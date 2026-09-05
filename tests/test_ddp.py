"""A4: the DDP wrapper, asserted under real processes.

These spawn actual torchrun jobs, so they are marked slow. They are also the only
tests here that can fail for A4's real failure modes - a job that hangs at a
collective, or one whose ranks quietly diverge - because neither is reproducible in
a single process.

The loopback flags are required on a host that cannot resolve its own hostname; see
`make doctor` and the A1 notes.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

import pytest

from dtp.checkpoint import CheckpointManager
from dtp.model import SmallCNN

TORCHRUN = [
    sys.executable,
    "-m",
    "torch.distributed.run",
    "--master-addr=127.0.0.1",
    "--local-addr=127.0.0.1",
    "--nproc_per_node=4",
]
ENV = {**os.environ, "GLOO_SOCKET_IFNAME": "lo0"}
MANIFEST = "data/crops.jsonl"


def _run(args: list[str], port: str, timeout: int = 600) -> subprocess.CompletedProcess:
    cmd = [*TORCHRUN[:3], f"--master-port={port}", *TORCHRUN[3:], *args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=ENV)


@pytest.mark.slow
def test_gradient_sync_across_ranks() -> None:
    """DDP must average gradients from ranks that saw *different* data.

    This is the assertion A4's stated done-when does not make. "Per-rank losses agree"
    is satisfied trivially when all ranks see identical data, whether or not the
    all-reduce happens at all.
    """
    result = _run(["-m", "dtp.checks", "gradient-sync"], port="29610")
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"gradient-sync failed\n{output}"
    assert "gradient-sync: PASS" in output, output
    assert "the check is vacuous" not in output, output


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(MANIFEST), reason="run `make crops` first")
def test_four_rank_training_run_completes_and_exits() -> None:
    """4 processes complete and the job exits cleanly.

    In A4 this asserted `spread=0.00e+00`. That assertion was correct then and is
    wrong now, and the change is the point of A5: once DistributedSampler gives each
    rank a different shard, per-rank losses *must* differ, because they are computed
    over different data. Loss agreement was never evidence of gradient
    synchronisation - `test_gradient_sync_across_ranks` asserts that property
    directly, and it is what still guarantees the ranks train one shared model.

    A hang here fails on the subprocess timeout rather than blocking forever, which
    is the difference between a test suite and a wedged terminal.
    """
    result = _run(["-m", "dtp.train", "--epochs", "1", "--batch-size", "128"], port="29611")
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"torchrun exited {result.returncode}\n{output}"
    assert "world_size=4" in output, output

    spread = re.search(r"spread=([0-9.e+-]+)", output)
    assert spread, f"no per-rank loss spread reported\n{output}"
    assert float(spread.group(1)) > 0, (
        "per-rank losses were identical, so the ranks are sharing data: "
        "DistributedSampler is not partitioning\n" + output
    )


@pytest.mark.slow
def test_sampler_partitions_across_real_processes() -> None:
    """The A5 done-when, verified by the ranks a live job actually has.

    tests/test_sampler.py checks the same properties more thoroughly in one process.
    This catches what that one assumes: that each rank reads its own rank and world
    size correctly from the environment.
    """
    result = _run(["-m", "dtp.checks", "sampler-coverage"], port="29613")
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"sampler-coverage failed\n{output}"
    assert "sampler-coverage: PASS" in output, output
    assert "covers_exactly=True no_overlap=True equal_shards=True" in output, output


@pytest.mark.slow
def test_exception_on_one_rank_does_not_hang_the_job(tmp_path) -> None:
    """The `finally` in `process_group` earns its place here.

    A rank that raises must still tear down its process group, or its peers block in
    the next collective until the backend timeout - minutes of a job that looks alive
    and is not. The job is expected to fail; what is asserted is that it fails
    *promptly*, rather than hanging until the subprocess timeout kills it.
    """
    script = tmp_path / "boom.py"
    script.write_text(
        "import torch.distributed as dist\n"
        "from dtp.dist import process_group\n"
        "with process_group() as ctx:\n"
        "    dist.barrier()\n"
        "    if ctx.rank == 2:\n"
        "        raise RuntimeError('deliberate failure on rank 2')\n"
        "    dist.barrier()\n"
    )
    result = _run([str(script)], port="29612", timeout=300)
    output = result.stdout + result.stderr
    assert result.returncode != 0, f"job should have failed\n{output}"
    assert "deliberate failure on rank 2" in output, output


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(MANIFEST), reason="run `make crops` first")
def test_only_rank_zero_writes_checkpoints(tmp_path) -> None:
    """A6 under real processes: one writer, and the result loads unwrapped.

    The single-process tests use a stand-in for DistributedDataParallel, because real
    DDP needs a process group. This is the one that proves the `module.` prefix is
    actually stripped from a checkpoint written by a genuine DDP-wrapped model.
    """
    ckpt_dir = tmp_path / "checkpoints"
    result = _run(
        [
            "-m",
            "dtp.train",
            "--epochs",
            "2",
            "--batch-size",
            "256",
            "--checkpoint-dir",
            str(ckpt_dir),
            "--run-dir",
            str(tmp_path / "runs"),
        ],
        port="29614",
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output

    written = sorted(p.name for p in ckpt_dir.glob("ckpt_epoch*.pt"))
    assert written == ["ckpt_epoch0000.pt", "ckpt_epoch0001.pt"], (
        f"expected exactly one writer, got {written}\n{output}"
    )
    assert not list(ckpt_dir.glob("*.tmp")), "temp files survived"

    manager = CheckpointManager(ckpt_dir)
    meta = manager.read_marker()
    assert meta is not None and meta.world_size == 4
    state = manager.load()  # verifies the checksum
    assert state["epoch"] == 1
    assert not any(k.startswith("module.") for k in state["model"]), (
        "checkpoint written by a real DDP-wrapped model kept the module. prefix"
    )
    SmallCNN(10).load_state_dict(state["model"])  # strict=True
