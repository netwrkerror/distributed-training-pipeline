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
import subprocess
import sys

import pytest

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
def test_four_rank_training_run_agrees_and_exits() -> None:
    """The A4 done-when: 4 processes complete, agree, and the job exits cleanly.

    A hang here fails on the subprocess timeout rather than blocking forever, which
    is the difference between a test suite and a wedged terminal.
    """
    result = _run(["-m", "dtp.train", "--epochs", "1", "--batch-size", "128"], port="29611")
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"torchrun exited {result.returncode}\n{output}"
    assert "spread=0.00e+00" in output, f"ranks disagreed on the loss\n{output}"
    assert "world_size=4" in output, output


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
