"""A8: the two hang signatures, and recovery from a killed worker.

Both failures here are hangs rather than crashes, because a collective is an
all-or-nothing rendezvous: `all_reduce` returns only once every rank has called it. A
rank that dies, or that disagrees with its peers about how many steps an epoch has,
produces an absence rather than an error.

The two signatures turn out to be very different under `torchrun`, and the difference
is the point of these tests.
"""

from __future__ import annotations

import contextlib
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ENV = {**os.environ, "GLOO_SOCKET_IFNAME": "lo0"}
MANIFEST = "data/crops.jsonl"
COLLECTIVE_TIMEOUT = 20
# Long enough that the agent's ~18s detection latency clearly beats it.
LONG_TIMEOUT = 60


def _seconds(stamp: str) -> float:
    hours, minutes, rest = stamp.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(rest)


def _torchrun(args: list[str], port: str, extra: list[str] | None = None, timeout: int = 900):
    """Run a torchrun job, and make sure nothing survives it.

    `subprocess.run(timeout=...)` kills the process it started - torchrun itself - and
    not the worker processes torchrun spawned. On a timeout those workers are orphaned
    and keep running: they hold rendezvous ports, burn CPU, and make every *later*
    distributed test slower and flakier for as long as the machine is up. This is not
    hypothetical; it happened here, and the resulting contention masqueraded as an
    unexplained hang for some time.

    So the job is started in its own process group and the whole group is signalled on
    the way out, whether or not it timed out.
    """
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--master-addr=127.0.0.1",
        f"--master-port={port}",
        "--local-addr=127.0.0.1",
        "--nproc_per_node=4",
        *(extra or []),
        *args,
    ]
    started = time.perf_counter()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=ENV,
        start_new_session=True,  # its own process group, so we can signal the whole tree
    )
    try:
        stdout, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, _ = process.communicate()
        raise
    finally:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)

    result = subprocess.CompletedProcess(cmd, process.returncode, stdout, "")
    return result, time.perf_counter() - started


@pytest.mark.slow
def test_a_killed_worker_fails_fast_rather_than_hanging() -> None:
    """Signature one, and it contradicts the expectation I started with.

    The received wisdom is that surviving ranks block at the next collective until the
    timeout fires. Under torchrun they do not: the elastic agent supervises its
    children, notices the dead one, and SIGTERMs the survivors before any collective
    times out.

    "Fast" is relative, though, and measuring it was worth the trouble: detection took
    ~18s on this machine, and that figure is independent of the collective timeout
    (18.01s at 20s, 18.02s at 60s) while `--monitor-interval` adds up to one interval
    on top. So the job is silent for eighteen seconds before anything is reported -
    fast compared with gloo's 30-minute default, slow compared with what "supervised"
    suggests.
    """
    result, _ = _torchrun(
        [
            "-m",
            "dtp.faults",
            "kill-rank",
            "--steps",
            "8",
            "--victim",
            "2",
            "--at",
            "3",
            "--timeout",
            str(COLLECTIVE_TIMEOUT),
        ],
        port="29740",
    )
    output = result.stdout
    assert result.returncode != 0, output
    assert "exitcode: 137" in output, f"expected the victim's exit code to be reported\n{output}"
    # SIGTERM to the survivors is the agent tearing the job down, not a collective timeout
    assert "closing signal SIGTERM" in output, output
    assert "Timed out waiting" not in output, (
        f"expected the agent to kill the job before any collective timed out\n{output}"
    )

    # No survivor completed the step the victim died in: they entered the collective
    # and never came out of it.
    assert "entering all_reduce for step 3" in output
    assert "step 3 all_reduce returned" not in output, (
        f"a rank completed the collective the victim was supposed to be in\n{output}"
    )

    # How long the agent took to notice, measured from the victim's own last words to
    # the agent's first teardown signal. This is what makes "fails fast" a measurement
    # rather than an impression.
    death = re.search(r"(\d\d:\d\d:\d\d\.\d+) \[rank \d\] dying abruptly", output)
    teardown = re.search(r"[WE]\d{4} (\d\d:\d\d:\d\d\.\d+).*closing signal SIGTERM", output)
    assert death and teardown, f"could not time the teardown\n{output}"
    detection = _seconds(teardown.group(1)) - _seconds(death.group(1))
    assert 0 <= detection < LONG_TIMEOUT, (
        f"agent took {detection:.2f}s to notice a dead worker, which is not sooner than "
        f"the {LONG_TIMEOUT}s collective timeout it is supposed to pre-empt"
    )


@pytest.mark.slow
def test_uneven_shards_hang_until_the_collective_times_out() -> None:
    """Signature two: the one that actually hangs.

    Nothing has failed here. All four processes are alive and healthy, so torchrun's
    agent sees a perfectly normal job. The ranks simply disagree about how many steps
    the epoch has, so one sits in `all_reduce` while the others sit in `barrier` - two
    different collectives, waiting for each other forever.

    This is what DistributedSampler's padding exists to prevent (see A5), and in
    production the default timeout is 30 minutes of a job that holds its allocation,
    logs nothing, and looks alive.
    """
    result, elapsed = _torchrun(
        [
            "-m",
            "dtp.faults",
            "uneven-shards",
            "--steps",
            "4",
            "--step-seconds",
            "0.2",
            "--timeout",
            str(COLLECTIVE_TIMEOUT),
        ],
        port="29750",
    )
    output = result.stdout
    assert result.returncode != 0
    assert "Timed out waiting" in output, f"expected a collective timeout\n{output}"

    # the diagnostic tell: the ranks are stuck in *different* collectives
    assert "entering all_reduce for step 4" in output, output
    assert "waiting at the end-of-epoch barrier" in output, output

    # and it really did block, rather than failing immediately
    assert elapsed > COLLECTIVE_TIMEOUT, (
        f"job ended in {elapsed:.1f}s, faster than the {COLLECTIVE_TIMEOUT}s collective "
        f"timeout - it cannot have hung"
    )


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(MANIFEST), reason="run `make crops` first")
def test_job_recovers_from_a_killed_worker(tmp_path: Path) -> None:
    """The A8 done-when: the job survives a killed worker and finishes.

    `--max-restarts` relaunches the workers; `--resume` makes them pick up the last
    checkpoint. The fault fires only on the first attempt, keyed on
    TORCHELASTIC_RESTART_COUNT, so the retry can succeed.
    """
    result, _ = _torchrun(
        [
            "-m",
            "dtp.train",
            "--epochs",
            "4",
            "--resume",
            "--crash-at-epoch",
            "1",
            "--batch-size",
            "256",
            "--checkpoint-dir",
            str(tmp_path / "ckpt"),
            "--run-dir",
            str(tmp_path / "runs"),
        ],
        port="29760",
        extra=["--rdzv-backend=c10d", "--rdzv-endpoint=127.0.0.1:29761", "--max-restarts=2"],
    )
    output = result.stdout
    assert result.returncode == 0, f"job did not recover\n{output}"

    assert "injected fault" in output, "the fault never fired; the test proves nothing"
    assert "exitcode: 137" in output, output
    assert "resumed from epoch 0" in output, f"restarted workers did not resume\n{output}"

    # ran to the target, and no further: --epochs is a target, not a per-attempt budget
    epochs = re.findall(r"epoch (\d+)/(\d+)", output)
    assert epochs, output
    assert epochs[-1] == ("4", "4"), f"expected to finish at epoch 4/4, got {epochs[-1]}"


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(MANIFEST), reason="run `make crops` first")
def test_epochs_is_a_target_not_a_budget_per_restart(tmp_path: Path) -> None:
    """Guards the bug the recovery run exposed.

    With `--epochs` meaning "run N more from wherever you resumed", every automatic
    restart silently extends the job: a run that crashed three times would train three
    extra epochs, and a resumed run would not be comparable with an uninterrupted one.
    """
    ckpt = tmp_path / "ckpt"
    _torchrun(
        [
            "-m",
            "dtp.train",
            "--epochs",
            "2",
            "--batch-size",
            "256",
            "--checkpoint-dir",
            str(ckpt),
            "--run-dir",
            str(tmp_path / "r1"),
        ],
        port="29770",
    )
    result, _ = _torchrun(
        [
            "-m",
            "dtp.train",
            "--epochs",
            "2",
            "--resume",
            "--batch-size",
            "256",
            "--checkpoint-dir",
            str(ckpt),
            "--run-dir",
            str(tmp_path / "r2"),
        ],
        port="29780",
    )
    output = result.stdout
    assert result.returncode == 0, output
    assert "already reached" in output, (
        f"resuming a finished run should be a no-op, not two more epochs\n{output}"
    )
