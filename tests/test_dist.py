"""A1: the environment contract and the four-process gloo run."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from dtp.dist import DistContext, context_from_env


def test_context_defaults_to_single_process() -> None:
    """No torchrun, no environment: rank 0 of a world of 1, not a crash."""
    ctx = context_from_env(env={})
    assert ctx == DistContext(rank=0, local_rank=0, world_size=1)
    assert ctx.is_master
    assert not ctx.is_distributed


def test_context_reads_torchrun_env() -> None:
    ctx = context_from_env(env={"RANK": "3", "LOCAL_RANK": "3", "WORLD_SIZE": "4"})
    assert (ctx.rank, ctx.local_rank, ctx.world_size) == (3, 3, 4)
    assert not ctx.is_master
    assert ctx.is_distributed


@pytest.mark.parametrize("rank,expected", [(0, True), (1, False)])
def test_is_master_is_rank_zero(rank: int, expected: bool) -> None:
    assert context_from_env(env={"RANK": str(rank), "WORLD_SIZE": "4"}).is_master is expected


@pytest.mark.slow
def test_four_process_gloo_run_exits_cleanly() -> None:
    """The A1 done-when condition, as an assertion.

    Asserts three things that are easy to conflate: the job exits 0, every rank
    printed, and the all_reduce inside hello.py agreed with its known answer. A run
    that hangs fails here on the timeout rather than blocking the suite forever.

    The loopback flags are not incidental. On a host that cannot resolve its own
    hostname, bare `torchrun` never completes rendezvous; see `make doctor`.
    """
    env = {**os.environ, "GLOO_SOCKET_IFNAME": "lo0"}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--master-addr=127.0.0.1",
            "--master-port=29600",
            "--local-addr=127.0.0.1",
            "--nproc_per_node=4",
            "-m",
            "dtp.hello",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"torchrun exited {result.returncode}\n{output}"
    for rank in range(4):
        assert f"[rank {rank}/4]" in output, f"rank {rank} never logged\n{output}"
    assert output.count("all_reduce ok") == 4, f"not every rank completed all_reduce\n{output}"
