"""Process-group setup and teardown.

Everything distributed in this repo goes through here so that rank/world_size are
read from exactly one place and the process group is always destroyed, including
on the exception path. A process that exits without calling destroy_process_group
leaves its peers blocked in the next collective until the backend timeout fires,
which is the slowest possible way to learn about a crash.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch.distributed as dist

DEFAULT_BACKEND = "gloo"


@dataclass(frozen=True)
class DistContext:
    """Who this process is in the job.

    torchrun sets RANK, LOCAL_RANK and WORLD_SIZE in each child's environment. We
    read them rather than computing them so that a single-process run (no torchrun)
    degrades to rank 0 of a world of 1 instead of crashing.
    """

    rank: int
    local_rank: int
    world_size: int

    @property
    def is_master(self) -> bool:
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1


def context_from_env(env: dict[str, str] | None = None) -> DistContext:
    """Build a DistContext from torchrun's environment variables."""
    env = os.environ if env is None else env
    return DistContext(
        rank=int(env.get("RANK", "0")),
        local_rank=int(env.get("LOCAL_RANK", "0")),
        world_size=int(env.get("WORLD_SIZE", "1")),
    )


def setup_logging(ctx: DistContext, level: int = logging.INFO) -> logging.Logger:
    """Rank-tagged logging.

    Interleaved output from N processes is unreadable without the rank on every
    line, and 'which rank printed this' is the first question of every distributed
    debugging session.
    """
    logging.basicConfig(
        level=level,
        format=f"%(asctime)s [rank {ctx.rank}/{ctx.world_size}] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    return logging.getLogger("dtp")


@contextmanager
def process_group(backend: str = DEFAULT_BACKEND) -> Iterator[DistContext]:
    """Initialize the process group, yield the context, always tear it down.

    init_process_group is a collective: it returns only once every rank in
    WORLD_SIZE has checked in at the rendezvous. If one process never starts, the
    others block here rather than failing fast.
    """
    ctx = context_from_env()
    dist.init_process_group(backend=backend)
    try:
        yield ctx
    finally:
        # try/finally, not a plain call at the end: an exception on one rank must
        # still tear down that rank's group, or the peers hang.
        if dist.is_initialized():
            dist.destroy_process_group()
