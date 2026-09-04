"""A1 done-when: four processes agree on who they are and can move bytes.

Run with:  torchrun --nproc_per_node=4 -m dtp.hello

Printing rank and world_size only proves each process read its environment. It
does not prove the ranks can communicate. The all_reduce here has a known answer
(0 + 1 + ... + (world_size - 1)), so a process group that initializes but cannot
actually exchange tensors fails loudly instead of looking like success.
"""

from __future__ import annotations

import os
import socket
import time

import torch
import torch.distributed as dist

from dtp.dist import process_group, setup_logging


def main() -> None:
    started = time.perf_counter()
    with process_group() as ctx:
        log = setup_logging(ctx)
        rendezvous_s = time.perf_counter() - started
        log.info(
            "up on %s pid=%d local_rank=%d backend=%s rendezvous=%.3fs",
            socket.gethostname(),
            os.getpid(),
            ctx.local_rank,
            dist.get_backend(),
            rendezvous_s,
        )

        # Every rank contributes its own rank number; the sum is arithmetic.
        payload = torch.tensor([float(ctx.rank)])
        dist.all_reduce(payload, op=dist.ReduceOp.SUM)
        expected = float(ctx.world_size * (ctx.world_size - 1) // 2)
        got = payload.item()
        if got != expected:
            raise RuntimeError(f"all_reduce returned {got}, expected {expected}")
        log.info("all_reduce ok: sum of ranks = %.0f (expected %.0f)", got, expected)

        # A barrier before exit makes the shutdown ordering explicit: without it,
        # rank 0 can finish and destroy its group while a slower rank is still
        # inside a collective.
        dist.barrier()
        if ctx.is_master:
            log.info("all %d ranks reached the barrier; exiting", ctx.world_size)


if __name__ == "__main__":
    main()
