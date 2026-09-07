"""Fault injection: make the two classic hangs happen on purpose.

    torchrun --nproc_per_node=4 -m dtp.faults kill-rank --timeout 30
    torchrun --nproc_per_node=4 -m dtp.faults uneven-shards --timeout 30

Both failures here are *hangs*, not crashes, and that is the whole lesson. A
collective is an all-or-nothing rendezvous: `all_reduce` returns only once every rank
has called it. A rank that dies, or that takes a different number of steps from its
peers, therefore does not raise anything - it produces an absence, and the survivors
block waiting for a participant that is never going to arrive.

In production the timeout is 30 minutes by default, so the symptom is a job that
holds a cluster allocation, burns budget, emits no logs and looks alive. `--timeout`
shortens it so the failure can be watched in under a minute.
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import signal
import sys
import time

import torch
import torch.distributed as dist

from dtp.dist import process_group, setup_logging


def enable_stack_dumps() -> None:
    """Let a hung rank be asked what it is waiting on, without root.

    `py-spy dump --pid <pid>` is the usual tool and is what I reached for first. On
    macOS it refuses:

        This program requires root on OSX.
        Try running again with elevated permissions by going 'sudo !!'

    faulthandler gives the same answer for a process that opted in ahead of time: send
    it SIGUSR1 and every thread's Python stack goes to stderr. It cannot attach to an
    arbitrary process the way py-spy can, but a training job can arm itself in advance,
    and then diagnosing a hang needs no privileges at all.
    """
    faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)


def _log_every_rank(ctx, message: str) -> None:
    """Print from all ranks - during a hang, who is where is the entire question.

    Timestamped because the interesting quantity in A8 is *when* things happened
    relative to each other: how long after a rank died did the agent notice, and how
    long did the survivors sit in a collective before anything gave way.
    """
    print(
        f"{time.strftime('%H:%M:%S')}.{int(time.time() % 1 * 1000):03d} "
        f"[rank {ctx.rank}] {message}",
        flush=True,
    )


def kill_rank(args: argparse.Namespace) -> int:
    """One rank dies abruptly; the survivors block at the next collective.

    `os._exit` rather than `sys.exit` on purpose: it skips atexit handlers and any
    chance to destroy the process group cleanly, which is what an OOM kill, a
    segfault or a preempted spot instance actually looks like.
    """
    with process_group(timeout_s=args.timeout) as ctx:
        log = setup_logging(ctx, master_only=True)
        log.info("running %d steps; rank %d will die at step %d", args.steps, args.victim, args.at)

        for step in range(args.steps):
            if ctx.rank == args.victim and step == args.at:
                _log_every_rank(ctx, f"dying abruptly at step {step} (os._exit)")
                sys.stdout.flush()
                os._exit(137)  # 137 = SIGKILL's exit code, as an OOM kill would give

            payload = torch.full((256,), float(ctx.rank))
            started = time.perf_counter()
            _log_every_rank(ctx, f"entering all_reduce for step {step}")
            dist.all_reduce(payload)
            _log_every_rank(
                ctx, f"step {step} all_reduce returned in {time.perf_counter() - started:.2f}s"
            )
            time.sleep(args.step_seconds)

        log.info("completed without incident (unexpected)")
        return 0


def uneven_shards(args: argparse.Namespace) -> int:
    """Ranks disagree about how many steps an epoch has.

    This is what `drop_last=False` protects against, and why DistributedSampler pads
    rather than letting the last shard be short (see A5). Here rank 0 is handed one
    extra batch: it enters an all_reduce that its peers have already finished the loop
    and moved past. Nothing is malformed, no rank has failed - they simply disagree
    about the length of the epoch.
    """
    with process_group(timeout_s=args.timeout) as ctx:
        log = setup_logging(ctx, master_only=True)
        steps = args.steps + (1 if ctx.rank == 0 else 0)
        _log_every_rank(ctx, f"my shard has {steps} batches")
        if ctx.is_master:
            log.info("rank 0 has one extra batch; the others will finish the epoch without it")

        for step in range(steps):
            payload = torch.full((256,), float(ctx.rank))
            _log_every_rank(ctx, f"entering all_reduce for step {step} of {steps}")
            dist.all_reduce(payload)
            _log_every_rank(ctx, f"step {step} done")
            time.sleep(args.step_seconds)

        _log_every_rank(ctx, "finished my loop; waiting at the end-of-epoch barrier")
        dist.barrier()
        log.info("completed without incident (unexpected)")
        return 0


FAULTS = {"kill-rank": kill_rank, "uneven-shards": uneven_shards}


def main() -> int:
    enable_stack_dumps()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fault", choices=sorted(FAULTS))
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--victim", type=int, default=2, help="rank to kill")
    ap.add_argument("--at", type=int, default=3, help="step at which to kill it")
    ap.add_argument("--step-seconds", type=float, default=0.5)
    ap.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="collective timeout in seconds (gloo's default is 30 minutes)",
    )
    return FAULTS[ap.parse_args().fault](ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
