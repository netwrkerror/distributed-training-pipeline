"""Report how DataLoader workers are actually started on this machine.

Run with:  python -m dtp.probe

This is not a test, it is a measurement of the platform. The multiprocessing start
method decides three things that bite much later:

  fork  - the child is a copy of the parent's memory. Worker startup is cheap, but
          the child inherits the parent's RNG state *exactly*, so every worker
          draws the same "random" augmentations (gotcha 3).
  spawn - the child is a fresh interpreter that re-imports the module. Startup is
          expensive, which is what persistent_workers buys back (B3), and dataset
          objects must be picklable.

macOS defaults to spawn; Linux defaults to fork. Numbers measured on a Mac
therefore do not transfer to a Linux training box, and a seeding bug that
reproduces in production may not reproduce here. Better to know that in A1 than to
discover it in B5.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import platform
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SeedProbe(Dataset):
    """Returns the RNG state each worker sees, not data."""

    def __init__(self, n: int = 8) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, int]:
        return {
            "idx": idx,
            "pid": os.getpid(),
            "torch_seed": torch.initial_seed() % (2**31),
            "numpy_draw": int(np.random.randint(0, 2**31)),
            "python_draw": random.randint(0, 2**31),
        }


def _run(num_workers: int, context: str | None) -> None:
    label = f"num_workers={num_workers}" + (f", context={context}" if context else "")
    try:
        loader = DataLoader(
            SeedProbe(8),
            batch_size=1,
            num_workers=num_workers,
            multiprocessing_context=context if num_workers > 0 else None,
        )
        rows = [
            (
                int(b["idx"][0]),
                int(b["pid"][0]),
                int(b["torch_seed"][0]),
                int(b["numpy_draw"][0]),
                int(b["python_draw"][0]),
            )
            for b in loader
        ]
    except Exception as exc:  # a crash here is itself the finding
        print(f"\n{label}\n  FAILED: {type(exc).__name__}: {exc}")
        return

    print(f"\n{label}")
    print(f"  {'idx':>3} {'pid':>7} {'torch_seed':>12} {'numpy_draw':>12} {'py_draw':>12}")
    for idx, pid, ts, nd, pd in rows:
        print(f"  {idx:>3} {pid:>7} {ts:>12} {nd:>12} {pd:>12}")

    if num_workers == 0:
        return

    by_pid: dict[int, list[int]] = {}
    for _, pid, _, nd, _ in rows:
        by_pid.setdefault(pid, []).append(nd)
    first_draws = {pid: draws[0] for pid, draws in by_pid.items()}
    distinct = len(set(first_draws.values()))
    verdict = (
        "independent seeding"
        if distinct == len(by_pid)
        else "SHARED NUMPY SEED -> every worker draws identical augmentations (gotcha 3)"
    )
    print(f"  -> {len(by_pid)} workers, {distinct} distinct first numpy draw(s): {verdict}")


def main() -> None:
    print(f"platform         : {platform.platform()}")
    print(f"python           : {platform.python_version()}")
    print(f"torch            : {torch.__version__}")
    print(f"cpu count        : {os.cpu_count()}")
    print(f"mp start method  : {mp.get_start_method()}")
    print(f"available methods: {mp.get_all_start_methods()}")

    _run(0, None)
    _run(2, None)  # whatever this platform defaults to
    _run(2, "fork")  # the Linux default, forced, to see whether gotcha 3 reproduces


if __name__ == "__main__":
    main()
