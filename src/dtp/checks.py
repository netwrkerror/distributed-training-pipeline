"""Runtime checks that a distributed run is doing what it claims.

    torchrun --nproc_per_node=4 -m dtp.checks gradient-sync

These are assertions about the *framework*, not about the model. They exist because
A4's stated done-when - "per-rank losses agree closely" - is satisfiable by a broken
implementation. When every rank sees identical data starting from identical weights,
the losses agree whether or not gradients are ever exchanged. Agreement is necessary,
not sufficient.

The check below makes the ranks disagree on purpose, then asserts that DDP reconciles
them.
"""

from __future__ import annotations

import argparse

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler

from dtp.dist import process_group, setup_logging
from dtp.model import SmallCNN

NUM_CLASSES = 4
TOLERANCE = 1e-6


def _rank_specific_batch(rank: int, n: int = 8, size: int = 32):
    """Deliberately different data on every rank."""
    generator = torch.Generator().manual_seed(1000 + rank)
    images = torch.rand(n, 3, size, size, generator=generator)
    targets = torch.randint(0, NUM_CLASSES, (n,), generator=generator)
    return images, targets


def gradient_sync() -> int:
    """Assert DDP averages gradients across ranks that saw different data.

    Three things are checked, and the third is what stops the test being vacuous:

    1. DDP's gradient equals the mean of the ranks' independent local gradients.
    2. Every rank ends with bit-identical gradients.
    3. The local, un-synchronised gradients actually *differ* between ranks - if the
       ranks had accidentally been handed the same data, checks 1 and 2 would pass
       while proving nothing at all.
    """
    with process_group() as ctx:
        log = setup_logging(ctx, master_only=True)
        if not ctx.is_distributed:
            log.warning("gradient-sync needs world_size > 1; run it under torchrun")
            return 1

        torch.manual_seed(0)
        reference = SmallCNN(NUM_CLASSES, width=8)
        # DDP broadcasts rank 0's parameters at construction, so the ranks start
        # identical even if their seeds had differed.
        model = DistributedDataParallel(SmallCNN(NUM_CLASSES, width=8))
        model.module.load_state_dict(reference.state_dict())

        images, targets = _rank_specific_batch(ctx.rank)
        criterion = nn.CrossEntropyLoss()

        # (a) the synchronised gradient, via DDP
        criterion(model(images), targets).backward()
        ddp_grads = [p.grad.detach().clone() for p in model.module.parameters()]

        # (b) the same gradient computed locally, with no communication at all
        local = SmallCNN(NUM_CLASSES, width=8)
        local.load_state_dict(reference.state_dict())
        criterion(local(images), targets).backward()
        local_grads = [p.grad.detach().clone() for p in local.parameters()]

        # (c) is our data actually different across ranks? Otherwise this proves nothing.
        probe = local_grads[0].clone()
        gathered = [torch.zeros_like(probe) for _ in range(ctx.world_size)]
        dist.all_gather(gathered, probe)
        max_local_difference = max(
            float((gathered[i] - gathered[0]).abs().max()) for i in range(ctx.world_size)
        )
        if max_local_difference < TOLERANCE:
            log.error(
                "ranks computed identical local gradients (%.2e); the check is vacuous",
                max_local_difference,
            )
            return 1

        # (a) == mean over ranks of (b)
        worst = 0.0
        for ddp_grad, local_grad in zip(ddp_grads, local_grads, strict=True):
            expected = local_grad.clone()
            dist.all_reduce(expected, op=dist.ReduceOp.SUM)
            expected /= ctx.world_size
            worst = max(worst, float((ddp_grad - expected).abs().max()))

        # every rank must hold the same gradient afterwards
        probe = ddp_grads[0].clone()
        gathered = [torch.zeros_like(probe) for _ in range(ctx.world_size)]
        dist.all_gather(gathered, probe)
        cross_rank = max(
            float((gathered[i] - gathered[0]).abs().max()) for i in range(ctx.world_size)
        )

        log.info(
            "local gradients differ across ranks by %.3e (good: the check is meaningful)",
            max_local_difference,
        )
        log.info("DDP gradient vs mean of local gradients: %.3e", worst)
        log.info("DDP gradient disagreement across ranks : %.3e", cross_rank)

        ok = worst < TOLERANCE and cross_rank < TOLERANCE
        log.info("gradient-sync: %s", "PASS" if ok else "FAIL")
        dist.barrier()
        return 0 if ok else 1


def sampler_coverage() -> int:
    """Assert the real partition across real processes, and expose gotcha 1.

    `tests/test_sampler.py` checks the same properties by constructing all four ranks'
    samplers inside one process, which is a stronger test in most respects. This one
    is complementary: it verifies that the ranks a *live job* actually has agree, so
    it would catch a rank misreading its own rank or world size from the environment -
    something the simulated version assumes correct by construction.
    """
    with process_group() as ctx:
        log = setup_logging(ctx, master_only=True)
        if not ctx.is_distributed:
            log.warning("sampler-coverage needs world_size > 1; run it under torchrun")
            return 1

        n = 3208
        dataset = list(range(n))
        sampler = DistributedSampler(dataset, shuffle=True, seed=0)

        ok = True
        orders: list[list[int]] = []
        for epoch in range(3):
            sampler.set_epoch(epoch)
            local = list(sampler)
            orders.append(local)

            gathered: list[list[int]] = [None] * ctx.world_size  # type: ignore[list-item]
            dist.all_gather_object(gathered, local)
            union = [i for shard in gathered for i in shard]

            exact = sorted(union) == dataset
            no_overlap = len(set(union)) == len(union)
            equal_sizes = len({len(shard) for shard in gathered}) == 1
            ok = ok and exact and no_overlap and equal_sizes
            log.info(
                "epoch %d: %d indices over %d ranks | covers_exactly=%s no_overlap=%s "
                "equal_shards=%s",
                epoch,
                len(union),
                ctx.world_size,
                exact,
                no_overlap,
                equal_sizes,
            )

        changed = len({tuple(o) for o in orders}) == len(orders)
        log.info("order differs every epoch with set_epoch: %s", changed)
        ok = ok and changed

        # gotcha 1: the same sampler, never told which epoch it is
        frozen = DistributedSampler(dataset, shuffle=True, seed=0)
        repeats = [list(frozen) for _ in range(3)]
        identical = repeats[0] == repeats[1] == repeats[2]
        log.info(
            "without set_epoch, every epoch is identical: %s  (this is the bug, and it "
            "never crashes and never logs)",
            identical,
        )
        ok = ok and identical

        log.info("sampler-coverage: %s", "PASS" if ok else "FAIL")
        dist.barrier()
        return 0 if ok else 1


CHECKS = {"gradient-sync": gradient_sync, "sampler-coverage": sampler_coverage}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("check", choices=sorted(CHECKS))
    return CHECKS[ap.parse_args().check]()


if __name__ == "__main__":
    raise SystemExit(main())
