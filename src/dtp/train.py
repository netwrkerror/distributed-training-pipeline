"""A3/A4: the training loop, single-process and under DDP.

    python -m dtp.train --epochs 8                      # A3: single process
    make ddp EPOCHS=3                                   # A4: 4 processes, gloo

This has to converge before any DDP code exists. Once there are four processes, a
loss curve that does not come down has two candidate explanations - the model or the
data distribution across ranks - and no way to tell them apart. This run is the
control that makes A4's failures attributable, so its numbers (loss per epoch, step
time, macro-recall) are the reference everything later is compared against.

Written single-process but *not* single-process-only: rank and world size come from
`context_from_env`, which returns rank 0 of world 1 when torchrun is absent. A4 wraps
the model in DistributedDataParallel without rewriting the loop.

A4 deliberately does NOT use DistributedSampler - that is A5. Every rank therefore
iterates the whole dataset in the same order, computes identical gradients, and
averaging identical gradients changes nothing. The four-process run should reproduce
the single-process loss curve while doing four times the work. That is the
duplicated-data pathology, demonstrated on purpose so A5 has something to fix.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset

from dtp.dataset import CropDataset
from dtp.dist import all_gather_scalar, context_from_env, process_group, setup_logging
from dtp.metrics import ClassificationMetrics
from dtp.model import SmallCNN, count_parameters
from dtp.splits import random_split, scene_split


def set_seed(seed: int) -> None:
    """Seed every generator this process will draw from.

    Note that this seeds the *parent* process. DataLoader workers are seeded
    separately by torch (base_seed + worker_id); see `python -m dtp.probe`.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, num_classes: int
) -> ClassificationMetrics:
    model.eval()
    metrics = ClassificationMetrics(num_classes)
    with torch.no_grad():
        for images, targets in loader:
            logits = model(images)
            loss = criterion(logits, targets)
            metrics.update(logits, targets, loss.item() * targets.size(0), targets.size(0))
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_classes: int,
    sync_each_step: bool,
) -> tuple[ClassificationMetrics, list[float]]:
    """Run one epoch; return metrics and the per-step wall times.

    `sync_each_step` exists to measure gotcha 7. Calling `.item()` on the loss inside
    the loop forces the process to wait for the computation to finish before it can
    read the number. On CPU that is a cheap barrier; on GPU it stalls the pipeline by
    forcing a device synchronisation every step, which is why the habit is worth
    breaking before there is a GPU to punish it. The default path accumulates the
    loss as a tensor and reads it once, at the end of the epoch.
    """
    model.train()
    metrics = ClassificationMetrics(num_classes)
    step_times: list[float] = []
    running_loss = torch.zeros((), dtype=torch.float64)
    seen = 0

    for images, targets in loader:
        start = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch = targets.size(0)
        if sync_each_step:
            running_loss += loss.item() * batch
        else:
            running_loss += loss.detach().double() * batch
        seen += batch

        step_times.append(time.perf_counter() - start)
        metrics.update(logits.detach(), targets, 0.0, 0)

    metrics.loss_sum = float(running_loss)
    metrics.n = seen
    return metrics, step_times


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data/crops.jsonl")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--crop-size", type=int, default=64)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument(
        "--split",
        choices=("scene", "random"),
        default="scene",
        help="scene: honest. random: leaks near-duplicate crops; diagnostic only",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-dir", default="runs")
    ap.add_argument("--backend", default="gloo", help="torch.distributed backend")
    ap.add_argument(
        "--threads",
        type=int,
        default=0,
        help="torch intra-op threads; 0 picks cpu_count//2 divided by world_size. "
        "torchrun sets OMP_NUM_THREADS=1, so leaving this unset makes a 4-process run "
        "and a single-process run use different threading and their step times "
        "incomparable",
    )
    ap.add_argument(
        "--log-all-ranks",
        action="store_true",
        help="log progress from every rank, not just rank 0",
    )
    ap.add_argument(
        "--sync-each-step",
        action="store_true",
        help="call loss.item() every step (gotcha 7); default accumulates on-device",
    )
    args = ap.parse_args()

    ctx = context_from_env()
    # A single process needs no process group; entering one would mean a rendezvous
    # with nobody to meet. nullcontext keeps the two paths structurally identical.
    launcher = process_group(args.backend) if ctx.is_distributed else nullcontext(ctx)

    with launcher as ctx:
        _run(args, ctx)


def _run(args: argparse.Namespace, ctx) -> None:
    log = setup_logging(ctx, master_only=not args.log_all_ranks)
    set_seed(args.seed)

    # Every rank runs its own intra-op thread pool on the same machine. Left at the
    # single-process default they would oversubscribe the cores four times over and
    # the step time would measure contention, not the model.
    threads = args.threads or max(1, (os.cpu_count() or 4) // 2 // ctx.world_size)
    torch.set_num_threads(threads)

    dataset = CropDataset(args.manifest, crop_size=args.crop_size)
    splitter = scene_split if args.split == "scene" else random_split
    train_idx, val_idx = splitter(dataset.records, args.val_fraction, args.seed)
    num_classes = len(dataset.classes)

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model: nn.Module = SmallCNN(num_classes)
    if ctx.is_distributed:
        # No device_ids: on CPU there is no device to bind to. On CUDA this would be
        # device_ids=[ctx.local_rank], which is what makes each rank own one GPU.
        model = DistributedDataParallel(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Only rank 0 writes. Four processes creating the same timestamped directory and
    # appending to the same file would interleave lines and race on creation.
    run_dir = Path(args.run_dir) / time.strftime("%Y%m%d-%H%M%S")
    metrics_path = run_dir / "metrics.jsonl"
    if ctx.is_master:
        run_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "dataset=%d train=%d val=%d classes=%d params=%s threads=%d split=%s "
        "world_size=%d per_device_batch=%d effective_batch=%d",
        len(dataset),
        len(train_idx),
        len(val_idx),
        num_classes,
        f"{count_parameters(model):,}",
        torch.get_num_threads(),
        args.split,
        ctx.world_size,
        args.batch_size,
        args.batch_size * ctx.world_size,
    )
    if ctx.is_master:
        with open(run_dir / "config.json", "w") as fh:
            json.dump(
                vars(args) | {"num_classes": num_classes, "classes": dataset.classes},
                fh,
                indent=2,
            )

    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        train_metrics, step_times = train_one_epoch(
            model, train_loader, criterion, optimizer, num_classes, args.sync_each_step
        )
        val_metrics = evaluate(model, val_loader, criterion, num_classes)
        epoch_s = time.perf_counter() - epoch_start

        # The A4 done-when, as a number rather than an impression. With gradients
        # synchronised the ranks are training one model, so their losses must agree.
        rank_losses = all_gather_scalar(train_metrics.loss)
        loss_spread = max(rank_losses) - min(rank_losses)

        record = {
            "epoch": epoch,
            "train": train_metrics.summary(dataset.classes),
            "val": val_metrics.summary(dataset.classes),
            "steps": len(step_times),
            "step_ms_median": round(1000 * statistics.median(step_times), 2),
            "step_ms_p90": round(1000 * sorted(step_times)[int(0.9 * len(step_times))], 2),
            "epoch_s": round(epoch_s, 2),
            "world_size": ctx.world_size,
            "rank_losses": [round(x, 6) for x in rank_losses],
            "loss_spread": loss_spread,
        }
        if ctx.is_master:
            with open(metrics_path, "a") as fh:
                fh.write(json.dumps(record) + "\n")

        log.info(
            "epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.3f  val_macro_recall=%.3f  "
            "step_median=%.1fms  epoch=%.1fs",
            epoch + 1,
            args.epochs,
            train_metrics.loss,
            val_metrics.loss,
            val_metrics.accuracy,
            val_metrics.macro_recall,
            record["step_ms_median"],
            epoch_s,
        )
        if ctx.is_distributed:
            log.info(
                "per-rank train_loss=%s spread=%.2e",
                [round(x, 5) for x in rank_losses],
                loss_spread,
            )

    if ctx.is_master:
        log.info("wrote %s", metrics_path)


if __name__ == "__main__":
    main()
