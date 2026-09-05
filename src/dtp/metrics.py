"""Metric accumulation for an imbalanced classification problem.

Overall accuracy is close to useless on this dataset. `car` is 48% of the crops and
`trailer` is 0.4%, so a model that predicts `car` for everything scores 48% and has
learned nothing. Worse, that number *rises* during early training in a way that
looks like progress.

So the accumulator keeps a full confusion matrix and reports macro-recall - the mean
of per-class recalls, where every class counts equally regardless of frequency. A
model ignoring the tail cannot hide in it.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


class ClassificationMetrics:
    """Confusion-matrix accumulator. Update per batch, summarise per epoch."""

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        self.confusion = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long)
        self.loss_sum = 0.0
        self.n = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss_sum: float, n: int) -> None:
        preds = logits.argmax(dim=1)
        # bincount over (true * C + pred) fills the matrix in one pass without a
        # Python loop over the batch.
        idx = targets.reshape(-1) * self.num_classes + preds.reshape(-1)
        self.confusion += torch.bincount(idx, minlength=self.num_classes**2).reshape(
            self.num_classes, self.num_classes
        )
        self.loss_sum += loss_sum
        self.n += n

    @property
    def loss(self) -> float:
        return self.loss_sum / max(1, self.n)

    @property
    def accuracy(self) -> float:
        total = self.confusion.sum().item()
        return self.confusion.diag().sum().item() / max(1, total)

    def per_class_recall(self) -> list[float]:
        support = self.confusion.sum(dim=1)
        correct = self.confusion.diag()
        return [
            (correct[i].item() / support[i].item()) if support[i] > 0 else float("nan")
            for i in range(self.num_classes)
        ]

    @property
    def macro_recall(self) -> float:
        """Mean recall over classes that actually appear. NaN classes are skipped."""
        recalls = [r for r in self.per_class_recall() if r == r]
        return sum(recalls) / max(1, len(recalls))

    def all_reduce(self) -> None:
        """Sum this rank's counts into every rank's, in place.

        Metrics have to be *reduced*, not sampled. Once DistributedSampler gives each
        rank a different shard, rank 0's local accuracy is computed over a quarter of
        the validation set - a number that is not wrong so much as answering a
        different question, and one that moves around as the shuffle changes. Summing
        the confusion matrices gives the metric over the whole set.

        Note this reduces *counts*, not rates. Averaging four ranks' accuracies would
        also be wrong whenever their shards differ in size.
        """
        if not dist.is_available() or not dist.is_initialized():
            return
        dist.all_reduce(self.confusion, op=dist.ReduceOp.SUM)
        totals = torch.tensor([self.loss_sum, float(self.n)], dtype=torch.float64)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        self.loss_sum = float(totals[0])
        self.n = int(totals[1])

    def summary(self, classes: list[str]) -> dict:
        return {
            "loss": round(self.loss, 5),
            "accuracy": round(self.accuracy, 4),
            "macro_recall": round(self.macro_recall, 4),
            "per_class_recall": {
                name: (None if r != r else round(r, 4))
                for name, r in zip(classes, self.per_class_recall(), strict=True)
            },
            "support": dict(zip(classes, self.confusion.sum(dim=1).tolist(), strict=True)),
        }
