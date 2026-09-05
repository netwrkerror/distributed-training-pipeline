"""A3: metrics must not flatter an imbalanced model."""

from __future__ import annotations

import torch

from dtp.metrics import ClassificationMetrics


def _onehot(preds: list[int], num_classes: int) -> torch.Tensor:
    logits = torch.zeros(len(preds), num_classes)
    for row, p in enumerate(preds):
        logits[row, p] = 1.0
    return logits


def test_perfect_predictions() -> None:
    m = ClassificationMetrics(3)
    targets = torch.tensor([0, 1, 2, 0])
    m.update(_onehot([0, 1, 2, 0], 3), targets, 0.0, 4)
    assert m.accuracy == 1.0
    assert m.macro_recall == 1.0


def test_majority_class_predictor_is_exposed_by_macro_recall() -> None:
    """The whole reason macro_recall is reported alongside accuracy.

    Nine `car` and one `trailer`; the model always says `car`. Accuracy says 90%,
    which on this dataset would look like a working model.
    """
    m = ClassificationMetrics(2)
    targets = torch.tensor([0] * 9 + [1])
    m.update(_onehot([0] * 10, 2), targets, 0.0, 10)
    assert m.accuracy == 0.9
    assert m.macro_recall == 0.5  # 1.0 on car, 0.0 on trailer
    assert m.per_class_recall() == [1.0, 0.0]


def test_absent_classes_are_skipped_not_counted_as_zero() -> None:
    """A class with no support is unknown, not wrong; counting it as 0 would
    understate macro_recall on any batch that happens to miss a rare class."""
    m = ClassificationMetrics(3)
    targets = torch.tensor([0, 0, 1])
    m.update(_onehot([0, 0, 1], 3), targets, 0.0, 3)
    recalls = m.per_class_recall()
    assert recalls[2] != recalls[2]  # NaN
    assert m.macro_recall == 1.0


def test_loss_is_a_weighted_mean_over_samples() -> None:
    m = ClassificationMetrics(2)
    t = torch.tensor([0, 1])
    m.update(_onehot([0, 1], 2), t, loss_sum=4.0, n=2)
    m.update(_onehot([0, 1], 2), t, loss_sum=2.0, n=2)
    assert m.loss == 1.5


def test_reset_clears_state() -> None:
    m = ClassificationMetrics(2)
    m.update(_onehot([0], 2), torch.tensor([0]), 1.0, 1)
    m.reset()
    assert m.confusion.sum().item() == 0
    assert m.n == 0
