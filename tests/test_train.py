"""A3: the training loop must actually reduce loss.

This is the assertion the whole milestone rests on. Once A4 introduces four
processes, a flat loss curve has two explanations - the model or the data
distribution across ranks - and telling them apart requires knowing that this loop,
on this model, converges when nothing is distributed.

It uses a synthetic separable dataset rather than nuScenes so it runs in under a
second and does not need the 9GB split on disk.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dtp.metrics import ClassificationMetrics
from dtp.model import SmallCNN, count_parameters
from dtp.train import evaluate, set_seed, train_one_epoch

NUM_CLASSES = 2


def _separable_loader(n: int = 64, size: int = 16) -> DataLoader:
    """Class 0 is dark, class 1 is bright. Trivially learnable, with noise."""
    set_seed(0)
    labels = torch.randint(0, NUM_CLASSES, (n,))
    images = torch.rand(n, 3, size, size) * 0.2 + labels.view(-1, 1, 1, 1).float() * 0.6
    return DataLoader(TensorDataset(images, labels), batch_size=8)


def test_loss_decreases_over_epochs() -> None:
    set_seed(0)
    loader = _separable_loader()
    model = SmallCNN(NUM_CLASSES, width=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    criterion = nn.CrossEntropyLoss()

    losses = []
    for _ in range(6):
        metrics, step_times = train_one_epoch(
            model, loader, criterion, optimizer, NUM_CLASSES, sync_each_step=False
        )
        losses.append(metrics.loss)
        assert len(step_times) == len(loader)

    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"
    assert losses[-1] < 0.4, f"loop failed to learn a separable problem: {losses}"


def test_accumulated_loss_matches_per_step_item() -> None:
    """The gotcha 7 workaround must not change the number, only when it is read."""
    results = []
    for sync in (False, True):
        set_seed(0)
        loader = _separable_loader()
        model = SmallCNN(NUM_CLASSES, width=8)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        metrics, _ = train_one_epoch(
            model, loader, nn.CrossEntropyLoss(), optimizer, NUM_CLASSES, sync_each_step=sync
        )
        results.append(metrics.loss)
    assert abs(results[0] - results[1]) < 1e-6, results


def test_evaluate_does_not_update_weights() -> None:
    set_seed(0)
    loader = _separable_loader()
    model = SmallCNN(NUM_CLASSES, width=8)
    before = [p.detach().clone() for p in model.parameters()]
    evaluate(model, loader, nn.CrossEntropyLoss(), NUM_CLASSES)
    for p, q in zip(model.parameters(), before, strict=True):
        assert torch.equal(p.detach(), q)


def test_model_shape_and_size() -> None:
    model = SmallCNN(10)
    assert model(torch.zeros(2, 3, 64, 64)).shape == (2, 10)
    # a guard on the "cheap model" decision: if this grows a lot, milestone B's
    # input-pipeline measurements stop being meaningful
    assert count_parameters(model) < 500_000


def test_metrics_are_wired_into_the_epoch() -> None:
    set_seed(0)
    loader = _separable_loader()
    model = SmallCNN(NUM_CLASSES, width=8)
    metrics, _ = train_one_epoch(
        model,
        loader,
        nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.01),
        NUM_CLASSES,
        sync_each_step=False,
    )
    assert isinstance(metrics, ClassificationMetrics)
    assert metrics.confusion.sum().item() == 64
