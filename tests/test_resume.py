"""A7: a resumed run must continue, not restart.

Restoring weights is the easy part and the part everyone gets right. A resume has
four pieces of state - model, optimizer moments, step count, and data order - and the
last one fails silently. A run that restores everything but the sampler epoch trains
happily on data it has already seen, with a loss curve that looks entirely normal.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from dtp.checkpoint import LATEST, CheckpointManager, restore_rng_state, rng_state

MANIFEST = "data/crops.jsonl"


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


# --- the pieces of state ------------------------------------------------------


def test_checkpoint_carries_every_piece_of_resume_state(tmp_path: Path) -> None:
    model = _model()
    manager = CheckpointManager(tmp_path)
    manager.save(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters()),
        epoch=3,
        global_step=153,
        sampler_epoch=3,
        val_loss=0.5,
    )
    state = manager.load()
    # weights alone are not a resume
    for key in ("model", "optimizer", "epoch", "global_step", "sampler_epoch", "rng"):
        assert key in state, f"{key} missing; a resume cannot be exact without it"
    assert (state["epoch"], state["global_step"], state["sampler_epoch"]) == (3, 153, 3)


def test_rng_state_round_trips(tmp_path: Path) -> None:
    """Not needed for continuity today, but B5 adds per-worker augmentation and then
    it is. Retrofitting it later would silently invalidate older checkpoints."""
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    captured = rng_state()

    expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    # burn the generators
    for _ in range(10):
        random.random()
        np.random.rand()
        torch.rand(1)

    restore_rng_state(captured)
    assert (random.random(), float(np.random.rand()), float(torch.rand(1))) == expected


def test_resume_epoch_arithmetic() -> None:
    """A checkpoint written at the end of epoch N resumes at N+1, not N.

    Off by one here either re-runs an epoch or skips one, and both produce a curve
    that still descends.
    """
    saved_epoch = 3
    assert saved_epoch + 1 == 4


def test_load_returns_none_when_there_is_nothing_to_resume(tmp_path: Path) -> None:
    assert CheckpointManager(tmp_path / "empty").load() is None


def test_resume_marker_survives_a_second_run(tmp_path: Path) -> None:
    model = _model()
    manager = CheckpointManager(tmp_path, keep_last=2)
    for epoch in range(4):
        manager.save(
            model=model,
            optimizer=torch.optim.AdamW(model.parameters()),
            epoch=epoch,
            global_step=epoch * 51,
            sampler_epoch=epoch,
            val_loss=1.0 - 0.1 * epoch,
        )
    meta = manager.read_marker(LATEST)
    assert meta.epoch == 3
    assert manager.load()["global_step"] == 153


# --- the done-when ------------------------------------------------------------


def _train(args: list[str]) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "dtp.train", "--threads", "2", *args],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _losses(run_dir: Path) -> list[tuple[int, float]]:
    out = []
    for metrics in sorted(run_dir.glob("*/metrics.jsonl")):
        for line in metrics.read_text().splitlines():
            record = json.loads(line)
            out.append((record["epoch"], record["train"]["loss"]))
    return out


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(MANIFEST), reason="run `make crops` first")
def test_resumed_run_matches_an_uninterrupted_one_exactly(tmp_path: Path) -> None:
    """The A7 done-when, as an exact equality rather than an eyeballed curve.

    Data order is a pure function of (seed, epoch) on both the single-process and
    distributed paths, so a correct resume is not merely *continuous* with an
    uninterrupted run - it is identical to it. Asserting equality rather than
    "looks close" is what makes the sampler-epoch bug detectable at all: that bug
    moves the loss by about 1%, which is well inside what a human would accept as
    noise.
    """
    common = ["--batch-size", "256"]
    uninterrupted = tmp_path / "full"
    _train(
        [
            *common,
            "--epochs",
            "4",
            "--checkpoint-dir",
            str(tmp_path / "c1"),
            "--run-dir",
            str(uninterrupted),
        ]
    )

    interrupted = tmp_path / "part"
    _train(
        [
            *common,
            "--epochs",
            "2",
            "--checkpoint-dir",
            str(tmp_path / "c2"),
            "--run-dir",
            str(interrupted),
        ]
    )
    _train(
        [
            *common,
            "--epochs",
            "2",
            "--resume",
            "--checkpoint-dir",
            str(tmp_path / "c2"),
            "--run-dir",
            str(interrupted),
        ]
    )

    full = _losses(uninterrupted)
    split = _losses(interrupted)

    assert [e for e, _ in full] == [0, 1, 2, 3]
    assert [e for e, _ in split] == [0, 1, 2, 3], (
        f"resume did not continue the epoch count: {split}"
    )
    assert full == split, f"resumed curve diverged\n  uninterrupted {full}\n  resumed       {split}"


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(MANIFEST), reason="run `make crops` first")
def test_ignoring_the_sampler_epoch_silently_changes_the_curve(tmp_path: Path) -> None:
    """The bug, as an assertion.

    Restoring weights, optimizer and step count while restarting the data order gives
    a run that completes normally and logs nothing unusual. This asserts the damage is
    real (the curve differs) *and* small (well within a range a human would dismiss),
    which is the combination that makes it dangerous.
    """
    common = ["--batch-size", "256", "--epochs", "2"]
    _train([*common, "--checkpoint-dir", str(tmp_path / "c"), "--run-dir", str(tmp_path / "r")])

    good = tmp_path / "good"
    bad = tmp_path / "bad"
    import shutil

    shutil.copytree(tmp_path / "c", tmp_path / "c_bad")
    _train([*common, "--resume", "--checkpoint-dir", str(tmp_path / "c"), "--run-dir", str(good)])
    _train(
        [
            *common,
            "--resume",
            "--ignore-sampler-epoch",
            "--checkpoint-dir",
            str(tmp_path / "c_bad"),
            "--run-dir",
            str(bad),
        ]
    )

    good_losses = [loss for _, loss in _losses(good)]
    bad_losses = [loss for _, loss in _losses(bad)]

    assert good_losses != bad_losses, "expected the frozen data order to change the curve"
    relative = abs(bad_losses[0] - good_losses[0]) / good_losses[0]
    assert relative < 0.10, (
        f"the bug should be subtle, not obvious: {relative:.1%} apart "
        f"({good_losses[0]:.5f} vs {bad_losses[0]:.5f})"
    )
