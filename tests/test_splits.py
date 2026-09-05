"""A3: the train/val split must not leak, and must be deterministic."""

from __future__ import annotations

import pytest

from dtp.splits import random_split, scene_split


def _records(n_scenes: int = 10, per_scene: int = 40) -> list[dict]:
    return [
        {"scene_token": f"scene{s}", "sample_token": f"s{s}-{i}", "class_name": "car"}
        for s in range(n_scenes)
        for i in range(per_scene)
    ]


def test_scene_split_shares_no_scene() -> None:
    """The property that makes the number trustworthy."""
    records = _records()
    train_idx, val_idx = scene_split(records)
    train_scenes = {records[i]["scene_token"] for i in train_idx}
    val_scenes = {records[i]["scene_token"] for i in val_idx}
    assert train_scenes & val_scenes == set()
    assert train_scenes | val_scenes == {r["scene_token"] for r in records}


def test_scene_split_shares_no_frame() -> None:
    """Stronger and more direct: no camera frame may appear on both sides."""
    records = _records()
    train_idx, val_idx = scene_split(records)
    train_samples = {records[i]["sample_token"] for i in train_idx}
    val_samples = {records[i]["sample_token"] for i in val_idx}
    assert train_samples & val_samples == set()


def test_scene_split_covers_every_record_exactly_once() -> None:
    records = _records()
    train_idx, val_idx = scene_split(records)
    assert sorted(train_idx + val_idx) == list(range(len(records)))


def test_scene_split_is_deterministic() -> None:
    records = _records()
    assert scene_split(records, seed=0) == scene_split(records, seed=0)


def test_scene_split_seed_changes_the_split() -> None:
    records = _records(n_scenes=10)
    assert scene_split(records, seed=0) != scene_split(records, seed=5)


def test_scene_split_needs_two_scenes() -> None:
    with pytest.raises(ValueError, match="at least 2 scenes"):
        scene_split(_records(n_scenes=1))


def test_random_split_leaks_frames() -> None:
    """Documents *why* random_split is diagnostic-only, as an assertion.

    If this ever stops leaking, the contrast experiment in NOTES.md A3 is no longer
    measuring what it claims to measure.
    """
    records = _records()
    train_idx, val_idx = random_split(records)
    assert sorted(train_idx + val_idx) == list(range(len(records)))
    # same-frame crops land on both sides: exactly the leak scene_split prevents
    scenes_both = {records[i]["scene_token"] for i in train_idx} & {
        records[i]["scene_token"] for i in val_idx
    }
    assert scenes_both, "random split should share scenes; that is the point of it"
