"""Scene-level train/validation split.

A random split of this dataset would be wrong, and wrong in the direction that
flatters the model. There are roughly ten crops per camera frame, and consecutive
frames within a scene show the same physical objects a fraction of a second apart.
Split by record and near-duplicate crops of the *same car* land on both sides, so
validation accuracy measures memorisation and reports it as generalisation.

Splitting by scene puts every frame of a driving sequence entirely on one side.
nuScenes mini has ten scenes, which is coarse - one scene is 10% of the data and
scenes differ in content - so the split is deterministic and reported, not resampled
until it looks good.
"""

from __future__ import annotations

import hashlib


def _scene_rank(scene_token: str, seed: int) -> str:
    """Stable pseudo-random ordering of scenes that does not depend on dict order."""
    return hashlib.sha256(f"{seed}:{scene_token}".encode()).hexdigest()


def scene_split(
    records: list[dict], val_fraction: float = 0.2, seed: int = 0
) -> tuple[list[int], list[int]]:
    """Return (train_indices, val_indices), disjoint by scene.

    Scenes are assigned whole. The realised validation fraction will not match
    `val_fraction` exactly because scenes differ in size; the caller should log what
    it actually got rather than assume.
    """
    scenes = sorted({r["scene_token"] for r in records})
    if len(scenes) < 2:
        raise ValueError(f"need at least 2 scenes to split, found {len(scenes)}")

    ordered = sorted(scenes, key=lambda s: _scene_rank(s, seed))
    n_val = max(1, round(len(ordered) * val_fraction))
    val_scenes = set(ordered[:n_val])

    train_idx = [i for i, r in enumerate(records) if r["scene_token"] not in val_scenes]
    val_idx = [i for i, r in enumerate(records) if r["scene_token"] in val_scenes]
    if not train_idx or not val_idx:
        raise ValueError("split produced an empty side; adjust val_fraction")
    return train_idx, val_idx


def random_split(
    records: list[dict], val_fraction: float = 0.2, seed: int = 0
) -> tuple[list[int], list[int]]:
    """Record-level split. **Leaks, by construction** - use only as a diagnostic.

    Near-duplicate crops of the same object land on both sides, so validation here
    measures memorisation. Its only legitimate use is as a contrast: if random-split
    validation tracks training loss while scene-split validation does not, the gap
    between them is the size of the generalisation problem, not a bug in the loop.
    """
    import random as _random

    rng = _random.Random(seed)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    n_val = max(1, round(len(idx) * val_fraction))
    return sorted(idx[n_val:]), sorted(idx[:n_val])
