"""A2: the crop Dataset's contract, including how it fails.

Most of these tests are about failure modes rather than happy paths. A dataset that
quietly drops a record it cannot read is worse than one that raises: its length
changes, and A5's whole no-duplicate argument is stated in terms of a stable length
and a stable index-to-record mapping.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dtp.dataset import CropDataset, MalformedRecordError

DATAROOT = os.environ.get(
    "DTP_NUSCENES_ROOT",
    "/Users/nabh/workspace/repos/github/ray-multimodal-pipeline/data/nuscenes",
)
REAL_MANIFEST = Path("data/crops.jsonl")


def _write_image(path: Path, size: tuple[int, int] = (200, 120)) -> None:
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(path)


def _write_manifest(path: Path, records: list[dict], classes: list[str] | None = None) -> None:
    with open(path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    if classes is not None:
        with open(path.with_suffix("").with_suffix(".classes.json"), "w") as fh:
            json.dump(classes, fh)


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    img = tmp_path / "frame.jpg"
    _write_image(img)
    manifest = tmp_path / "crops.jsonl"
    _write_manifest(
        manifest,
        [
            {"image_path": str(img), "box": [10, 10, 90, 90], "class_name": "car"},
            {"image_path": str(img), "box": [20, 5, 120, 100], "class_name": "pedestrian"},
        ],
        classes=["car", "pedestrian"],
    )
    return manifest


def test_yields_tensor_and_label(tiny_dataset: Path) -> None:
    ds = CropDataset(tiny_dataset, crop_size=32)
    assert len(ds) == 2
    tensor, label = ds[0]
    assert tensor.shape == (3, 32, 32)
    assert tensor.dtype.is_floating_point
    assert float(tensor.min()) >= 0.0 and float(tensor.max()) <= 1.0
    assert label == 0


def test_label_ids_come_from_the_class_sidecar(tiny_dataset: Path) -> None:
    """Label ids must not depend on which records happen to be present."""
    ds = CropDataset(tiny_dataset)
    assert ds.classes == ["car", "pedestrian"]
    assert ds[0][1] == 0 and ds[1][1] == 1


def test_missing_image_file_raises_with_index_and_path(tmp_path: Path) -> None:
    manifest = tmp_path / "crops.jsonl"
    _write_manifest(
        manifest,
        [{"image_path": str(tmp_path / "gone.jpg"), "box": [0, 0, 20, 20], "class_name": "car"}],
        classes=["car"],
    )
    ds = CropDataset(manifest)
    with pytest.raises(FileNotFoundError) as exc:
        _ = ds[0]
    message = str(exc.value)
    assert "gone.jpg" in message
    assert "record 0" in message


def test_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        CropDataset(tmp_path / "nope.jsonl")


@pytest.mark.parametrize(
    "line,expected",
    [
        ("{not json", "not valid JSON"),
        ('{"box": [0,0,1,1], "class_name": "car"}', "missing fields"),
        ('{"image_path": "x.jpg", "class_name": "car"}', "missing fields"),
        ('{"image_path": "x.jpg", "box": [0,0], "class_name": "car"}', "box must be"),
        ('{"image_path": "x.jpg", "box": [9,0,1,1], "class_name": "car"}', "non-positive extent"),
    ],
)
def test_malformed_record_raises_with_line_number(tmp_path: Path, line: str, expected: str) -> None:
    manifest = tmp_path / "crops.jsonl"
    good = '{"image_path": "a.jpg", "box": [0,0,10,10], "class_name": "car"}'
    manifest.write_text(good + "\n" + line + "\n")
    with pytest.raises(MalformedRecordError) as exc:
        CropDataset(manifest)
    message = str(exc.value)
    assert expected in message
    assert ":2:" in message, f"error should name the offending line: {message}"


def test_empty_manifest_raises(tmp_path: Path) -> None:
    manifest = tmp_path / "crops.jsonl"
    manifest.write_text("\n\n")
    with pytest.raises(MalformedRecordError, match="empty"):
        CropDataset(manifest)


def test_unlisted_class_raises(tmp_path: Path) -> None:
    img = tmp_path / "frame.jpg"
    _write_image(img)
    manifest = tmp_path / "crops.jsonl"
    _write_manifest(
        manifest,
        [{"image_path": str(img), "box": [0, 0, 20, 20], "class_name": "unicorn"}],
        classes=["car"],
    )
    with pytest.raises(MalformedRecordError, match="unicorn"):
        CropDataset(manifest)


@pytest.mark.skipif(not REAL_MANIFEST.exists(), reason="run `make crops` first")
def test_real_manifest_loads_and_decodes() -> None:
    ds = CropDataset(REAL_MANIFEST)
    assert len(ds) > 1000
    assert len(ds.classes) == 10
    tensor, label = ds[len(ds) // 2]
    assert tensor.shape == (3, 64, 64)
    assert 0 <= label < len(ds.classes)
