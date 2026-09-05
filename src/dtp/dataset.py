"""Map-style Dataset over the crop manifest.

A *map-style* dataset - one that implements `__len__` and integer `__getitem__` - is
not an arbitrary choice. `DistributedSampler` in A5 works by handing each rank a
strided subset of `range(len(dataset))`, so it needs a stable length and stable
index-to-record mapping. An IterableDataset cannot be sampled that way, which is why
B5 has to shard by hand instead.

Failures here are loud on purpose. A dataset that skips unreadable records changes
its own length, and every invariant A5 asserts - that the union of per-rank indices
covers the dataset exactly once - is stated in terms of that length.

Decoding cost is deliberately left unoptimised: each record re-opens and re-decodes
a full 1600x900 JPEG to take one crop out of it, and there are roughly ten crops per
image. That is the bottleneck milestone B exists to find and fix, so it should be
measured before it is engineered away.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

REQUIRED_FIELDS = ("image_path", "box", "class_name")
DEFAULT_CROP_SIZE = 64


class MalformedRecordError(ValueError):
    """A manifest line is missing fields, is not JSON, or has an unusable box."""


class CropDataset(Dataset):
    """Yields (image_tensor, label) for each crop named in the manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        classes: list[str] | None = None,
        crop_size: int = DEFAULT_CROP_SIZE,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.crop_size = crop_size
        self.records = self._load(self.manifest_path)

        if classes is None:
            classes = self._load_classes(self.manifest_path, self.records)
        self.classes = classes
        self.class_to_index = {name: i for i, name in enumerate(classes)}

        unknown = {r["class_name"] for r in self.records} - set(self.class_to_index)
        if unknown:
            raise MalformedRecordError(
                f"{self.manifest_path}: records reference classes not in the class list: "
                f"{sorted(unknown)}"
            )

    @staticmethod
    def _load(path: Path) -> list[dict]:
        if not path.exists():
            raise FileNotFoundError(f"crop manifest not found: {path}. Run `make crops`.")

        records: list[dict] = []
        with open(path) as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise MalformedRecordError(f"{path}:{lineno}: not valid JSON: {exc}") from exc

                missing = [f for f in REQUIRED_FIELDS if f not in record]
                if missing:
                    raise MalformedRecordError(f"{path}:{lineno}: missing fields {missing}")

                box = record["box"]
                if not (isinstance(box, list) and len(box) == 4):
                    raise MalformedRecordError(
                        f"{path}:{lineno}: box must be [x1, y1, x2, y2], got {box!r}"
                    )
                if box[2] <= box[0] or box[3] <= box[1]:
                    raise MalformedRecordError(
                        f"{path}:{lineno}: box has non-positive extent: {box!r}"
                    )
                records.append(record)

        if not records:
            raise MalformedRecordError(f"{path}: manifest is empty")
        return records

    @staticmethod
    def _load_classes(manifest_path: Path, records: list[dict]) -> list[str]:
        sidecar = manifest_path.with_suffix("").with_suffix(".classes.json")
        if sidecar.exists():
            with open(sidecar) as fh:
                return json.load(fh)
        # Deriving the class list from the data makes label ids depend on which
        # records happen to be present, so this is a fallback, not the default path.
        return sorted({r["class_name"] for r in records})

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        path = record["image_path"]

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"record {index} of {self.manifest_path} references a missing image: {path}"
            )
        try:
            with Image.open(path) as img:
                crop = img.convert("RGB").crop(tuple(record["box"]))
                crop = crop.resize((self.crop_size, self.crop_size), Image.BILINEAR)
                array = np.asarray(crop, dtype=np.float32) / 255.0
        except UnidentifiedImageError as exc:
            raise MalformedRecordError(
                f"record {index} of {self.manifest_path}: {path} is not a readable image"
            ) from exc

        # HWC uint8-derived float -> CHW, which is what conv layers expect.
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return tensor, self.class_to_index[record["class_name"]]
