"""Minimal nuScenes metadata reader.

The nuScenes "database" is a directory of JSON files where every row carries a
`token` and rows reference each other by token. nuscenes-devkit wraps this with a
lot of convenience, but it also drags in numpy<2, matplotlib, opencv, scikit-learn
and shapely - too much to put underneath a training loop for what is, at bottom,
`json.load` plus a dictionary keyed by token.

This class deliberately implements only what `manifest.py`'s `NuScenesLike`
protocol needs (`sample`, `dataroot`, `get`), plus the tables the crop builder
reads. It is not a devkit replacement and should not grow into one.
"""

from __future__ import annotations

import json
import os
from functools import cached_property

TABLES = (
    "sample",
    "sample_data",
    "sample_annotation",
    "ego_pose",
    "calibrated_sensor",
    "category",
    "instance",
    "sensor",
)


class NuScenesTables:
    """Token-indexed access to the nuScenes JSON metadata tables."""

    def __init__(self, dataroot: str, version: str = "v1.0-mini") -> None:
        self.dataroot = os.path.abspath(dataroot)
        self.version = version
        self._tables: dict[str, list[dict]] = {}
        self._index: dict[str, dict[str, dict]] = {}

        for table in TABLES:
            path = os.path.join(self.dataroot, version, f"{table}.json")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"missing nuScenes table {table}.json at {path}. "
                    f"Is --dataroot pointing at the directory containing {version}/?"
                )
            with open(path) as fh:
                rows = json.load(fh)
            self._tables[table] = rows
            self._index[table] = {row["token"]: row for row in rows}

        self._attach_keyframe_data()

    def _attach_keyframe_data(self) -> None:
        """Populate `sample["data"][channel] -> sample_data token`.

        The raw JSON has no such field: it is a reverse index the devkit computes at
        load time by walking sample_data for keyframes and resolving each one's
        channel through calibrated_sensor -> sensor. We rebuild it here so the
        manifest code copied from the sibling repo works against this loader
        unchanged.
        """
        channel_of_calibrated = {
            cs["token"]: self._index["sensor"][cs["sensor_token"]]["channel"]
            for cs in self._tables["calibrated_sensor"]
        }
        for row in self._tables["sample"]:
            row["data"] = {}
        for sd in self._tables["sample_data"]:
            if not sd["is_key_frame"]:
                continue
            sample = self._index["sample"].get(sd["sample_token"])
            if sample is not None:
                sample["data"][channel_of_calibrated[sd["calibrated_sensor_token"]]] = sd["token"]

    def get(self, table_name: str, token: str) -> dict:
        """Row lookup by token, matching the devkit's signature."""
        try:
            return self._index[table_name][token]
        except KeyError as exc:
            raise KeyError(f"no row {token!r} in table {table_name!r}") from exc

    def table(self, name: str) -> list[dict]:
        return self._tables[name]

    @property
    def sample(self) -> list[dict]:
        return self._tables["sample"]

    @cached_property
    def annotations_by_sample(self) -> dict[str, list[dict]]:
        """sample_token -> its annotations.

        Built once because the crop builder needs it per frame, and scanning 18.5k
        annotations for every one of 404 frames is quadratic for no reason.
        """
        grouped: dict[str, list[dict]] = {}
        for ann in self._tables["sample_annotation"]:
            grouped.setdefault(ann["sample_token"], []).append(ann)
        return grouped
