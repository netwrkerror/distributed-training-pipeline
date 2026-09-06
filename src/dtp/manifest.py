"""Manifest stage: enumerate nuScenes-mini samples into per-record file paths.

Copied from the sibling repo `ray-multimodal-pipeline`
(src/ray_multimodal_pipeline/manifest.py) with only this note added. Two personal
repos do not justify a shared package, and the duplication is deliberate: if this
file needs to diverge for training, it should diverge freely.

Kept because it is the clean expression of "the manifest is the interface" - the
storage layout underneath can change in B4 without the training code noticing.

Runs driver-side against the small `sample` / `sample_data` metadata tables (not the
actual image/lidar bytes), producing a lightweight list that the ingest stage can
turn into a Ray Dataset lazily.
"""

from __future__ import annotations

import os
from typing import Protocol

CAMERA_CHANNEL = "CAM_FRONT"
LIDAR_CHANNEL = "LIDAR_TOP"


class NuScenesLike(Protocol):
    """Structural subset of `nuscenes.nuscenes.NuScenes` this module depends on."""

    sample: list[dict]
    dataroot: str

    def get(self, table_name: str, token: str) -> dict: ...


def build_manifest(nusc: NuScenesLike) -> list[dict]:
    """Build one record per sample: token, camera/lidar file paths, timestamp.

    Paths are absolutized here on the driver: Ray workers execute from their own
    runtime working directory, so a path relative to the driver's cwd would not
    resolve inside the worker.
    """
    dataroot = os.path.abspath(nusc.dataroot)
    manifest = []
    for sample in nusc.sample:
        cam_data = nusc.get("sample_data", sample["data"][CAMERA_CHANNEL])
        lidar_data = nusc.get("sample_data", sample["data"][LIDAR_CHANNEL])
        manifest.append(
            {
                "sample_token": sample["token"],
                "cam_front_path": os.path.join(dataroot, cam_data["filename"]),
                "lidar_top_path": os.path.join(dataroot, lidar_data["filename"]),
                "timestamp": sample["timestamp"],
            }
        )
    return manifest
