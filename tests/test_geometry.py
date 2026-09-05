"""Check our hand-rolled 3D->2D projection against a devkit-generated fixture.

The fixture (tests/fixtures/box2d_oracle.json) was produced once by
tools/gen_oracle_fixture.py running nuscenes-devkit in a throwaway venv. The devkit
is not installed here and is not a dependency of this project; these numbers are the
reference implementation's answer, frozen.

Without this test the geometry in dtp/geometry.py is unfalsifiable: a transposed
rotation or a swapped width/length still yields boxes that look like boxes, and the
error would surface much later as a classifier that will not converge - at which
point it is indistinguishable from a modeling bug.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from dtp.geometry import annotation_to_camera, box_2d_from_corners, project_to_image
from dtp.nuscenes_tables import NuScenesTables

FIXTURE = Path(__file__).parent / "fixtures" / "box2d_oracle.json"
DATAROOT = os.environ.get(
    "DTP_NUSCENES_ROOT",
    "/Users/nabh/workspace/repos/github/ray-multimodal-pipeline/data/nuscenes",
)

# The fixture stores 4 decimal places; anything at that scale is rounding, not error.
CORNER_TOL = 1e-3
PIXEL_TOL = 1e-2


@pytest.fixture(scope="module")
def oracle() -> dict:
    if not FIXTURE.exists():
        pytest.skip(f"oracle fixture missing: {FIXTURE}")
    with open(FIXTURE) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def tables() -> NuScenesTables:
    if not os.path.isdir(DATAROOT):
        pytest.skip(f"nuScenes data not found at {DATAROOT}; set DTP_NUSCENES_ROOT")
    return NuScenesTables(DATAROOT)


def test_fixture_is_not_empty(oracle: dict) -> None:
    total = sum(len(f["boxes"]) for f in oracle["frames"])
    assert len(oracle["frames"]) >= 20
    assert total > 500, f"fixture only has {total} boxes; regenerate it"


def test_camera_frame_corners_match_devkit(oracle: dict, tables: NuScenesTables) -> None:
    """Every corner of every box, in camera coordinates, to within 1e-3 metres."""
    worst = 0.0
    checked = 0
    for frame in oracle["frames"]:
        sd = tables.get("sample_data", frame["sample_data_token"])
        ego = tables.get("ego_pose", sd["ego_pose_token"])
        cs = tables.get("calibrated_sensor", sd["calibrated_sensor_token"])

        for box in frame["boxes"]:
            ann = tables.get("sample_annotation", box["annotation_token"])
            _, corners = annotation_to_camera(
                ann["translation"],
                ann["size"],
                ann["rotation"],
                ego["translation"],
                ego["rotation"],
                cs["translation"],
                cs["rotation"],
            )
            expected = np.array(box["corners_cam"])
            worst = max(worst, float(np.abs(corners - expected).max()))
            checked += 1

    assert checked > 500
    assert worst < CORNER_TOL, f"worst corner disagreement {worst:.6f} m over {checked} boxes"


def test_projected_2d_boxes_match_devkit(oracle: dict, tables: NuScenesTables) -> None:
    """The pixel boxes we would actually crop with, to within 1e-2 px."""
    worst = 0.0
    checked = 0
    for frame in oracle["frames"]:
        sd = tables.get("sample_data", frame["sample_data_token"])
        ego = tables.get("ego_pose", sd["ego_pose_token"])
        cs = tables.get("calibrated_sensor", sd["calibrated_sensor_token"])
        intrinsic = np.array(cs["camera_intrinsic"])

        for box in frame["boxes"]:
            ann = tables.get("sample_annotation", box["annotation_token"])
            _, corners = annotation_to_camera(
                ann["translation"],
                ann["size"],
                ann["rotation"],
                ego["translation"],
                ego["rotation"],
                cs["translation"],
                cs["rotation"],
            )
            got = np.array(box_2d_from_corners(project_to_image(corners, intrinsic)))
            worst = max(worst, float(np.abs(got - np.array(box["box2d"])).max()))
            checked += 1

    assert checked > 500
    assert worst < PIXEL_TOL, f"worst pixel disagreement {worst:.6f} px over {checked} boxes"


def test_intrinsic_matches_fixture(oracle: dict, tables: NuScenesTables) -> None:
    """Guards against reading the intrinsic from the wrong calibrated_sensor row."""
    for frame in oracle["frames"]:
        sd = tables.get("sample_data", frame["sample_data_token"])
        cs = tables.get("calibrated_sensor", sd["calibrated_sensor_token"])
        assert np.allclose(cs["camera_intrinsic"], frame["camera_intrinsic"], atol=1e-4)
