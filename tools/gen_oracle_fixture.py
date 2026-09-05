"""Generate the golden fixture that `tests/test_geometry.py` checks our math against.

This is the ONLY code in the repo that uses nuscenes-devkit, and it is deliberately
not part of the package or the test run. The devkit pulls numpy<2 (plus matplotlib,
opencv, scikit-learn, shapely), and we are not downgrading numpy under the training
code for a reference implementation we consult once.

So: run it in a throwaway venv, commit the JSON it produces, and let the test suite
compare our hand-rolled projection against the committed numbers with no devkit
installed at all.

    uv venv /tmp/oracle-venv --python 3.11
    VIRTUAL_ENV=/tmp/oracle-venv uv pip install nuscenes-devkit
    /tmp/oracle-venv/bin/python tools/gen_oracle_fixture.py \
        --dataroot /path/to/nuscenes --out tests/fixtures/box2d_oracle.json

Regenerate only if the fixture's schema changes. If our geometry disagrees with
this file, our geometry is what changed and what should be fixed.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility, view_points

CHANNEL = "CAM_FRONT"
ROUND = 4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--frames", type=int, default=20, help="keyframes to record")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    frames = []
    for sample in nusc.sample[: args.frames]:
        sd_token = sample["data"][CHANNEL]
        _, boxes, intrinsic = nusc.get_sample_data(sd_token, box_vis_level=BoxVisibility.ANY)
        sd = nusc.get("sample_data", sd_token)

        records = []
        for box in boxes:
            corners_cam = box.corners()  # 3x8, camera frame
            pts = view_points(corners_cam, intrinsic, normalize=True)[:2]  # 2x8, pixels
            records.append(
                {
                    "annotation_token": box.token,
                    "category": box.name,
                    # camera-frame geometry, so a failure can be localised to the
                    # transform rather than the projection
                    "center_cam": np.round(box.center, ROUND).tolist(),
                    "wlh": np.round(box.wlh, ROUND).tolist(),
                    "corners_cam": np.round(corners_cam, ROUND).tolist(),
                    "box2d": np.round(
                        [pts[0].min(), pts[1].min(), pts[0].max(), pts[1].max()], ROUND
                    ).tolist(),
                }
            )

        frames.append(
            {
                "sample_token": sample["token"],
                "sample_data_token": sd_token,
                "filename": sd["filename"],
                "width": sd["width"],
                "height": sd["height"],
                "camera_intrinsic": np.round(intrinsic, ROUND).tolist(),
                "boxes": records,
            }
        )

    payload = {
        "_generated_by": "tools/gen_oracle_fixture.py",
        "_source": f"nuscenes-devkit, {args.version}, channel {CHANNEL}",
        "_box_vis_level": "ANY",
        "frames": frames,
    }
    with open(args.out, "w") as fh:
        json.dump(payload, fh)
    n = sum(len(f["boxes"]) for f in frames)
    print(f"wrote {args.out}: {len(frames)} frames, {n} boxes")


if __name__ == "__main__":
    main()
