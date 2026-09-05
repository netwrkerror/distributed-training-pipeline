"""Build the crop manifest: nuScenes 3D annotations -> jsonl of 2D object crops.

    python -m dtp.build_crops --dataroot /path/to/nuscenes --out data/crops.jsonl

This runs once, offline. It writes *records*, not pixels: each line names an image
file and a pixel box inside it. Nothing downstream - the Dataset, the training loop,
B1's throughput harness - reads nuScenes metadata again. That separation is the
whole point of A2: B4 will repack the underlying images into large sequential shards
and only this file's output format has to survive.

Records carry a little more than training strictly needs (scene token, visibility,
lidar point count) so that later work can filter or split without rebuilding.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from dtp.geometry import annotation_to_camera, box_2d_from_corners, project_to_image
from dtp.nuscenes_tables import NuScenesTables

CAMERA_CHANNEL = "CAM_FRONT"

# The standard nuScenes detection classes. nuScenes' own categories are far more
# specific (`vehicle.car`, `human.pedestrian.police_officer`); collapsing them keeps
# the label set small enough for a classifier to learn on 400 frames.
CATEGORY_TO_CLASS = {
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.trailer": "trailer",
    "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.barrier": "barrier",
}

MIN_SIDE_PX = 24
MIN_VISIBLE_FRACTION = 0.6


def _clip(box: tuple[float, float, float, float], w: int, h: int):
    x1, y1, x2, y2 = box
    return (max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2))


def _area(box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def build_crop_records(tables: NuScenesTables, channel: str = CAMERA_CHANNEL) -> list[dict]:
    records: list[dict] = []

    for sample in tables.sample:
        sd_token = sample["data"][channel]
        sd = tables.get("sample_data", sd_token)
        ego = tables.get("ego_pose", sd["ego_pose_token"])
        cs = tables.get("calibrated_sensor", sd["calibrated_sensor_token"])
        intrinsic = np.array(cs["camera_intrinsic"])
        width, height = sd["width"], sd["height"]

        for ann in tables.annotations_by_sample.get(sample["token"], []):
            category = tables.get(
                "category", tables.get("instance", ann["instance_token"])["category_token"]
            )["name"]
            class_name = CATEGORY_TO_CLASS.get(category)
            if class_name is None:
                continue

            _, corners = annotation_to_camera(
                ann["translation"],
                ann["size"],
                ann["rotation"],
                ego["translation"],
                ego["rotation"],
                cs["translation"],
                cs["rotation"],
            )

            # Every corner must be in front of the image plane. project_to_image
            # divides by z, so a single negative-z corner silently mirrors that
            # corner to the far side of the image and inflates the box across the
            # whole frame - a bug that produces valid-looking crops of nothing.
            if (corners[2] <= 0.1).any():
                continue

            box = box_2d_from_corners(project_to_image(corners, intrinsic))
            clipped = _clip(box, width, height)
            full_area = _area(box)
            if full_area <= 0 or _area(clipped) / full_area < MIN_VISIBLE_FRACTION:
                continue
            if (clipped[2] - clipped[0]) < MIN_SIDE_PX or (clipped[3] - clipped[1]) < MIN_SIDE_PX:
                continue

            records.append(
                {
                    "image_path": os.path.join(tables.dataroot, sd["filename"]),
                    "box": [round(v, 2) for v in clipped],
                    "class_name": class_name,
                    "category": category,
                    "sample_token": sample["token"],
                    "scene_token": sample["scene_token"],
                    "annotation_token": ann["token"],
                    "visibility_token": ann.get("visibility_token"),
                    "num_lidar_pts": ann.get("num_lidar_pts"),
                }
            )

    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataroot",
        default=os.environ.get("DTP_NUSCENES_ROOT"),
        help="directory containing v1.0-mini/ and samples/ (or set DTP_NUSCENES_ROOT)",
    )
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--channel", default=CAMERA_CHANNEL)
    ap.add_argument("--out", default="data/crops.jsonl")
    args = ap.parse_args()

    if not args.dataroot:
        ap.error("--dataroot is required (or set DTP_NUSCENES_ROOT)")

    tables = NuScenesTables(args.dataroot, version=args.version)
    records = build_crop_records(tables, channel=args.channel)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    classes = sorted({r["class_name"] for r in records})
    classes_path = os.path.splitext(args.out)[0] + ".classes.json"
    with open(classes_path, "w") as fh:
        json.dump(classes, fh, indent=2)

    counts: dict[str, int] = {}
    for rec in records:
        counts[rec["class_name"]] = counts.get(rec["class_name"], 0) + 1

    print(f"wrote {args.out}: {len(records)} crops from {len(tables.sample)} frames")
    print(f"wrote {classes_path}: {len(classes)} classes")
    for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {name:>22} {count:>6}  ({100 * count / len(records):.1f}%)")


if __name__ == "__main__":
    main()
