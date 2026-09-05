"""3D annotation -> 2D image box, written from scratch.

nuScenes annotates objects as 3D boxes in *world* coordinates. To crop an object out
of a camera image we need its 2D extent in *pixel* coordinates, which means walking
the sensor chain:

    world  --(ego_pose)-->  ego vehicle  --(calibrated_sensor)-->  camera  --(K)-->  pixels

Each arrow is a rigid transform stored as a translation vector plus a rotation
quaternion, and each is applied as an *inverse*: `ego_pose` describes where the
vehicle is in the world, so moving a world point into the vehicle's frame means
undoing it. Getting that inversion backwards is the classic silent failure here -
boxes land in plausible-looking but wrong places, crops contain the wrong object,
and the classifier trains happily on mislabeled data.

Nothing in this module depends on nuscenes-devkit. `tests/test_geometry.py` checks
every function here against a fixture the devkit generated, so "plausible" is not
the standard being applied - agreement to within a pixel is.
"""

from __future__ import annotations

import numpy as np

# nuScenes quaternions are stored [w, x, y, z].
Quaternion = np.ndarray  # shape (4,)


def quaternion_to_rotation_matrix(q: Quaternion) -> np.ndarray:
    """Rotation matrix for a unit quaternion [w, x, y, z]."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def quaternion_inverse(q: Quaternion) -> Quaternion:
    """Inverse rotation. For a unit quaternion this is just the conjugate.

    We divide by the squared norm anyway: the stored values are unit-norm only to
    float precision, and the error compounds across two chained transforms.
    """
    w, x, y, z = q
    norm_sq = w * w + x * x + y * y + z * z
    return np.array([w, -x, -y, -z]) / norm_sq


def quaternion_multiply(q1: Quaternion, q2: Quaternion) -> Quaternion:
    """Hamilton product: the rotation q2 followed by the rotation q1."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def box_corners(center: np.ndarray, wlh: np.ndarray, orientation: Quaternion) -> np.ndarray:
    """The box's eight corners as a 3x8 array, in the frame `center` is expressed in.

    nuScenes stores size as (width, length, height) but the box's local axes are
    (x=length, y=width, z=height) - x forward along the object, y left, z up. Mixing
    those up produces boxes that are correct in volume and wrong in shape, which is
    exactly the kind of error that survives a visual glance.
    """
    w, length, h = wlh
    x_corners = length / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    corners = np.vstack((x_corners, y_corners, z_corners))

    corners = quaternion_to_rotation_matrix(orientation) @ corners
    return corners + np.asarray(center).reshape(3, 1)


def transform_to_frame(
    center: np.ndarray,
    orientation: Quaternion,
    frame_translation: np.ndarray,
    frame_rotation: Quaternion,
) -> tuple[np.ndarray, Quaternion]:
    """Move a box into the frame described by (translation, rotation).

    `frame_translation`/`frame_rotation` say where the *frame* sits in the parent
    coordinate system, so entering that frame means subtracting the translation and
    applying the inverse rotation - in that order. Rotating first would rotate about
    the parent origin instead of the frame's own.
    """
    inv = quaternion_inverse(frame_rotation)
    new_center = quaternion_to_rotation_matrix(inv) @ (center - np.asarray(frame_translation))
    return new_center, quaternion_multiply(inv, orientation)


def project_to_image(points_3d: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """Perspective-project camera-frame points (3xN) to pixels (2xN).

    The divide by z is the perspective part. Points at or behind the image plane
    (z <= 0) are meaningless here; callers must filter them out *before* projecting,
    because a negative z silently flips a point to the opposite side of the image
    rather than failing.
    """
    projected = np.asarray(intrinsic) @ points_3d
    return projected[:2] / projected[2:3]


def box_2d_from_corners(corners_2d: np.ndarray) -> tuple[float, float, float, float]:
    """Axis-aligned pixel bounding box (x1, y1, x2, y2) enclosing projected corners."""
    return (
        float(corners_2d[0].min()),
        float(corners_2d[1].min()),
        float(corners_2d[0].max()),
        float(corners_2d[1].max()),
    )


def annotation_to_camera(
    ann_translation: np.ndarray,
    ann_size: np.ndarray,
    ann_rotation: Quaternion,
    ego_translation: np.ndarray,
    ego_rotation: Quaternion,
    sensor_translation: np.ndarray,
    sensor_rotation: Quaternion,
) -> tuple[np.ndarray, np.ndarray]:
    """World-frame annotation -> (center, corners) in the camera frame.

    This is the whole chain in one call: world -> ego -> camera.
    """
    center = np.asarray(ann_translation, dtype=float)
    orientation = np.asarray(ann_rotation, dtype=float)

    center, orientation = transform_to_frame(center, orientation, ego_translation, ego_rotation)
    center, orientation = transform_to_frame(
        center, orientation, sensor_translation, sensor_rotation
    )
    return center, box_corners(center, np.asarray(ann_size, dtype=float), orientation)
