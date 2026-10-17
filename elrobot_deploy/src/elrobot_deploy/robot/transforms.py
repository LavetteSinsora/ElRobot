"""Rotations and poses in numpy, in the one convention used everywhere in elrobot_deploy.

A pose is `[x y z qw qx qy qz]` (metres, unit quaternion w first); a rotation matrix is 3x3,
columns = the rotated frame's axes in the outer frame. No MuJoCo here: the phone source and the
tests use this without the IK stack.
"""

from __future__ import annotations

import numpy as np

IDENTITY_POSE = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])


def quat_mul(a, b) -> np.ndarray:
    """Hamilton product, wxyz. `quat_mul(a, b)` is "rotate by b, then by a"."""
    aw, ax, ay, az = np.asarray(a, dtype=np.float64)
    bw, bx, by, bz = np.asarray(b, dtype=np.float64)
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_conj(q) -> np.ndarray:
    """The inverse of a UNIT quaternion."""
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalise(q) -> np.ndarray:
    """Unit length; raises on a degenerate quaternion rather than inventing one."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if not np.isfinite(n) or n < 1e-9:
        raise ValueError(f"degenerate quaternion: {q}")
    return q / n


def quat_angle(q) -> float:
    """The rotation angle of a unit quaternion, radians in [0, pi]."""
    w = abs(float(np.asarray(q, dtype=np.float64)[0]))
    return 2.0 * float(np.arccos(min(1.0, w)))


def quat_from_mat(R) -> np.ndarray:
    """3x3 -> wxyz, w >= 0 (Shepperd's method: stable at 180 degrees too)."""
    R = np.asarray(R, dtype=np.float64)
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                      (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] >= R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                      0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                      (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    q = q / np.linalg.norm(q)
    return -q if q[0] < 0 else q


def mat_from_quat(q) -> np.ndarray:
    """wxyz -> 3x3."""
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def pose_to_mat(pose7) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = mat_from_quat(pose7[3:])
    T[:3, 3] = pose7[:3]
    return T


def mat_to_pose(T) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    return np.r_[T[:3, 3], quat_from_mat(T[:3, :3])]


def pose_mul(a, b) -> np.ndarray:
    """`a @ b` for two poses."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.r_[a[:3] + mat_from_quat(a[3:]) @ b[:3], quat_mul(a[3:], b[3:])]


def pose_inv(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    qi = quat_conj(a[3:])
    return np.r_[-(mat_from_quat(qi) @ a[:3]), qi]


__all__ = ["IDENTITY_POSE", "mat_from_quat", "mat_to_pose", "pose_inv", "pose_mul",
           "pose_to_mat", "quat_angle", "quat_conj", "quat_from_mat", "quat_mul",
           "quat_normalise"]
