"""Conversions an adapter usually needs, in plain numpy (no scipy required).

Copy this file next to your adapter. Every function takes arrays with any
number of leading dimensions — e.g. `(n, 2, 4)` quaternions for n actions of
two arms — and returns the same leading shape.

The elrobot_deploy contract (docs/deploying_a_policy/policy_contract.md, "The actions"):
    pose = [x, y, z, qw, qx, qy, qz], metres, unit wxyz quaternion, the FLANGE
    (wrist-yaw link), absolute, in the robot base frame.
"""

from __future__ import annotations

import numpy as np

# --- quaternions ----------------------------------------------------------------


def xyzw_to_wxyz(q):
    """scipy / ROS / many datasets store `[x, y, z, w]`; the contract is wxyz."""
    q = np.asarray(q, dtype=np.float64)
    return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)


def wxyz_to_xyzw(q):
    q = np.asarray(q, dtype=np.float64)
    return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)


def normalize_quat(q):
    """Unit length (the runtime refuses non-unit quaternions; it never fixes them)."""
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def matrix_from_quat(q):
    """wxyz `(..., 4)` -> rotation matrices `(..., 3, 3)`."""
    w, x, y, z = np.moveaxis(normalize_quat(q), -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def quat_from_matrix(R):
    """Rotation matrices `(..., 3, 3)` -> wxyz `(..., 4)`, `w >= 0`.

    Shepperd's method: solve for the largest of w, x, y, z first, then the
    rest from the off-diagonals, so it stays exact at 180-degree rotations."""
    R = np.asarray(R, dtype=np.float64)
    flat = R.reshape(-1, 3, 3)
    out = np.empty((len(flat), 4))
    for i, m in enumerate(flat):
        t = np.array([m[0, 0] + m[1, 1] + m[2, 2], m[0, 0], m[1, 1], m[2, 2]])
        k = int(np.argmax(t))
        if k == 0:
            s = 2.0 * np.sqrt(1.0 + t[0])
            q = [s / 4, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
                 (m[1, 0] - m[0, 1]) / s]
        elif k == 1:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            q = [(m[2, 1] - m[1, 2]) / s, s / 4, (m[0, 1] + m[1, 0]) / s,
                 (m[0, 2] + m[2, 0]) / s]
        elif k == 2:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, s / 4,
                 (m[1, 2] + m[2, 1]) / s]
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                 (m[1, 2] + m[2, 1]) / s, s / 4]
        q = np.asarray(q)
        out[i] = -q if q[0] < 0 else q
    return normalize_quat(out.reshape(R.shape[:-2] + (4,)))


def quat_from_rotvec(v):
    """Axis-angle `(..., 3)` (angle = norm, radians) -> wxyz."""
    v = np.asarray(v, dtype=np.float64)
    angle = np.linalg.norm(v, axis=-1, keepdims=True)
    half = angle / 2
    scale = np.where(angle > 1e-12, np.sin(half) / np.maximum(angle, 1e-12), 0.5)
    return np.concatenate([np.cos(half), v * scale], axis=-1)


def matrix_from_rot6d(d6, *, layout: str):
    """The 6D representation (Zhou et al., 2019) -> rotation matrices.

    `layout="columns"`: `d6 = [R[:, 0], R[:, 1]]`, the first two COLUMNS (the
    paper's). `layout="rows"`: the first two ROWS — pytorch3d's
    `matrix_to_rotation_6d`, and so Diffusion Policy, UMI and the stacks built
    on them. Required, on purpose: the two differ by a transpose, both give
    valid rotations, and a mix-up passes every automatic check. Read the
    training code's conversion to choose.
    """
    d6 = np.asarray(d6, dtype=np.float64)
    a, b = d6[..., :3], d6[..., 3:6]
    e1 = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b - np.sum(e1 * b, axis=-1, keepdims=True) * e1
    e2 = b / np.linalg.norm(b, axis=-1, keepdims=True)
    e3 = np.cross(e1, e2)
    R = np.stack([e1, e2, e3], axis=-1)          # e_i as columns
    if layout == "columns":
        return R
    if layout == "rows":
        return np.swapaxes(R, -1, -2)
    raise ValueError(f"layout must be 'columns' or 'rows', got {layout!r}")


# --- poses ----------------------------------------------------------------------


def pose_to_matrix(pose):
    """`[x, y, z, qw, qx, qy, qz]` `(..., 7)` -> homogeneous `(..., 4, 4)`."""
    pose = np.asarray(pose, dtype=np.float64)
    T = np.zeros(pose.shape[:-1] + (4, 4))
    T[..., :3, :3] = matrix_from_quat(pose[..., 3:])
    T[..., :3, 3] = pose[..., :3]
    T[..., 3, 3] = 1.0
    return T


def matrix_to_pose(T):
    T = np.asarray(T, dtype=np.float64)
    return np.concatenate([T[..., :3, 3], quat_from_matrix(T[..., :3, :3])], -1)


def invert(T):
    T = np.asarray(T, dtype=np.float64)
    R = np.swapaxes(T[..., :3, :3], -1, -2)
    out = np.zeros_like(T)
    out[..., :3, :3] = R
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", R, T[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def tcp_to_flange(tcp_pose, tcp):
    """Poses of a tool-centre point -> the flange poses the contract wants.

    `tcp` is the flange->TCP pose in the flange frame, `(7,)` — for the
    robot's own hand, `self.robot.tcp` (Dex-1: `[0.184759, 0, 0, 1, 0, 0, 0]`).
    If training used a different TCP, use that one: the question is where the
    model's poses were measured, not where the robot's jaws are.
    """
    return matrix_to_pose(pose_to_matrix(tcp_pose) @ invert(pose_to_matrix(tcp)))


def flange_to_tcp(flange_pose, tcp):
    return matrix_to_pose(pose_to_matrix(flange_pose) @ pose_to_matrix(tcp))


def apply_delta(reference, delta, *, frame: str):
    """Relative actions -> absolute poses.

    `reference` `(..., 7)`: the pose the deltas are relative to (e.g.
    `obs.state.flange`, or its TCP), broadcast against `delta` `(..., 7)`.
    `frame="local"`: the delta is expressed in the reference's own frame
    (`T = T_ref @ T_delta`); `"world"`: in the base frame, applied on the left
    for rotation and added for translation (`R = R_delta R_ref`,
    `p = p_ref + p_delta`). Which one training used is a fact to find in the
    dataset code, not to guess.
    """
    Tr, Td = pose_to_matrix(reference), pose_to_matrix(delta)
    if frame == "local":
        return matrix_to_pose(Tr @ Td)
    if frame == "world":
        T = np.zeros(np.broadcast_shapes(Tr.shape, Td.shape))
        T[..., :3, :3] = Td[..., :3, :3] @ Tr[..., :3, :3]
        T[..., :3, 3] = Tr[..., :3, 3] + Td[..., :3, 3]
        T[..., 3, 3] = 1.0
        return matrix_to_pose(T)
    raise ValueError(f"frame must be 'local' or 'world', got {frame!r}")
