"""Astra's frames and units, and the conversions to and from the runtime's.

What Astra sees (docs/astra.md, "Conventions"):

    base frame   x forward, y left, z up (REP-103): `base_link` itself (our URDF's, since
                 2026-10-18). `TaskFrame` maps `[teleop] base_forward / base_up` onto it, the
                 identity with the shipped profile; it stays so that a description with other
                 axes needs only those two keys.
    virtual TCP  the URDF's `tcp` link: x out of the jaws (the approach direction), y the
                 jaw-opening direction, z = x cross y. At the URDF zero pose its axes ARE the
                 base frame's (roll = pitch = yaw = 0); at home it is pitched 45 deg down.
    rotation     roll / pitch / yaw in degrees about fixed x, y, z: R = Rz(yaw) Ry(pitch)
                 Rx(roll) (scipy `from_euler("xyz", ..., degrees=True)`).
    gripper      the opening in millimetres (`GripperMap`); never the servo angle.

Everything here is numpy, no MuJoCo: the trajectory law, the harness and the tests use it.
"""

from __future__ import annotations

import math

import numpy as np

from elrobot_deploy.robot.transforms import mat_from_quat, quat_from_mat


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def mat_from_rpy(rpy_deg) -> np.ndarray:
    """`[roll, pitch, yaw]` degrees -> R = Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = np.radians(np.asarray(rpy_deg, dtype=np.float64).reshape(3))
    return rot_z(y) @ rot_y(p) @ rot_x(r)


def rpy_from_mat(R) -> np.ndarray:
    """The inverse of `mat_from_rpy`, degrees; pitch in [-90, 90]."""
    R = np.asarray(R, dtype=np.float64)
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    if abs(R[2, 0]) < 1 - 1e-9:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:                                   # gimbal lock: put it all in yaw
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    return np.degrees([roll, pitch, yaw])


def rotvec_from_mat(R) -> np.ndarray:
    """Axis x angle (radians) of a rotation, angle in [0, pi]."""
    R = np.asarray(R, dtype=np.float64)
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2))
    angle = math.acos(c)
    if angle < 1e-9:
        return np.zeros(3)
    if angle > math.pi - 1e-6:
        # Near 180 deg the antisymmetric part vanishes: take the axis from R + I.
        M = (R + np.eye(3)) / 2
        axis = M[:, int(np.argmax(np.diag(M)))]
        return axis / np.linalg.norm(axis) * angle
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w / (2 * math.sin(angle)) * angle


def mat_from_rotvec(r) -> np.ndarray:
    """Rodrigues."""
    r = np.asarray(r, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(r))
    if angle < 1e-12:
        return np.eye(3)
    k = r / angle
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def rotation_angle(R) -> float:
    """The angle of a rotation, radians."""
    return math.acos(max(-1.0, min(1.0, (np.trace(np.asarray(R)) - 1) / 2)))


class TaskFrame:
    """Astra's base frame (x forward, y left, z up) against the runtime's `base_link`.

    `M` has the task axes as rows, in base_link coordinates, so `p_task = M p_base` and a TCP
    orientation `R_base` (columns = the TCP's axes in base_link) reads `M R_base` in the task
    frame. The TCP's own axes never change: only the frame they are written in.
    """

    def __init__(self, base_forward=(1.0, 0.0, 0.0), base_up=(0.0, 0.0, 1.0)):
        f = np.asarray(base_forward, dtype=np.float64)
        u = np.asarray(base_up, dtype=np.float64)
        f, u = f / np.linalg.norm(f), u / np.linalg.norm(u)
        self.M = np.vstack([f, np.cross(u, f), u])

    @classmethod
    def from_config(cls, cfg) -> TaskFrame:
        return cls(cfg.teleop.base_forward, cfg.teleop.base_up)

    def to_task(self, pose7) -> tuple[np.ndarray, np.ndarray]:
        """A runtime pose `[x y z qw qx qy qz]` (base_link) -> (xyz, R) in the task frame."""
        pose7 = np.asarray(pose7, dtype=np.float64)
        return self.M @ pose7[:3], self.M @ mat_from_quat(pose7[3:])

    def to_base(self, p_task, R_task) -> np.ndarray:
        """(xyz, R) in the task frame -> a runtime pose, base_link."""
        p = self.M.T @ np.asarray(p_task, dtype=np.float64)
        return np.r_[p, quat_from_mat(self.M.T @ np.asarray(R_task, dtype=np.float64))]

    def describe(self, pose7) -> tuple[np.ndarray, np.ndarray]:
        """A runtime pose -> (xyz m, rpy deg) as Astra reads it."""
        p, R = self.to_task(pose7)
        return p, rpy_from_mat(R)


class GripperMap:
    """The jaw opening in mm against the gripper servo angle (rad), linear.

    `q_open` is the open end of the calibrated range (the servo's lower limit, 3 deg) and
    `q_closed` the closed end (139.8 deg); `open_mm` / `closed_mm` the inner gap there
    (`[astra] gripper_open_mm`, measured with a ruler: 55 mm).
    """

    def __init__(self, q_open: float, q_closed: float, open_mm: float, closed_mm: float = 0.0):
        self.q_open, self.q_closed = float(q_open), float(q_closed)
        self.open_mm, self.closed_mm = float(open_mm), float(closed_mm)

    @classmethod
    def from_config(cls, cfg) -> GripperMap:
        from elrobot_deploy.runtime.shared_memory import schema
        g = schema.GRIPPER
        return cls(cfg.limits.lower[g], cfg.limits.upper[g], cfg.astra.gripper_open_mm,
                   cfg.astra.gripper_closed_mm)

    def mm(self, q):
        """Servo angle (rad) -> opening (mm), clipped to the range."""
        f = (np.asarray(q, dtype=np.float64) - self.q_open) / (self.q_closed - self.q_open)
        return self.open_mm + np.clip(f, 0.0, 1.0) * (self.closed_mm - self.open_mm)

    def q(self, mm):
        """Opening (mm) -> servo angle (rad), clipped to the range."""
        f = (np.asarray(mm, dtype=np.float64) - self.open_mm) / (self.closed_mm - self.open_mm)
        return self.q_open + np.clip(f, 0.0, 1.0) * (self.q_closed - self.q_open)


__all__ = ["GripperMap", "TaskFrame", "mat_from_rotvec", "mat_from_rpy", "rot_x", "rot_y",
           "rot_z", "rotation_angle", "rotvec_from_mat", "rpy_from_mat"]
