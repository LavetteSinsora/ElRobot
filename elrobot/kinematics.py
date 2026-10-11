"""Minimal URDF forward kinematics.

Each joint contributes  T = Translate(origin.xyz) * Rot(origin.rpy) * Motion(axis, q),
where Motion is a rotation about `axis` (revolute) or a translation along it (prismatic).
A link's pose in the base frame is the product of these along the path from the root.

    uv run python -m elrobot.kinematics                       # joint positions at q = 0
    uv run python -m elrobot.kinematics --deg 0 -90 90 0 0 0 0 0
"""

import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

UPSTREAM_URDF = Path(__file__).resolve().parent.parent / "description/elrobot_upstream/elrobot_follower.urdf"
ARM_JOINTS = [f"rev_motor_0{i}" for i in range(1, 9)]


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    origin: np.ndarray  # 4x4
    axis: np.ndarray
    lower: float | None = None
    upper: float | None = None
    mimic: tuple[str, float, float] | None = None  # (joint, multiplier, offset)


def rpy_matrix(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def axis_angle_matrix(axis: np.ndarray, q: float) -> np.ndarray:
    x, y, z = axis
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(q) * k + (1 - np.cos(q)) * k @ k


def _floats(s: str | None, default: str) -> np.ndarray:
    return np.array([float(v) for v in (s or default).split()])


class Robot:
    def __init__(self, urdf: Path = UPSTREAM_URDF):
        root = ET.parse(urdf).getroot()
        self.joints: dict[str, Joint] = {}
        for j in root.findall("joint"):
            o = j.find("origin")
            T = np.eye(4)
            if o is not None:
                T[:3, :3] = rpy_matrix(*_floats(o.get("rpy"), "0 0 0"))
                T[:3, 3] = _floats(o.get("xyz"), "0 0 0")
            a = j.find("axis")
            axis = _floats(a.get("xyz") if a is not None else None, "1 0 0")
            joint = Joint(j.get("name"), j.get("type"), j.find("parent").get("link"),
                          j.find("child").get("link"), T, axis / np.linalg.norm(axis))
            if (lim := j.find("limit")) is not None:
                joint.lower, joint.upper = float(lim.get("lower", 0)), float(lim.get("upper", 0))
            if (m := j.find("mimic")) is not None:
                joint.mimic = (m.get("joint"), float(m.get("multiplier", 1)), float(m.get("offset", 0)))
            self.joints[joint.name] = joint
        self.by_child = {j.child: j for j in self.joints.values()}
        children = set(self.by_child)
        self.root = next(j.parent for j in self.joints.values() if j.parent not in children)

    def joint_transform(self, joint: Joint, q: dict[str, float]) -> np.ndarray:
        if joint.mimic:
            src, mult, off = joint.mimic
            value = q.get(src, 0.0) * mult + off
        else:
            value = q.get(joint.name, 0.0)
        M = np.eye(4)
        if joint.type in ("revolute", "continuous"):
            M[:3, :3] = axis_angle_matrix(joint.axis, value)
        elif joint.type == "prismatic":
            M[:3, 3] = joint.axis * value
        return joint.origin @ M

    def fk(self, q: dict[str, float], link: str) -> np.ndarray:
        """Pose of `link` in the root frame, as a 4x4 matrix."""
        T = np.eye(4)
        while link != self.root:
            joint = self.by_child[link]
            T = self.joint_transform(joint, q) @ T
            link = joint.parent
        return T

    def joint_frame(self, q: dict[str, float], joint_name: str) -> np.ndarray:
        """Pose of a joint's frame (= its child link's frame) in the root frame."""
        return self.fk(q, self.joints[joint_name].child)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deg", type=float, nargs=8, metavar="Q", help="joint angles 1-8 in degrees")
    ap.add_argument("--urdf", type=Path, default=UPSTREAM_URDF)
    args = ap.parse_args(argv)

    robot = Robot(args.urdf)
    q = dict(zip(ARM_JOINTS, np.radians(args.deg))) if args.deg else {}
    print(f"{'joint':<14} {'q deg':>7}  {'x':>7} {'y':>7} {'z':>7}  (mm, base frame: z up, arm reaches +y)")
    for name in ARM_JOINTS:
        x, y, z = robot.joint_frame(q, name)[:3, 3] * 1000
        j = robot.joints[name]
        flag = "" if j.lower <= q.get(name, 0) <= j.upper else "  OUTSIDE LIMITS"
        print(f"{name:<14} {np.degrees(q.get(name, 0)):>7.1f}  {x:>7.1f} {y:>7.1f} {z:>7.1f}{flag}")


if __name__ == "__main__":
    main()
