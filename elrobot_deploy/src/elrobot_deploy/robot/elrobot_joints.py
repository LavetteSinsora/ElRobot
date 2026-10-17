"""The ElRobot's 8 commanded joints and their limits, read from our URDF without MuJoCo.

The executor (`runtime/processes/p7_bus.py`) needs the joint order and the limits and nothing
else from the robot description, and like g1d's P7 it must not import the IK stack. So this
reads the URDF's `<limit>` tags with ElementTree; `robot/elrobot_model.py` is the MuJoCo side.

Order is the servo order, base to gripper: index `i` is servo ID `i + 1` and URDF joint
`rev_motor_0{i+1}`. Angles are radians in the URDF's convention (`elrobot/calibration.py` maps
servo readings onto it); for the gripper 0 is fully open and + closes.

The position limits in `description/elrobot_ours/elrobot.urdf` ARE the calibrated safe ranges
(`make_description.py` writes them from `calibration/calibration.json`), so the URDF is the one
source for both the IK and the executor.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from elrobot_deploy.robot.elrobot_model import DEFAULT_URDF

JOINT_NAMES = tuple(f"rev_motor_0{i}" for i in range(1, 9))
N_JOINTS = len(JOINT_NAMES)          # 7 arm joints + the gripper
GRIPPER = 7                          # index of the gripper (servo 8)
SERVO_IDS = tuple(range(1, N_JOINTS + 1))


@dataclass(frozen=True)
class JointLimits:
    lower: np.ndarray        # (8,) rad
    upper: np.ndarray        # (8,) rad
    velocity: np.ndarray     # (8,) rad/s, the URDF's own numbers (unscaled)

    def clip(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self.lower, self.upper)


def urdf_limits(urdf: Path | str = DEFAULT_URDF) -> JointLimits:
    root = ET.parse(urdf).getroot()
    rows = []
    for name in JOINT_NAMES:
        lim = root.find(f"joint[@name='{name}']/limit")
        if lim is None:
            raise ValueError(f"{urdf}: joint {name} has no <limit>")
        rows.append((float(lim.get("lower")), float(lim.get("upper")),
                     float(lim.get("velocity"))))
    a = np.array(rows, dtype=np.float64)
    return JointLimits(lower=a[:, 0], upper=a[:, 1], velocity=a[:, 2])


__all__ = ["GRIPPER", "JOINT_NAMES", "N_JOINTS", "SERVO_IDS", "JointLimits", "urdf_limits"]
