"""The ElRobot as a MuJoCo model, built from our description (description/elrobot_ours).

One model for everything that needs kinematics on this machine: the IK (`runtime/ik/tcp_qp.py`)
and, later, the state reader's FK. Two models would let the pose the operator sees and the pose
the IK solves for drift apart silently.

    model = ElRobotModel()              # loads description/elrobot_ours/elrobot.urdf
    T = model.fk(q7)                    # 4x4 TCP pose in the base frame
    model.limits()                      # (7, 2) joint limits, radians

What is changed on the way in, and why:
  * **the gripper has no DOF here.** `rev_motor_08` and the two jaw slides become fixed joints
    at the fully open pose. The TCP sits on `Gripper_Base`, so it does not depend on the
    gripper, and MuJoCo would ignore the URDF's mimic tags and give each jaw a free slide.
    The gripper is commanded separately, in servo angle.
  * **fixed links are kept as bodies** (`fusestatic="false"`), so `tcp` and
    `wrist_camera_optical` exist as frames the IK and FK can name.
  * **meshes are passed in memory**, keyed by file name, so the URDF's relative paths
    (`../elrobot_upstream/assets/...`) need no working-directory games.

Conventions: the base frame is `base_link` (z up, the arm reaches along +y at zero). Poses are
`[x y z qw qx qy qz]`, as everywhere in elrobot_deploy. Joint vectors are the 7 arm joints in
order, `rev_motor_01` (base yaw) to `rev_motor_07` (wrist roll), in radians, in the URDF's
convention (`calibration/calibration.json` maps servo readings onto it).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

#: The generated description: `description/elrobot_ours/make_description.py`.
DEFAULT_URDF = Path(__file__).resolve().parents[4] / "description" / "elrobot_ours" / "elrobot.urdf"

#: The 7 arm joints, in the order every joint vector uses.
ARM_JOINTS = tuple(f"rev_motor_0{i}" for i in range(1, 8))
#: The gripper joint and the jaw slides it drives: frozen open in this model.
GRIPPER_JOINTS = ("rev_motor_08", "rev_motor_08_1", "rev_motor_08_2")
TCP = "tcp"
CAMERA = "wrist_camera_optical"


def mjcf_ready_urdf(urdf: Path) -> tuple[str, dict[str, bytes]]:
    """Our URDF text, prepared for MuJoCo, and its meshes keyed by the name the text uses."""
    urdf = Path(urdf).resolve()
    root = ET.parse(urdf).getroot()
    for name in GRIPPER_JOINTS:
        j = root.find(f"joint[@name='{name}']")
        j.set("type", "fixed")
        for tag in ("axis", "limit", "mimic", "dynamics"):
            for el in j.findall(tag):
                j.remove(el)
    assets: dict[str, bytes] = {}
    for mesh in root.iter("mesh"):
        path = (urdf.parent / mesh.get("filename")).resolve()
        mesh.set("filename", path.name)
        assets[path.name] = path.read_bytes()
    ext = ET.SubElement(root, "mujoco")
    ET.SubElement(ext, "compiler", fusestatic="false", discardvisual="true",
                  strippath="false", balanceinertia="true")
    return ET.tostring(root, encoding="unicode"), assets


class ElRobotModel:
    """MuJoCo model + data for the 7-DoF arm, and the few lookups everything else needs."""

    def __init__(self, urdf: Path | str = DEFAULT_URDF):
        import mujoco

        self._mujoco = mujoco
        self.urdf = Path(urdf)
        xml, assets = mjcf_ready_urdf(self.urdf)
        self.model = mujoco.MjModel.from_xml_string(xml, assets)
        self.data = mujoco.MjData(self.model)
        m = self.model
        if m.nq != len(ARM_JOINTS):
            raise ValueError(f"expected {len(ARM_JOINTS)} DOF after freezing the gripper, got {m.nq}")
        self.qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in ARM_JOINTS])
        self.dadr = np.array([m.jnt_dofadr[m.joint(n).id] for n in ARM_JOINTS])
        self.tcp_id = m.body(TCP).id
        self.camera_id = m.body(CAMERA).id
        self.base_id = m.body("base_link").id

    # --- configuration ----------------------------------------------------------------------

    def set_q(self, q7) -> None:
        """Place the arm and refresh every body pose."""
        self.data.qpos[self.qadr] = np.asarray(q7, dtype=np.float64)
        self._mujoco.mj_kinematics(self.model, self.data)

    def q(self) -> np.ndarray:
        return self.data.qpos[self.qadr].copy()

    def limits(self) -> np.ndarray:
        """(7, 2) lower/upper joint limits from the URDF (our calibrated safe ranges), radians."""
        m = self.model
        return np.array([m.jnt_range[m.joint(n).id] for n in ARM_JOINTS], dtype=np.float64)

    def velocity_limits(self) -> np.ndarray:
        """(7,) the URDF's joint velocity limits, rad/s (MuJoCo drops them, so read the file)."""
        root = ET.parse(self.urdf).getroot()
        return np.array([float(root.find(f"joint[@name='{n}']/limit").get("velocity"))
                         for n in ARM_JOINTS])

    # --- poses -------------------------------------------------------------------------------

    def body_pose(self, body_id: int) -> np.ndarray:
        """4x4 pose of a body in the base frame, at the current configuration."""
        T = np.eye(4)
        T[:3, :3] = self.data.xmat[body_id].reshape(3, 3)
        T[:3, 3] = self.data.xpos[body_id]
        B = np.eye(4)
        B[:3, :3] = self.data.xmat[self.base_id].reshape(3, 3)
        B[:3, 3] = self.data.xpos[self.base_id]
        return np.linalg.inv(B) @ T

    def fk(self, q7=None, body: str = TCP) -> np.ndarray:
        """4x4 pose of `body` (the TCP by default) in the base frame, at `q7` (or as placed)."""
        if q7 is not None:
            self.set_q(q7)
        return self.body_pose(self.model.body(body).id)


# --- pose conversions (the contract's [x y z qw qx qy qz]) ------------------------------------

def mat_to_pose(T: np.ndarray) -> np.ndarray:
    import mujoco
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(T[:3, :3]).ravel())
    if q[0] < 0:
        q = -q
    return np.r_[T[:3, 3], q]


def pose_to_mat(p) -> np.ndarray:
    import mujoco
    p = np.asarray(p, dtype=np.float64)
    R = np.empty(9)
    mujoco.mju_quat2Mat(R, p[3:] / np.linalg.norm(p[3:]))
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R.reshape(3, 3), p[:3]
    return T


__all__ = [
    "ARM_JOINTS",
    "CAMERA",
    "DEFAULT_URDF",
    "GRIPPER_JOINTS",
    "TCP",
    "ElRobotModel",
    "mat_to_pose",
    "mjcf_ready_urdf",
    "pose_to_mat",
]
