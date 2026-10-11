"""Kinematics shared by P1's FK and P6's IK — and the reason they must agree.

`place()` is the only function in the system that decides where the trunk is
when a flange pose is computed. P1 calls it with the **measured** waist to
produce the observation's flange pose; P6 calls it with the trunk it solves
against before solving. If those two ever diverge, the flange pose
the policy observes and the flange pose the IK solves for are expressed against
different trunk configurations, and the difference is a silent bias in metres —
not a failure anything reports.

That is why the placement lives here rather than once in each process.

P1's waist is always the *measured* one, and on the reference robot the bow
sits at a permanent 0.039 rad (2.24°) that kp=300 never closes — so a caller
that placed the COMMANDED waist would diverge from P1 by exactly the bias this
docstring warns about, whenever the trunk is not where it was commanded.
Which axis that droop lands on is `WAIST_JOINT_NAMES`' business; that it lands
at all is a property of the hardware, and `home_tol_waist` is what decides how
much of it a run will accept.

**The model is the G1-D** (`g1d_deploy.robot.g1d_model`), and the base frame every pose in
this system is expressed against is the lift column, not a pelvis — the G1-D
has none. See `g1d_model.BASE_BODY` for why that link and not another, and
`g1d_model.PELVIS_FROM_TORSO` for why the frame is nonetheless placed exactly
where the humanoid G1's pelvis is: with the waist at zero this reports the same
flange pose a humanoid model does, to well under a micrometre.

Nothing here imports mujoco at module scope. `waist_limits` is called by P5,
which has no reason to carry a physics library, and `g1d_deploy` as a whole is
imported by P7, which must not touch this stack at all.

The frames and joint order as a policy author sees them:
`docs/deploying_a_policy/policy_contract.md`, "The base frame" and "Joint
order and limits".
"""

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple
from xml.etree import ElementTree

import numpy as np

from g1d_deploy.runtime.shared_memory import schema

#: The order every `(2, ...)` array in this system uses.
SIDES = ("left", "right")

#: Which model joint each of the three waist slots drives, in the order
#: the robot file's `[joints] waist` lists them — lowstate indices 12, 13, 14.
#:
#: **`Yaw_Joint` is the BOW and `torso_Joint` is the yaw.** Those two G1-D names
#: are backwards from their own axes; map by this tuple or by axis, never by the
#: name (`g1d_deploy.robot.g1d_model`). The vendored SDK enum (`G1_29_JointIndex`) is the
#: *humanoid* G1's map, and the G1-D keeps that wire format while having none of
#: the humanoid's lower body and only two of its three waist axes: it yaws about
#: the lift column and it bows forward.
#:
#: Reading index 13 as a roll puts the measured 0.039 rad trunk bow on a
#: sideways tilt instead of a forward bow: 14 mm of error on the left flange
#: and 3-7 mm on the right, in every sample, inside the observation the policy
#: sees.
#:
#: **Slot 2 is `None` because the joint does not exist.** Index 14 has no motor
#: behind it — across 495,958 pooled state samples from five rollouts `q[14]` and
#: `dq[14]` are bit-identically 0.0, a single distinct value, while index 13
#: carries ordinary encoder noise. The slot is retained rather than dropped:
#: `schema.WAIST_DOF` is 3 throughout — the body target, the chunk, `home_waist`,
#: `qdot_max_body`, the recorded dtype — and changing the arity would touch the
#: record format and the training contract for no gain. `waist_limits` bounds it
#: to exactly zero, so nothing can command an axis the robot does not have.
WAIST_JOINT_NAMES = ("torso_Joint", "Yaw_Joint", None)

#: Short labels for the same three slots, for operator messages. Written out
#: rather than derived from the joint names above, because those names lie.
WAIST_AXIS_NAMES = ("yaw", "bow", "absent")


@lru_cache(maxsize=4)
def waist_limits(urdf: str) -> np.ndarray:
    """`(3, 2)` of `[lower, upper]`, in `WAIST_JOINT_NAMES` order, from the URDF.

    That order is yaw, bow, absent — see the tuple. The rows follow it, so a
    caller indexing `limits[1]` gets the bow's bounds without knowing which model
    joint carries it.

    The absent slot comes back as `[0.0, 0.0]`: the only command the robot can
    honour on an axis it does not have is no command at all. That is a real
    bound and not a placeholder — it is what stops a policy that emits three
    live waist dimensions from writing the third one to the wire, where it would
    be published at kp=300 against nothing.

    Parsed with ElementTree rather than loaded through mujoco or pinocchio: this
    is called by P5, which has no reason to carry a physics library, and the
    numbers are two pairs of floats in an XML attribute.

    They matter because **the waist is the one commanded joint that never goes
    through IK**. The arm's limits are enforced inside the QP by mink's
    `ConfigurationLimit`; the waist is pinned at whatever the policy asked for
    and written straight to the ring, so the only place it can be bounded is
    where it enters the system.
    """
    tree = ElementTree.parse(urdf)
    out = np.zeros((len(WAIST_JOINT_NAMES), 2))
    for i, name in enumerate(WAIST_JOINT_NAMES):
        if name is None:
            continue                    # [0, 0]: no motor, so no travel
        joint = tree.find(f".//joint[@name='{name}']")
        limit = None if joint is None else joint.find("limit")
        if limit is None or "lower" not in limit.attrib:
            raise ValueError(f"{urdf} has no position limit for {name}")
        out[i] = (float(limit.attrib["lower"]), float(limit.attrib["upper"]))
    return out


#: The 14 arm joints in body-target order — `q[:14]` of every target, and the
#: order the robot file's `[joints] arm` lists their lowstate indices (15..28):
#: left 7 then right 7, each shoulder pitch/roll/yaw, elbow, wrist
#: roll/pitch/yaw. The same names as `mujoco_model.ARM_JOINTS`, restated here
#: because that module imports mujoco and P5 must not (`tests/test_kin_g1d.py`
#: asserts the two agree).
ARM_JOINT_NAMES = tuple(
    f"{side}_{j}_joint" for side in SIDES
    for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
              "wrist_roll", "wrist_pitch", "wrist_yaw"))


@lru_cache(maxsize=4)
def arm_limits(urdf: str) -> np.ndarray:
    """`(14, 2)` of `[lower, upper]`, in `ARM_JOINT_NAMES` order, from the URDF.

    Joint-mode actions (`JointActions`) are the one path where an arm joint reaches
    the wire without the QP's `ConfigurationLimit` in front of it, so P5 bounds
    them here, at the contract. Parsed with ElementTree for the same reason
    `waist_limits` is.
    """
    tree = ElementTree.parse(urdf)
    out = np.zeros((len(ARM_JOINT_NAMES), 2))
    for i, name in enumerate(ARM_JOINT_NAMES):
        joint = tree.find(f".//joint[@name='{name}']")
        limit = None if joint is None else joint.find("limit")
        if limit is None or "lower" not in limit.attrib:
            raise ValueError(f"{urdf} has no position limit for {name}")
        out[i] = (float(limit.attrib["lower"]), float(limit.attrib["upper"]))
    return out


# --- pose conversions -------------------------------------------------------


def quat_from_mat(R) -> np.ndarray:
    """3x3 -> wxyz (Shepperd's method).

    Branch-per-largest-diagonal rather than the naive trace formula: the naive
    one divides by a quantity that vanishes at 180 degrees of rotation.
    """
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
    return q / np.linalg.norm(q)


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
    """`[x y z qw qx qy qz]` -> 4x4."""
    T = np.eye(4)
    T[:3, :3] = mat_from_quat(pose7[3:])
    T[:3, 3] = pose7[:3]
    return T


def mat_to_pose(T) -> np.ndarray:
    """4x4 -> `[x y z qw qx qy qz]`, wxyz."""
    out = np.empty(schema.POSE_DIM)
    out[:3] = T[:3, 3]
    out[3:] = quat_from_mat(T[:3, :3])
    return out


# --- model placement --------------------------------------------------------


class JointMap(NamedTuple):
    """Where the contract's joints live in this model's qpos, by name.

    By name rather than by constant because the G1_29 index order and the model's
    qpos order agreeing is a property of a model file, not a guarantee — and on
    the G1-D they emphatically do not agree. Its qpos begins with two wheels and
    two column stages, and mounting the Dex-1 moves everything again, so
    `left_shoulder_pitch` is nowhere near index 15.
    """

    #: (14,) qpos addresses of the arm joints, left 7 then right 7.
    arm: np.ndarray
    #: qpos addresses of the waist joints that EXIST — two on this robot.
    waist: np.ndarray
    #: (same length) which contract slot each of those is, for indexing a
    #: `(3,)` waist vector. Slots with no joint behind them are simply absent
    #: here rather than written somewhere harmless-looking.
    waist_cols: np.ndarray


def joint_addresses(backend) -> JointMap:
    """Resolve the contract's joints against `backend`'s model."""
    arm = np.concatenate([backend.arm_qpos_adr["left"],
                          backend.arm_qpos_adr["right"]])
    if len(arm) != schema.ARM_DOF:
        raise RuntimeError(
            f"model exposes {len(arm)} arm DOF, the contract has {schema.ARM_DOF}")
    m = backend.model
    pairs = [(int(m.jnt_qposadr[m.joint(n).id]), i)
             for i, n in enumerate(WAIST_JOINT_NAMES) if n is not None]
    return JointMap(arm,
                    np.array([a for a, _ in pairs], dtype=np.intp),
                    np.array([c for _, c in pairs], dtype=np.intp))


def place(backend, arm_q14, waist_q3, jmap: JointMap | None = None) -> None:
    """Configure the model at these arm joints and this trunk. No FK is run.

    Zeroes the WHOLE qpos first — the wheels, both stages of the lift column and
    the gripper jaws included — and then writes the waist and the arms. The
    zeroing is what makes the placement a pure function of its two arguments
    rather than of whatever the model happened to hold before.

    Zeroing the column is safe precisely because of where the base frame sits:
    it rides the column (`g1d_model.BASE_BODY`), so the flange pose this produces does
    not depend on the column's real extension — which is just as well, since
    nothing on the wire reports it.
    """
    if jmap is None:
        jmap = joint_addresses(backend)
    d = backend.data
    d.qpos[:] = 0.0
    d.qpos[jmap.waist] = np.asarray(waist_q3, dtype=np.float64)[jmap.waist_cols]
    d.qpos[jmap.arm] = np.asarray(arm_q14, dtype=np.float64)


class ForwardKinematics:
    """P1's flange FK: (14,) arm joints + (3,) waist -> (2, 7) base-frame poses.

    Computed in P1 so the result carries exactly the `t_recv` of the joints that
    produced it, and so P5 and P6 cannot disagree about it.
    """

    def __init__(self, urdf: str | None = None):
        import mujoco
        from g1d_deploy.robot import g1d_model as _g1d

        self._mujoco = mujoco
        # the robot file's model path, or the module's own default. Passing it
        # through is what makes the config knob real: a second robot with a
        # different URDF is a new site file, not a code change.
        self._backend = _g1d.G1DBackend(urdf)
        self._jmap = joint_addresses(self._backend)

        # The base is fixed, so its inverse is constant: compute it once rather
        # than once per lowstate sample at ~1042 Hz.
        base = self._backend.base_pose()
        R = base[:3, :3]
        self._base_inv = np.eye(4)
        self._base_inv[:3, :3] = R.T
        self._base_inv[:3, 3] = -R.T @ base[:3, 3]

        self._out = np.zeros((schema.N_HANDS, schema.POSE_DIM))

    def flange(self, arm_q14, waist_q3) -> np.ndarray:
        place(self._backend, arm_q14, waist_q3, self._jmap)
        self._mujoco.mj_forward(self._backend.model, self._backend.data)
        for h, side in enumerate(SIDES):
            self._out[h] = mat_to_pose(
                self._base_inv @ self._backend.flange_pose(side))
        return self._out
