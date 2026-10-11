"""`rt/lowcmd` — the body command binding, and the gravity comp that rides on it.

Used by P7 (the 500 Hz publish) and by P0 (the damp). Design:
`docs/infra_logic/500hz_execution.md`.

The whole contract of this module: the 14 arm
joints and the 3 waist joints come from the interpolated target with per-joint
`kp`/`kd`, every actuated one of them additionally carries the interpolator's
own velocity and a `pin.rnea` feedforward torque computed from the **measured**
joints, and the 12 leg joints are held at whatever they measured at connect
time.

**Both feedforward channels exist for the same reason, and it is worth stating
once.** A PD joint produces force only by being wrong. At rest it must sit
`tau_load / kp` away from its target, because `kp * error` IS the holding
torque; while moving it must sit `(kd / kp) * velocity` behind, because with
`dq = 0` on the wire the damper reads the intended motion as error and brakes
it. Neither is a fault and neither can be tuned away — they are what the
control law is. What removes them is telling the joint what it is about to
need, so feedback is left holding only the part the model did not predict:

  * `tau` cancels the load    -> the resting error collapses
  * `dq`  cancels the damper  -> the following error collapses

Both were measured on the reference robot (2026-08-31), in a configuration
that sent `tau` for the arms only and `dq` for nothing:

  * the trunk sat 0.0393 rad (2.25 deg) below the zero it was commanded, all
    run — which is the 11.8 N.m it was holding divided by its own `kp` of 300,
    on the one joint group that got no `tau` at all;
  * every arm joint trailed its target by 0.0375 s of travel — which is
    `kd / kp` for all fourteen of them (3/80 and 1.5/40 are the same number),
    fitted at 42 ms against a predicted 37.5 ms over 21943 samples.

Neither number is a tuning constant. They are the two terms above, read off a
real robot.

Two things live here that P7 never calls:

* `damp()` — the latching e-stop. It is **P0's**, not P7's, and
  keeping it out of the servo loop is a deliberate design decision: P7 stays a
  pure interpolator with no notion of sessions or safety states. It is in this
  module because it is a `rt/lowcmd` message and this module owns that topic —
  putting it in P0 would give the arena two writers for one wire.
* `leg_q_hold` is passed *in* rather than read here. The measured legs live in
  the state ring, which is P1's; a binding that subscribed to `rt/lowstate` on
  its own would be a second reader of a stream the arena already carries, and
  the two could disagree about what "connect time" meant.

Gravity compensation is behind `GravityComp` for one reason: `PinocchioComp`
loads a URDF and builds a reduced model, which needs pinocchio.
`ZeroComp` lets every test in the system drive the real servo loop without
either. That P7's only kinematics is this one interface is what makes "P7
imports nothing from the IK / mujoco / mink stack" checkable rather than
merely intended (`tests/test_p7_body.py`).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from g1d_deploy.runtime.shared_memory import schema

# --- gravity compensation ---------------------------------------------------


class LowCmdError(RuntimeError):
    """The wire could not be brought up. Raised only from `connect`.

    A distinct type because of where it lands: P0 calls `_damp` from `finally`
    and catches broadly there, so this becomes a logged line rather than a
    second failure displacing the first. What it must never do is hang.
    """


class GravityComp(Protocol):
    """Measured joints in, body feedforward torque out."""

    def tau(self, q_measured29: np.ndarray) -> np.ndarray:
        """(29,) measured -> (17,) N.m in body order: 14 arm, then 3 waist.

        Body order, not arm order, because the trunk holds the arms up and is
        the joint group with the largest uncompensated load in the system. The
        third waist slot has no motor on this machine and is always 0.
        """
        ...


class ZeroComp:
    """No compensation. What a test drives, and what an unloadable URDF degrades
    to — with `kp` at 80 the arm still holds itself, it simply sags and sticks
    the way the pre-`tau` deploy stack did."""

    def tau(self, q_measured29: np.ndarray) -> np.ndarray:
        return np.zeros(schema.BODY_DOF)


#: The Dex-1's three bodies as upstream states them, in the DEX-1 BASE frame:
#: `(mass kg, com xyz m)`, read off
#: `unitree_ros/robots/g1_description/g1_29dof_mode_15_with_dex1_1.urdf` — the
#: same variant the meshes in `robot/assets/unitree_dex1` came from, and the mount
#: offset below matches the one recorded in that directory's PROVENANCE.md,
#: which is what says these are the same gripper.
#:
#: Both fingers hang off the base at `xyz="0 0 0"` on prismatic joints, so at
#: zero travel their link frames ARE the base frame and these numbers compose
#: directly.
_DEX1_BODIES = (
    (0.19138, (0.043853, -0.0001306, 1.4003e-5)),        # dex1_base_link
    (0.086783, (0.073453, -0.011173, 0.0086355)),        # dex1_finger_link_1
    (0.086783, (0.073453, 0.011173, -0.0086355)),        # dex1_finger_link_2
)

#: `{side}_base_joint`: the Dex-1 base sits this far along the wrist-yaw frame's
#: own +x, unrotated. Both sides identical — there is no mirroring in the mount.
_DEX1_MOUNT_X = 0.0415

#: The links the swap removes: Unitree's three-finger hand, everything distal to
#: the wrist-yaw flange. `{side}_hand_palm_link` is fixed to the flange and the
#: rest are its revolute fingers, which `buildReducedModel` locks and lumps.
_SHIPPED_HAND_LINKS = (
    "hand_palm_link",
    "hand_thumb_0_link", "hand_thumb_1_link", "hand_thumb_2_link",
    "hand_middle_0_link", "hand_middle_1_link",
    "hand_index_0_link", "hand_index_1_link",
)


def _real_end_effector(urdf_path: str, payload_mass: float,
                       payload_com_x: float) -> str:
    """The URDF with the shipped hand swapped for the Dex-1, as an XML string.

    Done in memory rather than by editing `robot/assets/unitree_g1d/g1_d.urdf`,
    because that file is what MuJoCo loads for P1's FK, P6's IK and the viewer,
    and all three want the geometry exactly as it is. This is a statement about
    MASS, and only the gravity feedforward reads it.

    The hand links are zeroed rather than deleted: deleting them would take
    their joints with them and change the reduced model's joint set, which
    `__init__` checks against `ARM_JOINT_NAMES` for a very good reason. A
    massless link is inert in `rnea` and leaves that structure alone.

    `payload_mass` is unmodelled hardware bolted to the wrist — the camera and
    its mount. Zero by default; see the class docstring.
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    links = {el.get("name"): el for el in root.iter("link")}

    for side in ("left", "right"):
        for stem in _SHIPPED_HAND_LINKS:
            el = links.get(f"{side}_{stem}")
            inertial = None if el is None else el.find("inertial")
            if inertial is None:
                raise RuntimeError(
                    f"{urdf_path} has no inertial on {side}_{stem}; the "
                    f"end-effector swap cannot state what it removed.")
            inertial.find("mass").set("value", "0")
            for a in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
                inertial.find("inertia").set(a, "0")

        # One lumped body, mounted on the flange. Exact for gravity: see the
        # class docstring on why the jaw's travel does not move this point.
        bodies = list(_DEX1_BODIES)
        if payload_mass > 0.0:
            bodies.append((payload_mass, (payload_com_x - _DEX1_MOUNT_X, 0.0, 0.0)))
        mass = sum(m for m, _ in bodies)
        com = [sum(m * c[i] for m, c in bodies) / mass for i in range(3)]

        link = ET.SubElement(root, "link")
        link.set("name", f"{side}_dex1_lumped_link")
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin").set(
            "xyz", " ".join(repr(v) for v in com))
        ET.SubElement(inertial, "mass").set("value", repr(mass))
        # Zero, and not a placeholder: `rnea(q, 0, 0)` never reads it. A
        # non-zero guess here would be a number nobody could defend.
        inertia = ET.SubElement(inertial, "inertia")
        for a in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
            inertia.set(a, "0")

        joint = ET.SubElement(root, "joint")
        joint.set("name", f"{side}_dex1_lumped_joint")
        joint.set("type", "fixed")
        ET.SubElement(joint, "parent").set("link", f"{side}_wrist_yaw_link")
        ET.SubElement(joint, "child").set("link", f"{side}_dex1_lumped_link")
        ET.SubElement(joint, "origin").set("xyz", f"{_DEX1_MOUNT_X} 0 0")

    return ET.tostring(root, encoding="unicode")


class PinocchioComp:
    """`pin.rnea` over the URDF, with everything but the arms and waist locked.

    Locking legs and fingers at the reference configuration leaves a 16-DoF
    model — the 14 arm joints and the two waist joints that carry a motor. The
    torque it returns is laid out in BODY order (14 arm, then 3 waist) by
    looking each joint up by NAME, never by position in the reduced model: a
    silently permuted model returns plausible torques for the wrong joints, and
    the reduction's output order is pinocchio's business, not ours.

    `rnea(q, 0, 0)` is gravity only: no velocity and no acceleration term. That
    is deliberate — the servo loop's job is to cancel the arm's weight, not to
    invert its dynamics, and an acceleration term computed from a noisy measured
    `dq` would inject that noise straight into the torque at 500 Hz.

    **The waist is compensated and configured at its MEASURED angle.** Locking
    it at zero would make this model assume an upright trunk and — far more
    consequentially — mean the one joint group carrying the entire upper body
    received no feedforward at all while the arms it holds up received
    theirs. Measured that way on the reference robot (2026-08-31), the
    bow sat 0.0393 rad low for the whole run: 11.8 N.m of holding torque over
    a `kp` of 300, against a gravity load this model puts at 9.7 N.m. With that
    9.7 supplied here the position error has only the ~2 N.m remainder to
    generate, which is a fifth of the droop.

    It does not go to zero, and the reason is the same one the arms have: the
    URDF does not know about anything mounted on the wrists (cameras, mounts).
    Measured on that same rollout, the arms' P term carries 6.84 N.m against a 16.93 N.m
    feedforward — the true load is ~31% heavier than this model, both elbows,
    independently. `wrist_payload` is where that goes once somebody weighs it;
    until then this under-compensates, which is the safe direction to be wrong
    in: the trunk sags less than uncompensated and never bows backwards.

    Applied from the first publish, like the arms' torque has always been,
    rather than ramped. The step it puts on the trunk is the one the arms have
    taken at every connect since `tau` shipped, and the joint it lands on is
    measurably the quietest in the machine — 0.0002 rad of standard deviation
    across a 29 s run in which its own gravity load swung by a factor of 2.8.
    A joint that ignores an 8.7 N.m load swing is not one that rings on a 9.7
    N.m step.

    The model is now built from the G1-D's own URDF rather than the humanoid's,
    and that changes the torque by **exactly zero**: the two files carry the same
    arm chain and the same three-finger hand inertias, so the reduced 14-DOF
    models are bit-identical (verified over 200 random postures, max |Δτ| = 0).

    **The end effector is now the Dex-1, and it is LIGHTER than what shipped.**
    The URDF carries Unitree's three-finger hand — 0.6965 kg with its CoM
    108.2 mm along the wrist-yaw +x — where the supported rigs carry a Dex-1 gripper:
    0.3649 kg at 99.4 mm (`_DEX1_*` below, from upstream's own
    `g1_29dof_mode_15_with_dex1_1.urdf`). `_real_end_effector` swaps them, which
    drops the first moment about the wrist by 52% and the arm's gravity torque
    with it.

    That direction is worth stating plainly, because it is the opposite of what
    the robot's own behaviour asks for. On the reference robot, holding home for
    30 s, the P term carries 6.84 N.m against a 16.93 N.m feedforward, and the
    sign says the TRUE load is heavier than the model — both elbows, ~31%,
    independently. Correcting the gripper alone therefore makes the sag worse,
    not better. The arithmetic is right and the remainder is honest: what is
    left over is mass this URDF does not know about (on the reference robot, a
    wrist camera and its mount), and the robot file's `[model.wrist_payload]`
    is where that goes once somebody weighs it. Until then it is 0.0: a
    fitted value would be indistinguishable from joint friction.

    Only mass and CoM are needed: `rnea(q, 0, 0)` has no velocity or
    acceleration term, so the rotational inertia never enters the result. The
    Dex-1 is lumped to a single body for the same reason, and that lumping is
    EXACT rather than an approximation — its two fingers are prismatic, share the
    base's origin, and slide symmetrically, so the pair's combined CoM does not
    move with the jaw.

    The model is built here rather than through `g1d_deploy.robot.g1d_model`, which owns
    every other use of this same URDF, and the reason is P7's import rule: that
    module subclasses the MuJoCo backend, so importing it would pull mujoco into
    P7 — the one 500 Hz process kept clear of the whole IK stack. The two
    engines want different things out of the file anyway (mass here, geometry
    there), and only this one has to run in a real-time loop.
    """

    ARM_JOINT_NAMES = (
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint", "left_elbow_joint",
        "left_wrist_roll_joint", "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint", "right_elbow_joint",
        "right_wrist_roll_joint", "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    )

    #: The waist joints that carry a motor, in the robot file's `[joints]
    #: waist` slot order — the same order as `g1d_deploy.robot.kinematics.WAIST_JOINT_NAMES`,
    #: which is where these names are explained. Slot 0 is the yaw, slot 1 is
    #: the BOW (whose URDF name says "Yaw" and is simply wrong about it), and
    #: slot 2 has no joint on this machine, so only two names appear here and
    #: the third torque is always 0.
    #:
    #: Duplicated from `robot/kinematics.py` rather than imported, exactly as
    #: `ARM_JOINT_NAMES` is: this module is on P7's import path, which is kept
    #: clear of the MuJoCo stack. `tests/test_kin_g1d.py`
    #: (`test_the_waist_names_gravity_comp_uses_are_the_ones_kin_uses`) is what stops the
    #: two copies drifting.
    WAIST_JOINT_NAMES = ("torso_Joint", "Yaw_Joint")

    def __init__(self, hw, urdf: str | None = None, *, waist_joints=None):
        # Imported here, not at module scope: this module is on P7's import
        # path, and pinocchio costs seconds to import. Nothing that only ever
        # uses ZeroComp should pay it.
        import pinocchio as pin

        self._hw = hw
        # `buildModelFromUrdf` reads kinematics and inertia and never touches a
        # mesh — which is all gravity compensation has ever needed, and which is
        # why the G1-D's un-vendored three-finger hand meshes are a problem for
        # MuJoCo and not for this. `RobotWrapper.BuildFromURDF`, which this used
        # to call, would try to load all 16.3 MB of them and fail.
        path = str(urdf if urdf is not None else hw.urdf)
        # `buildModelFromXML`, because the file on disk describes the shipped
        # three-finger hand and the rig has a Dex-1. See `_real_end_effector`.
        full = pin.buildModelFromXML(_real_end_effector(
            path,
            float(getattr(hw, "wrist_payload_mass", 0.0)),
            float(getattr(hw, "wrist_payload_com_x", 0.0))))
        waist = (self.WAIST_JOINT_NAMES if waist_joints is None
                 else tuple(waist_joints))
        free = self.ARM_JOINT_NAMES + waist

        # Checked BEFORE the reduction, because `getJointId` answers an unknown
        # name with an out-of-range id rather than raising — so a typo, or this
        # class pointed at a model whose waist is called something else, would
        # otherwise reduce to the wrong joint set and compensate confidently.
        missing = [n for n in free if n not in full.names]
        if missing:
            raise RuntimeError(
                f"{path} has no joint(s) {missing}. Gravity comp names every "
                f"joint it leaves free; a model that spells one differently "
                f"needs `waist_joints=`, not a silent omission.")

        lock = [full.getJointId(n) for n in full.names
                if n != "universe" and n not in free]
        reduced = pin.buildReducedModel(full, lock, np.zeros(full.nq))
        got = {n for n in reduced.names if n != "universe"}
        if got != set(free):
            raise RuntimeError(
                f"reduced model has joints {sorted(got)}, expected "
                f"{sorted(free)}. Gravity comp would apply torques to a "
                f"different robot than the one it was asked about.")
        # One coordinate per free joint. Everything here is revolute, so nq and
        # nv agree; asserting it is what lets the two index maps below be read
        # as "the slot this joint occupies" without qualification.
        if reduced.nq != len(free) or reduced.nv != len(free):
            raise RuntimeError(
                f"reduced model is nq={reduced.nq} nv={reduced.nv} over "
                f"{len(free)} joints; gravity comp assumes one coordinate each.")

        # BY NAME, never by position: this is the whole correspondence between
        # pinocchio's ordering and body order, and it is the one thing in this
        # class that cannot be checked by looking at the output.
        self._idx_q = np.array(
            [reduced.joints[reduced.getJointId(n)].idx_q for n in free], dtype=int)
        self._idx_v = np.array(
            [reduced.joints[reduced.getJointId(n)].idx_v for n in free], dtype=int)
        # Where each of those joints is measured on the wire. The arms first,
        # then as many waist slots as there are waist joints — `joints_waist`
        # lists three and the last has no motor, so it is never read.
        self._src = np.array(
            list(hw.joints_arm) + list(hw.joints_waist)[:len(waist)], dtype=int)

        self._pin = pin
        self._model = reduced
        self._data = reduced.createData()
        self._zero = np.zeros(reduced.nv)
        self._n_free = len(free)

    def tau(self, q_measured29: np.ndarray) -> np.ndarray:
        """(29,) measured -> (17,) N.m in body order.

        The waist enters at its MEASURED angle, so the arms' gravity term is
        computed about the trunk the robot actually has rather than the upright
        one it is being commanded toward.
        """
        q = np.zeros(self._model.nq)
        q[self._idx_q] = np.asarray(q_measured29, dtype=np.float64)[self._src]
        full = np.asarray(self._pin.rnea(self._model, self._data, q,
                                         self._zero, self._zero),
                          dtype=np.float64)
        out = np.zeros(schema.BODY_DOF)
        out[:self._n_free] = full[self._idx_v]
        # The waist's measured constant, added here so that every caller gets
        # it — P7, the probes, anything else that asks for a feedforward — and
        # so there is one place it can be wrong. `rnea` cannot produce it: it
        # is a spring, a harness or a driver offset, not a mass.
        # `Hardware.waist_tau_offset`.
        off = getattr(self._hw, "waist_tau_offset", None)
        if off is not None:
            out[schema.ARM_DOF:] += np.asarray(off, np.float64)
        return out


# --- the body command -------------------------------------------------------


class LowCmd(Protocol):
    """What P7 and P0 depend on. Two implementations: DDS and a recorder."""

    def connect(self, leg_q_hold: np.ndarray) -> None: ...
    def publish_body(self, q_body17: np.ndarray, tau_body17: np.ndarray,
                     dq_body17: np.ndarray) -> None: ...
    def damp(self) -> None: ...
    def close(self) -> None: ...


class _BodyCmdBase:
    """The index arithmetic and the latch, shared by both implementations.

    Both halves of the split matter. The index arithmetic is shared so that a
    fake and the real wire cannot disagree about which of the 17 body numbers is
    a waist angle. The latch is shared so that "no path publishes to
    `rt/lowcmd` after the damp latches" is one line of code rather than a
    property of two.
    """

    def __init__(self, hw):
        self.hw = hw
        self.arm = list(hw.joints_arm)
        self.waist = list(hw.joints_waist)
        self.leg = list(hw.joints_leg)
        self.connected = False
        self.damped = False
        self.leg_q_hold: np.ndarray | None = None

    def _accept(self, leg_q_hold) -> np.ndarray:
        legs = np.asarray(leg_q_hold, dtype=np.float64).reshape(-1)
        if legs.shape != (len(self.leg),):
            raise ValueError(
                f"leg_q_hold must be ({len(self.leg)},), got {legs.shape}")
        return legs.copy()

    @staticmethod
    def _split(q_body17, tau_body17, dq_body17
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The three body vectors, validated. All (17,), all in body order.

        One shape for all three: the trunk holds the arms up and is compensated
        with them, so "which of these numbers is a waist quantity" is the same
        question for every channel and is answered in one place.
        """
        out = []
        for name, v in (("q_body", q_body17), ("tau_body", tau_body17),
                        ("dq_body", dq_body17)):
            a = np.asarray(v, dtype=np.float64).reshape(-1)
            if a.shape != (schema.BODY_DOF,):
                raise ValueError(
                    f"{name} must be ({schema.BODY_DOF},), got {a.shape}")
            out.append(a)
        return tuple(out)


class FakeLowCmd(_BodyCmdBase):
    """Records what was published; commands nothing.

    `published` keeps the *body* vector rather than the 29-wide wire row because
    that is what the assertions in P7's tests are about — the interpolation, the
    freeze and the slew clamp all live in the 17 numbers. `wire()` reconstructs
    the full row for the rare test that cares about leg hold or gains.
    """

    def __init__(self, hw):
        super().__init__(hw)
        self.published: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.closed = False

    def connect(self, leg_q_hold: np.ndarray) -> None:
        self.leg_q_hold = self._accept(leg_q_hold)
        self.connected = True

    def publish_body(self, q_body17, tau_body17, dq_body17) -> None:
        if not self.connected:
            raise RuntimeError("publish_body before connect()")
        if self.damped:
            return                                  # latched: nothing after a damp
        q, tau, dq = self._split(q_body17, tau_body17, dq_body17)
        self.published.append((q.copy(), tau.copy(), dq.copy()))

    def set_gains(self, kp, kd) -> None:
        """Records the gains `DdsLowCmd.set_gains` would write."""
        self.gains = (np.asarray(kp, dtype=np.float64).copy(),
                      np.asarray(kd, dtype=np.float64).copy())

    def damp(self) -> None:
        self.damped = True

    def close(self) -> None:
        self.closed = True

    # --- test conveniences --------------------------------------------------

    @property
    def q_body(self) -> np.ndarray:
        """(n, 17) — every body command published, in order."""
        return self._column(0)

    @property
    def tau(self) -> np.ndarray:
        """(n, 17) — the feedforward torque that rode with each command."""
        return self._column(1)

    @property
    def dq(self) -> np.ndarray:
        """(n, 17) — the target velocity that rode with each command."""
        return self._column(2)

    def _column(self, i: int) -> np.ndarray:
        if not self.published:
            return np.zeros((0, schema.BODY_DOF))
        return np.array([row[i] for row in self.published])

    def wire(self, i: int = -1) -> np.ndarray:
        """The (29,) joint row the i-th publish would have put on the wire."""
        q = self.published[i][0]
        row = np.zeros(schema.N_JOINTS)
        row[self.arm] = q[:schema.ARM_DOF]
        row[self.waist] = q[schema.ARM_DOF:]
        row[self.leg] = self.leg_q_hold
        return row


class DdsLowCmd(_BodyCmdBase):
    """The real `rt/lowcmd` publisher (unitree_hg `LowCmd_`, 29 of 35 motors).

    Gains are written at connect, and again only by `set_gains` when a rollout
    starts with a different arm gain scale. The firmware re-reads the whole
    message every time, so per-publish work is three vector stores (q, dq, tau)
    plus the CRC — at 500 Hz the difference between that and rewriting 29 gain
    pairs is real.

    `mode_machine` is read from one `rt/lowstate` sample at connect. It is a
    firmware handshake byte, not state: a message carrying the wrong one is
    ignored outright, so this waits for a sample rather than guessing a default
    and publishing into the void. **Waits, bounded** — see `connect` for why an
    unbounded one was a bug in the damp rather than merely a slow start.
    """

    #: How long a single `Read` may wait. Short: it is only there so the loop
    #: gets the chance to notice its own deadline, and a bare `Read()` never
    #: returns at all on a silent topic.
    _READ_TIMEOUT_S = 0.1

    #: How long to wait for the one `rt/lowstate` sample `mode_machine` needs.
    #: Generous against a healthy robot — P0's own CHECK proves this stream at
    #: ~1000 Hz before anything gets here — and short against a dead one, since
    #: what is downstream of this wait is the damp.
    MODE_MACHINE_TIMEOUT_S = 5.0

    def __init__(self, hw, *, kp=None, kd=None):
        super().__init__(hw)
        self._pub = None
        self._msg = None
        self._crc = None
        # (29,) gains for this run; the robot file's when not given. P7 passes
        # `derive.servo_gains`, which scales the arms for a gain A/B.
        self.kp = np.asarray(hw.kp if kp is None else kp, dtype=np.float64)
        self.kd = np.asarray(hw.kd if kd is None else kd, dtype=np.float64)

    def connect(self, leg_q_hold: np.ndarray) -> None:
        import time

        from g1d_deploy.runtime.drivers.dds import channel
        from g1d_deploy.runtime.drivers.dds.crc import hg_lowcmd_crc
        from g1d_deploy.runtime.drivers.dds.idl.unitree_hg import LowCmd_, LowState_
        from g1d_deploy.runtime.drivers.dds.idl.unitree_hg import (default_LowCmd_ as unitree_hg_msg_dds__LowCmd_)

        # Through the module, so a test that stubs the channel is honoured.
        ChannelPublisher = channel.ChannelPublisher
        ChannelSubscriber = channel.ChannelSubscriber

        legs = self._accept(leg_q_hold)
        from g1d_deploy.runtime.drivers.dds import domain as dds
        dds.init(self.hw.dds_domain, self.hw.dds_iface)

        sub = ChannelSubscriber(self.hw.topic_lowstate, LowState_)
        sub.Init()
        state = None
        deadline = time.monotonic() + self.MODE_MACHINE_TIMEOUT_S
        while state is None:
            # **`Read()` WITH a timeout, never the bare form.** The bare call is
            # `take_one()` with no wait bound, and on this stack it does not
            # return when the topic is silent — it blocks forever. Measured
            # side by side against a robot that had stopped
            # publishing: `Read()` was still blocked after 6 s in a thread that
            # was then abandoned, while `Read(0.5)` returned None in 0.50 s.
            #
            # This is the DAMP's path. `_damp` is the only thing that ends a
            # `kp`-held setpoint, and P0 reaches it from `finally`
            # on EVERY exit — including the failure exits. So with the bare form a
            # robot that stops publishing lowstate makes P0 hang in EXITING,
            # which is the one state it must never hang in: it has failed, it is
            # on its way to a damp, and it never arrives.
            state = sub.Read(self._READ_TIMEOUT_S)
            if state is None and time.monotonic() >= deadline:
                sub.Close()
                raise LowCmdError(
                    f"no sample on {self.hw.topic_lowstate} within "
                    f"{self.MODE_MACHINE_TIMEOUT_S:.0f}s, so `mode_machine` "
                    f"cannot be read and no LowCmd this process sends would be "
                    f"accepted. The robot's low-level controller is not "
                    f"publishing — check that it is powered and out of e-stop.")
        sub.Close()

        self._pub = ChannelPublisher(self.hw.topic_lowcmd, LowCmd_)
        self._pub.Init()
        # The firmware drops a LowCmd whose CRC is wrong (`dds/crc.py`: the
        # SDK's packing, zlib's CRC, ~5 us — P7's budget is 2 ms).
        self._crc = hg_lowcmd_crc

        msg = unitree_hg_msg_dds__LowCmd_()
        msg.mode_pr = 0
        msg.mode_machine = state.mode_machine
        for i in range(schema.N_JOINTS):
            cmd = msg.motor_cmd[i]
            cmd.mode = 1
            cmd.kp = float(self.kp[i])
            cmd.kd = float(self.kd[i])
            cmd.dq = 0.0
            cmd.tau = 0.0
            cmd.q = 0.0
        for i, idx in enumerate(self.leg):
            msg.motor_cmd[idx].q = float(legs[i])
        self._msg = msg
        self.leg_q_hold = legs
        self.connected = True

    def set_gains(self, kp, kd) -> None:
        """New (29,) gains for the arm and waist, from the next publish on.

        P7 calls this when a rollout starts with a different
        `servo.arm_gain_scale` — between rollouts, with the arm
        holding at home. The legs keep the gains `connect` gave them: they are
        held, not driven, and no setting addresses them. A no-op once damped:
        the damp's gains are final.
        """
        self.kp = np.asarray(kp, dtype=np.float64).copy()
        self.kd = np.asarray(kd, dtype=np.float64).copy()
        if self._msg is None or self.damped:
            return
        for idx in self.arm + self.waist:
            cmd = self._msg.motor_cmd[idx]
            cmd.kp = float(self.kp[idx])
            cmd.kd = float(self.kd[idx])

    def publish_body(self, q_body17, tau_body17, dq_body17) -> None:
        """One body command: position, velocity and torque, arms and waist.

        All three channels go to both joint groups. Leaving `dq` at the 0.0
        `connect` wrote would make the firmware read every commanded motion as
        velocity error and brake it — see the module docstring for what that
        costs. The legs keep the zeros: they are held, so 0 is their
        true target velocity and they carry no feedforward.
        """
        if not self.connected:
            raise RuntimeError("publish_body before connect()")
        if self.damped:
            return                                  # latched: nothing after a damp
        q, tau, dq = self._split(q_body17, tau_body17, dq_body17)
        msg = self._msg
        # `self.arm` is body slots 0..13 and `self.waist` is 14..16, which is
        # exactly `enumerate` over the two in order — the same correspondence
        # `_split`'s docstring names, written once per group rather than per
        # channel.
        for slot, idx in enumerate(self.arm + self.waist):
            cmd = msg.motor_cmd[idx]
            cmd.q = float(q[slot])
            cmd.dq = float(dq[slot])
            cmd.tau = float(tau[slot])
        msg.crc = self._crc(msg)
        self._pub.Write(msg)

    def set_limp(self, *, kd: float = 2.0) -> None:
        """Make both arms back-drivable and weightless. NOT part of a run.

        Zero the arms' position gain and keep a little damping; the waist and
        legs are untouched and stay fully controlled. `publish_body` then still
        does the streaming — with `kp = 0` the joint torque is `-kd*dq + tau`,
        so the gravity-compensation feedforward it already sends is what makes
        the arm weightless, and `-kd*dq` is what stops it swinging. The arm
        floats where you put it.

        Gains persist in `self._msg` across publishes, so this is a one-time
        edit rather than something the streaming loop has to remember.

        **A limp arm falls under gravity.** Gravity compensation is a torque
        estimate, not a brake: a payload, or a model that disagrees with the
        real arm, still lets it sag. Support the arm before calling this.

        Deliberately not reachable from a run: nothing in `g1d_deploy`'s eight
        processes calls it. It exists for `tools.measure_pose`, which is how the
        postures that end up in a `home_q` get measured in the first place.
        """
        if not self.connected:
            raise RuntimeError("set_limp before connect()")
        if self.damped:
            return                                  # latched: nothing after a damp
        for idx in self.arm:
            cmd = self._msg.motor_cmd[idx]
            cmd.kp = 0.0
            cmd.kd = float(kd)

    def damp(self) -> None:
        """THE stop. kp=0, kd=`damp_kd` on every joint, and it latches.

        Publishing is the point: the firmware runs the last command it received
        forever, so silence leaves the robot stiff at its last setpoint. Sent a handful of times because a single datagram may be dropped
        and there is, by construction, no retry after this.
        """
        import time

        self.damped = True                          # latch first
        if not self.connected:
            return
        msg = self._msg
        for i in range(schema.N_JOINTS):
            cmd = msg.motor_cmd[i]
            cmd.kp = float(self.hw.damp_kp)
            cmd.kd = float(self.hw.damp_kd)
            cmd.dq = 0.0
            cmd.tau = 0.0
        for _ in range(5):
            msg.crc = self._crc(msg)
            self._pub.Write(msg)
            time.sleep(0.002)

    def close(self) -> None:
        self._pub = None
        self._msg = None
        self.connected = False


__all__ = ["DdsLowCmd", "FakeLowCmd", "GravityComp", "LowCmd", "LowCmdError",
           "PinocchioComp",
           "ZeroComp"]
