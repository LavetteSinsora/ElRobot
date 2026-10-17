"""The G1-D model: `robot/assets/unitree_g1d/g1_d.urdf`, the one file MuJoCo (FK,
IK) and pinocchio (gravity) both build from.

Until 2026-08-27 the deploy loop posed a humanoid G1 model instead. The
before/after comparisons against it were dropped with the humanoid assets
(elrobot_deploy ships only `unitree_g1d` and `unitree_dex1`); what is pinned here is
the G1-D model's own geometry, collision sets, waist and hand.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from elrobot_deploy.robot import kinematics as kin
from elrobot_deploy.robot import mujoco_model as g1, g1d_model as g1d

def _postures(n: int = 24, seed: int = 0) -> np.ndarray:
    """`n` arm postures, well inside every joint's range on both models."""
    return np.random.default_rng(seed).uniform(-0.6, 0.6, size=(n, 14))


def _se3_inv(T):
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def _flange_mats(fk, arm_q14, waist_q3) -> list[np.ndarray]:
    """`ForwardKinematics.flange` as 4x4s. It returns pose7 into a REUSED
    buffer, so this converts rather than keeping a reference."""
    return [kin.pose_to_mat(p) for p in fk.flange(arm_q14, waist_q3)]


def test_the_waist_names_gravity_comp_uses_are_the_ones_kin_uses():
    """`PinocchioComp` duplicates these names rather than importing them.

    It has to: that class is on P7's import path and P7's import rule keeps that path clear
    of `kin`'s MuJoCo stack. A duplicate is only safe while something checks it,
    and this is that something — a swap here would feed the bow's gravity torque
    to the yaw motor, at a magnitude (~9.7 N.m) large enough to move the trunk.

    Needs no pinocchio: it compares two tuples of strings.
    """
    from elrobot_deploy.robot.kinematics import WAIST_JOINT_NAMES
    from elrobot_deploy.runtime.drivers.arm_command import PinocchioComp

    actuated = tuple(n for n in WAIST_JOINT_NAMES if n is not None)
    assert PinocchioComp.WAIST_JOINT_NAMES == actuated
    # And the order is the SLOT order, not merely the same set: slot 0 is the
    # yaw and slot 1 is the bow, which is what `_src` indexes `joints_waist` by.
    assert WAIST_JOINT_NAMES.index(actuated[0]) == 0
    assert WAIST_JOINT_NAMES.index(actuated[1]) == 1


# --- and the base frame is anchored where it has to be ----------------------


def test_the_flange_pose_does_not_depend_on_the_unmeasured_lift_column():
    """Why `g1d_model.BASE_BODY` is above the column and not at `AGV_link`.

    The column's two prismatic stages are not on `rt/lowstate` — there is no
    index for them — so their extension is a number the deploy
    loop never learns. A base frame below them would take that unknown as a
    permanent z bias in every pose the system produces. Anchored above, the frame
    rides the column and the bias is identically zero, which is what lets
    `kin.place()` zero the column without inventing anything.
    """
    import mujoco

    backend = g1d.G1DBackend()
    jmap = kin.joint_addresses(backend)
    m = backend.model
    column = [int(m.jnt_qposadr[m.joint(n).id])
              for n in ("LZ_mt_Joint", "LZ_it_Joint")]
    q = _postures(n=1, seed=2)[0]

    poses, heights = [], []
    for h in (0.0, 0.11, 0.21):
        kin.place(backend, q, np.zeros(3), jmap)
        backend.data.qpos[column] = h
        mujoco.mj_forward(backend.model, backend.data)
        base_inv = _se3_inv(backend.base_pose())
        poses.append([base_inv @ backend.flange_pose(s)
                      for s in ("left", "right")])
        heights.append(float(backend.base_pose()[2, 3]))

    # The frame really did ride the column — otherwise the invariance below is
    # the trivial kind and proves nothing.
    assert heights[2] - heights[0] == pytest.approx(0.42, abs=1e-9)
    for other in poses[1:]:
        assert np.allclose(poses[0], other, atol=1e-12)


def test_bowing_moves_the_flange_within_the_base_frame():
    """The other half of the same constraint: the frame must be BELOW the bow.

    If it were above — anchored on `torso_link`, say — the trunk would carry the
    frame with it, and a bow would leave the flange pose unchanged. The waist
    would then be invisible to FK, and P6's IK would be pinning a trunk the
    target could not describe.
    """
    fk = kin.ForwardKinematics()
    q = _postures(n=1, seed=3)[0]
    upright = _flange_mats(fk, q, np.zeros(3))
    # 0.039 rad is this trunk's measured gravity droop, and column 1
    # is the bow — `kin.WAIST_AXIS_NAMES`, not the URDF's joint names, which lie.
    bowed = _flange_mats(fk, q, np.array([0.0, 0.039, 0.0]))
    moved = max(float(np.linalg.norm(bowed[h][:3, 3] - upright[h][:3, 3]))
                for h in (0, 1))
    assert moved > 4e-3, f"a 0.039 rad bow moved the flange only {moved * 1e3:.2f} mm"


# --- The end effector is the Dex-1, not the hand the URDF ships -------------


def test_the_feedforward_carries_the_dex1_and_not_the_shipped_hand():
    """The URDF describes Unitree's three-finger hand; the robot has a Dex-1.

    The direction matters and is the opposite of the intuitive one: the Dex-1 is
    0.3649 kg where the shipped hand is 0.6965 kg, so getting this right makes
    the feedforward SMALLER and the arm's sag larger. That is the correct model
    and a worse-tracking robot at the same time, and the remainder it exposes is
    the wrist camera nobody has weighed (`Hardware.wrist_payload_mass`).

    Asserted on the model's MASS, not on |tau| per joint: the gravity torque is
    signed, and at some postures removing distal mass raises a joint's magnitude
    rather than lowering it. Mass is the thing the swap is actually about.
    """
    pytest.importorskip("pinocchio")
    import pinocchio as pin

    from elrobot_deploy.runtime.drivers import arm_command as lowcmd

    def reduced(model):
        lock = [model.getJointId(n) for n in model.names
                if n != "universe"
                and n not in lowcmd.PinocchioComp.ARM_JOINT_NAMES]
        return pin.buildReducedModel(model, lock, np.zeros(model.nq))

    shipped = reduced(pin.buildModelFromUrdf(str(g1d.URDF)))
    swapped = reduced(pin.buildModelFromXML(
        lowcmd._real_end_effector(str(g1d.URDF), 0.0, 0.0)))

    dex1 = sum(m for m, _ in lowcmd._DEX1_BODIES)
    assert dex1 == pytest.approx(0.364946, abs=1e-6)

    # The hand that came off, read from the file rather than pinned to a
    # literal: the assertion is that the swap removed EXACTLY the eight links it
    # names and added the Dex-1, which stays true if upstream revises the hand.
    import xml.etree.ElementTree as ET

    links = {el.get("name"): el for el in ET.parse(str(g1d.URDF)).iter("link")}
    hand = sum(float(links[f"left_{stem}"].find("inertial").find("mass")
                     .get("value"))
               for stem in lowcmd._SHIPPED_HAND_LINKS)

    # Both wrists, and only the wrists: everything else is the same arm.
    delta = pin.computeTotalMass(swapped) - pin.computeTotalMass(shipped)
    assert delta == pytest.approx(2 * (dex1 - hand), abs=1e-9)
    assert delta < 0, "the Dex-1 is lighter than the hand the URDF ships"

    # And it reaches the torque, in every posture, not just the mass table.
    data_a, data_b = shipped.createData(), swapped.createData()
    zero = np.zeros(14)
    for q in _postures(seed=7):
        was = np.asarray(pin.rnea(shipped, data_a, q, zero, zero))[:14]
        now = np.asarray(pin.rnea(swapped, data_b, q, zero, zero))[:14]
        assert not np.allclose(was, now)


def test_the_wrist_payload_is_off_until_somebody_weighs_it():
    """`wrist_payload_mass = 0.0` means unmeasured, not absent — and it must be
    inert, so that shipping the swap does not smuggle in a fitted number.

    Then: a non-zero payload has to add exactly its own mass to each wrist, and
    `com_x` has to reach the torque. A knob that changed the mass but ignored
    where it sits would compensate the right weight in the wrong place.
    """
    pytest.importorskip("pinocchio")
    import pinocchio as pin

    from elrobot_deploy.runtime.drivers import arm_command as lowcmd

    def build(mass, com_x):
        model = pin.buildModelFromXML(
            lowcmd._real_end_effector(str(g1d.URDF), mass, com_x))
        lock = [model.getJointId(n) for n in model.names
                if n != "universe"
                and n not in lowcmd.PinocchioComp.ARM_JOINT_NAMES]
        return pin.buildReducedModel(model, lock, np.zeros(model.nq))

    # Zero mass is inert whatever `com_x` says.
    off, off_far = build(0.0, 0.0), build(0.0, 0.25)
    assert pin.computeTotalMass(off) == pytest.approx(
        pin.computeTotalMass(off_far), abs=1e-12)

    loaded = build(0.5, 0.10)
    assert pin.computeTotalMass(loaded) - pin.computeTotalMass(off) == \
        pytest.approx(1.0, abs=1e-6)          # 0.5 kg on each wrist

    # Same mass, further out: the same weight on a longer lever must change the
    # torque it asks the shoulder and elbow to hold.
    far = build(0.5, 0.25)
    d_near, d_far = loaded.createData(), far.createData()
    zero = np.zeros(14)
    moved = 0
    for q in _postures(seed=3):
        a = np.asarray(pin.rnea(loaded, d_near, q, zero, zero))[:14]
        b = np.asarray(pin.rnea(far, d_far, q, zero, zero))[:14]
        if not np.allclose(a, b, atol=1e-6):
            moved += 1
    assert moved, "com_x never reached the torque"


def test_the_swap_refuses_a_urdf_whose_hand_it_cannot_find():
    """Silently skipping the strip would leave the shipped hand AND add the
    Dex-1 — a 1.06 kg end effector nobody chose, and no error to notice."""
    pytest.importorskip("pinocchio")
    import xml.etree.ElementTree as ET

    from elrobot_deploy.runtime.drivers.arm_command import _real_end_effector

    tree = ET.parse(str(g1d.URDF))
    for el in tree.getroot().iter("link"):
        if el.get("name") == "left_hand_palm_link":
            el.set("name", "left_gripper_link")
    tmp = pathlib.Path(__file__).resolve().parent / "_renamed_hand.urdf"
    tmp.write_bytes(ET.tostring(tree.getroot()))
    try:
        with pytest.raises(RuntimeError, match="hand_palm_link"):
            _real_end_effector(str(tmp), 0.0, 0.0)
    finally:
        tmp.unlink()


# --- what the collision limit can actually see ------------------------------
#
# Every failure of this constraint so far has been a link the solver could not
# see, and none of them were visible in the solve, in the tracking error, or in
# the published joint angles — only in a video, weeks later. These tests are the
# thing that would have said so.


def test_every_arm_link_from_the_shoulder_to_the_jaws_is_collidable():
    """The gripper and the upper arm are IN the constraint, not just drawn.

    Either can go missing, for different reasons that look
    identical from the outside: the upper arm by being left out of
    `_ARM_COLLISION_BODIES`; the Dex-1 by being built with
    `<visual>` geometry only, so that it compiles to `contype=0` and was filtered out
    of the QP by mink regardless. Naming a body is not the same as the solver
    being able to see it, so this asserts the second thing.
    """
    model = g1d.build_model()
    groups = g1.collision_geom_groups(model, g1d.TRUNK_COLLISION_BODIES,
                                      g1d.ARM_COLLISION_BODIES)
    for side in ("left", "right"):
        got = {model.body(model.geom_bodyid[g]).name for g in groups[side]}
        # `{side}_dex1_base_link` is absent by design: MuJoCo merges it into the
        # wrist across its fixed joint, so its geom arrives under that name.
        assert got == {f"{side}_shoulder_roll_link", f"{side}_shoulder_yaw_link",
                       f"{side}_elbow_link", f"{side}_wrist_roll_link",
                       f"{side}_wrist_pitch_link", f"{side}_wrist_yaw_link",
                       f"{side}_dex1_finger_link_1", f"{side}_dex1_finger_link_2"}
    assert groups["trunk"], "the trunk group must not be empty"


def test_the_dex1_collides_against_the_meshes_upstream_ships_for_it():
    """The jaws use `dex1_col_*.stl`, not their visual mesh.

    MuJoCo collides the CONVEX HULL of a mesh, and the jaws' visual hull spans
    the gap between the fingers — a gripper that is a solid block whenever it is
    open. Upstream ships purpose-built collision meshes for exactly this and the
    model must be using them, so this compares hull volumes rather than trusting
    the filename: the collision hull has to be materially smaller.
    """
    import mujoco
    from scipy.spatial import ConvexHull

    model = g1d.build_model()

    def hull_volume(body: str, collidable: bool) -> float:
        bid = model.body(body).id
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != bid:
                continue
            if bool(model.geom_contype[g] or model.geom_conaffinity[g]) != collidable:
                continue
            mid = model.geom_dataid[g]
            adr, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            return float(ConvexHull(model.mesh_vert[adr:adr + n]).volume)
        raise AssertionError(f"{body} has no {'collision' if collidable else 'visual'} geom")

    for side in ("left", "right"):
        for n in (1, 2):
            body = f"{side}_dex1_finger_link_{n}"
            assert hull_volume(body, True) < 0.55 * hull_volume(body, False)
    # And the jaws really are separate bodies, i.e. not merged into the wrist.
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                             "left_dex1_finger_link_1") > 0


def test_a_collision_body_that_resolves_to_nothing_is_an_error():
    """The failure mode the two tests above exist for, made loud.

    `collision_geom_groups` used to return an empty list for a name no geom
    answered to, which is how the G1-D ran with a trunk group that silently
    dropped `LZ_mt_Link` and, under the humanoid's default set, would have
    dropped the trunk entirely.
    """
    model = g1d.build_model()
    with pytest.raises(ValueError, match="no collidable geom"):
        g1.collision_geom_groups(model, ("pelvis", "torso_link"),
                                 g1d.ARM_COLLISION_BODIES)
    with pytest.raises(ValueError, match="no collidable geom"):
        g1.collision_geom_groups(
            model, g1d.TRUNK_COLLISION_BODIES,
            {s: (f"{s}_dex1_base_link",) for s in ("left", "right")})



def test_the_arm_joint_names_kin_restates_are_the_models():
    """`kin.ARM_JOINT_NAMES` exists so P5 can bound joint-mode actions without
    importing mujoco; it must never drift from the model's own list."""
    assert kin.ARM_JOINT_NAMES == tuple(g1.ARM_JOINTS["left"]
                                        + g1.ARM_JOINTS["right"])


def test_arm_limits_come_from_the_urdf_in_body_order(hw):
    lim = kin.arm_limits(hw.urdf)
    assert lim.shape == (14, 2) and (lim[:, 0] < lim[:, 1]).all()
    # The shoulder roll is mirrored between the arms: left [-1.59, 2.25],
    # right [-2.25, 1.59]. Index 1 and 8 in body order.
    assert lim[1] == pytest.approx([-1.5882, 2.2515])
    assert lim[8] == pytest.approx([-2.2515, 1.5882])
