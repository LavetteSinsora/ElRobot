"""Posing the robot for the browser: the MuJoCo model the viewers draw, joint
maps from wire order to model order, geometry packing and pose conversion.

Shared by the live dashboard (`dashboards/live/scene.py`) and rollout
inspection (`dashboards/inspect/scene.py`), so the two cannot disagree about
what the robot looks like.
"""

from __future__ import annotations

import numpy as np

from elrobot_deploy.robot import g1d_model as g1d
from elrobot_deploy.runtime.shared_memory import schema
from elrobot_deploy.robot.kinematics import WAIST_JOINT_NAMES


#: Re-exported so the viewer's own code and its tests have one name for it.
FLANGE_TO_TCP_X = g1d.FLANGE_TO_TCP_X


#: `grip_meas` is 12 wide — `(2, 6)` flattened — and the Dex-1 uses only dim 0 of
#: each hand, the jaw radian at the encoder. Hand `h` occupies `[6h, 6h+6)`.
_GRIP_COLUMN = {"left": 0, "right": 6}


#: The 14 arm joints in the [left, right] order every 14-vector in `elrobot_deploy`
#: uses, and which model joint each of the three lowstate waist columns drives.
#: Both are the deploy loop's own, imported rather than restated so the viewer
#: cannot come to pose the robot differently from the process that drove it.
_ARM_JOINTS = tuple(n for side in ("left", "right") for n in g1d.ARM_JOINTS[side])


_WAIST_JOINTS = WAIST_JOINT_NAMES


# --- building the model -----------------------------------------------------


def build_model(urdf=None):
    """The G1-D with the Dex-1 on it, and a fresh `MjData` to pose it in.

    The model is `elrobot_deploy.robot.g1d_model`'s — the *same* one P1's FK and P6's IK run on.
    That shared definition is the point: the viewer draws the robot the loop
    actually solved against, so a geometry bug cannot hide by being present in
    one and absent from the other. `urdf` is the RECORD's model
    (`record_robot`), None the shipped one (the live dashboard's case).
    """
    import mujoco

    model = g1d.build_model(urdf)
    return model, mujoco.MjData(model)


def tcp_point(facts) -> np.ndarray:
    """The virtual TCP in the flange frame, `(3,)`: the recorded robot file's
    `[end_effector] tcp` translation (`record_robot`). A marker is a
    point, so the TCP's rotation plays no part.

    Facts that carry no `tcp` attribute at all get the Dex-1 jaw midpoint; a
    robot whose `tcp` is None (a hand nobody has measured one for) gets the
    flange itself, so residuals are then flange residuals — and the page draws
    no TCP marker (`Scene.meta()["tcp"]`)."""
    if not hasattr(facts, "tcp"):
        return np.array([FLANGE_TO_TCP_X, 0.0, 0.0])
    tcp = facts.tcp
    if tcp is None:
        return np.zeros(3)
    return np.asarray(tcp, dtype=np.float64)[:3].copy()


# --- posing it --------------------------------------------------------------


class _JointMap:
    """qpos addresses by joint name. The only thing that writes `qpos`."""

    def __init__(self, model, facts):
        import mujoco

        def adr(name):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise KeyError(f"{name!r} is not a joint in this model")
            return int(model.jnt_qposadr[jid])

        self.arm_adr = np.array([adr(n) for n in _ARM_JOINTS], dtype=np.intp)
        self.arm_cols = np.asarray(facts.joints_arm, dtype=np.intp)

        # Only the columns with a joint behind them. Lowstate 14 has no motor,
        # and this model has nowhere to put it, so it is dropped rather than
        # written somewhere harmless-looking.
        pairs = [(adr(j), c, s) for s, (j, c)
                 in enumerate(zip(_WAIST_JOINTS, facts.joints_waist)) if j]
        self.waist_adr = np.array([a for a, _, _ in pairs], dtype=np.intp)
        self.waist_cols = np.array([c for _, c, _ in pairs], dtype=np.intp)
        # The same surviving slots, but as indices into a BODY TARGET's waist
        # block — `q[14:17]`, which is `target.waist` and is ordered by slot,
        # not by lowstate column. `waist_cols` addresses the 29-wide measured
        # state and would silently read the wrong three numbers here.
        self.waist_slots = np.array([schema.ARM_DOF + s for _, _, s in pairs],
                                    dtype=np.intp)
        # The same slots again, but as indices into a bare `(3,)` waist vector —
        # `body_targets["waist_ik"]`, which is slot-ordered like `q[14:17]` but
        # does not have the 14 arm columns in front of it. Kept as its own array
        # rather than subtracting `ARM_DOF` at the two call sites, because an
        # off-by-fourteen here reads the wrong three numbers and still renders a
        # robot.
        self.waist_pin_cols = np.array([s for _, _, s in pairs], dtype=np.intp)

        #: The lift column's two telescoping stages. Nothing in a record moves
        #: them — their extension is not on the lowstate wire at all — so they
        #: sit at zero for the measured and solved overlays. `wbik`'s
        #: `waist_column` and `full` tiers do command them, and this is where
        #: those go. Both stages carry half the commanded height, because one
        #: command drives them together (`rt/cmd_hispeed`).
        self.column_adr = np.array([adr("LZ_mt_Joint"), adr("LZ_it_Joint")],
                                   dtype=np.intp)

        self.jaw_adr = {
            side: np.array([adr(f"{side}_dex1_finger_joint_1"),
                            adr(f"{side}_dex1_finger_joint_2")], dtype=np.intp)
            for side in ("left", "right")}

        self.wrist_body = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                              f"{s}_wrist_yaw_link") for s in ("left", "right")]


_PART_RULES = (
    ("gripper", ("dex1", "Dex1")),
    ("wheel", ("Wheel",)),
    ("column", ("LZ_", "AGV_", "Pitching_", "Yaw_Link")),
)


def _part(body: str, mesh: str = "") -> str:
    """Coarse classification, for what the viewer can switch off.

    The MESH name is consulted as well as the body's, because MuJoCo merges a
    link joined by a FIXED joint into its parent: `left_dex1_base_link` has no
    joint of its own, so its geometry ends up on `left_wrist_yaw_link` and would
    otherwise be filed as arm rather than gripper.
    """
    for name, fragments in _PART_RULES:
        if any(f in body or f in mesh for f in fragments):
            return name
    return "body"


def _pose7_to_mat(pose7) -> np.ndarray:
    """`[x y z qw qx qy qz]` -> 4x4. The record's pose convention."""
    w, x, y, z = (float(v) for v in pose7[3:])
    T = np.eye(4)
    T[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = [float(v) for v in pose7[:3]]
    return T


def _geometry(model, geom_ids):
    """Mesh data straight out of the compiled model, so what the viewer draws is
    exactly what the model contains — same vertices, same winding, same units."""
    verts, norms, faces, geoms = [], [], [], []
    v_off = f_off = 0
    for gid in geom_ids:
        mesh_id = int(model.geom_dataid[gid])
        v_adr = int(model.mesh_vertadr[mesh_id])
        v_num = int(model.mesh_vertnum[mesh_id])
        f_adr = int(model.mesh_faceadr[mesh_id])
        f_num = int(model.mesh_facenum[mesh_id])

        verts.append(model.mesh_vert[v_adr:v_adr + v_num].reshape(-1).astype(np.float32))
        norms.append(model.mesh_normal[v_adr:v_adr + v_num].reshape(-1).astype(np.float32))
        faces.append(model.mesh_face[f_adr:f_adr + f_num].reshape(-1).astype(np.uint32))

        rgba = model.geom_rgba[gid]
        mat = int(model.geom_matid[gid])
        if mat >= 0:
            rgba = model.mat_rgba[mat]
        body = model.body(int(model.geom_bodyid[gid])).name or ""
        mesh = model.mesh(mesh_id).name or ""
        geoms.append({
            "name": model.geom(gid).name or mesh or body or f"geom{gid}",
            "body": body,
            "mesh": mesh,
            "part": _part(body, mesh),
            "vertex_offset": v_off, "vertex_count": v_num,
            "index_offset": f_off, "index_count": f_num * 3,
            "rgba": [round(float(c), 4) for c in rgba],
        })
        v_off += v_num
        f_off += f_num * 3

    return (geoms,
            np.concatenate(verts) if verts else np.zeros(0, np.float32),
            np.concatenate(norms) if norms else np.zeros(0, np.float32),
            np.concatenate(faces) if faces else np.zeros(0, np.uint32))
