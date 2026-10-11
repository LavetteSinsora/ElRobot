"""The G1-D as one model, built once.

Used by P1 (FK), P6 (IK) and the dashboards' 3-D scenes; never by P7. The
user-facing account is `docs/setup/robot_toml.md`, "`g1d_base.toml`: the robot
itself"; the IK that drives it is `docs/infra_logic/ik_solvers.md`.

The G1-D is not the **humanoid** G1 (two 6-DOF legs, a 3-DOF waist, a pelvis at
the root): it is a wheeled base, a telescoping lift column, and a two-axis
waist that bows and yaws. Everything from `torso_link` outward is the same
machine — the arm chain is identical to seven significant figures. Below
`torso_link` it is a different robot, and this module is what makes the deploy
loop pose the real one.

`robot/assets/unitree_g1d/g1_d.urdf` is byte-identical to Unitree's; every change
below is made in memory. It is the ONLY model file the deploy loop names, and
two engines consume it, for two reasons:

  * **MuJoCo** wants geometry — FK, the IK's frames and its collision limit.
    That is this module. It gets the doctored tree: the three-finger hand
    upstream ships is removed (the supported rigs carry a Dex-1, and those
    16.3 MB of meshes are not vendored, so the model would not even compile) and a Dex-1 is
    bolted on in its place.
  * **pinocchio** wants mass — gravity compensation, and nothing else. That is
    `g1d_deploy.runtime.drivers.arm_command.PinocchioComp`, and it lives there rather than here
    because this module subclasses the MuJoCo backend and P7 may not import
    mujoco at all (the 500 Hz loop's import set is asserted by
    `tests/test_p7_body.py`). It reads the URDF exactly as shipped, meshes
    never loaded.

Two things here are traps that have each already cost a rollout, so read them
before touching anything:

**`Yaw_Joint` is the BOW and `torso_Joint` is the YAW.** The names are backwards
from their own axes (`Yaw_Joint` turns about +Y, `torso_Joint` about +Z). Map by
axis or by `WAIST_JOINTS`, never by the name — reading index 13 as a roll
puts the trunk's 2.24 degree forward bow onto a sideways tilt.

**MuJoCo merges links joined by a FIXED joint into their parent.** `AGV_link` is
the URDF root and becomes the worldbody, `Pitching_Link` disappears into
`LZ_it_Link`, and `left_dex1_base_link` into `left_wrist_yaw_link`. Names that
exist in the URDF are not all names that exist in the compiled model, which is
why `BASE_BODY` and `TRUNK_COLLISION_BODIES` read the way they do.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple
from xml.etree import ElementTree

import numpy as np

from g1d_deploy.robot.mujoco_model import _ARM_COLLISION_BODIES, ARM_JOINTS, G1Backend
from g1d_deploy.robot.asset_paths import assets_dir

#: The robot. Byte-identical to Unitree's `unitree_ros` `g1_d_description`.
URDF = assets_dir() / "unitree_g1d" / "g1_d.urdf"

#: The gripper's meshes, from the `g1_29dof_mode_15_with_dex1_1` variant.
DEX1_MESHES = assets_dir() / "unitree_dex1"

#: The two waist joints, proximal first: the bow, then the yaw about the column.
#: Which lowstate column drives which is deploy-side wiring and lives with the
#: rest of it, in `g1d_deploy.robot.kinematics.WAIST_JOINT_NAMES`.
WAIST_JOINTS = ("Yaw_Joint", "torso_Joint")

#: The flange, per side: the wrist-yaw link's own origin, zero offset and
#: identity rotation. Every pose in `g1d_deploy` is expressed at this frame, and
#: the policy's TCP sits a further `FLANGE_TO_TCP_X` along its +X.
#:
#: A BODY rather than a site, unlike the humanoid MJCF's `{side}_ee_site`. URDF
#: has no site element, and the site was pinned at `pos="0 0 0"` inside this very
#: link — so the body frame *is* the site frame, exactly, and adding one back
#: would be XML surgery to reproduce a transform that is already the identity.
FLANGE_FRAME = {side: (f"{side}_wrist_yaw_link", "body")
                for side in ("left", "right")}

#: The TCP, per side: a site on the wrist-yaw link at the robot file's
#: `[end_effector] tcp`, added only when a model is built with one
#: (`build_model(tcp=...)`). What every IK drives when the robot file gives a
#: TCP (`G1DBackend.ik_frame`).
TCP_SITE = {side: (f"{side}_tcp_site", "site") for side in ("left", "right")}

#: The link the pose contract is expressed against — `g1d_deploy.robot.kinematics`'s "base
#: frame", the G1-D's answer to the humanoid's pelvis.
#:
#: It has to be a link BELOW the bow, so that bowing moves the flange *within*
#: the frame rather than moving the frame; that is the whole point of a base
#: frame and it is why the flange pose is a function of the waist at all.
#:
#: Of the candidates — `AGV_link`, the three column tubes, `Pitching_Link` — this
#: is the one ABOVE the lift column, and that is the deciding constraint. The
#: column's two prismatic stages are **not in the lowstate wire format** (there
#: is no index for them; `[joints] waist` is 12/13/14 and 14 is dead), so their
#: extension is unmeasured. Anchoring at `AGV_link` would mean supplying the
#: column height as a constant and wearing any error in it as a permanent z bias
#: in every pose the system produces. Anchored here, the FK does not depend on
#: the column at all — the frame rides it.
#:
#: `Pitching_Link` in the URDF; MuJoCo merges it into `LZ_it_Link`.
BASE_BODY = "LZ_it_Link"

#: Where the humanoid's pelvis sat relative to `torso_link`, with the waist at
#: zero: `waist_yaw` at the pelvis origin, then `-0.0039635 0 0.044` to
#: `waist_roll`, then zero to `torso_link`.
#:
#: The base frame is placed to coincide with that point, which costs nothing and
#: buys exact continuity: with the waist at zero this model reports the same
#: flange pose a humanoid-G1 model does, to the micrometre, so poses and
#: thresholds measured against either stay directly comparable.
#:
#: The frame ORIGIN is otherwise a convention: it is what "the robot base
#: frame" means in the policy contract
#: (`docs/deploying_a_policy/policy_contract.md`, "The base frame"), so moving
#: it changes what every eef policy's absolute poses mean.
PELVIS_FROM_TORSO = np.array([0.0039635, 0.0, -0.044])

#: What the IK's collision limit must steer the arms around: the base, the
#: column, the two waist links and the torso.
#:
#: `world` is `AGV_link` — the URDF root, merged into the worldbody, carrying the
#: wheeled base's own geometry, and `LZ_ot_Link`'s with it: the column's OUTER
#: tube is bolted to the base by a fixed joint and merges the same way, which is
#: why the column needs no name of its own down there. `LZ_it_Link` and
#: `Yaw_Link` move with the bow, so unlike the humanoid's the trunk geometry here
#: is a function of the commanded waist, which is exactly why
#: `g1d_deploy.robot.kinematics.place` writes the waist before the solve rather than zeroing it.
#:
#: **`LZ_mt_Link` is deliberately absent.** The two
#: telescoping stages carry `<visual>` geometry only — no `<collision>` element
#: anywhere in `g1_d.urdf` — so naming it would add nothing (and
#: `collision_geom_groups` raises on it). Upstream is
#: right to leave them out: both tubes nest strictly inside `LZ_ot_Link` (radius
#: 73 mm and 66 mm inside its 112 mm, over the same z span), so the outer tube's
#: volume already covers them. Nothing is lost — over one recorded rollout the
#: nearest any arm geom came to it was 217 mm. `LZ_it_Link` stays because it
#: earns its place a different way: `Pitching_Link`, which IS collidable, merges
#: into it across a fixed joint.
TRUNK_COLLISION_BODIES = (
    "world", "Left_Wheel_Link", "RIght_Wheel_Link",
    "LZ_it_Link", "Yaw_Link", "torso_link",
)

#: The arm side of the same constraint: the shared chain, plus the two jaws.
#:
#: `{side}_dex1_base_link` is deliberately NOT here. MuJoCo merges it into
#: `{side}_wrist_yaw_link` across its fixed joint, so its geom already arrives in
#: the group under that name — naming it as well would raise, since no body in
#: the compiled model answers to it. The jaws are separate bodies (their joints
#: are prismatic) and have to be named.
ARM_COLLISION_BODIES = {
    side: _ARM_COLLISION_BODIES[side] + (f"{side}_dex1_finger_link_1",
                                         f"{side}_dex1_finger_link_2")
    for side in ("left", "right")
}

#: MuJoCo's URDF importer prefixes its own meshdir, so a mesh written as
#: `meshes/X.STL` is resolved at `meshes/meshes/X.STL`.
_MESH_KEY = "meshes/meshes/{}"

#: `{side}_base_joint` in the Dex-1's URDF: where the gripper bolts to the wrist.
DEX1_MOUNT_X = 0.0415

#: The Dex-1's own travel, metres, from the official URDF's prismatic limits.
JAW_CLOSED, JAW_OPEN = -0.02, 0.0245

#: Wrist-yaw origin -> the **virtual TCP**: the midpoint of the two jaw tips at
#: their furthest extent, along the wrist frame's +X with identity rotation.
#: It decomposes as `DEX1_MOUNT_X` + 0.143259 from the gripper base to the
#: tip midpoint. Symmetric jaws, so the midpoint does not move as they open.
#:
#: **The runtime reads the robot file, not this.** A TCP is a property of the
#: end effector (`[end_effector] tcp` in the Dex-1 rig profiles, loaded as
#: `Hardware.tcp`); this constant is the documented value that file
#: carries for the Dex-1, kept for the model-level code that measures against
#: the Dex-1 specifically (the IK planners' residuals, `waist_qp`'s TCP error).
FLANGE_TO_TCP_X = 0.184759


# --- building the model -----------------------------------------------------


def _strip_three_finger_hand(root: ElementTree.Element) -> None:
    """Remove the hand upstream ships. Ours is a Dex-1.

    Dropping these links also drops the only references to their meshes, which
    is why `robot/assets/unitree_g1d/meshes/` does not vendor them — and therefore why
    this is not optional for MuJoCo: the model does not compile without it.
    """
    for link in [e for e in root.findall("link") if "_hand_" in (e.get("name") or "")]:
        root.remove(link)
    for joint in root.findall("joint"):
        child = joint.find("child")
        if child is not None and "_hand_" in (child.get("link") or ""):
            root.remove(joint)


class _Dex1Link(NamedTuple):
    """One gripper link's geometry and mass, verbatim from upstream."""

    visual: str
    collision: str
    mass: str
    com: str
    inertia: dict[str, str]


#: The three Dex-1 links, transcribed from `unitree_ros`'s
#: `g1_29dof_mode_15_with_dex1_1.urdf` (see `robot/assets/unitree_dex1/PROVENANCE.md`).
#:
#: **The jaws collide against `dex1_col_*.stl`, not against their visual mesh.**
#: Upstream ships purpose-built collision hulls for them — 1516 and 1456
#: triangles against the visual meshes' 6276 — and using them rather than
#: substituting the visual mesh is not an optimisation: MuJoCo collides the
#: CONVEX HULL of whatever mesh it is given, and the jaw's visual hull swallows
#: the gap between the fingers. The base link has no such variant upstream and
#: collides against its own mesh, exactly as upstream does.
#:
#: The inertias are upstream's real ones. Nothing in this package integrates
#: them (P7's gravity model is pinocchio's, built from `g1_d.urdf` as
#: shipped, with the gripper's mass added in `arm_command._real_end_effector`).
_DEX1_LINKS = {
    "base": _Dex1Link(
        visual="Dex1_base_link.STL", collision="Dex1_base_link.STL",
        mass="0.19138", com="0.043853 -0.0001306 1.4003E-05",
        inertia=dict(ixx="7.4842E-05", ixy="1.6252E-07", ixz="-8.0516E-09",
                     iyy="5.861E-05", iyz="-3.5931E-08", izz="3.9364E-05")),
    1: _Dex1Link(
        visual="Dex1_finger_link_1.STL", collision="dex1_col_1.stl",
        mass="0.086783", com="0.073453 -0.011173 0.0086355",
        inertia=dict(ixx="2.5583E-05", ixy="5.1557E-07", ixz="-6.3178E-07",
                     iyy="1.4549E-05", iyz="7.8542E-08", izz="3.6013E-05")),
    2: _Dex1Link(
        visual="Dex1_finger_link_2.STL", collision="dex1_col_2.stl",
        mass="0.086783", com="0.073453 0.011173 -0.0086355",
        inertia=dict(ixx="2.5583E-05", ixy="-5.1557E-07", ixz="6.3178E-07",
                     iyy="1.4549E-05", iyz="7.8542E-08", izz="3.6013E-05")),
}


def _dex1_link(name: str, spec: _Dex1Link) -> ElementTree.Element:
    """One gripper link, transcribed from the official Dex-1 URDF.

    Both a `<visual>` and a `<collision>` — and they are DIFFERENT meshes on the
    jaws, exactly as upstream has it. The collision element is the whole point:
    a URDF link with visual geometry only compiles to `contype=0`, which drops it
    from `collision_geom_groups`, from mink's own contype/conaffinity filter, and
    therefore from the IK's avoidance constraint — silently: the gripper would
    be 185 mm of arm the solver cannot see.
    """
    link = ElementTree.Element("link", {"name": name})
    inertial = ElementTree.SubElement(link, "inertial")
    ElementTree.SubElement(inertial, "origin", {"xyz": spec.com, "rpy": "0 0 0"})
    ElementTree.SubElement(inertial, "mass", {"value": spec.mass})
    ElementTree.SubElement(inertial, "inertia", spec.inertia)
    for tag, mesh in (("visual", spec.visual), ("collision", spec.collision)):
        elem = ElementTree.SubElement(link, tag)
        ElementTree.SubElement(elem, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        geometry = ElementTree.SubElement(elem, "geometry")
        ElementTree.SubElement(geometry, "mesh", {"filename": f"meshes/{mesh}"})
    return link


def _mount_dex1(root: ElementTree.Element) -> None:
    """Bolt a Dex-1 onto each wrist, at the offset its own URDF specifies."""
    for side in ("left", "right"):
        base = f"{side}_dex1_base_link"
        root.append(_dex1_link(base, _DEX1_LINKS["base"]))
        joint = ElementTree.SubElement(
            root, "joint", {"name": f"{side}_base_joint", "type": "fixed"})
        ElementTree.SubElement(joint, "origin",
                               {"xyz": f"{DEX1_MOUNT_X} 0 0", "rpy": "0 0 0"})
        ElementTree.SubElement(joint, "parent", {"link": f"{side}_wrist_yaw_link"})
        ElementTree.SubElement(joint, "child", {"link": base})

        for n, axis in ((1, "0 -1 0"), (2, "0 1 0")):
            child = f"{side}_dex1_finger_link_{n}"
            root.append(_dex1_link(child, _DEX1_LINKS[n]))
            jaw = ElementTree.SubElement(
                root, "joint",
                {"name": f"{side}_dex1_finger_joint_{n}", "type": "prismatic"})
            ElementTree.SubElement(jaw, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
            ElementTree.SubElement(jaw, "parent", {"link": base})
            ElementTree.SubElement(jaw, "child", {"link": child})
            ElementTree.SubElement(jaw, "axis", {"xyz": axis})
            # effort/velocity are upstream's; nothing here actuates the jaws,
            # but a number we made up is a number someone will later believe.
            ElementTree.SubElement(jaw, "limit", dict(
                lower=str(JAW_CLOSED), upper=str(JAW_OPEN),
                effort="20", velocity="0.2"))


def _mesh_assets(urdf: Path) -> dict[str, bytes]:
    out = {}
    for directory in (urdf.parent / "meshes", DEX1_MESHES):
        for path in directory.iterdir():
            if path.suffix.upper() == ".STL":
                out[_MESH_KEY.format(path.name)] = path.read_bytes()
    return out


#: The three joints `build_model(mobile_base=True)` inserts above `AGV_link`, in
#: chain order: two prismatic then one revolute, so the base's configuration is
#: `(x, y, theta)` on the floor plane.
#:
#: Named rather than derived because `wbik` addresses them by name and a typo in
#: a string literal there would silently freeze the base instead of failing.
MOBILE_BASE_JOINTS = ("wbik_base_x", "wbik_base_y", "wbik_base_yaw")

#: `TRUNK_COLLISION_BODIES` for the mobile model. `AGV_link` is the URDF root in
#: the fixed model and merges into the worldbody, which is why the fixed set
#: names `"world"`; give it a parent and it becomes an ordinary body carrying its
#: own name — and `LZ_ot_Link`, still fixed to it, still merges into it.
MOBILE_TRUNK_COLLISION_BODIES = tuple(
    "AGV_link" if b == "world" else b for b in TRUNK_COLLISION_BODIES)

#: Travel of the inserted base joints: metres for the two slides, radians for the
#: yaw. Not a hardware limit — the base can drive anywhere — but a bound on how
#: far one 25-tick chunk is allowed to reposition, which is what keeps the QP
#: from answering "drive across the room" to a target the arm merely cannot
#: quite reach. `wbik` tightens it further through its own rate limits.
MOBILE_BASE_RANGE = {"wbik_base_x": 2.0, "wbik_base_y": 2.0, "wbik_base_yaw": np.pi}


def _add_mobile_base(root: ElementTree.Element) -> None:
    """Insert an (x, y, yaw) chain above `AGV_link`, making the base steerable.

    The deploy loop's model is fixed-base on purpose — "the machine does not
    drive while it manipulates" — and that is still the default. This exists for
    `g1d_deploy.runtime.ik.waist_qp`, where the base is one more column of the Jacobian and the
    QP decides whether it is worth paying for.

    Three single-DOF joints rather than one `planar`/`free` joint, because URDF
    has neither and this file's whole method is in-memory URDF surgery. It also
    keeps each DOF individually addressable by name, which is what
    `mink.VelocityLimit` and `wbik`'s per-DOF cost vector both want.

    The two carrier links are massless in spirit but not in the file: MuJoCo
    refuses a body that has a DOF and no inertia, so they carry 1 g and a
    negligible diagonal. Nothing integrates them — `wbik` is kinematic, and
    P7's gravity model is pinocchio's, built from the URDF **as shipped** and
    therefore untouched by anything in this function.
    """
    parent = "wbik_base_world"
    root.append(ElementTree.Element("link", {"name": parent}))
    spec = [(MOBILE_BASE_JOINTS[0], "prismatic", "1 0 0", "wbik_base_x_link"),
            (MOBILE_BASE_JOINTS[1], "prismatic", "0 1 0", "wbik_base_y_link"),
            (MOBILE_BASE_JOINTS[2], "revolute", "0 0 1", "AGV_link")]
    for name, kind, axis, child in spec:
        if child != "AGV_link":
            link = ElementTree.SubElement(root, "link", {"name": child})
            inertial = ElementTree.SubElement(link, "inertial")
            ElementTree.SubElement(inertial, "origin",
                                   {"xyz": "0 0 0", "rpy": "0 0 0"})
            ElementTree.SubElement(inertial, "mass", {"value": "0.001"})
            ElementTree.SubElement(inertial, "inertia",
                                   dict(ixx="1e-6", ixy="0", ixz="0",
                                        iyy="1e-6", iyz="0", izz="1e-6"))
        joint = ElementTree.SubElement(root, "joint", {"name": name,
                                                       "type": kind})
        ElementTree.SubElement(joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        ElementTree.SubElement(joint, "parent", {"link": parent})
        ElementTree.SubElement(joint, "child", {"link": child})
        ElementTree.SubElement(joint, "axis", {"xyz": axis})
        r = MOBILE_BASE_RANGE[name]
        ElementTree.SubElement(joint, "limit", dict(
            lower=str(-r), upper=str(r), effort="1000", velocity="10"))
        parent = child


def build_model(urdf: str | Path | None = None, *, mobile_base: bool = False,
                tcp=None):
    """The G1-D with the Dex-1 on it, compiled. Returns an `MjModel`.

    Fixed base by default: `AGV_link` is the URDF root and no free joint is
    added, so it compiles straight into the worldbody. That is what the deploy
    loop wants — the machine does not drive while it manipulates — and it is why
    `base_pose()` is constant and can be cached.

    `mobile_base=True` inserts an (x, y, yaw) chain above it instead, for
    `g1d_deploy.runtime.ik.waist_qp`. Nothing on the default path changes: the argument defaults
    to the behaviour every existing caller already gets, and the two models
    differ in exactly three joints.

    `tcp` — `(7,)` `[x y z qw qx qy qz]` in the flange frame, or None — adds a
    `TCP_SITE` on each wrist-yaw link there. Sites carry no mass and no
    collision; the kinematics are unchanged.
    """
    import mujoco

    path = Path(urdf) if urdf is not None else URDF
    root = ElementTree.parse(path).getroot()
    _strip_three_finger_hand(root)
    _mount_dex1(root)
    if mobile_base:
        _add_mobile_base(root)
    xml = ElementTree.tostring(root, encoding="unicode")
    assets = _mesh_assets(path)
    if tcp is None:
        return mujoco.MjModel.from_xml_string(xml, assets)
    tcp = np.asarray(tcp, dtype=np.float64).reshape(7)
    spec = mujoco.MjSpec.from_string(xml, assets=assets)
    for side, (name, _kind) in TCP_SITE.items():
        spec.body(FLANGE_FRAME[side][0]).add_site(
            name=name, pos=tcp[:3].tolist(), quat=tcp[3:].tolist())
    return spec.compile()


class G1DBackend(G1Backend):
    """`G1Backend` on the real machine: G1-D geometry, based at the column.

    Everything the humanoid backend does is inherited unchanged — the arm joints
    resolve by the same names, and `flange_pose` reads `FLANGE_FRAME`. What this
    overrides is the one thing that has no counterpart: `base_pose`, because
    there is no pelvis.
    """

    EE_FRAME = FLANGE_FRAME

    def __init__(self, urdf: str | Path | None = None, model=None, *,
                 mobile_base: bool = False, tcp=None):
        import mujoco

        #: The robot file's `(7,)` TCP this backend was built with, or None.
        #: With one, the model carries `TCP_SITE` and the IK drives it
        #: (`ik_frame`); without, the IK drives the flange.
        self.tcp = None if tcp is None else np.asarray(tcp, np.float64).reshape(7)
        self.ik_frame = TCP_SITE if self.tcp is not None else FLANGE_FRAME

        #: Whether this backend's model carries `MOBILE_BASE_JOINTS`. Read by
        #: `waist_qp` to pick the matching collision set, and by `base_pose`'s
        #: caching decision — with a steerable base the base frame is no longer
        #: constant and must not be treated as though it were.
        self.mobile_base = bool(mobile_base)
        super().__init__(model=build_model(urdf, mobile_base=mobile_base,
                                           tcp=self.tcp)
                         if model is None else model)
        self._base_body = int(self.model.body(BASE_BODY).id)

        # The base frame, once, as a constant offset from `BASE_BODY`. Evaluated
        # at the zero configuration: that is where `PELVIS_FROM_TORSO` is defined
        # (the humanoid's pelvis with its waist at zero), and the offset is rigid
        # thereafter — which is what makes the frame independent of both the
        # waist above it and the lift column below.
        self.data.qpos[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        torso = self._body_pose(int(self.model.body("torso_link").id))
        torso[:3, 3] += torso[:3, :3] @ PELVIS_FROM_TORSO
        self._base_local = _se3_inv(self._body_pose(self._base_body)) @ torso
        self.reset_nominal()

    def _body_pose(self, bid: int) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.data.xmat[bid].reshape(3, 3)
        T[:3, 3] = self.data.xpos[bid]
        return T

    def base_pose(self) -> np.ndarray:
        """The 4x4 world pose of the frame every state and action is expressed
        in. Constant while the base is fixed, which it always is here."""
        return self._body_pose(self._base_body) @ self._base_local


def _se3_inv(T: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


__all__ = [
    "ARM_COLLISION_BODIES",
    "ARM_JOINTS",
    "BASE_BODY",
    "MOBILE_BASE_JOINTS",
    "MOBILE_BASE_RANGE",
    "MOBILE_TRUNK_COLLISION_BODIES",
    "DEX1_MOUNT_X",
    "FLANGE_FRAME",
    "FLANGE_TO_TCP_X",
    "JAW_CLOSED",
    "JAW_OPEN",
    "PELVIS_FROM_TORSO",
    "TCP_SITE",
    "TRUNK_COLLISION_BODIES",
    "URDF",
    "WAIST_JOINTS",
    "G1DBackend",
    "build_model",
]
