"""MuJoCo kinematic backend + dual-arm differential IK.

The backend is the swappable "robot" layer: it writes IK solutions straight
into qpos (kinematic playback). **It has no model of its own**: the only
concrete backend is `g1d_model.G1DBackend`, which builds the G1-D from its URDF
(`robot/assets/unitree_g1d`).

IK is mink (QP differential IK on MuJoCo): one FrameTask per flange site +
a low-cost posture task, with every non-arm DoF frozen exactly by zeroing
its velocity before integration (legs/waist never move).

Used by P1 and P6 through `g1d_model.py`, and by the dashboards. The solvers
built on it are described in `docs/infra_logic/ik_solvers.md`.
"""

import os

import mujoco
import numpy as np

from g1d_deploy.robot.kinematics import quat_from_mat

ARM_JOINTS = {
    "left": ["left_shoulder_pitch_joint", "left_shoulder_roll_joint",
             "left_shoulder_yaw_joint", "left_elbow_joint",
             "left_wrist_roll_joint", "left_wrist_pitch_joint",
             "left_wrist_yaw_joint"],
    "right": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint",
              "right_shoulder_yaw_joint", "right_elbow_joint",
              "right_wrist_roll_joint", "right_wrist_pitch_joint",
              "right_wrist_yaw_joint"],
}
EE_SITES = {"left": "left_ee_site", "right": "right_ee_site"}

#: Where the flange is read, per side: `(frame name, MuJoCo frame type)`. mink's
#: `FrameTask` takes the same pair, so one declaration serves both the IK task
#: and the readback and they cannot come to disagree.
#:
#: A site here because this model has one; `g1d_model.G1DBackend` overrides it with a
#: BODY, because URDF has no site element and the site was pinned at zero offset
#: inside `wrist_yaw_link` anyway. Nothing downstream needs to know which.
EE_FRAME = {side: (EE_SITES[side], "site") for side in ("left", "right")}

#: `MjData` arrays holding world position and orientation, per frame type.
_FRAME_ARRAYS = {"site": ("site_xpos", "site_xmat"),
                 "body": ("xpos", "xmat"),
                 "geom": ("geom_xpos", "geom_xmat")}

# Ready pose (rad): all arm joints at zero. For the menagerie G1 this IS the
# natural "ready" configuration the user wants - the elbow link has a built-in
# 90-degree dogleg, so qpos 0 gives upper arm vertical along the torso,
# forearm horizontal pointing forward, hand in front of the body at
# chest/waist height (flange ~[0.20, +-0.15, 0.89], shoulder at z=1.085).
# A touch of shoulder roll keeps the arms clear of the torso shell.
NOMINAL_ARM_QPOS = {
    "left_shoulder_roll_joint": 0.1,
    "right_shoulder_roll_joint": -0.1,
}

# Bodies whose collision geoms take part in self-collision avoidance /
# clearance monitoring.
#
# **The upper arm is in this set, and leaving it out is a real collision.**
# Starting at the elbow looks reasonable, on the reasoning that the shoulder
# links sit permanently near the torso shell and would keep the constraint
# saturated for no benefit. Both halves of that are false on the G1-D, measured
# on one recorded rollout of the reference robot (2026-08-31): at the home pose `shoulder_yaw_link` clears
# the torso by 8.5 mm, and over the run's 912 commanded ticks the links inside
# the set were held at >= -0.5 mm while `left_shoulder_yaw_link` - just outside
# it - reached -38.6 mm and was under zero on 472 of them. The arm was driving
# into the trunk through the one segment nothing was watching, and the measured
# state shows the machine really did it (-9.4 mm; the shell is what stopped it).
#
# Re-solving the whole rollout closed-loop with these two links added takes the
# worst penetration from -38.9 mm to -3.4 mm and the penetrating ticks from 592
# to 12, for +0.065 mm of mean position error and +0.7 ms of solve time. The
# constraint does not saturate either: ticks under 5 mm FALL, from 825 to 431,
# because the arm stops being pressed into the shell in the first place.
#
# `shoulder_pitch_link` stays out and cannot be usefully added: it is
# `torso_link`'s direct child, so mink's parent-child rule drops every pair it
# would form with the trunk. It is also the one link that never moves relative
# to the torso - a constant 10.1 mm over the same run.
_ARM_COLLISION_BODIES = {
    side: (f"{side}_shoulder_roll_link", f"{side}_shoulder_yaw_link",
           f"{side}_elbow_link", f"{side}_wrist_roll_link",
           f"{side}_wrist_pitch_link", f"{side}_wrist_yaw_link")
    for side in ("left", "right")
}
_TRUNK_COLLISION_BODIES = (
    "pelvis", "torso_link",
    "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
)


def collision_geom_groups(model, trunk_bodies=_TRUNK_COLLISION_BODIES,
                          arm_bodies=_ARM_COLLISION_BODIES):
    """-> {'left': [geom ids], 'right': [...], 'trunk': [...]} of collidable
    (contype/conaffinity != 0) geoms grouped for self-collision checks.

    Both sets are parameters because neither is the same robot twice. The
    humanoid's trunk is a pelvis and six hip links; the G1-D's is a wheeled base
    and a lift column (`g1d_model.TRUNK_COLLISION_BODIES`). The arm chains are
    identical from the shoulder to the flange, but only the G1-D carries a Dex-1
    past it (`g1d_model.ARM_COLLISION_BODIES`).

    **A named body that contributes no collidable geom raises.** Every way this
    constraint has failed so far has been a link quietly missing from a group and
    nothing saying so: the G1-D's trunk names, which are not the humanoid's, so
    the default set would return an EMPTY trunk and an IK with no self-collision
    constraint at all; and the Dex-1's links, which were built with `<visual>`
    geometry only, so they compiled to `contype=0` and were filtered out even
    when named. Neither is visible in the solve, in the tracking error, or in the
    published joint angles - only in a video, weeks later. So a name that
    resolves to nothing is an error here rather than an empty list.
    """
    by_body = {}
    for g in range(model.ngeom):
        if model.geom_contype[g] or model.geom_conaffinity[g]:
            by_body.setdefault(model.body(model.geom_bodyid[g]).name, []).append(g)

    def collect(names, what):
        missing = [b for b in names if not by_body.get(b)]
        if missing:
            raise ValueError(
                f"{what} names {missing}, which contribute no collidable geom to "
                f"this model. Either the body does not exist under that name "
                f"(MuJoCo merges fixed-joint links into their parent), or it has "
                f"visual geometry only (contype=0) and needs a <collision> "
                f"element. Silently dropping it would leave the IK unconstrained "
                f"there.")
        return [g for b in names for g in by_body[b]]

    groups = {"trunk": collect(trunk_bodies, "trunk_bodies")}
    for side in ("left", "right"):
        groups[side] = collect(arm_bodies[side], f"arm_bodies[{side!r}]")
    return groups


def self_clearance(backend, groups, side, distmax=0.2):
    """Min signed distance (m, negative = penetration) from `side`'s arm geoms
    to the trunk and to the other arm, at the backend's current qpos. This is
    the deployment-faithfulness monitor: the real robot can never realize a
    state where this is negative.

    One caveat, measured on a recorded rollout: `mj_geomDistance` occasionally
    returns exactly 0.0 for a mesh pair that is genuinely far apart (52 mm in the
    case that found it). It is a convex-distance failure, not a near miss, and it
    is not fixed by the legacy MPR path, which is worse. It never went the other
    way over that whole run - it under-reports clearance, never over-reports it -
    so this stays a safe monitor and the QP stays conservative. But a lone 0.0
    here is worth confirming against the geometry before believing it."""
    other = "right" if side == "left" else "left"
    lo = distmax
    for ga in groups[side]:
        for gb in groups["trunk"] + groups[other]:
            d = mujoco.mj_geomDistance(backend.model, backend.data,
                                       ga, gb, distmax, None)
            lo = min(lo, d)
    return float(lo)



def _se3_inv(T):
    """4x4 rigid inverse. Local rather than shared: `g1d_model.py` carries its own
    copy for the same reason, and five lines of SE(3) is not a dependency."""
    R = T[:3, :3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ T[:3, 3]
    return out


def _pose_of(T):
    """4x4 -> (pos (3,), quat wxyz (4,)).

    `g1d_deploy.robot.kinematics.mat_to_pose` is this same maths returning a flat (7,); mink
    wants the pair. Importing `kinematics` from here is safe in one direction only:
    `robot/kinematics.py` reaches back for `g1d_model` lazily, inside a function."""
    return T[:3, 3].copy(), quat_from_mat(T[:3, :3])


class G1Backend:
    """Kinematic G1: set joint positions, read flange poses, render."""

    #: Overridden by `g1d_model.G1DBackend`; see the module constant.
    EE_FRAME = EE_FRAME

    #: The frame the IK DRIVES, per side, `(name, type)`: the flange unless a
    #: subclass says otherwise (`g1d_model.G1DBackend` with a `tcp` drives the TCP
    #: site). None = `EE_FRAME`.
    ik_frame = None

    def __init__(self, xml_path=None, model=None):
        if model is None and xml_path is None:
            raise ValueError("G1Backend needs a model or an MJCF path; the G1-D "
                             "is g1d.G1DBackend, built from its URDF")
        self.model = model if model is not None \
            else mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        # Resolved once: `flange_pose` is called per lowstate sample at ~1042 Hz
        # and per QP iterate, and a name lookup on that path is pure waste.
        self._ee = {}
        for side, (name, kind) in self.EE_FRAME.items():
            fid = mujoco.mj_name2id(
                self.model, getattr(mujoco.mjtObj, f"mjOBJ_{kind.upper()}"), name)
            if fid < 0:
                raise ValueError(f"{name!r} is not a {kind} in this model")
            self._ee[side] = (fid, *_FRAME_ARRAYS[kind])
        self._ik = {}
        for side, (name, kind) in (self.ik_frame or self.EE_FRAME).items():
            fid = mujoco.mj_name2id(
                self.model, getattr(mujoco.mjtObj, f"mjOBJ_{kind.upper()}"), name)
            if fid < 0:
                raise ValueError(f"{name!r} is not a {kind} in this model")
            self._ik[side] = (fid, *_FRAME_ARRAYS[kind])
        self.arm_qpos_adr = {}
        self.arm_dof_adr = {}
        for side, names in ARM_JOINTS.items():
            self.arm_qpos_adr[side] = np.array(
                [self.model.joint(n).qposadr[0] for n in names])
            self.arm_dof_adr[side] = np.array(
                [self.model.joint(n).dofadr[0] for n in names])
        self.all_arm_dofs = np.concatenate(
            [self.arm_dof_adr["left"], self.arm_dof_adr["right"]])
        self._renderer = None
        self.reset_nominal()

    def reset_nominal(self):
        self.data.qpos[:] = 0.0
        for name, val in NOMINAL_ARM_QPOS.items():
            self.data.qpos[self.model.joint(name).qposadr[0]] = val
        mujoco.mj_forward(self.model, self.data)

    def set_qpos(self, q):
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)

    def flange_pose(self, side):
        """IK-achieved flange pose as 4x4 (this is what "measured" anchors
        read - the sim stand-in for encoder FK on the real robot)."""
        fid, xpos, xmat = self._ee[side]
        T = np.eye(4)
        T[:3, :3] = getattr(self.data, xmat)[fid].reshape(3, 3)
        T[:3, 3] = getattr(self.data, xpos)[fid]
        return T

    def ik_pose(self, side):
        """The pose of the frame the IK drives (`ik_frame`): the TCP when the
        robot file gives one, else the flange. 4x4, world."""
        fid, xpos, xmat = self._ik[side]
        T = np.eye(4)
        T[:3, :3] = getattr(self.data, xmat)[fid].reshape(3, 3)
        T[:3, 3] = getattr(self.data, xpos)[fid]
        return T

    def ik_xpos(self, side):
        """The driven frame's world position (3,)."""
        fid, xpos, _ = self._ik[side]
        return getattr(self.data, xpos)[fid]

    def flange_xpos(self, side):
        """Just the flange's world position (3,), for callers measuring a
        distance. Avoids assembling a 4x4 to throw three quarters of it away."""
        fid, xpos, _ = self._ee[side]
        return getattr(self.data, xpos)[fid]

    def base_pose(self):
        """Pelvis body's 4x4 world pose (the robot base frame states/actions
        are expressed in; constant while the base is fixed)."""
        bid = self.model.body("pelvis").id
        T = np.eye(4)
        T[:3, :3] = self.data.xmat[bid].reshape(3, 3)
        T[:3, 3] = self.data.xpos[bid]
        return T

    def world_to_base(self, T):
        """Re-express a world-frame 4x4 pose in the pelvis (base) frame."""
        return _se3_inv(self.base_pose()) @ T

    def shoulder_anchor(self, side):
        """World position of the shoulder-pitch joint axis (fixed while the
        base/waist are frozen) - the center of the arm's reach sphere."""
        jid = self.model.joint(f"{side}_shoulder_pitch_joint").id
        return self.data.xanchor[jid].copy()

    def max_reach(self, side):
        """Radius of the arm's reach sphere: max shoulder-anchor -> flange
        distance over the elbow's range (other joints at 0; qpos 0 has the
        elbow partly bent, so a single sample underestimates)."""
        saved = self.data.qpos.copy()
        elbow_adr = self.model.joint(f"{side}_elbow_joint").qposadr[0]
        lo, hi = self.model.joint(f"{side}_elbow_joint").range
        r = 0.0
        for e in np.linspace(lo, hi, 40):
            self.data.qpos[:] = 0.0
            self.data.qpos[elbow_adr] = e
            mujoco.mj_forward(self.model, self.data)
            r = max(r, float(np.linalg.norm(
                self.flange_pose(side)[:3, 3] - self.shoulder_anchor(side))))
        self.data.qpos[:] = saved
        mujoco.mj_forward(self.model, self.data)
        return r

    def render(self, width=480, height=360, cam_kwargs=None):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height, width)
            self._cam = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(self.model, self._cam)
            self._cam.azimuth = 150
            self._cam.elevation = -12
            self._cam.distance = 1.25
            self._cam.lookat[:] = [0.2, 0.0, 1.0]
        if cam_kwargs:
            for k, v in cam_kwargs.items():
                setattr(self._cam, k, v)
        self._renderer.update_scene(self.data, self._cam)
        return self._renderer.render()


class DualArmIK:
    """mink-based differential IK; arms only, everything else frozen."""

    def __init__(self, backend, position_cost=1.0, orientation_cost=0.3,
                 posture_cost=1e-3, solver=None, collision_min_dist=0.005,
                 trunk_bodies=_TRUNK_COLLISION_BODIES,
                 arm_bodies=_ARM_COLLISION_BODIES):
        import mink
        import qpsolvers
        self._mink = mink
        self.backend = backend
        self.solver = solver or ("daqp" if "daqp" in qpsolvers.available_solvers
                                 else qpsolvers.available_solvers[0])
        self.config = mink.Configuration(backend.model)
        self.tasks = {}
        # The DRIVEN frame: the TCP site when the backend was built with one,
        # else the flange (`G1Backend.ik_frame`).
        frames = backend.ik_frame or backend.EE_FRAME
        for side in ("left", "right"):
            frame_name, frame_type = frames[side]
            self.tasks[side] = mink.FrameTask(
                frame_name=frame_name, frame_type=frame_type,
                position_cost=position_cost,
                orientation_cost=orientation_cost, lm_damping=1e-2)
        self.posture = mink.PostureTask(backend.model, cost=posture_cost)
        self.config.update(backend.data.qpos.copy())
        self.posture.set_target_from_configuration(self.config)
        self._task_list = [self.tasks["left"], self.tasks["right"], self.posture]
        self._mask = np.zeros(backend.model.nv)
        self._mask[backend.all_arm_dofs] = 1.0
        # Freeze legs/waist INSIDE the QP via a zero velocity limit - masking
        # the returned velocity is not enough (the solver would route motion
        # through the frozen joints and the arm components stall far from the
        # target). The post-hoc mask stays as belt and braces.
        m = backend.model
        arm_names = {n for names in ARM_JOINTS.values() for n in names}
        vel = {m.joint(i).name: (100.0 if m.joint(i).name in arm_names else 1e-9)
               for i in range(m.njnt)}
        self._limits = [mink.ConfigurationLimit(m), mink.VelocityLimit(m, vel)]
        # Self-collision avoidance INSIDE the QP: the solver may never emit a
        # velocity that drives the distal arm into the trunk or the other
        # arm. This is what keeps sim FK states inside the deployment state
        # distribution - the real robot physically cannot self-penetrate, so
        # the kinematic replay must not either. Targets inside the body then
        # show up as large tracking error and are filtered, not silently
        # "achieved". Detection distance is generous (10 cm) so a fast
        # approach cannot tunnel through the constraint in one 33 ms step.
        if collision_min_dist is not None:
            groups = collision_geom_groups(m, trunk_bodies, arm_bodies)
            pairs = [(groups["left"], groups["trunk"] + groups["right"]),
                     (groups["right"], groups["trunk"])]
            self._limits.append(mink.CollisionAvoidanceLimit(
                m, pairs,
                minimum_distance_from_collisions=collision_min_dist,
                collision_detection_distance=0.10))

    def _set_target(self, side, T):
        pos, quat = _pose_of(T)
        se3 = self._mink.SE3.from_rotation_and_translation(
            self._mink.SO3(wxyz=quat), pos)
        self.tasks[side].set_target(se3)

    def solve_tick(self, T_target_left, T_target_right, iters=5, dt=1 / 30.0):
        """Iterate the QP a few times toward the two targets; returns the
        full qpos vector (already applied to the backend)."""
        self._set_target("left", T_target_left)
        self._set_target("right", T_target_right)
        for _ in range(iters):
            try:
                v = self._mink.solve_ik(self.config, self._task_list, dt,
                                        solver=self.solver, damping=1e-3,
                                        limits=self._limits)
            except self._mink.NoSolutionFound:
                # QP infeasible (constraint corner case): hold the current
                # configuration - the tracking error stays large and the
                # tick is filtered downstream, never silently mis-solved
                break
            self.config.integrate_inplace(v * self._mask, dt)
        q = self.config.q.copy()
        self.backend.set_qpos(q)
        return q

    def solve_static(self, T_left, T_right, iters=300, dt=0.02, tol_pos=1e-3):
        """Converge onto a single pose pair (used for the t0 placement gate).
        Returns (pos_err_l, pos_err_r) in meters after convergence."""
        for i in range(iters):
            self.solve_tick(T_left, T_right, iters=1, dt=dt)
            errs = [np.linalg.norm(self.backend.ik_xpos(s) - T[:3, 3])
                    for s, T in (("left", T_left), ("right", T_right))]
            if max(errs) < tol_pos:
                break
        return errs
