"""P10's view of the robot: this model, posed from the arena, one frame at a
time.

`dashboards/inspect/scene.py` poses the same model `n` times from a finished
record; this poses it once per dashboard frame from shared memory. Which geoms
exist, how a jaw fraction becomes a slide, how a commanded flange places a
gripper — all of it is shared (`dashboards/web/robot_scene.py`) rather than restated. A dashboard
that drew the robot slightly differently from the inspector would be worse than
one that did not draw it at all: two pictures of the same rollout that disagree,
with nothing to say which of them is wrong.

**Three configurations, which are the three the operator can ask for.**

* **Measured** — the state ring, and the only one always drawn. It is a
  measurement, so it exists in every phase: at INIT, mid-ramp, at READY, and
  during a rollout. The dashboard is a window onto the robot before it is a
  window onto a policy.
* **Solved** — the joint angles P6 published for the tick being executed, which
  is what P7 was actually told to reach.
* **Ghost** — the absolute flange pose the IK was *given* for that tick, drawn
  as a gripper with no arm behind it.

The last two are gated on validity rather than drawn unconditionally, and the
gate is not cosmetic. Outside a rollout there is no policy producing targets at
all, and during P0's homing and exit ramps the target ring carries joint angles
with `pose` deliberately NaN — those ramps have no Cartesian target behind them.
Drawing a ghost there would invent one. The operator asked for exactly
this: measured always, commanded only when something is commanding.

Used by the live server (`dashboards/live/server.py`), inside P10. For the
operator: `docs/deploying_a_policy/running_and_live_dashboard.md`, "The live
dashboard".
"""

from __future__ import annotations

import numpy as np

from elrobot_deploy.runtime.shared_memory import schema
from elrobot_deploy.robot import g1d_model as g1d
from elrobot_deploy.dashboards.web import robot_scene as scene_mod

#: Reused rather than re-derived — see the module docstring. These are private
#: to `web/robot_scene.py` in the sense that nothing outside this package
#: should touch them, not in the sense that the live view should own a second
#: copy: the whole point is that both views pose the same model the same way.
_JointMap = scene_mod._JointMap
_geometry = scene_mod._geometry
_pose7_to_mat = scene_mod._pose7_to_mat
_GRIP_COLUMN = scene_mod._GRIP_COLUMN

#: How many live state samples the base transform is averaged over. The same
#: reasoning as `scene.world_from_base`: a single sample is a reading, a few are
#: a centred estimate, and the quantity is a constant of the two models'
#: relationship rather than anything that moves.
_BASE_SAMPLES = 5


def _jaw_qpos(frac: float) -> float:
    """A normalised aperture onto the model's jaw slide.

    Clamped because the fraction comes from a MEASURED gear angle divided by a
    configured range, and a gripper holding an object reads slightly outside its
    own limits. Clamping draws it at the limit; not clamping would drive the
    slide through the finger.
    """
    frac = min(1.0, max(0.0, frac))
    return g1d.JAW_CLOSED + frac * (g1d.JAW_OPEN - g1d.JAW_CLOSED)


class LiveScene:
    """The model, the static geometry, and one pose per call.

    Built once when P10 starts, because compiling the model and unpacking its
    meshes costs far more than a frame does and neither depends on the robot.
    """

    def __init__(self, cfg, hw, rings):
        import mujoco

        self.mujoco = mujoco
        self.cfg, self.hw, self.rings = cfg, hw, rings
        self.model, self.data = scene_mod.build_model()
        # `_JointMap` wants an object carrying `joints_arm` / `joints_waist`,
        # which is exactly what `Hardware` is — offline it is handed the frozen
        # copy the record kept, and the two are the same numbers.
        self.jmap = _JointMap(self.model, hw)

        geom_ids = [g for g in range(self.model.ngeom)
                    if int(self.model.geom_contype[g]) == 0
                    and int(self.model.geom_type[g]) == mujoco.mjtGeom.mjGEOM_MESH]
        self.geoms, self.vertices, self.normals, self.indices = _geometry(
            self.model, geom_ids)
        self.gid = np.asarray(geom_ids, dtype=np.intp)

        self.span = max(1e-9, hw.hand_max - hw.hand_min)
        # The end effector's TCP in the flange frame — the robot file's
        # `[end_effector] tcp` (`Hardware.tcp`). Only its offset matters
        # for a drawn point. A hand whose robot file states none is drawn at
        # the flange rather than at a TCP borrowed from another hand.
        self.tcp_local = (np.zeros(3) if hw.tcp is None
                          else np.asarray(hw.tcp, dtype=np.float64)[:3].copy())

        # The solved overlay draws only what a body target can move. Wheels and
        # the lift column are excluded for `scene.build`'s reason: a translucent
        # copy of a wheel sitting exactly on the wheel is not a comparison.
        self.solved_slots = [s for s in range(len(self.gid))
                             if self.geoms[s]["part"] in ("body", "gripper")]
        self.solved_gid = self.gid[np.asarray(self.solved_slots, dtype=np.intp)]
        self.solved_data = mujoco.MjData(self.model)

        # The ghost draws the gripper alone, per hand.
        ghost_ids = [[], []]
        for slot, g in enumerate(self.gid):
            body = self.model.body(int(self.model.geom_bodyid[g])).name or ""
            mesh = self.model.mesh(int(self.model.geom_dataid[g])).name or ""
            if "Dex1" not in mesh and "dex1" not in body:
                continue
            ghost_ids[0 if body.startswith("left") else 1].append(slot)
        self.ghost_slots = ghost_ids[0] + ghost_ids[1]
        self.hand_of = [0] * len(ghost_ids[0]) + [1] * len(ghost_ids[1])
        self.ghost_data = mujoco.MjData(self.model)

        self.T_wp: np.ndarray | None = None

    # --- the base transform -------------------------------------------------

    def base_transform(self) -> np.ndarray | None:
        """`T_world_base`, derived from live samples, cached once it exists.

        Same identity as `scene.world_from_base` —
        `T_world_flange @ inv(T_base_flange)` — with the samples coming from the
        state ring instead of a record. P1 computes `flange` at runtime, so this
        is reading its FK rather than recomputing one that could differ.

        Returns None until P1 has published, which is the only reason it is not
        computed in `__init__`: P10 may be up before the state stream is.
        """
        if self.T_wp is not None:
            return self.T_wp
        mats = []
        for _ in range(_BASE_SAMPLES):
            latest = self.rings["state"].latest()
            if latest is None:
                return None
            q = np.asarray(latest["q"], dtype=float)
            self.data.qpos[:] = 0.0
            self.data.qpos[self.jmap.arm_adr] = q[self.jmap.arm_cols]
            self.data.qpos[self.jmap.waist_adr] = q[self.jmap.waist_cols]
            self.mujoco.mj_forward(self.model, self.data)
            for hand, bid in enumerate(self.jmap.wrist_body):
                T_world_flange = np.eye(4)
                T_world_flange[:3, :3] = self.data.xmat[bid].reshape(3, 3)
                T_world_flange[:3, 3] = self.data.xpos[bid]
                T_base_flange = _pose7_to_mat(
                    np.asarray(latest["flange"], dtype=float)[hand])
                mats.append(T_world_flange @ np.linalg.inv(T_base_flange))

        mean = np.mean(mats, axis=0)
        # Averaging rotation matrices elementwise does not give a rotation. The
        # inputs agree closely, so this is a projection rather than a repair.
        u, _, vt = np.linalg.svd(mean[:3, :3])
        out = np.eye(4)
        out[:3, :3] = u @ vt
        out[:3, 3] = mean[:3, 3]
        self.T_wp = out
        return out

    # --- posing -------------------------------------------------------------

    def _pose_into(self, data, q_arm_waist, jaw_fracs) -> None:
        """Write one configuration into `data` and run FK.

        `q_arm_waist` is addressed by the caller, because the two sources index
        differently: a measured `q` is 29 lowstate columns and a body target is
        17 (14 arm, then 3 waist). Both arrive here already reduced to (adr,
        value) pairs the model can take.
        """
        arm_adr, arm_q, waist_adr, waist_q = q_arm_waist
        data.qpos[:] = 0.0
        data.qpos[arm_adr] = arm_q
        data.qpos[waist_adr] = waist_q
        for side, frac in jaw_fracs.items():
            data.qpos[self.jmap.jaw_adr[side]] = _jaw_qpos(frac)
        self.mujoco.mj_forward(self.model, data)

    def _tcp(self, data) -> np.ndarray:
        """The Dex-1's virtual TCP: the wrist-yaw frame's origin plus a rigid
        offset along its own +X (`g1d_model.FLANGE_TO_TCP_X`)."""
        out = np.zeros((2, 3), dtype=np.float32)
        for h, bid in enumerate(self.jmap.wrist_body):
            R = data.xmat[bid].reshape(3, 3)
            out[h] = data.xpos[bid] + R @ self.tcp_local
        return out

    def measured(self):
        """The state ring's newest sample, posed. `None` if P1 has not published.

        Newest rather than interpolated onto anything: there is no other clock
        here to line up with. The inspector interpolates because it is stating a
        measured pose at the same instant as a commanded one it is about to be
        compared against; a live view has only now.
        """
        latest = self.rings["state"].latest()
        if latest is None:
            return None
        q = np.asarray(latest["q"], dtype=float)
        grip = np.asarray(latest["grip_meas"], dtype=float)
        self._pose_into(
            self.data,
            (self.jmap.arm_adr, q[self.jmap.arm_cols],
             self.jmap.waist_adr, q[self.jmap.waist_cols]),
            {side: (float(grip[col]) - self.hw.hand_min) / self.span
             for side, col in _GRIP_COLUMN.items()})

        transforms = np.zeros((len(self.gid), 12), dtype=np.float32)
        transforms[:, :3] = self.data.geom_xpos[self.gid]
        transforms[:, 3:] = self.data.geom_xmat[self.gid]
        return transforms, self._tcp(self.data), q, grip

    def solved(self, target):
        """The joint angles P6 published for this tick, posed as a whole robot.

        Gated on the body target alone and never on `pose`, which is the entire
        point of keeping the two overlays apart: P0's ramps command joints with
        no Cartesian target behind them, and during those the arm is being driven
        by something the ghost cannot show.

        **The trunk here is the COMMANDED one, `q[14:17]`, and the offline viewer
        can show the other.** The target also carries `waist_ik` — the trunk
        P6 pinned the model at, which is the frame `q[:14]` was solved in and is
        the measured trunk rather than the commanded one — and
        rollout inspection poses both and puts a switch between them. This one
        deliberately does not: the live pane draws beside a robot that is
        MEASURED at this same instant, so the trunk the IK used is on screen
        already, as the real one. What the offline viewer buys with the second
        pose is the ability to read a still frame with nothing else to compare
        against; here, moving the overlay's waist onto the measured trunk would
        hide the only difference the operator can see at a glance. If that ever
        stops being true — a policy that drives the waist, where the two trunks
        differ by tracking error rather than a constant droop — the array to pose
        it from is already in the ring.
        """
        q = np.asarray(target["q"], dtype=float)
        hand = self.rings["hand"].get(int(target["k"]))
        jaws = (np.asarray(hand.payload, dtype=float)[:, 0] if hand is not None
                else np.full(2, self.hw.hand_min))
        self._pose_into(
            self.solved_data,
            (self.jmap.arm_adr, q[:schema.ARM_DOF],
             self.jmap.waist_adr, q[self.jmap.waist_slots]),
            {side: (float(jaws[0 if side == "left" else 1]) - self.hw.hand_min)
             / self.span for side in _GRIP_COLUMN})

        transforms = np.zeros((len(self.solved_slots), 12), dtype=np.float32)
        transforms[:, :3] = self.solved_data.geom_xpos[self.solved_gid]
        transforms[:, 3:] = self.solved_data.geom_xmat[self.solved_gid]
        return transforms, self._tcp(self.solved_data), jaws

    def ghost(self, target, jaws):
        """Where the policy asked the gripper to be. `None` when nothing did.

        `target["pose"]` is the FLANGE in base coordinates — the policy
        contract's frame — so one transform
        puts it in the world. NaN means this tick had no Cartesian target, which
        is a fact about the tick and not a missing value to be filled in.

        The jaws are posed once and each gripper geom read RELATIVE to its own
        wrist, then re-anchored on the commanded flange. The ghost's virtual TCP
        then lands on the commanded TCP by construction — same rigid offset, same
        geometry — rather than by an alignment step that could drift from it.
        """
        T_wp = self.base_transform()
        pose = np.asarray(target["pose"], dtype=float)
        if T_wp is None or not np.isfinite(pose).all():
            return None

        self.ghost_data.qpos[:] = 0.0
        for side in _GRIP_COLUMN:
            frac = ((float(jaws[0 if side == "left" else 1]) - self.hw.hand_min)
                    / self.span)
            self.ghost_data.qpos[self.jmap.jaw_adr[side]] = _jaw_qpos(frac)
        self.mujoco.mj_forward(self.model, self.ghost_data)

        T_cmd = [T_wp @ _pose7_to_mat(pose[h]) for h in (0, 1)]
        transforms = np.zeros((len(self.ghost_slots), 12), dtype=np.float32)
        for j, slot in enumerate(self.ghost_slots):
            h = self.hand_of[j]
            wid = self.jmap.wrist_body[h]
            R_w = self.ghost_data.xmat[wid].reshape(3, 3)
            g = int(self.gid[slot])
            rel_R = R_w.T @ self.ghost_data.geom_xmat[g].reshape(3, 3)
            rel_p = R_w.T @ (self.ghost_data.geom_xpos[g]
                             - self.ghost_data.xpos[wid])
            transforms[j, :3] = T_cmd[h][:3, 3] + T_cmd[h][:3, :3] @ rel_p
            transforms[j, 3:] = (T_cmd[h][:3, :3] @ rel_R).reshape(-1)

        tcp = np.zeros((2, 3), dtype=np.float32)
        for h in (0, 1):
            tcp[h] = T_cmd[h][:3, 3] + T_cmd[h][:3, :3] @ self.tcp_local
        return transforms, tcp

    # --- what the browser is sent -------------------------------------------

    def meta(self) -> dict:
        """The static half: what geoms exist and how the mesh buffers are cut.

        Sent once. Nothing in here depends on the robot, the session or the
        rollout — it is the model, and the model does not change while P10 runs.
        """
        return {
            "n_geoms": int(len(self.gid)),
            "n_vertices": int(len(self.vertices) // 3),
            "n_triangles": int(len(self.indices) // 3),
            "geoms": self.geoms,
            "ghost_geoms": [int(s) for s in self.ghost_slots],
            "solved_geoms": [int(s) for s in self.solved_slots],
        }
