"""MuJoCo, offline: the meshes once, then a transform per geom per frame.

MuJoCo never runs in the browser. This module poses the model at every display
frame and emits two payloads — static geometry, and a flat array of world
transforms — which a few hundred lines of WebGL can draw without knowing what a
robot is. That split is the whole reason the viewer can orbit freely: the
expensive, physics-shaped work happens here, once, and the browser only
multiplies matrices.

**The model is `g1d_deploy.robot.g1d_model`'s — the same one the deploy loop
poses.** There is one model file for one robot, and the picture is of the
geometry P6 actually solved against. The shared posing code is
`dashboards/web/robot_scene.py`.

**Joints are addressed by NAME, never by qpos slice.** `d.qpos[:] = q` would
only be right for a model whose qpos is in lowstate order. The G1-D's qpos begins with two
wheels and two column stages, so `left_shoulder_pitch` sits at 6 and
`right_wrist_yaw` at 26, and mounting the Dex-1 moves them again. A positional
assignment would still run and still look like a robot.

**Three configurations are posed, not one, and the reason is a diagnosis.**
A rollout that misses its target misses it for one of two reasons, and they need
opposite fixes: either the IK could not reach the pose the policy asked for, or
it solved it fine and the arm did not go there. Those are indistinguishable from
a picture of the commanded POSE and the measured arm alone. So the joint
angles P6 published — `body_targets["q"]` — are posed as well, and the three sit in one scene:

    commanded pose  ->  the ghost gripper, the IK's INPUT
    commanded q     ->  the solved overlay, the IK's OUTPUT
    measured q      ->  the opaque robot

The solved overlay is posed TWICE and shipped twice, because a body target says
two things about the waist and only one of them is what the arm angles mean.
`q[14:17]` is the waist commanded for the tick, which is what P7 drove; the
`waist_ik` column beside it is the waist P6 pinned the model at to solve for
`q[:14]`, which is the measured trunk and on the reference robot is
permanently 0.039 rad away from the commanded one. Posing the solved arm on the
commanded trunk draws a configuration that existed nowhere — the arm is right,
the torso is 19 mm out, and the two errors are not the same error. Both arrays
go to the browser over the same geoms in the same order and a switch picks one;
the second costs one `mj_forward` per commanded frame. The alternative was to
rotate the first array about the waist in JavaScript, which is exact only while
nobody adds a geom proximal to the trunk and is a second definition of this
robot's kinematics living somewhere the model cannot check it.

The gap from the first to the second is the IK's own failure to reach, and P6
recorded its size in `body_targets["err"]`. The gap from the second to the third
is everything downstream of the solve: servo lag, gravity sag, a joint on its
limit. With neither on screen, 82 mm of under-reach can hide behind
ordinary-looking joint angles.

**The waist mapping is by physics, not by the record's names.** A record says
which *model* joint each waist column drove, and a record may name joints of
another model (`waist_pitch_joint` and friends exist in the humanoid G1's MJCF
and not here). What the hardware is does not depend on that: lowstate 12 is
the yaw about the column, lowstate 13 is the bow, lowstate 14 has no motor. So
the three columns map to `torso_Joint`, `Yaw_Joint` and nothing — and note that
those two URDF names are backwards from their own axes, which is exactly the
trap this comment exists to disarm.

Built once per Load by the inspection server (`server.py`). For a user:
`docs/deploying_a_policy/rollout_inspection.md`, "The view".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from g1d_deploy.robot import g1d_model as g1d

from g1d_deploy.runtime.shared_memory import schema
from g1d_deploy.dashboards.web.robot_scene import (
    _GRIP_COLUMN, _JointMap, _geometry,
    _pose7_to_mat, build_model, tcp_point)


@dataclass
class Scene:
    """Static geometry, and where every piece of it is at each frame."""

    transforms: np.ndarray          # (n_frames, n_geom, 12): xyz then row-major R
    geoms: list[dict]
    vertices: np.ndarray
    normals: np.ndarray
    indices: np.ndarray
    tcp: np.ndarray                 # (n_frames, 2, 3) world virtual-TCP positions

    #: The commanded overlay — a Dex-1 with no robot behind it, placed where the
    #: policy asked the gripper to be. `(n_frames, 6, 12)`, the six gripper geoms
    #: named by `ghost_geoms`.
    ghost: np.ndarray
    #: Which entries of `geoms` the ghost's six transforms correspond to, so the
    #: browser reuses those meshes instead of being sent them twice.
    ghost_geoms: list[int]
    ghost_tcp: np.ndarray           # (n_frames, 2, 3)
    #: `(n_frames,)` uint8. 0 means no target was in force — homing, warmup, the
    #: wait at READY, the exit ramp. The overlay hides rather than holding.
    ghost_valid: np.ndarray

    #: The SOLVED overlay: the robot posed at the joint angles P6 published,
    #: which is what the IK produced from the ghost's pose and what P7 was told
    #: to drive to. `(n_frames, len(solved_geoms), 12)`.
    solved: np.ndarray
    #: Which entries of `geoms` those transforms are for — every geom the arms
    #: and the waist can move, so the browser reuses meshes it already holds.
    #: The wheels and the lift column are excluded: nothing in a body target
    #: moves them, so a second copy would only alpha-blend onto itself.
    solved_geoms: list[int]
    solved_tcp: np.ndarray          # (n_frames, 2, 3)
    #: `(n_frames,)` uint8, and DELIBERATELY wider than `ghost_valid`: a body
    #: target always carries joint angles, including P0's homing and exit ramps,
    #: which carry no Cartesian target at all. Where this is 1 and `ghost_valid`
    #: is 0, the arm was being commanded in joint space by P0 and there is no IK
    #: in the picture to blame. Shared by both waist variants below.
    solved_valid: np.ndarray

    #: The SAME published joint angles, posed on the trunk the IK was PINNED at
    #: rather than on the one the tick commanded — `body_targets["waist_ik"]`.
    #: `(n_frames, len(solved_geoms), 12)`, the same geoms in the same order, so
    #: the browser swaps one array for the other and nothing else changes.
    #:
    #: Two arrays and not one because the choice is a question, not a setting.
    #: The arm angles only mean what they mean in the frame they were solved in,
    #: so this one is where the robot actually was; the other is what P7 was told
    #: to drive every joint to, waist included. Today they differ by 0.039 rad of
    #: gravity droop the commanded zero never reaches, and the day a policy
    #: commands a nonzero waist they will differ by whatever the trunk fails to
    #: track. Neither answer is derivable from the other on this side.
    #:
    #: EMPTY — `(0,)` — when the record predates `waist_ik`. Not a copy of
    #: `solved`: a switch that redraws the identical picture is worse than one
    #: that is visibly unavailable, and the empty array is what tells the browser
    #: which it has.
    solved_pinned: np.ndarray
    solved_pinned_tcp: np.ndarray   # (n_frames, 2, 3), or (0,)

    #: The WHOLE-BODY overlays, `{tier: (n_frames, len(wbik_geoms), 12)}`, from
    #: the IK bench's sidecar (`ik_bench.npz`). Empty when the recording has not been
    #: solved — the viewer greys those switches rather than offering a picture
    #: it cannot draw.
    #:
    #: **Over EVERY geom, unlike `solved`.** The solved overlay skips the wheels
    #: and the lift column because nothing in a body target moves them, and a
    #: translucent wheel on top of a wheel is haze rather than information. Here
    #: they genuinely move: two of the three tiers command the column, and one
    #: commands the base, which carries the whole machine with it.
    wbik: dict = field(default_factory=dict)
    #: `{tier: (n_frames, 2, 3)}` world virtual-TCP positions per tier.
    wbik_tcp: dict = field(default_factory=dict)
    #: `{tier: (n_frames,)}` TCP residual in METRES, for the readout. The
    #: distance from the commanded TCP to the one this tier reached — the
    #: quantity the whole comparison is about, and deliberately not the flange
    #: error `BODY_TARGET.err` carries: the TCP sits 184.759 mm along the
    #: flange's +X, so orientation error becomes position error there and the
    #: flange metric cannot see it.
    wbik_err: dict = field(default_factory=dict)
    #: Which entries of `geoms` the whole-body transforms correspond to.
    wbik_geoms: list = field(default_factory=list)
    #: `(n_frames,)` uint8 — a solved tick was in force AND the sidecar covers
    #: it. Shared by every tier, because they are solved from the same ticks.
    wbik_valid: np.ndarray = None
    #: `(n_frames,)` the arms-only TCP residual over the same ticks, re-solved
    #: rather than read from the record: `BODY_TARGET.err` is at the flange and
    #: there is no way to convert it after the fact, because that needs the
    #: orientation and the record does not store it.
    wbik_off_err: np.ndarray = None
    #: What each overlay name IS, for the switch it gets: `{name: {"source":
    #: "wbik" | "bench", "mode", "describe", ...}}` from `overlays.load`. Empty
    #: when there is no sidecar. `notes` carries anything the merge had to
    #: drop, so a stale cache says so on screen instead of silently vanishing.
    wbik_meta: dict = field(default_factory=dict)
    #: Which half of the action contract the rollout's policy spoke
    #: (`loader.action_mode`). In "joint" the commanded POSE exists only
    #: where an operator intervened, and the page greys the switch otherwise.
    action_mode: str = "eef"
    #: How many display frames an operator's (PICO) tick was in force on.
    intervention_frames: int = 0
    #: Does the recorded robot define a virtual TCP (`record_robot.tcp`)?
    has_tcp: bool = True

    def meta(self) -> dict:
        return {
            "n_frames": int(self.transforms.shape[0]),
            "n_geoms": int(self.transforms.shape[1]),
            "n_vertices": int(len(self.vertices) // 3),
            "n_triangles": int(len(self.indices) // 3),
            "geoms": self.geoms,
            "ghost_geoms": self.ghost_geoms,
            "ghost_frames": int(self.ghost_valid.sum()),
            "solved_geoms": self.solved_geoms,
            "solved_frames": int(self.solved_valid.sum()),
            # Whether the solved overlay can be posed on the IK's own trunk.
            # False for a record without `waist_ik`; the switch greys out.
            "solved_pinned": bool(self.solved_pinned.size),
            # The methods this recording has been solved for, in the order
            # they should appear. Empty means no sidecar — IK mode solves them.
            "wbik_tiers": list(self.wbik),
            "wbik_meta": self.wbik_meta,
            "wbik_geoms": self.wbik_geoms,
            "wbik_frames": (int(self.wbik_valid.sum())
                            if self.wbik_valid is not None else 0),
            "action_mode": self.action_mode,
            "intervention_frames": int(self.intervention_frames),
            # False when the recorded robot has no TCP (`record_robot`): the
            # markers would sit on the flange, so the page does not draw them.
            "tcp": bool(self.has_tcp),
        }


def world_from_base(rollout, model, data, jmap, *, samples: int = 5) -> np.ndarray:
    """`T_world_base`: the record's frame -> the viewer's frame.

    Derived from the record rather than looked up, and it stays derived now that
    the deploy loop and the viewer share a model — because a record does not.
    Rollouts written before the switch express their poses in the humanoid MJCF's
    PELVIS frame, and that model bows about a different point than this one
    (`g1d.BASE_BODY`): the same measured 0.039 rad of trunk droop puts the flange
    about 4 mm apart in the two. Deriving the transform absorbs that, so an old
    rollout still draws with its own commanded poses landing on its own arm.

    Both models agree on the arm chain, so posing both with the same measured
    joints cancels it and leaves only the roots' relationship:

        T_world_base = T_world_flange @ inv(T_base_flange)

    `T_base_flange` is read straight from `state.bin` — P1 computed it at
    runtime, and a fresh recompute matches it to 1e-16, so recomputing would only
    add a dependency on the recording machine's model being loadable here.

    It is a constant: the base sits below the waist in either model, and the lift
    column — the one thing that would move it — is drawn retracted because its
    height is not recorded. Averaging a few samples turns a
    single-sample reading into a centred estimate. For a rollout recorded on the
    G1-D model this must come back as `G1DBackend.base_pose()` exactly, and the
    tests assert precisely that.
    """
    import mujoco

    n = len(rollout.state)
    picks = np.linspace(0, n - 1, min(samples, n)).astype(int)
    arm_cols = jmap.arm_cols
    waist_cols = jmap.waist_cols
    mats = []
    for i in picks:
        q = rollout.state["q"][i]
        data.qpos[:] = 0.0
        data.qpos[jmap.arm_adr] = q[arm_cols]
        data.qpos[jmap.waist_adr] = q[waist_cols]
        mujoco.mj_forward(model, data)
        for hand, bid in enumerate(jmap.wrist_body):
            T_world_flange = np.eye(4)
            T_world_flange[:3, :3] = data.xmat[bid].reshape(3, 3)
            T_world_flange[:3, 3] = data.xpos[bid]
            T_base_flange = _pose7_to_mat(rollout.state["flange"][i][hand])
            mats.append(T_world_flange @ np.linalg.inv(T_base_flange))

    mean = np.mean(mats, axis=0)
    # Re-orthonormalise: averaging rotation matrices elementwise does not give a
    # rotation. The inputs agree to 0.01 deg so this is a projection, not a fix.
    u, _, vt = np.linalg.svd(mean[:3, :3])
    out = np.eye(4)
    out[:3, :3] = u @ vt
    out[:3, 3] = mean[:3, 3]
    return out


def _poses_from_record(body, rows):
    """The recorded IK input, straight off `body_targets`.

    No reconstruction: `body["pose"]` IS
    `chunk_blend.apply(fn, chunks.covering(k))` — the array P6 handed the solver — so
    there is no chunk lookup, no coverage test, no blend replay and no mode
    distinction: sync and async are the same two lines, and a custom blend
    function that this package could never have imported is already resolved.

    NaN marks a tick with no Cartesian target behind it (P0's homing and exit
    ramps), which is a recorded fact rather than the `k < 1` inference the
    reconstruction had to make.
    """
    poses = body["pose"][rows]
    return poses, np.isfinite(poses).all(axis=(1, 2))


def _poses_from_chunks(body, chunks, rows):
    """The same poses reconstructed for a record written before `pose` existed.

    **Sync only, and it declines rather than guesses.** A tick's contributors are
    whatever `covering()` found RESIDENT in the ring when P6 asked, which depends
    on a `chunk_id % chunk_slots` eviction schedule and on a wall-clock instant
    no record contains — `deadline` is when the tick should execute, and
    `t_observed` is P9 seeing the write already finished. In sync that does not
    matter: `n_actions == P`, so chunks partition the tick line and the single
    covering chunk is the only contributor there could have been.

    In async two chunks overlap by design and the answer is a blend of both.
    Reconstructing it would mean picking contributors from a ~35 ms window (the
    IK budget plus P9's sweep period) that straddles a replan whenever P is
    small — the exact regime async exists for. So an overlap returns invalid, and
    the overlay stays empty rather than drawing a pose nobody commanded.
    """
    kf = chunks["k_first_action"].astype(np.int64)
    na = chunks["n_actions"].astype(np.int64)
    ks = body["k"][rows].astype(np.int64)

    poses = np.zeros((len(rows), 2, 7))
    valid = np.zeros(len(rows), dtype=bool)
    for j, k in enumerate(ks):
        if k < 1:
            # Homing and the exit ramp are P0's: joint ramps with no Cartesian
            # target behind them. Drawing the previous chunk's pose would invent
            # a goal nobody commanded.
            continue
        hit = np.flatnonzero((kf <= k) & (k < kf + na))
        if len(hit) != 1:
            continue
        c = chunks[hit[0]]
        poses[j] = c["pose"][k - int(c["k_first_action"])]
        # A joint-mode chunk's `pose` is NaN: it covers the tick, but there is
        # no Cartesian target in it to draw.
        valid[j] = bool(np.isfinite(poses[j]).all())
    return poses, valid


class Commanded(NamedTuple):
    """Everything a body target says, resampled onto the display frames."""

    poses: np.ndarray               # (n, 2, 7) commanded flange, base frame
    jaws: np.ndarray                # (n, 2) commanded jaw, gear radians
    #: A CARTESIAN target was in force and is drawable. Gates the ghost.
    valid: np.ndarray               # (n,) uint8
    #: (n, 17) the joint angles P6 published for the tick in force, and...
    q: np.ndarray
    #: ...whether there was a tick in force at all. Wider than `valid`: P0's
    #: ramps are joint-space, so they set this and not that.
    q_valid: np.ndarray             # (n,) uint8
    #: (n, 3) the trunk each tick's IK was PINNED at, in body-target slot order —
    #: `body_targets["waist_ik"]`, the third argument P6 handed the solver.
    #: Already resolved against `q[14:17]`, so there are no NaNs here and no
    #: second validity byte: see `_commands` for why that fallback is the honest
    #: answer rather than a hole.
    waist_ik: np.ndarray
    #: Whether a pinned waist was RECORDED at all, for the whole rollout — a
    #: property of the record, not of the frame. False leaves `waist_ik` equal to
    #: the commanded waist everywhere, so posing it is defined but tells you
    #: nothing new, and the switch that picks it is disabled instead.
    has_pinned_waist: bool = False
    #: `(n,)` int64: which row of `body_targets` is in force at each frame, or
    #: -1. The whole-body overlays are stored per SOLVED TICK and keyed by `k`,
    #: while everything here is per display frame, and this is the only bridge
    #: between the two indexings. Derived rather than re-derived in `build`,
    #: because `timeline.index["command"]` is the authority on which tick a frame
    #: belongs to and two answers to that question is one too many.
    row: np.ndarray = None
    #: `(n,)` uint8: the tick in force was an operator's (PICO takeover).
    #: Its pose is an EEF pose whatever the policy's mode, so in a joint-mode
    #: rollout these are the only frames the commanded pose can be drawn on.
    intervened: np.ndarray = None


def _commands(rollout, timeline) -> Commanded:
    """Per display frame: the whole body target in force, resampled.

    Two sources for the POSE, and which one a record gets is a property of the
    record. `body_targets` carrying a `pose` column means P6 wrote down the
    array it handed the IK, and that is read verbatim. Older records have to
    reconstruct it from the chunk ring, which is exact in sync and impossible in
    async — see the two helpers above.

    Nothing else is part of that split. `q` has been the ring's whole point
    since the first record — it is what P7 drives to — and `hand["cmd"]` has
    always been the blend's own output, stored untouched because nothing
    consumes it on the way to the gripper. Only the pose was ever lost, because
    only the pose was eaten by the solver, which is exactly why `q` can be posed
    with no generation check and no reconstruction: the IK's output was never in
    doubt, only its input.

    **The waist is TWO answers and this builds both.** `q[14:17]` is the waist
    the tick commanded, which is what P7 put on the wire; `waist_ik` is the waist
    P6 pinned the model at to solve for `q[:14]`, which is the
    MEASURED trunk and on the reference robot is permanently 0.039 rad away from the
    commanded one. Both are true statements about the same tick, they pose to
    visibly different robots, and neither is the answer to the other's question —
    so the record carries both and the viewer switches between them.

    Where `waist_ik` is NaN it falls back to the commanded waist, and that is not
    a patch over missing data. NaN marks a tick P0 wrote — the homing and exit
    ramps — where there is no IK and therefore no frame anything was pinned in;
    the commanded waist is then the only waist in the picture and posing it is
    exactly right. The fallback also covers a record written before the column
    existed, where it collapses the pinned overlay onto the commanded one, which
    is precisely what such a record can support.
    """
    n = timeline.n
    out = Commanded(
        poses=np.zeros((n, 2, 7)),
        jaws=np.zeros((n, 2)),
        valid=np.zeros(n, dtype=np.uint8),
        q=np.zeros((n, schema.BODY_DOF)),
        q_valid=np.zeros(n, dtype=np.uint8),
        waist_ik=np.zeros((n, schema.WAIST_DOF)),
        has_pinned_waist=rollout.has_pinned_waist,
        row=np.full(n, -1, dtype=np.int64),
        intervened=np.zeros(n, dtype=np.uint8))
    column = timeline.index.get("command")
    if column is None:
        return out

    body = rollout.body_targets
    frames = np.flatnonzero(column >= 0)
    if not len(frames):
        return out
    rows = column[frames].astype(np.intp)

    # The joint side first, and unconditionally: it needs neither a recorded
    # pose nor a chunk ring, so a record whose ghost declines still poses its
    # solved overlay. That is the case the overlay is most wanted in.
    out.q[frames] = body["q"][rows]
    out.q_valid[frames] = 1
    out.row[frames] = rows

    # The pinned trunk, defaulted to the commanded one and overwritten wherever
    # the record actually has a solve behind the tick. Written this way round so
    # that a missing column, a P0 ramp and a NaN from any future cause all land
    # on the same defined answer without a branch each.
    out.waist_ik[frames] = out.q[frames][:, schema.ARM_DOF:]
    if rollout.has_pinned_waist:
        pinned = body["waist_ik"][rows]
        got = np.isfinite(pinned).all(axis=1)
        out.waist_ik[frames[got]] = pinned[got]

    if rollout.has_commands:
        if rollout.has_recorded_pose:
            got, ok = _poses_from_record(body, rows)
        else:
            got, ok = _poses_from_chunks(body, rollout.chunks, rows)
        out.poses[frames] = got
        out.valid[frames] = ok

    # The operator's ticks. In a joint-mode rollout these are the
    # only rows with a Cartesian target, and the finite-pose test above has
    # already let exactly them through; the mask is carried so the page can
    # say that is what it is drawing.
    out.intervened[frames] = rollout.intervened[rows]

    # `k` -> the hand target written in the same P6 iteration (body and hand
    # share one row), so the jaw shown is the one commanded for this very tick.
    # Looked up for every commanded frame rather than only the drawable ones,
    # because the solved overlay carries a gripper through P0's ramps, where the
    # ghost does not exist — and P0 ramps BOTH rings, so there is an answer.
    hand_by_k = {int(r["k"]): r["cmd"] for r in rollout.hand_targets}
    for f, row in zip(frames, rows):
        cmd = hand_by_k.get(int(body["k"][row]))
        if cmd is not None:
            out.jaws[f] = [float(cmd[0][0]), float(cmd[1][0])]

    return out


def _wbik_overlay(rollout):
    """Every re-solved overlay this recording carries, or None.

    `overlays.load` reads the IK bench's sidecar (`ik_bench.npz`) into one
    dict-like, so every method is posed through the same loop below.

    Deliberately tolerant: a missing file, a stale version and an unreadable one
    all mean the same thing to the viewer — this recording has not been solved,
    show fewer switches. Raising here would make a recording that opens fine
    today stop opening because of a cache.
    """
    try:
        from g1d_deploy.dashboards.inspect.overlays import load as _load
    except Exception:                                     # noqa: BLE001
        return None
    return _load(rollout.path)


def build(rollout, timeline) -> Scene:
    """Pose the model at every display frame."""
    import mujoco

    model, data = build_model(getattr(rollout.model, "urdf", None))
    jmap = _JointMap(model, rollout.model)

    # The URDF carries no `class="visual"` convention, so visual geometry is
    # identified the way URDF actually marks it: collision geoms have contact
    # enabled, visual ones do not.
    geom_ids = [g for g in range(model.ngeom)
                if int(model.geom_contype[g]) == 0
                and int(model.geom_type[g]) == mujoco.mjtGeom.mjGEOM_MESH]
    geoms, vertices, normals, indices = _geometry(model, geom_ids)
    gid = np.asarray(geom_ids, dtype=np.intp)

    # Per FRAME, not per record: `timeline` has already interpolated the ~1040 Hz
    # state onto the exact frame instants, so the measured pose is stated at the
    # same instant as the commanded one it is about to be compared against. See
    # `timeline`'s docstring for why state is interpolated where the targets and
    # the cameras are held.
    q_all = timeline.q
    grip = timeline.grip
    span = max(1e-9, rollout.model.hand_max - rollout.model.hand_min)
    tcp_local = tcp_point(rollout.model)

    n = timeline.n
    transforms = np.zeros((n, len(gid), 12), dtype=np.float32)
    tcp = np.zeros((n, 2, 3), dtype=np.float32)

    # --- the commanded overlay ---------------------------------------------
    # A second MjData, posed with the arms at zero and only the jaws moving. It
    # is never rendered directly; it exists to read each gripper geom's pose
    # RELATIVE to its wrist, which is all the ghost needs — the Dex-1 is rigid
    # to the flange apart from the two jaw slides, so that relative pose plus a
    # commanded flange places the whole gripper.
    ghost_data = mujoco.MjData(model)
    ghost_ids = [[], []]
    for slot, g in enumerate(gid):
        body = model.body(int(model.geom_bodyid[g])).name or ""
        mesh = model.mesh(int(model.geom_dataid[g])).name or ""
        if "Dex1" not in mesh and "dex1" not in body:
            continue
        ghost_ids[0 if body.startswith("left") else 1].append(slot)
    ghost_slots = ghost_ids[0] + ghost_ids[1]
    hand_of = [0] * len(ghost_ids[0]) + [1] * len(ghost_ids[1])

    cmd = _commands(rollout, timeline)
    cmd_poses, cmd_jaws, cmd_valid = cmd.poses, cmd.jaws, cmd.valid
    ghost = np.zeros((n, len(ghost_slots), 12), dtype=np.float32)
    ghost_tcp = np.zeros((n, 2, 3), dtype=np.float32)
    T_wp = world_from_base(rollout, model, data, jmap)

    # --- the solved overlay -------------------------------------------------
    # A third MjData, posed at the joint angles P6 published. Unlike the ghost
    # this is a real forward kinematic pose of the whole robot, so it is drawn
    # from the same transforms the measured arm is — a subset of the same geom
    # list, over the parts a body target can actually move. The wheels and the
    # lift column are left out because nothing commands them: a translucent copy
    # of a wheel sitting exactly on the wheel is not a comparison, it is haze.
    solved_data = mujoco.MjData(model)
    solved_slots = [slot for slot in range(len(gid))
                    if geoms[slot]["part"] in ("body", "gripper")]
    solved = np.zeros((n, len(solved_slots), 12), dtype=np.float32)
    solved_tcp = np.zeros((n, 2, 3), dtype=np.float32)
    solved_gid = gid[np.asarray(solved_slots, dtype=np.intp)] if solved_slots \
        else np.zeros(0, dtype=np.intp)

    # ...and the same overlay again on the trunk the IK was pinned at, when the
    # record says what that was. One extra `mj_forward` per commanded frame and
    # one more transform array; the alternative was rotating the first array in
    # the browser about the waist, which is exact only for as long as nobody adds
    # a geom proximal to it and is a second definition of the robot's kinematics
    # written in JavaScript.
    pinned_on = bool(cmd.has_pinned_waist)
    shape = (n, len(solved_slots), 12) if pinned_on else (0,)
    solved_pinned = np.zeros(shape, dtype=np.float32)
    solved_pinned_tcp = np.zeros((n, 2, 3) if pinned_on else (0,),
                                 dtype=np.float32)

    def pose_solved(f, waist_q) -> None:
        """The published joint angles into `solved_data`, on the given trunk.

        Everything except the three waist numbers is the same in both variants —
        same arm angles, same commanded jaw — so it is written once here rather
        than twice in the loop, where the two copies would drift the first time
        the jaw mapping changed and produce two overlays that disagree about the
        gripper for a reason having nothing to do with the waist."""
        solved_data.qpos[:] = 0.0
        solved_data.qpos[jmap.arm_adr] = cmd.q[f, :schema.ARM_DOF]
        solved_data.qpos[jmap.waist_adr] = waist_q
        for side in _GRIP_COLUMN:
            frac = (float(cmd_jaws[f, 0 if side == "left" else 1])
                    - rollout.model.hand_min) / span
            solved_data.qpos[jmap.jaw_adr[side]] = (
                g1d.JAW_CLOSED
                + min(1.0, max(0.0, frac)) * (g1d.JAW_OPEN - g1d.JAW_CLOSED))
        mujoco.mj_forward(model, solved_data)

    def read_solved(f, xforms, tcps) -> None:
        """Whatever `solved_data` currently holds, into row `f` of a pair."""
        xforms[f, :, :3] = solved_data.geom_xpos[solved_gid]
        xforms[f, :, 3:] = solved_data.geom_xmat[solved_gid]
        for h, bid in enumerate(jmap.wrist_body):
            R = solved_data.xmat[bid].reshape(3, 3)
            tcps[f, h] = solved_data.xpos[bid] + R @ tcp_local

    # --- the whole-body overlays -------------------------------------------
    # One more MjData, posed from the IK bench's sidecar. Absent sidecar means
    # empty dicts and a viewer that greys those switches: a recording nobody has
    # solved offers fewer overlays, never a wrong one.
    #
    # Drawn over EVERY geom, unlike `solved`, because two of the three tiers
    # move the lift column and one moves the base — parts a body target cannot
    # touch and the solved overlay therefore rightly omits.
    #
    # **The base is applied as a rigid pre-multiplication, not as a joint.** The
    # sidecar stores the base as `(x, y, yaw)` in the snapshot frame, and the
    # base joints sit at the ROOT of the kinematic chain, so translating and
    # rotating every geom transform by that pose is exactly what posing them
    # would produce. Doing it here rather than compiling `g1d`'s mobile-base
    # model keeps one model, one geom list and one set of meshes in play — a
    # second model would renumber every geom and break the slot indices the
    # browser reuses.
    ov = _wbik_overlay(rollout)
    wbik, wbik_tcp, wbik_err = {}, {}, {}
    wbik_valid = np.zeros(n, dtype=np.uint8)
    wbik_off_err = np.zeros(n, dtype=np.float32)
    wbik_row = np.full(n, -1, dtype=np.int64)
    wbik_slots = list(range(len(gid)))
    if ov is not None:
        # frame -> sidecar row, through the body-target row `_commands` resolved
        # and the `k` it carries. Two indexings meet here and nowhere else.
        k_to_ov = {int(kk): i for i, kk in enumerate(ov["k"])}
        ks = rollout.body_targets["k"]
        for f in range(n):
            r = int(cmd.row[f])
            if r >= 0:
                wbik_row[f] = k_to_ov.get(int(ks[r]), -1)
        wbik_valid[:] = (wbik_row >= 0).astype(np.uint8)
        off_err = ov["off/tcp_err"]
        wbik_off_err[wbik_row >= 0] = off_err[wbik_row[wbik_row >= 0]]
        for tier in (str(t) for t in ov["tiers"]):
            wbik[tier] = np.zeros((n, len(wbik_slots), 12), dtype=np.float32)
            wbik_tcp[tier] = np.zeros((n, 2, 3), dtype=np.float32)
            e = np.zeros(n, dtype=np.float32)
            e[wbik_row >= 0] = ov[f"{tier}/tcp_err"][wbik_row[wbik_row >= 0]]
            wbik_err[tier] = e
        wb_data = mujoco.MjData(model)

    def pose_wbik(f, tier, row) -> None:
        """The sidecar's configuration for this tick, into `wb_data`.

        Arm, waist and column go in as joint angles; the base comes back as a
        4x4 for the caller to pre-multiply, because it is not a joint in this
        model. The jaws follow the COMMANDED aperture, exactly as the solved
        overlay's do — the whole-body solver does not touch the gripper and a
        different jaw here would read as a difference it did not cause.
        """
        wb_data.qpos[:] = 0.0
        wb_data.qpos[jmap.arm_adr] = ov[f"{tier}/arm"][row]
        wb_data.qpos[jmap.waist_adr] = ov[f"{tier}/waist"][row][jmap.waist_pin_cols]
        # Both telescoping stages carry half the commanded height: one command
        # drives them together (`rt/cmd_hispeed`), which is how `wbik` solves
        # them and how the machine moves them.
        half = float(ov[f"{tier}/column"][row]) * 0.5
        for adr in jmap.column_adr:
            wb_data.qpos[adr] = half
        for side, col in _GRIP_COLUMN.items():
            frac = (float(cmd_jaws[f, 0 if side == "left" else 1])
                    - rollout.model.hand_min) / span
            wb_data.qpos[jmap.jaw_adr[side]] = (
                g1d.JAW_CLOSED
                + min(1.0, max(0.0, frac)) * (g1d.JAW_OPEN - g1d.JAW_CLOSED))
        mujoco.mj_forward(model, wb_data)
        bx, by, byaw = (float(v) for v in ov[f"{tier}/base"][row])
        c, sn = np.cos(byaw), np.sin(byaw)
        T = np.eye(4)
        T[:3, :3] = [[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]]
        T[:3, 3] = (bx, by, 0.0)
        return T

    for f in range(n):
        q = q_all[f]
        data.qpos[:] = 0.0
        data.qpos[jmap.arm_adr] = q[jmap.arm_cols]
        data.qpos[jmap.waist_adr] = q[jmap.waist_cols]

        g = grip[f]
        for side, col in _GRIP_COLUMN.items():
            # Uncalibrated linear map, gear radians -> jaw metres. Indicative,
            # not measured; see the dex1 PROVENANCE.md.
            frac = (float(g[col]) - rollout.model.hand_min) / span
            data.qpos[jmap.jaw_adr[side]] = (
                g1d.JAW_CLOSED + min(1.0, max(0.0, frac)) * (g1d.JAW_OPEN - g1d.JAW_CLOSED))

        mujoco.mj_forward(model, data)
        transforms[f, :, :3] = data.geom_xpos[gid]
        transforms[f, :, 3:] = data.geom_xmat[gid]

        # The Dex-1's virtual TCP:
        # the wrist-yaw frame's origin plus a rigid offset along its own +X.
        # Taken from the body rather than a site because the URDF has none, and
        # the two are the same point — `ee_site` was verified to sit at zero
        # offset and identity rotation from `wrist_yaw_link`.
        for h, bid in enumerate(jmap.wrist_body):
            R = data.xmat[bid].reshape(3, 3)
            tcp[f, h] = data.xpos[bid] + R @ tcp_local

        # The SOLVED configuration: what the IK produced, posed as a whole
        # robot. This is gated on `q_valid` and not on `cmd_valid`, which is the
        # entire point of separating the two — P0's homing and exit ramps
        # command joints with no Cartesian target behind them, and during those
        # the arm is being driven by something the ghost cannot show at all.
        if cmd.q_valid[f]:
            pose_solved(f, cmd.q[f, jmap.waist_slots])
            read_solved(f, solved, solved_tcp)
            if pinned_on:
                # The same tick again, differing in the three waist numbers and
                # nothing else. On a P0 ramp `waist_ik` IS the commanded waist
                # (`_commands` resolves the NaN), so these two rows are then
                # identical by construction rather than by a `k < 1` test here.
                pose_solved(f, cmd.waist_ik[f, jmap.waist_pin_cols])
                read_solved(f, solved_pinned, solved_pinned_tcp)

        if wbik_row[f] >= 0:
            row = int(wbik_row[f])
            for tier, arr in wbik.items():
                T_b = pose_wbik(f, tier, row)
                R_b, p_b = T_b[:3, :3], T_b[:3, 3]
                arr[f, :, :3] = wb_data.geom_xpos[gid] @ R_b.T + p_b
                # Each geom's 3x3, pre-rotated by the base. Written as a single
                # einsum rather than a loop: this is n_frames x n_geoms x 3
                # matrix products per tier and the loop version dominated the
                # build.
                R_g = wb_data.geom_xmat[gid].reshape(-1, 3, 3)
                arr[f, :, 3:] = np.einsum("ij,njk->nik", R_b, R_g).reshape(
                    len(gid), 9)
                for h, bid in enumerate(jmap.wrist_body):
                    R = wb_data.xmat[bid].reshape(3, 3)
                    local = wb_data.xpos[bid] + R @ tcp_local
                    wbik_tcp[tier][f, h] = R_b @ local + p_b

        if not cmd_valid[f]:
            continue

        # Where the policy asked the gripper to be. `chunk.pose` is the FLANGE
        # in base coordinates (the policy contract's frame), so one transform
        # puts it in the world.
        T_cmd = [T_wp @ _pose7_to_mat(cmd_poses[f, h]) for h in (0, 1)]
        for h in (0, 1):
            ghost_tcp[f, h] = T_cmd[h][:3, 3] + T_cmd[h][:3, :3] @ tcp_local

        # Pose the jaws at the COMMANDED aperture, then read each gripper geom
        # relative to its own wrist and re-anchor it on the commanded flange.
        # The ghost's virtual TCP lands on the commanded TCP by construction —
        # same rigid 0.184759 offset, same geometry — rather than by a separate
        # alignment step that could drift from it.
        ghost_data.qpos[:] = 0.0
        for side, col in _GRIP_COLUMN.items():
            frac = (float(cmd_jaws[f, 0 if side == "left" else 1])
                    - rollout.model.hand_min) / span
            ghost_data.qpos[jmap.jaw_adr[side]] = (
                g1d.JAW_CLOSED + min(1.0, max(0.0, frac)) * (g1d.JAW_OPEN - g1d.JAW_CLOSED))
        mujoco.mj_forward(model, ghost_data)

        for j, slot in enumerate(ghost_slots):
            h = hand_of[j]
            wid = jmap.wrist_body[h]
            R_w = ghost_data.xmat[wid].reshape(3, 3)
            g = int(gid[slot])
            # geom pose expressed in the wrist frame, then into the world at the
            # commanded flange.
            rel_R = R_w.T @ ghost_data.geom_xmat[g].reshape(3, 3)
            rel_p = R_w.T @ (ghost_data.geom_xpos[g] - ghost_data.xpos[wid])
            ghost[f, j, :3] = T_cmd[h][:3, 3] + T_cmd[h][:3, :3] @ rel_p
            ghost[f, j, 3:] = (T_cmd[h][:3, :3] @ rel_R).reshape(-1)

    return Scene(transforms, geoms, vertices, normals, indices, tcp,
                 ghost=ghost, ghost_geoms=[int(s) for s in ghost_slots],
                 ghost_tcp=ghost_tcp, ghost_valid=cmd_valid,
                 solved=solved, solved_geoms=[int(s) for s in solved_slots],
                 solved_tcp=solved_tcp, solved_valid=cmd.q_valid,
                 solved_pinned=solved_pinned,
                 solved_pinned_tcp=solved_pinned_tcp,
                 wbik=wbik, wbik_tcp=wbik_tcp, wbik_err=wbik_err,
                 wbik_geoms=[int(x) for x in wbik_slots] if wbik else [],
                 wbik_valid=wbik_valid, wbik_off_err=wbik_off_err,
                 wbik_meta=(ov.meta if ov is not None else {}),
                 action_mode=rollout.action_mode,
                 intervention_frames=int(cmd.intervened.sum()),
                 has_tcp=getattr(rollout.model, "tcp", 0) is not None)
