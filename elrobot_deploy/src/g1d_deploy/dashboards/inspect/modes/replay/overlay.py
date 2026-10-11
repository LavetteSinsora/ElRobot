"""One replay, posed onto the display frames the viewer already has.

The viewer draws an amber "commanded pose" ghost: the Dex-1 gripper pair placed
at the flange pose P6 handed the IK, per display frame. A replay is the same
picture computed from a different chunk stream, so this builds exactly that —
plus the whole chunk as a trajectory, which is the part the live viewer has no
reason to show and this tool exists for.

**Joint chunks are drawn through forward kinematics** (`ChunkPoses`): a
joint-mode policy's rows carry arm angles, and the gripper they put somewhere is
the recorded robot's FK of them, on the waist the policy commanded or else the
trunk measured at the chunk's anchor — the frame P6 would have posed them in.
One code path then serves both halves of the action contract, for the replay
and for the recording's own chunks alike.

    frame f  ->  the body target in force   (timeline.index["command"])
             ->  its tick k                 (body_targets["k"])
             ->  the chunk covering k       (newest whose k_first_action <= k)
             ->  row j = k - k_first_action

That chain is exact rather than approximate: a replay copies `chunk_id` and
`k_first_action` from the live replan it stands in for, so row `j` of a replay
and row `j` of the live chunk are commands for the same tick, and subtracting
them needs no interpolation.

**Rows past the end of a chunk are held, not dropped.** A chunk covers `P` ticks
and the next replan lands a few ticks later, so the last ticks of every period
run on a chunk that has run dry — which is exactly what P7 does with them (it
holds the last action). Holding here draws what the robot would have done;
blanking would make the overlay blink off for a reason that is about
bookkeeping rather than about the policy.

The geometry is `scene.py`'s, imported rather than restated: same model, same
geom list, same ghost slots, same jaw mapping, same virtual TCP. The slot
indices are the *loaded Scene's* `ghost_geoms`, so the browser indexes the
meshes it already uploaded.

Used by the replay mode (`__init__.py`) per Load and per finished replay.
"""

from __future__ import annotations

import numpy as np

from g1d_deploy.robot import g1d_model as g1d
from g1d_deploy.dashboards.inspect import scene as _scene

#: What `_pack` wants: `(name, array)` pairs, in a fixed order.
Parts = list[tuple[str, np.ndarray]]

#: `chunk.mode` for a joint chunk (`schema.chunk_dtype`), copied: a record's
#: integers mean what they meant when it was written.
_JOINT = 1


def rows_present(chunk) -> int:
    """How many rows of this chunk carry an action.

    `n_plan` says it when the record has the field (P5 publishes that many
    resident rows; a replay stores every row the policy returned). A record
    without it is asked of the data: the live recorder left the unused tail as
    zeros, and a zero quaternion is not a pose.
    """
    names = chunk.dtype.names or ()
    if "n_plan" in names and int(chunk["n_plan"]) > 0:
        return min(int(chunk["n_plan"]), int(np.shape(chunk["hand"])[0]))
    if "mode" in names and int(chunk["mode"]) == _JOINT:
        q = np.asarray(chunk["arm_q"])
        return int(np.count_nonzero(np.isfinite(q).all(axis=(1, 2))))
    quat = np.asarray(chunk["pose"])[:, :, 3:]
    ok = (np.isfinite(quat).all(axis=(1, 2))
          & np.any(np.abs(quat) > 1e-12, axis=(1, 2)))
    return int(np.count_nonzero(ok))


class ChunkPoses:
    """Flange poses `(n, 2, 7)`, base frame, for any chunk row: an eef row's
    own `pose`, a joint row's FK on the recorded robot.

    The waist a joint row is posed on is the chunk's own `waist` — the
    policy's `waist_q`, required in joint mode. A record whose joint chunk
    carries no finite waist falls back to the trunk MEASURED at its anchor.
    """

    def __init__(self, facts, state):
        self.facts = facts
        self.state = state
        self._fk = None
        self._cache: dict[tuple, np.ndarray] = {}

    def _forward(self):
        if self._fk is None:
            from g1d_deploy.robot.kinematics import ForwardKinematics
            self._fk = ForwardKinematics(getattr(self.facts, "urdf", None))
        return self._fk

    def _waist_at(self, t: float) -> np.ndarray:
        t_state = np.asarray(self.state["t_recv"], np.float64)
        i = int(np.argmin(np.abs(t_state - t)))
        return np.asarray(self.state["q"][i], np.float64)[
            np.asarray(self.facts.joints_waist, dtype=np.intp)]

    def of(self, chunk, n: int | None = None) -> np.ndarray | None:
        """The first `n` rows' flange poses (every present row by default), or
        None when the chunk carries nothing drawable."""
        rows = rows_present(chunk) if n is None else int(n)
        names = chunk.dtype.names or ()
        if rows <= 0:
            return None
        if "mode" not in names or int(chunk["mode"]) != _JOINT:
            pose = np.asarray(chunk["pose"][:rows], np.float64)
            return pose if np.isfinite(pose).all() else None
        arm = np.asarray(chunk["arm_q"][:rows], np.float64).reshape(rows, -1)
        if not np.isfinite(arm).all():
            return None
        key = (int(chunk["chunk_id"]), float(chunk["t_anchor"]), arm.tobytes())
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        own = bool(np.isfinite(np.asarray(chunk["waist"][:rows])).all())
        meas = self._waist_at(float(chunk["t_anchor"]))
        fk = self._forward()
        out = np.zeros((rows, 2, 7))
        for j in range(rows):
            waist = np.asarray(chunk["waist"][j], np.float64) if own else meas
            out[j] = fk.flange(arm[j], waist)
        self._cache[key] = out
        return out


def flange_poses(chunk, n: int, facts, cache: dict, state=None):
    """`ChunkPoses.of` for a caller without a builder (the replay's summary);
    `cache` keeps the `ChunkPoses`, and its FK model, between calls."""
    cp = cache.get("poses")
    if cp is None:
        cp = cache["poses"] = ChunkPoses(facts, state if state is not None
                                         else cache.get("state"))
    return cp.of(chunk, n)


class OverlayBuilder:
    """Holds the mujoco model once; builds one overlay per replay.

    Built once per loaded rollout because `build_model()` compiles the URDF and
    `world_from_base` fits the base transform, and neither depends on which
    replay is being drawn. Toggling six replays on should not pay for that six
    times.
    """

    def __init__(self, rollout, timeline, scene):
        import mujoco

        self.rollout = rollout
        self.timeline = timeline
        self.scene = scene
        self.model, data = _scene.build_model(getattr(rollout.model, "urdf", None))
        self.jmap = _scene._JointMap(self.model, rollout.model)
        geom_ids = [g for g in range(self.model.ngeom)
                    if int(self.model.geom_contype[g]) == 0
                    and int(self.model.geom_type[g]) == mujoco.mjtGeom.mjGEOM_MESH]
        self.gid = np.asarray(geom_ids, dtype=np.intp)
        # The browser's slot numbering, taken from the Scene it already has.
        self.ghost_slots = list(scene.ghost_geoms)
        self.hand_of = [
            0 if (self.model.body(int(self.model.geom_bodyid[self.gid[s]])).name
                  or "").startswith("left") else 1
            for s in self.ghost_slots]
        self.T_wp = _scene.world_from_base(rollout, self.model, data, self.jmap)
        self.tcp_local = _scene.tcp_point(rollout.model)
        self.poses = ChunkPoses(rollout.model, rollout.state)
        self._ghost_data = mujoco.MjData(self.model)
        self.hand_min = rollout.model.hand_min
        self.span = max(1e-9, rollout.model.hand_max - rollout.model.hand_min)

        # frame -> commanded tick, -1 where nothing was in force.
        n = timeline.n
        self.tick = np.full(n, -1, dtype=np.int64)
        rows = timeline.index.get("command")
        if rows is not None and len(rollout.body_targets):
            ks = rollout.body_targets["k"]
            for f in range(n):
                r = int(rows[f])
                if r >= 0:
                    self.tick[f] = int(ks[r])

    # --- the mapping -------------------------------------------------------

    def _cover(self, chunks, n_rows) -> tuple[np.ndarray, np.ndarray]:
        """Per frame: which chunk is in force, and which row of it.

        "In force" is the NEWEST chunk whose first action is at or before this
        tick — the same rule P6's `blend="newest"` follows, and the reason a
        replay's overlay lines up with the live ghost tick for tick.
        """
        n = self.timeline.n
        chunk_of = np.full(n, -1, dtype=np.int32)
        row_of = np.full(n, -1, dtype=np.int32)
        if not len(chunks):
            return chunk_of, row_of
        first = chunks["k_first_action"].astype(np.int64)
        order = np.argsort(first, kind="stable")
        first_sorted = first[order]
        for f in range(n):
            k = int(self.tick[f])
            if k < 0:
                continue
            pos = int(np.searchsorted(first_sorted, k, side="right")) - 1
            if pos < 0:
                continue
            c = int(order[pos])
            if n_rows[c] <= 0:
                continue
            # Held past the end of the chunk — see the module docstring.
            chunk_of[f] = c
            row_of[f] = min(k - int(first[c]), n_rows[c] - 1)
        return chunk_of, row_of

    # --- the geometry ------------------------------------------------------

    def _pose_jaws(self, grip_pair) -> None:
        import mujoco

        self._ghost_data.qpos[:] = 0.0
        for side in _scene._GRIP_COLUMN:
            frac = (float(grip_pair[0 if side == "left" else 1])
                    - self.hand_min) / self.span
            self._ghost_data.qpos[self.jmap.jaw_adr[side]] = (
                g1d.JAW_CLOSED
                + min(1.0, max(0.0, frac)) * (g1d.JAW_OPEN - g1d.JAW_CLOSED))
        mujoco.mj_forward(self.model, self._ghost_data)

    def _place(self, T_cmd, out_row) -> None:
        """Each ghost geom, read relative to its wrist and re-anchored —
        `scene.build`'s own loop, restated for the same eight lines."""
        for j, slot in enumerate(self.ghost_slots):
            h = self.hand_of[j]
            wid = self.jmap.wrist_body[h]
            R_w = self._ghost_data.xmat[wid].reshape(3, 3)
            g = int(self.gid[slot])
            rel_R = R_w.T @ self._ghost_data.geom_xmat[g].reshape(3, 3)
            rel_p = R_w.T @ (self._ghost_data.geom_xpos[g]
                             - self._ghost_data.xpos[wid])
            out_row[j, :3] = T_cmd[h][:3, 3] + T_cmd[h][:3, :3] @ rel_p
            out_row[j, 3:] = (T_cmd[h][:3, :3] @ rel_R).reshape(-1)

    def _world_tcp(self, pose7) -> np.ndarray:
        """`(2, 7)` flange poses in the base frame -> `(2, 3)` world TCP."""
        out = np.zeros((2, 3), dtype=np.float64)
        for h in (0, 1):
            T = self.T_wp @ _scene._pose7_to_mat(pose7[h])
            out[h] = T[:3, 3] + T[:3, :3] @ self.tcp_local
        return out

    # --- the product -------------------------------------------------------

    def build(self, chunks: np.ndarray, live: np.ndarray | None = None) -> dict:
        """Everything the browser needs to draw one replay.

        Returns `{"parts": [(name, array)...], "meta": {...}}`; the server packs
        the parts and sends the spans, exactly as it does for `transforms.bin`.
        """
        n = self.timeline.n
        chunks = chunks[np.argsort(chunks["chunk_id"], kind="stable")]
        n_chunks = len(chunks)
        H = int(chunks["hand"].shape[1]) if n_chunks else 0
        poses = [self.poses.of(chunks[c]) for c in range(n_chunks)]
        n_rows = [0 if p is None else len(p) for p in poses]

        chunk_of, row_of = self._cover(chunks, n_rows)
        ghost = np.zeros((n, len(self.ghost_slots), 12), dtype=np.float32)
        ghost_tcp = np.zeros((n, 2, 3), dtype=np.float32)
        valid = np.zeros(n, dtype=np.uint8)

        arc = np.zeros((n_chunks, H, 2, 3), dtype=np.float32)
        arc_n = np.zeros(n_chunks, dtype=np.int32)
        arc_exec = np.zeros(n_chunks, dtype=np.int32)
        arc_anchor = np.zeros((n_chunks, 2, 3), dtype=np.float32)

        for c in range(n_chunks):
            rows = n_rows[c]
            arc_n[c] = rows
            arc_exec[c] = min(int(chunks[c]["n_actions"]), rows)
            for j in range(rows):
                arc[c, j] = self._world_tcp(poses[c][j])
            if rows:
                arc_anchor[c] = arc[c, 0]

        for f in range(n):
            c, j = int(chunk_of[f]), int(row_of[f])
            if c < 0 or j < 0:
                continue
            pose7 = poses[c][j]
            T_cmd = [self.T_wp @ _scene._pose7_to_mat(pose7[h]) for h in (0, 1)]
            self._pose_jaws(chunks[c]["hand"][j, :, 0])
            self._place(T_cmd, ghost[f])
            for h in (0, 1):
                ghost_tcp[f, h] = T_cmd[h][:3, 3] + T_cmd[h][:3, :3] @ self.tcp_local
            valid[f] = 1

        err_mm = np.zeros((n, 2), dtype=np.float32)
        if live is not None and len(live):
            err_mm = self._errors(chunks, chunk_of, row_of, ghost_tcp, valid, live)

        parts: Parts = [
            ("ghost", ghost), ("ghost_tcp", ghost_tcp), ("valid", valid),
            ("chunk_of", chunk_of), ("row_of", row_of),
            ("arc", arc), ("arc_n", arc_n), ("arc_exec", arc_exec),
            ("arc_anchor", arc_anchor), ("err_mm", err_mm),
        ]
        meta = {
            "n_frames": int(n), "n_chunks": n_chunks, "H": H,
            "ghost_geoms": [int(s) for s in self.ghost_slots],
            "frames_valid": int(valid.sum()),
            "err_mm_p50": _pct(err_mm[valid > 0], 50),
            "err_mm_p99": _pct(err_mm[valid > 0], 99),
        }
        return {"parts": parts, "meta": meta}

    def _errors(self, chunks, chunk_of, row_of, ghost_tcp, valid, live):
        """Per frame, the distance to what the LIVE chunk asked for, matched by
        `chunk_id` (a replay's rows carry the live chunk's id)."""
        n = self.timeline.n
        out = np.zeros((n, 2), dtype=np.float32)
        live_by_id = {int(r["chunk_id"]): r for r in live}
        cache: dict[tuple[int, int], np.ndarray] = {}
        for f in range(n):
            c, j = int(chunk_of[f]), int(row_of[f])
            if c < 0 or not valid[f]:
                continue
            ref = live_by_id.get(int(chunks[c]["chunk_id"]))
            if ref is None:
                continue
            ref_poses = self.poses.of(ref)
            if ref_poses is None:
                continue
            jj = min(j, len(ref_poses) - 1)
            key = (int(ref["chunk_id"]), jj)
            if key not in cache:
                cache[key] = self._world_tcp(ref_poses[jj])
            out[f] = np.linalg.norm(ghost_tcp[f] - cache[key], axis=-1) * 1e3
        return out


def action_diff(chunks, live, poses: ChunkPoses, t0: float) -> dict:
    """The timeline strip for one replay: per anchor, how far its actions are
    from the recorded ones.

    For each replayed chunk, matched to the live chunk by `chunk_id` (a replay
    copies it): the mean, over the rows the live chunk EXECUTED
    (`n_actions`, capped at what both carry), of the worst hand's flange
    translation distance, mm. Joint rows are compared through FK
    (`ChunkPoses`). `t` is the anchor (`t_anchor`) in seconds from the
    display window's start `t0`.
    """
    live_by_id = {int(r["chunk_id"]): r for r in live}
    rows = []
    for ch in chunks[np.argsort(chunks["chunk_id"], kind="stable")]:
        ref = live_by_id.get(int(ch["chunk_id"]))
        if ref is None:
            continue
        a, b = poses.of(ch), poses.of(ref)
        if a is None or b is None:
            continue
        n = min(len(a), len(b), max(1, int(ref["n_actions"])))
        d = np.linalg.norm(a[:n, :, :3] - b[:n, :, :3], axis=-1).max(axis=1)
        rows.append((float(ref["t_anchor"]) - t0, float(d.mean()) * 1e3,
                     int(ch["chunk_id"])))
    rows.sort()
    return {"t": [round(r[0], 4) for r in rows],
            "diff_mm": [round(r[1], 3) for r in rows],
            "chunk_id": [r[2] for r in rows], "unit": "mm",
            "metric": "mean over the executed rows of the worst hand's flange "
                      "translation difference, replay vs recorded"}


def _pct(values, q) -> float | None:
    values = np.asarray(values)
    if not values.size:
        return None
    return round(float(np.percentile(values, q)), 2)


__all__ = ["ChunkPoses", "OverlayBuilder", "action_diff", "flange_poses",
           "rows_present"]
