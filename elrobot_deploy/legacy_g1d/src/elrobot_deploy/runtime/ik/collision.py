"""A collision limit that cannot be tunnelled through by a phantom distance.

Used by every shipped solver (`install`). Design:
`docs/infra_logic/ik_solvers.md`, "The four shipped solvers".

The failure it exists for, found on one recorded rollout (the branch planner,
and the per-tick QP the same way): the right wrist-pitch link sat 1.5 mm off the torso,
inside the 5 mm margin, so mink's `CollisionAvoidanceLimit` forbade any
approach. On the next iterate `mj_geomDistance` returned exactly 0.0 with
witness points 330 mm away from the wrist (outside its bounding sphere), the
constraint row took its normal from those points and pointed nowhere useful,
and the QP moved the wrist 14 mm INTO the torso in one iterate for a better
hand fit. From there the limit only holds (`bound_relaxation` = 0 inside the
margin), so it stayed in.

Phantom zeros are ~1.3% of near-pair queries on the G1-D meshes, and no
`opt.ccd_iterations` / `ccd_tolerance` setting removes them (measured: 35 to
2000 iterations, 59 zeros unchanged; tighter tolerance makes more).

`RobustCollisionLimit` changes three things, each switchable:

* **phantom guard** — a distance of exactly 0.0, or witness points further from
  their geoms than the geoms' bounding radii, is not believed. The pair's last
  valid witness points (stored in each body's frame) are carried forward to
  the current configuration and supply both the distance and the normal. With
  no history the pair's centre line is used with a zero bound (hold).
* **push-out** (`push` m/s, 0 = mink's behaviour) — inside the margin the
  separation-rate bound is `max(gain·(d − d_min)/dt, −push)` instead of 0, so a
  configuration that starts inside recovers instead of being held.
* **step guard** (`install(..., step_guard=True)`) — after every iterate, the
  near pairs are re-measured; an iterate that took a pair deeper than
  `min(previous, d_min) − tol` is rolled back, and the solve loop's own
  "iterate moved nothing" exit ends the tick. Belt and braces for any residual
  linearisation tunnel.

`install(solver.ik, **GUARD)` swaps the limit into a built `DualArmIK` (or the
whole-body solver's), the way every solver in this package is built.
"""

from __future__ import annotations

import mujoco
import numpy as np

#: What every solver here installs: bounds in displacement units (the QP
#: variable is Δq; mink's /dt lets one iterate close 25x the remaining gap),
#: the phantom guard, the step guard, no push-out.
GUARD = {"push": 0.0, "guard": True, "step_guard": True, "units": True}


class RobustCollisionLimit:
    """Wraps a built `mink.CollisionAvoidanceLimit` (same pairs, same gains)."""

    def __init__(self, base, *, push=0.0, guard=True, units=False):
        import mink

        self._mink = mink
        self.base = base
        self.model = base.model
        self.push = float(push)
        self.guard = bool(guard)
        #: Bound the approach in DISPLACEMENT units, gain·(d − d_min), as the
        #: QP variable is Δq. mink divides by dt (a velocity), which at the
        #: solver's dt = 1/30 s lets one iterate close 25× the remaining gap.
        self.units = bool(units)
        self._cache = {}      # pair idx -> (l1, l2, sign) witness points in body frames
        self.phantoms = 0
        self.rollbacks = 0
        #: pair idx -> (l1, l2, s): contact points in body frames and the world
        #: separation direction, taken from a penetration the step guard saw.
        #: Cleared whenever a new solve places a seed.
        self.cuts = {}
        m = base.model
        self._rb = m.geom_rbound

    # mink reads these off limits
    def __getattr__(self, name):
        return getattr(self.base, name)

    def _body_frame(self, data, gid):
        b = self.model.geom_bodyid[gid]
        return data.xpos[b], data.xmat[b].reshape(3, 3)

    # --- one distance sweep per configuration, shared by the constraint build
    # and the step guard (sweeping separately would be 2× the cost).

    def _pairs_arrays(self):
        if not hasattr(self, "_g1s"):
            pairs = np.asarray(self.base.geom_id_pairs, dtype=int)
            self._g1s, self._g2s = pairs[:, 0], pairs[:, 1]
            self._b1s = self.model.geom_bodyid[self._g1s]
            self._b2s = self.model.geom_bodyid[self._g2s]
            n = len(pairs)
            self._c_l1 = np.zeros((n, 3))
            self._c_l2 = np.zeros((n, 3))
            self._c_sign = np.ones(n)
            self._c_has = np.zeros(n, bool)
            self._sweep_key = None

    def sweep(self, data):
        """-> (idx, dist, fromto (k,6), valid) for the broadphase survivors at
        `data`'s configuration; cached on the configuration."""
        self._pairs_arrays()
        key = data.qpos.tobytes()
        if self._sweep_key == key:
            return self._sweep
        base = self.base
        idx = np.asarray(list(self.survivors(data)), dtype=int)
        distmax = base.collision_detection_distance
        dist = np.empty(len(idx))
        fromto = np.zeros((len(idx), 6))
        ft = np.zeros(6)
        m = self.model
        g1s, g2s = self._g1s[idx], self._g2s[idx]
        for j in range(len(idx)):
            dist[j] = mujoco.mj_geomDistance(m, data, int(g1s[j]), int(g2s[j]), distmax, ft)
            fromto[j] = ft
        near = np.abs(dist - distmax) >= 1e-12
        valid = np.ones(len(idx), bool)
        if self.guard and near.any():
            x1, x2 = data.geom_xpos[g1s], data.geom_xpos[g2s]
            d1 = np.linalg.norm(fromto[:, :3] - x1, axis=1)
            d2 = np.linalg.norm(fromto[:, 3:] - x2, axis=1)
            valid = (dist != 0.0) & (d1 <= self._rb[g1s] + 2e-3) & (d2 <= self._rb[g2s] + 2e-3)
            valid |= ~near
            b1, b2 = self._b1s[idx], self._b2s[idx]
            p1, p2 = data.xpos[b1], data.xpos[b2]
            R1, R2 = data.xmat[b1].reshape(-1, 3, 3), data.xmat[b2].reshape(-1, 3, 3)
            good = valid & near
            if good.any():
                gi = idx[good]
                self._c_l1[gi] = np.einsum("nji,nj->ni", R1[good], fromto[good, :3] - p1[good])
                self._c_l2[gi] = np.einsum("nji,nj->ni", R2[good], fromto[good, 3:] - p2[good])
                self._c_sign[gi] = np.where(dist[good] >= 0, 1.0, -1.0)
                self._c_has[gi] = True
            bad = ~valid
            if bad.any():
                self.phantoms += int(bad.sum())
                has = self._c_has[idx] & bad
                nohist = bad & ~has
                # no history: centre line, distance 0 (hold)
                fromto[nohist, :3] = x1[nohist]
                fromto[nohist, 3:] = x2[nohist]
                dist[nohist] = 0.0
                if has.any():
                    hi = idx[has]
                    a = p1[has] + np.einsum("nij,nj->ni", R1[has], self._c_l1[hi])
                    b = p2[has] + np.einsum("nij,nj->ni", R2[has], self._c_l2[hi])
                    fromto[has, :3], fromto[has, 3:] = a, b
                    dist[has] = self._c_sign[hi] * np.linalg.norm(b - a, axis=1)
                    valid[has] = True
        self._sweep_key = key
        self._sweep = (idx, dist, fromto, valid)
        return self._sweep

    def measure(self, data, idx, fromto):
        """-> (dist, valid) for pair `idx` (single-pair form of `sweep`)."""
        ids, dist, ft, valid = self.sweep(data)
        j = np.where(ids == idx)[0]
        if not len(j):
            return self.base.collision_detection_distance, True
        j = int(j[0])
        fromto[:] = ft[j]
        return float(dist[j]), bool(valid[j])

    def survivors(self, data):
        base = self.base
        if base.broadphase and base.max_num_contacts >= base.broadphase_min_pairs:
            return base._broadphase_survivors(data)
        return range(base.max_num_contacts)

    def near(self, data):
        """-> {idx: dist} for every pair inside the detection distance."""
        ids, dist, _, _ = self.sweep(data)
        dm = self.base.collision_detection_distance
        keep = np.abs(dist - dm) >= 1e-12
        return dict(zip(ids[keep].tolist(), dist[keep].tolist()))

    def compute_qp_inequalities(self, configuration, dt):
        from mink.limits.collision_avoidance_limit import (
            compute_contact_normal_jacobian,
        )
        from mink.limits.limit import Constraint

        base, model, data = self.base, self.model, configuration.data
        n = base.max_num_contacts
        h = np.full((n,), np.inf)
        G = np.zeros((n, model.nv))
        ft, nrm, j1, j2 = base._fromto, base._normal, base._jac1, base._jac2
        dmax = base.collision_detection_distance
        dmin = base.minimum_distance_from_collisions
        gain, relax = base.gain, base.bound_relaxation
        ids, dists, fts, valids = self.sweep(data)
        scale = 1.0 if self.units else 1.0 / dt
        for j in np.where(np.abs(dists - dmax) >= 1e-12)[0]:
            idx = int(ids[j])
            dist, valid = float(dists[j]), bool(valids[j])
            ft[:] = fts[j]
            if not np.any(ft[3:] - ft[:3]):
                continue
            g1, g2 = base.geom_id_pairs[idx]
            row = compute_contact_normal_jacobian(model, data, g1, g2, ft, nrm, j1, j2)
            if dist > dmin:
                h[idx] = gain * (dist - dmin) * scale + relax
            elif self.push > 0 and valid and dist < 0:
                h[idx] = max(gain * dist, -self.push * dt)
            else:
                h[idx] = relax
            G[idx] = (-1.0 if dist >= 0 else 1.0) * row
        if self.cuts:
            rows = []
            for idx, (l1, l2, sep) in self.cuts.items():
                g1, g2 = base.geom_id_pairs[idx]
                p1, R1 = self._body_frame(data, g1)
                p2, R2 = self._body_frame(data, g2)
                mujoco.mj_jac(model, data, j1, None, p1 + R1 @ l1, model.geom_bodyid[g1])
                mujoco.mj_jac(model, data, j2, None, p2 + R2 @ l2, model.geom_bodyid[g2])
                rows.append(-sep @ (j2 - j1))
            G = np.vstack([G, np.array(rows)])
            h = np.concatenate([h, np.zeros(len(rows))])
        return Constraint(G=G, h=h)

    def add_cut(self, data, idx):
        """Forbid pair `idx` from approaching along the direction of the
        penetration it is in at `data`'s configuration."""
        g1, g2 = self.base.geom_id_pairs[idx]
        ft = np.zeros(6)
        d = mujoco.mj_geomDistance(self.model, data, g1, g2,
                                   self.base.collision_detection_distance, ft)
        v = ft[3:] - ft[:3]
        nv = float(np.linalg.norm(v))
        if d == 0.0 or nv < 1e-9:
            v = data.geom_xpos[g2] - data.geom_xpos[g1]
            nv = float(np.linalg.norm(v))
            sep = v / nv
        else:
            sep = (v / nv) * (1.0 if d >= 0 else -1.0)
        p1, R1 = self._body_frame(data, g1)
        p2, R2 = self._body_frame(data, g2)
        self.cuts[idx] = (R1.T @ (ft[:3] - p1), R2.T @ (ft[3:] - p2), sep)


def install(ik, *, push=0.0, guard=True, step_guard=True, tol=1e-4, max_retries=2, units=False):
    """Swap `ik`'s collision limit for the robust one; optionally wrap
    `ik.solve_tick` with the step guard. Returns the limit (or None)."""
    import mink

    lim = None
    for i, L in enumerate(ik._limits):
        if isinstance(L, mink.CollisionAvoidanceLimit):
            lim = RobustCollisionLimit(L, push=push, guard=guard, units=units)
            ik._limits[i] = lim
    if lim is None or not step_guard:
        return lim
    orig = ik.solve_tick
    # A new seed is a new solve: forget last solve's cuts.
    cfg_update = ik.config.update

    def update(q=None):
        lim.cuts.clear()
        floor.clear()
        return cfg_update(q)

    ik.config.update = update

    last = [b"", {}]
    # per-solve depth floor: a pair may never end an iterate deeper than it was
    # at the seed (or than 0 if it started clear) — no creep across iterates.
    floor = {}

    def solve_tick(T_left, T_right, iters=5, dt=1 / 30.0):
        q = None
        for _ in range(iters):
            for _attempt in range(max_retries + 1):
                q_prev = ik.config.q.copy()
                key = q_prev.tobytes()
                before = last[1] if last[0] == key else lim.near(ik.config.data)
                q = orig(T_left, T_right, iters=1, dt=dt)
                after = lim.near(ik.config.data)
                for i, d in before.items():
                    floor.setdefault(i, d)
                bad = [i for i, d in after.items() if d < min(floor.get(i, np.inf), 0.0) - tol]
                if not bad:
                    last[0], last[1] = ik.config.q.tobytes(), after
                    break
                lim.rollbacks += 1
                for i in bad:
                    lim.add_cut(ik.config.data, i)
                cfg_update(q_prev)
                ik.backend.set_qpos(q_prev)
                q = q_prev.copy()
                last[0], last[1] = key, before
            else:
                break          # still tunnelling after the retries: hold
        return q

    ik.solve_tick = solve_tick
    return lim


__all__ = ["GUARD", "RobustCollisionLimit", "install"]
