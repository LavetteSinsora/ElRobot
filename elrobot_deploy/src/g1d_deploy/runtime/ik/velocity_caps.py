"""Motion caps INSIDE the QP, anchored on the tick's seed. Used by every
shipped solver through `arm_planner.Guarded` (`docs/infra_logic/ik_solvers.md`).

P6 clips every published arm step to `hw.qdot_max_body` per joint AFTER the
solve (`p6_target` step 5). A per-joint clip of a large QP step is a
configuration the QP never evaluated — on one recorded rollout, at a chunk seam,
six clipped joints put the wrist 57 mm into the torso past a collision limit
that had passed the QP's own answer. So every solver here carries the same
velocity limit as a QP LIMIT: the clamp then never binds, and what is
published is what was checked. Two caps, both measured on a recorded
rollout:

* `speed`       per joint, |q − q_seed| ≤ ω·dt over the whole tick — every
                iterate of every stage of a pull-polish shares the one anchor.
* `self_speed`  per arm, |n·(q − q_seed)| ≤ ω_self·dt where n is the arm's
                1-D Jacobian null space (the elbow swivel) at the current
                iterate: the motion the hand targets did not ask for. A reach
                is untouched; a spin is capped. Without it `arm_qp` commands
                shoulder-yaw or elbow speeds above 3 rad/s on 116 recorded
                ticks, and the branch planner on 176.

`TickAnchor` is what ties the cap to the tick. `Guarded` sets it from the seed
around the per-tick solve; plan-time solves (branch families, the trunk's
passes) run inside `suspended()` and are never capped, because they are never
published. A deliberate branch change (`transiting`) lifts the swivel cap — it
would forbid exactly the motion being asked for — and paces the joints at its
own speed instead.
"""

from __future__ import annotations

import contextlib

import numpy as np


class TickAnchor:
    """The tick's seed, and whether the caps are live. One per built solver."""

    def __init__(self, dt: float = 1 / 30.0):
        self.q: np.ndarray | None = None
        self.active = False
        self.dt = float(dt)
        self.transit = False
        self.transit_speed: float | None = None

    @contextlib.contextmanager
    def ticking(self, seed_q14):
        """The per-tick solve: caps on, anchored at `seed_q14`."""
        self.q = np.asarray(seed_q14, np.float64).copy()
        self.active = True
        try:
            yield
        finally:
            self.active = False

    @contextlib.contextmanager
    def suspended(self):
        """A plan-time solve inside a tick: no caps."""
        was = self.active
        self.active = False
        try:
            yield
        finally:
            self.active = was

    @contextlib.contextmanager
    def transiting(self, speed: float | None):
        """A branch transit: the swivel cap lifted, the joints paced at `speed`."""
        self.transit, self.transit_speed = True, speed
        try:
            yield
        finally:
            self.transit, self.transit_speed = False, None


class MotionLimit:
    """A mink limit: per-joint speed box and/or per-arm self-motion band."""

    def __init__(self, backend, anchor: TickAnchor, *, speed=None, self_speed=None):
        from g1d_deploy.robot import g1d_model as g1d

        m = backend.model
        self.anchor = anchor
        names = [n for s in ("left", "right") for n in g1d.ARM_JOINTS[s]]
        self.qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in names])
        self.dadr = np.array([m.jnt_dofadr[m.joint(n).id] for n in names])
        self.nv = m.nv
        self.speed = (None if speed is None
                      else np.broadcast_to(np.asarray(speed, float), (14,)).copy())
        self.self_speed = self_speed
        self.bid = [m.body(f"{s}_wrist_yaw_link").id for s in ("left", "right")]
        self.jp, self.jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))

    def compute_qp_inequalities(self, configuration, dt):
        import mujoco
        from mink.limits.limit import Constraint

        a = self.anchor
        if not a.active or a.q is None:
            return Constraint()
        q = configuration.q[self.qadr]
        dq = q - a.q
        T = a.dt
        G, h = [], []
        speed = self.speed
        if a.transit and a.transit_speed is not None:
            cap = np.full(14, float(a.transit_speed))
            speed = cap if speed is None else np.minimum(speed, cap)
        if speed is not None:
            E = np.zeros((14, self.nv))
            E[np.arange(14), self.dadr] = 1.0
            room_hi = speed * T - dq
            room_lo = speed * T + dq
            G += [E, -E]
            # mink's QP variable is the displacement Δq (v = Δq/dt), so bounds
            # are displacements too — no /dt.
            h += [np.maximum(room_hi, 0.0), np.maximum(room_lo, 0.0)]
        if self.self_speed is not None and not a.transit:
            m, d = configuration.model, configuration.data
            for side, bid in enumerate(self.bid):
                mujoco.mj_jacBody(m, d, self.jp, self.jr, bid)
                dofs = self.dadr[7 * side:7 * side + 7]
                J = np.vstack([self.jp[:, dofs], self.jr[:, dofs]])
                n = np.linalg.svd(J)[2][-1]
                row = np.zeros(self.nv)
                row[dofs] = n
                s = float(n @ dq[7 * side:7 * side + 7])
                G += [row[None], -row[None]]
                h += [np.array([max(self.self_speed * T - s, 0.0)]),
                      np.array([max(self.self_speed * T + s, 0.0)])]
        if not G:
            return Constraint()
        return Constraint(G=np.vstack(G), h=np.concatenate(h))


__all__ = ["MotionLimit", "TickAnchor"]
