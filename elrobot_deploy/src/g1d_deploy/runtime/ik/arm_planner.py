"""The streamed branch planner: `ik = "arm_planner"`, the default solver.

Design: `docs/infra_logic/ik_solvers.md`, "`arm_planner` — the streamed branch
planner". Tuning numbers below were measured on the reference robot's recorded rollouts (2026-09)
with the inspection dashboard's IK mode; re-measure before changing a
threshold.

The per-tick QP (`arm_qp.DualArmSolver`) answers one tick at a time from the
last tick's answer. That is exactly right until the policy asks for a pose the
current elbow BRANCH cannot reach: the arm folds onto a joint stop, the QP
reports a miss, and nothing in the loop ever swings the elbow to the branch
that can (measured: 19 of the 47 worst ticks of one set of rollouts were
reachable from another seed — a one-way trap: reachable, unreachable from
here).

So this solver looks at the CHUNK:

  1. **families** — per chunk, four candidate arms are chained through the
     chunk's knots (every `stride`-th pose and the last): the seed itself
     ("chain"), the joint-range midpoint ("mid"), and the seed swung ±0.6 rad
     along each arm's elbow self-motion ("sw+", "sw-"). One (knot, family)
     cell is solved per tick, so the plan streams in as the chunk plays.
  2. **a DP over the knots** picks a branch per knot: TCP miss, plus `beta`
     mm/rad of travel, plus `gamma` mm/rad of joint-limit margin under `m0`,
     plus a shoulder-roll winding hinge (`sr_gain` mm/rad inside the last
     `sr_rad` before a roll stop — the specific trap measured above), over switches
     the arm can make within a knot span at `vfrac` of the clamp.
  3. **per tick**, three pulled QP iterates toward the interpolated planned
     posture at `pull_cost`, then a polish with the plain solve from that
     answer (the pull removes the trap, the polish removes the pull's bias).
  4. **an escape** (`transit_*`): when the plan's best family beats the chain
     by `transit_mm` at an upcoming knot, REACHES there (within
     `transit_reach_mm`), and the executed arm already misses by
     `transit_stuck_mm`, the tick pulls hard toward that posture with no
     polish, paced at `transit_speed` rad/s, until within `transit_rad`.
     Ungated from the DP path (`transit_any`): the DP only admits switches the
     arm can make in one knot span, so a branch 3 rad away is never on its path
     and the escape would never fire (one recorded rollout sat on the L
     shoulder-roll limit for 17 chunks).
  5. **smoothness** — the seed-to-plan ramp spans a knot instead of snapping
     (`seam_ramp`), and the elbow self-motion is capped at 2.5 rad/s in the QP
     (`self_speed`): elbow spins above 3 rad/s 478 -> 69 ticks on the recordings.

Measured (2026-09-18, 32 recorded rollouts, seam-adjusted):
`arm_qp` 3.99 / 9 mm mean / p95 with 93 hits > 2 mm into the body; this
solver 1.94 / 7 with 11 (max 4.0 mm), joint p99 3.57 rad/s. Eleven variants of
its thresholds were all within 0.1 mm of it — the escape does the work.

Every QP in it carries the phantom-proof collision limit (`collision.py`) and
the tick's motion caps (`velocity_caps.py`); plan-time solves run uncapped, since
they are never published. `Guarded` is the object P6 holds.
"""

from __future__ import annotations

import time

import numpy as np

from g1d_deploy.robot import kinematics as kin
from g1d_deploy.runtime.ik.arm_qp import DualArmSolver
from g1d_deploy.robot.kinematics import SIDES
from g1d_deploy.runtime.ik import collision
from g1d_deploy.runtime.ik.velocity_caps import MotionLimit, TickAnchor

_X = np.array([1.0, 0.0, 0.0])


# --- pieces --------------------------------------------------------------------


class Residuals:
    """Residuals of a configuration against a tick's targets, on the deploy
    model. Owns its own backend so scoring never disturbs a solver's."""

    def __init__(self, tcp=None):
        """`tcp`: the robot file's (7,) TCP or None (`runtime/ik/__init__.py::tcp_point`)."""
        import mujoco

        from g1d_deploy.robot import g1d_model as g1d
        from g1d_deploy.runtime.ik import tcp_point

        self._mujoco = mujoco
        self.backend = g1d.G1DBackend()
        self.jmap = kin.joint_addresses(self.backend)
        self.base = self.backend.base_pose()
        self.tcp = tcp_point(tcp)           # (3,) in the flange frame

    def place(self, arm14, waist3):
        kin.place(self.backend, arm14, waist3, self.jmap)
        self._mujoco.mj_forward(self.backend.model, self.backend.data)

    def tcp_error(self, arm14, waist3, pose_2x7) -> float:
        """Worst virtual-TCP position error, metres, at this configuration.
        The TCP is the robot file's (`[end_effector] tcp`)."""
        self.place(arm14, waist3)
        worst = 0.0
        for h, side in enumerate(SIDES):
            T = self.base @ kin.pose_to_mat(np.asarray(pose_2x7[h], np.float64))
            cur = self.backend.flange_pose(side)
            a = cur[:3, 3] + cur[:3, :3] @ self.tcp
            b = T[:3, 3] + T[:3, :3] @ self.tcp
            worst = max(worst, float(np.linalg.norm(a - b)))
        return worst


def arm_ranges(solver: DualArmSolver) -> np.ndarray:
    """(14, 2) joint ranges in the arm's 14-vector order."""
    from g1d_deploy.robot import g1d_model as g1d

    m = solver.backend.model
    names = [n for s in SIDES for n in g1d.ARM_JOINTS[s]]
    return np.array([m.joint(n).range for n in names])


def posture_pull(solver: DualArmSolver, cost: float):
    """A `PostureTask` over the arm DOFs at `cost`, added to `solver`'s QP.
    Returns `(task, template)`: set a target with `template` holding the
    solver's full qpos and the 14 arm entries overwritten."""
    import mink

    m = solver.backend.model
    c = np.zeros(m.nv)
    c[solver.backend.all_arm_dofs] = cost
    task = mink.PostureTask(m, cost=c)
    solver.add_task(task)
    return task, solver.backend.data.qpos.copy()


class PullPolish:
    """Two solves per tick: `pulled` (a few iterates under a posture pull),
    then `polish` (the plain solve, pull absent) from that answer.

    The polish gets whatever the tick has LEFT, not a fixed half. With a fixed
    split the first quiet latency pass abandoned 1 tick in ~12,000 while the
    slowest whole tick took 23 ms of 33: the polish ran out of its half with
    the tick's budget unspent.
    """

    def __init__(self, pulled: DualArmSolver, polish: DualArmSolver, split: float = 0.5):
        self.pulled, self.polish, self.split = pulled, polish, split

    def solve(self, pose, waist, seed_q, budget=None):
        b = 1 / 30.0 if budget is None else budget
        t0 = time.perf_counter()
        q1, _, ok1 = self.pulled.solve(pose, waist, seed_q, budget=b * self.split)
        left = max(b - (time.perf_counter() - t0), 1e-4)
        return self.polish.solve(pose, waist, q1 if ok1 else seed_q, budget=left)


# --- the planner ---------------------------------------------------------------


class StreamPlannerSolver:
    """The chunk-horizon branch planner, ticked as P6 ticks (module docstring).

    P6's contract (`runtime/ik/__init__.py`): `plan_chunk(...)` when a chunk lands,
    `solve(...)` per tick. `plan_chunk` returns None — no trunk is commanded by
    an arm-mode solver. A waist planner (`waist_planner.py`) passes `waists`, the
    per-tick trunk it predicts the servo will be at, and `use_plan_waist` then
    poses each tick on that prediction rather than on the trunk P6 hands in.
    """

    mode = "arm"
    plans_chunks = True

    def __init__(self, cfg, anchor: TickAnchor, *, urdf=None,
                 families=("chain", "mid", "sw+", "sw-"), stride=8, beta=10.0, vfrac=0.8,
                 gamma=25.0, m0=0.3, swivel_rad=0.6, sr_gain=0.0, sr_rad=1.0,
                 pull_cost=1e-2, pull_iters=3, rel=0.05, plan_iters=10, seam_ramp=False,
                 transit_cost=None, transit_mm=10.0, transit_rad=0.15, transit_iters=8,
                 transit_speed=3.0, transit_stuck_mm=0.0, transit_reach_mm=15.0,
                 transit_any=False, use_plan_waist=False, tcp=None):
        self.anchor = anchor
        self.fams = tuple(families)
        self.stride, self.beta, self.vfrac = stride, beta, vfrac
        self.gamma, self.m0, self.swivel_rad = gamma, m0, swivel_rad
        self.sr_gain, self.sr_rad = float(sr_gain), float(sr_rad)
        self.SR = np.array([1, 8])           # L/R shoulder roll in the 14-vector
        self.transit_mm, self.transit_rad = transit_mm, transit_rad
        self.transit_speed, self.transit_stuck_mm = transit_speed, transit_stuck_mm
        self.transit_any, self.transit_reach_mm = transit_any, transit_reach_mm
        self.seam_ramp = seam_ramp
        self.use_plan_waist = use_plan_waist
        self.dt = float(cfg.dt)
        self.vmax = np.full(14, float(getattr(cfg, "qdot_max", 2 * np.pi))) * vfrac

        # Three QPs: the families' (plan-time, convergence-stopped), the
        # pulled one (a few iterates, no stop — the pull's bias defeats it),
        # and the polish (the deployed solve with the stop).
        self.family_solver = DualArmSolver(cfg, urdf=urdf, rel=0.05, max_iters=plan_iters, tcp=tcp)
        self.residuals = Residuals(tcp)
        rng = arm_ranges(self.family_solver)
        self.lo, self.hi = rng[:, 0], rng[:, 1]
        self.fixed = {"mid": rng.mean(axis=1)}

        pulled = DualArmSolver(cfg, urdf=urdf, max_iters=pull_iters, tcp=tcp)
        self.task, self.template = posture_pull(pulled, pull_cost)
        self.jmap_arm = pulled.jmap.arm
        self.pp = PullPolish(pulled, DualArmSolver(cfg, urdf=urdf, rel=rel, tcp=tcp))
        self.solvers = [self.family_solver, pulled, self.pp.polish]

        self.transit = None
        if transit_cost is not None:
            self.transit = DualArmSolver(cfg, urdf=urdf, rel=0.02, max_iters=transit_iters, tcp=tcp)
            self.ttask, self.ttemplate = posture_pull(self.transit, transit_cost)
            self.solvers.append(self.transit)
        self.st = None
        self.transits = 0
        self.err_last = 0.0
        self.solved_waist = None

    # --- chunk --------------------------------------------------------------

    def swivel(self, q, waist, sign):
        """`q` swung `sign * swivel_rad` along each arm's elbow self-motion."""
        import mujoco

        b = self.residuals.backend
        kin.place(b, q, waist, self.residuals.jmap)
        mujoco.mj_forward(b.model, b.data)
        out = np.asarray(q, np.float64).copy()
        jp, jr = np.zeros((3, b.model.nv)), np.zeros((3, b.model.nv))
        for h, side in enumerate(SIDES):
            mujoco.mj_jacBody(b.model, b.data, jp, jr, b.model.body(f"{side}_wrist_yaw_link").id)
            dofs = b.arm_dof_adr[side]
            v = np.linalg.svd(np.vstack([jp[:, dofs], jr[:, dofs]]))[2][-1]
            out[7 * h:7 * h + 7] += sign * self.swivel_rad * v / np.abs(v).max()
        return np.clip(out, self.lo, self.hi)

    def plan_chunk(self, poses, waist_meas, seed_q14, *, budget=None, ks=None, waists=None,
                   seed_k=None):
        """Start streaming this chunk's plan. `ks` (K,) tick numbers and
        `waists` (K, 3) per-tick trunks are optional; P6 passes neither, and
        consecutive ticks on the measured trunk are assumed."""
        poses = np.asarray(poses, np.float64)
        K = len(poses)
        ks = np.arange(K, dtype=float) if ks is None else np.asarray(ks, np.float64)
        waists = (np.repeat(np.asarray(waist_meas, np.float64)[None], K, axis=0)
                  if waists is None else np.asarray(waists, np.float64))
        seed = np.asarray(seed_q14, np.float64).copy()
        knots = sorted({*range(0, K, self.stride), K - 1})
        state = {}
        for f in self.fams:
            if f == "chain":
                state[f] = seed
            elif f in ("sw+", "sw-"):
                state[f] = self.swivel(seed, waists[0], 1.0 if f == "sw+" else -1.0)
            else:
                state[f] = self.fixed[f]
        self.st = {"poses": poses, "ks": ks, "waists": waists, "knots": knots, "next": 0,
                   "fam": 0, "row": [], "cands": [], "ref": None, "state": state,
                   "q0": seed.copy(), "k0": (ks[0] - 1.0) if seed_k is None else float(seed_k),
                   "in_transit": False, "calls": 0}

    def _advance_cell(self):
        st = self.st
        if st is None or st["next"] >= len(st["knots"]):
            return
        i = st["knots"][st["next"]]
        f = self.fams[st["fam"]]
        with self.anchor.suspended():
            q, _, _ = self.family_solver.solve(st["poses"][i], st["waists"][i], st["state"][f],
                                               budget=1.0)
        st["state"][f] = q
        e = self.residuals.tcp_error(q, st["waists"][i], st["poses"][i]) * 1e3
        st["row"].append((q, e))
        st["fam"] += 1
        if st["fam"] == len(self.fams):
            st["cands"].append(st["row"])
            st["row"], st["fam"] = [], 0
            st["next"] += 1
            st["ref"] = self._dp(st["knots"][:st["next"]], st["cands"], st["q0"], st["k0"])

    def _dp(self, knots, cands, q_now, k_now):
        st = self.st
        K, C = len(knots), len(self.fams)
        cost = np.full((K, C), np.inf)
        back = np.zeros((K, C), dtype=int)
        kk = np.array([st["ks"][i] for i in knots], np.float64)

        def node(q, e):
            if self.sr_gain > 0:
                m = np.minimum(q[self.SR] - self.lo[self.SR], self.hi[self.SR] - q[self.SR])
                e = e + self.sr_gain * float(np.sum(np.maximum(0.0, self.sr_rad - m)))
            if self.gamma <= 0:
                return e
            mg = float(np.min(np.minimum(q - self.lo, self.hi - q)))
            return e + self.gamma * max(0.0, self.m0 - mg)

        for c, (q, e) in enumerate(cands[0]):
            span = max(1.0, kk[0] - k_now) * self.dt
            if np.all(np.abs(q - q_now) <= self.vmax * span) or self.fams[c] == "chain":
                cost[0, c] = node(q, e) + self.beta * float(np.linalg.norm(q - q_now))
        for j in range(1, K):
            span = (kk[j] - kk[j - 1]) * self.dt
            for c, (q, e) in enumerate(cands[j]):
                best, arg = np.inf, 0
                for c0, (q0, _) in enumerate(cands[j - 1]):
                    if not np.isfinite(cost[j - 1, c0]):
                        continue
                    if not (np.all(np.abs(q - q0) <= self.vmax * span) or c == c0):
                        continue
                    v = cost[j - 1, c0] + self.beta * float(np.linalg.norm(q - q0))
                    if v < best:
                        best, arg = v, c0
                cost[j, c] = best + node(q, e)
                back[j, c] = arg
        c = int(np.argmin(cost[-1]))
        if not np.isfinite(cost[-1, c]):
            path = [self.fams.index("chain")] * K
        else:
            path = [c]
            for j in range(K - 1, 0, -1):
                c = back[j, c]
                path.append(c)
            path = path[::-1]
        return kk, np.array([cands[j][path[j]][0] for j in range(K)]), list(path)

    # --- tick ---------------------------------------------------------------

    def _tick_index(self, pose):
        """Which row of the chunk this tick is: the rows whose pose matches,
        the one nearest the tick expected from the call count (a held target
        repeats its pose; a hole skips a call)."""
        st = self.st
        d = np.abs(st["poses"] - np.asarray(pose, np.float64)).reshape(len(st["poses"]), -1).max(axis=1)
        hits = np.flatnonzero(d < 1e-6)
        expected = st["calls"]
        st["calls"] += 1
        if not len(hits):
            return None
        return int(hits[np.argmin(np.abs(hits - expected))])

    def _transit_goal(self, k, seed_q):
        st = self.st
        if st["ref"] is None:
            return None
        kk, qs, path = st["ref"]
        chain_i = self.fams.index("chain")
        j0 = int(np.searchsorted(kk, k, side="left"))
        for j in range(j0, len(kk)):
            cands = st["cands"][j]
            if self.transit_any:
                # The best family at the knot, on the DP's path or not; the
                # transit itself is paced at `transit_speed` over as many
                # ticks as it takes.
                fi = int(np.argmin([c[1] for c in cands]))
                if fi == chain_i or cands[chain_i][1] - cands[fi][1] < self.transit_mm:
                    continue
                if cands[fi][1] > self.transit_reach_mm:
                    # "Less bad" is not a reason to swing the arm across a
                    # branch: where no branch reaches (190940, 141126) it made
                    # p95 104 -> 223 / 105 -> 135. The goal must reach.
                    continue
                goal = cands[fi][0]
            else:
                if path[j] == chain_i:
                    continue
                if cands[chain_i][1] - cands[path[j]][1] < self.transit_mm:
                    continue
                goal = qs[j]
            travel = float(np.abs(goal - seed_q).max())
            if travel < self.transit_rad:
                st["in_transit"] = False
                return None
            need = (0 if self.transit_speed is None
                    else int(np.ceil(travel / (self.transit_speed * self.dt))) + 2)
            stuck = self.err_last * 1e3 >= self.transit_stuck_mm
            if st["in_transit"] or (stuck and (kk[j] - k <= need or self.transit_speed is None)):
                st["in_transit"] = True
                return goal
            return None
        st["in_transit"] = False
        return None

    def solve(self, pose, waist, seed_q, budget=None):
        st = self.st
        if st is None:
            return self.pp.solve(pose, waist, seed_q, budget=budget)
        for _ in range(len(self.fams)):
            self._advance_cell()
        i = self._tick_index(pose)
        k = st["ks"][i] if i is not None else st["ks"][0]
        if self.use_plan_waist and i is not None:
            waist = st["waists"][i]
        self.solved_waist = np.asarray(waist, np.float64).copy()
        ref = self.fixed["mid"]
        if st["ref"] is not None:
            kk, qs, _ = st["ref"]
            if self.seam_ramp and len(kk) > 1:
                # Ramp from the seed to the SECOND knot instead of snapping to
                # the first: a family switch the DP made at knot 0 is spread
                # over a knot span rather than pulled through in a tick.
                kk = np.r_[st["k0"], kk[1:]]
                qs = np.vstack([st["q0"], qs[1:]])
            if k <= kk[-1]:
                ref = np.array([np.interp(k, kk, qs[:, jj]) for jj in range(14)])
            elif st["next"] >= len(st["knots"]):
                ref = qs[-1]
        if self.transit is not None:
            goal = self._transit_goal(k, np.asarray(seed_q, np.float64))
            if goal is not None:
                t = self.ttemplate.copy()
                t[self.transit.jmap.arm] = goal
                self.ttask.set_target(t)
                with self.anchor.transiting(self.transit_speed):
                    q, e, ok = self.transit.solve(pose, waist, seed_q, budget=budget)
                self.transits += 1
                if ok:
                    self.err_last = e
                return q, e, ok
        t = self.template.copy()
        t[self.jmap_arm] = ref
        self.task.set_target(t)
        q, e, ok = self.pp.solve(pose, waist, seed_q, budget=budget)
        if ok:
            self.err_last = e
        return q, e, ok


# --- what P6 holds -----------------------------------------------------------


class Guarded:
    """The per-tick wrapper: anchors the motion caps on the tick's seed."""

    def __init__(self, inner, anchor: TickAnchor):
        self.inner, self.anchor = inner, anchor
        self.mode = inner.mode
        self.plans_chunks = getattr(inner, "plans_chunks", hasattr(inner, "plan_chunk"))

    def plan_chunk(self, *a, **k):
        return self.inner.plan_chunk(*a, **k)

    @property
    def solved_waist(self):
        return getattr(self.inner, "solved_waist", None)

    def solve(self, pose, waist, seed_q, budget=None):
        with self.anchor.ticking(seed_q):
            return self.inner.solve(pose, waist, seed_q, budget=budget)


def guard(solvers, anchor: TickAnchor, *, speed, self_speed=None) -> None:
    """Every QP in `solvers` gets the phantom-proof collision limit and the
    tick's motion caps (`speed` rad/s per joint, `self_speed` on the swivel)."""
    for s in solvers:
        collision.install(s.ik, **collision.GUARD)
        s.add_limit(MotionLimit(s.backend, anchor, speed=speed, self_speed=self_speed))


#: The shipped arm planner's settings: the escape (ungated transit, roll hinge
#: 120 mm/rad, pull 0.05) with the seam ramp; the 2.5 rad/s swivel cap goes in
#: `guard`. Chosen on recorded rollouts (see the module docstring); re-bench
#: before changing one.
ARM_PLANNER = {"pull_cost": 5e-2, "sr_gain": 120.0, "sr_rad": 1.0, "seam_ramp": True,
               "transit_cost": 0.3, "transit_speed": 3.0, "transit_stuck_mm": 10.0,
               "transit_any": True}


def arm_planner(cfg, *, urdf=None, qdot_max=2 * np.pi, tcp=None):
    anchor = TickAnchor(cfg.dt)
    s = StreamPlannerSolver(cfg, anchor, urdf=urdf, tcp=tcp, **ARM_PLANNER)
    guard(s.solvers, anchor, speed=qdot_max, self_speed=2.5)
    return Guarded(s, anchor)


arm_planner.mode = "arm"

__all__ = ["ARM_PLANNER", "Guarded", "PullPolish", "Residuals", "StreamPlannerSolver",
           "arm_planner", "arm_ranges", "guard", "posture_pull"]
