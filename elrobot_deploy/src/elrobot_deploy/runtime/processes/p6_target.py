"""P6: the operator's motion in, joint targets out.

Spawned by P0 at each ACTIVE and stopped at RESET or exit. Reads the intervention ring (the
phone source's rows), the target ring (for the seed) and the state ring (for the command the
bus process is holding); writes the target ring and nothing else. Each target is one IK solve
(`runtime/ik`, behind its interface: P6 never knows how a pose is solved).

WHO DRIVES: the parity of `control.intervene`

Odd: the operator holds the clutch, and each phone row becomes a target. Even: nobody drives
and P6 writes nothing; the bus process finds no row of the current world and holds its last
command. (A policy will be the even world's second source of targets later; g1d's chunk path is
in `legacy_g1d/src/elrobot_deploy/runtime/processes/p6_target.py`. Everything after the
source, `_commit`, is meant to be shared by both.)

THE ANCHOR, at every change of the counter to odd (a clutch press)

The rows carry the operator's motion RELATIVE to the press (`schema.INTERVENE.delta`), so P6
freezes the TCP pose at the same instant: `fk` of the command the bus process is holding (the
state ring's `q_cmd`, which is where the arm is being held, rather than the newest target, which
the bus abandoned when the counter moved). That configuration also seeds the first solve and is
handed to the solver as its nominal pose (`set_nominal`), so the redundant DOF leans toward the
posture the operator started from. An optional workspace box (`[teleop] workspace_min/max`,
none by default) is widened to contain the anchor, so a press outside it never jumps.

THE GRID

Rows arrive at the phone's ~60 Hz and are DECIMATED onto the target grid (`cfg.dt`): the next
row is taken once its deadline reaches the grid point `_due`, which advances by exactly `dt` per
target written, never to the deadline of the row that happened to be taken (that dithers at the
boundary and judders). `GUIDE_SLACK` (a little under half a frame) makes the rule "the row
nearest this grid point", kept off the exact half-frame knife edge a stream at twice the grid
rate would sit on. A row a whole `dt` late re-founds the grid on itself rather than firing a
backlog.

THE VELOCITY CLAMP

Against the previous target THIS engagement wrote (or the anchor), over the DEADLINE difference
between the two: a late or dropped phone frame can then license neither too fast nor too slow a
step. The bus process clamps again per tick; this bound keeps a far IK answer from being
written at all.

Abandoning a tick means writing nothing: a slot believed unsolved is never published. The bus
process holds through the hole.
"""

from __future__ import annotations

import time
from typing import NamedTuple

import numpy as np

from elrobot_deploy.runtime import timing as clock
from elrobot_deploy.runtime.shared_memory import Arena, layout_for, open_all, schema
from elrobot_deploy.runtime.teleop.intervention import apply_delta, is_intervening

#: How early (in grid periods) a row may be and still be taken as its grid point's.
GUIDE_SLACK = 0.45

#: Sleep this long when an iteration found nothing to do (no new row): P6 has a 16.7 ms grid,
#: and spinning would take a core from the bus process on a laptop.
IDLE_SLEEP_S = 0.0005


class Step(NamedTuple):
    """What one iteration did (for tests; the process ignores it)."""

    k: int | None
    written: bool
    deadline: float | None
    reason: str     # idle | released | decimated | unanchored | abandoned | gated | written


class Target:
    """The body of P6's loop, one `step()` per iteration."""

    def __init__(self, rings, cfg, solver, *, now=clock.now):
        self.cfg = cfg
        self.dt = float(cfg.dt)
        self.guide_slack = GUIDE_SLACK * self.dt
        self.control = rings["control"]
        self.targets = rings["targets"]
        self.state = rings["state"]
        self.rows = rings["intervene"]
        self.solver = solver
        self._now = now
        self.qdot_max = np.asarray(cfg.qdot_max, dtype=np.float64)
        self.lower = np.asarray(cfg.limits.lower, dtype=np.float64)
        self.upper = np.asarray(cfg.limits.upper, dtype=np.float64)
        self.box = None if cfg.teleop.workspace_min is None else (
            np.asarray(cfg.teleop.workspace_min, dtype=np.float64),
            np.asarray(cfg.teleop.workspace_max, dtype=np.float64))

        self._intervene: int | None = None          # the counter as last seen; None = never
        self.k: int | None = None                   # the last tick written this engagement
        self._row_index = 0
        self._due: float | None = None
        #: The engagement's anchor: TCP pose and configuration at the press.
        self.anchor_pose: np.ndarray | None = None
        self.anchor_q: np.ndarray | None = None
        self._box = self.box
        #: The velocity clamp's reference: the last written (q, deadline), or the anchor.
        self._seed_q: np.ndarray | None = None
        self._seed_t: float | None = None

    # --- one iteration ------------------------------------------------------------------

    def step(self) -> Step:
        counter = self.control.intervene
        if counter != self._intervene:
            self._intervene = counter
            self._switch(counter)
        if not is_intervening(counter):
            return Step(None, False, None, "released")
        if self.anchor_pose is None:
            return Step(None, False, None, "unanchored")
        return self._guided(counter)

    def _switch(self, counter: int) -> None:
        """The counter moved: everything latched against the old world goes. On a press,
        freeze the anchor."""
        self.k = None
        self._row_index = 0
        self._due = None
        self.anchor_pose = self.anchor_q = None
        self._seed_q = self._seed_t = None
        if not is_intervening(counter):
            return
        q = self._held_command()
        if q is None:
            return                      # nothing measured and nothing written: cannot anchor
        self.anchor_q = q
        self.anchor_pose = np.asarray(self.solver.fk(q[:schema.ARM_DOF]), dtype=np.float64)
        set_nominal = getattr(self.solver, "set_nominal", None)
        if callable(set_nominal):
            set_nominal(q[:schema.ARM_DOF])
        if self.box is not None:
            lo, hi = self.box
            p = self.anchor_pose[:3]
            self._box = (np.minimum(lo, p), np.maximum(hi, p))
        self._seed_q, self._seed_t = q.copy(), self._now()

    def _held_command(self) -> np.ndarray | None:
        """The 8 joints the bus process is holding: its newest `q_cmd`, else the newest target."""
        snap = self.state.latest()
        if snap is not None and np.all(np.isfinite(snap["q_cmd"])):
            return np.asarray(snap["q_cmd"], dtype=np.float64).copy()
        row = self.targets.newest_full()
        return None if row is None else np.asarray(row["q"], dtype=np.float64).copy()

    def _guided(self, counter: int) -> Step:
        row = self.rows.newest(counter, since=self._row_index)
        if row is None:
            return Step(None, False, None, "idle")
        self._row_index = row.index
        if self._due is not None and row.deadline < self._due - self.guide_slack:
            return Step(None, False, None, "decimated")
        k = (self.control.k_exec + 2) if self.k is None else self.k + 1
        pose = apply_delta(self.anchor_pose, row.delta)
        if self._box is not None:
            pose[:3] = np.clip(pose[:3], *self._box)
        step = self._commit(k, pose, row.grip, row.deadline, counter)
        if step.written:
            self.k = k
            on_grid = self._due is not None and row.deadline < self._due + self.dt
            self._due = (self._due if on_grid else row.deadline) + self.dt
        return step

    # --- the shared tail (the policy path will share it) --------------------------------

    def _commit(self, k: int, pose, grip: float, deadline: float, counter: int) -> Step:
        """Solve `pose`, clamp, gate, write tick `k`, or write nothing."""
        seed = self._seed_q
        q_arm, err, ok = self.solver.solve(pose, seed[:schema.ARM_DOF], budget=self.dt)
        if not ok:
            return Step(k, False, deadline, "abandoned")
        q = np.empty(schema.N_JOINTS)
        q[:schema.ARM_DOF] = q_arm
        q[schema.GRIPPER] = grip
        q = np.clip(q, self.lower, self.upper)
        # The span is the DEADLINE difference to the seed: a late or dropped frame licenses
        # neither too fast nor too slow a step. At least one grid period, so the first target
        # after the anchor (due one lead after it) is not held to a few milliseconds of motion.
        span = max(deadline - self._seed_t, self.dt)
        limit = self.qdot_max * span
        q = np.clip(q, seed - limit, seed + limit)
        k_exec = self.control.k_exec                    # fresh: the write gate
        if k == k_exec or k == k_exec - 1:
            return Step(k, False, deadline, "gated")
        self.targets.write(k, deadline, q, intervene=counter, err=float(err), pose=pose)
        self._seed_q, self._seed_t = q, deadline
        return Step(k, True, deadline, "written")


# --- the process ----------------------------------------------------------------------------


def run(shm_name: str, cfg) -> None:
    """P6 as a process. Runs while `session == ACTIVE`."""
    arena = Arena.attach(shm_name, layout_for(cfg))
    try:
        rings = open_all(arena)
        # mujoco / mink are imported here, by the factory, and nowhere else in the runtime.
        from elrobot_deploy.runtime.ik import build as build_solver
        loop = Target(rings, cfg, build_solver(cfg))
        control = rings["control"]
        while control.session == schema.ACTIVE:
            if not loop.step().written:
                time.sleep(IDLE_SLEEP_S)
    finally:
        arena.close()


__all__ = ["GUIDE_SLACK", "IDLE_SLEEP_S", "Step", "Target", "run"]
