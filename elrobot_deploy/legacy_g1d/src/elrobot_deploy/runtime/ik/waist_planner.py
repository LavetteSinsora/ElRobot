"""The waist-mode planner: `ik = "waist_planner"`, the waist-mode default.

Design: `docs/infra_logic/ik_solvers.md`, "`waist_planner` — trunk plan + arm
planner". The numbers below were measured on the reference robot's recorded rollouts (2026-09);
the trunk constants are that unit's and want re-measuring on another.

Waist mode = a per-chunk TRUNK plan + the per-tick ARM solve, so the arm-only
machinery (`arm_planner.py`) drops in for the second half. In order:

  1. **the trunk plan** is `waist_qp.WholeBodySolver`'s passes A+B (waist group)
     with the bow floored at the physical stop and the posture pull toward
     nominal (`WbikWeights.bow_floor` / `nominal_trunk`), convergence-stopped
     at 2%: the trunk answers the chunk, and comes back when it is no longer
     needed forward.
  2. **slewed**: the plan is rate-limited against the last commanded trunk —
     0.2 rad/s bending, 0.1 unbending, 0.3 yaw (`SlewTrunk`). The raw plan
     bent and unbent every chunk; the trunk lagged by up to 0.027 rad and the
     arm, solved on the planned trunk, missed by that much at the hand.
  3. **settled**: the last 6 ticks of every chunk hold the trunk command
     (`settle`). Mid-swing at the seam the trunk carried on 30 mrad past the
     held command under its own weight.
  4. **lag-aware**: the arm is planned and ticked on the trunk the servo will
     ACTUALLY be at, not the one commanded. `trunk_servo` is the fitted model
     (38 runs, 2026-09-15): per tick the measured trunk moves 0.7 of the way to
     the command, at most 1 rad/s.

Measured: p95 10.3 mm / executed p95 13.6 on one set of waist-mode
recordings. **On a second set, arm mode was better** — every waist mode was far
worse there (the bow creeps to 0.8 rad). It is the default OF waist mode (the
best of the waist solvers), for the runs that want the reach; arm mode is the
recommendation.
"""

from __future__ import annotations

import numpy as np

from elrobot_deploy.runtime.ik.waist_qp import WbikWeights, WholeBodySolver
from elrobot_deploy.runtime.ik.velocity_caps import TickAnchor
from elrobot_deploy.runtime.ik.arm_planner import Guarded, StreamPlannerSolver, guard

#: Where the trunk rests: the physical stop, +0.040 rad (measured on the
#: reference robot; re-measure for another unit).
BOW_STOP = 0.040

#: The trunk servo, fitted on 38 recorded runs of the reference robot
#: (2026-09-15): per tick the measured bow/yaw move `TRUNK_GAIN` of
#: the way to the command, at most `TRUNK_VMAX` rad/s.
TRUNK_GAIN = 0.7
TRUNK_VMAX = 1.0


def trunk_servo(cmd: np.ndarray, k: np.ndarray, start: np.ndarray, dt: float) -> np.ndarray:
    """(N, 3) modelled trunk given the (N, 3) commands at ticks `k`, from
    `start` at `k[0]`: row i is where the trunk is when command i is issued."""
    out = np.zeros_like(cmd)
    out[0] = start
    for i in range(1, len(cmd)):
        span = max(1, int(k[i] - k[i - 1]))
        step = TRUNK_GAIN * (cmd[i - 1] - out[i - 1])
        lim = TRUNK_VMAX * dt * span
        out[i] = out[i - 1] + np.clip(step, -lim, lim)
    return out


class SlewTrunk:
    """Rate-limit a trunk plan against the last commanded trunk, and hold the
    command for the chunk's last `settle` ticks."""

    def __init__(self, bend=0.2, unbend=0.1, yaw=0.3, settle=6):
        self.bend, self.unbend, self.yaw, self.settle = bend, unbend, yaw, settle
        self.last = None

    def __call__(self, traj, waist_meas, dt):
        traj = np.asarray(traj, np.float64).copy()
        last = np.asarray(waist_meas, np.float64).copy() if self.last is None else self.last
        for i in range(len(traj)):
            b = traj[i, 1]
            traj[i, 1] = float(np.clip(b, last[1] - self.unbend * dt, last[1] + self.bend * dt))
            traj[i, 0] = float(np.clip(traj[i, 0], last[0] - self.yaw * dt, last[0] + self.yaw * dt))
            last = traj[i]
        K = len(traj)
        if self.settle > 0 and K > self.settle:
            traj[K - self.settle:] = traj[K - self.settle - 1]
        self.last = traj[-1].copy()
        return traj


class ReturnTrunk:
    """The chunk's trunk trajectory: whole-body passes A+B, slewed and settled."""

    def __init__(self, cfg, *, urdf=None, posture_waist=0.10, stride=8, slew=None,
                 tcp=None):
        weights = WbikWeights(posture_waist=posture_waist, pass_a_stride=stride,
                              bow_floor=BOW_STOP, nominal_trunk=True)
        self.ws = WholeBodySolver(cfg, groups=("waist",), weights=weights, urdf=urdf,
                                  rel=0.02, tcp=tcp)
        self.slew = SlewTrunk() if slew is None else slew
        self.dt = float(cfg.dt)

    def plan_chunk(self, poses, waist_meas, seed_q14, *, budget=None):
        traj = self.ws.plan_chunk(poses, waist_meas, seed_q14, budget=budget)
        return self.slew(traj, waist_meas, self.dt)


class LagAwareWaistPlanner:
    """Trunk plan -> predicted trunks via the servo model -> the arm planner
    planned and ticked on the prediction. P6 passes each tick's planned trunk
    as `waist`; `solved_waist` is the trunk the arm was actually posed on."""

    mode = "waist"
    plans_chunks = True

    def __init__(self, cfg, anchor: TickAnchor, *, urdf=None, trunk: ReturnTrunk, **arm_kw):
        self.trunk = trunk
        self.arm = StreamPlannerSolver(cfg, anchor, urdf=urdf, use_plan_waist=True, **arm_kw)
        self.solvers = [self.trunk.ws, *self.arm.solvers]
        self.dt = float(cfg.dt)

    def plan_chunk(self, poses, waist_meas, seed_q14, *, budget=None, ks=None, seed_k=None):
        traj = np.asarray(self.trunk.plan_chunk(poses, waist_meas, seed_q14, budget=budget),
                          np.float64)
        kk = np.arange(len(traj)) if ks is None else np.asarray(ks)
        pred = trunk_servo(traj, kk, np.asarray(waist_meas, np.float64), self.dt)
        self.arm.plan_chunk(poses, waist_meas, seed_q14, waists=pred, ks=ks, seed_k=seed_k)
        return traj

    @property
    def solved_waist(self):
        return self.arm.solved_waist

    def solve(self, pose, waist, seed_q, budget=None):
        return self.arm.solve(pose, waist, seed_q, budget=budget)


#: The waist pick's arm settings: the round-8 gated transit (bench
#: `waist_swm4_slew2s_lag_cg` was built before the round-12 escape).
WAIST_PLANNER = {"transit_cost": 0.3, "transit_speed": 3.0, "transit_stuck_mm": 10.0}


def waist_planner(cfg, *, urdf=None, qdot_max=2 * np.pi, tcp=None):
    anchor = TickAnchor(cfg.dt)
    s = LagAwareWaistPlanner(cfg, anchor, urdf=urdf,
                             trunk=ReturnTrunk(cfg, urdf=urdf, tcp=tcp),
                             tcp=tcp, **WAIST_PLANNER)
    guard(s.solvers, anchor, speed=qdot_max)
    return Guarded(s, anchor)


waist_planner.mode = "waist"

__all__ = ["BOW_STOP", "WAIST_PLANNER", "LagAwareWaistPlanner", "ReturnTrunk", "SlewTrunk",
           "trunk_servo", "waist_planner"]
