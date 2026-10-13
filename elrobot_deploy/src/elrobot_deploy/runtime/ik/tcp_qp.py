"""The ElRobot's per-tick IK: a TCP pose in, 7 arm joints out.

A mink differential-IK QP on `robot/elrobot_model.py`'s model. It is g1d's `arm_qp`
(`arm_qp.py` in this folder, kept for reference until the prune) reduced to one arm and one
frame, with the same four properties the deploy loop relies on:

  * **an explicit seed on every solve.** The seed is the previous *published* target, never
    the measurement (a measured seed feeds tracking error back into the next target). It sets
    both the QP configuration and the smoothness posture target.
  * **early exit** once the TCP position error is inside `early_stop` (0.1 mm), with
    `max_iters` as a secondary cap.
  * **a wall-clock budget.** `ok=False` means one thing only: the budget ran out. An
    unreachable target is not a failure; the arm goes as far as it can and the residual is
    returned, so "cannot go there" and "machine too busy" stay distinguishable.
  * **the per-tick speed cap inside the QP**, anchored on the seed: |q - seed| <= qdot_max*dt
    over the whole tick, however many iterates it takes. Clipping afterwards would publish a
    configuration the QP never evaluated.

And one addition for teleoperation, where a session lasts minutes rather than one chunk:

  * **a weak pull toward a nominal pose** (`nominal_cost`, `set_nominal`). The arm has one
    redundant DOF, mostly the forearm roll and the wrist roll turning against each other (their
    axes nearly line up when the wrist bend is small). Solving each tick from the last one lets
    that drift: measured on a 5 cm circle, the two rolls wandered up to 200 deg in four loops
    while the TCP followed. Teleop re-anchors the pull at every clutch press (`set_nominal`
    with the configuration at the press), so it leans toward where the operator started.
    At the default strength it costs nothing on paths that are feasible.

WHAT THIS ARM CAN AND CANNOT FOLLOW (measured on the description, 60 Hz, 5 cm circles at
0.5 Hz, `velocity_scale` 0.5):

  * gripper pitched down (the usual tabletop case): 0.05-0.1 mm, the speed cap never binds.
    The wrist roll's axis then has a vertical component, so it can yaw the gripper.
  * gripper LEVEL and pointing straight ahead, moved sideways at constant heading: it depends on
    the wrist's configuration, not on the motion. Only the base joint has a vertical axis;
    joints 2, 3, 4 (and the wrist bend, with the forearm roll at 0) are parallel horizontal
    pitch axes, and both rolls point along the forearm. So with a straight, un-rolled wrist,
    nothing but the base can turn the gripper about the vertical, and the base also swings the
    TCP sideways. A pure sideways move cannot cancel that turn: the configuration is singular
    for it, and following it needs very fast wrist motion (measured: speed cap bound on most
    ticks, up to ~20 mm behind).
    Rolling the wrist first (forearm roll +90, wrist roll -90: the gripper looks the same)
    stands the wrist-bend axis upright, turning it into a yaw joint. From there a level 5 cm
    sideways circle tracks to 0.1 mm. Mixed sideways-and-vertical circles were still 5-15 mm
    behind, because the solve drifts back to the un-rolled branch; holding the branch is
    `nominal_q` / `set_nominal`'s job, and how teleop picks it is still open.
    Whatever it cannot follow, the QP trades orientation first (cost 0.3 against 1.0 for
    position), stops at the limits, and reports the residual. It never jumps.

The target is the TCP (the midpoint of the jaw tips; `description/elrobot_ours/README.md`), in
the base frame, as `[x y z qw qx qy qz]`. There is no flange anywhere in this package.
"""

from __future__ import annotations

import numpy as np

from elrobot_deploy.robot.elrobot_model import TCP, ElRobotModel, pose_to_mat
from elrobot_deploy.runtime.timing import now


class SeedSpeedLimit:
    """A mink limit: |q - seed| <= speed * dt per joint, over the whole tick.

    mink's own `VelocityLimit` bounds each *iterate*, so a tick of N iterates could move N
    times as far. This one is anchored on the tick's seed, like g1d's `MotionLimit`."""

    def __init__(self, robot: ElRobotModel, speed: np.ndarray, dt: float):
        self.qadr, self.dadr = robot.qadr, robot.dadr
        self.nv = robot.model.nv
        self.speed = np.asarray(speed, dtype=np.float64)
        self.dt = float(dt)
        self.seed: np.ndarray | None = None
        E = np.zeros((len(self.qadr), self.nv))
        E[np.arange(len(self.qadr)), self.dadr] = 1.0
        self._G = np.vstack([E, -E])

    def compute_qp_inequalities(self, configuration, dt):
        from mink.limits.limit import Constraint

        if self.seed is None:
            return Constraint()
        dq = configuration.q[self.qadr] - self.seed
        room = self.speed * self.dt
        # mink's QP variable is the displacement, so the bounds are displacements too
        h = np.concatenate([np.maximum(room - dq, 0.0), np.maximum(room + dq, 0.0)])
        return Constraint(G=self._G, h=h)


class TcpSolver:
    """Base-frame TCP pose + seed -> 7 arm joints. See the module docstring."""

    def __init__(self, robot: ElRobotModel | None = None, *, dt: float = 1 / 60,
                 qdot_max=None, velocity_scale: float = 0.5,
                 position_cost: float = 1.0, orientation_cost: float = 0.3,
                 posture_cost: float = 1e-3, nominal_q=None, nominal_cost: float = 1e-3,
                 early_stop: float = 1e-4, max_iters: int = 20):
        """`qdot_max`: (7,) rad/s, or None for the URDF's velocity limits times
        `velocity_scale`. `nominal_q`: the pose the null-space pull leans toward; None is the
        middle of each joint's range. `dt` is the tick the speed cap is per (60 Hz teleop)."""
        import mink
        import qpsolvers

        self._mink = mink
        self.robot = robot or ElRobotModel()
        self.dt = float(dt)
        self.early_stop = float(early_stop)
        self.max_iters = int(max_iters)
        self.limits_q = self.robot.limits()
        self.qdot_max = (self.robot.velocity_limits() * float(velocity_scale) if qdot_max is None
                         else np.broadcast_to(np.asarray(qdot_max, np.float64), (7,)).copy())
        self.solver = "daqp" if "daqp" in qpsolvers.available_solvers else qpsolvers.available_solvers[0]

        m = self.robot.model
        self.config = mink.Configuration(m)
        self.task = mink.FrameTask(frame_name=TCP, frame_type="body", position_cost=position_cost,
                                   orientation_cost=orientation_cost, lm_damping=1e-2)
        self.posture = mink.PostureTask(m, cost=posture_cost)
        self.nominal = mink.PostureTask(m, cost=nominal_cost)
        self.set_nominal(self.limits_q.mean(1) if nominal_q is None else nominal_q)
        self.tasks = [self.task, self.posture] + ([self.nominal] if nominal_cost > 0 else [])
        self.speed_limit = SeedSpeedLimit(self.robot, self.qdot_max, self.dt)
        self.limits = [mink.ConfigurationLimit(m), self.speed_limit]

    def set_nominal(self, q7) -> None:
        """Lean the redundant DOF toward `q7` from now on (teleop: the pose at the clutch press)."""
        self.nominal_q = np.asarray(q7, np.float64).reshape(7).copy()
        q = np.zeros(self.robot.model.nq)
        q[self.robot.qadr] = self.nominal_q
        self.nominal.set_target(q)

    # --- one solve ---------------------------------------------------------------------------

    def place(self, seed_q7) -> None:
        """The seed sets the configuration, the smoothness target and the speed cap's anchor."""
        seed = np.clip(np.asarray(seed_q7, np.float64), self.limits_q[:, 0], self.limits_q[:, 1])
        q = np.zeros(self.robot.model.nq)
        q[self.robot.qadr] = seed
        self.config.update(q)
        self.posture.set_target_from_configuration(self.config)
        self.speed_limit.seed = seed

    def tcp_error(self, T_target: np.ndarray) -> tuple[float, float]:
        """(position error m, rotation error rad) of the TCP at the current configuration."""
        T = self.config.get_transform_frame_to_world(TCP, "body")
        p_err = float(np.linalg.norm(T.translation() - T_target[:3, 3]))
        R = T.rotation().as_matrix().T @ T_target[:3, :3]
        r_err = float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))
        return p_err, r_err

    def solve(self, tcp_pose, seed_q7, budget: float | None = None):
        """-> `(q7, position_error_m, ok)`. `tcp_pose` is `[x y z qw qx qy qz]` (or a 4x4) in
        the base frame. `budget` is wall-clock seconds, default one tick."""
        mink = self._mink
        T_target = np.asarray(tcp_pose, np.float64)
        if T_target.shape != (4, 4):
            T_target = pose_to_mat(T_target)
        budget = self.dt if budget is None else float(budget)
        self.place(seed_q7)
        self.task.set_target(mink.SE3.from_matrix(T_target))

        expires = now() + budget
        err = self.tcp_error(T_target)[0]
        iters = 0
        while err > self.early_stop:
            if now() >= expires:
                return self.q(), err, False
            if iters >= self.max_iters:
                break
            q_prev = self.config.q.copy()
            try:
                v = mink.solve_ik(self.config, self.tasks, self.dt, solver=self.solver,
                                  damping=1e-3, limits=self.limits)
            except mink.NoSolutionFound:
                break               # infeasible corner: keep what we have, report the residual
            self.config.integrate_inplace(v, self.dt)
            iters += 1
            if np.allclose(self.config.q, q_prev, atol=1e-12):
                break               # at the floor: further iterates move nothing
            err = self.tcp_error(T_target)[0]
        return self.q(), err, True

    def q(self) -> np.ndarray:
        return self.config.q[self.robot.qadr].copy()


__all__ = ["SeedSpeedLimit", "TcpSolver"]
