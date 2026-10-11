"""A toy JOINT policy: both elbows bend on a small sinusoid.

    g1d_deploy run examples/policies/sine_joint.toml

The smallest policy that exercises joint mode: `JointActions.arm_q` in the
per-arm order of `docs/deploying_a_policy/policy_contract.md`, "Joint order and
limits" (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw), an explicit upright `waist_q` of zeros (required in joint mode;
zeros are what P6 commands in eef mode under the arm IK, so the trunk does
exactly what it does there), and P6 skipping only the IK — epoch, cursor,
blend, clamps and the target ring are the eef path's.

Zeros, not the measured waist latched: the bow rests on a stop a hair outside
its URDF range (`docs/deploying_a_policy/policy_contract.md`, "The waist"), and
a latched measurement there would be refused as out of range.

At its first inference it latches the measured arm joints — read out of the
29-wide `obs.state.q` through `robot.joints_arm`, never by a hardcoded index —
and every action `j` adds `AMPLITUDE_RAD * sin(2π t / PERIOD_S)` to both elbows
(per-arm index 3), with `t` the time since the latch of the instant action `j`
is due. The result is clipped to the URDF's joint range, because joint mode is
the one path where nothing downstream solves against those limits and P5
REFUSES a chunk that leaves them. Re-latches after a gap, like `examples/policies/hold.py`.
"""

from __future__ import annotations

import numpy as np

from g1d_deploy import policy_contract as g1d
from g1d_deploy.runtime.shared_memory import schema
from g1d_deploy.robot.kinematics import arm_limits
from g1d_deploy.policy_contract import JointActions

AMPLITUDE_RAD = 0.15
PERIOD_S = 4.0
ELBOW = 3                           # per-arm joint index, docs/deploying_a_policy/policy_contract.md
RELATCH_GAP_S = 2.0
H, CONTROL_HZ = 50, 30.0



class Policy(g1d.Policy):
    action_mode = "joint"
    end_effector = "dex1"          # `hand` is the Dex-1 jaw, gear radians
    anchor_camera = "head_left"    # reads no pixels; every G1-D robot file has a head
    trained_control_hz = CONTROL_HZ
    trained_action_horizon = H

    def __init__(self, checkpoint, prompt, robot):
        super().__init__(checkpoint, prompt, robot)
        self._arm = np.asarray(robot.joints_arm, dtype=np.intp)     # (14,)
        lim = arm_limits(robot.urdf).reshape(schema.N_HANDS, 7, 2)
        self._lo, self._hi = lim[..., 0], lim[..., 1]
        self._latch = None          # (t0, arm_q (2, 7), hand (2,))
        self._t_last = None

    def warmup(self) -> None:
        """Nothing to compile, and a dummy would latch all-zero joints."""

    def infer(self, obs, reader) -> JointActions:
        if (self._latch is None or self._t_last is None
                or obs.t_anchor - self._t_last > RELATCH_GAP_S):
            arm_q = obs.state.q[self._arm].reshape(schema.N_HANDS, 7)
            self._latch = (obs.t_anchor, arm_q.copy(),
                           np.clip(obs.state.hand, 0.0, 5.4))
        self._t_last = obs.t_anchor
        t0, arm_q, hand = self._latch

        dt = 1.0 / CONTROL_HZ
        t = obs.t_anchor + dt * np.arange(1, H + 1) - t0          # (H,)
        q = np.repeat(arm_q[None], H, axis=0)                      # (H, 2, 7)
        q[:, :, ELBOW] += AMPLITUDE_RAD * np.sin(2 * np.pi * t / PERIOD_S)[:, None]
        return JointActions(arm_q=np.clip(q, self._lo, self._hi),
                            waist_q=np.zeros((H, schema.WAIST_DOF)),
                            hand=np.repeat(hand[None], H, axis=0))

