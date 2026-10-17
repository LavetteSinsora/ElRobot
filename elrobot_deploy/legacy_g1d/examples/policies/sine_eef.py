"""A toy EEF policy: both flanges trace a small vertical sinusoid.

    elrobot_deploy run examples/policies/sine_eef.toml

The smallest policy that exercises the whole eef contract with visible motion
and no model: absolute flange poses in the base frame, timed from `t_anchor`,
re-timed by the infra's deadline shift, cut to `P`, solved by the IK against
the measured trunk, with the waist commanded by P6 (zero). The contract it
exercises: `docs/deploying_a_policy/policy_contract.md`, "`EefActions`
(`action_mode = "eef"`)".

WHAT IT DOES

At its first inference it latches the measured flange poses and hand, then
every action `j` is that pose raised by `AMPLITUDE_M * sin(2π t / PERIOD_S)`,
with `t` the time since the latch of the instant action `j` is due:
`obs.t_anchor + (j + 1) * dt`. Timing the wave off `t_anchor` rather than off a
counter is the point of the demo: consecutive chunks agree on the trajectory
wherever they overlap, and the executed motion is a clean sinusoid only if the
infra's timing (anchor, shift, epoch) is right. It starts at `sin(0) = 0`, so
the first chunk continues from where the arm already is.

It re-latches after a gap of `RELATCH_GAP_S` between inferences (an operator
took over, or a new rollout began), like `examples/policies/hold.py`, and for the same
reason: holding the MEASURED pose every replan would ratchet down by the droop.

Safe amplitude: 2 cm at 0.25 Hz is ~3 cm/s peak at the flange.
"""

from __future__ import annotations

import numpy as np

from elrobot_deploy import policy_contract as g1d
from elrobot_deploy.policy_contract import EefActions

AMPLITUDE_M = 0.02
PERIOD_S = 4.0
RELATCH_GAP_S = 2.0
H, CONTROL_HZ = 50, 30.0



class Policy(g1d.Policy):
    action_mode = "eef"
    end_effector = "dex1"          # `hand` is the Dex-1 jaw, gear radians
    anchor_camera = "head_left"    # reads no pixels; every G1-D robot file has a head
    trained_control_hz = CONTROL_HZ
    trained_action_horizon = H

    def __init__(self, checkpoint, prompt, robot):
        super().__init__(checkpoint, prompt, robot)
        self._latch = None          # (t0, flange (2, 7), hand (2,))
        self._t_last = None

    def warmup(self) -> None:
        """Nothing to compile, and a dummy would latch a pose of zeros."""

    def infer(self, obs, reader) -> EefActions:
        if (self._latch is None or self._t_last is None
                or obs.t_anchor - self._t_last > RELATCH_GAP_S):
            self._latch = (obs.t_anchor, obs.state.flange.copy(),
                           np.clip(obs.state.hand, 0.0, 5.4))
        self._t_last = obs.t_anchor
        t0, flange, hand = self._latch

        dt = 1.0 / CONTROL_HZ
        t = obs.t_anchor + dt * np.arange(1, H + 1) - t0          # (H,)
        pose = np.repeat(flange[None], H, axis=0)                  # (H, 2, 7)
        pose[:, :, 2] += AMPLITUDE_M * np.sin(2 * np.pi * t / PERIOD_S)[:, None]
        return EefActions(pose=pose, hand=np.repeat(hand[None], H, axis=0))

