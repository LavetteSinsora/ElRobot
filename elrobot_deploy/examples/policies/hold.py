"""A policy that holds still, for bringing up the intervention.

    elrobot_deploy run examples/deploy.toml --intervene
    # the test robot, no fisheye rig (head camera only):
    elrobot_deploy run examples/deploy.toml --intervene \
        --robot g1d_normal_dex1

Step one of the intervention bring-up: the whole system, the real
cameras, the real IK, the real wire, and a "policy" that asks the arm to stay
where it is. **Any motion at all that the operator did not cause is a framework
bug**, which is what makes it the right thing to have underneath the first
takeover.

It is deliberately not a special mode. P5 runs, the schedule runs, chunks are
published and adopted, P6 solves them, and the handback goes through
`first_inference` exactly as it will with any other policy. It is also the
smallest example of a policy that is not a neural network: an adapter is
anything that returns actions in the contract
(`docs/deploying_a_policy/policy_contract.md`).

WHAT IT HOLDS — A LATCH, NOT THE LIVE MEASUREMENT

Re-reading the MEASURED flange at every inference is a ratchet: the arm sits
below its command by the steady-state droop (~13 mm on the reference robot), so
commanding the measurement lowers the target, the arm sags to the new one, and
the next replan lowers it again — every ~830 ms, for the whole rollout.

So this LATCHES the measurement once and holds that, which is a fixed point:
commanding what is already commanded moves nothing. The one-time cost is a
single droop's worth of sag at the first chunk.

WHEN IT RE-LATCHES

When the time since its last inference exceeds `RELATCH_GAP_S`. P5 infers once
per replan period (~0.83 s here) for as long as the policy is driving, and not
at all while an operator is or between rollouts — so a long gap means
somebody else has had the arm, and the right thing to hold is where they left
it: no swing back to where the rollout started, and no step. Set
`LATCH_ONCE = True` to latch only at the first inference of the process
instead — a louder demonstration that the policy really does have the robot
again at a handback — and be ready for the motion.

"""

from __future__ import annotations

import numpy as np

from elrobot_deploy import policy_contract as g1d
from elrobot_deploy.policy_contract import EefActions

#: Latch only once per process. Off by default — see the module docstring.
LATCH_ONCE = False

#: A gap between two inferences longer than this re-latches. Two and a bit
#: replan periods: long enough that no ordinary schedule jitter trips it, short
#: enough that any intervention or rollout boundary does.
RELATCH_GAP_S = 2.0

H, CONTROL_HZ = 50, 30.0


class Policy(g1d.Policy):
    """Hold. Every action is the latched pose, repeated `H` times."""

    action_mode = "eef"
    end_effector = "dex1"          # `hand` is the Dex-1 jaw, gear radians
    anchor_camera = "head_left"    # reads no pixels; every G1-D robot file has a head
    trained_control_hz = CONTROL_HZ
    trained_action_horizon = H

    def __init__(self, checkpoint, prompt, robot):
        super().__init__(checkpoint, prompt, robot)   # no checkpoint to load
        self._latched = None
        self._t_last = None

    def warmup(self) -> None:
        """Nothing to compile; and running `infer` on the dummy would latch a
        pose of all zeros, which is the one thing this must never hold."""

    def infer(self, obs, reader) -> EefActions:
        gap = (float("inf") if self._t_last is None
               else obs.t_anchor - self._t_last)
        self._t_last = obs.t_anchor
        if self._latched is None or (not LATCH_ONCE and gap > RELATCH_GAP_S):
            # The measured hand is in the same units `hand` is commanded in
            # (the Dex-1 jaw, gear radians), so it is held as read.
            self._latched = (obs.state.flange.copy(),
                             np.clip(obs.state.hand, 0.0, 5.4))
        flange, hand = self._latched
        return EefActions(pose=np.repeat(flange[None], H, axis=0),
                          hand=np.repeat(hand[None], H, axis=0))


