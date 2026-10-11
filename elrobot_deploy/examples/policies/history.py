"""An adapter that reads HISTORY: stacked frames and a state window.

    g1d_deploy check-policy examples/policies/history.toml
    g1d_deploy run examples/policies/history.toml

It holds the arm still (like `examples/policies/hold.py`) — the point is the reading a
policy with a temporal input does before it computes its actions:

* `history` DECLARES what it reads: one second of state and one second of the
  right wrist. The declaration sizes that camera's ring at launch and is what
  lets `reader` serve it (an undeclared stream raises `LookupError`).
* Three frames at fixed offsets before `t_anchor`, chosen by TIME with
  `nearest_to_times` from the metadata-only `frame_ids_between`, so only those
  three are copied (~1.7 MB each for a fisheye wrist), not all thirty. The
  wrist is a fisheye, so each frame is already the square around its lens
  circle: the infra cut it where CHECK found the lens this run.
* A state window, resampled by the adapter to the rate its policy expects.

Early in a rollout the windows hold less than a second; the adapter pads by
repeating the oldest item, the usual choice for a model trained on full
windows. `docs/deploying_a_policy/policy_contract.md`, "History: `reader`".
"""

from __future__ import annotations

import numpy as np

from g1d_deploy import policy_contract as g1d
from g1d_deploy.policy_contract import EefActions, nearest_to_times

H, CONTROL_HZ = 50, 30.0
FRAME_OFFSETS_S = np.array([1.0, 0.5, 0.0])     # frames at t_anchor - these
STATE_HZ = 50.0                                  # the rate "training" used
RELATCH_GAP_S = 2.0


class Policy(g1d.Policy):
    action_mode = "eef"
    end_effector = "dex1"
    anchor_camera = "rwrist"
    history = {"state": 1.0, "rwrist": 1.0}
    trained_control_hz = CONTROL_HZ
    trained_action_horizon = H

    def __init__(self, checkpoint, prompt, robot):
        super().__init__(checkpoint, prompt, robot)
        self._latch = None
        self._t_last = None

    def warmup(self) -> None:
        super().warmup()            # one dummy infer: full-length dummy history
        self._latch = None          # ...which latched a pose of zeros; forget it

    def model_inputs(self, obs, reader):
        """What a temporal model would be fed: (3, s, s, 3) frames, (50, 14) q."""
        t = obs.t_anchor
        ids = reader.frame_ids_between("rwrist", t - 1.0, t)
        picks = nearest_to_times(ids, t - FRAME_OFFSETS_S)
        frames = reader.frames_by_id("rwrist", picks.ids)
        if not frames:                                   # nothing captured yet
            frames = [obs.frames["rwrist"]]
        stack = [f.pixels for f in frames]
        stack = [stack[0]] * (len(FRAME_OFFSETS_S) - len(stack)) + stack

        w = reader.states_between(t - 1.0, t)
        grid = t - np.arange(int(STATE_HZ))[::-1] / STATE_HZ
        if len(w.t_recv):
            idx = np.clip(np.searchsorted(w.t_recv, grid), 0, len(w.t_recv) - 1)
            q = w.q[idx][:, self.robot.joints_arm]
        else:
            q = np.repeat(obs.state.q[None, self.robot.joints_arm], len(grid), 0)
        return np.stack(stack), q

    def infer(self, obs, reader) -> EefActions:
        images, q = self.model_inputs(obs, reader)       # a model would run here
        assert images.shape[0] == len(FRAME_OFFSETS_S) and q.shape == (50, 14)

        if (self._latch is None or self._t_last is None
                or obs.t_anchor - self._t_last > RELATCH_GAP_S):
            self._latch = (obs.state.flange.copy(),
                           np.clip(obs.state.hand, self.robot.hand_min,
                                   self.robot.hand_max))
        self._t_last = obs.t_anchor
        flange, hand = self._latch
        return EefActions(pose=np.repeat(flange[None], H, axis=0),
                          hand=np.repeat(hand[None], H, axis=0))
