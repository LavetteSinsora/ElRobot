"""An async adapter that reads the arena: the bookkeeping of real-time chunking.

    g1d_deploy check-policy examples/rtc/deploy.toml
    g1d_deploy run examples/rtc/deploy.toml        # async, P = 10

Real-time chunking (RTC) generates each new chunk so that its first rows agree
with what the PREVIOUS chunk will still be executing while this one is being
inferred; with a blend that lets the newest chunk take over (`blend.py`) the
switch to the new chunk is then seamless. `README.md` beside this file explains
the whole example and the reasons for each read below.
A real RTC policy does that inside its sampler (inpainting the frozen prefix).
This example has no model: its "model" is the toy sinusoid of
`examples/policies/sine_eef.py`, and it enforces agreement the crude way — by copying the
frozen rows — so that the part worth copying is the bookkeeping:

* `reader.arena.chunks.get(newest_chunk_id)` — the previous chunk as P5
  published it (after the deadline shift, so exactly what will run), with its
  `k_first_action`;
* this chunk will start `P` ticks later (`self.settings.replan_period`, the
  rollout's own);
* it lands about `control.latency_s` from now, and P6 takes it over at the
  tick executing then, plus two (`control.k_exec + 2`). Every row before that
  is the previous chunk's, whatever this one says.

Three cases freeze nothing, because "the previous chunk" says nothing about
what runs next: a rollout's first inference (no chunk yet), a previous chunk
already past, and the first inference after a PICO operator handed back (P5
re-founds the tick line on `k_exec`, not on the old chunk plus P; seen as a
change of `control.intervene`).

The frozen rows match what runs to within a fraction of a tick: the
published rows are already re-timed by the deadline shift, and P5 re-times
this chunk again (by the sub-tick gap between its trigger and its anchor
frame). A strict RTC sampler conditions on them as a soft prefix.

`reader.arena` is None in warmup, check-policy and a policy replay: the
adapter then just returns its own chunk.
`docs/infra_logic/arena_layout_and_reads.md`, "Worked example: where is my last
chunk?"; `docs/infra_logic/chunk_blending.md`, "Real-time chunking is not a
blend".
"""

from __future__ import annotations

import numpy as np

from g1d_deploy import policy_contract as g1d
from g1d_deploy.policy_contract import EefActions

AMPLITUDE_M = 0.02
PERIOD_S = 4.0
RELATCH_GAP_S = 2.0
H, CONTROL_HZ = 50, 30.0


def frozen_prefix(arena, replan_period: int, control_hz: float, *,
                  intervene_seen: int | None = None):
    """`(rows, n)`: the previous chunk's rows that will run before a chunk
    inferred now can take over, as this chunk's first `n` rows — or
    `(None, 0)` when there is nothing to agree with (see the module doc).
    `intervene_seen`: `control.intervene` at the previous inference."""
    control = arena.control
    if control.newest_chunk_id == 0:                     # a rollout's first
        return None, 0
    if intervene_seen is not None and control.intervene != intervene_seen:
        return None, 0                                   # after a handback
    prev = arena.chunks.get(control.newest_chunk_id)
    if prev is None or int(prev["chunk_id"]) == 0:
        return None, 0
    k_first, n_prev = int(prev["k_first_action"]), int(prev["n_actions"])
    if k_first + n_prev <= control.k_exec + 2:           # already past
        return None, 0
    k_new_first = k_first + replan_period
    k_takeover = (control.k_exec + 2
                  + int(np.ceil(control.latency_s * control_hz)))
    n = max(0, min(k_takeover - k_new_first, n_prev - replan_period))
    return prev["pose"][replan_period:replan_period + n].copy(), n


class Policy(g1d.Policy):
    action_mode = "eef"
    end_effector = "dex1"
    anchor_camera = "head_left"
    trained_control_hz = CONTROL_HZ
    trained_action_horizon = H

    def __init__(self, checkpoint, prompt, robot):
        super().__init__(checkpoint, prompt, robot)
        self._latch = None
        self._t_last = None
        self.frozen_rows = 0            # how many rows the last chunk copied
        self._intervene = None          # control.intervene at the last inference

    def warmup(self) -> None:
        super().warmup()
        self._latch = None

    def _model(self, obs):
        """The stand-in for a model: the sine_eef example's vertical sinusoid, (H, 2, 7)."""
        if (self._latch is None or self._t_last is None
                or obs.t_anchor - self._t_last > RELATCH_GAP_S):
            self._latch = (obs.t_anchor, obs.state.flange.copy(),
                           np.clip(obs.state.hand, self.robot.hand_min,
                                   self.robot.hand_max))
        self._t_last = obs.t_anchor
        t0, flange, hand = self._latch
        t = obs.t_anchor + np.arange(1, H + 1) / CONTROL_HZ - t0
        pose = np.repeat(flange[None], H, axis=0)
        pose[:, :, 2] += AMPLITUDE_M * np.sin(2 * np.pi * t / PERIOD_S)[:, None]
        return pose, np.repeat(hand[None], H, axis=0)

    def infer(self, obs, reader) -> EefActions:
        pose, hand = self._model(obs)
        self.frozen_rows = 0
        if reader.arena is not None and self.settings is not None \
                and not self.settings.sync:
            rows, n = frozen_prefix(reader.arena, self.settings.replan_period,
                                    self.settings.control_hz,
                                    intervene_seen=self._intervene)
            self._intervene = reader.arena.control.intervene
            if n:
                pose[:n] = rows          # a real RTC sampler inpaints instead
                self.frozen_rows = n
        return EefActions(pose=pose, hand=hand)
