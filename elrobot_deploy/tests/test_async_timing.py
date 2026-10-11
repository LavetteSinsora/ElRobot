"""Async that cannot keep up is refused or reported, never silent.

Found by simulation (2026-09-29), with P6 and P7 running during inference:

* an inference longer than the replan period puts every trigger in the past,
  each chunk lands later than the last, and soon covers only ticks already
  gone — the arm holds forever and nothing says why;
* with too little horizon left past the replan period, every seam skips
  actions;
* `covering` handed P6 a chunk whose payload had landed before its id, so it
  could be blended into a tick one iteration before P6 adopted it.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from g1d_deploy.config_loading import resolved as cfgmod
from g1d_deploy.config_loading.errors import ConfigError
from g1d_deploy.config_loading.validate import check_async_timing
from g1d_deploy.runtime.processes.p5_policy import RolloutLatch
from g1d_deploy.runtime.shared_memory import schema

from .test_arena import _chunk
from .test_p0_launcher import (FakeClock, _measured, _play_p5,
                               _spawning_launcher)
from .test_p5_policy import _loop
from .test_settings import _Choosy, _activate, _dummy, _launch


def _async(cfg, **kw):
    return cfgmod.replace(cfg, sync=False, **kw)


# --- The rule ---------------------------------------------------------------


def test_a_fast_enough_policy_passes(cfg):
    # Typical numbers: 123 ms at 30 Hz, P = 25, H = 50.
    check_async_timing(_async(cfg, replan_period_ticks=25, action_horizon=50),
                       0.123)


def test_an_inference_longer_than_the_replan_period_is_refused(cfg):
    with pytest.raises(ConfigError, match="longer than a replan period") as e:
        check_async_timing(_async(cfg, replan_period_ticks=5, action_horizon=50),
                           0.200)
    assert "replan_period 7..42" in str(e.value), "says which P works"


def test_too_little_horizon_past_the_replan_period_is_refused(cfg):
    with pytest.raises(ConfigError, match="would skip actions"):
        check_async_timing(_async(cfg, replan_period_ticks=8, action_horizon=8),
                           0.120)


def test_when_no_replan_period_works_it_says_so(cfg):
    with pytest.raises(ConfigError, match="no replan period"):
        check_async_timing(_async(cfg, replan_period_ticks=4, action_horizon=8),
                           0.200)


def test_sync_and_an_unmeasured_latency_check_nothing(cfg):
    check_async_timing(cfgmod.replace(cfg, sync=True), 10.0)
    check_async_timing(_async(cfg), 0.0)


# --- P6 blends only what it adopted ------------------------------------------


def test_covering_leaves_out_a_chunk_newer_than_the_adopted_one(rings, cfg):
    _chunk(rings, cfg, 1, 10, cfg.P, 1.0)
    _chunk(rings, cfg, 2, 11, cfg.P, 2.0)          # landed, not yet adopted
    assert [c.chunk_id for c in rings["chunks"].covering(11)] == [1, 2]
    got = rings["chunks"].covering(11, adopted_through=1)
    assert [c.chunk_id for c in got] == [1]


# --- P5 measures, P0 refuses -------------------------------------------------


def test_the_latch_publishes_the_90th_percentile_plus_the_anchor_wait(rings, cfg):
    control = rings["control"]
    latch = RolloutLatch(control, _Choosy(cfg), cfg, clock=FakeClock(),
                         anchor_wait=0.010)
    for ms in [100] * 9 + [400]:                   # one outlier in ten
        latch.observe(ms * 1e-3)
    assert control.latency_s == pytest.approx(0.110, abs=1e-6)


class _Timed(_Choosy):
    """Each dummy inference takes `cost` seconds on the fake clock."""

    def __init__(self, cfg, clock, cost):
        super().__init__(cfg)
        self.clock, self.cost = clock, cost

    def infer(self, obs, reader):
        self.clock.t += self.cost
        return super().infer(obs, reader)


def test_changed_async_settings_are_remeasured_before_they_are_applied(rings,
                                                                        cfg):
    control = rings["control"]
    clock = FakeClock()
    policy = _Timed(cfg, clock, 0.050)
    latch = RolloutLatch(control, policy, cfg, clock=clock)
    for _ in range(5):
        latch.observe(0.300)                        # the old settings' cost
    new = dataclasses.replace(_launch(cfg), sync=False, sample_steps=4)
    _activate(control, new, cfg.prompt, 1)
    latch.poll(_dummy(cfg))
    assert control.settings_applied == 1
    assert control.latency_s == pytest.approx(0.050, abs=1e-6)
    assert len(policy.inferred) > 1, "the recompile dummy, then measured ones"


def test_changed_sync_settings_cost_one_dummy_only(rings, cfg):
    control = rings["control"]
    clock = FakeClock()
    policy = _Timed(cfg, clock, 0.050)
    latch = RolloutLatch(control, policy, cfg, clock=clock)
    _activate(control, dataclasses.replace(_launch(cfg), sample_steps=4),
              cfg.prompt, 1)
    latch.poll(_dummy(cfg))
    assert len(policy.inferred) == 1


def test_p0_refuses_an_async_start_the_policy_cannot_keep_up_with(rings, cfg,
                                                                  hw, derived):
    """Checked AFTER P5 took the settings (a new `sample_steps` changes the
    latency): the session stays READY, the reason is in the arena, and a
    START once the policy is fast enough goes through."""
    control = rings["control"]
    seen = {}

    class Clock(FakeClock):
        def sleep_until(self, t, **_):
            if not seen.get("pressed"):
                seen["pressed"] = 1
                control.request = schema.REQ_START
            elif control.refusal()[0] == 2 and "refusal" not in seen:
                seen["session"] = control.session
                seen["refusal"] = control.refusal()[1]
                control.latency_s = 0.020           # a faster policy
                control.request = schema.REQ_START
            _play_p5(control)
            return super().sleep_until(t)

    lch, _ = _spawning_launcher(cfg, hw, rings, derived, clock=Clock())
    control.request_config(dataclasses.replace(_launch(cfg), sync=False)
                           .to_record())
    control.latency_s = 0.500                       # 15 ticks > P = 3
    _measured(rings, hw, np.zeros(schema.BODY_DOF))

    lch.ready()                                     # returns on the 2nd START

    assert seen["session"] == schema.READY
    assert "longer than a replan period" in seen["refusal"]
    assert lch.rollout_cfg.sync is False
    assert control.rollout_generation == 2


# --- and says so when it drifts mid-rollout ----------------------------------


def test_a_chunk_landing_after_all_its_ticks_is_reported(rings, cfg, capsys):
    """500 ms at 30 Hz is 15 ticks: an 8-action chunk is entirely in the past.
    Judged by time, because a starved arm's `k_exec` has stopped too."""
    loop, _ = _loop(_async(cfg), rings, clock=FakeClock(t=10.0))
    loop._warn_if_late(7, n=8, t_trigger=9.5)
    assert "landed after all 8 of its ticks" in capsys.readouterr().out


def test_an_overrun_that_still_leaves_ticks_is_reported_as_late(rings, cfg,
                                                               capsys):
    loop, _ = _loop(_async(cfg), rings, clock=FakeClock(t=10.0))
    loop._warn_if_late(7, n=50, t_trigger=9.5)     # 500 ms > P·dt = 100 ms
    out = capsys.readouterr().out
    assert "longer than the replan period" in out and "landed after" not in out


def test_an_on_time_chunk_says_nothing(rings, cfg, capsys):
    loop, _ = _loop(_async(cfg), rings, clock=FakeClock(t=10.0))
    loop._warn_if_late(7, n=8, t_trigger=9.95)
    assert "WARNING" not in capsys.readouterr().out
