"""`Policy.settings`: the current rollout's settings, readable any time.

`on_settings` fires only when a Start CHANGES them, so without this an
adapter never learns the launch's replan period or control rate — which an
adapter that reasons about ticks (real-time chunking) needs from the first
rollout on."""

from __future__ import annotations

import dataclasses

from g1d_deploy.config_loading.rollout_settings import RolloutSettings
from g1d_deploy.policy_contract import bind_runtime
from g1d_deploy.runtime.processes.p5_policy import RolloutLatch

from .test_p0_launcher import FakeClock
from .test_settings import _Choosy, _activate, _dummy, _launch


def test_bound_with_the_launch_settings_and_updated_at_every_start(rings, cfg):
    control = rings["control"]
    policy = _Choosy(cfg)
    assert policy.settings is None
    bind_runtime(policy, None, None, None, RolloutSettings.of(cfg))
    assert policy.settings == _launch(cfg)

    latch = RolloutLatch(control, policy, cfg, clock=FakeClock())
    _activate(control, _launch(cfg), "go", 1)          # unchanged: no on_settings
    latch.poll(_dummy(cfg))
    assert policy.settings_seen == [] and policy.settings == _launch(cfg)

    new = dataclasses.replace(_launch(cfg), replan_period=2)
    _activate(control, new, "go", 2)
    latch.poll(_dummy(cfg))
    assert policy.settings == new


def test_a_refused_start_leaves_the_old_settings(rings, cfg):
    control = rings["control"]
    policy = _Choosy(cfg)
    bind_runtime(policy, None, None, None, RolloutSettings.of(cfg))
    latch = RolloutLatch(control, policy, cfg, clock=FakeClock())
    _activate(control, dataclasses.replace(_launch(cfg), sample_steps=99), "x", 1)
    latch.poll(_dummy(cfg))                            # _Choosy refuses > 50
    assert policy.settings == _launch(cfg)
