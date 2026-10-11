"""Settings and prompt switched between rollouts.

What has to hold, and what nobody would notice if it did not:

* a request made DURING a rollout changes nothing until the NEXT start — the
  whole reason there are two blocks and only P0 reads the first;
* P0 refuses a START it cannot honour, and says why, rather than running a
  rollout with settings nobody asked for;
* the prompt channel cannot hand a reader half of one prompt and half of
  another;
* every process that latches reads the SAME settings for a rollout.
"""

from __future__ import annotations

import dataclasses
import threading
import time

import numpy as np
import pytest

from g1d_deploy.config_loading import resolved as cfgmod
from g1d_deploy.config_loading.derive import servo_gains
from g1d_deploy.runtime.shared_memory import schema
from g1d_deploy.config_loading.resolved import ServoShaping
from g1d_deploy.config_loading.errors import ConfigError
from g1d_deploy.policy_contract import EefActions, Policy
from g1d_deploy.runtime.processes.p5_policy import RolloutLatch, warmup
from g1d_deploy.runtime.processes.p7_body import BodyServo, relatch
from g1d_deploy.runtime.execution.servo_filter import LINEAR
from g1d_deploy.config_loading.rollout_settings import (LAUNCH_IK, RolloutSettings,
                                                        SettingsError, check, latched)
from g1d_deploy.runtime.drivers.arm_command import FakeLowCmd, ZeroComp

from .test_p0_launcher import (FakeClock, _measured, _play_p5,
                               _spawning_launcher)


def _launch(cfg) -> RolloutSettings:
    return RolloutSettings.of(cfg)


def _activate(control, settings: RolloutSettings, prompt: str, gen: int):
    """What P0 does at an accepted START: the block, then the counter."""
    control.activate(settings.to_record(), prompt)
    control.rollout_generation = gen


# --- The record ---------------------------------------------------------------


def test_settings_survive_the_arena_record_exactly(cfg):
    s = dataclasses.replace(
        _launch(cfg), sync=False, replan_period=5, control_hz=15.0, ik="waist_qp",
        sample_steps=3,
        servo=ServoShaping(filter_hz=6.0, filter_lead=False, velocity_ff=0.5,
                           torque_ff=0.0, arm_gain_scale=1.5))
    assert RolloutSettings.from_record(s.to_record()) == s


def test_every_shipped_solver_survives_the_record_and_the_shorthands_normalize(cfg):
    for name in ("arm_planner", "arm_qp", "waist_planner", "waist_qp"):
        s = dataclasses.replace(_launch(cfg), ik=name)
        assert RolloutSettings.from_record(s.to_record()).ik == name
    assert dataclasses.replace(_launch(cfg), ik="arm").ik == "arm_planner"
    assert dataclasses.replace(_launch(cfg), ik="waist").ik == "waist_planner"
    assert _launch(cfg).updated({"ik": "waist"}).ik == "waist_planner"


def test_a_launch_named_module_solver_travels_as_the_launchs_own(cfg, hw):
    """A `module:factory` has no index: the record says "the launch's", and
    `apply` gives the launch's name back. A rollout may leave it for a shipped
    solver, but not switch to some other module."""
    from . import test_ik_solvers

    mod = f"{test_ik_solvers.__name__}:factory_arm"
    launch = dataclasses.replace(cfg, ik=mod)
    s = _launch(launch)
    back = RolloutSettings.from_record(s.to_record())
    assert back.ik == LAUNCH_IK and back.apply(launch).ik == mod
    assert check(back, launch, hw).ik == mod
    assert check(dataclasses.replace(back, ik="arm_qp"), launch, hw).ik == "arm_qp"
    with pytest.raises(ConfigError, match="chosen at launch"):
        check(dataclasses.replace(
            back, ik=f"{test_ik_solvers.__name__}:factory_waist"),
              launch, hw)
    with pytest.raises(ConfigError, match="chosen at launch"):
        check(dataclasses.replace(_launch(cfg), ik=mod), cfg, hw)


def test_apply_changes_the_switchable_fields_and_nothing_else(cfg):
    s = dataclasses.replace(_launch(cfg), replan_period=2, control_hz=60.0)
    out = s.apply(cfg)
    assert (out.P, out.control_hz, out.dt) == (2, 60.0, 1 / 60.0)
    for name in ("prompt", "action_horizon", "cameras", "checkpoint",
                 "primary_camera", "state_history_s"):
        assert getattr(out, name) is getattr(cfg, name), name


def test_updated_takes_a_partial_request_and_refuses_what_it_cannot_parse(cfg):
    base = _launch(cfg)
    s = base.updated({"replan_period": 4, "servo": {"filter_hz": 0}})
    assert s.replan_period == 4 and s.servo.filter_hz == 0.0
    assert s.servo.torque_ff == base.servo.torque_ff, "untouched servo fields kept"
    for bad in ({"replan": 4}, {"servo": {"gain": 1}}, {"sync": "yes"},
                {"replan_period": 2.5}, {"ik": "legs"}, {"control_hz": "30"},
                {"replan_period": -3}, {"servo": {"filter_lead": 1}}):
        with pytest.raises(SettingsError):
            base.updated(bad)


# --- validation (P0 at START, P10 at submit) ----------------------------------


def test_check_refuses_a_replan_period_past_the_horizon(cfg, hw):
    with pytest.raises(ConfigError, match="action_horizon"):
        check(dataclasses.replace(_launch(cfg), replan_period=cfg.H + 1), cfg, hw)
    with pytest.raises(ConfigError, match=">= 2"):
        check(dataclasses.replace(_launch(cfg), replan_period=1), cfg, hw)
    rollout = check(dataclasses.replace(_launch(cfg), replan_period=cfg.H),
                    cfg, hw)
    assert rollout.P == cfg.H


def test_check_refuses_what_the_launch_cannot_run(cfg, hw):
    base = _launch(cfg)
    for waist in ("waist_planner", "waist_qp"):
        with pytest.raises(ConfigError, match="sync = false"):
            check(dataclasses.replace(base, ik=waist, sync=False), cfg, hw)
    with pytest.raises(ConfigError, match="control_hz"):
        check(dataclasses.replace(base, control_hz=0.0), cfg, hw)
    with pytest.raises(ConfigError, match="arm_gain_scale"):
        check(dataclasses.replace(base, servo=ServoShaping(arm_gain_scale=9.0)),
              cfg, hw)
    # --intervene was built against the plain servo and the per-tick arm QP:
    # the arm_planner default is refused under it, like the waist solvers.
    linear = dataclasses.replace(base, servo=ServoShaping.linear(), ik="arm_qp")
    check(linear, cfg, hw, intervene=True)
    with pytest.raises(ConfigError, match="linear servo"):
        check(dataclasses.replace(base, ik="arm_qp"), cfg, hw, intervene=True)
    for ik in ("arm_planner", "waist_planner", "waist_qp"):
        with pytest.raises(ConfigError, match="currently unsupported"):
            check(dataclasses.replace(linear, ik=ik), cfg, hw, intervene=True)


# --- The latch ------------------------------------------------------------------


def test_generation_zero_is_the_launch_config(rings, cfg):
    """Before any START nothing has been accepted, whatever the block holds."""
    out, prompt, gen = latched(rings["control"], cfg)
    assert (out, prompt, gen) == (cfg, cfg.prompt, 0)


def test_the_latch_reads_the_active_block_never_the_request(rings, cfg):
    control = rings["control"]
    _activate(control, dataclasses.replace(_launch(cfg), replan_period=2),
              "pick the cup", 1)
    control.request_config(dataclasses.replace(
        _launch(cfg), replan_period=7).to_record())
    out, prompt, gen = latched(control, cfg)
    assert (out.P, prompt, gen) == (2, "pick the cup", 1)


# --- The prompt channel ---------------------------------------------------------


def test_prompt_requests_are_numbered_and_round_trip_utf8(rings):
    control = rings["control"]
    assert control.prompt_request() == (0, "")
    a = control.request_prompt("open the box")
    b = control.request_prompt("öffne die Schachtel — 箱を開けて")
    assert (a, b) == (2, 4), "each request moves the seqlock by two"
    assert control.prompt_request() == (4, "öffne die Schachtel — 箱を開けて")


def test_a_prompt_that_does_not_fit_the_arena_is_refused_not_cut(rings):
    control = rings["control"]
    with pytest.raises(ValueError, match="arena holds"):
        control.request_prompt("x" * (schema.PROMPT_BYTES + 1))
    assert control.prompt_request() == (0, ""), "nothing half-written"


def test_a_reader_never_returns_a_prompt_mid_write(rings):
    """The seqlock: a reader that finds the counter odd waits it out, and
    returns the finished prompt — never the old length over the new bytes."""
    control = rings["control"]
    control.request_prompt("old prompt that is long")
    c = rings["control"]._c
    c["prompt_seq"][0] += 1                     # a writer is mid-write
    c["prompt"][0][:3] = np.frombuffer(b"new", dtype=np.uint8)
    got = []
    reader = threading.Thread(target=lambda: got.append(control.prompt_request()))
    reader.start()
    time.sleep(0.05)
    assert not got, "read a block whose writer had not finished"
    c["prompt"][0][:] = 0
    c["prompt"][0][:3] = np.frombuffer(b"new", dtype=np.uint8)
    c["prompt_len"][0] = 3
    c["prompt_seq"][0] += 1                     # the write lands
    reader.join(timeout=2.0)
    assert got == [(4, "new")]


# --- P0: latched at START, validated first ---------------------------------------


def _ready_launcher(cfg, hw, rings, derived, script):
    """A launcher whose clock runs `script` (a list of callables, one per P0
    sleep) and then plays P5 — the operator acting while P0 polls."""
    class Clock(FakeClock):
        steps = list(script)

        def sleep_until(self, t, **_):
            if self.steps:
                self.steps.pop(0)(rings["control"])
            _play_p5(rings["control"], refuse_prompt="TOO LONG")
            return super().sleep_until(t)

    lch, events = _spawning_launcher(cfg, hw, rings, derived, clock=Clock())
    return lch, events


def _press(name):
    return lambda control: setattr(control, "request",
                                   getattr(schema, f"REQ_{name.upper()}"))


def test_a_request_during_active_does_not_apply_until_the_next_start(
        rings, cfg, hw, derived):
    """THE property. The request block may be written at any time; the active
    block only at an accepted START, so a rollout's settings cannot move under
    it — and the next START does pick the request up."""
    control = rings["control"]
    two = dataclasses.replace(_launch(cfg), replan_period=2)
    five = dataclasses.replace(_launch(cfg), replan_period=5, control_hz=15.0)
    lch, _ = _ready_launcher(cfg, hw, rings, derived, [
        _press("start"),
        lambda c: c.request_config(five.to_record()),   # during ACTIVE
        _press("reset")])
    control.request_config(two.to_record())
    _measured(rings, hw, np.zeros(schema.BODY_DOF))

    lch.ready()
    assert control.rollout_generation == 1
    assert lch.rollout_cfg.P == 2
    lch.active()                                # the request lands in here
    rec, _ = control.active()
    assert RolloutSettings.from_record(rec) == two, "changed mid-rollout"
    assert latched(control, cfg)[0].P == 2

    lch.reset()
    lch.clock.steps.append(_press("start"))
    lch.ready()
    assert control.rollout_generation == 2
    assert RolloutSettings.from_record(control.active()[0]) == five
    assert lch.rollout_cfg.P == 5 and lch.rollout_cfg.control_hz == 15.0


def test_p0_refuses_a_bad_request_and_says_why(rings, cfg, hw, derived):
    """P > H is refused at START: the session stays READY, nothing is
    activated, the reason is in the arena for the dashboard — and the operator
    fixing the request and pressing again is all it takes."""
    control = rings["control"]
    bad = dataclasses.replace(_launch(cfg), replan_period=cfg.H + 1)
    good = dataclasses.replace(_launch(cfg), replan_period=4)
    seen = {}

    def after_refusal(c):
        seen["session"] = c.session
        seen["generation"] = c.rollout_generation
        seen["refusal"] = c.refusal()
        c.request_config(good.to_record())

    lch, _ = _ready_launcher(cfg, hw, rings, derived, [
        _press("start"), after_refusal, _press("start")])
    control.request_config(bad.to_record())

    lch.ready()                                 # returns only on the 2nd START

    assert seen["session"] == schema.READY
    assert seen["generation"] == 0, "a refused START activated something"
    seq, reason = seen["refusal"]
    assert seq == 2 and "action_horizon" in reason
    assert control.rollout_generation == 1
    assert RolloutSettings.from_record(control.active()[0]) == good


def test_p0_refuses_a_start_whose_prompt_the_policy_refused(rings, cfg, hw,
                                                           derived):
    control = rings["control"]
    control.request_prompt("TOO LONG " * 5)
    seen = {}

    def after_refusal(c):
        seen["refusal"] = c.refusal()[1]
        seen["prompt"] = c.active()[1]
        c.request_prompt("pick up the red cup")

    lch, _ = _ready_launcher(cfg, hw, rings, derived, [
        _press("start"), after_refusal, _press("start")])
    lch.ready()

    assert "refused the prompt" in seen["refusal"]
    assert "too many tokens" in seen["refusal"]
    assert seen["prompt"] == cfg.prompt, "the refused prompt became active"
    assert control.active()[1] == "pick up the red cup"


def test_no_prompt_request_keeps_the_active_prompt(rings, cfg, hw, derived):
    lch, _ = _ready_launcher(cfg, hw, rings, derived, [_press("start")])
    lch.ready()
    assert rings["control"].active()[1] == cfg.prompt


def test_a_policy_that_refuses_the_settings_refuses_the_start(rings, cfg, hw,
                                                              derived):
    control = rings["control"]
    steps = [_press("start")]

    class Clock(FakeClock):
        def sleep_until(self, t, **_):
            if steps:
                steps.pop(0)(control)
            elif control.refusal()[0] == 0:
                _play_p5(control, fail_settings=True)
            elif control.request == schema.REQ_NONE and \
                    control.refusal()[0] == 2 and not getattr(self, "again", 0):
                self.again = 1
                control.request = schema.REQ_START
            else:
                _play_p5(control)
            return super().sleep_until(t)

    lch, _ = _spawning_launcher(cfg, hw, rings, derived, clock=Clock())
    lch.ready()
    assert control.refusal()[0] == 2
    assert "policy refused" in control.refusal()[1]
    assert control.rollout_generation == 2 and control.settings_applied == 2


def test_p10_learns_the_step_and_port_from_the_launch_not_the_environment(
        cfg, hw, monkeypatch):
    """P10 used to read environment variables for these; the step now comes
    from the resolved deploy file via P0 and the port from the machine file."""
    import inspect

    from g1d_deploy.runtime import launcher as p0_launcher
    from g1d_deploy.runtime.processes import p10_dashboard

    assert "os.environ" not in inspect.getsource(p10_dashboard)
    lch = p0_launcher.Launcher(
        cfgmod.replace(cfg, checkpoint="/runs/x/29999"), hw,
        provenance={"deploy": {"resolved_tree": {"checkpoint": {"step": "latest"}}}})
    assert lch._checkpoint_step() == "latest"
    lch = p0_launcher.Launcher(cfgmod.replace(cfg, checkpoint="/runs/x/29999/"), hw)
    assert lch._checkpoint_step() == "29999"
    assert p0_launcher.Launcher(cfg, hw)._checkpoint_step() is None

    monkeypatch.setenv("G1D_DASHBOARD_PORT", "8123")
    assert cfgmod.load_hardware().dashboard_port == 8123
    monkeypatch.delenv("G1D_DASHBOARD_PORT")
    assert cfgmod.load_hardware().dashboard_port == 8080


# --- P5: judging prompts, taking a generation --------------------------------------


class _Choosy(Policy):
    """A policy with a token limit and a sampler it recompiles per step count."""

    action_mode = "eef"
    end_effector = "dex1"
    anchor_camera = "rwrist"
    max_prompt_tokens = 4

    def __init__(self, cfg):
        super().__init__(None, cfg.prompt, None)
        self.settings_seen = []
        self.prompts_seen = []
        self.inferred = []

    def count_prompt_tokens(self, text):
        return len(text.split())

    def set_prompt(self, prompt):
        super().set_prompt(prompt)
        self.prompts_seen.append(prompt)

    def on_settings(self, settings):
        if settings.sample_steps > 50:
            raise ValueError("no sampler compiled for that many steps")
        self.settings_seen.append(settings)

    def infer(self, obs, reader):
        self.inferred.append(obs.prompt)
        return EefActions(pose=np.zeros((1, 2, 7)), hand=np.zeros((1, 2)))


def _dummy(cfg):
    from g1d_deploy.policy_contract import dummy_observation
    return dummy_observation({c: cfg.cameras[c] for c in cfg.observed}, cfg.prompt)


def test_p5_judges_a_prompt_request_with_the_policys_own_check(rings, cfg):
    control = rings["control"]
    policy = _Choosy(cfg)
    latch = RolloutLatch(control, policy, cfg, clock=FakeClock())
    ok = control.request_prompt("pick the cup")
    latch.poll(_dummy(cfg))
    assert control.verdict() == (ok, schema.PROMPT_ACCEPTED, "")
    bad = control.request_prompt("please pick up the red cup now")
    latch.poll(_dummy(cfg))
    of, verdict, reason = control.verdict()
    assert (of, verdict) == (bad, schema.PROMPT_REFUSED)
    assert reason.startswith("Prompt exceeds the policy's maximum token length")
    assert policy.prompts_seen == [], "judging is not applying"


def test_p5_takes_a_generation_once_and_pays_the_compile_before_active(rings,
                                                                     cfg):
    control = rings["control"]
    policy = _Choosy(cfg)
    latch = RolloutLatch(control, policy, cfg, clock=FakeClock())
    dummy = _dummy(cfg)
    new = dataclasses.replace(_launch(cfg), sample_steps=4, replan_period=2)
    _activate(control, new, "open it", 1)

    dummy = latch.poll(dummy)
    assert control.settings_applied == 1
    assert policy.prompts_seen == ["open it"]
    assert policy.settings_seen == [new]
    assert policy.inferred == ["open it"], "one dummy inference, new prompt"
    assert dummy.prompt == "open it"
    assert latch.cfg.P == 2 and latch.cfg.sample_steps == 4

    latch.poll(dummy)                           # same generation: nothing
    assert len(policy.settings_seen) == 1 and len(policy.inferred) == 1

    # A new generation with the SAME settings re-delivers nothing to the policy.
    _activate(control, new, "open it", 2)
    latch.poll(dummy)
    assert control.settings_applied == 2
    assert policy.settings_seen == [new] and policy.prompts_seen == ["open it"]


def test_p5_reports_a_generation_the_policy_refused(rings, cfg):
    control = rings["control"]
    policy = _Choosy(cfg)
    latch = RolloutLatch(control, policy, cfg, clock=FakeClock())
    _activate(control, dataclasses.replace(_launch(cfg), sample_steps=99), "x", 1)
    latch.poll(_dummy(cfg))
    assert control.settings_applied == 1 | schema.APPLIED_FAILED
    assert latch.cfg is cfg, "a refused generation is not latched"


def test_warmup_polls_the_latch_between_dummies(rings, cfg):
    control = rings["control"]
    control.session = schema.READY
    policy = _Choosy(cfg)
    latch = RolloutLatch(control, policy, cfg, clock=FakeClock())
    _activate(control, _launch(cfg), "go", 1)
    real_infer = policy.infer

    def infer(obs, reader):
        out = real_infer(obs, reader)
        if len(policy.inferred) == 4:
            control.session = schema.ACTIVE
        return out

    policy.infer = infer
    dummy = warmup(control, policy, _dummy(cfg), clock=FakeClock(), latch=latch)
    assert control.settings_applied == 1
    assert dummy.prompt == "go"
    assert set(policy.inferred) == {"go"}


# --- P7: servo settings at the first tick of a rollout ----------------------------


def test_p7_retunes_at_active_and_only_for_a_new_generation(rings, cfg, hw):
    control = rings["control"]
    out = FakeLowCmd(hw)
    out.connect(np.zeros(len(hw.joints_leg)))
    servo = BodyServo(ring=rings["body"], control=control, state=rings["state"],
                      out=out, gravity=ZeroComp(), hw=hw,
                      q_cmd_last=np.zeros(schema.BODY_DOF), t0=0.0,
                      shaping=LINEAR)
    stiff = dataclasses.replace(
        _launch(cfg), servo=ServoShaping(filter_hz=0.0, torque_ff=0.0,
                                         arm_gain_scale=1.5))
    _activate(control, stiff, "p", 1)

    control.session = schema.READY              # accepted, not yet running
    assert relatch(servo, control, cfg, hw, 0) == 0
    assert servo.shaping is LINEAR and not hasattr(out, "gains")

    control.session = schema.ACTIVE
    assert relatch(servo, control, cfg, hw, 0) == 1
    kp, _ = servo_gains(stiff.apply(cfg), hw)
    np.testing.assert_allclose(out.gains[0], kp)
    arm = list(hw.joints_arm)
    np.testing.assert_allclose(out.gains[0][arm], np.asarray(hw.kp)[arm] * 1.5)

    shaping = servo.shaping
    assert relatch(servo, control, cfg, hw, 1) == 1
    assert servo.shaping is shaping, "re-latched the same generation"
