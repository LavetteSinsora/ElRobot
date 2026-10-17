"""P7's command shaping (`runtime/execution/servo_filter.py`, `resolved.ServoShaping`).

What must hold, in order of how badly it would hurt to get wrong:

* a LINEAR servo publishes exactly what P7 always published;
* the look-ahead never reads past the last written target, and never skips one:
  at a chunk seam it heads for the new chunk's first target;
* the filter is exact, stable on any period, and its lead cancels its lag;
* a stall still ends in a hold, reached without overshoot;
* the feedforward is bounded and touches only the arms;
* the defaults are the shaped servo, and the record says what ran.
"""

from __future__ import annotations

import dataclasses
import json
import os
import uuid

import numpy as np
import pytest

from elrobot_deploy.config_loading import resolved as cfgmod
from elrobot_deploy.config_loading.derive import derive, servo_gains
from elrobot_deploy.config_loading.errors import ConfigError
from elrobot_deploy.config_loading.validate import validate
from elrobot_deploy.runtime.shared_memory import Arena, build, open_all, schema
from elrobot_deploy.runtime.execution import servo_filter as shaping
from elrobot_deploy.runtime.processes.p7_body import BodyServo
from elrobot_deploy.runtime.processes.p9_record import Recorder
from elrobot_deploy.runtime.execution.servo_segment import Segment
from elrobot_deploy.runtime.drivers.arm_command import FakeLowCmd, ZeroComp

PERIOD = 1.0 / 500.0
DT = 1.0 / 30.0


@pytest.fixture
def rings(cfg, hw):
    """A deploy-sized target ring (128 slots): the shared fixture's 16 would
    wrap under the 40-target lines these tests walk."""
    big = cfgmod.replace(cfg, action_horizon=50, replan_period_ticks=25)
    a = Arena.create(f"elr-sv{os.getpid()}-{uuid.uuid4().hex[:8]}",
                     build(big, derive(big, hw)))
    try:
        yield open_all(a, big)
    finally:
        a.close()
        a.unlink()


def make(rings, hw, servo=None, *, q0=None, command=True):
    out = FakeLowCmd(hw)
    out.connect(np.zeros(len(hw.joints_leg)))
    q0 = np.zeros(schema.BODY_DOF) if q0 is None else np.asarray(q0, float)
    shape = shaping.LINEAR if servo is None else shaping.build(servo, hw, q0)
    body = BodyServo(ring=rings["body"], control=rings["control"],
                     state=rings["state"], out=out, gravity=ZeroComp(), hw=hw,
                     q_cmd_last=q0, t0=0.0, shaping=shape,
                     command=rings["command"] if command else None)
    return body, out


def run(body, seconds):
    for i in range(int(round(seconds / PERIOD))):
        for _ in range(3):
            if body.step(i * PERIOD):
                break


def ramp(rings, *, n=40, slope=0.3, t0=0.0, k0=1, jitter=0.0):
    """A 30 Hz target line: q = slope * t, optionally zig-zagged."""
    rng = np.random.default_rng(0)
    for k in range(k0, k0 + n):
        t = t0 + (k - k0 + 1) * DT
        q = np.full(schema.BODY_DOF, slope * t) + jitter * rng.standard_normal(schema.BODY_DOF)
        rings["targets"].write(k, t, q, intervene=0)


FILTER = cfgmod.ServoShaping(filter_hz=4.0, torque_ff=0.0)


# --- The filter --------------------------------------------------------------------


def test_the_filter_is_exact_so_the_period_split_does_not_matter():
    a, b = shaping.CritDamped(5.0, np.zeros(3)), shaping.CritDamped(5.0, np.zeros(3))
    r = np.array([1.0, -0.5, 0.2])
    a.step(r, 0.004)
    b.step(r, 0.002)
    b.step(r, 0.002)
    assert np.allclose(a.q, b.q, atol=1e-14) and np.allclose(a.v, b.v, atol=1e-13)


def test_the_filter_step_response_is_critically_damped_and_never_overshoots():
    w = 2 * np.pi * 4.0
    f = shaping.CritDamped(4.0, np.zeros(1))
    qs = [f.step(np.ones(1), PERIOD)[0][0] for _ in range(2000)]
    t = 100 * PERIOD
    assert abs(qs[99] - (1 - (1 + w * t) * np.exp(-w * t))) < 1e-12
    assert max(qs) <= 1.0 + 1e-12


# --- The feedforward ------------------------------------------------------------------


def test_the_feedforward_is_bounded_and_arm_only(hw):
    model = cfgmod.ServoModel(inertia=(0.02,) * 14, viscous=(0.4,) * 14,
                              coulomb=(0.3,) * 14, tau_max=0.5)
    ff = shaping.ModelFeedforward(model, scale=1.0)
    assert np.all(ff.tau(np.full(17, 10.0), np.zeros(17))[:14] == 0.5)
    tau = ff.tau(np.full(17, 0.1), np.full(17, 1.0))
    assert np.allclose(tau[:14], 0.02 + 0.4 * 0.1 + 0.3 * np.tanh(0.1 / 0.02))
    assert not tau[14:].any()
    negative = dataclasses.replace(model, inertia=(0,) * 14, viscous=(0,) * 14,
                                   coulomb=(-1.0,) * 14)
    assert not shaping.ModelFeedforward(negative, 1.0).tau(np.ones(17), np.zeros(17)).any()


# --- The look-ahead -------------------------------------------------------------------


def test_the_look_ahead_follows_the_written_line_and_holds_its_end(rings):
    ramp(rings, n=6, slope=0.3)                      # line ends at t = 0.2 s
    seg = Segment(rings["body"], rings["control"], np.zeros(17), 0.0, cursor=False)
    seg.phase(0.01, np.zeros(17))
    assert np.allclose(shaping.line_ahead(seg, rings["body"], 0.01, 0.08), 0.3 * 0.09)
    assert np.allclose(shaping.line_ahead(seg, rings["body"], 0.01, 0.5), 0.3 * 0.2)


def test_at_a_seam_the_look_ahead_heads_for_the_first_target_not_past_it(rings):
    """A stalled servo (inference) and a new chunk whose first target is 66 ms
    out: less than the 80 ms lead. The reference must be that target, not a
    later one, and not the held pose."""
    hold = np.zeros(17)
    seg = Segment(rings["body"], rings["control"], hold, 0.0, cursor=False)
    assert seg.phase(1.0, hold).name == "HOLD"
    q1 = np.full(17, 0.01)
    rings["targets"].write(30, 1.066, q1, intervene=0)
    assert np.allclose(shaping.line_ahead(seg, rings["body"], 1.0, 0.08), q1)
    rings["targets"].write(31, 1.1, np.full(17, 0.02), intervene=0)
    ref = shaping.line_ahead(seg, rings["body"], 1.0, 0.08)
    assert np.allclose(ref, 0.01 + 0.01 * (0.08 - 0.066) / (1.1 - 1.066))


def test_the_look_ahead_ignores_another_worlds_targets(rings):
    seg = Segment(rings["body"], rings["control"], np.zeros(17), 0.0, cursor=False)
    seg.phase(0.0, np.zeros(17))
    rings["targets"].write(5, 0.05, np.ones(17), intervene=1)
    assert not shaping.line_ahead(seg, rings["body"], 0.0, 0.08).any()


# --- P7 ---------------------------------------------------------------------------------


def test_a_linear_servo_publishes_the_line_and_logs_it(rings, hw):
    ramp(rings, jitter=0.01)
    a, out_a = make(rings, hw, command=True)
    b, out_b = make(rings, hw, command=False)
    run(a, 1.2)
    run(b, 1.2)
    assert np.array_equal(out_a.q_body, out_b.q_body)
    log = rings["command"]._rec
    assert np.array_equal(log["q"], log["q_lin"]) and not log["tau_ff"].any()


def test_the_filtered_command_follows_the_line_without_lag(rings, hw):
    ramp(rings, slope=0.3)
    body, out = make(rings, hw, FILTER)
    run(body, 1.2)
    t = np.arange(len(out.q_body)) * PERIOD
    steady = (t > 0.5) & (t < 1.2)
    assert np.abs(out.q_body[steady, 0] - 0.3 * t[steady]).max() < 1e-3
    rings["control"].k_exec = 0
    lagging, out2 = make(rings, hw, dataclasses.replace(FILTER, filter_lead=False))
    run(lagging, 1.2)
    assert np.median(out2.q_body[steady, 0] - 0.3 * t[steady]) < -0.02


def test_the_filtered_velocity_is_continuous_where_the_linear_one_steps(rings, hw):
    ramp(rings, jitter=0.004)
    lin, out_l = make(rings, hw)
    run(lin, 1.2)
    rings["control"].k_exec = 0
    filt, out_f = make(rings, hw, FILTER)
    run(filt, 1.2)
    ramping = slice(int(0.3 / PERIOD), int(1.2 / PERIOD))   # past the start from rest
    step_linear = np.abs(np.diff(out_l.dq[ramping, 0])).max()
    step_filtered = np.abs(np.diff(out_f.dq[ramping, 0])).max()
    assert step_linear > 0.2 and step_filtered < 0.05 * step_linear


def test_a_stall_ends_in_a_hold_on_the_held_line_without_overshoot(rings, hw):
    """The filter settles on what the line holds: like the linear servo, the
    last walked sample, at most one period of travel short of the target."""
    ramp(rings, n=10, slope=0.6)                     # line ends at 1/3 s, q = 0.2
    body, out = make(rings, hw, FILTER)
    run(body, 1.5)
    q = out.q_body[:, 0]
    assert q.max() <= 0.2 + 1e-9
    assert np.allclose(q[-1], body.q_line_last[0], atol=1e-6)
    assert abs(q[-1] - 0.2) <= 0.6 * PERIOD + 1e-9 and abs(out.dq[-1, 0]) < 1e-3


def test_the_feedforward_rides_on_gravity_and_is_logged(rings, hw):
    ramp(rings, n=20)
    body, out = make(rings, hw, cfgmod.ServoShaping())
    run(body, 0.6)
    log = rings["command"]._rec
    newest = log[np.argmax(log["index"])]
    assert np.allclose(out.tau[-1], newest["tau_ff"])          # ZeroComp gravity
    assert np.abs(out.tau[:, :14]).max() > 0.01 and not out.tau[:, 14:].any()


# --- config -----------------------------------------------------------------------------


def test_the_default_servo_is_smoothed_with_feedforward_and_linear_is_the_old_one():
    default, linear = cfgmod.ServoShaping(), cfgmod.ServoShaping.linear()
    assert default.filter_hz == 4.0 and default.torque_ff == 0.5
    assert linear.filter_hz == 0.0 and linear.torque_ff == 0.0 and linear.lead_s == 0.0


def test_hardware_carries_the_servo_model(hw):
    m = hw.servo_model
    assert m is not None and len(m.inertia) == len(m.viscous) == len(m.coulomb) == 14


def test_torque_ff_without_a_model_is_refused(cfg, hw):
    with pytest.raises(ConfigError, match="servo_model"):
        validate(cfg, dataclasses.replace(hw, servo_model=None))
    validate(cfgmod.replace(cfg, servo=cfgmod.ServoShaping.linear()),
                    dataclasses.replace(hw, servo_model=None))


@pytest.mark.parametrize("bad, match", [
    (dict(arm_gain_scale=3.0), "arm_gain_scale"),
    (dict(filter_hz=0.5), "filter_hz"),
    (dict(velocity_ff=2.0), "velocity_ff"),
    (dict(torque_ff=-1.0), "torque_ff"),
])
def test_a_servo_that_cannot_mean_what_it_says_is_refused(cfg, hw, bad, match):
    with pytest.raises(ConfigError, match=match):
        validate(cfgmod.replace(cfg, servo=cfgmod.ServoShaping(**bad)), hw)


def test_servo_gains_scale_the_arms_only(cfg, hw):
    kp0, kd0 = servo_gains(cfg, hw)
    kp, kd = servo_gains(
        cfgmod.replace(cfg, servo=cfgmod.ServoShaping(arm_gain_scale=2.0)), hw)
    arm = list(hw.joints_arm)
    rest = [i for i in range(schema.N_JOINTS) if i not in arm]
    assert np.allclose(kp[arm], 2 * kp0[arm]) and np.allclose(kd[arm], np.sqrt(2) * kd0[arm])
    assert np.array_equal(kp[rest], kp0[rest]) and np.array_equal(kd[rest], kd0[rest])


# --- The record ---------------------------------------------------------------------------


def test_the_record_carries_the_command_log_servo_model_gains_and_labels(cfg, hw, rings,
                                                                          tmp_path):
    c = cfgmod.replace(cfg, labels={"variant": "default"})
    ramp(rings, n=10)
    rec = Recorder(tmp_path, c, rings)
    body, _ = make(rings, hw, c.servo)
    for i in range(400):
        body.step(i * PERIOD)
        if i % 50 == 0:
            rec.sweep()
    rec.sweep()
    rec.close(hw)
    m = json.loads((tmp_path / "manifest.json").read_text())
    log = np.fromfile(tmp_path / "command.bin", dtype=schema.COMMAND)
    assert m["streams"]["command"]["records"] == len(log) > 300
    assert m["deploy_config"]["servo"]["filter_hz"] == 4.0
    assert m["deploy_config"]["labels"] == {"variant": "default"}
    assert len(m["servo_model"]["inertia"]) == 14 and "kp" in m["gains"]
