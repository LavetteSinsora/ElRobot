"""The examples the docs point to, one per extension point, actually work.

Each is exercised the way the runtime uses it: resolved from its deploy file
(plug-ins named by FILE, relative to the deploy file), built by the same
function P6 / P7 / the inspection server use, and run on something real.
"""

from __future__ import annotations

import dataclasses
import types
from pathlib import Path

import numpy as np
import pytest

from elrobot_deploy.cli import main as cli
from elrobot_deploy.config_loading import deploy_file
from elrobot_deploy.config_loading.rollout_settings import RolloutSettings
from elrobot_deploy.policy_contract import (DummyReader, Reader, bind_runtime,
                                        camera_info, dummy_observation)
from elrobot_deploy.policy_contract.loading import load_attr
from elrobot_deploy.runtime import ik
from elrobot_deploy.runtime.execution import chunk_blend, servo_filter
from elrobot_deploy.runtime.shared_memory import open_read_only, schema
from elrobot_deploy.runtime.shared_memory.rings import Contribution


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
NEW = ["policies/history.toml", "rtc/deploy.toml", "custom_modules/deploy.toml"]
MODULES = EXAMPLES / "custom_modules"


def _resolve(name):
    return deploy_file.resolve(EXAMPLES / name, record=False)


@pytest.mark.parametrize("example", NEW)
def test_every_example_passes_check_policy(example, capfd):
    with pytest.raises(SystemExit) as got:
        cli.main(["check-policy", str(EXAMPLES / example)])
    out = capfd.readouterr()
    assert got.value.code == 0, out.out + out.err
    assert "[ FAIL ]" not in out.out


def test_plugins_named_by_file_resolve_against_the_deploy_file():
    cfg = _resolve("custom_modules/deploy.toml").cfg
    assert cfg.ik == f"{MODULES / 'ik_solver.py'}:timed_arm_qp"
    assert cfg.blend == f"{MODULES / 'blend.py'}:consistent_ensemble"
    assert cfg.servo_smoother.startswith(str(MODULES / "servo.py"))
    assert _resolve("rtc/deploy.toml").cfg.blend == f"{EXAMPLES / 'rtc' / 'blend.py'}:newest"


# --- custom_ik ---------------------------------------------------------------


def test_the_custom_solver_solves_a_reachable_pose_and_times_it():
    r = _resolve("custom_modules/deploy.toml")
    solver = ik.build(r.cfg, r.hw)
    from elrobot_deploy.robot.kinematics import ForwardKinematics
    home = np.asarray(r.cfg.home_q, np.float64)
    target = ForwardKinematics(r.hw.urdf).flange(home, np.zeros(3)).copy()
    seed = home + 0.05
    q, err, ok = solver.solve(target, np.zeros(3), seed, budget=1 / 30)
    assert ok and err < 5e-3
    assert len(solver.solve_ms) == 1 and ik.mode_of(r.cfg.ik) == "arm"


# --- custom_blend ------------------------------------------------------------


def _pose(z):
    p = np.zeros((2, 7))
    p[:, 2], p[:, 3] = z, 1.0
    return p


def test_the_mode_aware_ensemble_averages_agreeing_chunks_and_drops_others():
    fn = chunk_blend.get(_resolve("custom_modules/deploy.toml").cfg.blend)
    near = Contribution(1, _pose(0.01), np.zeros((2, 6)), None, None, row=10)
    far = Contribution(2, _pose(0.50), np.zeros((2, 6)), None, None, row=5)
    new = Contribution(3, _pose(0.00), np.ones((2, 6)), None, None, row=0)
    got = chunk_blend.apply(fn, [near, far, new])
    assert 0.0 < got.pose[0, 2] < 0.01                # far's plan is dropped
    assert np.allclose(np.linalg.norm(got.pose[:, 3:], axis=1), 1.0)
    alone = chunk_blend.apply(fn, [far, new])
    assert alone.pose[0, 2] == 0.0                    # disagreement: the newest


# --- custom_servo ------------------------------------------------------------


def test_the_custom_smoother_and_feedforward_are_what_p7_builds():
    r = _resolve("custom_modules/deploy.toml")
    q0 = np.zeros(schema.BODY_DOF)
    shaping = servo_filter.build(r.cfg.servo, r.hw, q0, smoother=r.cfg.servo_smoother,
                                 feedforward=r.cfg.servo_feedforward)
    assert type(shaping.smoother).__name__ == "SecondOrderLowPass"
    assert type(shaping.feedforward).__name__ == "ViscousFeedforward"
    ref = np.full(schema.BODY_DOF, 0.1)
    for _ in range(1000):                              # 2 s at 500 Hz
        q, v, a = shaping.smoother.step(ref, 0.002)
    assert np.allclose(q, ref, atol=1e-3)
    tau = shaping.feedforward.tau(np.ones(schema.BODY_DOF), q0)
    assert np.all(tau[14:] == 0) and np.all(tau[:14] > 0)


# --- rtc ---------------------------------------------------------------------


def test_the_rtc_blend_is_the_newest_chunk_like_the_builtin():
    fn = chunk_blend.get(_resolve("rtc/deploy.toml").cfg.blend)
    old = Contribution(1, _pose(0.30), np.zeros((2, 6)), None, None, row=10)
    new = Contribution(2, _pose(0.10), np.ones((2, 6)), None, None, row=0)
    got, ref = chunk_blend.apply(fn, [old, new]), chunk_blend.apply(chunk_blend.newest, [old, new])
    assert np.array_equal(got.pose, ref.pose) and np.array_equal(got.hand, ref.hand)
    assert got.pose[0, 2] == 0.10



def test_rtc_freezes_exactly_the_rows_that_run_before_takeover(arena, rings, cfg):
    mod = load_attr(f"{EXAMPLES / 'rtc' / 'policy.py'}:frozen_prefix")
    ro = open_read_only(arena, cfg)
    n = cfg.H                                          # 8; the test arena's H
    rows = np.arange(n, dtype=float)
    rings["chunks"].write(
        chunk_id=3, k_first_action=100, n_actions=n,
        pose=np.repeat(np.repeat(rows[:, None, None], 2, 1), 7, 2),
        hand=np.zeros((n, 2, 6)), t_trigger=0.0, t_anchor=0.0,
        t_obs_received=0.0, t_cpu_processed=0.0, t_chunk_received=0.0,
        t_chunk_written=0.0)
    rings["control"].newest_chunk_id = 3
    rings["control"].k_exec = 101
    rings["control"].latency_s = 0.05                  # 1.5 ticks -> 2
    # P = 3: the next chunk starts at 103; takeover at 101 + 2 + 2 = 105.
    got, frozen = mod(ro, 3, 30.0)
    assert frozen == 2 and np.allclose(got[:, 0, 0], [3.0, 4.0])


def _rtc_arena(arena, rings, cfg, *, k_exec, newest=3, intervene=0):
    n = cfg.H
    rings["chunks"].write(
        chunk_id=3, k_first_action=100, n_actions=n,
        pose=np.zeros((n, 2, 7)), hand=np.zeros((n, 2, 6)), t_trigger=0.0,
        t_anchor=0.0, t_obs_received=0.0, t_cpu_processed=0.0,
        t_chunk_received=0.0, t_chunk_written=0.0)
    rings["control"].newest_chunk_id = newest
    rings["control"].k_exec = k_exec
    rings["control"].intervene = intervene
    rings["control"].latency_s = 0.05
    return open_read_only(arena, cfg)


def test_rtc_freezes_nothing_at_a_start_after_expiry_or_after_a_handback(
        arena, rings, cfg):
    fp = load_attr(f"{EXAMPLES / 'rtc' / 'policy.py'}:frozen_prefix")
    # a rollout's first inference: newest_chunk_id = 0 (slot 0 is empty)
    assert fp(_rtc_arena(arena, rings, cfg, k_exec=101, newest=0), 3, 30.0) == (None, 0)
    # the previous chunk (ticks 100..107) is already past
    assert fp(_rtc_arena(arena, rings, cfg, k_exec=400), 3, 30.0) == (None, 0)
    # an operator took over and handed back since the last inference
    ro = _rtc_arena(arena, rings, cfg, k_exec=101, intervene=2)
    assert fp(ro, 3, 30.0, intervene_seen=0) == (None, 0)
    rows, n = fp(ro, 3, 30.0, intervene_seen=2)
    assert n == 2


def test_rtc_policy_without_an_arena_is_its_own_chunk(cfg, hw):
    cls = load_attr(f"{EXAMPLES / 'rtc' / 'policy.py'}:Policy")
    pol = cls(None, "x", hw)
    dummy = dummy_observation({c: cfg.views[c] for c in cfg.views}, "x")
    settings = dataclasses.replace(RolloutSettings.of(cfg), sync=False)
    bind_runtime(pol, dummy, Reader(), camera_info(cfg), settings)
    pol.infer(dummy, Reader())
    assert pol.frozen_rows == 0


# --- history_policy ----------------------------------------------------------


def test_history_policy_stacks_square_fisheye_frames_and_a_state_window():
    r = _resolve("policies/history.toml")
    cls = load_attr(r.cfg.policy_entry)
    pol = cls(None, "x", r.hw)
    views = r.cfg.views
    dummy = dummy_observation({c: views[c] for c in ("rwrist", "head_left")}, "x")
    reader = DummyReader(views, dict(cls.history), state_hz=r.hw.lowstate_hz)
    bind_runtime(pol, dummy, reader, camera_info(r.cfg), RolloutSettings.of(r.cfg))
    images, q = pol.model_inputs(dummy, reader)
    x0, y0, x1, y1 = pol.cameras["rwrist"].crop
    side = x1 - x0
    assert images.shape == (3, side, side, 3) and q.shape == (50, 14)


# --- inspect_mode ------------------------------------------------------------


def test_the_inspection_mode_loads_by_file_and_plots_latency():
    from elrobot_deploy.dashboards.inspect import modes
    mode = modes.build(f"{EXAMPLES / 'inspect_mode' / 'mode.py'}:LatencyMode")
    assert mode.describe()["script"] == "/static/mode_latency.js"
    assert (mode.static_dir / "mode_latency.js").is_file()
    dt = schema.chunk_dtype(4)
    chunks = np.zeros(3, dt)
    chunks["chunk_id"] = [1, 2, 0]
    chunks["t_trigger"] = [10.0, 11.0, 0.0]
    chunks["t_anchor"] = [10.01, 11.02, 0.0]
    chunks["t_chunk_written"] = [10.12, 11.15, 0.0]
    loaded = types.SimpleNamespace(rollout=types.SimpleNamespace(chunks=chunks),
                                   timeline=types.SimpleNamespace(t0=9.0))
    got = load_attr(f"{EXAMPLES / 'inspect_mode' / 'mode.py'}:latency_strip")(loaded)
    assert got["t"] == [1.0, 2.0]
    assert got["total_ms"] == [120.0, 150.0] and got["anchor_ms"] == [10.0, 20.0]
