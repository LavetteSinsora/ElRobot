"""The inspection page's second timeline: each mode's per-tick metric over the
whole rollout (`/api/ik/strip`, `/api/executor/strip`, `/api/replay/strip`).

The record stands at home and commands home (`inspect_fixtures`), so every
residual is known in advance: the deployed IK is exact, a command stream
offset by `x` rad tracks at exactly `x`, a replay raised by `d` mm differs by
exactly `d`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from g1d_deploy.dashboards.inspect.options import Options
from g1d_deploy.dashboards.inspect import loader, server
from g1d_deploy.dashboards.inspect.modes import ik as ik_mode
from g1d_deploy.dashboards.inspect.producers import ik_bench
from g1d_deploy.dashboards.inspect.modes import executor as ex
from g1d_deploy.dashboards.inspect.modes import replay as replay_mode
from g1d_deploy.dashboards.inspect.modes.replay import anchors, store
from g1d_deploy.dashboards.inspect.modes.replay.overlay import ChunkPoses, action_diff

from . import inspect_fixtures as fx

pytest.importorskip("mujoco")


def _mounted(record_dir, mode, root=None):
    opts = Options(root=Path(root or record_dir),
                   state_file=Path(record_dir).parent / "state.json")
    app = server.build_app(opts, modes=[])
    mode.options = opts
    mode.mount(app, app.state)
    key = "." if root is None else str(Path(record_dir).relative_to(root))
    app.state.load(key, 30.0)
    return app


def _deadlines(rec):
    ro = loader.load(rec)
    try:
        return np.asarray(ro.body_targets["deadline"], float)
    finally:
        ro.close()


# --- The page ------------------------------------------------------------------


def test_the_strip_sits_under_the_scrubber_and_every_mode_feeds_it():
    """One strip, in the transport under the slider, drawn with the shared
    strip chart; each built-in mode fills it from its own route."""
    static = Path(server.__file__).parent / "static"
    html = (static / "index.html").read_text()
    footer = html[html.index('<footer id="transport">'):html.index("</footer>")]
    assert footer.index('id="controls"') < footer.index('id="metric-row"')
    assert '<canvas id="metric">' in footer
    metric = (static / "metric.js").read_text()
    assert 'from "/static/strip.js"' in metric
    app = (static / "app.js").read_text()
    assert 'from "/static/metric.js"' in app
    assert "metric.setCursor(t)" in app          # the main playhead drives it
    assert "goto(frameAt(t))" in app             # and it seeks the main view
    for mode, route in (("ik", "/api/ik/strip"), ("executor", "/api/executor/strip"),
                        ("replay", "/api/replay/strip")):
        js = (static / f"mode_{mode}.js").read_text()
        assert route in js and "hooks.strip.show(" in js, mode


# --- ik ------------------------------------------------------------------------


def test_ik_strip_has_the_deployed_line_without_a_bench(tmp_path):
    rec = fx.write(tmp_path / "r", n_chunks=2)
    mode = ik_mode.IkMode()
    app = _mounted(rec, mode)
    try:
        d = mode.strip()
        cur = app.state.current
        assert d["applicable"] and d["point"] == "virtual TCP"
        assert [s["name"] for s in d["series"]] == ["deployed"]
        n = len(d["k"])
        assert n == len(d["t"]) == len(d["series"][0]["tcp_mm"]) > 0
        # On the main timeline's axis: the tick's deadline from the window start.
        want = _deadlines(rec)[:n] - cur.timeline.t0
        assert np.allclose(d["t"], want, atol=1e-3)
        assert max(d["series"][0]["tcp_mm"]) < 1e-3
        assert max(d["series"][0]["rot_deg"]) < 1e-2
    finally:
        app.close()


def test_ik_strip_adds_one_line_per_benched_method(tmp_path):
    rec = fx.write(tmp_path / "r", n_chunks=2)
    ik_bench.solve(rec, ["arm_qp"], log=lambda *_: None)
    mode = ik_mode.IkMode()
    app = _mounted(rec, mode)
    try:
        d = mode.strip()
        names = [s["name"] for s in d["series"]]
        assert names == ["deployed", "arm_qp"]
        qp = d["series"][1]
        assert qp["source"] == "bench"
        assert len(qp["tcp_mm"]) == len(d["k"])
        assert all(v is not None and v < 2.0 for v in qp["tcp_mm"])   # mm
        assert all(v is not None for v in qp["rot_deg"])
        # The mm are the sidecar's own metres, scaled.
        with np.load(rec / "ik_bench.npz") as z:
            assert np.allclose(qp["tcp_mm"], z["arm_qp/tcp_err"] * 1e3, atol=1e-3)
            assert np.allclose(qp["rot_deg"], np.degrees(z["arm_qp/rot_err"]),
                               atol=1e-2)
    finally:
        app.close()


def test_ik_strip_scores_the_flange_when_the_robot_has_no_tcp(tmp_path):
    """A hand nobody measured a TCP for (the BrainCo): the residual is at the
    flange, and the strip says so."""
    def no_tcp(m):
        m["robot"]["files"].append({
            "path": "/rig/no_tcp.toml",
            "text": 'extends = "g1d_fisheye_dex1"\n[end_effector]\ntype = "brainco"\n'})
    rec = fx.write(tmp_path / "r", n_chunks=2, manifest_edit=no_tcp)
    ro = loader.load(rec)
    assert ro.model.tcp is None
    ro.close()
    mode = ik_mode.IkMode()
    app = _mounted(rec, mode)
    try:
        assert mode.strip()["point"] == "flange"
    finally:
        app.close()


def test_ik_strip_says_why_a_joint_rollout_has_none(tmp_path):
    rec = fx.write(tmp_path / "j", example="sine_joint", n_chunks=2)
    mode = ik_mode.IkMode()
    app = _mounted(rec, mode)
    try:
        d = mode.strip()
        assert d["applicable"] is False and "joint" in d["why"]
        assert d["series"] == []
    finally:
        app.close()


# --- executor ------------------------------------------------------------------


def test_executor_strip_is_the_mean_arm_tracking_error_per_tick(tmp_path):
    root = tmp_path / "records"
    rec = fx.write(root / "20260926-120000", command_offset=0.01)
    fx.write(rec / ex.REPLAYS_DIR / "20260926-130000", command_offset=0.03)
    mode = ex.ExecutorMode()
    app = _mounted(rec, mode, root=root)
    try:
        own = mode.strip("")
        other = mode.strip("20260926-130000")
        for d, off in ((own, 0.01), (other, 0.03)):
            assert d["has_command"] and d["unit"] == "deg"
            err = np.array([v for v in d["err_deg"] if v is not None])
            assert len(err) >= len(d["t"]) - 1
            assert np.allclose(err, np.degrees(off), atol=1e-3)
        # The rollout's own ticks land on its own deadlines on the display axis.
        t0 = app.state.current.timeline.t0
        assert np.allclose(own["t"], _deadlines(rec) - t0, atol=1e-3)
        with pytest.raises(KeyError):
            mode.strip("nope")
    finally:
        app.close()


def test_executor_strip_without_a_command_stream_is_empty(tmp_path):
    rec = fx.write(tmp_path / "r")
    mode = ex.ExecutorMode()
    app = _mounted(rec, mode)
    try:
        d = mode.strip("")
        assert d["has_command"] is False and d["err_deg"] == []
    finally:
        app.close()


# --- replay --------------------------------------------------------------------


def test_action_diff_is_the_flange_distance_per_anchor(tmp_path):
    rec = fx.write(tmp_path / "r")
    ro = loader.load(rec)
    try:
        live = anchors.live_chunks(ro.chunks)
        rep = live.copy()
        rep["pose"][:, :, 1, 2] += 0.005              # right hand up 5 mm
        d = action_diff(rep, live, ChunkPoses(ro.model, ro.state), 100.0)
        assert d["chunk_id"] == list(live["chunk_id"])
        assert np.allclose(d["diff_mm"], 5.0, atol=1e-6)
        assert np.allclose(d["t"], live["t_anchor"] - 100.0, atol=1e-3)
        same = action_diff(live, live, ChunkPoses(ro.model, ro.state), 0.0)
        assert np.allclose(same["diff_mm"], 0.0)
    finally:
        ro.close()


def test_replay_strip_route_serves_a_ready_replay(tmp_path):
    pytest.importorskip("cv2")
    from g1d_deploy.dashboards.inspect.modes.replay import run

    rec = fx.write(tmp_path / "records" / "20260926-120000")
    out = run.replay(rec, log=lambda *_: None)
    mode = replay_mode.ReplayMode()
    app = _mounted(rec, mode)
    try:
        d = mode.strip(out.name)
        assert len(d["t"]) == len(d["diff_mm"]) == fx.N_CHUNKS
        assert all(v >= 0 for v in d["diff_mm"])
        assert mode.strip(store.LIVE_ID)["t"] == []
        with pytest.raises(KeyError):
            mode.strip("nope")
    finally:
        app.close()
