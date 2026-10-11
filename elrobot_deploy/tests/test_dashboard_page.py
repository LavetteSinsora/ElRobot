"""The live dashboard is one page: no view modes, a status line with the
rollout's settings, and nothing on it writes the arena except the page's own
requests. Run against a synthetic arena (`live_fixtures.fake_run`).
"""

from __future__ import annotations

import dataclasses
import os
import urllib.error
import urllib.request

import numpy as np
import pytest

from g1d_deploy.config_loading import resolved as cfgmod
from g1d_deploy.runtime.shared_memory import schema
from g1d_deploy.dashboards.live.scene import LiveScene
from g1d_deploy.dashboards.live.server import STATIC, Live, serve
from g1d_deploy.config_loading.machine_file import load_machine

from .live_fixtures import fake_run


@pytest.fixture
def cfg():
    return cfgmod.DeployConfig(
        prompt="test", checkpoint="", sync=True, replan_period_ticks=3,
        action_horizon=8, control_hz=30.0,
        cameras={"rwrist": cfgmod.CameraSpec(8, 8),
                 "head": cfgmod.CameraSpec(8, 16, layout="side_by_side",
                                           binocular=True)},
        primary_camera="rwrist",
        home_q=np.zeros(schema.ARM_DOF), home_waist=np.zeros(schema.WAIST_DOF),
        home_hand=np.zeros((schema.N_HANDS, schema.HAND_DIM)),
        exit_q=np.zeros(schema.ARM_DOF), exit_waist=np.zeros(schema.WAIST_DOF),
        exit_hand=np.zeros((schema.N_HANDS, schema.HAND_DIM)),
        state_history_s=1.0)


@pytest.fixture
def live(cfg, hw, rings):
    return Live(cfg, hw, rings, scene=None)


def _html() -> str:
    return (STATIC / "index.html").read_text()


# --- one view ------------------------------------------------------------------


def test_there_are_no_view_modes():
    """No picker, no mode stage, no mode scripts, no `dashboard.modes`."""
    html = _html()
    for gone in ('id="sel-mode"', 'id="mode-stage"', 'id="mode-banner"',
                 "view-picker"):
        assert gone not in html, gone
    assert not list(STATIC.glob("mode_*.js"))
    with pytest.raises(ImportError):
        import g1d_deploy.dashboards.live.modes  # noqa: F401
    app = (STATIC / "app.js").read_text()
    assert "/api/modes" not in app and 'from "/static/live.js"' in app


def test_the_page_keeps_the_scene_the_cameras_and_the_controls():
    html = _html()
    for needle in ('<canvas id="gl">', 'id="cameras"', 'id="btn-start"',
                   'id="btn-reset"', 'id="btn-exit"', 'id="prompt-form"',
                   'id="btn-settings"', 'id="chk-ghost"', 'id="chk-solved"'):
        assert needle in html, needle
    # The stage is on screen from the start (no mode unhides it).
    assert '<main id="stage">' in html


def test_joint_mode_still_greys_the_commanded_pose():
    js = (STATIC / "live.js").read_text()
    assert "head.ghost_absent" in js and '"inactive"' in js


def test_the_machine_file_has_no_dashboard_modes_key(tmp_path):
    p = tmp_path / "machine.toml"
    p.write_text('[network]\niface = "lo"\n[dashboard]\nport = 8099\n')
    m = load_machine(p)
    assert m.dashboard_port == 8099
    assert not hasattr(m, "dashboard_modes")
    assert "dashboard_modes" not in {
        f.name for f in dataclasses.fields(cfgmod.Hardware)}


# --- The status line -----------------------------------------------------------


def test_the_status_line_labels():
    """No rollout number; `P` is "replan period", `schedule` is "inference
    mode" — in the status line and in the Deploy settings panel."""
    html = _html()
    line = html[html.index('id="statusline"'):html.index("</div>",
                                                        html.index('id="statusline"'))]
    labels = [s.split("</i>")[0] for s in line.split("<i>")[1:]]
    assert labels == ["phase", "ik", "inference mode", "replan period", "rate",
                      "prompt"]
    assert 'id="sl-gen"' not in html
    panel = html[html.index('id="settings-panel"'):]
    assert "<th>inference mode</th>" in panel
    assert "<th>replan period <small" in panel
    assert "replan period P" not in panel
    assert "sl-gen" not in (STATIC / "app.js").read_text()


def test_the_banner_carries_the_rollouts_settings(live, cfg, hw, rings):
    fake_run(rings, cfg, hw, generation=4)
    b = live.banner()
    assert b["prompt"] == "pick up the cube"
    r = b["rollout"]
    assert r["ik"] == cfg.ik and r["ik_mode"] == "arm"
    assert r["sync"] is True and r["replan_period"] == 3
    assert r["control_hz"] == pytest.approx(30.0)


# --- nothing is written --------------------------------------------------------


def test_no_route_writes_to_the_arena(cfg, hw, rings, arena):
    """Every GET, over HTTP, against a byte copy of the whole arena; the old
    mode routes are gone (404), GET or POST."""
    fake_run(rings, cfg, hw)
    live = Live(cfg, hw, rings, scene=LiveScene(cfg, hw, rings))
    server = serve(live, host="127.0.0.1", port=0)
    try:
        before = bytes(arena.shm.buf)
        url = f"http://127.0.0.1:{server.server_address[1]}"
        for route in ("/", "/api/banner", "/api/settings", "/api/frame",
                      "/api/meta", "/static/live.js", "/static/strip.js"):
            with urllib.request.urlopen(f"{url}{route}") as r:
                assert r.status == 200, route
        for route in ("/api/modes", "/api/ik", "/api/servo", "/api/timing",
                      "/api/cameras"):
            with pytest.raises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(f"{url}{route}")
            assert e.value.code == 404, route
        req = urllib.request.Request(f"{url}/api/ik", data=b"{}", method="POST")
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req)
        assert e.value.code == 404
        assert bytes(arena.shm.buf) == before
    finally:
        server.stopping.set()
        server.shutdown()
        server.server_close()


def test_a_port_in_use_is_the_bind_error(live):
    first = serve(live, host="127.0.0.1", port=0)
    try:
        with pytest.raises(OSError):
            serve(live, host="127.0.0.1", port=first.server_address[1])
    finally:
        first.stopping.set()
        first.shutdown()
        first.server_close()


def test_fake_arena_is_the_suites_own(arena):
    """The fixture arena is a fresh segment with the test's pid: a deploy run
    on this box (`/dev/shm/g1d-<pid>`) is never touched."""
    assert arena.shm.name.lstrip("/").startswith(f"g1d-t{os.getpid()}-")
