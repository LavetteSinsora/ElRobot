"""Modes are pluggable; the built-ins ship with the infra.

An extra mode is `pkg.module:Class` from `--mode` or the machine file's
`[inspect] modes`; it mounts routes, may serve its own static directory, and a
broken one is reported and left out. The executor mode's producer is plugged
in the same way, by module path, and without one the mode still lists what
exists.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from g1d_deploy.dashboards.inspect import options as options_mod
from g1d_deploy.dashboards.inspect import modes as modes_mod
from g1d_deploy.dashboards.inspect import server
from g1d_deploy.dashboards.inspect.modes import executor as ex

from . import inspect_fixtures as fx

_PLUGIN = '''
from pathlib import Path
from g1d_deploy.dashboards.inspect.modes import describe


class Hello:
    name = "hello"
    title = "hello"
    blurb = "a plug-in"
    static_dir = Path(__file__).parent / "hello_static"

    def __init__(self, options):
        self.options = options
        self.loaded = []

    def mount(self, app, state):
        state.on_load(lambda loaded, d: self.loaded.append(d))

        @app.get("/api/hello")
        def hello(req):
            req.json({"root": str(self.options.root)})

    def describe(self):
        return describe(self)


class Broken:
    def __init__(self):
        raise RuntimeError("nope")
'''


@pytest.fixture
def plugin(tmp_path, monkeypatch):
    pkg = tmp_path / "plug"
    (pkg / "hello_static").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "hello.py").write_text(textwrap.dedent(_PLUGIN))
    (pkg / "hello_static" / "mode_hello.js").write_text("export function activate() {}")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield "plug.hello"
    for name in [n for n in sys.modules if n.startswith("plug")]:
        del sys.modules[name]


class _Req:
    def __init__(self, path):
        self.path, self.query, self.replied, self.sent = path, {}, False, None

    def json(self, obj, code=200):
        self.sent, self.replied = (code, obj), True

    def bytes(self, payload, ctype, code=200, extra=None):
        self.sent, self.replied = (code, payload), True

    def error(self, exc, code=400):
        self.json({"error": str(exc)}, code)


def test_the_builtins_are_ik_and_replay_and_executor_is_opt_in(tmp_path):
    assert list(modes_mod.BUILTIN) == ["ik", "replay"]
    assert list(modes_mod.OPTIONAL) == ["executor"]
    for spec in [*modes_mod.BUILTIN.values(), *modes_mod.OPTIONAL.values()]:
        assert spec.startswith("g1d_deploy.dashboards.inspect.")
    for opts in (options_mod.Options(root=tmp_path, modes=["executor"]),
                 options_mod.Options(root=tmp_path, executor_producer="pkg.sim")):
        req = _Req("/api/modes")
        server.build_app(opts).dispatch("GET", req)
        assert [m["name"] for m in req.sent[1]] == ["ik", "replay", "executor"]


def test_an_extra_mode_mounts_by_module_path(tmp_path, plugin):
    opts = options_mod.Options(root=tmp_path, modes=[f"{plugin}:Hello",
                                                     f"{plugin}:Broken"])
    app = server.build_app(opts)
    names = []
    req = _Req("/api/modes")
    app.dispatch("GET", req)
    names = [m["name"] for m in req.sent[1]]
    # Built-ins first, then the plug-in; the broken one is left out.
    assert names == ["ik", "replay", "hello"]
    req = _Req("/api/hello")
    app.dispatch("GET", req)
    assert req.sent == (200, {"root": str(tmp_path)})
    # Its own static directory is served, ahead of the built-in files.
    req = _Req("/static/mode_hello.js")
    app.dispatch("GET", req)
    assert req.sent[0] == 200 and b"activate" in req.sent[1]
    app.close()


def test_modes_come_from_the_machine_file_and_the_flags(tmp_path, monkeypatch):
    machine = tmp_path / "machine.toml"
    machine.write_text(textwrap.dedent('''
        [inspect]
        modes = ["a.b:C"]
        executor_producer = "my.simrig"
        runs = ["/data/runs"]
        port = 9999
    '''))
    monkeypatch.delenv("G1D_MACHINE", raising=False)
    o = options_mod.resolve(tmp_path, machine=str(machine), modes=["d.e:F", "a.b:C"],
                            runs=["/x"])
    assert o.modes == ["a.b:C", "d.e:F"]
    assert o.executor_producer == "my.simrig"
    assert [str(r) for r in o.runs] == ["/data/runs", "/x"]
    assert o.port == 9999
    o = options_mod.resolve(tmp_path, machine=str(machine), port=1234,
                            executor_producer="other.mod")
    assert o.port == 1234 and o.executor_producer == "other.mod"


def test_the_executor_mode_without_a_producer_lists_and_explains(tmp_path):
    """Replays of the rollout are found by contract (a record whose labels
    name it, or one under its `executor_replays/`); running new ones needs a
    producer, and the mode says how to plug one in."""
    pytest.importorskip("mujoco")
    root = tmp_path / "records"
    rec = fx.write(root / "20260926-120000")
    fx.write(rec / ex.REPLAYS_DIR / "20260926-130000")
    fx.write(root / "replays" / "20260926-140000",
             manifest_edit=lambda m: m["deploy_config"].__setitem__(
                 "labels", {"source_name": "20260926-120000", "variant": "sim"}))
    fx.write(root / "other" / "20260926-150000")

    found = ex.replays_of(rec, root)
    assert [p.name for p in found] == ["20260926-130000", "20260926-140000"]

    mode = ex.ExecutorMode(options_mod.Options(root=root))
    app = server.build_app(options_mod.Options(root=root), modes=[])
    mode.mount(app, app.state)
    app.state.load("20260926-120000", 30.0)
    out = mode.replays()
    assert out["producer"] == "" and "executor_producer" in out["note"]
    assert {r["name"] for r in out["replays"]} == {"20260926-130000",
                                                    "20260926-140000"}
    with pytest.raises(RuntimeError, match="no executor producer"):
        mode.replay({"variants": ["x"]})
    app.close()


def test_the_executor_producer_is_spawned_by_module_path(tmp_path, monkeypatch):
    pytest.importorskip("mujoco")
    rec = fx.write(tmp_path / "20260926-120000")
    seen = {}

    def fake_submit(**kw):
        seen.update(kw)

        class _Job:
            def as_json(self):
                return {"id": "j"}
        return _Job()

    monkeypatch.setattr(ex.RUNNER, "submit", fake_submit)
    monkeypatch.setattr(ex.RUNNER, "busy", lambda: False)
    opts = options_mod.Options(root=rec, executor_producer="my.simrig")
    mode = ex.ExecutorMode(opts)
    app = server.build_app(opts, modes=[])
    mode.mount(app, app.state)
    app.state.load(".", 30.0)
    mode.replay({"variants": ["linear", "shaped"]})
    assert seen["cmd"][:3] == [sys.executable, "-m", "my.simrig"]
    assert seen["cmd"][seen["cmd"].index("--out") + 1] == str(rec / ex.REPLAYS_DIR)
    app.close()


def test_both_dashboards_pose_with_the_shared_robot_scene():
    """`dashboards/web/robot_scene.py` is the one model and joint map both the
    live dashboard and inspection pose with; neither imports the other."""
    pytest.importorskip("mujoco")
    from g1d_deploy.dashboards.inspect import scene as inspect_scene
    from g1d_deploy.dashboards.live import scene as live_scene
    from g1d_deploy.dashboards.web import robot_scene

    model, data = robot_scene.build_model()
    assert model.ngeom > 0
    assert live_scene._JointMap is robot_scene._JointMap is inspect_scene._JointMap
    assert np.allclose(robot_scene.tcp_point(object()),
                       [robot_scene.FLANGE_TO_TCP_X, 0, 0])
    assert "dashboards.inspect" not in Path(live_scene.__file__).read_text()
