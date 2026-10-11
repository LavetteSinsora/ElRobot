"""P5 under the policy's own interpreter (`[policy] python`).

What is tested here, end to end, with real processes:

* `codec` carries `DeployConfig` and `Hardware` across an interpreter
  boundary bit-exactly, without pickle;
* `policy_contract.describe` answers a class's declarations from another
  interpreter, and `deploy_file.resolve` reads them through it;
* P0's spawn path (`_spawn_process` with `interpreter=`) runs P5 in another
  interpreter against an arena the runtime created: it attaches (the layout
  hash agrees), warms up, and on ACTIVE publishes a chunk the runtime reads;
* **P5's import graph is the minimal core**: the child runs with an import
  hook (a `sitecustomize` on its PYTHONPATH) that refuses every heavy
  dependency outright and records every module the process loaded. What it
  loaded must be the stdlib, numpy, and the `CORE` modules of g1d_deploy —
  nothing else.

The same tests run against a SECOND venv when `G1D_TEST_POLICY_PYTHON` names
its interpreter — one holding numpy 2 and only g1d_deploy's minimal core
(`pip install --no-deps -e g1d_deploy` + `numpy>=2`), while this runtime runs
numpy 1.x. That is the case the feature exists for.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
import time
import uuid

import numpy as np
import pytest

from g1d_deploy.runtime import timing as clock
from g1d_deploy.config_loading import json_transport as codec, resolved as cfgmod
from g1d_deploy.config_loading.derive import derive
from g1d_deploy.config_loading.errors import ConfigError
from g1d_deploy.config_loading.validate import validate
from g1d_deploy.runtime.shared_memory import Arena, build, open_all, schema
from g1d_deploy.runtime.launcher import children

#: The minimal core: every g1d_deploy module P5 (and `policy_contract.describe`) may
#: load. A policy venv installs g1d_deploy with NO extras, and these import
#: nothing but numpy and the stdlib.
CORE = {
    "g1d_deploy",
    "g1d_deploy.config_loading",
    "g1d_deploy.config_loading.errors",
    "g1d_deploy.config_loading.json_transport",
    "g1d_deploy.config_loading.resolved",
    "g1d_deploy.config_loading.rollout_settings",
    "g1d_deploy.policy_contract",
    "g1d_deploy.policy_contract.actions",
    "g1d_deploy.policy_contract.anchor",
    "g1d_deploy.policy_contract.base",
    "g1d_deploy.policy_contract.deadline_shift",
    "g1d_deploy.policy_contract.describe",
    "g1d_deploy.policy_contract.loading",
    "g1d_deploy.policy_contract.observation",
    "g1d_deploy.robot",
    "g1d_deploy.robot.end_effectors",
    "g1d_deploy.robot.kinematics",
    "g1d_deploy.runtime",
    "g1d_deploy.runtime.cpu_pinning",
    "g1d_deploy.runtime.ik",
    "g1d_deploy.runtime.processes",
    "g1d_deploy.runtime.processes.p5_policy",
    "g1d_deploy.runtime.processes.policy_host",
    "g1d_deploy.runtime.shared_memory",
    "g1d_deploy.runtime.shared_memory.layout",
    "g1d_deploy.runtime.shared_memory.primitives",
    "g1d_deploy.runtime.shared_memory.rings",
    "g1d_deploy.runtime.shared_memory.schema",
    "g1d_deploy.runtime.teleop.intervention",
    "g1d_deploy.runtime.timing",
}

#: Refused outright in the child: importing one fails loudly at the import.
FORBIDDEN = ("mujoco", "pinocchio", "pin", "cv2", "cyclonedds",
             "unitree_sdk2py", "scipy", "h5py", "av", "pyarrow", "pandas",
             "mink", "qpsolvers", "PIL", "zmq", "pyrealsense2", "paramiko",
             "yaml", "torch", "jax", "glfw", "daqp")

_HOOK = textwrap.dedent('''
    import atexit, json, os, sys
    _BASE = set(sys.modules)
    _FORBIDDEN = set(json.loads(os.environ["G1D_FORBID"]))

    class _Refuse:
        def find_spec(self, name, path=None, target=None):
            if name.partition(".")[0] in _FORBIDDEN:
                raise ImportError(f"forbidden in the minimal core: {name}")
            return None

    sys.meta_path.insert(0, _Refuse())

    def _dump():
        with open(os.environ["G1D_IMPORTS_OUT"], "w") as f:
            json.dump({"new": sorted(set(sys.modules) - _BASE),
                       "stdlib": sorted(sys.stdlib_module_names)}, f)

    atexit.register(_dump)
''')

POLICY_SRC = textwrap.dedent('''
    import numpy as np
    from g1d_deploy.policy_contract import EefActions, Policy

    class Hold(Policy):
        action_mode = "eef"
        end_effector = "dex1"
        anchor_camera = "rwrist"
        trained_control_hz = 30.0
        trained_action_horizon = 8
        history = {"state": 0.05}
        max_prompt_tokens = 12

        def infer(self, obs, reader):
            n = 8
            return EefActions(pose=np.repeat(obs.state.flange[None], n, 0),
                              hand=np.zeros((n, 2)))
''')


def _interpreters():
    out = [pytest.param(sys.executable, id="runtime-python")]
    other = os.environ.get("G1D_TEST_POLICY_PYTHON")
    out.append(pytest.param(other, id="policy-venv", marks=pytest.mark.skipif(
        not other, reason="G1D_TEST_POLICY_PYTHON names no second venv")))
    return out


def _wrapper(tmp_path, python):
    """`python` behind a DIFFERENT path, so `same_interpreter` says no and
    the subprocess path is taken even for this venv's own interpreter."""
    w = tmp_path / "policy-python"
    w.write_text(f"#!/bin/sh\nexec {python} \"$@\"\n")
    w.chmod(0o755)
    return str(w)


def _hooked_env(tmp_path, monkeypatch):
    hook = tmp_path / "hook"
    hook.mkdir()
    (hook / "sitecustomize.py").write_text(_HOOK)
    out = tmp_path / "imports.json"
    src = os.path.join(os.path.dirname(cfgmod.__file__), os.pardir)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(
        [str(hook), os.path.abspath(src)]
        + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]))
    monkeypatch.setenv("G1D_FORBID", json.dumps(FORBIDDEN))
    monkeypatch.setenv("G1D_IMPORTS_OUT", str(out))
    return out


def _check_imports(out_file, *, extra=(), core=True):
    got = json.loads(out_file.read_text())
    std = set(got["stdlib"])
    top = {m.partition(".")[0] for m in got["new"]}
    # `cython_runtime` is created by numpy's own compiled extensions.
    foreign = sorted(t for t in top if t not in std and not t.startswith("_")
                     and t not in {"numpy", "cython_runtime", "g1d_deploy", *extra})
    assert not foreign, f"P5 loaded modules outside the minimal core: {foreign}"
    ours = {m for m in got["new"] if m.partition(".")[0] == "g1d_deploy"}
    assert not core or ours <= CORE, f"g1d_deploy modules outside CORE: {sorted(ours - CORE)}"


# --- codec -----------------------------------------------------------------


def _same(a, b):
    if isinstance(a, np.ndarray):
        return (isinstance(b, np.ndarray) and a.dtype == b.dtype
                and a.shape == b.shape and np.array_equal(a, b, equal_nan=True))
    return type(a) is type(b) and a == b


def test_codec_carries_the_configs_exactly(cfg, hw):
    import dataclasses
    for obj in (cfg, hw):
        back = codec.loads(codec.dumps(obj))
        assert type(back) is type(obj)
        for f in dataclasses.fields(obj):
            assert _same(getattr(obj, f.name), getattr(back, f.name)), f.name


def test_codec_refuses_what_it_cannot_carry():
    with pytest.raises(TypeError):
        codec.dumps({"x": object()})
    with pytest.raises(TypeError):
        codec.dumps({1: 2})
    bad = json.dumps({"__g1d__": "dataclass", "cls": "os:PathLike", "fields": {}})
    with pytest.raises(ValueError):
        codec.loads(bad)


def test_codec_keeps_non_finite_floats_and_scalars():
    tree = {"a": np.array([1.5, np.inf, np.nan]), "b": float("-inf"),
            "c": np.float32(2.5), "d": (1, [2, (3,)])}
    back = codec.loads(codec.dumps(tree))
    assert _same(tree["a"], back["a"])
    assert back["b"] == float("-inf") and back["d"] == (1, [2, (3,)])
    assert type(back["c"]) is np.float32 and back["c"] == 2.5


# --- describe ------------------------------------------------------------------


@pytest.mark.parametrize("python", _interpreters())
def test_describe_answers_from_the_policy_interpreter(tmp_path, monkeypatch,
                                                      python):
    from g1d_deploy.policy_contract.describe import describe
    (tmp_path / "pol.py").write_text(POLICY_SRC)
    out = _hooked_env(tmp_path, monkeypatch)
    info = describe(f"{tmp_path / 'pol.py'}:Hold",
                    python=_wrapper(tmp_path, python))
    assert info.is_policy and info.invalid is None
    assert (info.action_mode, info.anchor_camera, info.end_effector) == (
        "eef", "rwrist", "dex1")
    assert info.history == {"state": 0.05} and info.max_prompt_tokens == 12
    assert info.trained_action_horizon == 8 and info.trained_control_hz == 30.0
    assert info.schema_version == schema.VERSION
    _check_imports(out, extra=("pol",))


def test_describe_reports_the_childs_error(tmp_path):
    from g1d_deploy.policy_contract.describe import DescribeError, describe
    (tmp_path / "bad.py").write_text("raise RuntimeError('no torch here')\n")
    with pytest.raises(DescribeError, match="no torch here"):
        describe(f"{tmp_path / 'bad.py'}:X",
                 python=_wrapper(tmp_path, sys.executable))


def test_describe_says_how_to_install_the_core(tmp_path, monkeypatch):
    from g1d_deploy.policy_contract.describe import DescribeError, describe
    monkeypatch.setenv("PYTHONPATH", "")
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    # The system interpreter, isolated from every site-packages: no g1d_deploy.
    bare = tmp_path / "bare-python"
    bare.write_text(f"#!/bin/sh\nexec {sys.executable} -S \"$@\"\n")
    bare.chmod(0o755)
    with pytest.raises(DescribeError, match="minimal core"):
        describe("x.y:Z", python=str(bare))


def test_the_deploy_file_names_the_interpreter_and_resolve_describes_there(
        tmp_path, monkeypatch):
    from g1d_deploy.config_loading import deploy_file as deployfile
    (tmp_path / "pol.py").write_text(POLICY_SRC.replace('"rwrist"', '"head_left"'))
    py = _wrapper(tmp_path, sys.executable)
    tree = {"policy": {"entry": "pol.py:Hold", "python": "./policy-python"},
            "home": {"arm": [0.0] * schema.ARM_DOF},
            "control": {"replan_period": 4}}
    out = _hooked_env(tmp_path, monkeypatch)       # also: PYTHONPATH absolute
    monkeypatch.chdir(tmp_path)
    r = deployfile.resolve(tree=tree, record=False)
    assert r.cfg.policy_python == py
    assert r.cfg.primary_camera == "head_left"
    assert r.cfg.policy_action_mode == "eef"
    assert r.sources["action_horizon"] == "trained" and r.cfg.H == 8
    validate(r.cfg, r.hw)
    _check_imports(out, extra=("pol",))


def test_a_missing_interpreter_is_refused_at_validate(cfg, hw):
    with pytest.raises(ConfigError, match="not an executable"):
        validate(cfgmod.replace(cfg, policy_python="/nonexistent/python"), hw)


# --- P5 in another interpreter, over a runtime-made arena ------------------------


def _feed(rings, stop):
    """What P1 and P2 would publish: frames at 30 Hz, state at 1 kHz."""
    t_frame = 0.0
    while not stop.is_set():
        t = clock.now()
        if t - t_frame >= 1 / 30:
            for cam in rings["cameras"].values():
                cam.spare()["pixels"][:] = 7
                cam.publish(t_recv=t)
            t_frame = t
        rings["state"].write(
            t_recv=t, q=np.zeros(schema.N_JOINTS), dq=np.zeros(schema.N_JOINTS),
            flange=np.array([[0.3, 0.1, 0.2, 1.0, 0.0, 0.0, 0.0]] * schema.N_HANDS),
            grip_meas=np.zeros(schema.GRIP_DIM), t_grip_recv=t)
        time.sleep(0.001)


def _wait(pred, timeout, proc, what):
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        if pred():
            return
        assert proc.is_alive(), f"P5 exited before {what}"
        time.sleep(0.01)
    raise AssertionError(f"P5 never reached {what}")


@pytest.mark.parametrize("python", _interpreters())
def test_p5_runs_in_the_policy_interpreter_against_the_runtimes_arena(
        cfg, hw, tmp_path, monkeypatch, python):
    (tmp_path / "pol.py").write_text(POLICY_SRC)
    cfg = cfgmod.replace(cfg, policy_entry=f"{tmp_path / 'pol.py'}:Hold",
                         policy_python=python)
    out = _hooked_env(tmp_path, monkeypatch)
    layout = build(cfg, derive(cfg, hw))
    name = f"g1d-t{os.getpid()}-{uuid.uuid4().hex[:8]}"
    arena = Arena.create(name, layout)
    stop = threading.Event()
    proc = None
    try:
        rings = open_all(arena, cfg)
        control = rings["control"]
        control.session = schema.WARMUP
        feeder = threading.Thread(target=_feed, args=(rings, stop), daemon=True)
        feeder.start()
        # P0's own spawn, with P0's own argument: `interpreter=`.
        proc = children._spawn_process(
            "p5_policy", "g1d_deploy.runtime.processes.p5_policy.run",
            (name, cfg, hw), {"interpreter": cfg.policy_python})
        assert isinstance(proc, children._InterpreterChild)
        _wait(lambda: control.warm, 60, proc, "warm")

        control.session = schema.ACTIVE
        _wait(lambda: control.newest_chunk_id >= 1, 30, proc, "its first chunk")
        got = rings["chunks"].covering(2)
        assert got and got[0].chunk_id == 1
        np.testing.assert_allclose(got[0].pose[0, :3], [0.3, 0.1, 0.2])

        control.session = schema.EXITING
        proc.join(20)
        assert not proc.is_alive(), "P5 did not leave on EXITING"
        assert proc._proc.returncode == 0
    finally:
        stop.set()
        if proc is not None and proc.is_alive():
            proc.terminate()
            proc.join(5)
        arena.close()
        arena.unlink()
    _check_imports(out, extra=("pol",))


def test_the_layout_and_dtypes_do_not_depend_on_numpy(cfg, hw, tmp_path):
    """The arena is raw bytes, read through dtypes each side builds itself:
    the layout hash (which folds in every region's `dtype.str`, size and
    offset) and every structured dtype's `descr` must agree across numpy
    majors. Asked of the second venv when there is one."""
    other = os.environ.get("G1D_TEST_POLICY_PYTHON")
    if not other:
        pytest.skip("G1D_TEST_POLICY_PYTHON names no second venv")
    script = textwrap.dedent('''
        import json, sys
        import numpy as np
        from g1d_deploy import codec, config
        from g1d_deploy.runtime.shared_memory import build
        cfg, hw = codec.loads(sys.stdin.read())
        lay = build(cfg, config.derive(cfg, hw))
        print(json.dumps({"hash": lay.hash, "size": lay.size,
                          "np": np.__version__,
                          "descr": {n: str(r.dtype.descr) for n, r in lay.regions.items()}}))
    ''')
    proc = subprocess.run([other, "-c", script], input=codec.dumps((cfg, hw)),
                          capture_output=True, text=True, check=True)
    there = json.loads(proc.stdout.strip().splitlines()[-1])
    lay = build(cfg, derive(cfg, hw))
    assert there["hash"] == lay.hash and there["size"] == lay.size
    assert there["descr"] == {n: str(r.dtype.descr) for n, r in lay.regions.items()}


# --- inspection: the replay's light subprocesses, in the policy interpreter ----


_WORDS = textwrap.dedent('''
    from g1d_deploy.policy_contract import Policy

    class Words(Policy):
        action_mode = "eef"
        end_effector = "dex1"
        anchor_camera = "head_left"
        max_prompt_tokens = 3

        def count_prompt_tokens(self, text):
            return len(text.split())

        def infer(self, obs, reader):
            raise AssertionError("never")
''')


@pytest.mark.parametrize("python", _interpreters())
def test_describe_and_check_prompt_run_in_the_policy_interpreter(
        tmp_path, monkeypatch, python):
    """The dashboard's "describe" and "Check prompt" subprocesses, spawned by
    module path (`replay.PRODUCER`) in the policy's interpreter: they need
    nothing beyond the core — no mujoco, no cv2 — so a numpy-2 policy venv
    answers them."""
    pytest.importorskip("cv2")                    # the fixture record's JPEGs
    from g1d_deploy.dashboards.inspect.modes.replay import PRODUCER

    from . import inspect_fixtures as fx
    record = fx.write(tmp_path / "records" / "exp" / "20260926-120000")
    (tmp_path / "words.py").write_text(_WORDS)
    out = _hooked_env(tmp_path, monkeypatch)
    base = [python, "-m", PRODUCER, "--record", str(record),
            "--entry", f"{tmp_path / 'words.py'}:Words"]

    d = subprocess.run(base + ["--describe-entry"], capture_output=True,
                       text=True, timeout=120)
    assert d.returncode == 0, d.stderr
    desc = json.loads(d.stdout.strip().splitlines()[-1])
    assert desc["max_prompt_tokens"] == 3 and desc["anchor_camera"] == "head_left"
    _check_imports(out, extra=("words",), core=False)   # + inspection, numpy-only

    c = subprocess.run(base + ["--checkpoint", "",
                               "--check-prompt=one two three four"],
                       capture_output=True, text=True, timeout=120)
    assert c.returncode == 0, c.stderr
    verdict = json.loads(c.stdout.strip().splitlines()[-1])
    assert verdict["ok"] is False and verdict["n_tokens"] == 4
    got = json.loads(out.read_text())
    assert not {m.partition(".")[0] for m in got["new"]} & set(FORBIDDEN)


def test_a_full_replay_runs_in_a_policy_venv_with_the_replay_extra(tmp_path):
    """A replay decodes JPEGs and runs FK, so the policy venv needs the
    `replay` extra (cv2, mujoco) — which works under numpy 2. Skipped unless
    the second venv has them."""
    other = os.environ.get("G1D_TEST_POLICY_PYTHON")
    if not other:
        pytest.skip("G1D_TEST_POLICY_PYTHON names no second venv")
    has = subprocess.run([other, "-c", "import cv2, mujoco"], capture_output=True)
    if has.returncode != 0:
        pytest.skip("the second venv has no replay extra (cv2, mujoco)")
    pytest.importorskip("cv2")
    from g1d_deploy.dashboards.inspect.modes.replay import PRODUCER, store

    from . import inspect_fixtures as fx
    record = fx.write(tmp_path / "records" / "exp" / "20260926-120000")
    src = os.path.abspath(os.path.join(os.path.dirname(cfgmod.__file__), os.pardir))
    done = subprocess.run([other, "-m", PRODUCER, "--record", str(record),
                           "--ignore-live"], capture_output=True, text=True,
                          timeout=300, env={**os.environ, "PYTHONPATH": src})
    assert done.returncode == 0, done.stderr
    rows = store.list_replays(record)
    assert rows and rows[-1].state == "ready"
