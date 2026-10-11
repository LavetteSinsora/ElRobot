"""Policy replay's "Check prompt": the policy's own tokenizer judges a prompt
without a replay — constructed, never warmed up, never inferred, GPU hidden.

The fake policies count words, so no model and no GPU is involved; their
`warmup` and `infer` raise, so a check that called either would fail.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from elrobot_deploy.dashboards.web import gpu
from elrobot_deploy.dashboards.inspect.modes import replay as replay_mode
from elrobot_deploy.dashboards.inspect.modes.replay import run

from . import inspect_fixtures as fx

pytest.importorskip("mujoco")
pytest.importorskip("cv2")

SRC = Path(__file__).resolve().parents[1] / "src"

_FAKE = '''
import json, os
from pathlib import Path
from elrobot_deploy.policy_contract import Policy as _Base

LOG = Path(__file__).with_suffix(".log")


def _log(**kw):
    with LOG.open("a") as f:
        f.write(json.dumps(kw) + "\\n")


class Words(_Base):
    action_mode = "eef"
    end_effector = "dex1"
    anchor_camera = "head_left"
    max_prompt_tokens = 3

    def __init__(self, checkpoint, prompt, robot):
        super().__init__(checkpoint, prompt, robot)
        _log(event="init", checkpoint=str(checkpoint),
             cuda=os.environ.get("CUDA_VISIBLE_DEVICES"),
             jax=os.environ.get("JAX_PLATFORMS"))

    def count_prompt_tokens(self, text):
        return len(text.split())

    def warmup(self):
        _log(event="warmup")
        raise AssertionError("a prompt check must never warm up")

    def infer(self, obs, reader):
        _log(event="infer")
        raise AssertionError("a prompt check must never infer")


class NoLimit(Words):
    max_prompt_tokens = None
'''


@pytest.fixture
def record(tmp_path):
    return fx.write(tmp_path / "records" / "exp" / "20260926-120000")


@pytest.fixture
def fake(tmp_path):
    path = tmp_path / "words_policy.py"
    path.write_text(textwrap.dedent(_FAKE))
    return path


def _events(fake):
    log = fake.with_suffix(".log")
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines()]


def _cli(record, *extra):
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    return subprocess.run([sys.executable, "-m", "elrobot_deploy", "replay-policy",
                           "--record", str(record), *extra],
                          capture_output=True, text=True, env=env, timeout=120)


def test_the_check_accepts_a_prompt_that_fits(record, fake):
    v = run.check_prompt(record, "pick it up", entry=f"{fake}:Words",
                         checkpoint="/ckpt/1000")
    assert v["ok"] is True
    assert (v["n_tokens"], v["max_tokens"]) == (3, 3)
    assert v["message"] == "OK (3 tokens ≤ 3)"
    assert v["checkpoint"] == "/ckpt/1000"
    events = _events(fake)
    assert [e["event"] for e in events] == ["init"]
    assert events[0]["checkpoint"] == "/ckpt/1000"


def test_the_check_refuses_with_the_dashboards_sentence(record, fake):
    v = run.check_prompt(record, "pick the red one up", entry=f"{fake}:Words")
    assert v["ok"] is False
    assert (v["n_tokens"], v["max_tokens"]) == (5, 3)
    assert v["message"] == ("Prompt exceeds the policy's maximum token length "
                            "(5 tokens > 3).")
    assert [e["event"] for e in _events(fake)] == ["init"]


def test_a_policy_with_no_limit_accepts_anything(record, fake):
    v = run.check_prompt(record, "a b c d e f g", entry=f"{fake}:NoLimit")
    assert v["ok"] is True and v["max_tokens"] is None
    assert v["message"] == "OK (no limit declared)"


def test_the_subcommand_prints_a_verdict_with_the_gpu_hidden(record, fake):
    ok = _cli(record, "--entry", f"{fake}:Words", "--checkpoint", "/ckpt/7",
              "--check-prompt=pick it up")
    assert ok.returncode == 0, ok.stderr
    v = json.loads(ok.stdout.strip().splitlines()[-1])
    assert v["ok"] and v["message"] == "OK (3 tokens ≤ 3)"

    # A prompt that starts with "-" is a prompt, not a flag.
    bad = _cli(record, "--entry", f"{fake}:Words", "--check-prompt=-a b c d")
    assert bad.returncode == 0, bad.stderr
    v = json.loads(bad.stdout.strip().splitlines()[-1])
    assert not v["ok"] and "(4 tokens > 3)" in v["message"]

    events = _events(fake)
    assert [e["event"] for e in events] == ["init", "init"]
    assert all(e["cuda"] == "" and e["jax"] == "cpu" for e in events)


def test_the_subcommand_reports_a_class_that_does_not_load(record, fake):
    out = _cli(record, "--entry", f"{fake}:Missing", "--check-prompt=x")
    assert out.returncode == 1
    assert out.stderr.strip().splitlines()[-1].startswith("REFUSED: ")


def _mode(record_dir):
    from elrobot_deploy.dashboards.inspect.options import Options
    from elrobot_deploy.dashboards.inspect import server

    opts = Options(root=Path(record_dir),
                   state_file=Path(record_dir).parent / "state.json")
    app = server.build_app(opts, modes=[])
    mode = replay_mode.ReplayMode(opts)
    mode.mount(app, app.state)
    app.state.load(".", 30.0)
    return mode


@pytest.fixture
def counted(monkeypatch):
    """The mode's subprocess.run, counted, with PYTHONPATH so the child finds
    this checkout."""
    calls = []
    real = subprocess.run

    def spy(cmd, **kw):
        calls.append(cmd)
        kw["env"] = {**kw.get("env", os.environ), "PYTHONPATH": str(SRC)}
        return real(cmd, **kw)

    monkeypatch.setattr(replay_mode.subprocess, "run", spy)
    return calls


def test_the_mode_checks_and_caches_per_entry_checkpoint_and_prompt(
        record, fake, counted):
    mode = _mode(record)
    req = {"entry": f"{fake}:Words", "prompt": "pick it up", "checkpoint": "/c/1"}
    first = mode.check_prompt(req)
    assert first["ok"] and first["cached"] is False
    assert len(counted) == 1
    cmd = counted[0]
    assert cmd[1:3] == ["-m", replay_mode.PRODUCER]
    assert "--check-prompt=pick it up" in cmd
    assert cmd[cmd.index("--checkpoint") + 1] == "/c/1"

    again = mode.check_prompt(req)
    assert again["cached"] is True and again["message"] == first["message"]
    assert len(counted) == 1

    refused = mode.check_prompt({**req, "prompt": "pick the red one up"})
    assert not refused["ok"] and refused["cached"] is False
    assert refused["message"].startswith("Prompt exceeds the policy's maximum")
    mode.check_prompt({**req, "checkpoint": "/c/2"})
    mode.check_prompt({**req, "entry": f"{fake}:NoLimit"})
    assert len(counted) == 4
    assert all(e["event"] == "init" for e in _events(fake))


def test_the_mode_check_is_not_refused_while_a_deployment_is_live(
        record, fake, counted, tmp_path, monkeypatch):
    shm = tmp_path / "shm"
    shm.mkdir()
    (shm / f"{gpu.PREFIX}{os.getpid()}").write_bytes(b"")    # a live P0: us
    monkeypatch.setattr(gpu, "SHM", shm)
    mode = _mode(record)
    assert "live deployment" in mode.defaults()["gpu"]
    v = mode.check_prompt({"entry": f"{fake}:Words", "prompt": "go"})
    assert v["ok"] and v["message"] == "OK (1 tokens ≤ 3)"


def test_the_endpoint_answers_over_http(record, fake, counted):
    """POST /api/replay/check-prompt through the mounted app."""
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    from elrobot_deploy.dashboards.web import http
    from elrobot_deploy.dashboards.inspect.options import Options
    from elrobot_deploy.dashboards.inspect import server

    opts = Options(root=Path(record), state_file=Path(record).parent / "state.json")
    app = server.build_app(opts, modes=[])
    mode = replay_mode.ReplayMode(opts)
    mode.mount(app, app.state)
    app.state.load(".", 30.0)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), http.make_handler(app))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        port = httpd.server_address[1]
        body = json.dumps({"entry": f"{fake}:Words",
                           "prompt": "one two three four"}).encode()
        r = urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{port}/api/replay/check-prompt", data=body,
            headers={"Content-Type": "application/json"}), timeout=120)
        v = json.loads(r.read())
        assert v["ok"] is False and "(4 tokens > 3)" in v["message"]
    finally:
        httpd.shutdown()


def test_a_check_that_times_out_says_why(record, fake, monkeypatch):
    def slow(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw.get("timeout"))

    monkeypatch.setattr(replay_mode.subprocess, "run", slow)
    with pytest.raises(ValueError, match="timed out after .* __init__ should load"):
        _mode(record).check_prompt({"entry": f"{fake}:Words", "prompt": "go"})
