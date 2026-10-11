"""Policy replay: a rollout's own observations, re-run through a policy.

Exercised with the model-free example policies (`examples/policies/sine_eef.py`,
`examples/policies/sine_joint.py`) and small policies written here, so nothing touches a
GPU: the replay path is the same whatever the class does inside `infer`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from g1d_deploy.dashboards.web import gpu
from g1d_deploy.dashboards.inspect import loader, scene, timeline
from g1d_deploy.dashboards.inspect.modes import replay as replay_mode
from g1d_deploy.dashboards.inspect.modes.replay import anchors, browse, run, store
from g1d_deploy.dashboards.inspect.modes.replay.overlay import OverlayBuilder

from . import inspect_fixtures as fx

pytest.importorskip("mujoco")
pytest.importorskip("cv2")

SRC = Path(__file__).resolve().parents[1] / "src"

#: A policy that shows what reached it: the flange raised by `sample_steps`
#: millimetres (so `on_settings` is visible in the actions), refusing prompts
#: longer than two words (so `check_prompt_length` is exercised), and reading
#: a second of head history (so the record-backed reader is).
_PROBE = '''
import numpy as np
from g1d_deploy.policy_contract import EefActions, Policy as _Base


class Policy(_Base):
    action_mode = "eef"
    end_effector = "dex1"
    anchor_camera = "head_left"
    history = {"head_left": 1.0, "state": 0.5}
    max_prompt_tokens = 2
    seen = []

    def __init__(self, checkpoint, prompt, robot):
        super().__init__(checkpoint, prompt, robot)
        self.steps = 10

    def count_prompt_tokens(self, text):
        return len(text.split())

    def warmup(self):
        pass

    def on_settings(self, settings):
        self.steps = settings.sample_steps

    def infer(self, obs, reader):
        t = obs.t_anchor
        ids = reader.frame_ids_between("head_left", t - 1.0, t)
        states = reader.states_between(t - 0.5, t)
        frames = reader.frames_by_id("head_left", ids.ids[-1:])
        Policy.seen.append((len(ids.ids), len(states.t_recv),
                            float(ids.t_recv.max()) if len(ids.ids) else None,
                            frames[0].pixels.shape if frames else None,
                            self.prompt, str(self.checkpoint)))
        pose = np.repeat(obs.state.flange[None], 50, axis=0)
        pose[:, :, 2] += self.steps * 1e-3
        return EefActions(pose=pose,
                          hand=np.repeat(obs.state.hand[None], 50, axis=0))
'''


@pytest.fixture
def record(tmp_path):
    return fx.write(tmp_path / "records" / "exp" / "20260926-120000")


@pytest.fixture
def probe(tmp_path):
    path = tmp_path / "probe_policy.py"
    path.write_text(textwrap.dedent(_PROBE))
    from g1d_deploy.policy_contract import loading as entry
    cls = entry.load_entry(f"{path}:Policy")
    cls.seen.clear()
    return f"{path}:Policy", cls


def _manifest(d):
    return json.loads((Path(d) / "manifest.json").read_text())


def test_a_replay_of_toy_eef_lands_beside_the_record(record):
    """Every default is the rollout's own: the recorded policy entry, its
    checkpoint, its prompt and its settings."""
    out = run.replay(record, log=lambda *_: None)
    assert out.parent == record / store.DIR
    m = _manifest(out)
    assert m["schema"] == store.SCHEMA
    rec = _manifest(record)
    assert m["entry"] == rec["policy"]["entry"]
    assert m["prompt"] == rec["rollouts"][0]["prompt"]
    assert m["settings"] == rec["rollouts"][0]["active_config"]
    assert m["n_chunks"] == fx.N_CHUNKS and m["action_mode"] == "eef"
    assert m["anchors"]["primary"] == "head"
    assert m["anchors"]["primary_exact"] == fx.N_CHUNKS

    # Row for row with the live replans: same ids, same first ticks, the
    # whole chunk the policy returned, `n_actions` what would have run.
    chunks = store.load_chunks(out)
    live = anchors.live_chunks(loader.load(record).chunks)
    assert list(chunks["chunk_id"]) == list(live["chunk_id"])
    assert list(chunks["k_first_action"]) == list(live["k_first_action"])
    assert (chunks["n_actions"] == 25).all() and (chunks["n_plan"] == 50).all()
    assert np.isfinite(chunks["pose"]).all()

    rows = store.list_replays(record)
    assert [r.id for r in rows] == [out.name] and rows[0].state == "ready"
    assert rows[0].summary["rows_compared"] == 25 * fx.N_CHUNKS


def test_repeats_are_separate_replays(record):
    """The spread between repeats of one setting is the sampler's noise
    floor, so each is its own directory and its own switch."""
    a = run.replay(record, log=lambda *_: None)
    b = run.replay(record, log=lambda *_: None)
    assert a != b
    rows = store.list_replays(record)
    assert len(rows) == 2 and rows[0].label != rows[1].label


def test_sample_steps_reach_the_policy_through_on_settings(record, probe):
    entry, cls = probe
    base = run.replay(record, entry=entry, log=lambda *_: None)
    five = run.replay(record, entry=entry, settings={"sample_steps": 5},
                      log=lambda *_: None)
    assert _manifest(five)["settings"]["sample_steps"] == 5
    assert _manifest(base)["settings"]["sample_steps"] == 10
    # Row 0 of a shifted chunk is blended with the measured flange (the
    # deadline shift's grid starts at the anchor state), so compare after it.
    dz = store.load_chunks(five)["pose"][:, 1:25, :, 2] \
        - store.load_chunks(base)["pose"][:, 1:25, :, 2]
    assert np.allclose(dz, -5e-3, atol=1e-6)
    labels = {r.id: r.label for r in store.list_replays(record)}
    assert "sample steps 5" in labels[five.name]


def test_the_reader_serves_history_from_the_record_and_never_the_future(record, probe):
    entry, cls = probe
    run.replay(record, entry=entry, log=lambda *_: None)
    replans = anchors.live_chunks(loader.load(record).chunks)
    assert len(cls.seen) == 1 + len(replans)        # the START dummy, then one each
    for (n_frames, n_state, t_last, shape, _p, _c), c in zip(cls.seen[1:], replans):
        assert n_frames >= 1 and n_state >= 1
        assert t_last <= float(c["t_obs_received"])
        # The head is recorded whole; the view is its left half.
        assert shape == (480, 640, 3)


def test_a_prompt_the_policy_cannot_fit_is_refused_with_the_dashboards_sentence(
        record, probe):
    entry, _ = probe
    with pytest.raises(Exception, match=r"maximum token length \(3 tokens > 2\)"):
        run.replay(record, entry=entry, prompt="one two three", log=lambda *_: None)
    rows = store.list_replays(record)
    assert rows[0].state == "broken" and "maximum token length" in rows[0].note


def test_a_run_and_step_resolve_by_the_numbered_subdirectory_rule(record, probe, tmp_path):
    entry, cls = probe
    runs = tmp_path / "runs" / "my_run"
    for step in (100, 250):
        (runs / str(step)).mkdir(parents=True)
    out = run.replay(record, entry=entry, run=str(runs), step="latest",
                     log=lambda *_: None)
    assert _manifest(out)["checkpoint"] == str(runs / "250")
    assert cls.seen[-1][-1] == str(runs / "250")


def test_a_joint_replay_is_drawn_through_forward_kinematics(tmp_path):
    rec = fx.write(tmp_path / "j", example="sine_joint")
    out = run.replay(rec, log=lambda *_: None)
    m = _manifest(out)
    assert m["action_mode"] == "joint"
    chunks = store.load_chunks(out)
    assert (chunks["mode"] == 1).all() and np.isnan(chunks["pose"]).all()
    ro = loader.load(rec)
    tl = timeline.build(ro)
    built = OverlayBuilder(ro, tl, scene.build(ro, tl)).build(
        chunks, anchors.live_chunks(ro.chunks))
    assert built["meta"]["frames_valid"] > 0
    # The toy bends each elbow 0.15 rad on a sinusoid; the live chunks are the
    # home pose, so the replay's TCP sits a few centimetres off it.
    assert 5.0 < built["meta"]["err_mm_p99"] < 150.0


def test_settings_that_do_not_change_the_policy_are_not_replayable(record):
    mode = _mode(record)
    assert mode.check_settings({"sample_steps": 4})["sample_steps"] == 4
    with pytest.raises(ValueError, match="recorded replans"):
        mode.check_settings({"replan_period": 10})


def _mode(record_dir):
    """The mode mounted on a dashboard served on one record, loaded."""
    from g1d_deploy.dashboards.inspect.options import Options
    from g1d_deploy.dashboards.inspect import server

    opts = Options(root=Path(record_dir),
                   state_file=Path(record_dir).parent / "state.json")
    app = server.build_app(opts, modes=[])
    mode = replay_mode.ReplayMode(opts)
    mode.mount(app, app.state)
    app.state.load(".", 30.0)
    return mode


def test_the_mode_defaults_to_what_the_rollout_ran(record):
    d = _mode(record).defaults()
    rec = _manifest(record)
    assert d["entry"] == rec["policy"]["entry"]
    assert d["prompt"] == rec["rollouts"][0]["prompt"]
    assert d["settings"] == rec["rollouts"][0]["active_config"]
    assert d["editable"] == ["sample_steps"]


def test_the_mode_spawns_the_replay_command(record, monkeypatch):
    seen = []

    def fake_submit(**kw):
        seen.append(kw)

        class _Job:
            def as_json(self):
                return {"id": str(len(seen))}
        return _Job()

    monkeypatch.setattr(replay_mode.RUNNER, "submit", fake_submit)
    monkeypatch.setattr(replay_mode.gpu, "refusal", lambda: "")
    mode = _mode(record)
    out = mode.start({"settings": {"sample_steps": 3}, "repeats": 2,
                      "prompt": "Wave twice."})
    assert len(out["jobs"]) == 2 and len(seen) == 2
    cmd = seen[0]["cmd"]
    assert cmd[:3] == [sys.executable, "-m", replay_mode.PRODUCER]
    assert cmd[cmd.index("--record") + 1] == str(record)
    assert json.loads(cmd[cmd.index("--settings") + 1]) == {"sample_steps": 3}
    assert cmd[cmd.index("--prompt") + 1] == "Wave twice."
    assert seen[0]["jobs_dir"] == record / store.DIR / "_jobs"


def test_a_replay_is_refused_while_a_live_deployment_holds_the_gpu(record, tmp_path,
                                                                  monkeypatch):
    shm = tmp_path / "shm"
    shm.mkdir()
    (shm / f"{gpu.PREFIX}{os.getpid()}").write_bytes(b"")    # a live P0: us
    (shm / f"{gpu.PREFIX}999999999").write_bytes(b"")        # a dead one
    live = gpu.live_deployments(shm)
    assert [d["pid"] for d in live] == [os.getpid()]
    monkeypatch.setattr(gpu, "SHM", shm)
    mode = _mode(record)
    with pytest.raises(RuntimeError, match="live deployment"):
        mode.start({})
    assert "live deployment" in mode.defaults()["gpu"]


def test_the_run_picker_lists_directories_and_steps(tmp_path):
    runs = tmp_path / "runs" / "r1"
    for step in (10, 2000):
        (runs / str(step)).mkdir(parents=True)
    (runs / "norm_stats").mkdir()
    d = browse.listing(runs)
    assert [s["step"] for s in d["steps"]] == [10, 2000]
    assert [c["name"] for c in d["dirs"]] == ["norm_stats"]
    assert d["parent"] == str(runs.parent)
    with pytest.raises(ValueError):
        browse.listing("relative/path")

    state = tmp_path / "state.json"
    browse.remember(state, str(runs))
    browse.remember(state, str(tmp_path))
    browse.remember(state, str(runs))
    assert browse.recent(state) == [str(runs), str(tmp_path)]


def test_the_cli_replays_and_describes(record):
    """`--ignore-live`: the toy policy takes no GPU, and a deployment may be
    running on the box the suite runs on (the refusal has its own test)."""
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    base = [sys.executable, "-m", "g1d_deploy", "replay-policy", "--record",
            str(record)]
    desc = subprocess.run(base + ["--describe-entry"], capture_output=True,
                          text=True, env=env, timeout=120)
    assert desc.returncode == 0, desc.stderr
    d = json.loads(desc.stdout.strip().splitlines()[-1])
    assert d["action_mode"] == "eef" and d["resolves_checkpoints"] is False

    done = subprocess.run(base + ["--sample-steps", "7", "--ignore-live"],
                          capture_output=True,
                          text=True, env=env, timeout=300)
    assert done.returncode == 0, done.stderr
    rows = store.list_replays(record)
    assert rows and rows[-1].settings["sample_steps"] == 7
