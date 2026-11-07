"""Offline inspection against small real-format records, including clocks and HTTP isolation."""

import json
import urllib.error
import urllib.request

import numpy as np
import pytest

from elrobot_deploy.inspection.loader import Catalog, Rollout
from elrobot_deploy.inspection.server import serve


@pytest.fixture
def recording(tmp_path):
    root = tmp_path / "records"
    p = root / "example"
    p.mkdir(parents=True)
    m = {"format": "elrobot-record/1", "complete": True, "created": "2026-10-18T12:00:00",
         "config": {}, "rollout": {"t_start": 100.}, "streams": {}, "cameras": {}}
    state = np.zeros(3, dtype=[("t", "f8"), ("q", "f8", (8,)), ("q_cmd", "f8", (8,)),
                              ("load", "f8", (8,)), ("temp", "f8", (8,))])
    state["t"] = [100.01, 100.5, 101.]
    state["q"] = np.arange(3)[:, None] * .1
    state["q_cmd"] = state["q"] + .02
    state["temp"] = np.nan
    targets = np.zeros(1, dtype=[("deadline", "f8"), ("q", "f8", (8,)),
                                ("pose", "f8", (7,)), ("err", "f8")])
    targets["deadline"] = 100.4
    targets["pose"] = [.2, 0, .1, 1, 0, 0, 0]
    targets["err"] = .001
    session = np.array([(5, 1, 100.02), (6, 5, 100.99)],
                       dtype=[("session", "u8"), ("bus", "u8"), ("t_observed", "f8")])
    for name, rows in {"state": state, "targets": targets, "session": session,
                       "intervene": np.zeros(0, dtype=[("t", "f8")])}.items():
        rows.tofile(p / f"{name}.bin")
        m["streams"][name] = {"file": f"{name}.bin", "descr": rows.dtype.descr,
                              "records": len(rows)}
    for name, times in {"exo": [100.1, 100.2, 100.8], "wrist": [100.05, 100.55]}.items():
        frames = [b"\xff\xd8" + name.encode() + bytes([i]) + b"\xff\xd9"
                  for i in range(len(times))]
        ix = np.zeros(len(times), dtype=[("t_capture", "f8"), ("offset", "u8"), ("nbytes", "u8")])
        ix["t_capture"] = times
        ix["offset"] = np.r_[0, np.cumsum([len(x) for x in frames[:-1]])]
        ix["nbytes"] = [len(x) for x in frames]
        (p / f"{name}.jpg.bin").write_bytes(b"".join(frames))
        ix.tofile(p / f"{name}.idx.bin")
        m["cameras"][name] = {"blob": f"{name}.jpg.bin", "index": f"{name}.idx.bin",
                               "index_descr": ix.dtype.descr, "dropped": 0}
    (p / "manifest.json").write_text(json.dumps(m))
    return root, p


def test_sample_uses_recorded_clocks_and_never_future_robot_targets(recording):
    _, p = recording
    r = Rollout(p)
    before = r.sample(0)
    assert before["measured"] is None and before["target"] is None
    sample = r.sample(.49)
    assert sample["state_t"] == pytest.approx(.01)
    assert sample["measured"] == [0.] * 8
    assert sample["position_mm"] == 1
    assert sample["cameras"]["exo"]["index"] == 1
    assert sample["cameras"]["wrist"]["index"] == 1
    assert sample["cameras"]["wrist"]["delta_ms"] == pytest.approx(60)
    assert sample["temp"] == [None] * 8
    assert r.sample(1)["bus"] == "RECOVERING"
    assert r.duration == pytest.approx(1)
    assert not r.data()["model_available"]


def test_episode_actions_are_associated_by_record_path(recording, tmp_path):
    root, p = recording
    episodes = tmp_path / "episodes"
    ep = episodes / "pen"
    action = ep / "actions" / "0001"
    action.mkdir(parents=True)
    (ep / "episode.json").write_text(json.dumps({"task": "Draw a line", "record": str(p),
                                                "clock": {"t0_monotonic": 100.1}}))
    (action / "execution_outcome.json").write_text(json.dumps(
        {"action": 1, "tick_start": 0, "tick_end": 15, "interrupted": "session EXITING"}))
    cat = Catalog(root, episodes)
    record = cat.get("example").data()
    assert record["task"] == "Draw a line"
    assert record["actions"][0]["start"] == pytest.approx(.1)
    assert record["actions"][0]["end"] == pytest.approx(.6)
    assert Catalog(p).records()[0]["id"] == "."
    with pytest.raises(KeyError):
        cat.get("../example")


def test_aborted_opening_with_no_camera_frames_or_episode_clock(recording, tmp_path):
    root, p = recording
    episodes = tmp_path / "episodes"
    ep = episodes / "aborted"
    ep.mkdir(parents=True)
    (ep / "episode.json").write_text(json.dumps({"task": "Opening failed", "record": str(p),
                                                "clock": None, "finish": None}))
    for camera in ("exo", "wrist"):
        (p / f"{camera}.idx.bin").write_bytes(b"")
        (p / f"{camera}.jpg.bin").write_bytes(b"")
    rollout = Catalog(root, episodes).get("example")
    assert rollout.data()["task"] == "Opening failed"
    assert rollout.data()["episode_available"]
    assert rollout.data()["actions"] == []
    assert all(c["frames"] == 0 for c in rollout.data()["cameras"])
    assert rollout.sample(.5)["cameras"] == {"exo": None, "wrist": None}


@pytest.fixture
def inspector(recording):
    root, _ = recording
    server = serve(root, port=0)
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.headers.get("Content-Type"), r.read()


def test_http_record_cameras_and_strict_json(inspector):
    assert b"Rollout inspector" in get(inspector + "/")[1]
    assert get(inspector + "/static/app.js")[0] == "text/javascript"
    assert get(inspector + "/vendor/three/three.module.min.js")[0] == "text/javascript"
    entries = json.loads(get(inspector + "/api/records")[1])["records"]
    assert entries[0]["id"] == "example"
    record = json.loads(get(inspector + "/api/record?id=example")[1])
    assert record["state_count"] == 3 and len(record["cameras"]) == 2
    raw = get(inspector + "/api/sample?id=example&t=0.49")[1]
    assert b"NaN" not in raw and b"Infinity" not in raw
    kind, jpg = get(inspector + "/api/frame?id=example&camera=wrist&index=1")
    assert kind == "image/jpeg" and jpg == b"\xff\xd8wrist\x01\xff\xd9"


@pytest.mark.parametrize("path", ["/static/../loader.py", "/static/%2e%2e/loader.py",
    "/vendor/../server.py", "/api/record?id=../example", "/api/sample?id=unknown&t=0",
    "/api/frame?id=example&camera=exo&index=-1", "/api/frame?id=example&camera=exo&index=99",
    "/mesh/-1?id=example", "/api/request"])
def test_http_cannot_escape_catalog_or_control_robot(inspector, path):
    with pytest.raises(urllib.error.HTTPError) as e:
        get(inspector + path)
    assert e.value.code == 404


def test_invalid_time_and_post_are_rejected(inspector):
    with pytest.raises(urllib.error.HTTPError) as e:
        get(inspector + "/api/sample?id=example&t=nan")
    assert e.value.code == 400
    req = urllib.request.Request(inspector + "/api/request", data=b'{"request":"START"}')
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=5)
    assert e.value.code == 501
