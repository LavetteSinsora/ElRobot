"""P9, the recorder, and `elrobot_deploy.record`, which reads its records back (live too)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from elrobot_deploy.config_loading.config import load
from elrobot_deploy.record import Tail
from elrobot_deploy.record import load as load_record
from elrobot_deploy.runtime.processes.p9_record import Recorder
from elrobot_deploy.runtime.shared_memory import Arena, layout_for, open_all, schema
from elrobot_deploy.runtime.timing import now
from tests.conftest import arena_name

pytest.importorskip("cv2")

SQUARE = (297, 22, 964, 689)            # the wrist lens square at 720p (side 667)


@pytest.fixture
def setup(tmp_path):
    cfg = load(sets=["bus.driver=sim", f'record.dir="{tmp_path}"']).cameras_for(
        astra=True).with_cameras(fake=True)
    a = Arena.create(arena_name(), layout_for(cfg))
    try:
        yield cfg, open_all(a), tmp_path
    finally:
        a.close()
        a.unlink()


def state(rings, t, q=0.1):
    rings["state"].write(t=t, q=np.full(8, q), q_cmd=np.full(8, q))


def frame(ring, t, value, crop):
    slot = ring.spare()
    slot["pixels"][...] = value
    return ring.publish(t, t + 0.001, crop)


def test_one_rollout_is_recorded_whole_and_reads_back_live_and_after(setup):
    cfg, rings, tmp = setup
    wrist, exo = rings["cameras"]["wrist"], rings["cameras"]["exo"]
    t = now()
    state(rings, t - 1.0, q=9.0)                      # before the rollout: never recorded
    frame(wrist, t - 1.0, 9, SQUARE)
    rings["targets"].write(5, t - 0.5, np.zeros(8), intervene=0)
    rec = Recorder(tmp / "r", cfg, rings, t_floor=t, code={"sha": None})
    m = json.loads((tmp / "r" / "manifest.json").read_text())
    assert m["complete"] is False and set(m["cameras"]) == {"exo", "wrist"}

    rings["control"].session = schema.ACTIVE
    for i in range(10):
        state(rings, t + 0.01 * i)
    rings["targets"].write(10, t + 0.1, np.full(8, 0.2), intervene=1, err=0.001,
                           pose=np.array([0.25, 0, 0.1, 1, 0, 0, 0.0]))  # 25 cm ahead
    rings["intervene"].write(intervene=1, deadline=t + 0.1, delta=np.r_[np.zeros(3), 1, 0, 0, 0],
                             grip=0.5)
    frame(wrist, t + 0.01, 100, SQUARE)
    frame(exo, t + 0.01, 50, (0, 0, 1280, 720))
    rec.sweep()
    rec.flush()

    tail = Tail(tmp / "r")
    new = tail.poll()
    assert len(new["state"]) == 10 and np.all(new["state"]["q"][:, 0] == 0.1)
    assert len(new["targets"]) == 1 and new["targets"]["k"][0] == 10
    assert len(new["intervene"]) == 1 and new["intervene"]["grip"][0] == 0.5
    assert len(new["wrist"]) == 1 and tuple(new["wrist"]["crop"][0]) == SQUARE
    assert len(tail.jpeg("wrist", new["wrist"][0])) == new["wrist"]["nbytes"][0]
    assert len(tail.poll()["state"]) == 0               # nothing twice

    # More frames than the ring holds, without a sweep: the lapped ones are counted, not lost
    # silently.
    for i in range(wrist.n + 3):
        frame(wrist, t + 0.1 + 0.03 * i, 120, SQUARE)
    rec.sweep()
    rec.close()

    r = load_record(tmp / "r")
    assert r.manifest["complete"] is True
    assert len(r.state) == 10 and len(r.targets) == 1
    assert r.targets["pose"][0][0] == 0.25 and r.targets["err"][0] == 0.001
    cam = r.cameras["wrist"]
    assert len(cam.index) == 1 + wrist.n and r.manifest["cameras"]["wrist"]["dropped"] == 3
    img = cam.image(0)
    assert img.shape == (667, 667, 3)                   # the lens square, cut, not resized
    assert abs(int(img[333, 333, 0]) - 100) <= 3
    assert r.cameras["exo"].image(0).shape == (720, 1280, 3)     # a normal camera whole
    assert r.session["session"][-1] == schema.ACTIVE
    assert r.manifest["config"]["astra"]["gripper_open_mm"] == 55.0
