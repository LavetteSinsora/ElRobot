"""Camera rings, the binocular head split and the history reader.

What can go wrong here goes wrong silently. A ring whose frame-id -> slot
mapping drifts hands a policy the wrong instant with a plausible image; a torn
slot is half of one frame and half of another; a split with the eyes swapped is
a mirrored stereo pair; a ring sized from the wrong history returns a window
cut short with no error. Each of those is pinned below.
"""

from __future__ import annotations

import os
import types
import uuid

import numpy as np
import pytest

from elrobot_deploy.config_loading import resolved as cfgmod
from elrobot_deploy.config_loading.derive import derive
from elrobot_deploy.config_loading.errors import ConfigError
from elrobot_deploy.config_loading.validate import validate
from elrobot_deploy.config_loading import deploy_file as deployfile
from elrobot_deploy.runtime.shared_memory import (Arena, ShmTooSmall, build, check_shm,
                                              open_all, rings as rings_mod, schema)
from elrobot_deploy.policy_contract import (DummyReader, EefActions, Frame, FrameTimes,
                                        Policy, PolicyError, Reader, check_against,
                                        dummy_observation, nearest_to_times,
                                        subsample_every)
from elrobot_deploy.runtime.processes import p2_camera
from elrobot_deploy.runtime.processes.p5_policy import history_reader

HZ = 30.0
T0 = 1000.0

#: A head the way every robot file declares it: one side_by_side frame. 8x16
#: so the arena stays tiny; the split makes two 8x8 eyes.
HEAD = cfgmod.CameraSpec(8, 16, layout="side_by_side", binocular=True)


def _split_cfg(cfg, **history):
    return cfgmod.replace(cfg, cameras={"rwrist": cfg.cameras["rwrist"],
                                        "head": HEAD},
                          primary_camera="head_left",
                          camera_history_s=dict(history))


class _Arena:
    """An arena for a config other than conftest's, unlinked on exit."""

    def __init__(self, cfg, hw):
        self.cfg = cfg
        self.layout = build(cfg, derive(cfg, hw))
        self.arena = Arena.create(f"elr-t{os.getpid()}-{uuid.uuid4().hex[:8]}",
                                  self.layout)
        self.rings = open_all(self.arena, cfg)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.arena.close()
        self.arena.unlink()


def _publish(ring, t, value=0, jpeg=None):
    ring.spare()["pixels"][...] = value
    return ring.publish(t, jpeg)


# --- sizing ------------------------------------------------------


def test_ring_depth_follows_the_declared_history():
    """N = ceil(s x fps x 1.25) + 3; no history is the old triple buffer."""
    assert cfgmod.camera_slots(0.0, 30.0) == 3
    assert cfgmod.camera_slots(4.0, 30.0) == 153
    assert cfgmod.camera_slots(1.0, 30.0) == 41
    assert cfgmod.camera_slots(0.1, 30.0) == 7


def test_history_reaches_the_layout_through_the_config(cfg, hw):
    c = _split_cfg(cfg, head_left=4.0)
    d = derive(c, hw)
    assert d.camera_slots == {"head_left": 153, "head_right": 3, "rwrist": 3}
    layout = build(c, d)
    assert layout.cameras == ("head_left", "head_right", "rwrist")
    assert layout.region("cam_head_left_rec").count == 153
    assert layout.region("cam_head_left_seq").count == 153
    assert layout.region("cam_head_right_rec").count == 3
    assert "head_left:153" in d.describe()
    # A different history is a different layout: every process must agree.
    assert build(_split_cfg(cfg, head_left=2.0), derive(
        _split_cfg(cfg, head_left=2.0), hw)).hash != layout.hash


def test_resolve_copies_the_policy_classes_camera_history(tmp_path):
    (tmp_path / "pol.py").write_text(
        "from elrobot_deploy.policy_contract import Policy\n"
        "class P(Policy):\n"
        "    action_mode = 'eef'\n"
        "    end_effector = 'dex1'\n"
        "    anchor_camera = 'head_left'\n"
        "    history = {'state': 0.5, 'head_left': 4.0, 'head_right': 2.0}\n"
        "    def infer(self, obs, reader): ...\n")
    f = tmp_path / "deploy.toml"
    f.write_text('robot = "g1d_normal_dex1"\n[policy]\nentry = "pol.py:P"\n'
                 "[home]\narm = [0,0,0,0,0,0,0, 0,0,0,0,0,0,0]\n")
    r = deployfile.resolve(f)
    assert r.cfg.camera_history_s == {"head_left": 4.0, "head_right": 2.0}
    assert r.sources["camera_history_s"] == "policy"
    assert {n: v.slots for n, v in r.cfg.views.items()} == {
        "head_left": 153, "head_right": 78}
    check_against(r.policy, r.cfg)


def test_resolve_refuses_history_for_the_unsplit_name(tmp_path):
    (tmp_path / "pol.py").write_text(
        "from elrobot_deploy.policy_contract import Policy\n"
        "class P(Policy):\n"
        "    action_mode = 'eef'\n"
        "    end_effector = 'dex1'\n"
        "    anchor_camera = 'head_left'\n"
        "    history = {'head': 4.0}\n"
        "    def infer(self, obs, reader): ...\n")
    f = tmp_path / "deploy.toml"
    f.write_text('robot = "g1d_normal_dex1"\n[policy]\nentry = "pol.py:P"\n'
                 "[home]\narm = [0,0,0,0,0,0,0, 0,0,0,0,0,0,0]\n")
    with pytest.raises(ConfigError, match="head_left"):
        deployfile.resolve(f)


class _HistoryPolicy(Policy):
    action_mode = "eef"
    end_effector = "dex1"
    anchor_camera = "head_left"
    history = {"head_left": 4.0}

    def infer(self, obs, reader):
        raise AssertionError("not called")


def test_a_history_longer_than_the_ring_is_refused_at_load(cfg):
    """A config that was not resolved from this class (a deploy.py launch, a
    hand-built config) has smaller rings than the class wants: refused, never
    a window silently cut short."""
    with pytest.raises(PolicyError, match="camera rings hold"):
        check_against(_HistoryPolicy, _split_cfg(cfg))
    check_against(_HistoryPolicy, _split_cfg(cfg, head_left=4.0))


# --- The ring ----------------------------------------------------


def test_the_ring_wraps_and_keeps_the_last_n_frames(cfg, hw):
    with _Arena(_split_cfg(cfg, head_left=0.1), hw) as a:      # N = 7
        ring = a.rings["cameras"]["head_left"]
        assert ring.n == 7
        for i in range(1, 18):
            assert _publish(ring, T0 + i / HZ, value=i) == i
        # frame f lives in slot (f - 1) % N
        rec = a.arena.view("cam_head_left_rec")
        assert int(rec[(17 - 1) % 7]["frame_id"]) == 17
        ids, t = ring.window(-np.inf, np.inf)
        assert ids.tolist() == list(range(11, 18))            # the newest 7
        assert np.all(np.diff(t) > 0)
        assert ring.peek() == (pytest.approx(T0 + 17 / HZ), 17)
        px, t17 = ring.read_frame(17)
        assert np.all(px == 17) and t17 == pytest.approx(T0 + 17 / HZ)
        assert ring.read_frame(10) is None, "lapped: gone, not a stale frame"
        assert ring.read_frame(18) is None, "not yet published"


def test_a_slot_being_written_is_never_read(rings):
    """`spare()` marks the slot odd for the whole decode: the window, a frame
    read and the newest-frame readers all skip it."""
    ring = rings["cameras"]["rwrist"]                          # N = 3
    for i in range(1, 4):
        _publish(ring, T0 + i, value=i)
    ring.spare()["pixels"][...] = 99                           # writing frame 4
    ids, _ = ring.window(-np.inf, np.inf)
    assert ids.tolist() == [2, 3], "frame 1's slot is being overwritten"
    assert ring.read_frame(1) is None
    assert ring.peek()[1] == 3
    out = ring.empty()
    assert ring.read_into(out) and int(out["frame_id"]) == 3
    ring.publish(T0 + 4)
    assert ring.window(-np.inf, np.inf)[0].tolist() == [2, 3, 4]


def test_a_frame_overwritten_during_the_copy_is_dropped(rings, monkeypatch):
    """Copy-and-verify: the writer lapping the slot WHILE the reader copies it
    moves the counter, and the copy is thrown away rather than returned."""
    ring = rings["cameras"]["rwrist"]
    for i in range(1, 4):
        _publish(ring, T0 + i, value=i)
    seq = ring._seq

    def lapping_copy(a):
        seq[ring._slot(2)] += 2                    # a whole write, mid-copy
        return np.array(a, copy=True)

    fake_np = types.SimpleNamespace(**{k: getattr(np, k) for k in dir(np)
                                       if not k.startswith("__")})
    fake_np.ascontiguousarray = lapping_copy
    monkeypatch.setattr(rings_mod, "np", fake_np)
    assert ring.read_frame(2, flip_channels=True) is None
    monkeypatch.undo()
    reader = Reader(rings["state"], {"rwrist": 1.0}, {"rwrist": ring})
    monkeypatch.setattr(rings_mod, "np", fake_np)
    got = reader.frames_by_id("rwrist", [1, 2, 3])
    assert [f.frame_id for f in got] == [1, 3], "the torn one is left out"


def test_the_writer_resumes_the_frame_id_mapping(rings, arena, cfg):
    """A second writer over a live ring (a restarted camera process) continues
    the ids, so `(f - 1) % N` still names every frame's slot."""
    ring = rings["cameras"]["rwrist"]
    for i in range(1, 5):
        _publish(ring, T0 + i, value=i)
    again = open_all(arena, cfg)["cameras"]["rwrist"]
    assert _publish(again, T0 + 5, value=5) == 5
    assert again.window(-np.inf, np.inf)[0].tolist() == [3, 4, 5]
    assert np.all(again.read_frame(5)[0] == 5)


def test_jpegs_since_catches_up_and_skips_the_uncaptured(rings):
    ring = rings["cameras"]["rwrist"]
    ring.publish(T0 + 1, b"\xff\xd8a")
    ring.publish(T0 + 2, None)                      # not captured
    ring.publish(T0 + 3, b"\xff\xd8c")
    got, newest = ring.jpegs_since(0)
    assert [(fid, p) for _, fid, p in got] == [(1, b"\xff\xd8a"), (3, b"\xff\xd8c")]
    assert newest == 3
    assert ring.jpegs_since(3) == ([], 3)


# --- /dev/shm ----------------------------------------------------


def test_an_arena_dev_shm_cannot_hold_is_refused_with_the_numbers(layout):
    line = check_shm(layout, free=lambda: (1 << 34, 1 << 35))
    assert "MB free" in line
    with pytest.raises(ShmTooSmall) as exc:
        check_shm(layout, free=lambda: (layout.size // 2, 1 << 30))
    msg = str(exc.value)
    assert f"{layout.size / 1e6:.1f} MB" in msg and "history" in msg
    assert "rwrist" in msg and "slots" in msg


def test_p0_refuses_before_the_arena_exists(cfg, hw):
    from elrobot_deploy.runtime.launcher import Launcher

    name = f"elr-shm-{uuid.uuid4().hex[:8]}"
    lch = Launcher(cfg, hw, arena_name=name, log=lambda *_: None,
                   shm_free=lambda: (1 << 20, 1 << 30))
    with pytest.raises(ConfigError, match="not enough"):
        lch.init()
    assert lch.arena is None
    assert not os.path.exists(f"/dev/shm/{name}")


def test_a_realistic_config_fits_and_says_how_much(hw):
    """The box's rig: two 720x1280 fisheye wrists and the split head, with 4 s
    of both eyes. The numbers the operator will see at INIT."""
    robot = __import__("elrobot_deploy.config_loading.robot_file", fromlist=["load_robot"]).load_robot(
        "g1d_fisheye_dex1")
    c = cfgmod.DeployConfig(
        prompt="", checkpoint="", sync=True, replan_period_ticks=25,
        action_horizon=50, control_hz=30.0, cameras=dict(robot.cameras),
        primary_camera="rwrist", home_q=np.zeros(14), home_waist=np.zeros(3),
        home_hand=np.zeros((2, 6)), exit_q=np.zeros(14),
        exit_waist=np.zeros(3), exit_hand=np.zeros((2, 6)),
        camera_history_s={"head_left": 4.0, "head_right": 4.0})
    layout = build(c, derive(c, hw))
    left = layout.region("cam_head_left_rec")
    right = layout.region("cam_head_right_rec")
    assert left.count == right.count == 153
    # The left eye's slots also hold the whole wire JPEG; the right's hold none.
    assert right.dtype.itemsize == 480 * 640 * 3 + 32
    assert left.dtype.itemsize == right.dtype.itemsize + 480 * 1280 * 3 // 4
    assert 330e6 < layout.size < 380e6


# --- The head split ----------------------------------------------


def test_a_side_by_side_camera_is_two_named_views(cfg):
    c = _split_cfg(cfg)
    views = c.views
    assert sorted(views) == ["head_left", "head_right", "rwrist"]
    left, right = views["head_left"], views["head_right"]
    assert (left.source, left.half, left.x0, left.width) == ("head", "left", 0, 8)
    assert (right.source, right.half, right.x0, right.width) == ("head", "right", 8, 8)
    assert left.carries_jpeg and not right.carries_jpeg
    assert right.jpeg_capacity == 0
    assert sorted(c.cameras) == ["head", "rwrist"], "the robot file's, untouched"
    assert c.observed == ("head_left", "head_right", "rwrist")


def test_the_unsplit_name_is_refused_with_the_names_it_meant(cfg, hw):
    with pytest.raises(ConfigError, match=r"head_left.*head_right"):
        validate(cfgmod.replace(_split_cfg(cfg), primary_camera="head"), hw)
    with pytest.raises(ConfigError, match="split into"):
        cfgmod.camera_views({"head": HEAD, "head_left": cfg.cameras["rwrist"]})


class _StereoSource:
    """A side_by_side source: the left eye is BGR (10+i, 1, 2), the right eye
    (200, 3, 4) — distinct in every channel, so a swap or a missed BGR->RGB
    conversion cannot pass."""

    def __init__(self, n, *, t0=T0, hz=HZ):
        self.n, self.i, self.t0, self.hz = n, 0, t0, hz
        self.t_recv, self.jpeg, self.seq = 0.0, None, None

    def connect(self):
        pass

    def close(self):
        pass

    def read_into(self, pixels):
        if self.i >= self.n:
            return False
        w = pixels.shape[1] // 2
        pixels[:, :w] = (10 + self.i % 200, 1, 2)
        pixels[:, w:] = (200, 3, 4)
        self.t_recv = self.t0 + self.i / self.hz
        self.jpeg = b"\xff\xd8frame%d" % self.i
        self.seq = 1000 + self.i
        self.i += 1
        return True


def test_p2_splits_the_head_into_its_two_eyes(cfg, hw):
    c = _split_cfg(cfg)
    with _Arena(c, hw) as a:
        src = _StereoSource(5)
        p2_camera.run(a.arena.shm.name, c, hw, camera="head", source=src)
        left, right = a.rings["cameras"]["head_left"], a.rings["cameras"]["head_right"]
        assert left.peek() == right.peek() == (pytest.approx(T0 + 4 / HZ), 5)
        out = left.empty_pixels()
        left.read_pixels_into(out)
        assert out.shape == (8, 8, 3) and np.all(out == (14, 1, 2))
        right.read_pixels_into(out)
        assert np.all(out == (200, 3, 4))
        assert left.peek_counters() == right.peek_counters() == (5, 1004)
        # The wire JPEG — the whole binocular frame — rides with the left eye
        # only, and that is what the recorder and the dashboard read, under the
        # robot file's name.
        assert left.read_jpeg()[2] == b"\xff\xd8frame4"
        assert right.read_jpeg() is None
        assert a.rings["camera_sources"] == {"head": left,
                                             "rwrist": a.rings["cameras"]["rwrist"]}


def test_the_recorder_and_dashboard_keep_the_whole_head(cfg, hw, tmp_path):
    from elrobot_deploy.dashboards.live.server import Live
    from elrobot_deploy.runtime.processes.p9_record import Recorder

    c = _split_cfg(cfg)
    with _Arena(c, hw) as a:
        rec = Recorder(tmp_path, c, a.rings)
        p2_camera.run(a.arena.shm.name, c, hw, camera="head",
                      source=_StereoSource(2))
        rec.sweep()
        assert set(rec.frames) == {"head", "rwrist"}
        assert rec.frames["head"][1].count == 2
        rec.close(hw)
        import json
        manifest = json.loads((rec.dir / "manifest.json").read_text())
        assert manifest["camera_views"]["head_right"] == {
            "source": "head", "half": "right", "x0": 8, "y0": 0, "cropped": False, "height": 8,
            "width": 8, "slots": 3, "history_s": 0.0}

        live = Live(c, hw, a.rings, scene=None)
        panes = {p["name"]: p for p in live.cameras()}
        assert set(panes) == {"head", "rwrist"}
        assert panes["head"]["primary"] and panes["head"]["used"]
        assert panes["head"]["binocular"]


# --- The reader ----------------------------------------------------


def test_frame_ids_between_is_a_closed_ascending_window(rings):
    ring = rings["cameras"]["rwrist"]
    for i in range(1, 4):
        _publish(ring, T0 + i, value=i)
    r = Reader(rings["state"], {"rwrist": 1.0}, {"rwrist": ring})
    ft = r.frame_ids_between("rwrist", T0 + 2, T0 + 3)
    assert isinstance(ft, FrameTimes)
    assert ft.ids.tolist() == [2, 3] and ft.t_recv.tolist() == [T0 + 2, T0 + 3]
    assert r.frame_ids_between("rwrist", T0 + 2.5, T0 + 2.6).ids.tolist() == []
    # Early in a rollout: what exists, no padding, no error.
    assert r.frame_ids_between("rwrist", T0 - 100, T0 + 100).ids.tolist() == [1, 2, 3]
    with pytest.raises(ValueError, match="empty window"):
        r.frame_ids_between("rwrist", T0 + 3, T0 + 2)


def test_frames_are_rgb_private_copies_of_exactly_those_ids(cfg, hw):
    c = _split_cfg(cfg, head_left=1.0)
    with _Arena(c, hw) as a:
        p2_camera.run(a.arena.shm.name, c, hw, camera="head",
                      source=_StereoSource(10))
        ring = a.rings["cameras"]["head_left"]
        r = Reader(a.rings["state"], {"head_left": 1.0}, {"head_left": ring})
        got = r.frames_by_id("head_left", [7, 3])
        assert [f.frame_id for f in got] == [7, 3], "in the order asked"
        assert np.all(got[0].pixels == (2, 1, 10 + 6)), "BGR (16,1,2) -> RGB"
        assert got[1].t_recv == pytest.approx(T0 + 2 / HZ)
        got[0].pixels[...] = 0                       # a private copy
        assert np.all(r.frames_by_id("head_left", [7])[0].pixels == (2, 1, 16))
        every_frame = r.frames_between("head_left", T0, T0 + 1.0)
        assert [f.frame_id for f in every_frame] == list(range(1, 11))


def test_nearest_and_every_work_on_whatever_the_reader_returned():
    t = T0 + np.arange(10) / 10.0
    ft = FrameTimes(np.arange(1, 11), t)
    picks = nearest_to_times(ft, [T0 + 0.24, T0 + 0.26, T0 + 5.0, T0 - 1.0])
    assert picks.ids.tolist() == [3, 4, 10, 1]
    frames = [Frame(np.zeros((1, 1, 3), np.uint8), float(x), i + 1)
              for i, x in enumerate(t)]
    assert [f.frame_id for f in nearest_to_times(frames, [T0 + 0.5])] == [6]
    sub = subsample_every(ft, 0.3)
    assert sub.ids.tolist() == [1, 4, 7, 10], "anchored at the newest"
    assert subsample_every(FrameTimes(np.empty(0, np.int64), np.empty(0)), 0.3).ids.size == 0
    assert nearest_to_times(ft, []).ids.size == 0


# --- warmup: the dummy carries the whole declared history ---------------------


def test_the_dummy_reader_serves_every_declared_history_at_full_length(cfg):
    c = _split_cfg(cfg, head_left=4.0, head_right=4.0)
    r = DummyReader(c.views, {"head_left": 4.0, "head_right": 4.0,
                              "state": 0.5}, state_hz=500.0)
    ft = r.frame_ids_between("head_left", -4.0, 0.0)
    assert len(ft.ids) == 121 and ft.t_recv[-1] == 0.0
    assert np.all(np.diff(ft.ids) == 1) and np.all(np.diff(ft.t_recv) > 0)
    frames = r.frames_between("head_right", -4.0, 0.0)
    assert len(frames) == 121 and frames[0].pixels.shape == (8, 8, 3)
    w = r.states_between(-0.5, 0.0)
    assert w.q.shape == (251, schema.N_JOINTS) and w.flange[0, 0, 3] == 1.0
    with pytest.raises(LookupError):
        r.frame_ids_between("rwrist", -1.0, 0.0)                # not declared


class ToyStereoPolicy(Policy):
    """Reads 4 s of BOTH head views every inference — the 'done
    when'. Keeps what it saw so the test can check it."""

    action_mode = "eef"
    end_effector = "dex1"
    anchor_camera = "head_left"
    uses_cameras = ("head_left", "head_right")
    history = {"head_left": 4.0, "head_right": 4.0}

    def infer(self, obs, reader):
        t1 = obs.t_anchor
        self.seen = {cam: reader.frames_between(cam, t1 - 4.0, t1)
                     for cam in ("head_left", "head_right")}
        self.stacked = {cam: np.stack([f.pixels for f in fs])
                        for cam, fs in self.seen.items()}
        pose = np.repeat(obs.state.flange[None], 4, axis=0)
        return EefActions(pose=pose, hand=np.zeros((4, 2)))


def test_a_toy_policy_reads_four_seconds_of_both_head_views(cfg, hw):
    c = _split_cfg(cfg, head_left=4.0, head_right=4.0)
    check_against(ToyStereoPolicy, c)
    pol = ToyStereoPolicy(None, "", hw)

    # Warmup first: the dummy has the declared history at full length, so the
    # shapes the policy stacks at warmup are the shapes it stacks live.
    from elrobot_deploy.policy_contract import bind_runtime
    dummy = dummy_observation({v: c.views[v] for v in pol.uses_cameras}, "")
    bind_runtime(pol, dummy, DummyReader(c.views, pol.history))
    pol.warmup()
    warm_shapes = {k: v.shape for k, v in pol.stacked.items()}
    assert warm_shapes == {"head_left": (121, 8, 8, 3),
                           "head_right": (121, 8, 8, 3)}

    with _Arena(c, hw) as a:
        # 5 s of head frames at 30 Hz: more than the policy asks for, less than
        # the 153-slot ring holds.
        p2_camera.run(a.arena.shm.name, c, hw, camera="head",
                      source=_StereoSource(150))
        t_anchor = a.rings["cameras"]["head_left"].peek()[0]
        obs = dummy_observation({v: c.views[v] for v in pol.uses_cameras}, "")
        obs = type(obs)(t_anchor=t_anchor, frames=obs.frames, state=obs.state,
                        prompt="")
        pol.infer(obs, history_reader(a.rings, pol))

    left, right = pol.seen["head_left"], pol.seen["head_right"]
    assert len(left) == len(right) == 121
    assert [f.frame_id for f in left] == [f.frame_id for f in right]
    assert left[-1].t_recv == t_anchor
    assert left[-1].t_recv - left[0].t_recv == pytest.approx(4.0)
    assert {k: v.shape for k, v in pol.stacked.items()} == warm_shapes
    assert np.all(pol.stacked["head_right"] == (4, 3, 200)), "RGB, right eye"
    assert np.all(pol.stacked["head_left"][:, :, :, :2] == (2, 1))
