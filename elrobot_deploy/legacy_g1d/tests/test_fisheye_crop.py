"""The infra crops every fisheye: a `fisheye180` view's ring holds the square
around its lens circle, never the sensor (`p2_camera._loop_crop`).

The circle is fitted at INIT, before the arena exists (the square is the ring's
shape), so nothing about the square comes from the robot file; its optional
`fallback_lens_circle` is only a fallback. Pinned here: the cut (black off the
sensor), the view the square makes, P2 cutting at it, INIT fitting before the
arena and refusing when there is nothing to fall back to, and the replay
cutting the same square from the recorded whole frame.
"""

from __future__ import annotations

import os
import uuid

import numpy as np
import pytest

from elrobot_deploy.config_loading import resolved as cfgmod
from elrobot_deploy.config_loading.derive import derive
from elrobot_deploy.config_loading.errors import ConfigError
from elrobot_deploy.dashboards.inspect.modes.replay.anchors import views_of
from elrobot_deploy.policy_contract import camera_info
from elrobot_deploy.runtime.drivers import fisheye_circle_fit as fit
from elrobot_deploy.runtime.processes import p2_camera
from elrobot_deploy.runtime.shared_memory import Arena, build, open_all

HZ = 30.0

#: A 20x30 "sensor" whose fitted 16 px square runs 3 px off its top.
FISH = cfgmod.CameraSpec(20, 30, source="zmq", kind="fisheye180",
                         crop=(6, -3, 22, 13), lens_circle=(14.0, 5.0, 6.0))


def sensor(i: int) -> np.ndarray:
    """A BGR frame whose every pixel says where it is: (row, col, i)."""
    yy, xx = np.mgrid[0:20, 0:30]
    return np.stack([yy, xx, np.full_like(yy, i)], -1).astype(np.uint8)


class _Source:
    def __init__(self, n):
        self.n, self.i = n, 0
        self.t_recv, self.jpeg, self.seq = 0.0, None, None

    def connect(self):
        pass

    def close(self):
        pass

    def read_into(self, pixels):
        if self.i >= self.n:
            return False
        pixels[...] = sensor(self.i)
        self.t_recv = 1000.0 + self.i / HZ
        self.jpeg = b"\xff\xd8whole%d" % self.i
        self.i += 1
        return True


class _Arena:
    def __init__(self, cfg, hw):
        self.arena = Arena.create(f"elr-t{os.getpid()}-{uuid.uuid4().hex[:8]}",
                                  build(cfg, derive(cfg, hw)))
        self.rings = open_all(self.arena, cfg)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.arena.close()
        self.arena.unlink()


def _fish_cfg(cfg, spec=FISH):
    return cfgmod.replace(cfg, cameras={"lwrist": spec}, primary_camera="lwrist")


def disc_frames(cx, cy, r, *, n=10, h=720, w=1280):
    """A lit disc with a dark bar moving across it (the gripper's stand-in)."""
    yy, xx = np.mgrid[0:h, 0:w]
    disc = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r
    out = []
    for i in range(n):
        f = np.zeros((h, w), np.uint8)
        f[disc] = 180
        x0 = int(cx - r + (2 * r - 120) * (i / max(n - 1, 1)))
        f[:, max(x0, 0):x0 + 120] = 0
        out.append(f)
    return out


# --- The cut -----------------------------------------------------------------


def test_the_cut_is_the_window_with_black_off_the_sensor():
    f = sensor(1)
    sq = fit.cut(f, 6, -3, 16)
    assert sq.shape == (16, 16, 3)
    assert np.all(sq[:3] == 0), "the 3 rows above the sensor are black"
    assert np.array_equal(sq[3:], f[0:13, 6:22])


def test_a_cut_inside_the_sensor_is_a_plain_window_into_a_given_buffer():
    f = sensor(2)
    out = np.full((10, 10, 3), 77, np.uint8)
    fit.cut(f, 4, 5, 10, out=out)
    assert np.array_equal(out, f[5:15, 4:14])


def test_a_cut_entirely_off_the_sensor_is_black():
    assert not fit.cut(sensor(0), 100, 100, 5).any()


# --- The view ----------------------------------------------------------------


def test_a_fitted_fisheye_is_one_square_view_at_its_fitted_square():
    v = cfgmod.camera_views({"lwrist": FISH})["lwrist"]
    assert v.cropped and (v.height, v.width) == (16, 16)
    assert (v.x0, v.y0) == (6, -3)
    assert (v.source_height, v.source_width) == (20, 30)
    assert v.jpeg_capacity == 20 * 30 * 3 // 4, "the JPEG is the whole frame"


def test_before_a_fit_the_square_is_fallback_lens_circles_then_a_placeholder():
    spec = cfgmod.CameraSpec(720, 1280, kind="fisheye180",
                             fallback_lens_circle=(675.48, 342.61, 343.44))
    assert cfgmod.lens_square(spec) == (fit.crop_for(675.48, 342.61, 343.44),
                                        "fallback_lens_circle")
    bare = cfgmod.CameraSpec(720, 1280, kind="fisheye180")
    assert cfgmod.lens_square(bare) == ((280, 0, 1000, 720), "placeholder")
    assert cfgmod.camera_views({"w": bare})["w"].width == 720


def test_a_normal_camera_is_not_cropped():
    v = cfgmod.camera_views({"head": cfgmod.CameraSpec(8, 16)})["head"]
    assert not v.cropped and (v.height, v.width) == (8, 16)


def test_the_policy_is_told_the_square_it_gets(cfg):
    info = camera_info(_fish_cfg(cfg))["lwrist"]
    assert (info.height, info.width) == (16, 16)
    assert info.crop == (6, -3, 22, 13) and info.crop_source == "fitted"
    assert info.lens_circle == (14.0, 5.0, 6.0)


def test_the_square_nearly_fills_with_the_disc():
    x0, y0, x1, y1 = fit.crop_for(648.0, 386.0, 375.0)
    assert x1 - x0 == y1 - y0 == 2 * (375 + fit.CROP_MARGIN_PX)


# --- P2 ------------------------------------------------------------------------


def test_p2_publishes_only_the_square_and_the_jpeg_stays_whole(cfg, hw):
    c = _fish_cfg(cfg)
    with _Arena(c, hw) as a:
        p2_camera.run(a.arena.shm.name, c, hw, camera="lwrist", source=_Source(2))
        ring = a.rings["cameras"]["lwrist"]
        px = ring.read_frame(2)[0]
        assert px.shape == (16, 16, 3)
        assert np.array_equal(px, fit.cut(sensor(1), 6, -3, 16))
        assert ring.read_jpeg()[2] == b"\xff\xd8whole1"


def test_the_arena_holds_nothing_for_the_crop_but_the_ring():
    """The square is decided before the arena exists: no tap, no corner word."""
    names = set(build(_fish_cfg(_cfg_stub()), derive(_fish_cfg(_cfg_stub()),
                                                     cfgmod.load_hardware())).regions)
    assert {n for n in names if n.startswith("cam_lwrist")} == {
        "cam_lwrist_head", "cam_lwrist_seq", "cam_lwrist_rec"}


def _cfg_stub():
    return cfgmod.DeployConfig(
        prompt="", checkpoint="", sync=True, replan_period_ticks=3, action_horizon=8,
        control_hz=30.0, cameras={"lwrist": FISH}, primary_camera="lwrist",
        home_q=np.zeros(14), home_waist=np.zeros(3), home_hand=np.zeros((2, 6)),
        exit_q=np.zeros(14), exit_waist=np.zeros(3), exit_hand=np.zeros((2, 6)))


# --- The fit -------------------------------------------------------------------


def test_the_fit_alone_decides_centre_and_size():
    spec = cfgmod.CameraSpec(720, 1280, source="zmq", kind="fisheye180")
    d = fit.decide("lwrist", spec, disc_frames(660.0, 340.0, 368.0))
    assert d.used == "fit" and not d.warning
    assert d.lens_circle == pytest.approx((660.0, 340.0, 368.0), abs=4)
    assert d.crop == fit.crop_for(*d.lens_circle)
    assert d.drift is None, "no fallback_lens_circle, nothing to drift from"


def test_a_failed_fit_without_fallback_lens_circle_refuses_the_launch():
    spec = cfgmod.CameraSpec(720, 1280, source="zmq", kind="fisheye180")
    with pytest.raises(ConfigError, match="fallback_lens_circle"):
        fit.decide("lwrist", spec, [np.zeros((720, 1280), np.uint8)] * 4)


def test_a_failed_fit_falls_back_to_fallback_lens_circle_with_a_warning():
    spec = cfgmod.CameraSpec(720, 1280, source="zmq", kind="fisheye180",
                             fallback_lens_circle=(648.0, 386.0, 375.0))
    d = fit.decide("lwrist", spec, [])
    assert d.used == "fallback_lens_circle" and d.warning
    assert d.crop == fit.crop_for(648.0, 386.0, 375.0)
    assert "WARNING" in d.line()


def test_capture_source_settles_then_pools_and_never_hangs():
    class Blocking(_Source):
        def read_into(self, pixels):
            import time
            time.sleep(10)
            return False

    got = fit.capture_source(_Source(50), (20, 30, 3), n=5, settle_s=0.0)
    assert len(got) == 5 and got[0].shape == (20, 30)
    assert fit.capture_source(Blocking(1), (20, 30, 3), settle_s=0.0, timeout_s=0.2) == []


def test_init_fits_before_the_arena_and_hands_every_process_the_square(cfg, hw):
    from elrobot_deploy.runtime.launcher import Launcher

    spec = cfgmod.CameraSpec(720, 1280, source="zmq", kind="fisheye180")
    c = _fish_cfg(cfg, spec)
    lch = Launcher(c, hw, arena_name=f"elr-t{os.getpid()}-{uuid.uuid4().hex[:8]}",
                   log=lambda *_: None,
                   lens_frames=lambda cam: disc_frames(655.0, 380.0, 372.0))
    try:
        lch.init()
        got = lch.cfg.cameras["lwrist"]
        assert got.crop == lch.crops[0].crop and got.lens_circle == lch.crops[0].lens_circle
        side = got.crop[2] - got.crop[0]
        assert lch.rings["cameras"]["lwrist"].dtype["pixels"].shape == (side, side, 3)
    finally:
        if lch.arena is not None:
            lch.arena.close()
            lch.arena.unlink()


# --- The replay ------------------------------------------------------------------


def test_the_replay_cuts_the_square_the_live_run_cut():
    manifest = {"camera_views": {"lwrist": {"source": "lwrist", "x0": 10, "y0": 2,
                                            "cropped": True, "height": 16, "width": 16}}}
    assert views_of(manifest, {"lwrist"})["lwrist"].square == (10, 2, 16)
