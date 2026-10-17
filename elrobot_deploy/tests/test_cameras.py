"""The cameras (`runtime/cameras/`, P2): the lens fit and square, the exposure modes (above all
`camera_first`: the camera's own settings until they are not good enough), the camera ring,
the profile tables, the UVC plumbing that can be checked without a camera, P2 as a process
and the dashboard's view. The real camera's side is `elrobot_deploy camera probe` / `test`."""

from __future__ import annotations

import json
import threading
import time
import urllib.request

import numpy as np
import pytest

from elrobot_deploy.config_loading.config import load
from elrobot_deploy.config_loading.errors import ConfigError
from elrobot_deploy.runtime.cameras import exposure, lens, sources, uvc
from elrobot_deploy.runtime.cameras.exposure import Controller, Measurement, Patience
from elrobot_deploy.runtime.cameras.sources import FakeCamera, FakeControls
from elrobot_deploy.runtime.processes import p2_camera
from elrobot_deploy.runtime.shared_memory import Arena, layout_for, open_all, schema
from tests.conftest import arena_name

SHAPE = (360, 640, 3)


def meas(p995, clip=0.0, mean=120.0):
    return Measurement(p995=p995, mean=mean, clip_frac=clip, pixels=10_000)


def cam_cfg(**sets):
    s = ['cameras.wrist.source="fake"', "cameras.wrist.width=640", "cameras.wrist.height=360"]
    s += [f"cameras.wrist.{k}={json.dumps(v)}" for k, v in sets.items()]
    return load(sets=["bus.driver=sim"] + s)


# --- the lens ---------------------------------------------------------------------------


def test_the_fit_finds_the_fake_lens_and_cuts_its_square():
    cam = FakeCamera(SHAPE, fps=500, circle=(310.0, 185.0, 160.0))
    for name, value in (("ae_mode", 1), ("exposure", 200), ("gain", 64)):
        cam.ctl.write(name, value)
    px = np.empty(SHAPE, np.uint8)
    frames = []
    for _ in range(lens.REFIT_FRAMES):
        cam.read_into(px)
        frames.append(lens.luma(px).copy())
    d = lens.decide("wrist", frames)
    assert d.used == "fit"
    assert np.allclose(d.lens_circle, (310, 185, 160), atol=2.0)
    x0, y0, x1, y1 = d.crop
    assert x1 - x0 == y1 - y0 == round(2 * (d.lens_circle[2] + lens.CROP_MARGIN_PX))
    sq = lens.cut(px, x0, y0, x1 - x0)
    assert sq.shape[:2] == (x1 - x0,) * 2 and sq[sq.shape[0] // 2, sq.shape[1] // 2].max() > 0


def test_a_failed_fit_falls_back_then_to_the_centred_square():
    dark = [np.zeros(SHAPE[:2], np.uint8)] * 4
    d = lens.decide("wrist", dark, (300.0, 180.0, 150.0))
    assert d.used == "fallback_lens_circle" and d.lens_circle == (300.0, 180.0, 150.0)
    d = lens.decide("wrist", dark)
    assert d.used == "placeholder" and d.crop == (140, 0, 500, 360) and d.warning


def test_cut_blacks_out_what_overhangs_the_sensor():
    frame = np.full((10, 10, 3), 7, np.uint8)
    sq = lens.cut(frame, -2, -3, 8)
    assert sq[:3].max() == 0 and sq[:, :2].max() == 0 and sq[3:, 2:].min() == 7


# --- exposure: the modes ------------------------------------------------------------------


def test_camera_first_restores_the_cameras_own_settings_and_leaves_them_alone():
    c = Controller("camera_first")
    writes = c.begin(0.0)
    assert writes == [("gain", None), ("ae_mode", None), ("wb_auto", None)]   # not exposure
    for i in range(100):                                    # 10 s of a good image
        assert c.update(meas(240.0), 0.1 * i) == []
    assert c.owner == "camera" and not c.takeover


def test_camera_first_leaves_a_lit_screen_alone():
    # What our camera's own loop did on 2026-10-17 with a monitor in view: 7% of the disc
    # clipped (the monitor), mean 157, the workspace well exposed. Not a failure.
    c = Controller("camera_first")
    c.begin(0.0)
    for i in range(200):
        assert c.update(meas(255.0, clip=0.07, mean=157.0), 0.1 * i) == []
    assert c.owner == "camera" and not c.takeover
    assert c.spell.need == "bright"          # what sticky, owning it, would have acted on


def test_camera_first_ignores_a_brief_glare_but_takes_over_from_a_lasting_one():
    p = Patience()
    c = Controller("camera_first")
    c.begin(0.0)
    t = p.takeover_grace_s + 0.1
    for i in range(int(p.takeover_bright_s / 0.1) - 3):     # clipped, but not for long enough
        assert c.update(meas(255.0, clip=0.26), t + 0.1 * i) == []
    t += 2.0
    assert c.update(meas(240.0), t) == []                   # the glare is gone: start over
    out = []
    for i in range(int(p.takeover_bright_s / 0.1) + 2):     # g1d's failure: 26% clipped
        out = c.update(meas(255.0, clip=0.26), t + 0.1 * (i + 1), current=lambda: (300, 40))
        if out:
            break
    assert c.owner == "ours" and "clipped" in c.takeover
    names = [n for n, _ in out]
    assert names[0] == "ae_mode" and dict(out)["ae_mode"] == exposure.AE_MANUAL
    assert "gain" in names and "exposure" in names
    # it started from what the camera had reached (300 x gain 40), and cut from there
    light_then = exposure.light_index(300, 40)
    assert c.state.light < light_then


def test_camera_first_without_a_control_channel_only_says_it_wanted_to():
    c = Controller("camera_first", can_control=False)
    c.begin(0.0)
    for i in range(100):
        assert c.update(meas(255.0, clip=0.3), 0.1 * i) == []
    assert c.owner == "camera" and c.wants_takeover and c.events


def test_camera_mode_never_takes_over():
    c = Controller("camera")
    c.begin(0.0)
    assert all(c.update(meas(255.0, clip=0.3), 0.1 * i) == [] for i in range(100))
    assert c.owner == "camera" and c.spell.need == "panic"


def test_sticky_holds_through_a_lamp_and_corrects_a_lasting_need_then_holds():
    p = Patience()
    c = Controller("sticky")
    first = c.begin(0.0)
    assert dict(first)["ae_mode"] == exposure.AE_MANUAL
    t = 0.0
    for _ in range(int(p.bright_s / 0.1) - 2):              # a lamp swept through the view
        t += 0.1
        assert c.update(meas(254.0, clip=0.02), t) == []
    for _ in range(5):
        t += 0.1
        assert c.update(meas(225.0), t) == []               # acceptable: never touched
    wrote = []
    for _ in range(int(p.bright_s / 0.1) + 2):              # a window, for good
        t += 0.1
        wrote += c.update(meas(254.0, clip=0.02), t)
    assert wrote and c.corrections == 1
    t += 0.1
    c.update(meas(240.0), t)                                # back inside 235..247
    assert c.phase == "hold"
    assert c.update(meas(215.0), t + 0.1) == []             # darker, but acceptable: holds


def test_sticky_waits_out_its_cooldown_but_not_for_a_panic():
    p = Patience()
    c = Controller("sticky")
    c.begin(0.0)
    c.t_settled = 10.0
    t = 10.0
    for _ in range(int(p.dark_s / 0.1) + 2):                # dark long enough, inside cooldown
        t += 0.1
        if t - 10.0 < p.cooldown_s:
            assert c.update(meas(150.0), t) == []
    c2 = Controller("sticky")
    c2.begin(0.0)
    c2.t_settled = 10.0
    wrote = []
    for i in range(4):                                      # heavy clipping: at once
        wrote += c2.update(meas(255.0, clip=0.3), 10.05 + 0.1 * i)
    assert wrote


def test_pinned_writes_once_and_servo_follows_every_tick():
    c = Controller("pinned", pinned=(150, 30))
    assert dict(c.begin(0.0)) == {"ae_mode": 1, "gain": 30, "exposure": 150}
    assert c.update(meas(255.0, clip=0.5), 1.0) == []
    s = Controller("servo")
    s.begin(0.0)
    assert s.update(meas(200.0), 0.1)                       # below the band: climbs at once


# --- the runner (P2 without the arena) ------------------------------------------------------


@pytest.fixture
def quick(monkeypatch):
    monkeypatch.setattr(lens, "SETTLE_S", 0.2)


def test_the_runner_takes_over_from_the_fakes_blown_out_auto_exposure(quick):
    cfg = cam_cfg()
    cam = cfg.camera("wrist")
    runner = p2_camera.CameraRunner(cam, source=FakeCamera(cam.shape, fps=120), log=lambda s: None)
    px = np.empty(cam.shape, np.uint8)
    runner.start(px)
    assert runner.decision.used == "fit" and runner.controls_note == "ok"
    for _ in range(120 * 6):                                # 6 s of frames
        runner.frame(px)
    st = runner.status()["exposure"]
    assert st["owner"] == "ours" and st["takeover"] and st["phase"] == "hold"
    assert 230 <= st["p995"] <= 250 and st["clip_pct"] < 1.0
    owner, exp, _ = runner.exposure_in_effect()
    assert owner == schema.EXPOSURE_OURS and exp in exposure.EXPOSURE_BANDS
    ctl = runner.ctl
    runner.close()
    assert ctl.get("ae_mode") == FakeControls.RANGES["ae_mode"]["def"]    # handed back
    assert runner.control_faults == 0       # no write the real camera would refuse


def test_an_explicit_mode_without_a_control_channel_refuses(quick):
    class NoControls(FakeCamera):
        def controls(self):
            return None

    cam = cam_cfg(exposure="sticky").camera("wrist")
    runner = p2_camera.CameraRunner(cam, source=NoControls(cam.shape, fps=120),
                                    log=lambda s: None)
    with pytest.raises(sources.CameraError, match="control channel"):
        runner.start(np.empty(cam.shape, np.uint8))


# --- the ring and P2 as a process -----------------------------------------------------------


def test_the_camera_ring_publishes_whole_frames_with_their_square():
    cfg = cam_cfg()
    a = Arena.create(arena_name(), layout_for(cfg))
    try:
        ring = open_all(a)["cameras"]["wrist"]
        assert ring.shape == SHAPE and ring.peek() is None
        slot = ring.spare()
        slot["pixels"][...] = 9
        ring.publish(1.5, 1.6, (1, 2, 3, 4), owner=schema.EXPOSURE_OURS, exposure=200, gain=64)
        out = np.empty(SHAPE, np.uint8)
        meta = ring.read_newest(out)
        assert meta.frame_id == 1 and meta.crop == (1, 2, 3, 4) and out.min() == 9
        assert (meta.exposure, meta.gain, meta.exposure_owner) == (200, 64, 1)
        assert ring.read_newest(out, after=1) is None       # nothing newer
        ring.spare()                                        # being written: odd
        ring._head[0] = 2                                   # a reader racing the writer
        assert ring.read_newest(out, after=1) is None       # never torn
        ring.set_status('{"phase": "running"}')
        assert json.loads(ring.status()) == {"phase": "running"}
    finally:
        a.close()
        a.unlink()


def test_p2_streams_into_the_arena_until_stopped(quick):
    cfg = cam_cfg()
    name = arena_name()
    a = Arena.create(name, layout_for(cfg))
    rings = open_all(a)
    rings["control"].session = schema.CHECK
    ready = threading.Event()
    th = threading.Thread(target=p2_camera.run, args=(name, cfg), daemon=True, kwargs={
        "camera": "wrist", "ready": ready, "source": FakeCamera(SHAPE, fps=60)})
    th.start()
    try:
        assert ready.wait(10.0)
        ring = rings["cameras"]["wrist"]
        t_end = time.monotonic() + 5.0
        while ring.newest_id() < 10 and time.monotonic() < t_end:
            time.sleep(0.02)
        meta = ring.peek()
        assert meta.frame_id >= 10 and meta.crop[2] - meta.crop[0] == meta.crop[3] - meta.crop[1]
        st = json.loads(ring.status())
        assert st["phase"] == "running" and st["lens_used"] == "fit"
    finally:
        rings["control"].session = schema.STOPPED
        th.join(5.0)
        a.close()
        a.unlink()
    assert not th.is_alive()


# --- the profile ------------------------------------------------------------------------------


def test_the_profile_declares_the_wrist_camera_and_sets_reach_into_it():
    cfg = load()
    (wrist,) = cfg.cameras
    assert wrist.name == "wrist" and wrist.lens == "fisheye" and wrist.exposure == "camera_first"
    cfg = load(sets=['cameras.wrist.exposure="sticky"',
                     "cameras.wrist.fallback_lens_circle=[663, 334, 344]"])
    assert cfg.camera("wrist").exposure == "sticky"
    assert cfg.camera("wrist").fallback_lens_circle == (663.0, 334.0, 344.0)
    assert cfg.with_cameras(none=True).cameras == ()
    assert cfg.with_cameras(fake=True).camera("wrist").source == "fake"
    with pytest.raises(ConfigError, match="exposure"):
        load(sets=['cameras.wrist.exposure="auto"'])
    with pytest.raises(ConfigError, match="no table"):
        load(sets=['cameras.exo.exposure="camera"'])
    with pytest.raises(ConfigError, match="no key"):
        load(sets=["cameras.wrist.colour=1"])


def test_each_camera_has_its_own_rings_in_the_layout():
    cfg = load()
    lay = layout_for(cfg)
    assert lay.cameras == ("wrist",)
    assert lay.region("cam_wrist_rec").dtype["pixels"].shape == (720, 1280, 3)
    assert layout_for(cfg.with_cameras(none=True)).hash != lay.hash


# --- UVC and AVFoundation plumbing (no camera needed) -----------------------------------------


def test_usb_ids_and_locations():
    assert sources.parse_usb_uid("0x1421000010bb2b08") == (0x14210000, 0x10BB, 0x2B08)
    # as this Mac printed the wrist camera: the location is not zero-padded
    assert sources.parse_usb_uid("0x213000010bb2b08") == (0x02130000, 0x10BB, 0x2B08)
    assert sources.parse_usb_uid("1FD4B3A2-236E-492B-8CE5-255DD288CE50") is None
    assert sources.parse_usb_id("10bb:2b08@0x14210000") == (0x10BB, 0x2B08, 0x14210000)

    class Dev:
        bus, port_numbers = 2, (1, 1)

    assert uvc.location_of(Dev()) == 0x02110000            # the servo adapter, as ioreg says


def test_the_videocontrol_descriptors_name_the_units():
    header = bytes([13, 0x24, 0x01, 0x00, 0x01, 0, 0, 0, 0, 0, 0, 1, 1])
    camera = bytes([18, 0x24, 0x02, 1, 0x01, 0x02, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0x0A, 0, 0])
    pu = bytes([11, 0x24, 0x05, 2, 1, 0, 0, 2, 0x7F, 0x15, 0])

    class Intf:
        bInterfaceClass, bInterfaceSubClass, bInterfaceNumber = 0x0E, 0x01, 0
        extra_descriptors = list(header + camera + pu)

    assert uvc._vc_layout([[Intf()]]) == (0, {"ct": 1, "pu": 2})


def test_auto_picks_the_one_usb_camera_never_an_iphone():
    devices = [
        {"name": "FaceTime HD", "uid": "1FD4", "usb": None},
        {"name": "iPhone camera", "uid": "6435F80F-0001", "usb": None},
        {"name": "HX200S", "uid": "0x1421000010bb2b08", "usb": (0x14210000, 0x10BB, 0x2B08)},
    ]
    assert sources.resolve_avf("auto", devices)["name"] == "HX200S"
    assert sources.resolve_avf("10bb:2b08", devices)["name"] == "HX200S"
    assert sources.resolve_avf("facetime", devices)["name"] == "FaceTime HD"
    with pytest.raises(sources.CameraError, match="no a USB camera|no USB"):
        sources.resolve_avf("auto", devices[:2])
    twin = dict(devices[2], uid="0x1422000010bb2b08", usb=(0x14220000, 0x10BB, 0x2B08))
    with pytest.raises(sources.CameraError, match="2 cameras"):
        sources.resolve_avf("auto", devices + [twin])
    assert sources.resolve_avf("10bb:2b08@0x14220000", devices + [twin])["uid"] == twin["uid"]


# --- the dashboard ------------------------------------------------------------------------------


def test_the_dashboard_reports_and_streams_the_camera():
    from elrobot_deploy.dashboard.server import Source, serve

    cfg = cam_cfg()
    a = Arena.create(arena_name(), layout_for(cfg))
    rings = open_all(a)
    rings["control"].session = schema.ACTIVE
    ring = rings["cameras"]["wrist"]
    ring.set_status(json.dumps({"phase": "running", "exposure": {"owner": "camera"}}))
    slot = ring.spare()
    slot["pixels"][...] = 128
    ring.publish(time.monotonic(), time.monotonic(), (140, 0, 500, 360))
    httpd = serve(Source(cfg, rings), "127.0.0.1", 0)
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/api/state", timeout=5) as r:
            st = json.loads(r.read())
        assert st["cameras"]["wrist"]["phase"] == "running"
        assert st["cameras"]["wrist"]["age_s"] is not None
        with urllib.request.urlopen(base + "/camera/wrist.mjpg", timeout=5) as r:
            assert r.headers["Content-Type"].startswith("multipart/x-mixed-replace")
            assert r.readline() == b"--frame\r\n"
            headers = dict(r.readline().decode().strip().split(": ", 1) for _ in range(2))
            assert r.readline() == b"\r\n" and headers["Content-Type"] == "image/jpeg"
            jpg = r.read(int(headers["Content-Length"]))
        assert jpg[:2] == b"\xff\xd8"                           # one JPEG part
    finally:
        rings["control"].session = schema.STOPPED
        httpd.shutdown()
        httpd.server_close()
        a.close()
        a.unlink()
