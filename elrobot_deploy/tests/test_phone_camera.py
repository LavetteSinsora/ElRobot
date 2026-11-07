"""The iPhone camera (`runtime/cameras/phone_camera.py`, the phone app's Camera role): the wire, the
clock map, finding the phone, the profile keys, and P2's runner on it. `FakePhone` is the app's
side of the protocol on loopback, its clock deliberately far from ours."""

from __future__ import annotations

import json
import socket
import threading
import time

import cv2
import numpy as np
import pytest

from elrobot_deploy.config_loading.config import load
from elrobot_deploy.config_loading.errors import ConfigError
from elrobot_deploy.runtime.cameras import CameraError, lens, make_source
from elrobot_deploy.runtime.cameras import phone_camera as pc
from elrobot_deploy.runtime.processes import p2_camera
from elrobot_deploy.runtime.timing import now

SHAPE = (360, 640, 3)


class FakePhone:
    """The app on loopback: answers the hello, pongs, streams JPEGs captured `age_s` before
    they are sent, on a clock `offset_s` ahead of ours."""

    def __init__(self, *, offset_s=1234.5, age_s=0.020, fps=60.0, shape=SHAPE, refuse="",
                 dropped=0):
        self.offset_s, self.age_s, self.fps = offset_s, age_s, fps
        self.shape, self.refuse, self.dropped = shape, refuse, dropped
        self.hello: dict = {}
        self.sent: list[float] = []             # capture times, OUR clock (the truth)
        self.srv = socket.create_server(("127.0.0.1", 0))
        self.port = self.srv.getsockname()[1]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def clock(self) -> float:
        return now() + self.offset_s

    def _serve(self):
        try:
            conn, _ = self.srv.accept()
        except OSError:
            return
        lock = threading.Lock()

        def send(kind, payload=b""):
            with lock:
                conn.sendall(pc.pack(kind, payload))

        try:
            kind, payload = pc.read_message(conn)
            self.hello = json.loads(payload)
            h, w = self.shape[:2]
            if self.refuse:
                send("i", json.dumps({"v": 1, "ok": False, "error": self.refuse}).encode())
                return
            send("i", json.dumps({"v": 1, "ok": True, "model": "iPhone15,3", "name": "exo",
                                  "width": w, "height": h, "fps": self.fps}).encode())
            threading.Thread(target=self._frames, args=(send,), daemon=True).start()
            while not self._stop.is_set():
                kind, payload = pc.read_message(conn)
                if kind == "p":
                    send("q", pc.PONG.pack(pc.PING.unpack(payload)[0], self.clock()))
        except (OSError, ConnectionError):
            pass
        finally:
            conn.close()

    def _frames(self, send):
        h = self.shape[0]
        seq = 0
        try:
            while not self._stop.is_set():
                img = np.full(self.shape, (seq * 7) % 256, np.uint8)
                cv2.putText(img, str(seq), (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (255, 255, 255), 3)
                jpeg = cv2.imencode(".jpg", img)[1].tobytes()
                t_ours = now() - self.age_s
                self.sent.append(t_ours)
                send("f", pc.FRAME.pack(seq, self.dropped, t_ours + self.offset_s, 0.01, 100.0)
                     + jpeg)
                seq += 1
                time.sleep(1.0 / self.fps)
        except OSError:
            pass

    def close(self):
        self._stop.set()
        self.srv.close()


@pytest.fixture
def phone():
    made = []

    def make(**kw):
        p = FakePhone(**kw)
        made.append(p)
        return p

    yield make
    for p in made:
        p.close()


def test_frames_arrive_with_capture_times_on_our_clock(phone):
    p = phone()
    cam = pc.PhoneCamera(f"127.0.0.1:{p.port}", SHAPE, fps=60, quality=70, lens="ultrawide",
                         exposure="auto", rotate=180)
    cam.connect()
    try:
        assert p.hello == {"v": 1, "width": 640, "height": 360, "fps": 60.0, "quality": 70,
                           "lens": "ultrawide", "exposure": "auto", "rotate": 180}
        px = np.empty(SHAPE, np.uint8)
        ages, last = [], -1.0
        for _ in range(30):
            assert cam.read_into(px)
            assert cam.t_capture > last and cam.t_capture <= cam.t_recv
            last = cam.t_capture
            ages.append(cam.t_recv - cam.t_capture)
            # the truth: the fake stamped this capture `age_s` before sending, on its clock
            assert min(abs(cam.t_capture - t) for t in p.sent[-10:]) < 0.003
        assert np.median(ages) == pytest.approx(0.020, abs=0.005)
        assert cam.sync.offset == pytest.approx(1234.5, abs=0.002)
        assert "iPhone15,3" in cam.describe() and "clock synced" in cam.describe()
        assert cam.controls() is None
    finally:
        cam.close()
    assert not cam.read_into(px, timeout=0.2)               # closed: no more frames


def test_the_phones_own_drops_are_counted(phone):
    p = phone(dropped=7)
    cam = pc.PhoneCamera(f"127.0.0.1:{p.port}", SHAPE, fps=60)
    cam.connect()
    try:
        assert cam.dropped >= 7
    finally:
        cam.close()


def test_a_mode_the_phone_cannot_run_is_refused_with_its_reason(phone):
    p = phone(refuse="the wide camera has no 640x360 mode at 60 fps")
    with pytest.raises(CameraError, match="no 640x360 mode"):
        pc.PhoneCamera(f"127.0.0.1:{p.port}", SHAPE, fps=60).connect()


def test_a_phone_at_another_size_is_refused(phone):
    p = phone(shape=(720, 1280, 3))
    with pytest.raises(CameraError, match="runs 1280x720"):
        pc.PhoneCamera(f"127.0.0.1:{p.port}", SHAPE, fps=60).connect()


def test_the_phones_address_on_our_subnet_comes_first(monkeypatch):
    import elrobot_deploy.runtime.processes.p11_phone as p11
    monkeypatch.setattr(p11, "_local_ip", lambda: "192.168.1.16")
    assert pc._rank(["169.254.3.4", "192.168.2.1", "192.168.1.30"]) == [
        "192.168.1.30", "192.168.2.1", "169.254.3.4"]


def test_nobody_listening_says_so():
    s = socket.create_server(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    with pytest.raises(CameraError, match="cannot reach"):
        pc.PhoneCamera(f"127.0.0.1:{port}", SHAPE).connect()


def test_the_fastest_round_trip_sets_the_clock():
    s = pc.ClockSync(window_s=10.0)
    s.add(0.0, 100.0 + 0.030, 0.060)        # slow: 60 ms, its midpoint says offset 100.000
    s.add(1.0, 101.0 + 0.002 + 0.001, 1.004)    # fast: 4 ms, says 100.001
    assert s.offset == pytest.approx(100.001) and s.rtt == pytest.approx(0.004)
    s.add(20.0, 120.0 + 0.010, 20.020)      # 19 s later: the fast one has left the window
    assert s.offset == pytest.approx(100.0) and s.rtt == pytest.approx(0.020)
    assert s.to_local(130.0) == pytest.approx(30.0)


def test_device_names_an_address_or_a_bonjour_name():
    assert pc.parse_address("192.168.1.20") == ("192.168.1.20", pc.PORT)
    assert pc.parse_address("192.168.1.20:6000") == ("192.168.1.20", 6000)
    assert pc.parse_address("chris-iphone.local") == ("chris-iphone.local", pc.PORT)
    assert pc.parse_address("exo") is None and pc.parse_address("auto") is None

    one = [{"name": "exo", "host": "10.0.0.5", "port": 5006}]
    two = one + [{"name": "side", "host": "10.0.0.6", "port": 5006}]
    assert pc.resolve_phone("auto", browse=lambda: one) == (["10.0.0.5"], 5006, "exo")
    assert pc.resolve_phone("SIDE", browse=lambda: two) == (["10.0.0.6"], 5006, "side")
    assert pc.resolve_phone("192.168.1.20", browse=list) == (["192.168.1.20"], pc.PORT,
                                                             "192.168.1.20")
    with pytest.raises(CameraError, match="2 camera phones"):
        pc.resolve_phone("auto", browse=lambda: two)
    with pytest.raises(CameraError, match="no camera phone on the LAN"):
        pc.resolve_phone("auto", browse=list)
    with pytest.raises(CameraError, match="named 'top'"):
        pc.resolve_phone("top", browse=lambda: two)


def test_the_profile_keys_reach_the_binding():
    cfg = load(sets=['cameras.exo.phone_lens="ultrawide"', "cameras.exo.phone_quality=60",
                     'cameras.exo.phone_exposure="auto"', "cameras.exo.phone_rotate=180"])
    exo = cfg.camera("exo")
    src = make_source(exo)
    assert isinstance(src, pc.PhoneCamera)
    assert (src.lens, src.quality, src.phone_exposure, src.rotate) == ("ultrawide", 60, "auto",
                                                                       180)
    for bad in ('cameras.exo.phone_lens="tele"', 'cameras.exo.phone_exposure="manual"',
                "cameras.exo.phone_rotate=90", "cameras.exo.phone_quality=0",
                'cameras.exo.exposure="camera_first"'):
        with pytest.raises(ConfigError):
            load(sets=[bad])


def test_p2s_runner_streams_the_phone(phone, monkeypatch):
    monkeypatch.setattr(lens, "SETTLE_S", 0.2)
    p = phone()
    cam = load(sets=[f'cameras.exo.device="127.0.0.1:{p.port}"', "cameras.exo.width=640",
                     "cameras.exo.height=360", "cameras.exo.fps=60"]).camera("exo")
    runner = p2_camera.CameraRunner(cam, log=lambda s: None)
    px = np.empty(cam.shape, np.uint8)
    runner.start(px)
    try:
        assert runner.phase == "running" and runner.ctl is None
        for _ in range(20):
            assert runner.frame(px)
        st = runner.status()
        assert st["fps"] > 30 and "exo" in st["device"]         # the fake paces by sleep
    finally:
        runner.close()
