"""P2 / P3 / P4 — the camera readers.

Two things here fail silently and nothing else really does. If the decoder ever
writes into the buffer a reader is holding, the reader gets half of one frame
and half of another — a plausible image, at 30 Hz, forever. And if `t_recv` or
the published index lags the pixels, P5 anchors its observation to the wrong
instant, which looks exactly like a policy that reacts late.

So: which slot is being written relative to which is published, and whether
the published header describes the frame that is actually there. The split of a
side_by_side head into two views is tested here too, against real pixels.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from g1d_deploy.config_loading import resolved as cfgmod
from g1d_deploy.runtime import cpu_pinning as affinity
from g1d_deploy.runtime.processes import p2_camera
from g1d_deploy.runtime.shared_memory import schema
from g1d_deploy.runtime.drivers.cameras import (FakeCamera, RealSenseCamera, V4l2Camera,
                                                ZmqCamera, _fourcc_name, _is_jpeg)

SHAPE = (8, 8, 3)          # the conftest `rwrist` camera, at native resolution


class TracingCamera(FakeCamera):
    """A fake that records, per frame, the slot it was handed to decode into,
    the frame published at that moment, and every slot's counter.

    The slot is recorded as a raw address and resolved to an index only after
    the run. `run()` attaches its own mapping of the same segment, so the
    addresses it hands out are in a different address space from the fixture's
    — which is the reason the arena stores indices and never pointers.
    Within one mapping the slots are contiguous and equally sized, so the
    offset from the lowest observed address names the slot.
    """

    def __init__(self, shape, head, seq, **kw):
        super().__init__(shape, **kw)
        self.head = head
        self.counters = seq            # NOT `seq`: P2 reads that as the source counter
        #: (published frame_id, target address, counters at decode time)
        self.trace: list[tuple[int, int, np.ndarray]] = []

    def read_into(self, pixels):
        published = int(self.head[0])
        addr = pixels.__array_interface__["data"][0]
        counters = self.counters.copy()
        if not super().read_into(pixels):
            return False
        self.trace.append((published, addr, counters))
        return True

    def targets(self, stride: int) -> list[int]:
        base = min(addr for _, addr, _ in self.trace)
        out = []
        for _, addr, _ in self.trace:
            i, rem = divmod(addr - base, stride)
            assert rem == 0, "the decode target is not aligned to a slot"
            out.append(i)
        return out


# --- The ring's whole purpose --------------------------


def test_decodes_into_a_slot_that_is_not_the_published_one(arena, cfg, hw):
    """A ~1 MB frame is decoded in place, so the slot being written must never
    be the newest one a reader can be looking at — and it must SAY it is being
    written (odd counter) for as long as the decode runs."""
    cam = TracingCamera(SHAPE, arena.view("cam_rwrist_head"),
                        arena.view("cam_rwrist_seq"), n=6)
    p2_camera.run(arena.shm.name, cfg, hw, camera="rwrist", source=cam)

    assert len(cam.trace) == 6
    n = arena.layout.region("cam_rwrist_rec").count
    assert n == 3                                 # no history: the old triple
    targets = cam.targets(arena.layout.region("cam_rwrist_rec").dtype.itemsize)
    assert set(targets) == {0, 1, 2}              # all three are in the rotation
    for (published, _, counters), target in zip(cam.trace, targets):
        if published:
            assert target != (published - 1) % n
        assert counters[target] % 2 == 1, "decoding into a slot marked readable"
    # and consecutive frames move on, rather than ping-ponging onto the slot
    # published one frame ago.
    assert all(a != b for a, b in zip(targets, targets[1:]))
    # At EOF the loop had already opened the next slot for a decode that never
    # came: that one slot stays odd ("being written" — its old frame, the
    # oldest, is unreadable), every other one is even.
    seq = arena.view("cam_rwrist_seq")
    newest = int(arena.view("cam_rwrist_head")[0])
    assert [int(s) % 2 for s in seq] == [int(i == newest % n) for i in range(n)]


def test_the_published_header_describes_the_pixels_beside_it(arena, cfg, hw,
                                                             rings):
    """`peek` is what P5 polls to find the first frame after its trigger
    instant, and it reads only the header. The header therefore has to be the
    header *of the frame in that buffer* — which is why it lives inside the
    buffer rather than beside the region."""
    cam = FakeCamera(SHAPE, n=4)
    p2_camera.run(arena.shm.name, cfg, hw, camera="rwrist", source=cam)

    region = rings["cameras"]["rwrist"]
    t_recv, frame_id = region.peek()
    assert frame_id == 4                      # the newest, not the first
    assert t_recv == pytest.approx(cam.t_recv)

    out = region.empty()
    assert region.read_into(out)
    assert int(out["frame_id"]) == 4
    assert float(out["t_recv"]) == t_recv
    assert np.all(out["pixels"] == 4)         # FakeCamera's frame i is `1 + i`
    assert out["pixels"].shape == SHAPE       # native resolution, no resize


def test_arrival_stamps_are_monotone_and_precede_publication(arena, cfg, hw):
    """`t_recv` is stamped when the frame arrives, not when it is published, so
    it must sit before the publish that carries it — and the whole system is on
    one monotonic clock."""
    t0 = time.monotonic()
    cam = FakeCamera(SHAPE, n=5, delay=1e-3)
    p2_camera.run(arena.shm.name, cfg, hw, camera="rwrist", source=cam)
    t1 = time.monotonic()

    stamps = sorted(float(b["t_recv"]) for b in arena.view("cam_rwrist_rec")
                    if int(b["frame_id"]) > 0)
    assert len(stamps) == 3                   # three slots, five frames
    assert stamps == sorted(set(stamps))
    assert t0 < stamps[0] and stamps[-1] < t1


def test_the_loop_never_runs_once_the_session_has_stopped(arena, cfg, hw, rings):
    rings["control"].session = schema.STOPPED
    cam = FakeCamera(SHAPE, n=5)
    p2_camera.run(arena.shm.name, cfg, hw, camera="rwrist", source=cam)

    assert cam.i == 0
    assert cam.closed
    assert not arena.view("cam_rwrist_rec")["frame_id"].any()


# --- which binding a camera gets ----------------------------------


def _camera(cfg, name, spec):
    return cfgmod.replace(cfg, cameras={**cfg.cameras, name: spec})


def test_the_spec_chooses_the_binding(cfg, hw):
    """A rig's cameras may all be on the deploy PC — two fisheye wrists on
    USB and a RealSense — so which class P2 constructs is a property of the
    task's config, not of the camera's name."""
    zmq_cfg = _camera(cfg, "rwrist", cfgmod.CameraSpec(8, 8))
    assert isinstance(p2_camera.make_source(zmq_cfg, hw, "rwrist"), ZmqCamera)

    v4l2_cfg = _camera(cfg, "rwrist", cfgmod.CameraSpec(
        720, 1280, source="v4l2", device="/dev/v4l/by-path/pci-0-usb-0:1"))
    source = p2_camera.make_source(v4l2_cfg, hw, "rwrist")
    assert isinstance(source, V4l2Camera)
    assert source.device == "/dev/v4l/by-path/pci-0-usb-0:1"
    assert source.shape == (720, 1280, 3)

    rs_cfg = _camera(cfg, "exo", cfgmod.CameraSpec(
        480, 640, source="realsense", device="342222071420"))
    source = p2_camera.make_source(rs_cfg, hw, "exo")
    assert isinstance(source, RealSenseCamera)
    assert source.serial == "342222071420"
    assert source.shape == (480, 640, 3)


def test_p4_is_the_heads_alone(hw):
    """P4 belongs to the head and is shared with nothing.

    `head` and `exo` used to map here together, on the written assumption that
    no configuration declares both. The head is now recorded on every rollout,
    so that assumption is gone: two 30 Hz decode loops on one core is not a
    refusal anywhere, it is one camera's frames arriving late.
    """
    assert p2_camera.PROCESS["head"] == "p4_camera"
    assert "p4_camera" in hw.cpu
    assert "exo" not in p2_camera.PROCESS


def test_an_unlisted_camera_is_pinned_by_its_own_name(hw):
    """Which is why `exo` needs no entry above, and no code change to get a core.

    `run` pins `PROCESS.get(camera, camera)`, so a camera absent from the table
    resolves to its own name and `[cpu.pin] exo = <core>` in the machine file
    is the whole change — tests/machine.toml states exactly that, 17. This is
    the mechanism the comment in `p2_camera` promises, asserted rather than
    described.
    """
    assert p2_camera.PROCESS.get("exo", "exo") == "exo"
    assert set(affinity.cores_for(hw, "exo")) == {17}

    # And an unpinned camera is a camera that RUNS, not one that refuses: the
    # fallback is only useful if a name with no core is survivable.
    assert affinity.cores_for(hw, "table") in (None, (), [], set())
    assert affinity.pin(hw, "table") is False


# --- The real bindings --------------------------------------------


def test_fourcc_round_trips():
    """`_verify_format` compares the driver's negotiated FOURCC against the
    requested one, and that comparison is the whole guard against the YUYV
    fallback — which delivers ~5 fps rather than refusing."""
    cv2 = pytest.importorskip("cv2")
    from g1d_deploy.runtime.drivers.cameras import _fourcc

    for name in ("MJPG", "YUYV"):
        assert _fourcc_name(int(_fourcc(cv2, name))) == name


def test_jpeg_detection_decides_whether_the_recorder_gets_frames():
    """`CAP_PROP_CONVERT_RGB = 0` is driver-dependent, so V4l2Camera probes for
    it rather than assuming. Decoded pixels must not be mistaken for a JPEG."""
    cv2 = pytest.importorskip("cv2")

    ok, encoded = cv2.imencode(".jpg", np.zeros(SHAPE, np.uint8))
    assert ok
    assert _is_jpeg(encoded)
    assert not _is_jpeg(np.zeros(SHAPE, np.uint8))
    assert not _is_jpeg(np.zeros(0, np.uint8))


def test_v4l2_camera_refuses_a_device_it_cannot_open():
    """The message has to name the by-path directory: the failure a person
    actually hits is a node number that moved across a reboot."""
    pytest.importorskip("cv2")
    cam = V4l2Camera("/dev/v4l/by-path/no-such-camera", SHAPE)
    with pytest.raises(ValueError, match="by-path"):
        cam.connect()


# --- The ZMQ binding ----------------------------------------------


def test_zmq_camera_decodes_a_real_jpeg_into_the_region(arena, rings):
    """The wire format is raw JPEG bytes on a SUB socket — no header, no
    timestamp, no sequence number — so this drives the actual binding against a
    local PUB socket rather than trusting the fake.

    The frame is compared by content, not bytes: JPEG is lossy, and asserting
    equality would only prove that nothing decoded at all.
    """
    cv2 = pytest.importorskip("cv2")
    zmq = pytest.importorskip("zmq")

    frame = np.zeros(SHAPE, np.uint8)
    frame[:4, :, 0] = 200
    frame[:, :4, 2] = 120
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    payload = encoded.tobytes()

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    port = pub.bind_to_random_port("tcp://127.0.0.1")
    stop = threading.Event()

    def publish():                            # SUB is a slow joiner: keep sending
        while not stop.is_set():
            pub.send(payload)
            time.sleep(5e-3)

    thread = threading.Thread(target=publish, daemon=True)
    thread.start()

    cam = ZmqCamera("127.0.0.1", port, SHAPE)
    wide = ZmqCamera("127.0.0.1", port, (8, 16, 3))
    try:
        cam.connect()
        wide.connect()
        cam._sock.setsockopt(zmq.RCVTIMEO, 5000)    # fail, never hang
        wide._sock.setsockopt(zmq.RCVTIMEO, 5000)

        region = rings["cameras"]["rwrist"]
        dst = region.spare()["pixels"]
        assert cam.read_into(dst)
        assert cam.t_recv > 0.0
        assert np.abs(dst.astype(int) - frame.astype(int)).mean() < 12

        # A head frame arriving where a wrist frame is expected is wrong by
        # exactly 2x in width, and silently cropped it looks like an image.
        with pytest.raises(ValueError, match="the arena region is"):
            wide.read_into(np.zeros((8, 16, 3), np.uint8))
    finally:
        stop.set()
        thread.join(timeout=1.0)
        cam.close()
        wide.close()
        pub.close()
        ctx.term()


# --- The sequence number (fish_eye_image_server) -----------------------------
#
# Two publishers speak to `ZmqCamera` and it tells them apart by the bytes: our
# board server prefixes a 16-byte header, the vendor's `teleimager` sends a bare
# JPEG. Getting that discrimination wrong is ugly in both directions — 16 bytes
# stripped off a bare JPEG is a corrupt image, and 16 left on a headered one is
# the same — so it is tested from both sides rather than one.


class _SocketStub:
    """Stands in for the SUB socket, returning canned payloads in order."""

    def __init__(self, payloads):
        self._payloads = list(payloads)

    def recv(self):
        return self._payloads.pop(0)


def _jpeg(shape) -> bytes:
    import cv2
    ok, buf = cv2.imencode(".jpg", np.zeros(shape, dtype=np.uint8))
    assert ok
    return buf.tobytes()


def _zmq_camera_reading(payloads, shape=SHAPE):
    cam = ZmqCamera("127.0.0.1", 1, shape)
    cam._sock = _SocketStub(payloads)
    return cam


def test_zmq_camera_strips_our_header_and_reports_the_sequence_number():
    from g1d_deploy.runtime.drivers.board.fisheye_image_server.publisher import (pack_header)

    jpeg = _jpeg(SHAPE)
    cam = _zmq_camera_reading([pack_header(7) + jpeg])
    pixels = np.zeros(SHAPE, dtype=np.uint8)

    assert cam.read_into(pixels)
    assert cam.seq == 7
    # The recorder must get the JPEG, never the header — a second lossy
    # generation is bad, but a corrupt first byte is worse.
    assert cam.jpeg == jpeg


def test_zmq_camera_still_reads_the_vendor_servers_bare_jpeg():
    """`teleimager` publishes no header at all, and stock rigs use it."""
    jpeg = _jpeg(SHAPE)
    cam = _zmq_camera_reading([jpeg])
    pixels = np.zeros(SHAPE, dtype=np.uint8)

    assert cam.read_into(pixels)
    assert cam.seq is None            # not 0 — "no counter" is not "frame zero"
    assert cam.jpeg == jpeg


def test_a_jpeg_can_never_be_mistaken_for_our_header():
    """The discrimination is exact, not heuristic: SOI is FF D8, magic is G1DF."""
    from g1d_deploy.runtime.drivers.board.fisheye_image_server.publisher import MAGIC

    assert _jpeg(SHAPE)[:2] == b"\xff\xd8"
    assert MAGIC[:2] != b"\xff\xd8"


def test_the_sequence_number_reaches_the_arena_beside_its_own_frame(arena, cfg):
    """`source_seq` rides in the slot with the pixels it numbers."""
    from g1d_deploy.runtime.shared_memory import open_all

    region = open_all(arena, cfg)["cameras"]["rwrist"]
    region.publish(1.0, b"\xff\xd8ab", seq=41)
    region.publish(2.0, b"\xff\xd8cd", seq=42)

    frame_id, seq = region.peek_counters()
    assert (frame_id, seq) == (2, 42)


def test_a_source_without_a_counter_records_minus_one_not_zero(arena, cfg):
    """Zero is a real sequence number. Conflating it with "absent" would make
    CHECK report a fabricated drop count for the vendor server."""
    from g1d_deploy.runtime.shared_memory import open_all

    region = open_all(arena, cfg)["cameras"]["rwrist"]
    region.publish(1.0, None, seq=None)
    assert region.peek_counters() == (1, -1)


# --- UVC control read-back tolerance ----------------------------------------


class _Ctl:
    def __init__(self, step, lo, hi):
        self.step, self.min_val, self.max_val = step, lo, hi


def test_a_bitmask_controls_nonsense_step_does_not_disable_the_readback_check():
    """MEASURED on the wrist units: `Auto Exposure Mode` is [1..8] with step 9.

    Trusting that step would accept any in-range value, so a declined write to
    the one control that gates all the others — auto-exposure must be off before
    a manual exposure time sticks — would have passed silently.
    """
    from g1d_deploy.runtime.drivers.board.fisheye_image_server.publisher import (_tolerance)

    assert _tolerance(_Ctl(step=9, lo=1, hi=8)) == 0


def test_a_real_quantum_is_still_allowed():
    """`Pan control` steps by 3600 over +-57600, `White Balance temperature` by
    10 over 2800..6500. Those snaps are the driver working, not refusing."""
    from g1d_deploy.runtime.drivers.board.fisheye_image_server.publisher import (_tolerance)

    assert _tolerance(_Ctl(step=3600, lo=-57600, hi=57600)) == 3600
    assert _tolerance(_Ctl(step=10, lo=2800, hi=6500)) == 10
    assert _tolerance(_Ctl(step=1, lo=1, hi=500)) == 0      # exact


# --- photometry modes -------------------------------------------------------


def test_servo_is_the_default_photometry_mode():
    """Was `auto` until 2026-09-20, on an argument whose facts did not hold.

    The old reasoning: training jittered brightness 0.3 / contrast 0.4 over
    every frame, so landing anywhere inside that band is enough, and an
    adaptive camera stays inside it. What was then measured:

      * `auto` does not land inside the band, it lands OUTSIDE the sensor —
        it meters the whole frame, ~60% of which is the fisheye's black
        surround, and drives 26.3% of the actual picture past 250.
      * the jitter is a monotone [0,1]->[0,1] curve, so it never produced a
        clipped training frame and cannot undo one at inference.

    `servo` keeps what was right about that argument — adapt to the room,
    don't trust device integers copied off another camera model — and meters
    the lens circle instead of the whole sensor.
    """
    from g1d_deploy.runtime.drivers.board.fisheye_image_server.publisher import (DEFAULT_PHOTOMETRY,
                                                                                 resolve_controls)

    assert DEFAULT_PHOTOMETRY == "servo"
    mode, _ = resolve_controls("lwrist", {"circle": {"cx": 1, "cy": 1, "r": 1}})
    assert mode == "servo"


def test_a_camera_without_a_circle_falls_back_to_auto():
    """The head is rectilinear: the whole frame IS picture, so its own
    auto-exposure meters the right thing and measures 0.16% clipped. There is
    also no circle for the servo to meter. Falling back rather than refusing
    is what lets `servo` be the file-wide default."""
    from g1d_deploy.runtime.drivers.board.fisheye_image_server.publisher import (resolve_controls)

    mode, controls = resolve_controls(
        "head", {"controls_auto": {"Auto Exposure Mode": 8}})
    assert mode == "auto"
    assert controls["Auto Exposure Mode"] == 8


def test_each_mode_selects_its_own_control_block():
    from g1d_deploy.runtime.drivers.board.fisheye_image_server.publisher import (resolve_controls)

    spec = {"controls_auto": {"Auto Exposure Mode": 8},
            "controls_pinned": {"Auto Exposure Mode": 1,
                                "Absolute Exposure Time": 110}}
    assert resolve_controls("w", {**spec, "photometry": "auto"}) == (
        "auto", {"Auto Exposure Mode": 8})
    assert resolve_controls("w", {**spec, "photometry": "pinned"})[1][
        "Absolute Exposure Time"] == 110


def test_an_unknown_photometry_mode_is_refused():
    """A typo must not silently fall back to driver defaults — that is a
    photometry nobody chose."""
    import pytest as _pytest

    from g1d_deploy.runtime.drivers.board.fisheye_image_server.publisher import (resolve_controls)

    with _pytest.raises(RuntimeError, match="photometry"):
        resolve_controls("w", {"photometry": "atuo"})


def test_the_shipped_config_servos_the_wrists_and_autos_the_head():
    from g1d_deploy.runtime.drivers.board.fisheye_image_server import publisher as pub

    cfg = pub.load_config(pub.DEFAULT_CONFIG)
    for topic, spec in cfg["cameras"].items():
        mode, controls = pub.resolve_controls(topic, spec)
        if "wrist" in topic:
            assert mode == "servo", topic
            # Manual is ASSERTED, not merely left unset. While auto-exposure
            # is on, the driver accepts writes to exposure and gain and then
            # ignores them — the servo would run, log plausible numbers, and
            # change nothing at all.
            assert controls["Auto Exposure Mode"] == 1, topic
            # White balance pinned, so the loop has
            # one variable to chase rather than two.
            assert controls["White Balance temperature,Auto"] == 0, topic
            assert controls["White Balance temperature"] == 3800, topic
            # Exposure and gain must NOT be here: the loop owns them, and a
            # value in this block would be overwritten within 100 ms while
            # looking authoritative in the config.
            assert "Absolute Exposure Time" not in controls, topic
            assert "Gain" not in controls, topic
        else:
            assert mode == "auto", topic
            assert controls["Auto Exposure Mode"] == 8, topic
            assert controls["White Balance temperature,Auto"] == 1, topic


def test_the_wrists_declare_the_circle_the_servo_meters():
    """Without a circle a wrist silently falls back to `auto`, which is the
    exact failure this whole mode exists to remove — so the config having one
    is part of the contract, not an optional extra.

    The values must agree with the robot file whose `crop_default` the model
    falls back to: one decides what the model sees, the other what the loop
    exposes for. Both are the reference robot's fit of 2026-09-21.
    """
    from g1d_deploy.config_loading import robot_file as robot
    from g1d_deploy.runtime.drivers.board.fisheye_image_server import publisher as pub

    cfg = pub.load_config(pub.DEFAULT_CONFIG)
    cams = robot.load_robot("g1d_fisheye_dex1").cameras

    for topic, arena in (("left_wrist_camera", "lwrist"),
                         ("right_wrist_camera", "rwrist")):
        circle = cfg["cameras"][topic].get("circle")
        assert circle, topic
        fitted = dict(zip(("cx", "cy", "r"), cams[arena].fallback_lens_circle))
        for key in ("cx", "cy", "r"):
            assert circle[key] == pytest.approx(fitted[key], abs=0.01), (
                f"{topic}.{key}: cameras.yaml says {circle[key]}, the robot "
                f"file says {fitted[key]} — these must be the same circle")
        # and the board serves the shape the robot file declares
        assert list(cfg["cameras"][topic]["image_shape"]) == [
            cams[arena].height, cams[arena].width], topic


def test_the_wrists_carry_the_collection_settings_for_pinned():
    """The `pinned` block holds one recorded set of manual controls, so
    switching modes needs no archaeology."""
    from g1d_deploy.runtime.drivers.board.fisheye_image_server import publisher as pub

    cfg = pub.load_config(pub.DEFAULT_CONFIG)
    for topic in ("left_wrist_camera", "right_wrist_camera"):
        pinned = pub.resolve_controls(
            topic, {**cfg["cameras"][topic], "photometry": "pinned"})[1]
        assert pinned["Auto Exposure Mode"] == 1        # manual
        assert pinned["Absolute Exposure Time"] == 110
        assert pinned["Gain"] == 64
        assert pinned["White Balance temperature,Auto"] == 0
        assert pinned["White Balance temperature"] == 3800
