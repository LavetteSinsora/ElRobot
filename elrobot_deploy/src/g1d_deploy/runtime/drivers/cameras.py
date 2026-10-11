"""The camera wire: three bindings, one contract, decoded into the arena.

Opened by P2 (one process per camera) and by P0's lens fit. The robot-file side
is `docs/setup/robot_toml.md`, "`[cameras.<name>]`".

`ZmqCamera` is a SUB socket per camera onto the robot board's image server,
subscribe-all; from the vendor's server **every message is a raw JPEG frame**
— no header, no timestamp, no sequence number. That absence is exactly why `t_recv` has to be stamped on arrival and
why the camera-to-joint offset is a measured constant rather than something read
off the wire.

`V4l2Camera` and `RealSenseCamera` are for cameras plugged into the **deploy
PC**, not the robot board: e.g. fisheye wrists on USB, or an Intel RealSense
for a fixed exocentric view, which the board's image server cannot serve.

**All three sources publish BGR**, which is cv2's convention and therefore
`ZmqCamera`'s. The RealSense is asked for `bgr8` for that reason alone — it
would just as happily give RGB. One convention at the wire is worth a redundant
channel flip downstream, because the alternative is a per-camera colour
flag threaded through the arena, and a colour convention that varies by source
is a skew that raises nothing and reads as a weak policy. The conversion to
RGB happens once, in `policy_contract/observation.py`.

Three details are deliberate:

**`read_into` writes into a caller-provided view**, which is the spare slot of
the camera's ring. The point is that a ~1 MB frame
is never copied between the decoder and the reader, so the
destination has to reach the decoder rather than the decoder returning an array.

**The source stamps `t_recv`, and only this source does.** Everywhere else the
loop owns the clock; here it cannot. The stamp belongs *between* the arrival of
the frame and its decode, and the seam between those two is inside `recv` +
`imdecode` where only the driver can see it. Stamping in the loop instead would
add the whole JPEG decode — milliseconds, not microseconds — to every
`t_img_recv`, and `t_img_recv` is what anchors the entire observation.
The clock is still the one clock: `g1d_deploy.runtime.timing.now`.

**Native resolution, and the bytes as the source produced them.** No resize and
no colour conversion beyond the one BGR convention above: the head arrives as
one 480x1280 binocular frame, a stock wrist as 480x640, a USB fisheye as
720x1280. Any transform belongs to the policy, where it can match what the
policy expects, and the head's split into two eyes belongs to P2, after
the decode. A resize here would be a resize no policy could undo.
"""

from __future__ import annotations

import time

import numpy as np

from g1d_deploy.runtime.timing import now

# The image server never queues on our behalf: one slot, and the newest frame
# wins. A frame that sat in a receive queue is a frame whose arrival stamp is a
# lie by however long it waited, and P5's anchor is that stamp.
RCVHWM = 1


# What P2/P3/P4 need from a camera, implemented by `ZmqCamera` and `FakeCamera`
# below: `connect()`, `read_into(pixels) -> bool` (block for one frame, decode
# into `pixels`, False at EOF), `close()`, and a `t_recv` attribute holding the
# arrival of the frame `read_into` last decoded.
#
# `t_recv` is the one place in the wire layer that touches the clock, and it has
# to be: the arrival/decode seam is inside `recv()`, so stamping it in the loop
# instead would fold milliseconds of decode into the instant that anchors the
# whole observation.
#
# `jpeg` holds the bytes that produced the last decoded frame, or None. It is a
# reference, not a copy — the copy happens once, in P2, into the arena slot.


class ZmqCamera:
    """One camera off an image server's ZMQ PUB socket, at native resolution.

    `shape` is the configured native resolution. It is checked against every
    decoded frame rather than trusted: camera regions are not uniform, and a
    head frame arriving where a wrist frame is expected is wrong by exactly 2x
    in width — which, silently cropped, looks like a plausible image.

    **Two publishers speak to this class, and it tells them apart by the bytes.**
    The vendor `teleimager` sends a bare JPEG. Our own
    `g1d_deploy/runtime/drivers/board/fisheye_image_server/` prefixes a 16-byte header carrying a
    sequence number. A JPEG begins with the SOI marker `FF D8` and can therefore
    never be mistaken for the header's `G1DF` magic, so the discrimination is
    exact rather than heuristic — and getting it wrong in either direction is
    ugly: 16 bytes stripped off a bare JPEG is a corrupt image, and 16 bytes left
    on a headered one is the same.

    `parse_header` is IMPORTED from the board package rather than reimplemented,
    so the wire format has exactly one definition. That import is safe in both
    directions: `publisher.py`'s module scope is stdlib-only (`struct` and
    friends), and it imports nothing from `g1d_deploy`, so it stays self-contained
    when it is copied to the robot board where `g1d_deploy` does not exist.
    """

    def __init__(self, host: str, port: int, shape: tuple[int, int, int]):
        self.host = str(host)
        self.port = int(port)
        self.shape = tuple(int(v) for v in shape)
        self.t_recv = 0.0
        self.jpeg: bytes | None = None      # the compressed source of the last frame
        #: The publisher's counter for the last frame, or None from a source
        #: that has none. Gaps in it are frames ZMQ dropped at its high-water
        #: mark; CHECK turns that into a number.
        self.seq: int | None = None
        self._ctx = None
        self._sock = None

    @property
    def device(self) -> str:
        """`"host:port"` — this camera's address, in the one form that names it.

        Spelled the same way `V4l2Camera.device` is, so anything that records
        "which camera produced this" can take it off the source without knowing
        which transport it is.
        """
        return f"{self.host}:{self.port}"

    def connect(self) -> None:
        import zmq

        self._ctx = zmq.Context()
        sock = self._ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.RCVHWM, RCVHWM)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(f"tcp://{self.host}:{self.port}")
        sock.setsockopt_string(zmq.SUBSCRIBE, "")
        self._sock = sock

    def read_into(self, pixels: np.ndarray) -> bool:
        import cv2

        from g1d_deploy.runtime.drivers.board.fisheye_image_server.publisher import (parse_header)

        payload = self._sock.recv()             # blocks until a frame arrives
        self.t_recv = now()                     # stamped before the decode
        # Split the header off BEFORE anything else touches the bytes, so that
        # everything downstream — the decode, the recorder's copy, the shape
        # check — sees the JPEG and only the JPEG.
        self.seq, payload = parse_header(payload)
        # The compressed frame, kept rather than dropped. It costs a reference
        # here and one ~50 KB memcpy in P2, and it is what lets the recorder
        # store what the robot actually sent instead of a re-encoding of our
        # decode of it.
        self.jpeg = payload
        frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8),
                             cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(
                f"camera {self.host}:{self.port} sent {len(payload)} bytes that "
                f"are not a decodable image")
        if frame.shape != self.shape:
            raise ValueError(
                f"camera {self.host}:{self.port} sent {frame.shape}, the arena "
                f"region is {self.shape}. The region is sized from the config's "
                f"native resolution and cameras are not uniform.")
        np.copyto(pixels, frame)
        return True

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        if self._ctx is not None:
            self._ctx.term()
            self._ctx = None


class V4l2Camera:
    """A UVC camera on the deploy PC's own USB, at native resolution.

    Built for USB fisheye wrists (MSK R4231) plugged into the machine that
    runs the policy.

    **MJPG is forced, and that is a bandwidth requirement, not a preference.**
    At 1280x720 the uncompressed YUYV alternative is 2 bytes per pixel, i.e.
    55 MB/s per camera at 30 fps, which does not fit down a shared USB 2.0 bus.
    A driver that falls back to YUYV does not error — it quietly delivers ~5 fps,
    and a camera stream at a sixth of its declared rate looks like a policy that
    reacts late rather than like a misconfigured camera. So the negotiated format
    and the *achieved* frame rate are both verified at `connect`, where the
    failure is a refusal rather than a rollout nobody can explain.

    **`t_recv` is stamped between `grab` and `retrieve`**, which is exactly the
    seam the stamp belongs at and which V4L2 gives us properly: `grab` blocks for the
    frame's arrival and `retrieve` does the decode, so unlike the ZMQ path there
    is no guesswork about where the boundary is.

    The compressed frame is kept for the recorder by asking OpenCV
    not to convert — `CAP_PROP_CONVERT_RGB = 0` hands back the MJPEG buffer
    itself. That is driver-dependent, so it is *probed* rather than assumed: if
    the first frame back is not a JPEG, the camera is reopened in OpenCV's
    ordinary converting mode and `jpeg` stays None. Pixels are identical either
    way — both paths end at `cv2.imdecode` (the difference from another
    decoder, e.g. PIL, is well under one LSB). A resize is where
    interpolation actually matters, and that lives in the policy.
    """

    #: Below this fraction of the requested rate, refuse. Generous, because it
    #: is separating "30 fps" from "the YUYV fallback's 5", not measuring jitter.
    MIN_FPS_RATIO = 0.6
    #: Frames to time at connect. ~0.5 s at 30 fps — enough to tell those two
    #: apart, short enough to pay on every process start.
    FPS_PROBE_FRAMES = 15

    def __init__(self, device: str, shape: tuple[int, int, int],
                 *, fps: float = 30.0, fourcc: str = "MJPG"):
        self.device = str(device)
        self.shape = tuple(int(v) for v in shape)
        self.fps = float(fps)
        self.fourcc = str(fourcc)
        self.t_recv = 0.0
        self.jpeg: bytes | None = None
        #: True once the raw-MJPEG probe has succeeded. Read it to know whether
        #: the recorder is getting compressed frames from this camera.
        self.raw_mode = False
        self._cap = None

    # --- connect ----------------------------------------------------------

    def connect(self) -> None:
        import cv2

        self._open(cv2, raw=True)
        if not self._probe_raw():
            # The driver ignored CONVERT_RGB. Reopen rather than flipping the
            # property on a running capture: some backends only apply format
            # changes at open, and a half-applied one is the worst of both.
            self._cap.release()
            self._open(cv2, raw=False)
        self._verify_rate()

    def _open(self, cv2, *, raw: bool) -> None:
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise ValueError(
                f"cannot open camera {self.device!r}. `ls -l /dev/v4l/by-path/` "
                f"lists what is actually attached; the node numbering under "
                f"/dev/videoN is not stable across reboots, which is why this "
                f"takes a by-path node.")
        # FOURCC first. Setting the resolution before the format lets the driver
        # pick a mode for the OLD format and then keep it.
        cap.set(cv2.CAP_PROP_FOURCC, _fourcc(cv2, self.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.shape[1])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.shape[0])
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        # One frame of depth: the newest frame wins, for the same reason the ZMQ
        # socket sets RCVHWM = 1. A frame that sat in a queue is a frame whose
        # arrival stamp is a lie by however long it waited, and P5's anchor is
        # that stamp.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if raw:
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        self._cap = cap
        self.raw_mode = raw
        self._verify_format(cv2)

    def _verify_format(self, cv2) -> None:
        got = _fourcc_name(int(self._cap.get(cv2.CAP_PROP_FOURCC)))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        if got != self.fourcc:
            raise ValueError(
                f"camera {self.device!r} negotiated {got!r}, not "
                f"{self.fourcc!r}. At {width}x{height} the uncompressed "
                f"alternative does not fit down USB 2.0 and the driver will "
                f"silently deliver a fraction of the requested frame rate. "
                f"`v4l2-ctl -d {self.device} --list-formats-ext` shows which "
                f"modes this unit actually offers.")
        if (height, width) != self.shape[:2]:
            raise ValueError(
                f"camera {self.device!r} negotiated {width}x{height}, but the "
                f"arena region is {self.shape[1]}x{self.shape[0]}. The region "
                f"is sized from the config's native resolution, and a wrist "
                f"stream at the wrong resolution is a different fisheye "
                f"projection — the crop box fitted for this rig would not "
                f"describe it.")

    def _probe_raw(self) -> bool:
        """Did `CONVERT_RGB = 0` actually hand back the compressed frame?"""
        for _ in range(5):              # the first frames after open can be empty
            ok, payload = self._cap.read()
            if ok and payload is not None and payload.size:
                return _is_jpeg(payload)
        return False

    def _verify_rate(self) -> None:
        """Time real frames. The negotiated FOURCC can be right and the rate
        still wrong — a driver that cannot sustain the mode drops frames rather
        than refusing it."""
        t0 = time.monotonic()
        for _ in range(self.FPS_PROBE_FRAMES):
            if not self._cap.grab():
                raise ValueError(
                    f"camera {self.device!r} stopped delivering frames during "
                    f"the rate probe, {self.FPS_PROBE_FRAMES} frames after "
                    f"opening.")
        measured = self.FPS_PROBE_FRAMES / max(time.monotonic() - t0, 1e-9)
        if measured < self.MIN_FPS_RATIO * self.fps:
            raise ValueError(
                f"camera {self.device!r} delivers {measured:.1f} fps against a "
                f"requested {self.fps:.1f}. The format negotiated as "
                f"{self.fourcc!r}, so this is the bus or the sensor mode, not "
                f"the pixel format — check what else shares the USB controller.")

    # --- the loop ---------------------------------------------------------

    def read_into(self, pixels: np.ndarray) -> bool:
        import cv2

        if not self._cap.grab():        # blocks until a frame arrives
            return False
        self.t_recv = now()             # after arrival, before decode
        ok, payload = self._cap.retrieve()
        if not ok or payload is None:
            return False

        if self.raw_mode:
            # A reference for the recorder, and one decode for the pixels. The
            # copy into the arena slot happens once, in P2.
            self.jpeg = payload.tobytes()
            frame = cv2.imdecode(payload.reshape(-1), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(
                    f"camera {self.device!r} delivered {payload.size} bytes "
                    f"that are not a decodable image")
        else:
            frame = payload

        if frame.shape != self.shape:
            raise ValueError(
                f"camera {self.device!r} produced {frame.shape}, the arena "
                f"region is {self.shape}. The region is sized from the config's "
                f"native resolution and cameras are not uniform.")
        np.copyto(pixels, frame)
        return True

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class RealSenseCamera:
    """An Intel RealSense colour stream, straight off the deploy PC's USB.

    For an exocentric view. There is no JPEG anywhere on this path — the SDK
    hands over raw pixels and they go into the arena unchanged, so `jpeg` stays
    None and the recorder stores this camera as "not captured", which is the
    same case a dropped frame already produces.

    **The resolution is a field of view, not a pixel count.** The D435I's colour
    sensor is natively 4:3; its 16:9 modes crop vertically. A policy made for
    one mode sees a different field of view (and different letterboxing after
    its own resize) in another, so the robot file's `shape` must be the mode
    the policy expects.

    `bgr8` is requested because this package publishes BGR from every source;
    the sensor would give RGB just as happily, and the redundant flip is the
    price of having one convention at the wire.
    """

    #: `wait_for_frames` raises on timeout rather than blocking forever. There
    #: are no watchdogs in this system by design, but librealsense
    #: gives this one away, and a dead USB camera that raises beats one that
    #: stalls P2 silently.
    TIMEOUT_MS = 5000

    def __init__(self, shape: tuple[int, int, int], *, serial: str | None = None,
                 fps: float = 30.0):
        self.shape = tuple(int(v) for v in shape)
        self.serial = str(serial) if serial else None
        self.fps = float(fps)
        self.t_recv = 0.0
        self.jpeg: bytes | None = None
        self._pipeline = None

    def connect(self) -> None:
        import pyrealsense2 as rs

        height, width, _ = self.shape
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8,
                             int(self.fps))
        self._pipeline = rs.pipeline()
        profile = self._pipeline.start(config)

        # Newest frame wins, as with the ZMQ socket's RCVHWM and the V4L2
        # capture's BUFFERSIZE: a queued frame's arrival stamp is a lie by
        # however long it waited, and that stamp anchors the observation.
        try:
            sensor = profile.get_device().first_color_sensor()
            if sensor.supports(rs.option.frames_queue_size):
                sensor.set_option(rs.option.frames_queue_size, 1)
        except RuntimeError:
            pass                        # an option this unit does not expose

    def read_into(self, pixels: np.ndarray) -> bool:
        frames = self._pipeline.wait_for_frames(self.TIMEOUT_MS)
        self.t_recv = now()             # as close to arrival as the SDK allows
        color = frames.get_color_frame()
        if not color:
            return False
        frame = np.asanyarray(color.get_data())
        if frame.shape != self.shape:
            raise ValueError(
                f"RealSense {self.serial or '(first device)'} produced "
                f"{frame.shape}, the arena region is {self.shape}. The colour "
                f"sensor is natively 4:3 and its 16:9 modes crop vertically, so "
                f"a resolution mismatch here is a different field of view — the "
                f"exo letterbox bars, and with them the dropped patch tokens, "
                f"are computed from 640x480.")
        np.copyto(pixels, frame)
        return True

    def close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None


def _fourcc(cv2, name: str) -> float:
    """`"MJPG"` -> the packed integer OpenCV's property wants."""
    fn = getattr(cv2, "VideoWriter_fourcc", None) or cv2.VideoWriter.fourcc
    return float(fn(*name))


def _fourcc_name(code: int) -> str:
    """The inverse, for reporting what the driver actually negotiated."""
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))


def _is_jpeg(payload: np.ndarray) -> bool:
    """A JPEG starts with the SOI marker `FF D8`.

    This is what decides whether `CAP_PROP_CONVERT_RGB = 0` was honoured. A
    driver that ignored it returns decoded pixels, whose first two bytes are
    image data and would match `FF D8` only by coincidence — and the cost of
    that coincidence is one failed `imdecode` on the first frame, which raises.
    """
    flat = np.asarray(payload).reshape(-1)
    return flat.size > 2 and flat[0] == 0xFF and flat[1] == 0xD8


class FakeCamera:
    """Deterministic frames, for driving the loop with no board and no network.

    Frame `i` is the constant image `1 + i`, so a test can tell which buffer
    holds which frame by looking at any one pixel — and `1 +` because a zeroed
    buffer is one that was never written.
    """

    def __init__(self, shape: tuple[int, int, int], *, n: int | None = None,
                 delay: float = 0.0):
        self.shape = tuple(int(v) for v in shape)
        self.n = n
        self.delay = float(delay)
        self.i = 0
        self.t_recv = 0.0
        #: Fakes have no compressed source; P2 stores `jpeg_nbytes = 0` for
        #: them, which is exactly the "not captured" case a real dropped frame
        #: produces — so the recorder's handling of it is exercised off-robot.
        self.jpeg: bytes | None = None
        self.connected = False
        self.closed = False

    def connect(self) -> None:
        self.connected = True

    def read_into(self, pixels: np.ndarray) -> bool:
        if self.n is not None and self.i >= self.n:
            return False
        if self.delay:
            time.sleep(self.delay)
        self.t_recv = now()
        pixels[...] = np.uint8((1 + self.i) % 256)
        self.i += 1
        return True

    def close(self) -> None:
        self.closed = True
