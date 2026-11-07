"""Where a camera's frames come from: one binding per kind of camera, one contract.

What P2 (and `elrobot_deploy camera`) needs from a source:

    src.connect()                  open, start streaming, verify the mode; raises if it cannot
    src.read_into(pixels) -> bool  block for the next frame, write it (BGR, uint8, native
                                   resolution) into `pixels`; False = the stream ended
    src.t_capture, src.t_recv      that frame's capture time and arrival time (`timing.now`'s
                                   clock); a source with no capture time sets both to arrival
    src.dropped                    frames the source itself discarded (late), if it knows
    src.controls()                 the camera's control channel (`uvc.UvcControls` or
                                   compatible), or None if this binding has none
    src.describe()                 one line: the device and the negotiated mode
    src.close()

**BGR, native resolution, no resize.** Any transform belongs to the consumer. The one cut, the
fisheye square, is metadata on every frame (`lens.py`), not a resize.

The bindings:

* `AvfCamera` (`source = "avfoundation"`, the Mac): AVFoundation through pyobjc. NOT OpenCV's
  macOS backend, deliberately: OpenCV 5 opens every camera at `AVCaptureSessionPresetMedium`
  and only SCALES the output to the size asked for, never selecting the camera's own mode, so
  a 1280x720 request becomes an upscaled medium-preset stream at whatever rate that preset
  picks. Here the device's own format is chosen (size, frame rate, MJPEG preferred), frames
  arrive BGRA unscaled, and each carries its presentation time on the host clock, which on
  macOS is the clock `time.monotonic` reads.
* `V4l2Camera` (`source = "v4l2"`, Linux): OpenCV's V4L2 backend, MJPG forced. Ported from
  g1d (`legacy_g1d/.../drivers/cameras.py`), where it ran USB fisheyes on the deploy PC. No
  control channel yet (V4L2 has its own; add it when a Linux box runs a camera).
* `RealSenseCamera` (`source = "realsense"`): an Intel RealSense colour stream (an
  exocentric view), ported unchanged from g1d. No control channel.
* `FakeCamera` (`source = "fake"`): a synthetic fisheye whose brightness answers to
  `FakeControls`, including a camera "auto-exposure" that meters the whole frame and so
  blows the disc out, the failure g1d measured on the real lens. For tests and `--sim`.

A new kind of camera is one class with this contract, added to `cameras.SOURCES`.
"""

from __future__ import annotations

import re
import sys
import threading
import time
import typing

import numpy as np

from elrobot_deploy.runtime.timing import now


class CameraError(RuntimeError):
    """A camera could not be opened, or does not deliver what the profile says."""


#: Below this fraction of the requested rate, refuse (g1d: separates "30 fps" from a 5 fps
#: fallback mode, not jitter).
MIN_FPS_RATIO = 0.6
#: Frames timed at connect.
FPS_PROBE_FRAMES = 15
#: A capture timestamp further than this from the arrival is not on our clock: use arrival.
MAX_PTS_SKEW_S = 0.5


def parse_usb_uid(uid: str) -> tuple[int, int, int] | None:
    """A macOS USB camera's AVFoundation uniqueID -> (location, vid, pid), else None.
    `0x213000010bb2b08` = location 0x02130000, 10bb:2b08 (the location is not zero-padded:
    the last eight digits are always the ids)."""
    m = re.fullmatch(r"0x([0-9a-fA-F]{1,8})([0-9a-fA-F]{4})([0-9a-fA-F]{4})", uid or "")
    if m is None:
        return None
    return int(m.group(1), 16), int(m.group(2), 16), int(m.group(3), 16)


def parse_usb_id(text: str) -> tuple[int, int, int | None] | None:
    """`"10bb:2b08"` or `"10bb:2b08@0x14210000"` -> (vid, pid, location or None)."""
    m = re.fullmatch(r"\s*([0-9a-fA-F]{4}):([0-9a-fA-F]{4})(?:@(0x[0-9a-fA-F]+|\d+))?\s*",
                     text or "")
    if m is None:
        return None
    loc = None if m.group(3) is None else int(m.group(3), 0)
    return int(m.group(1), 16), int(m.group(2), 16), loc


# --- AVFoundation (macOS) -------------------------------------------------------------------

_FOURCC_MJPEG = ("dmb1", "jpeg", "mjpa", "mjpg")


def _fourcc(code: int) -> str:
    return "".join(chr((int(code) >> s) & 0xFF) for s in (24, 16, 8, 0))


def avf_devices() -> list[dict]:
    """Every video device AVFoundation offers: name, uniqueID, model, type, and the USB
    (location, vid, pid) its uniqueID encodes (None for built-in and Continuity cameras)."""
    import AVFoundation as AVF

    out = []
    for d in AVF.AVCaptureDevice.devicesWithMediaType_(AVF.AVMediaTypeVideo):
        uid = str(d.uniqueID())
        out.append({"name": str(d.localizedName()), "uid": uid, "model": str(d.modelID()),
                    "type": str(d.deviceType()), "usb": parse_usb_uid(uid)})
    return out


def avf_formats(device) -> list[dict]:
    """A device's formats: width, height, fourcc, frame-rate ranges."""
    import CoreMedia

    out = []
    for f in device.formats():
        desc = f.formatDescription()
        dims = CoreMedia.CMVideoFormatDescriptionGetDimensions(desc)
        ranges = list(f.videoSupportedFrameRateRanges())
        rates = [(float(r.minFrameRate()), float(r.maxFrameRate())) for r in ranges]
        out.append({"format": f, "width": int(dims.width), "height": int(dims.height),
                    "fourcc": _fourcc(CoreMedia.CMFormatDescriptionGetMediaSubType(desc)),
                    "rates": rates, "ranges": ranges})
    return out


def verify_rate(src, *, hint: str = "") -> None:
    """A first frame within 5 s, then `FPS_PROBE_FRAMES` at no less than `MIN_FPS_RATIO` of the
    requested rate. For a binding with `device`, `mode`, `fps`, `shape`, `_fault` and a
    `read_into(pixels, timeout=)`."""
    pixels = np.empty(src.shape, np.uint8)
    if not src.read_into(pixels, timeout=5.0):
        raise CameraError(f"{src.device}: no frame within 5 s of starting"
                          + (f" ({src._fault})" if src._fault else ""))
    t0 = now()
    for _ in range(FPS_PROBE_FRAMES):
        if not src.read_into(pixels, timeout=2.0):
            raise CameraError(f"{src.device} stopped delivering frames during the rate "
                              f"probe" + (f" ({src._fault})" if src._fault else ""))
    measured = FPS_PROBE_FRAMES / max(now() - t0, 1e-9)
    if measured < MIN_FPS_RATIO * src.fps:
        raise CameraError(f"{src.device} delivers {measured:.1f} fps against "
                          f"{src.fps:g} requested ({src.mode}). {hint}".rstrip())


def resolve_avf(spec: str, devices: list[dict] | None = None) -> dict:
    """The AVFoundation device a profile's `device` names.

    "auto" = the one USB camera (never the built-in one, never an iPhone's Continuity
    Camera, which the teleop phone would lose its ARKit session to). "continuity" = the one
    iPhone or iPad Continuity Camera (the exocentric view of an Astra session). Otherwise the
    exact uniqueID, a `vid:pid[@location]`, or a case-insensitive part of the name.
    """
    devices = avf_devices() if devices is None else devices
    listing = "; ".join(f"{d['name']!r} ({d['uid']})" for d in devices) or "none"
    spec = (spec or "auto").strip()
    if spec == "auto":
        found = [d for d in devices if d["usb"] is not None]
        what = "a USB camera"
    elif spec == "continuity":
        found = [d for d in devices if d["usb"] is None
                 and d.get("model", "").startswith(("iPhone", "iPad"))]
        what = ("an iPhone Continuity Camera (same Apple ID, WiFi and Bluetooth on, the phone "
                "near the Mac)")
    elif parse_usb_id(spec) is not None:
        vid, pid, loc = parse_usb_id(spec)
        found = [d for d in devices if d["usb"] is not None and d["usb"][1:] == (vid, pid)
                 and (loc is None or d["usb"][0] == loc)]
        what = f"USB id {spec}"
    else:
        found = [d for d in devices if d["uid"] == spec] or \
                [d for d in devices if spec.lower() in d["name"].lower()]
        what = f"a camera named {spec!r}"
    if not found:
        raise CameraError(f"no {what} among AVFoundation's cameras ({listing}). Is it plugged "
                          f"in? `system_profiler SPUSBDataType` must list it; "
                          f"`elrobot_deploy camera list` shows what is seen.")
    if len(found) > 1:
        raise CameraError(f"{len(found)} cameras match {what}: "
                          + "; ".join(f"{d['name']!r} ({d['uid']})" for d in found)
                          + ". Name one by its uniqueID or vid:pid@location.")
    return found[0]


def _authorize() -> None:
    """Camera access for this process (macOS asks the person once, for the terminal app)."""
    import AVFoundation as AVF

    status = int(AVF.AVCaptureDevice.authorizationStatusForMediaType_(AVF.AVMediaTypeVideo))
    if status == 3:                                         # authorized
        return
    if status == 0:                                         # not determined: ask
        done, granted = threading.Event(), []

        def handler(ok):
            granted.append(bool(ok))
            done.set()

        AVF.AVCaptureDevice.requestAccessForMediaType_completionHandler_(
            AVF.AVMediaTypeVideo, handler)
        done.wait(120.0)
        if granted and granted[0]:
            return
    raise CameraError(
        "camera access is not allowed for this terminal. System Settings > Privacy & "
        "Security > Camera: turn it on for the terminal app you run elrobot_deploy from, "
        "then restart that app.")


_DELEGATE = None


def _delegate_class():
    """The pyobjc delegate class, defined once per process (Objective-C class names are
    global)."""
    global _DELEGATE
    if _DELEGATE is None:
        import objc
        from Foundation import NSObject

        class ElrCameraDelegate(NSObject, protocols=[objc.protocolNamed(
                "AVCaptureVideoDataOutputSampleBufferDelegate")]):
            def captureOutput_didOutputSampleBuffer_fromConnection_(self, out, sb, conn):
                self.owner._on_frame(sb)

            def captureOutput_didDropSampleBuffer_fromConnection_(self, out, sb, conn):
                self.owner.dropped += 1

        _DELEGATE = ElrCameraDelegate
    return _DELEGATE


class AvfCamera:
    """A camera on this Mac, through AVFoundation, at its own native mode."""

    def __init__(self, device: str, shape: tuple[int, int, int], *, fps: float = 30.0):
        self.spec = str(device)
        self.shape = tuple(int(v) for v in shape)
        self.fps = float(fps)
        self.t_capture = self.t_recv = 0.0
        self.dropped = 0
        self.errors = 0
        self.device = ""                # resolved: "name (uniqueID)"
        self.uid = ""
        self.mode = ""
        self._cond = threading.Condition()
        self._back = np.zeros(self.shape, np.uint8)
        self._back_t = (0.0, 0.0)
        self._n = 0                     # frames delivered by AVFoundation
        self._read = 0                  # the last one `read_into` returned
        self._session = None
        self._delegate = None
        self._queue = None
        self._fault = ""

    # --- the callback (AVFoundation's dispatch queue) ----------------------------------------

    def _on_frame(self, sb) -> None:
        import CoreMedia
        import cv2
        import Quartz

        t_recv = now()
        try:
            img = CoreMedia.CMSampleBufferGetImageBuffer(sb)
            Quartz.CVPixelBufferLockBaseAddress(img, 1)     # read-only
            try:
                h = Quartz.CVPixelBufferGetHeight(img)
                w = Quartz.CVPixelBufferGetWidth(img)
                bpr = Quartz.CVPixelBufferGetBytesPerRow(img)
                if (h, w) != self.shape[:2]:
                    self._fault = (f"AVFoundation delivered {w}x{h}, the profile says "
                                   f"{self.shape[1]}x{self.shape[0]}")
                    return
                buf = Quartz.CVPixelBufferGetBaseAddress(img).as_buffer(bpr * h)
                bgra = np.frombuffer(buf, np.uint8).reshape(h, bpr)[:, :w * 4].reshape(h, w, 4)
                pts = CoreMedia.CMTimeGetSeconds(
                    CoreMedia.CMSampleBufferGetPresentationTimeStamp(sb))
                with self._cond:
                    cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR, dst=self._back)
                    t_cap = pts if abs(pts - t_recv) < MAX_PTS_SKEW_S else t_recv
                    self._back_t = (t_cap, t_recv)
                    self._n += 1
                    self._cond.notify_all()
            finally:
                Quartz.CVPixelBufferUnlockBaseAddress(img, 1)
        except Exception as exc:  # noqa: BLE001
            self.errors += 1
            self._fault = f"{type(exc).__name__}: {exc}"

    # --- the contract --------------------------------------------------------------------------

    def connect(self) -> None:
        import AVFoundation as AVF
        import dispatch
        import Quartz

        _authorize()
        info = resolve_avf(self.spec)
        dev = AVF.AVCaptureDevice.deviceWithUniqueID_(info["uid"])
        if dev is None:
            raise CameraError(f"{info['name']!r} disappeared")
        self.device, self.uid = f"{info['name']} ({info['uid']})", info["uid"]
        h, w = self.shape[:2]
        formats = avf_formats(dev)
        fits = [f for f in formats if (f["height"], f["width"]) == (h, w)
                and any(lo - 0.5 <= self.fps <= hi + 0.5 for lo, hi in f["rates"])]
        if not fits:
            have = sorted({f"{f['width']}x{f['height']} {f['fourcc']} "
                           f"{max(hi for _, hi in f['rates']):.0f}fps" for f in formats})
            raise CameraError(f"{self.device} has no {w}x{h} mode at {self.fps:g} fps. Its "
                              f"modes: {', '.join(have)}")
        # The compressed mode first: over USB 2 raw YUYV ("yuvs") manages 10 fps at 720p. macOS
        # shows a UVC camera's MJPEG modes as "420v" (it decodes them), and they are the ones
        # that reach the highest rate, so the top rate is what picks them.
        fits.sort(key=lambda f: (f["fourcc"] not in _FOURCC_MJPEG,
                                 -max(hi for _, hi in f["rates"])))
        chosen = fits[0]
        self.mode = f"{w}x{h} {chosen['fourcc']} @ {self.fps:g} fps"

        session = AVF.AVCaptureSession.alloc().init()
        inp, err = AVF.AVCaptureDeviceInput.deviceInputWithDevice_error_(dev, None)
        if inp is None:
            raise CameraError(f"cannot open {self.device}: {err}")
        out = AVF.AVCaptureVideoDataOutput.alloc().init()
        out.setVideoSettings_({Quartz.kCVPixelBufferPixelFormatTypeKey:
                               Quartz.kCVPixelFormatType_32BGRA})
        out.setAlwaysDiscardsLateVideoFrames_(True)         # the newest frame wins
        self._delegate = _delegate_class().alloc().init()
        self._delegate.owner = self
        self._queue = dispatch.dispatch_queue_create(b"elrobot.camera", None)
        out.setSampleBufferDelegate_queue_(self._delegate, self._queue)

        # The device's own mode, set inside the session's configuration and held locked
        # until the session runs: otherwise the session's preset overrides it.
        session.beginConfiguration()
        if not session.canAddInput_(inp) or not session.canAddOutput_(out):
            raise CameraError(f"cannot stream {self.device} (in use by another program?)")
        session.addInput_(inp)
        session.addOutput_(out)
        ok, err = dev.lockForConfiguration_(None)
        if not ok:
            raise CameraError(f"cannot configure {self.device}: {err}")
        try:
            dev.setActiveFormat_(chosen["format"])
            # The range's own duration, not 1/fps: this camera's "30 fps" is exactly
            # 1000000/30000030 s, and AVFoundation refuses anything else.
            rate = min(chosen["ranges"], key=lambda r: abs(float(r.maxFrameRate()) - self.fps))
            period = rate.minFrameDuration()
            dev.setActiveVideoMinFrameDuration_(period)
            dev.setActiveVideoMaxFrameDuration_(period)
            session.commitConfiguration()
            session.startRunning()
        finally:
            dev.unlockForConfiguration()
        self._session = session
        self._verify_rate()

    def _verify_rate(self) -> None:
        verify_rate(self, hint="Another program using it? A USB hub shared with something busy?")

    def read_into(self, pixels: np.ndarray, *, timeout: float = 2.0) -> bool:
        with self._cond:
            if not self._cond.wait_for(lambda: self._n > self._read, timeout):
                return False
            np.copyto(pixels, self._back)
            self.t_capture, self.t_recv = self._back_t
            self._read = self._n
        return True

    def controls(self):
        from elrobot_deploy.runtime.cameras.uvc import UvcControls
        usb = parse_usb_uid(self.uid)
        if usb is None:
            return None
        return UvcControls.open(vid=usb[1], pid=usb[2], location=usb[0])

    def describe(self) -> str:
        return f"{self.device}: {self.mode}"

    def close(self) -> None:
        if self._session is not None:
            self._session.stopRunning()
            self._session = None
        self._delegate = None


# --- V4L2 (Linux) ---------------------------------------------------------------------------


class V4l2Camera:
    """A UVC camera on a Linux box's USB, through OpenCV's V4L2 backend, MJPG forced (at
    1280x720 the uncompressed alternative does not fit USB 2 and the driver silently delivers
    ~5 fps). Ported from g1d's `V4l2Camera`; `device` is a `/dev/v4l/by-path/...` node (the
    /dev/videoN numbering is not stable across reboots)."""

    def __init__(self, device: str, shape: tuple[int, int, int], *, fps: float = 30.0,
                 fourcc: str = "MJPG"):
        self.device = str(device)
        self.shape = tuple(int(v) for v in shape)
        self.fps = float(fps)
        self.fourcc = fourcc
        self.t_capture = self.t_recv = 0.0
        self.dropped = 0
        self._cap = None

    def connect(self) -> None:
        import cv2

        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise CameraError(f"cannot open camera {self.device!r} (`ls -l /dev/v4l/by-path/`)")
        cap.set(cv2.CAP_PROP_FOURCC, float(cv2.VideoWriter.fourcc(*self.fourcc)))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.shape[1])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.shape[0])
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)                 # the newest frame wins
        self._cap = cap
        got = _fourcc(int(cap.get(cv2.CAP_PROP_FOURCC)))[::-1]
        size = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        if got != self.fourcc or size != self.shape[:2]:
            raise CameraError(f"{self.device} negotiated {got} {size[1]}x{size[0]}, not "
                              f"{self.fourcc} {self.shape[1]}x{self.shape[0]} "
                              f"(`v4l2-ctl -d {self.device} --list-formats-ext`)")
        t0 = time.monotonic()
        for _ in range(FPS_PROBE_FRAMES):
            if not cap.grab():
                raise CameraError(f"{self.device} stopped delivering frames")
        measured = FPS_PROBE_FRAMES / max(time.monotonic() - t0, 1e-9)
        if measured < MIN_FPS_RATIO * self.fps:
            raise CameraError(f"{self.device} delivers {measured:.1f} fps against {self.fps:g}")

    def read_into(self, pixels: np.ndarray, **_) -> bool:
        if not self._cap.grab():                            # blocks until a frame arrives
            return False
        self.t_capture = self.t_recv = now()                # after arrival, before decode
        ok, frame = self._cap.retrieve()
        if not ok or frame is None or frame.shape != self.shape:
            return False
        np.copyto(pixels, frame)
        return True

    def controls(self):
        return None

    def describe(self) -> str:
        return f"{self.device}: {self.shape[1]}x{self.shape[0]} {self.fourcc} @ {self.fps:g} fps"

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


# --- RealSense ------------------------------------------------------------------------------


class RealSenseCamera:
    """An Intel RealSense colour stream (e.g. a fixed exocentric view), ported unchanged from
    g1d. `bgr8` is requested so every source publishes BGR. The resolution is a field of
    view, not just a pixel count: the colour sensor is natively 4:3 and its 16:9 modes crop."""

    TIMEOUT_MS = 5000

    def __init__(self, shape: tuple[int, int, int], *, serial: str | None = None,
                 fps: float = 30.0):
        self.shape = tuple(int(v) for v in shape)
        self.serial = str(serial) if serial and serial != "auto" else None
        self.fps = float(fps)
        self.t_capture = self.t_recv = 0.0
        self.dropped = 0
        self._pipeline = None

    def connect(self) -> None:
        import pyrealsense2 as rs

        h, w, _ = self.shape
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, int(self.fps))
        self._pipeline = rs.pipeline()
        profile = self._pipeline.start(config)
        try:
            sensor = profile.get_device().first_color_sensor()
            if sensor.supports(rs.option.frames_queue_size):
                sensor.set_option(rs.option.frames_queue_size, 1)
        except RuntimeError:
            pass

    def read_into(self, pixels: np.ndarray, **_) -> bool:
        frames = self._pipeline.wait_for_frames(self.TIMEOUT_MS)
        self.t_capture = self.t_recv = now()
        color = frames.get_color_frame()
        if not color:
            return False
        frame = np.asanyarray(color.get_data())
        if frame.shape != self.shape:
            raise CameraError(f"RealSense produced {frame.shape}, the profile says {self.shape}")
        np.copyto(pixels, frame)
        return True

    def controls(self):
        return None

    def describe(self) -> str:
        return f"RealSense {self.serial or '(first)'}: {self.shape[1]}x{self.shape[0]} bgr8"

    def close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None


# --- the fake -------------------------------------------------------------------------------


class FakeControls:
    """The registers a `FakeCamera` reads, with the same interface as `uvc.UvcControls`. The
    ranges and defaults are what our wrist camera (10bb:2b08) reported over UVC on this Mac,
    2026-10-17; like it, the fake starts in auto-exposure (mode 8, aperture priority)."""

    RANGES: typing.ClassVar[dict] = {
        "ae_mode": {"min": 0, "max": 1, "res": 0b1001, "def": 8},
        "exposure": {"min": 1, "max": 10000, "res": 1, "def": 166},
        "gain": {"min": 0, "max": 128, "res": 1, "def": 64},
        "wb_auto": {"min": 0, "max": 1, "res": 1, "def": 1},
        "wb_temp": {"min": 2800, "max": 6500, "res": 10, "def": 4600}}

    def __init__(self):
        self.values = {k: r["def"] for k, r in self.RANGES.items()}
        self.writes: list[tuple[str, int]] = []
        self.lock = threading.Lock()

    def range(self, name):
        return dict(self.RANGES.get(name, {}))

    def default(self, name):
        return self.RANGES.get(name, {}).get("def")

    def get(self, name, req=None):
        with self.lock:
            return self.values[name]

    def write(self, name, value):
        r = self.RANGES[name]
        if name == "exposure" and self.values["ae_mode"] != 1:
            # as the real one does: in auto-exposure, the exposure time is not writable
            raise RuntimeError("exposure: [Errno 32] Pipe error (auto-exposure is on)")
        if value is None:
            value = r["def"]
        elif name != "ae_mode":             # a bitmask: its min/max (0..1 here!) mean nothing
            value = min(max(int(value), r["min"]), r["max"])
        with self.lock:
            self.values[name] = value
            self.writes.append((name, value))
        return value

    def probe(self):
        return {k: dict(r, cur=self.values[k]) for k, r in self.RANGES.items()}

    def close(self):
        pass


class FakeCamera:
    """A synthetic 210 deg fisheye: a lit, textured disc on black, whose brightness follows
    the registers in `FakeControls` (exposure x gain, linear, as g1d measured the real one).

    In auto-exposure (`ae_mode` 8) it behaves like the real camera does by g1d's measurement:
    it meters the WHOLE frame, black surround included, and raises the light until the
    full-frame mean is `ae_mean`: at the default 90 about a third of the disc is on 255, the
    failure g1d measured on its rig (26%). (Our own camera, in our room, did far better: 7%.)
    `scene` scales the room's light.
    """

    def __init__(self, shape: tuple[int, int, int] = (720, 1280, 3), *, fps: float = 30.0,
                 circle: tuple[float, float, float] | None = None, scene: float = 1.0,
                 n: int | None = None, controls: FakeControls | None = None,
                 ae_mean: float = 90.0):
        self.shape = tuple(int(v) for v in shape)
        h, w = self.shape[:2]
        self.circle = circle or (w / 2 - 10.0, h / 2 - 5.0, 0.46 * h)
        self.fps = float(fps)
        self.scene = float(scene)
        self._ae_mean = float(ae_mean)
        self.n = n
        self.ctl = controls or FakeControls()
        self.t_capture = self.t_recv = 0.0
        self.dropped = 0
        self.i = 0
        self.device = "fake"
        self._t_next = 0.0
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy, r = self.circle
        rr = np.hypot(xx - cx, yy - cy)
        inside = rr <= r
        # A scene with a highlight: a gradient, a grid of objects, and a bright window.
        base = 0.35 + 0.25 * (xx / w) + 0.08 * np.sin(xx / 23.0) * np.cos(yy / 17.0)
        base += 0.45 * ((np.abs(xx - cx - 0.4 * r) < 0.12 * r) & (np.abs(yy - cy + 0.3 * r)
                                                                   < 0.10 * r))
        self._radiance = np.where(inside, base, 0.0).astype(np.float32)
        self._frac = float(inside.mean())

    def connect(self) -> None:
        self._t_next = now()

    def _light(self) -> float:
        """Exposure x gain as relative light (1.0 = exposure 200, gain 64)."""
        c = self.ctl
        if c.get("ae_mode") == 1:
            exp, gain = c.get("exposure"), c.get("gain")
            return exp * (1 + gain / 21.0) / (200 * (1 + 64 / 21.0)) * self.scene
        # the camera's own loop: the FULL-FRAME mean held at `ae_mean`, whatever the disc does
        return self._ae_mean / max(220.0 * float(self._radiance.mean()), 1e-6)

    def read_into(self, pixels: np.ndarray, **_) -> bool:
        if self.n is not None and self.i >= self.n:
            return False
        from elrobot_deploy.runtime.timing import sleep_until
        self._t_next = max(self._t_next + 1.0 / self.fps, now() - 1.0)
        sleep_until(self._t_next)
        v = np.clip(self._radiance * (220.0 * self._light()), 0, 255).astype(np.uint8)
        pixels[...] = v[..., None]
        self.t_capture = self.t_recv = now()
        self.i += 1
        return True

    def controls(self):
        return self.ctl

    def describe(self) -> str:
        return f"fake: {self.shape[1]}x{self.shape[0]} @ {self.fps:g} fps"

    def close(self) -> None:
        pass


def default_usb_source() -> str:
    """What `source = "usb"` means on this machine."""
    return "avfoundation" if sys.platform == "darwin" else "v4l2"


__all__ = ["AvfCamera", "CameraError", "FakeCamera", "FakeControls", "RealSenseCamera",
           "V4l2Camera", "avf_devices", "avf_formats", "default_usb_source", "parse_usb_id",
           "parse_usb_uid", "resolve_avf", "verify_rate"]
