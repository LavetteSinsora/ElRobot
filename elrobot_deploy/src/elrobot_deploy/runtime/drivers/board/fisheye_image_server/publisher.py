"""The fisheye image server: UVC cameras -> one ZMQ PUB socket each, on the board.

RUNS ON THE ROBOT BOARD (`192.168.123.164`), under conda env `tv`. It is a
deliberate, minimal stand-in for the vendor's `teleimager` for a rig with
fisheye wrist cameras, and it mirrors that server's conventions on purpose — same config
schema, same camera topic names, same socket options, same mode selection — so
that reading one teaches you the other.

WHY THIS EXISTS AT ALL. `teleimager` already does MJPEG passthrough on its ZMQ
path (`frame.jpeg_buffer` straight to the socket, no decode, no re-encode), so
there is nothing to improve about the transport. Two things it does not do:

  1. **Camera controls.** `cam_config_server.yaml` has no exposure, gain or
     white-balance field. A policy made from images recorded with tuned
     controls, served a systematically brighter or dimmer image, is a skew
     nothing reports. `cameras.yaml` here carries them, and
     they are read back and verified rather than hoped for.
  2. **A sequence number.** ZMQ PUB drops silently under back-pressure — by
     design, since both ends set a high-water mark of 1 so the newest frame
     wins. With the vendor's server a dropped frame is invisible. The 16-byte header below makes it
     a number the deploy side can report.

WHAT IT DELIBERATELY DOES NOT DO: no WebRTC, no decode, no crop, no resize, no
colour conversion. The crop is the deploy box's (`elrobot_deploy.runtime.drivers.fisheye_circle_fit`) and any
resize is the policy's. A resize here would be one no policy could undo.

⚠️ **THIS CANNOT RUN AT THE SAME TIME AS `teleimager`.** The vendor server's
`reload_uvc_driver()` runs `sudo modprobe -r uvcvideo` at startup, and its
systemd unit is `Restart=always`, so any restart of it rips the module out from
under whatever else is capturing. Exactly one of the two owns the cameras. See
README.md for the switch procedure.

ONE DELIBERATE DEVIATION FROM THE VENDOR. `teleimager` captures in one thread
(into a triple buffer) and publishes from another at a fixed rate. That
decoupling means a slow capture republishes the same frame and a fast one skips
frames, neither of which is visible downstream. Here, capture and publish are
the SAME loop: one captured frame is one published message, and `seq` increments
once per captured frame. That is what makes `seq` mean anything — a rate-paced
publisher would emit duplicate sequence numbers for one image and gaps for
frames that were never dropped at all.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import struct
import sys
import threading
import time

#: DUAL-MODE, AND BOTH MODES ARE REAL — do not "tidy" this into one line.
#:
#: The deploy box imports this as
#: `elrobot_deploy.runtime.drivers.board.fisheye_image_server.publisher` (the
#: tests and `drivers/cameras.py` do), but the BOARD runs it as a bare script:
#: the directory is copied to the robot file's `[image_server] dir` and
#: `image_server_launch._launch` runs `python publisher.py` from inside it, with no package
#: context. A bare relative import works in the first case and raises
#: "attempted relative import with no known parent package" in the second —
#: i.e. it would pass every test here and break the robot.
try:
    from elrobot_deploy.runtime.drivers.board.fisheye_image_server import photometry
except ImportError:                             # running as a script on the board
    import photometry

# --- the wire format --------------------------------------------------------
#
# One ZMQ message per frame: this 16-byte header, then the camera's own JPEG
# bytes verbatim. A single message rather than a multipart one, because a single
# message cannot half-arrive — and with a receive high-water mark of 1, reasoning
# about partially-delivered multipart frames is a cost with no benefit.
#
# There is NO timestamp, and that is a decision rather than an omission. A board
# timestamp is only meaningful against a board<->box clock mapping, and the two
# are unsynchronised free-running monotonics. `version` and `flags` leave room to
# add one later without a flag day.

MAGIC = b"G1DF"
HEADER_VERSION = 1
#: `<` = little-endian, no padding: 4 + 1 + 1 + 2 + 8 = 16 bytes.
HEADER_STRUCT = struct.Struct("<4sBBHQ")
HEADER_SIZE = HEADER_STRUCT.size

#: Camera topic names, matching the vendor's `cam_config_server.yaml` keys so the
#: two configs read the same way.
CAMERA_TOPICS = ("head_camera", "left_wrist_camera", "right_wrist_camera")

DEFAULT_CONFIG = pathlib.Path(__file__).resolve().parent / "cameras.yaml"

log = logging.getLogger("fish_eye_image_server")


def pack_header(seq: int, *, flags: int = 0) -> bytes:
    """The 16-byte prefix for frame `seq`."""
    return HEADER_STRUCT.pack(MAGIC, HEADER_VERSION, flags, 0, seq & 0xFFFFFFFFFFFFFFFF)


def parse_header(payload: bytes):
    """`(seq, jpeg)` if this carries our header, else `(None, payload)`.

    The fallback is what lets one client read both servers: the vendor publishes
    a bare JPEG, whose first two bytes are the SOI marker `FF D8` and therefore
    never our magic. A client that cannot tell them apart would silently strip 16
    bytes off a vendor frame and hand a corrupt JPEG to the decoder.
    """
    if len(payload) >= HEADER_SIZE and payload[:4] == MAGIC:
        _magic, _version, _flags, _pad, seq = HEADER_STRUCT.unpack_from(payload)
        return seq, payload[HEADER_SIZE:]
    return None, payload


# --- finding a camera -------------------------------------------------------
#
# pyuvc addresses devices by `uid`, which is `"{bus}:{device_address}"` — and the
# device address CHANGES on every replug. It is therefore never something to
# write in a config file. The two stable identifiers are the sysfs USB path
# (which socket the cable is in) and the serial number, and this resolves either
# of them to a uid at startup.
#
# The vendor resolves `physical_path` by reloading `uvcvideo`, reading
# `/sys/class/video4linux` and mapping back. We do not need that dance: sysfs
# already publishes `busnum` and `devnum` for the USB device itself, which IS the
# uid. Skipping it is why this server never has to touch the kernel module — and
# so is strictly less invasive than the one it stands in for.

SYSFS_USB = pathlib.Path("/sys/bus/usb/devices")


def uid_from_physical_path(physical_path: str) -> str:
    """`"1-3.2"` -> `"1:7"`, by reading the USB device's own sysfs attributes."""
    node = SYSFS_USB / physical_path
    try:
        bus = (node / "busnum").read_text().strip()
        dev = (node / "devnum").read_text().strip()
    except OSError as exc:
        raise RuntimeError(
            f"cannot read USB attributes for physical_path {physical_path!r} "
            f"({exc}). `ls {SYSFS_USB}` lists what is actually attached. The "
            f"path is the USB topology — which controller, which hub, which "
            f"port — so it changes only if the cable moves.") from None
    return f"{int(bus)}:{int(dev)}"


def uid_from_serial(serial: str) -> str:
    """The uid of the attached camera reporting this serial number.

    Refuses on a duplicate rather than picking one. Fisheye pairs have been
    seen reporting the same serial, and choosing arbitrarily between two wrist cameras
    gives the policy a mirrored view of its own arms with nothing raised.
    """
    import uvc

    matches = [d for d in uvc.device_list() if str(d.get("serialNumber")) == str(serial)]
    if not matches:
        seen = sorted(str(d.get("serialNumber")) for d in uvc.device_list())
        raise RuntimeError(
            f"no attached camera reports serial {serial!r}. Attached: {seen}")
    if len(matches) > 1:
        raise RuntimeError(
            f"{len(matches)} attached cameras report serial {serial!r}. These "
            f"units ship with duplicate serials — address them by "
            f"`physical_path` instead, which is unique by construction.")
    return matches[0]["uid"]


def resolve_uid(topic: str, spec: dict) -> str:
    """`physical_path` first, then `serial_number` — the vendor's own priority.

    `video_id` is NOT supported and cannot be: this board has no `/dev/video*` at
    all. The cameras are claimed from userspace through `usbfs` by pyuvc, so
    there is no device node to number.
    """
    if spec.get("physical_path"):
        return uid_from_physical_path(str(spec["physical_path"]))
    if spec.get("serial_number"):
        return uid_from_serial(str(spec["serial_number"]))
    raise RuntimeError(
        f"camera {topic!r} declares neither `physical_path` nor "
        f"`serial_number`. One is required; `video_id` does not work on this "
        f"board because it has no /dev/video* nodes (pyuvc claims the cameras "
        f"through usbfs).")


# --- opening it -------------------------------------------------------------


def choose_mode(cap, *, height: int, width: int, fps: int):
    """The MJPG mode matching the configured geometry exactly, or a refusal.

    Same predicate the vendor's `UVCCamera._choose_mode` uses. MJPG is required
    rather than preferred: at 1280x720 the uncompressed alternative is ~55 MB/s
    per camera, which does not fit down a shared USB 2.0 bus, and it is also the
    format whose bytes we pass through without re-encoding.
    """
    for mode in cap.available_modes:
        if (mode.width == width and mode.height == height
                and mode.fps == fps and mode.format_name == "MJPG"):
            return mode
    offered = sorted({(m.width, m.height, m.fps, m.format_name)
                      for m in cap.available_modes})
    raise RuntimeError(
        f"this camera does not offer {width}x{height} @ {fps} MJPG. It offers: "
        f"{offered}. The configured geometry is what the robot file and the "
        f"lens circle describe; change cameras.yaml and the robot file "
        f"together.")


#: How a camera's photometry is decided. Both are reproducible in the sense that
#: matters — neither changes behaviour between runs — but they trade differently.
#:
#: `auto`   the camera's own exposure/white-balance loop. Adapts to the lab's
#:          lighting, survives a hardware swap, and needs no per-unit numbers.
#: `pinned` explicit device values. Frozen, so two runs see identical transfer
#:          functions, at the cost of not adapting to anything at all.
#:
#: `servo`  this server's own exposure loop (`photometry.py`), metering only
#:          inside the lens circle. **The default.**
#:
#: **Why not `auto` on a fisheye: it blows the image out.** The fisheye's
#: circle fills ~35-41% of the sensor, so ~60% of what the camera meters is a
#: black surround no exposure can lift. AE raises exposure and gain to drag that
#: average up and takes the picture through the ceiling: measured 26.3% of the
#: disk clipped on both wrists, against 1.5%/4.5% in the reference recordings made with tuned
#: manual controls. The head camera, same room and same `Auto Exposure Mode: 8`
#: but rectilinear, sits at 0.16% — so the room is fine and the metering window
#: is the fault.
#:
#: A learned policy's brightness augmentation does not cover it: a jitter that
#: is a monotone [0,1]->[0,1] curve cannot clip and cannot flatten, so it never
#: teaches a model anything about blown-out frames. A photometric band and a
#: clipped highlight are different things.
#:
#: `pinned` device integers copied from ANOTHER camera model are not a reliable
#: way to reproduce an image either: exposure and gain are firmware-scoped.
#:
#: `servo` closes the loop here instead, metering only inside the lens circle.
#: The camera cannot be asked to do this itself: probed live, it exposes no
#: metering-region control of any kind.
PHOTOMETRY_MODES = ("auto", "pinned", "servo")
DEFAULT_PHOTOMETRY = "servo"


def resolve_controls(topic: str, spec: dict) -> tuple[str, dict]:
    """`(mode, controls)` for one camera, from its `photometry` selector.

    Both blocks live in the config side by side so the alternative is documented
    where it is switched, rather than in a comment somewhere else that goes
    stale. Switching is one word.
    """
    mode = str(spec.get("photometry", DEFAULT_PHOTOMETRY))
    if mode not in PHOTOMETRY_MODES:
        raise RuntimeError(
            f"camera {topic!r} has photometry {mode!r}; expected one of "
            f"{list(PHOTOMETRY_MODES)}")
    if mode == "servo" and not spec.get("circle"):
        # A rectilinear camera has nothing for the loop to meter and does not
        # need it. Falling back rather than refusing keeps `servo` usable as
        # the file-wide default while the head stays on its own auto-exposure.
        return "auto", dict(spec.get("controls_auto") or {})
    return mode, dict(spec.get(f"controls_{mode}") or {})


def _tolerance(control) -> int:
    """How far a read-back may sit from what we wrote before it counts as declined.

    A driver may legitimately snap a continuous control to its own quantum — the
    wrists' `Pan control` steps by 3600, `White Balance temperature` by 10 — so
    the step is the natural tolerance. But `step` is only meaningful for a
    control that is actually a range, and UVC drivers report nonsense for the
    ones that are not.

    Measured on these units: **`Auto Exposure Mode` is `[1..8]` with `step 9`** —
    a bitmask (1 manual, 2 auto, 4 shutter priority, 8 aperture priority) whose
    step is larger than its entire span. Taken at face value that tolerance
    accepts ANY in-range value, so a declined write to the single most important
    control here would have passed silently: auto-exposure must be off before a
    manual `Absolute Exposure Time` will stick, and if it is not, every frame is
    exposed by the camera's own judgement instead of the configured one.

    So a step is trusted only when it is a plausible quantum — bigger than 1 and
    no bigger than the range it divides. Otherwise the value must come back
    exactly as written.
    """
    step = abs(int(getattr(control, "step", 1) or 1))
    try:
        span = abs(int(control.max_val) - int(control.min_val))
    except (TypeError, ValueError):
        return 0
    return step if 1 < step <= span else 0


def apply_controls(cap, controls: dict, *, topic: str) -> dict:
    """Set each named UVC control, read it back, and refuse on a mismatch.

    **Order is significant and is the config's order.** An auto mode must be
    turned off before the manual value it overrides will stick — setting
    absolute exposure while auto-exposure is still on is accepted by the driver
    and then quietly ignored. PyYAML preserves document order into a dict, so
    listing `Auto Exposure Mode` before `Absolute Exposure Time` does the right
    thing.

    Read-back is the whole point. A control the driver silently declined is
    exactly the failure this file exists to catch: the run proceeds, the images
    are systematically wrong, and nothing anywhere says so.
    """
    if not controls:
        log.warning("[%s] no controls configured — the camera keeps its driver "
                    "defaults, which are nobody's deliberate choice", topic)
        return {}

    by_name = {c.display_name: c for c in cap.controls}
    applied = {}
    for name, wanted in controls.items():
        control = by_name.get(name)
        if control is None:
            raise RuntimeError(
                f"[{topic}] no UVC control named {name!r}. This unit exposes: "
                f"{sorted(by_name)}. Names are the driver's, verbatim.")
        control.value = int(wanted)
        # `refresh()` re-reads from the device rather than returning the value we
        # just wrote back out of a cache.
        try:
            control.refresh()
        except Exception:                       # noqa: BLE001 - not all controls implement it
            pass
        got = control.value
        if abs(int(got) - int(wanted)) > _tolerance(control):
            raise RuntimeError(
                f"[{topic}] control {name!r} was set to {wanted} but reads back "
                f"as {got} (step {getattr(control, 'step', '?')}, range "
                f"{getattr(control, 'min_val', '?')}..{getattr(control, 'max_val', '?')}). "
                f"The driver declined it. If this control depends on an auto "
                f"mode, that mode must be disabled EARLIER in the config's "
                f"`controls` list.")
        applied[name] = int(got)
        log.info("[%s] control %s = %s", topic, name, got)
    return applied


# --- the loop ---------------------------------------------------------------


class CameraPublisher(threading.Thread):
    """One camera: capture a frame, publish it, once per frame, forever."""

    def __init__(self, topic: str, spec: dict, *, host: str, stop: threading.Event):
        super().__init__(name=topic, daemon=True)
        self.topic = topic
        self.spec = spec
        self.host = host
        self.stop = stop
        self.port = int(spec["zmq_port"])
        self.height, self.width = (int(v) for v in spec["image_shape"])
        self.fps = int(spec["fps"])
        self.seq = 0
        self.dropped = 0            # frames the PUB socket refused at its HWM
        self._cap = None
        self._sock = None
        self.servo = None           # photometry.ServoRunner, under `servo` only

    # -- setup, all of which may refuse ------------------------------------

    def open(self) -> None:
        import uvc

        uid = resolve_uid(self.topic, self.spec)
        log.info("[%s] opening uid=%s (%s)", self.topic, uid,
                 self.spec.get("physical_path") or self.spec.get("serial_number"))
        self._cap = uvc.Capture(uid)
        self._cap.frame_mode = choose_mode(
            self._cap, height=self.height, width=self.width, fps=self.fps)
        mode, controls = resolve_controls(self.topic, self.spec)
        log.info("[%s] photometry=%s (%d control(s))", self.topic, mode,
                 len(controls))
        apply_controls(self._cap, controls, topic=self.topic)

        if mode == "servo":
            # `controls_servo` has already turned auto-exposure OFF and been
            # read back, so the two registers the loop owns will now stick.
            # Writing them before that would have been accepted and ignored.
            by_name = {c.display_name: c for c in self._cap.controls}

            def write(name, value, _by_name=by_name, _topic=self.topic):
                control = _by_name.get(name)
                if control is None:
                    raise RuntimeError(
                        f"[{_topic}] the servo needs a UVC control named "
                        f"{name!r}; this unit exposes {sorted(_by_name)}")
                control.value = int(value)

            self.servo = photometry.runner_for(
                self.topic, self.spec, write=write, log=log)
            if self.servo is None:              # resolve_controls guards this
                raise RuntimeError(f"[{self.topic}] servo mode without a circle")
            self.servo.prime()

    def bind(self) -> None:
        import zmq

        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.PUB)
        # Both options match the vendor's publisher exactly. SNDHWM = 1 keeps
        # only the newest frame: a frame that waited in a send queue is a frame
        # whose arrival stamp on the far side is a lie by however long it sat,
        # and that stamp anchors the whole observation (elrobot_deploy's anchor rule).
        sock.setsockopt(zmq.SNDHWM, 1)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(f"tcp://{self.host}:{self.port}")
        self._sock = sock
        log.info("[%s] publishing on tcp://%s:%d (%dx%d @ %d MJPG)",
                 self.topic, self.host, self.port, self.width, self.height, self.fps)

    # -- the hot loop -------------------------------------------------------

    def run(self) -> None:
        import zmq

        try:
            while not self.stop.is_set():
                frame = self._cap.get_frame_robust()
                if frame is None or frame.jpeg_buffer is None:
                    # An MJPG mode that yields no compressed buffer means the
                    # negotiated format is not what we think it is. Passing
                    # through is the entire design, so there is no fallback.
                    raise RuntimeError(
                        f"[{self.topic}] frame carried no jpeg_buffer despite "
                        f"an MJPG mode")

                # The seq is assigned to the frame BEFORE the send, and is not
                # rolled back if the send is dropped. That is what makes a gap
                # on the subscriber mean "a frame existed that you did not get".
                jpeg = bytes(frame.jpeg_buffer)
                payload = pack_header(self.seq) + jpeg
                self.seq += 1
                try:
                    self._sock.send(payload, zmq.NOBLOCK)
                except zmq.Again:
                    self.dropped += 1

                # AFTER the send, and never on the frame's critical path. One
                # frame in three costs a 2.30 ms half-scale grayscale decode
                # (measured on the board) against a 33 ms frame period, and
                # `offer` swallows its own faults — a photometry bug must not
                # be able to stop an image server.
                if self.servo is not None:
                    self.servo.offer(jpeg)
        except Exception:                       # noqa: BLE001 - any camera fault stops the server
            log.exception("[%s] publisher stopped", self.topic)
            self.stop.set()

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        if self._cap is not None:
            try:
                self._cap.close()
            except Exception:                   # noqa: BLE001 - a yanked USB hub can hang close()
                pass
            self._cap = None


# --- config -----------------------------------------------------------------


def load_config(path: pathlib.Path) -> dict:
    import yaml

    if not path.exists():
        raise SystemExit(f"no config at {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    cameras = {}
    for topic, spec in raw.items():
        if topic in ("server", "settings"):
            continue
        if not isinstance(spec, dict):
            continue
        if not spec.get("enable_zmq", True):
            log.info("[%s] disabled in config (enable_zmq: false)", topic)
            continue
        for required in ("zmq_port", "image_shape", "fps"):
            if required not in spec:
                raise SystemExit(f"camera {topic!r} is missing {required!r}")
        cameras[topic] = spec
    if not cameras:
        raise SystemExit(f"{path} enables no cameras")
    ports = [s["zmq_port"] for s in cameras.values()]
    if len(set(ports)) != len(ports):
        raise SystemExit(f"two cameras share a zmq_port: {ports}")
    return {"cameras": cameras, "server": raw.get("server") or {}}


def pin_cores(cores) -> None:
    """Mirror the vendor unit's `CPUAffinity=0 1 2`, so this server competes for
    the same cores the one it replaces did and no others."""
    import os

    if not cores:
        return
    try:
        os.sched_setaffinity(0, set(int(c) for c in cores))
        log.info("pinned to cores %s", sorted(set(cores)))
    except (AttributeError, OSError) as exc:
        log.warning("could not set CPU affinity to %s: %s", cores, exc)


# --- bring-up helpers -------------------------------------------------------


def cmd_list_devices() -> int:
    """Every attached UVC camera. Does NOT claim anything, so it is safe to run
    while `teleimager` is serving."""
    import uvc

    devices = uvc.device_list()
    if not devices:
        print("no UVC devices found")
        return 1
    for d in devices:
        path = _sysfs_path_for(d)
        print(f"uid={d['uid']:8s} serial={str(d.get('serialNumber')):16s} "
              f"name={d.get('name')!r:28s} physical_path={path or '?'}")
    print("\nUse physical_path in cameras.yaml — uid is bus:device_address and "
          "changes on replug.")
    return 0


def _sysfs_path_for(device: dict) -> str | None:
    """The sysfs USB path whose busnum/devnum match this device's uid."""
    want = (str(device.get("bus_number")), str(device.get("device_address")))
    for node in sorted(SYSFS_USB.glob("*")):
        try:
            bus = (node / "busnum").read_text().strip()
            dev = (node / "devnum").read_text().strip()
        except OSError:
            continue
        if (bus, dev) == want:
            return node.name
    return None


def cmd_probe(config: dict) -> int:
    """Open each configured camera and dump its modes and current controls.

    This is the bring-up tool: it answers "does this unit offer 1280x720 MJPG?"
    and "what are its controls actually set to right now?" — the latter being how
    a reference rig's settings get read off and written into `cameras.yaml`.

    REQUIRES `teleimager` STOPPED, because it claims the cameras.
    """
    import uvc

    rc = 0
    for topic, spec in config["cameras"].items():
        print(f"\n===== {topic} =====")
        try:
            uid = resolve_uid(topic, spec)
            cap = uvc.Capture(uid)
        except Exception as exc:                # noqa: BLE001
            print(f"  FAILED: {exc}")
            rc = 1
            continue
        try:
            modes = sorted({(m.width, m.height, m.fps, m.format_name)
                            for m in cap.available_modes})
            print(f"  uid={uid}  modes:")
            for m in modes:
                mark = " <= configured" if (
                    m[0] == int(spec["image_shape"][1])
                    and m[1] == int(spec["image_shape"][0])
                    and m[2] == int(spec["fps"]) and m[3] == "MJPG") else ""
                print(f"    {m[0]}x{m[1]} @ {m[2]:>3} {m[3]}{mark}")
            print("  controls (name = value [min..max] step):")
            for c in cap.controls:
                print(f"    {c.display_name!r} = {c.value} "
                      f"[{c.min_val}..{c.max_val}] step {c.step}")
        finally:
            try:
                cap.close()
            except Exception:                   # noqa: BLE001
                pass
    return rc


# --- entrypoint -------------------------------------------------------------


def serve(config: dict) -> int:
    server = config["server"]
    pin_cores(server.get("cpu_affinity"))
    host = str(server.get("bind_host", "0.0.0.0"))

    stop = threading.Event()
    publishers = [CameraPublisher(topic, spec, host=host, stop=stop)
                  for topic, spec in config["cameras"].items()]
    try:
        for p in publishers:
            p.open()
            p.bind()
        for p in publishers:
            p.start()
        log.info("serving %d camera(s); ctrl-c to stop", len(publishers))

        report = float(server.get("report_period_s", 10.0))
        last = {p.topic: 0 for p in publishers}
        while not stop.is_set():
            time.sleep(report)
            if stop.is_set():
                break
            for p in publishers:
                rate = (p.seq - last[p.topic]) / report
                last[p.topic] = p.seq
                log.info("[%s] %.1f Hz  seq=%d  hwm_dropped=%d",
                         p.topic, rate, p.seq, p.dropped)
    except KeyboardInterrupt:
        log.info("interrupted")
    finally:
        stop.set()
        for p in publishers:
            # `is_alive()` before `join()`: when one camera fails to open, the
            # threads after it were never started, and joining an unstarted
            # Thread raises — burying the real error ("Resource busy") under a
            # RuntimeError from the cleanup path.
            if p.is_alive():
                p.join(timeout=2.0)
            p.close()
    return 1 if any(p.seq == 0 for p in publishers) else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Fisheye image server (robot board).")
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--list-devices", action="store_true",
                        help="list attached UVC cameras; claims nothing")
    parser.add_argument("--probe", action="store_true",
                        help="dump modes and controls per configured camera "
                             "(requires teleimager stopped)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(message)s")

    if args.list_devices:
        return cmd_list_devices()
    config = load_config(args.config)
    if args.probe:
        return cmd_probe(config)
    return serve(config)


if __name__ == "__main__":
    sys.exit(main())
