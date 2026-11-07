"""An iPhone running the phone app in its Camera role, as a camera: `source = "phone"`.

The app (`phone_teleop/ios`, the Camera role) is a TCP server on the LAN that advertises
itself over Bonjour as `_elrobotcam._tcp` under the name typed in the app (default "exo"). It
needs no Apple ID in common with this Mac (unlike a Continuity Camera). This binding:

    finds it       `device` = "auto" (the one camera phone on the LAN), part of its Bonjour
                   name, or "host[:port]" (port `PORT` by default)
    configures it  a hello with the profile's mode (width, height, fps) and the phone_* keys
                   (lens, JPEG quality, exposure policy, rotation); the app answers with what it
                   runs, or why it cannot
    syncs clocks   pings at 5 Hz; the phone answers with its own clock (the host clock its
                   capture timestamps are on). offset = phone time - the ping's midpoint, from
                   the fastest round trip of the last `SYNC_WINDOW_S`: the one least delayed
                   by WiFi queueing, symmetric to within a millisecond or so
    streams        one JPEG per frame with its capture time on the phone's clock, mapped onto
                   ours; the newest frame wins. The app drops frames itself when the WiFi
                   cannot keep up (never more than two in flight) and says how many

THE WIRE (TCP, big-endian). Every message is `HEADER` (payload length, type) and the payload.

    Mac -> phone   h  hello   JSON {"v", "width", "height", "fps", "quality" (1-100),
                              "lens" ("wide" | "ultrawide"), "exposure" ("auto" | "lock"),
                              "rotate" (0 | 180)}
                   p  ping    `PING`  t_mac
    phone -> Mac   i  info    JSON {"v", "ok", "error", "model", "name", "width", "height",
                              "fps", "lens", "locked"}; again whenever "locked" changes
                   q  pong    `PONG`  t_mac (echoed), t_phone (when the ping arrived)
                   f  frame   `FRAME` seq, dropped (by the phone so far), t_capture (phone
                              clock), exposure duration (s), ISO; then the JPEG bytes

No control channel: exposure is the phone's own, so the profile says `exposure = "camera"`.
`phone_exposure = "lock"` lets the phone settle for 2 s, then locks exposure, white balance and
focus for the connection, so every recorded frame is exposed alike.
"""

from __future__ import annotations

import ipaddress
import json
import math
import socket
import struct
import threading
import time
from collections import deque

import numpy as np

from elrobot_deploy.runtime.cameras.sources import MAX_PTS_SKEW_S, CameraError, verify_rate
from elrobot_deploy.runtime.timing import now

PROTOCOL = 1
PORT = 5006
SERVICE = "_elrobotcam._tcp.local."
HEADER = struct.Struct("!IB")
PING = struct.Struct("!d")
PONG = struct.Struct("!dd")
FRAME = struct.Struct("!IIdff")
#: Larger than any JPEG the app sends (a 4K frame at quality 100 is ~5 MB).
MAX_MESSAGE = 32 << 20
PING_EVERY_S = 0.2
#: The fastest round trip of this long wins: short enough that clock drift (tens of ppm,
#: under a millisecond here) does not matter, long enough to hold a quiet moment of the WiFi.
SYNC_WINDOW_S = 10.0
#: How long "auto" (or a name) listens for Bonjour answers.
BROWSE_S = 3.0
CONNECT_TIMEOUT_S = 5.0
#: The app reconfigures its capture session before it answers the hello.
HELLO_TIMEOUT_S = 8.0


def pack(kind: str, payload: bytes = b"") -> bytes:
    return HEADER.pack(len(payload), ord(kind)) + payload


def _read_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    view, got = memoryview(buf), 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("the phone closed the connection")
        got += k
    return bytes(buf)


def read_message(sock: socket.socket) -> tuple[str, bytes]:
    n, kind = HEADER.unpack(_read_exact(sock, HEADER.size))
    if n > MAX_MESSAGE:
        raise ConnectionError(f"a {n}-byte message: not the camera phone protocol")
    return chr(kind), _read_exact(sock, n)


class ClockSync:
    """The phone's clock -> ours, from ping round trips (the fastest in the window wins)."""

    def __init__(self, window_s: float = SYNC_WINDOW_S):
        self.window_s = float(window_s)
        self._samples: deque = deque()          # (t_recv, rtt, offset)
        self.offset: float | None = None        # phone - ours, s
        self.rtt: float | None = None           # of the sample in use, s

    def add(self, t_sent: float, t_phone: float, t_recv: float) -> None:
        rtt = t_recv - t_sent
        if rtt < 0:
            return
        self._samples.append((t_recv, rtt, t_phone - 0.5 * (t_sent + t_recv)))
        while t_recv - self._samples[0][0] > self.window_s:
            self._samples.popleft()
        _, self.rtt, self.offset = min(self._samples, key=lambda s: s[1])

    def to_local(self, t_phone: float) -> float | None:
        return None if self.offset is None else t_phone - self.offset


# --- finding the phone ----------------------------------------------------------------------


def _rank(addrs: list[str]) -> list[str]:
    """A phone's addresses, the reachable first: on this Mac's own subnet (/24), then any but
    link-local. Bonjour lists every interface a device has (a hotspot bridge, say)."""
    from elrobot_deploy.runtime.processes.p11_phone import _local_ip
    lan = _local_ip().rsplit(".", 1)[0] + "."
    return sorted(addrs, key=lambda a: (not a.startswith(lan), a.startswith("169.254.")))


def browse(timeout: float = BROWSE_S) -> list[dict]:
    """Every camera phone on the LAN: [{"name", "host", "hosts", "port"}] (IPv4; `host` is the
    likeliest of `hosts`)."""
    from zeroconf import IPVersion, ServiceBrowser, ServiceListener, Zeroconf

    names: set[str] = set()

    class Listener(ServiceListener):
        def add_service(self, zc, type_, name):
            names.add(name)

        def update_service(self, zc, type_, name):
            names.add(name)

        def remove_service(self, zc, type_, name):
            names.discard(name)

    zc = Zeroconf(ip_version=IPVersion.V4Only)
    try:
        ServiceBrowser(zc, SERVICE, Listener())
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end and not names:
            time.sleep(0.05)
        time.sleep(min(0.3, max(0.0, t_end - time.monotonic())))    # a second phone, too
        out = []
        for name in sorted(names):
            info = zc.get_service_info(SERVICE, name, timeout=2000)
            addrs = _rank(info.parsed_addresses(IPVersion.V4Only)) if info else []
            if addrs:
                out.append({"name": name.removesuffix("." + SERVICE), "host": addrs[0],
                            "hosts": addrs, "port": int(info.port)})
        return out
    finally:
        zc.close()


def parse_address(spec: str) -> tuple[str, int] | None:
    """"host[:port]" when `spec` is an address (an IP, or a name ending in .local), else None."""
    host, _, port = spec.rpartition(":") if spec.count(":") == 1 else (spec, "", "")
    host = host or spec
    try:
        ipaddress.ip_address(host)
    except ValueError:
        if not (host.endswith(".local") or host == "localhost"):
            return None
    if port and not port.isdigit():
        return None
    return host, int(port) if port else PORT


def resolve_phone(spec: str, *, browse=browse) -> tuple[list[str], int, str]:
    """([host, ...], port, name) of the phone a profile's `device` names: its addresses, the
    likeliest first."""
    spec = (spec or "auto").strip()
    addr = parse_address(spec)
    if addr is not None:
        return [addr[0]], addr[1], spec
    found = browse()
    listing = ", ".join(f"{p['name']!r} ({p['host']}:{p['port']})" for p in found) or "none"
    if spec != "auto":
        found = [p for p in found if spec.lower() in p["name"].lower()]
    if not found:
        raise CameraError(
            f"no camera phone {'on the LAN' if spec == 'auto' else f'named {spec!r}'} "
            f"(found: {listing}). Is the app open on the phone in the Camera role, on this Mac's WiFi, with Local "
            f"Network allowed? Or give its address: device = \"192.168.1.20\".")
    if len(found) > 1:
        raise CameraError(f"{len(found)} camera phones match {spec!r}: {listing}. Name one "
                          f"(device = part of its name) or give its address.")
    p = found[0]
    return list(p.get("hosts") or [p["host"]]), p["port"], p["name"]


# --- the binding ----------------------------------------------------------------------------


class PhoneCamera:
    """An iPhone running the app in its Camera role, over the LAN, at the profile's mode."""

    def __init__(self, device: str, shape: tuple[int, int, int], *, fps: float = 30.0,
                 quality: int = 85, lens: str = "wide", exposure: str = "lock", rotate: int = 0,
                 browse=browse):
        self.spec = str(device)
        self.shape = tuple(int(v) for v in shape)
        self.fps = float(fps)
        self.quality, self.lens = int(quality), str(lens)
        self.phone_exposure, self.rotate = str(exposure), int(rotate)
        self._browse = browse
        self.t_capture = self.t_recv = 0.0
        self.dropped = 0                # by the phone (WiFi too slow) + overwritten unread here
        self.errors = 0                 # undecodable frames
        self.device = ""
        self.mode = ""
        self.info: dict = {}
        self.exposure_s = self.iso = math.nan
        self.sync = ClockSync()
        self._cond = threading.Condition()
        self._back = np.zeros(self.shape, np.uint8)
        self._back_t = (0.0, 0.0)
        self._n = 0                     # frames received
        self._read = 0                  # the last one `read_into` returned
        self._overwritten = 0
        self._closed = False
        self._fault = ""
        self._sock: socket.socket | None = None
        self._send_lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    # --- the contract --------------------------------------------------------------------------

    def connect(self) -> None:
        hosts, port, name = resolve_phone(self.spec, browse=self._browse)
        sock, tried = None, []
        for host in hosts:
            try:
                sock = socket.create_connection(
                    (host, port), timeout=CONNECT_TIMEOUT_S if len(hosts) == 1 else 2.0)
                break
            except OSError as exc:
                tried.append(f"{host}: {exc}")
        if sock is None:
            raise CameraError(f"cannot reach the phone camera {name!r} on port {port} "
                              f"({'; '.join(tried)}). Is the app open in the Camera role (in "
                              f"front, the phone unlocked)?")
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = sock
        h, w = self.shape[:2]
        hello = {"v": PROTOCOL, "width": w, "height": h, "fps": self.fps,
                 "quality": self.quality, "lens": self.lens, "exposure": self.phone_exposure,
                 "rotate": self.rotate}
        try:
            sock.sendall(pack("h", json.dumps(hello).encode()))
            sock.settimeout(HELLO_TIMEOUT_S)
            kind, payload = read_message(sock)
        except (OSError, ConnectionError) as exc:
            self.close()
            raise CameraError(f"the camera phone {name!r} at {host}:{port} did not answer the "
                              f"hello: {exc}") from None
        info = json.loads(payload) if kind == "i" else {}
        if not info.get("ok"):
            self.close()
            raise CameraError(f"the camera phone {name!r}: "
                              f"{info.get('error') or f'unexpected first message {kind!r}'}")
        if (int(info.get("width", 0)), int(info.get("height", 0))) != (w, h):
            self.close()
            raise CameraError(f"the camera phone {name!r} runs {info.get('width')}x"
                              f"{info.get('height')}, the profile says {w}x{h}")
        self.info = info
        self.device = (f"{info.get('name') or name} ({info.get('model') or 'iPhone'}, "
                       f"{host}:{port})")
        self.mode = (f"{w}x{h} JPEG q{self.quality} @ {self.fps:g} fps, {self.lens} lens, "
                     f"exposure {self.phone_exposure}")
        sock.settimeout(None)           # the reader blocks; `close` shuts the socket to wake it
        self._threads = [threading.Thread(target=self._reader, name="phone-camera-rx",
                                          daemon=True),
                         threading.Thread(target=self._pinger, name="phone-camera-ping",
                                          daemon=True)]
        for t in self._threads:
            t.start()
        try:
            verify_rate(self, hint="The WiFi too slow for this mode? Lower phone_quality or "
                                   "the resolution.")
        except CameraError:
            self.close()
            raise

    def read_into(self, pixels: np.ndarray, *, timeout: float = 2.0) -> bool:
        with self._cond:
            if not self._cond.wait_for(lambda: self._n > self._read or self._closed, timeout):
                return False
            if self._n <= self._read:
                return False                            # closed, nothing new
            np.copyto(pixels, self._back)
            self.t_capture, self.t_recv = self._back_t
            self._read = self._n
        return True

    def controls(self):
        return None

    def describe(self) -> str:
        sync = (f", clock synced (rtt {self.sync.rtt * 1e3:.1f} ms)"
                if self.sync.rtt is not None else ", clock not synced yet")
        locked = ", locked" if self.info.get("locked") else ""
        return f"{self.device}: {self.mode}{locked}{sync}"

    def close(self) -> None:
        self._stop.set()
        sock, self._sock = self._sock, None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()
        for t in self._threads:
            if t is not threading.current_thread():
                t.join(timeout=1.0)
        self._threads = []
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    # --- the threads ---------------------------------------------------------------------------

    def _send(self, data: bytes) -> None:
        sock = self._sock
        if sock is None:
            raise OSError("closed")
        with self._send_lock:
            sock.sendall(data)

    def _pinger(self) -> None:
        while not self._stop.is_set():
            try:
                self._send(pack("p", PING.pack(now())))
            except OSError:
                return
            # Quickly at first: frames wait for the first offset.
            self._stop.wait(PING_EVERY_S if self.sync.offset is not None else 0.02)

    def _reader(self) -> None:
        import cv2

        t_last = -math.inf
        try:
            while not self._stop.is_set():
                kind, payload = read_message(self._sock)
                t_recv = now()
                if kind == "q":
                    t_sent, t_phone = PONG.unpack(payload)
                    self.sync.add(t_sent, t_phone, t_recv)
                elif kind == "i":
                    self.info = {**self.info, **json.loads(payload)}
                elif kind == "f":
                    _, dropped, t_cap, exp_s, iso = FRAME.unpack_from(payload)
                    t_local = self.sync.to_local(t_cap)
                    if t_local is None:
                        continue                        # no clock yet: its time would be wrong
                    img = cv2.imdecode(np.frombuffer(payload, np.uint8, offset=FRAME.size),
                                       cv2.IMREAD_COLOR)
                    if img is None:
                        self.errors += 1
                        continue
                    if img.shape != self.shape:
                        self._fault = (f"the phone sent {img.shape[1]}x{img.shape[0]}, the "
                                       f"profile says {self.shape[1]}x{self.shape[0]}")
                        continue
                    # As AvfCamera does: a capture time that is not plausibly ours, arrival.
                    if not (-MAX_PTS_SKEW_S < t_recv - t_local < MAX_PTS_SKEW_S):
                        t_local = t_recv
                    # Never after arrival, never before the previous frame (a better clock
                    # sample can move the map by a millisecond).
                    t_c = max(min(t_local, t_recv), t_last + 1e-4)
                    t_last = t_c
                    with self._cond:
                        np.copyto(self._back, img)
                        self._back_t = (t_c, t_recv)
                        if self._n > self._read:
                            self._overwritten += 1
                        self._n += 1
                        self.exposure_s, self.iso = float(exp_s), float(iso)
                        self.dropped = int(dropped) + self._overwritten
                        self._cond.notify_all()
        except (OSError, ConnectionError, struct.error, ValueError) as exc:
            if not self._stop.is_set():
                self._fault = f"{type(exc).__name__}: {exc}"
        finally:
            with self._cond:
                self._closed = True
                self._cond.notify_all()


__all__ = ["PORT", "SERVICE", "ClockSync", "PhoneCamera", "browse", "parse_address",
           "resolve_phone"]
