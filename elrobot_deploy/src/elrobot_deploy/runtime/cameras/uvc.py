"""A UVC camera's controls (exposure, gain, white balance, ...) over USB, beside its stream.

WHY THIS EXISTS. On Linux, V4L2 gives frames and controls through one device node. macOS's
AVFoundation gives the frames of a USB camera but none of its UVC controls, so the exposure
loop (`exposure.py`) needs a second route to the same camera. This is it: the UVC class
requests (GET_CUR / SET_CUR / GET_MIN ... on the camera terminal and the processing unit),
sent as USB control transfers through libusb while AVFoundation streams.

**The requests go to the DEVICE, the interface is never claimed.** macOS's own camera driver
owns the VideoControl interface while the camera streams; claiming it would fail (or steal
the stream). A control transfer addressed to an interface can still be sent on the device's
default pipe without claiming anything, which is how macOS camera-control tools work. pyusb's
`Device.ctrl_transfer` would try to claim the interface first, so this goes one level down,
to the backend's `ctrl_transfer`. Whether this unit accepts it while streaming is what
`elrobot_deploy camera probe` checks (PORTING_PLAN.md D1, the spike).

libusb itself comes from the `libusb-package` wheel (no Homebrew), or the system's.

Which USB device is which AVFoundation camera: a USB camera's AVFoundation `uniqueID` is its
USB location ID, vendor and product in hex (`0x1421000010bb2b08` = location 0x14210000,
10bb:2b08), and libusb's bus and port numbers give the same location, so the two are matched
exactly, even between two identical cameras (`location_of`, `sources.parse_usb_uid`).
"""

from __future__ import annotations

import dataclasses

# --- the UVC vocabulary (USB Video Class 1.1 / 1.5) ----------------------------------------

CC_VIDEO = 0x0E
SC_VIDEOCONTROL = 0x01
CS_INTERFACE = 0x24
VC_INPUT_TERMINAL = 0x02
VC_PROCESSING_UNIT = 0x05
ITT_CAMERA = 0x0201

SET_CUR = 0x01
GET_CUR, GET_MIN, GET_MAX, GET_RES, GET_LEN, GET_INFO, GET_DEF = (
    0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87)
REQ_NAMES = {GET_CUR: "cur", GET_MIN: "min", GET_MAX: "max", GET_RES: "res", GET_DEF: "def"}

#: The controls this package uses, by our name: (unit, selector, size in bytes, signed).
#: "ct" is the camera terminal, "pu" the processing unit. Units: exposure in 100 us; white
#: balance in kelvin; the AE mode is a bitmask value (1 manual, 2 auto, 4 shutter priority,
#: 8 aperture priority) and its GET_RES is the set of modes the camera supports.
CONTROLS = {
    "ae_mode": ("ct", 0x02, 1, False),
    "ae_priority": ("ct", 0x03, 1, False),
    "exposure": ("ct", 0x04, 4, False),
    "backlight": ("pu", 0x01, 2, False),
    "brightness": ("pu", 0x02, 2, True),
    "contrast": ("pu", 0x03, 2, False),
    "gain": ("pu", 0x04, 2, False),
    "power_line": ("pu", 0x05, 1, False),
    "hue": ("pu", 0x06, 2, True),
    "saturation": ("pu", 0x07, 2, False),
    "sharpness": ("pu", 0x08, 2, False),
    "gamma": ("pu", 0x09, 2, False),
    "wb_temp": ("pu", 0x0A, 2, False),
    "wb_auto": ("pu", 0x0B, 1, False),
}

TIMEOUT_MS = 200


class UvcError(RuntimeError):
    """The control channel could not be opened, or a request failed."""


def _backend():
    try:
        import libusb_package
        be = libusb_package.get_libusb1_backend()
        if be is not None:
            return be
    except ImportError:
        pass
    import usb.backend.libusb1
    be = usb.backend.libusb1.get_backend()
    if be is None:
        raise UvcError("no libusb: `uv sync --extra camera` installs it (libusb-package)")
    return be


def location_of(dev) -> int:
    """macOS's USB location ID from libusb's bus and port numbers: the bus in the top byte,
    then one hex digit per hub port (0x02110000 = bus 2, port 1, port 1)."""
    loc = (int(dev.bus) & 0xFF) << 24
    for i, port in enumerate(getattr(dev, "port_numbers", None) or ()):
        if i >= 6:
            break
        loc |= (int(port) & 0xF) << (20 - 4 * i)
    return loc


@dataclasses.dataclass(frozen=True)
class UvcDevice:
    """A UVC camera on the bus: what `list_devices` reports."""

    vid: int
    pid: int
    location: int
    product: str
    interface: int              # the VideoControl interface number
    units: dict                 # {"ct": id, "pu": id}

    @property
    def usb_id(self) -> str:
        return f"{self.vid:04x}:{self.pid:04x}"

    def line(self) -> str:
        return (f"{self.usb_id} @ {self.location:#010x}  {self.product or '(no name)'}  "
                f"VC interface {self.interface}, units {self.units}")


def _vc_layout(dev) -> tuple[int, dict] | None:
    """(VideoControl interface number, {"ct": camera terminal id, "pu": processing unit id})
    from the configuration descriptor's class-specific bytes, or None if not a UVC camera."""
    for cfg in dev:
        for intf in cfg:
            if intf.bInterfaceClass != CC_VIDEO or intf.bInterfaceSubClass != SC_VIDEOCONTROL:
                continue
            extra = bytes(intf.extra_descriptors or b"")
            units, i = {}, 0
            while i + 3 <= len(extra):
                n = extra[i]
                if n < 3 or i + n > len(extra):
                    break
                d = extra[i:i + n]
                if d[1] == CS_INTERFACE:
                    if d[2] == VC_INPUT_TERMINAL and n >= 6 and \
                            int.from_bytes(d[4:6], "little") == ITT_CAMERA:
                        units["ct"] = d[3]
                    elif d[2] == VC_PROCESSING_UNIT:
                        units["pu"] = d[3]
                i += n
            return intf.bInterfaceNumber, units
    return None


def list_devices() -> list[UvcDevice]:
    """Every UVC camera libusb can see (no device is opened beyond reading descriptors)."""
    import usb.core
    import usb.util

    out = []
    for dev in usb.core.find(find_all=True, backend=_backend()):
        try:
            got = _vc_layout(dev)
        except Exception:  # noqa: BLE001, S112
            continue
        if got is None:
            continue
        try:
            product = usb.util.get_string(dev, dev.iProduct) if dev.iProduct else ""
        except Exception:  # noqa: BLE001
            product = ""
        out.append(UvcDevice(dev.idVendor, dev.idProduct, location_of(dev), product or "",
                             got[0], got[1]))
    return out


class UvcControls:
    """The control channel of one camera.

        ctl = UvcControls.open(vid=0x10bb, pid=0x2b08, location=0x14210000)
        ctl.get("exposure"); ctl.set("gain", 64); ctl.range("gain")

    `location` picks one of several identical cameras; None = the only one with that id.
    """

    def __init__(self, dev, info: UvcDevice):
        self._dev = dev
        self.info = info
        self._ranges: dict = {}

    @classmethod
    def open(cls, *, vid: int | None = None, pid: int | None = None,
             location: int | None = None) -> UvcControls:
        cams = [d for d in list_devices()
                if (vid is None or d.vid == vid) and (pid is None or d.pid == pid)
                and (location is None or d.location == location)]
        want = (f"{vid:04x}:{pid:04x}" if vid is not None and pid is not None else "a UVC camera") \
            + (f" at {location:#010x}" if location is not None else "")
        if not cams:
            seen = "; ".join(d.line() for d in list_devices()) or "none"
            raise UvcError(f"no {want} on USB (UVC cameras libusb sees: {seen})")
        if len(cams) > 1:
            raise UvcError(f"{len(cams)} cameras match {want}: "
                           + "; ".join(d.line() for d in cams) + ". Name it by location.")
        info = cams[0]
        if "ct" not in info.units and "pu" not in info.units:
            raise UvcError(f"{info.line()}: no camera terminal or processing unit described")
        import usb.core
        dev = usb.core.find(backend=_backend(), custom_match=lambda d: (
            d.idVendor == info.vid and d.idProduct == info.pid
            and location_of(d) == info.location))
        if dev is None:
            raise UvcError(f"{info.line()} disappeared")
        return cls(dev, info)

    # --- requests ------------------------------------------------------------------------

    def _transfer(self, bm, req, name, data_or_len):
        unit_kind, selector, _, _ = CONTROLS[name]
        unit = self.info.units.get(unit_kind)
        if unit is None:
            raise UvcError(f"{name}: this camera describes no {unit_kind} unit")
        ctx = self._dev._ctx
        ctx.managed_open()          # opens the device handle; claims nothing
        w_value, w_index = selector << 8, (unit << 8) | self.info.interface
        import array
        if isinstance(data_or_len, int):
            buf = array.array("B", bytes(data_or_len))
            n = ctx.backend.ctrl_transfer(ctx.handle, bm, req, w_value, w_index, buf,
                                          TIMEOUT_MS)
            return bytes(buf[:n])
        buf = array.array("B", data_or_len)
        return ctx.backend.ctrl_transfer(ctx.handle, bm, req, w_value, w_index, buf, TIMEOUT_MS)

    def get(self, name: str, req: int = GET_CUR) -> int:
        _, _, size, signed = CONTROLS[name]
        try:
            raw = self._transfer(0xA1, req, name, size)
        except UvcError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise UvcError(f"{name} {REQ_NAMES.get(req, hex(req))}: {exc}") from None
        if len(raw) != size:
            raise UvcError(f"{name}: {len(raw)} bytes back, expected {size}")
        return int.from_bytes(raw, "little", signed=signed)

    def set(self, name: str, value: int) -> None:
        _, _, size, signed = CONTROLS[name]
        try:
            self._transfer(0x21, SET_CUR, name, int(value).to_bytes(size, "little", signed=signed))
        except UvcError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise UvcError(f"{name} := {value}: {exc}") from None

    def range(self, name: str) -> dict:
        """{min, max, res, def} of a control (cached), or {} if the camera has no such
        control. The AE mode's `res` is the bitmask of modes it supports."""
        if name not in self._ranges:
            out = {}
            try:
                for req in (GET_MIN, GET_MAX, GET_RES, GET_DEF):
                    out[REQ_NAMES[req]] = self.get(name, req)
            except UvcError:
                out = {}
            self._ranges[name] = out
        return self._ranges[name]

    def default(self, name: str) -> int | None:
        return self.range(name).get("def")

    def write(self, name: str, value: int | None) -> int:
        """Set `name` (None = its default), clamped into its range. Returns what was sent."""
        r = self.range(name)
        if value is None:
            if "def" not in r:
                raise UvcError(f"{name}: no default to restore (the control is not supported)")
            value = r["def"]
        if name != "ae_mode" and "min" in r and "max" in r:
            value = min(max(int(value), r["min"]), r["max"])
        self.set(name, int(value))
        return int(value)

    def probe(self) -> dict:
        """Every known control's {min, max, res, def, cur}, or {"error": ...}."""
        out = {}
        for name in CONTROLS:
            r = dict(self.range(name))
            if not r:
                out[name] = {"error": "not supported"}
                continue
            try:
                r["cur"] = self.get(name)
            except UvcError as exc:
                r["cur_error"] = str(exc)
            out[name] = r
        return out

    def close(self) -> None:
        try:
            import usb.util
            usb.util.dispose_resources(self._dev)
        except Exception:  # noqa: BLE001, S110
            pass


__all__ = ["CONTROLS", "UvcControls", "UvcDevice", "UvcError", "list_devices", "location_of"]
