"""The iPhone's packets (`phone_teleop/ios`), and the Mac's echo back to it.

One UTF-8 JSON object per UDP datagram, ~60 Hz (one per ARFrame):

    {"v": 2, "seq": 1234,
     "t": 5123.456,          ARFrame capture time, s (the phone's uptime clock): the motion's clock
     "t_send": 5123.470,     send time, same clock (echoed back for the round trip)
     "pose": [16 floats],    T_world_phone, ROW-major, metres; ARKit world: gravity-aligned, +Y up
     "track": "normal",      ARKit tracking state; anything else is not to be trusted
     "world_epoch": 0,       bumped when the world origin is reset or ARKit relocalizes: poses
                             before and after are in different frames            (v2; v1 has none)
     "buttons": {"engage": false, "grip": false}}

Unknown keys are ignored, so the app can grow without breaking this side. A v1 app (no
`world_epoch`) works, without that one guard.

The echo, one datagram back per packet, also JSON:

    {"seq": 1234, "t_send": 5123.470,     the phone's own, for its round-trip time
     "t_mac": 812.123,                     this machine's arrival time (for logs)
     "state": "engaged",                   what the arm is doing for the operator (`STATES`)
     "why": "",                            why the last forced release happened
     "intervene": 7}                       the counter: changes = every press and release
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np

TRACK_NORMAL, TRACK_LIMITED, TRACK_UNAVAILABLE, TRACK_UNKNOWN = range(4)

#: The echo's `state`: `inactive` (no rollout is running: the clutch does nothing), `idle`
#: (ready: press the clutch), `engaged` (the arm follows), `released` (a guard forced a release:
#: let go of the clutch and press again).
STATES = ("inactive", "idle", "engaged", "released")


class BadPacket(ValueError):
    pass


@dataclass(frozen=True)
class PhonePacket:
    v: int
    seq: int
    t: float                    # capture time, phone clock, s
    t_send: float
    T: np.ndarray               # (4, 4) T_world_phone
    track: int                  # TRACK_*
    track_text: str
    world_epoch: int            # 0 for a v1 phone
    engage: bool
    grip: bool


def _track(text: str) -> int:
    if text == "normal":
        return TRACK_NORMAL
    if text.startswith("limited"):
        return TRACK_LIMITED
    if text == "notAvailable":
        return TRACK_UNAVAILABLE
    return TRACK_UNKNOWN


def parse(data: bytes) -> PhonePacket:
    """One datagram -> a packet, or `BadPacket` (never anything half-parsed)."""
    try:
        obj = json.loads(data)
        pose = np.asarray(obj["pose"], dtype=np.float64)
        if pose.shape != (16,):
            raise BadPacket(f"pose has {pose.size} numbers, not 16")
        T = pose.reshape(4, 4)
        if not np.all(np.isfinite(T)):
            raise BadPacket("pose is not finite")
        R = T[:3, :3]
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-3):
            raise BadPacket("pose rotation is not orthonormal")
        buttons = obj.get("buttons") or {}
        t = float(obj["t"])
        if not math.isfinite(t):
            raise BadPacket("t is not finite")
        track = str(obj.get("track", ""))
        return PhonePacket(v=int(obj.get("v", 1)), seq=int(obj["seq"]), t=t,
                           t_send=float(obj.get("t_send", t)), T=T, track=_track(track),
                           track_text=track, world_epoch=int(obj.get("world_epoch", 0)),
                           engage=bool(buttons.get("engage", False)),
                           grip=bool(buttons.get("grip", False)))
    except BadPacket:
        raise
    except (ValueError, KeyError, TypeError) as exc:
        raise BadPacket(f"{type(exc).__name__}: {exc}") from None


def echo(pkt: PhonePacket, *, t_mac: float, state: str, why: str = "",
         intervene: int = 0) -> bytes:
    return json.dumps({"seq": pkt.seq, "t_send": pkt.t_send, "t_mac": round(t_mac, 6),
                       "state": state, "why": why, "intervene": int(intervene)}).encode()


def packet(seq: int, t: float, T, *, engage=False, grip=False, track="normal",
           world_epoch=0, t_send=None) -> bytes:
    """Build a v2 datagram (the fake phone and the tests)."""
    T = np.asarray(T, dtype=np.float64)
    return json.dumps({"v": 2, "seq": int(seq), "t": float(t),
                       "t_send": float(t if t_send is None else t_send),
                       "pose": T.reshape(16).tolist(), "track": track,
                       "world_epoch": int(world_epoch),
                       "buttons": {"engage": bool(engage), "grip": bool(grip)}}).encode()


#: The user's grip (screen facing left, back camera right, top forward), as a rotation in an
#: ARKit world whose -Z is the operator's forward: columns are the device axes. The fake phone
#: holds it; `RoomFrame` never needs it (it takes the heading at each press).
GRIP_IN_WORLD = np.c_[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]


__all__ = ["GRIP_IN_WORLD", "STATES", "TRACK_LIMITED", "TRACK_NORMAL", "TRACK_UNAVAILABLE",
           "TRACK_UNKNOWN", "BadPacket", "PhonePacket", "echo", "packet", "parse"]
