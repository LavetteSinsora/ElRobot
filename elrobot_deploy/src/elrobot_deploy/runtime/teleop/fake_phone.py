"""A fake iPhone: sends packet-v2 datagrams like the app does, for a run without the phone.

    elrobot_deploy fake-phone                      # to this machine, the profile's port
    elrobot_deploy fake-phone --host 127.0.0.1 --circle-cm 3 --seconds 20

It holds the user's grip (screen left, top forward), waits until the Mac's echo says `idle`
(a rollout is running and the clutch is free), then `--idle` more seconds with the clutch up,
then holds the clutch and moves the phone around a circle in the operator's left/up plane
(and back and forth along forward) while squeezing the gripper on and off, then lets go. With
`elrobot_deploy run --sim` it drives the simulated arm through the whole chain: P11, P6 and the
IK, the bus process. It prints the Mac's echo state as it changes.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time

import numpy as np

from elrobot_deploy.runtime.teleop.phone import GRIP_IN_WORLD, packet

#: The ARKit world's directions for an operator facing -Z (the app's start heading).
FORWARD, LEFT, UP = np.array([0, 0, -1.0]), np.array([-1.0, 0, 0]), np.array([0, 1.0, 0])


def trajectory(t: float, *, radius: float, freq: float, idle: float, seconds: float):
    """(T_world_phone, engage, grip) at time `t` since the start."""
    T = np.eye(4)
    T[:3, :3] = GRIP_IN_WORLD
    T[:3, 3] = [0.0, 1.2, 0.0]
    if t < idle or t > idle + seconds:
        return T, False, False
    s = t - idle
    a = 2 * math.pi * freq * s
    ramp = min(1.0, s / 1.0)                    # ease into the circle over the first second
    T[:3, 3] += ramp * radius * (math.sin(a) * LEFT + (1 - math.cos(a)) * UP
                                 + 0.5 * math.sin(0.5 * a) * FORWARD)
    grip = int(s / 2.0) % 2 == 1                # 2 s squeezing, 2 s open
    return T, True, grip


def main(argv=None) -> int:
    from elrobot_deploy.config_loading.config import load
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=None, help="default: the profile's teleop.port")
    ap.add_argument("--hz", type=float, default=60.0)
    ap.add_argument("--idle", type=float, default=1.0, help="s with the clutch up first")
    ap.add_argument("--seconds", type=float, default=12.0, help="s with the clutch held")
    ap.add_argument("--circle-cm", type=float, default=3.0)
    ap.add_argument("--freq", type=float, default=0.25, help="circles per second")
    ap.add_argument("--tail", type=float, default=1.0, help="s with the clutch up at the end")
    args = ap.parse_args(argv)
    port = args.port or load().teleop.port

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)
    t0 = time.monotonic()
    t_go = None                                   # when the Mac first said "idle"
    seq, state = 0, None
    total = args.idle + args.seconds + args.tail
    print(f"fake phone -> {args.host}:{port}: waiting for the Mac to be ready, then {total:.0f} s")
    while True:
        t = 0.0 if t_go is None else time.monotonic() - t_go
        if t > total:
            break
        T, engage, grip = trajectory(t, radius=args.circle_cm / 100, freq=args.freq,
                                     idle=args.idle, seconds=args.seconds)
        seq += 1
        now = time.monotonic()
        try:
            sock.sendto(packet(seq, now, T, engage=engage, grip=grip, t_send=now),
                        (args.host, port))
        except OSError as exc:
            print(f"send failed: {exc}")
        try:
            while True:
                reply = json.loads(sock.recv(65535))
                if reply.get("state") != state:
                    state = reply.get("state")
                    why = f" ({reply['why']})" if reply.get("why") else ""
                    print(f"{t:6.2f} s  mac: {state}{why}")
                if t_go is None and state == "idle":
                    t_go = time.monotonic()
        except (BlockingIOError, json.JSONDecodeError):
            pass
        except OSError:
            pass
        time.sleep(max(0.0, (seq / args.hz) - (time.monotonic() - t0)))
    sock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
