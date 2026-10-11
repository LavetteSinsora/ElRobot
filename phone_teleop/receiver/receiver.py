#!/usr/bin/env python3
"""Receive PhoneTeleop pose packets over UDP (standard library only).

    python3 receiver.py              # listen on 0.0.0.0:5005, print live stats
    python3 receiver.py --port 6000 --no-echo

Each datagram is one JSON packet (see ios/PhoneTeleop/PoseStreamer.swift).
By default every packet is echoed back as {"seq", "t_send"} so the phone can
show round-trip time on its own clock.

Library use, for the teleop loop later:

    from receiver import PoseReceiver
    rx = PoseReceiver(port=5005).start()
    pkt = rx.latest()            # newest packet dict, or None
    T = rx.latest_matrix()       # 4x4 nested list, T_world_phone
"""
import argparse
import json
import socket
import threading
import time


def local_ip() -> str:
    """LAN IP of the Wi-Fi/Ethernet interface.

    On macOS, ask the interfaces directly (en0, en1, ...). The routing trick below
    can return a VPN/proxy TUN address (e.g. 198.18.x.x from Clash/Surge), which
    the phone cannot reach.
    """
    import subprocess
    for iface in ("en0", "en1", "en2", "en3"):
        try:
            ip = subprocess.run(["ipconfig", "getifaddr", iface], capture_output=True,
                                text=True, timeout=1).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            break
        if ip:
            return ip
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))  # no packet is sent; just picks the default interface
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


class PoseReceiver:
    def __init__(self, port: int = 5005, echo: bool = True):
        self.port, self.echo = port, echo
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("0.0.0.0", port))
        self._lock = threading.Lock()
        self._latest = None
        self._recv_time = None
        self.received = 0
        self.lost = 0
        self._last_seq = None

    def start(self) -> "PoseReceiver":
        threading.Thread(target=self._run, daemon=True).start()
        return self

    def _run(self):
        while True:
            data, addr = self._sock.recvfrom(65535)
            now = time.monotonic()
            try:
                pkt = json.loads(data)
            except json.JSONDecodeError:
                continue
            if self.echo:
                self._sock.sendto(
                    json.dumps({"seq": pkt.get("seq"), "t_send": pkt.get("t_send")}).encode(), addr
                )
            seq = pkt.get("seq")
            with self._lock:
                if self._last_seq is not None and seq is not None:
                    if seq <= self._last_seq:      # reordered/duplicate: keep the newer one
                        continue
                    self.lost += seq - self._last_seq - 1
                self._last_seq = seq
                self._latest, self._recv_time = pkt, now
                self.received += 1

    def latest(self):
        with self._lock:
            return self._latest

    def age(self):
        """Seconds since the newest packet arrived (None if nothing yet)."""
        with self._lock:
            return None if self._recv_time is None else time.monotonic() - self._recv_time

    def latest_matrix(self):
        pkt = self.latest()
        if pkt is None:
            return None
        p = pkt["pose"]
        return [p[4 * r: 4 * r + 4] for r in range(4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--no-echo", action="store_true")
    args = ap.parse_args()

    rx = PoseReceiver(args.port, echo=not args.no_echo).start()
    print(f"Listening on {local_ip()}:{args.port}  (enter this IP in the phone app)")

    last_n, last_t = 0, time.monotonic()
    while True:
        time.sleep(0.25)
        pkt = rx.latest()
        now = time.monotonic()
        rate = (rx.received - last_n) / (now - last_t)
        last_n, last_t = rx.received, now
        if pkt is None:
            print("\rwaiting for packets...", end="", flush=True)
            continue
        p, b = pkt["pose"], pkt.get("buttons", {})
        print(
            f"\r{rate:5.1f} Hz  lost {rx.lost:4d}  age {rx.age() * 1000:5.1f} ms  "
            f"track {pkt.get('track', '?'):<28s} "
            f"xyz [{p[3]:+.3f} {p[7]:+.3f} {p[11]:+.3f}]  "
            f"engage {int(b.get('engage', False))} grip {int(b.get('grip', False))}   ",
            end="",
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
