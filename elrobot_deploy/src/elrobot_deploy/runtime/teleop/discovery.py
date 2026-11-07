"""This Mac on the LAN, for the phone app: the teleop port advertised over Bonjour.

The app (`phone_teleop/ios`, Controller role) browses for `SERVICE` and lists every Mac it
finds by its computer name; it connects to the chosen one by that name, so nobody types an
IP address, and a Mac whose DHCP address changed is still found. Only the LAN address is
advertised (`_local_ip` in P11): a VPN's or a proxy's tunnel address would be listed too,
and is unreachable from the phone.

A failure to advertise is not fatal: the app can still be given the address by hand.
"""

from __future__ import annotations

import socket
import subprocess
import threading

SERVICE = "_elrobot._udp.local."


def computer_name() -> str:
    """The name macOS shows for this machine ("Chris's MacBook Pro"), else the hostname."""
    try:
        name = subprocess.run(["scutil", "--get", "ComputerName"], capture_output=True,
                              text=True, timeout=1, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        name = ""
    return name or socket.gethostname().removesuffix(".local")


class Advert:
    """The teleop port on `ip`, under this Mac's name, until `close`. Registration (Bonjour
    probes the LAN for a name clash, about a second) runs on its own thread: nothing waits
    for it. `log` says the name it got, or why it failed (the app can still be given the
    address by hand)."""

    def __init__(self, port: int, ip: str, *, name: str | None = None, log=print):
        self.ip, self.port, self.log = ip, int(port), log
        self.name = (name or computer_name())[:60]
        self._zc = self._info = None
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._register, name="bonjour", daemon=True)
        self._thread.start()

    def _register(self) -> None:
        try:
            from zeroconf import ServiceInfo, Zeroconf

            zc = Zeroconf(interfaces=[self.ip])
            info = ServiceInfo(SERVICE, f"{self.name}.{SERVICE}",
                               addresses=[socket.inet_aton(self.ip)], port=self.port,
                               properties={"v": "2"})
            # A second session's advert (or a stale one) under the same name: take a new one
            # ("name (2)") rather than fail.
            zc.register_service(info, allow_name_change=True)
            self._zc, self._info = zc, info
            self.name = info.name.removesuffix("." + SERVICE)
            if not self._closed.is_set():
                self.log(f"[P11] on the LAN as {self.name!r}: the app lists this Mac by that "
                         f"name")
        except Exception as exc:  # noqa: BLE001
            self.log(f"[P11] not advertised on the LAN ({type(exc).__name__}: {exc}): give "
                     f"the app this Mac's IP, {self.ip}")

    def close(self) -> None:
        self._closed.set()
        self._thread.join(timeout=3.0)
        if self._zc is not None:
            try:
                self._zc.unregister_service(self._info)
            finally:
                self._zc.close()


def advertise(port: int, ip: str, *, log=print) -> Advert | None:
    """`Advert`, or None (said once) without a LAN address."""
    if ip.startswith("127."):
        log("[P11] not advertised on the LAN (no LAN address): give the app this Mac's IP")
        return None
    return Advert(port, ip, log=log)


__all__ = ["SERVICE", "Advert", "advertise", "computer_name"]
