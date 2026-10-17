"""Bring up the DDS domain on the robot's subnet — by ADDRESS when a name is ambiguous.

`channel.py` hands CycloneDDS `<NetworkInterface name="enp4s0"/>` (the config
unitree_sdk2py generates, kept verbatim). When that NIC
carries more than one IPv4 address, Cyclone binds and ADVERTISES one of them,
and it is not necessarily the robot's subnet. Measured with a NIC holding both
a 192.168.1.x address (another network's DHCP) and a 192.168.123.x one (the
robot's): by name, 0 `rt/lowstate` samples in 5 s; by the robot-subnet address,
5197. The robot publishes; it just sends to a locator it cannot
reach. P0 then sits in CHECK "waiting on p1_state" forever, and whether it
happens depends only on whether the box was also plugged into another
network at launch — which is why it looks intermittent.

So: an interface NAME with several IPv4 addresses is resolved to its
192.168.123.x address (the robot subnet) and the config template
is switched to `address=`. A name with one address, an explicit
address, or no interface at all are passed through unchanged.

For the operator: `docs/setup/machine_toml.md` (`[network] iface`);
`elrobot_deploy check` prints this module's diagnosis.
"""

from __future__ import annotations

import dataclasses
import ipaddress
import os
import re
import subprocess
from collections.abc import Callable

#: The robot's subnet: every G1-D board and this machine's cable address.
ROBOT_NET = ipaddress.ip_network("192.168.123.0/24")


def ipv4_addresses(iface: str) -> list[str]:
    """IPv4 addresses on `iface`, in the kernel's order. Empty if unknown."""
    try:
        out = subprocess.run(("ip", "-4", "-o", "addr", "show", "dev", iface), check=False, capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    addrs = []
    for line in out.splitlines():
        parts = line.split()
        if "inet" in parts:
            addrs.append(parts[parts.index("inet") + 1].split("/")[0])
    return addrs


def bind_address(iface: str, addrs: list[str] | None = None) -> str | None:
    """The address to bind by, or None to keep binding by name."""
    try:
        ipaddress.ip_address(iface)
        return iface                          # already an address
    except ValueError:
        pass
    addrs = ipv4_addresses(iface) if addrs is None else addrs
    if len(addrs) <= 1:
        return None                           # unambiguous: unchanged behaviour
    robot = [a for a in addrs if ipaddress.ip_address(a) in ROBOT_NET]
    return robot[0] if robot else None


#: The address `remedy` suggests this machine hold on the robot subnet,
#: permanently — a form to fill in, since any FREE address on the subnet does
#: (not the robot's own .161/.164, and not another computer's). A DHCP lease
#: alone, or an address added by hand, does not survive a re-plug; the remedy
#: pins one in NetworkManager next to DHCP, so it is there whatever DHCP does.
STATIC_ADDR = "192.168.123.<a free number, e.g. 222>/24"

#: Where the config template sends Cyclone's trace (`channel.py`,
#: Verbosity=config, as unitree_sdk2py's does). Its `ownip` line is the one fact about what DDS actually
#: bound, from outside the process that bound it.
CDDS_LOG = "/tmp/cdds.LOG"


def nm_connection(iface: str) -> str | None:
    """The NetworkManager profile active on `iface`, for the remedy. None if unknown."""
    try:
        out = subprocess.run(("nmcli", "-t", "-f", "NAME,DEVICE", "con", "show", "--active"), check=False, capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        name, _, dev = line.rpartition(":")
        if dev == iface:
            return name
    return None


def remedy(iface: str) -> str:
    """How to make the robot-subnet address permanent on `iface` (NM, beside DHCP)."""
    con = nm_connection(iface) or "<the NM connection on " + iface + ">"
    return (f'give {iface} a permanent robot-subnet address: sudo nmcli con mod '
            f'"{con}" ipv4.method auto ipv4.addresses "{STATIC_ADDR}" && '
            f'sudo nmcli con up "{con}"')


def has_nic(iface: str) -> bool:
    try:
        return subprocess.run(("ip", "link", "show", "dev", iface), check=False, capture_output=True,
                              timeout=5).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def own_ip(since: float = 0.0) -> str | None:
    """What Cyclone bound, per its trace — the LAST `ownip` line, since every
    process that inits the factory rewrites the same file. None if there is no
    trace, or none written after `since` (a wall-clock time): yesterday's run
    must not testify about today's."""
    try:
        if os.stat(CDDS_LOG).st_mtime < since:
            return None
        with open(CDDS_LOG, encoding="utf-8", errors="replace") as f:
            hits = re.findall(r"ownip:\s*udp/(\d+\.\d+\.\d+\.\d+)", f.read())
    except OSError:
        return None
    return hits[-1] if hits else None


def answers(robot: str, src: str) -> bool:
    """One ICMP echo to `robot` FROM `src`, so it proves the cable behind the
    address DDS binds and not some other route. Unprivileged on Linux."""
    try:
        return subprocess.run(("ping", "-c", "1", "-W", "1", "-I", src, robot), check=False, capture_output=True, timeout=3).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@dataclasses.dataclass(frozen=True)
class Diagnosis:
    """Why `rt/lowstate` might be silent, checked in dependency order.

    Each field is a precondition for the next; `explain()` names the FIRST one
    that fails, with what to do about it, and — when none fails — says so, so
    the operator knows to look at the robot and not the network.
    """
    iface: str
    robot: str
    nic: bool                     # `iface` is a NIC on this machine
    addrs: tuple[str, ...]        # its IPv4 addresses
    bound: str | None             # the robot-subnet address DDS binds (or would)
    ownip: str | None             # what a running Cyclone says it bound, if any
    reach: bool | None            # robot answers ping from `bound`; None = not tried

    @property
    def ok(self) -> bool:
        return (self.nic and self.bound is not None
                and (self.ownip is None or self.ownip == self.bound)
                and self.reach is not False)

    def explain(self) -> str:
        i, r = self.iface, self.robot
        if not self.nic:
            return (f"{i} is not a NIC on this machine — no 192.168.123.x "
                    f"address was found on any NIC. "
                    f"{remedy('<the NIC cabled to the robot>')}")
        if self.bound is None:
            held = ", ".join(self.addrs) or "no IPv4 address"
            return (f"{i} holds {held}: nothing on {ROBOT_NET}, so DDS binds off the "
                    f"robot's subnet and {r} can never reach it. {remedy(i)}")
        if self.ownip is not None and self.ownip != self.bound:
            return (f"DDS bound {self.ownip} ({CDDS_LOG}), not {self.bound} — the "
                    f"robot sends to a locator it cannot reach. Restart the run; if it "
                    f"repeats, `elrobot_deploy.runtime.drivers.dds.domain.init` is not being used for this NIC.")
        if self.reach is False:
            return (f"{i} is {self.bound} and DDS binds it, but {r} does not answer "
                    f"ping from it: the robot is off, or the cable between them is not.")
        return (f"{i} is {self.bound}, DDS binds it and {r} answers from it: the "
                f"network is fine and the robot is up, but it publishes nothing — is "
                f"it out of e-stop and in the mode that publishes rt/lowstate?")


def diagnose(iface: str, robot: str, *, ping: bool | Callable[[str, str], bool] = True,
             addrs: list[str] | None = None, nic: bool | None = None,
             ownip: str | None | Callable[[], str | None] = own_ip) -> Diagnosis:
    """Check the preconditions for hearing the robot on `iface`, in order.

    `ping=False` skips the one probe that costs time (up to 1 s), for a call
    on the CHECK critical path; the keyword injections are for tests.
    """
    if nic is None:
        nic = has_nic(iface)
    if addrs is None:
        addrs = ipv4_addresses(iface) if nic else []
    on_net = [a for a in addrs if ipaddress.ip_address(a) in ROBOT_NET]
    bound = bind_address(iface, addrs) or (on_net[0] if on_net else None)
    own = ownip() if callable(ownip) else ownip
    reach: bool | None = None
    if bound is not None and ping is not False:
        reach = (answers if ping is True else ping)(robot, bound)
    return Diagnosis(iface, robot, nic, tuple(addrs), bound, own, reach)


def init(domain: int, iface: str | None) -> str | None:
    """`ChannelFactoryInitialize`, bound where the robot can reach us.

    Returns the address bound by, or None when binding by name / auto.
    Idempotent: the first call in a process brings the domain up.
    """
    import importlib

    # Through sys.modules, so a test that stubs the channel module is honoured.
    channel = importlib.import_module("elrobot_deploy.runtime.drivers.dds.channel")

    if not iface:
        channel.ChannelFactoryInitialize(int(domain))
        return None
    addr = bind_address(iface)
    if addr is not None:
        tpl = getattr(channel, "ChannelConfigHasInterface", "")
        if 'name="$__IF_NAME__$"' in tpl:
            channel.ChannelConfigHasInterface = tpl.replace(
                'name="$__IF_NAME__$"', 'address="$__IF_NAME__$"')
        channel.ChannelFactoryInitialize(int(domain), addr)
        return addr
    channel.ChannelFactoryInitialize(int(domain), iface)
    return None


__all__ = ["CDDS_LOG", "ROBOT_NET", "STATIC_ADDR", "Diagnosis", "bind_address",
           "diagnose", "init", "ipv4_addresses", "own_ip", "remedy"]
