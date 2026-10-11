"""DDS channels over CycloneDDS — the part of unitree_sdk2py g1d_deploy needs.

g1d_deploy needs exactly three things of `unitree_sdk2py.core.channel`:
`ChannelFactoryInitialize(domain, interface)`, `ChannelPublisher(topic, type)`
and `ChannelSubscriber(topic, type)` without a handler. This module is those
three, with the same names and call shapes, written directly against
`cyclonedds` so that the SDK (and its opencv dependency, and its Linux-only
CRC library) is not a dependency. The message types are
`idl/`, vendored.

**What is kept identical to the SDK (unitree_sdk2py 1.0.1,
`core/channel.py` + `core/channel_config.py`), because the robot is the
other end:**

* The domain: `Domain(id, <config>)` then `DomainParticipant(id)`, once per
  process. The config XML is the SDK's `ChannelConfigHasInterface` verbatim
  (one `<NetworkInterface name=... priority="default" multicast="default"/>`,
  Cyclone's trace at Verbosity=config into `/tmp/cdds.LOG` — which
  `domain.own_ip` reads) or `ChannelConfigAutoDetermine` with no interface.
  `domain.init` switches `name=` to `address=` for a NIC with several
  addresses.
* **QoS: none.** The SDK passes `qos=None` to the `Topic`, the `DataWriter`
  and the `DataReader` (its factory's `__qos` is never set by
  `ChannelFactoryInitialize`), so every entity takes Cyclone's defaults —
  a writer reliable, a reader best-effort, volatile, keep-last 1. Same here.
* A topic per channel (`Topic(participant, name, type)`), a writer created
  with a publication-matched listener and followed by the SDK's 0.2 s sleep
  (discovery time before the first write), a reader with no listener whose
  `Read(timeout)` is `take_one(timeout=...)`.

**What differs, none of it on the wire:** a `Read` that times out returns
None silently (the SDK printed "[Reader] take sample timeout" on every one —
P7's bounded wait polls at 10 Hz); `Write(sample, timeout)`'s wait for a
matched reader is the SDK's, with `False` on no match.

UNVERIFIED ON THE ROBOT (`docs/deploying_a_policy/running_and_live_dashboard.md`,
"Known limits"): byte-equality with the SDK's
messages and type information is tested; that the robot accepts a LowCmd
from this layer needs one supervised check.

`cyclonedds` is imported inside the functions, not at the top, so this
module imports (and can be stubbed by tests) where cyclonedds is absent.
"""

from __future__ import annotations

import threading
import time
from typing import Any

#: unitree_sdk2py `core/channel_config.py`, verbatim.
ChannelConfigHasInterface = '''<?xml version="1.0" encoding="UTF-8" ?>
    <CycloneDDS>
        <Domain Id="any">
            <General>
                <Interfaces>
                    <NetworkInterface name="$__IF_NAME__$" priority="default" multicast="default"/>
                </Interfaces>
            </General>
            <Tracing>
                <Verbosity>config</Verbosity>
            <OutputFile>/tmp/cdds.LOG</OutputFile>
        </Tracing>
        </Domain>
    </CycloneDDS>'''

#: unitree_sdk2py `core/channel_config.py`, verbatim.
ChannelConfigAutoDetermine = '''<?xml version="1.0" encoding="UTF-8" ?>
    <CycloneDDS>
        <Domain Id="any">
            <General>
                <Interfaces>
                    <NetworkInterface autodetermine=\"true\" priority=\"default\" multicast=\"default\" />
                </Interfaces>
            </General>
        </Domain>
    </CycloneDDS>'''

#: The SDK's pause after creating a writer, before the first write can be
#: matched: discovery time. Kept, so a first command is not published into a
#: writer the robot has not yet matched.
WRITER_SETTLE_S = 0.2

_lock = threading.Lock()
_domain = None
_participant = None


class ChannelError(RuntimeError):
    """The DDS domain could not be brought up."""


def ChannelFactoryInitialize(id: int = 0, networkInterface: str | None = None) -> None:
    """Bring up this process's DDS domain on `networkInterface` (a NIC name,
    or an address once `domain.init` has switched the template). Idempotent:
    the first call wins, as the SDK's singleton factory did."""
    global _domain, _participant
    if _participant is not None:
        return
    from cyclonedds.core import DDSException
    from cyclonedds.domain import Domain, DomainParticipant

    with _lock:
        if _participant is not None:
            return
        config = (ChannelConfigAutoDetermine if networkInterface is None else
                  ChannelConfigHasInterface.replace("$__IF_NAME__$",
                                                    networkInterface))
        try:
            domain = Domain(int(id), config)
            participant = DomainParticipant(int(id))
        except DDSException as exc:
            raise ChannelError(f"DDS domain {id} on {networkInterface!r}: "
                               f"{getattr(exc, 'msg', exc)}") from exc
        _domain, _participant = domain, participant


def _require():
    if _participant is None:
        raise ChannelError("ChannelFactoryInitialize() was not called in this "
                           "process (wire.dds.init)")
    return _participant


def _topic(name: str, type_: Any):
    from cyclonedds.topic import Topic
    return Topic(_require(), name, type_, None)


class ChannelPublisher:
    """`topic` as a writer of `type_` (qos None: Cyclone's defaults)."""

    def __init__(self, name: str, type_: Any):
        self._name = name
        self._topic = _topic(name, type_)
        self._writer = None
        self._matched = 0

    def Init(self) -> None:
        if self._writer is not None:
            return
        from cyclonedds.core import Listener
        from cyclonedds.pub import DataWriter

        def matched(writer, status) -> None:
            self._matched = status.current_count

        self._writer = DataWriter(_require(), self._topic, None,
                                  Listener(on_publication_matched=matched))
        time.sleep(WRITER_SETTLE_S)

    def Write(self, sample: Any, timeout: float | None = None) -> bool:
        """Write one sample. With `timeout`, first wait up to that long for a
        matched reader and return False if none appears (the SDK's rule)."""
        if timeout is not None:
            left = float(timeout)
            while left > 0.0 and self._matched == 0:
                time.sleep(0.1)
                left -= 0.1
            if self._matched == 0:
                return False
        from cyclonedds.core import DDSException
        try:
            self._writer.write(sample)
        except DDSException as exc:
            print(f"[dds] write on {self._name}: {getattr(exc, 'msg', exc)}",
                  flush=True)
            return False
        return True

    def Close(self) -> None:
        self._writer = None


class ChannelSubscriber:
    """`topic` as a reader of `type_` (qos None: Cyclone's defaults). No
    handler: `Read` takes on the caller's thread, so P1 is woken by the sample
    itself rather than by a callback thread's queue."""

    def __init__(self, name: str, type_: Any):
        self._name = name
        self._topic = _topic(name, type_)
        self._reader = None

    def Init(self, handler=None, queueLen: int = 0) -> None:
        if handler is not None:
            raise NotImplementedError("g1d_deploy reads on its own thread; "
                                      "no handler")
        if self._reader is None:
            from cyclonedds.sub import DataReader
            self._reader = DataReader(_require(), self._topic, None)

    def Read(self, timeout: float | None = None):
        """One sample, or None on a timeout or an error. `timeout=None`
        blocks with no bound (see `DdsLowCmd.connect` for why that is only
        ever P1's choice)."""
        from cyclonedds.core import DDSException
        from cyclonedds.util import duration
        try:
            if timeout is None:
                return self._reader.take_one()
            return self._reader.take_one(timeout=duration(seconds=timeout))
        except TimeoutError:
            return None
        except DDSException as exc:
            print(f"[dds] read on {self._name}: {getattr(exc, 'msg', exc)}",
                  flush=True)
            return None

    def Close(self) -> None:
        self._reader = None


__all__ = ["ChannelConfigAutoDetermine", "ChannelConfigHasInterface",
           "ChannelError", "ChannelFactoryInitialize", "ChannelPublisher",
           "ChannelSubscriber", "WRITER_SETTLE_S"]
