"""The lift column's wire: a height in, a velocity out.

`rt/hispeed_state` (`geometry_msgs/Point32`, `.y` = height in metres) and
`rt/cmd_hispeed` (the same type, `.z` = vertical velocity in [-1, 1], + is up).
Both are on the control board's own firmware, so they answer whenever the robot
is powered — the vendor's teleop need not be running.

P1 reads the height into the state ring (`column_h`); nothing in a rollout
commands the column.

**This is not a joint and the difference is the whole file.** Every other
actuator in this system takes a POSITION on `rt/lowcmd` at 500 Hz and reports
itself in the same 29-slot message. The column takes a VELOCITY, on its own
topic, at ~30 Hz, and reports on a third. Two consequences that shape
everything downstream:

  * **a velocity command persists until it is contradicted.** The publish rate
    is therefore a SAFETY parameter and not a tracking one: it bounds how far
    the column travels after something decides it should stop. At 30 Hz and the
    full 76.5 mm/s that is 2.5 mm; at 2 Hz it is 38 mm. Nothing here needs
    30 Hz to follow a trajectory — the column is slow — and everything needs it
    to stop.

  * **it must be zeroed on the way out, by whoever is still alive.** An arm that
    loses its publisher holds position; a column that loses its publisher keeps
    moving. `stop()` exists for that and is deliberately idempotent and
    repeat-safe, because the only thing worse than sending it is not sending it.

This is the raw-topic interface, verified on the reference robot: 30 Hz on
the raw topic rather than the AGV service's `HeightAdjust` (api 1002) that
wraps it.
"""

from __future__ import annotations

#: What `.z = 1.0` means, metres per second. Unitree high-level API 1002: the
#: normalised command is scaled by 76.5 mm/s.
FULL_SCALE_MPS = 0.0765

#: The command topic's own rate, Hz: what has been driven on the reference
#: robot. A SAFETY parameter (see the module docstring), not a tracking one.
COMMAND_HZ = 30.0

#: How many stops `stop()` sends. `move_g1d` sends five on the way out and does
#: not wait for an acknowledgement, because there is none to wait for: the topic
#: is fire-and-forget and a single dropped datagram would leave the column
#: moving.
STOP_REPEATS = 5


class DdsColumnState:
    """`rt/hispeed_state` — the column's height, metres.

    `read()` blocks for one sample and returns the height, or None at EOF, which
    is the shape `p1_state.ColumnLatch` pumps. It is deliberately NOT the
    `(dtype, read(out) -> bool)` shape the lowstate sources use: there is one
    float here, and a one-field structured record would be ceremony.
    """

    def __init__(self, hw):
        self._hw = hw
        self._sub = None

    def connect(self) -> None:
        from g1d_deploy.runtime.drivers.dds.channel import ChannelSubscriber
        from g1d_deploy.runtime.drivers.dds.idl.geometry_msgs import Point32_

        from g1d_deploy.runtime.drivers.robot_state import _init_channels

        _init_channels(self._hw)
        self._sub = ChannelSubscriber("rt/hispeed_state", Point32_)
        self._sub.Init()

    def read(self) -> float | None:
        msg = self._sub.Read()              # blocks until a sample arrives
        return None if msg is None else float(msg.y)

    def close(self) -> None:
        if self._sub is not None:
            self._sub.Close()
            self._sub = None


class DdsColumnCmd:
    """`rt/cmd_hispeed` — the column's velocity, normalised to [-1, 1].

    `publish` takes **metres per second** and does the scaling here, so no
    caller has to remember that the wire is normalised. It clamps rather than
    refuses: this is the last point a command passes through before the robot,
    and the same argument P8's range clamp is built on.
    """

    def __init__(self, hw):
        self._hw = hw
        self._pub = None
        self._msg = None

    def connect(self) -> None:
        from g1d_deploy.runtime.drivers.dds.channel import ChannelPublisher
        from g1d_deploy.runtime.drivers.dds.idl.geometry_msgs import Point32_
        from g1d_deploy.runtime.drivers.dds.idl.geometry_msgs import (default_Point32_ as MkPoint)

        from g1d_deploy.runtime.drivers.robot_state import _init_channels

        _init_channels(self._hw)
        self._pub = ChannelPublisher("rt/cmd_hispeed", Point32_)
        self._pub.Init()
        self._msg = MkPoint()

    def publish(self, v_mps: float) -> None:
        if self._pub is None:
            raise RuntimeError("publish before connect()")
        z = float(v_mps) / FULL_SCALE_MPS
        self._msg.z = max(-1.0, min(1.0, z))
        self._pub.Write(self._msg)

    def stop(self) -> None:
        """Zero velocity, sent `STOP_REPEATS` times. Safe to call at any point,
        including before `connect` and more than once."""
        if self._pub is None:
            return
        self._msg.z = 0.0
        for _ in range(STOP_REPEATS):
            self._pub.Write(self._msg)

    def close(self) -> None:
        self.stop()
        self._pub = None


# --- fakes ------------------------------------------------------------------


class FakeColumnState:
    """Replays a sequence of heights, then EOF. For driving P1 with no robot."""

    def __init__(self, heights=()):
        self.heights = list(heights)
        self._i = 0

    def connect(self) -> None:
        pass

    def read(self) -> float | None:
        if self._i >= len(self.heights):
            return None
        self._i += 1
        return float(self.heights[self._i - 1])

    def close(self) -> None:
        pass


class FakeColumnCmd:
    """Records what would have gone on the wire, in metres per second."""

    def __init__(self):
        self.sent: list[float] = []
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def publish(self, v_mps: float) -> None:
        if not self.connected:
            raise RuntimeError("publish before connect()")
        # Clamped exactly as the real one clamps, so a test sees what the robot
        # would see rather than what the caller asked for.
        z = max(-1.0, min(1.0, float(v_mps) / FULL_SCALE_MPS))
        self.sent.append(z * FULL_SCALE_MPS)

    def stop(self) -> None:
        self.sent.extend([0.0] * STOP_REPEATS)

    def close(self) -> None:
        self.stop()
        self.connected = False


__all__ = ["COMMAND_HZ", "FULL_SCALE_MPS", "STOP_REPEATS",
           "DdsColumnCmd", "DdsColumnState", "FakeColumnCmd", "FakeColumnState"]
