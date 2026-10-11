"""The two subscriptions P1 reads — `rt/lowstate` and the end-effector state.

Both are exposed as the same three-call source protocol so that P1's
loop (`runtime/processes/p1_state.py`) contains the *ordering* that matters and
nothing else, and so the
whole loop can be driven on a laptop from `FakeLowState` / `FakeHandState`.

**`read()` blocks.** A source is a stream, not a latest-value poll: P1 runs at
whatever rate the topic delivers (~1042 Hz), and the only honest arrival stamp
is the one taken the instant a sample lands. A polling source would report the
poll rate and would put an arbitrary sleep between arrival and stamp.

**The clock is stamped by the loop, not here.** Nothing in this module calls
`monotonic()`; `t_recv` is P1's, taken the instant `read()` returns. The cost
of that choice is stated plainly: cyclonedds deserialises the
whole sample inside `take_one`, and `read()` then copies 58 scalars out of it,
so the stamp is a few tens of microseconds after the packet actually arrived.
That constant is common to every sample and does not reorder anything — what
is required, and what the loop guarantees, is that the stamp precedes the FK,
the grip latch and the seqlock write.

**EOF is `False`, not an exception.** A source that has run out (a finite fake,
a closed subscriber) ends P1's loop the way every loop ends: by
returning. There is no reconnect and no retry here — a dead stream is invisible
by design.

What P1 writes from these: `docs/infra_logic/arena_layout_and_reads.md`, "The
records, field by field".
"""

from __future__ import annotations

import time
from typing import Protocol

import numpy as np

from g1d_deploy.runtime.shared_memory import schema

# One sample of each topic, as a 0-d structured record the caller owns. The
# caller-provided buffer is what keeps the hot path allocation-free at 1042 Hz.
LOWSTATE = np.dtype([
    ("q", "<f8", (schema.N_JOINTS,)),
    ("dq", "<f8", (schema.N_JOINTS,)),
    # `LowState_.tick`: the robot's own millisecond clock at the sample. Beside
    # P1's `t_recv` it is the delivery delay of every sample — the number that
    # says whether a stall in the state stream is the robot not producing
    # samples (tick jumps) or samples produced on time and delivered late
    # (tick advances at 2 ms, t_recv does not). 190525 showed 29 stalls of
    # ~90 ms with the sample index continuous; this is what settles where.
    ("tick", "<u4"),
])

HANDSTATE = np.dtype([
    ("grip", "<f8", (schema.GRIP_DIM,)),
])


def record(dtype) -> np.ndarray:
    """A zeroed 0-d record to hand to `read()`."""
    return np.zeros((), dtype=dtype)


class StateSource(Protocol):
    """`LowStateSource` and the hand-state source share this shape."""

    dtype: np.dtype

    def connect(self) -> None: ...

    def read(self, out: np.ndarray) -> bool:
        """Block for one sample, fill `out`, return False at EOF."""

    def close(self) -> None: ...


# The two topics differ only in their record dtype, so P1 depends on one shape
# for both. `LowStateSource` names the arm stream.
LowStateSource = StateSource


# --- DDS --------------------------------------------------------------------


def _init_channels(hw) -> None:
    """Bring up the DDS domain. Idempotent — one participant per process
    (`dds/channel.py`), so the second source joins the first one's."""
    from g1d_deploy.runtime.drivers.dds import domain as dds

    # No interface argument joins an existing domain rather than fighting a
    # running ROS 2 for it. A NIC holding several IPv4 addresses is bound by its
    # 192.168.123.x address (`dds/domain.py::init`): by name, Cyclone may advertise
    # another network's address and the robot's samples never arrive.
    addr = dds.init(hw.dds_domain, hw.dds_iface)
    if addr:
        print(f"[dds] {hw.dds_iface} has several IPv4 addresses: bound by {addr}",
              flush=True)


class DdsLowState:
    """`rt/lowstate` — 29 measured joint positions and velocities.

    Subscribed with **no handler**, which is what makes `Read()` block on the
    reader's waitset instead of delivering into a callback thread. P1 is the
    consumer and P1 is the thread that must be woken; a callback would add a
    queue between the arrival and the stamp.
    """

    dtype = LOWSTATE

    def __init__(self, hw):
        self._hw = hw
        self._sub = None

    def connect(self) -> None:
        from g1d_deploy.runtime.drivers.dds.channel import ChannelSubscriber
        from g1d_deploy.runtime.drivers.dds.idl.unitree_hg import LowState_

        _init_channels(self._hw)
        self._sub = ChannelSubscriber(self._hw.topic_lowstate, LowState_)
        self._sub.Init()

    def read(self, out: np.ndarray, timeout: float | None = None) -> bool:
        # Blocks until a sample arrives — forever, in the bare form, and a
        # blocked native read does not see Ctrl-C. P1 wants that (the ring is
        # fed by arrival, not by polling); a script that just wants to know
        # whether the robot is there passes `timeout` and asks `dds/domain.py::diagnose`
        # when this returns False.
        msg = self._sub.Read() if timeout is None else self._sub.Read(timeout)
        if msg is None:
            return False
        # `motor_state` carries 35 slots; the G1_29 contract is the first 29
        # (legs 0-11, waist 12-14, arms 15-28). The tail is unpopulated on this
        # robot and is not part of any ring.
        motors = msg.motor_state
        q, dq = out["q"], out["dq"]
        for i in range(schema.N_JOINTS):
            m = motors[i]
            q[i] = m.q
            dq[i] = m.dq
        out["tick"] = int(getattr(msg, "tick", 0)) & 0xFFFFFFFF
        return True

    def close(self) -> None:
        if self._sub is not None:
            self._sub.Close()
            self._sub = None


class DdsHandState:
    """`rt/dex1/{left,right}/state` — the measured end-effector state, 12 wide.

    The 12 dims are the `(2, 6)` contract flattened, so hand `h`
    occupies `[6h, 6h+6)`. For the Dex-1 gripper only **dim 0** of each hand is
    meaningful — the jaw radian at the encoder/gear — and the remaining ten dims
    stay zero.

    **One topic per hand, one motor per topic.** `dex1_1_gripper_server -l -r`
    runs a server per gripper and each publishes its own `MotorStates_` with the
    jaw at `states[0]`; that is what `hw.topic_hand_state[h]` names, and it is
    what the vendor's own `Dex1_Gripper_Controller` subscribes to
    (`robot_configs.dex1_default_factory`, `Gripper_Single_JointIndex.kGripper =
    0`). A single `rt/unitree_actuator/state` — two motors on
    one topic, left then right — is only the `Dex1GripperConfig` class default,
    overridden at every vendor call site, and measured silent on the reference
    robot.

    `read()` blocks on **both** topics, in order, and returns one sample only
    when each has produced. That is the conservative reading: a hand that stops
    publishing stops the latch entirely, so `t_grip_recv` freezes and a
    staleness test sees it. Taking whichever arrived first instead would let a
    live right gripper keep `t_grip_recv` moving while the left froze at its
    last value — the exact indistinguishable-constant failure P0's CHECK exists
    to catch, reintroduced one hand at a time. Both topics run at ~492 Hz, so
    the pairing costs the latch nothing but the phase between them.

    `--hand brainco` reads `rt/brainco/{left,right}/state` instead, six motors
    each, and reports dim 0 as the MEAN OF THE FOUR FINGERS — the motors that
    follow the grip value 1:1 (`gripper.brainco_targets`) — so the measured
    grip stays comparable to the commanded one. The other five dims stay zero
    there too: a per-motor layout is undecided.
    """

    dtype = HANDSTATE

    #: Each per-gripper topic carries exactly one motor, the jaw.
    JAW = 0
    #: `--hand brainco`: index, middle, ring, pinky — the motors the grip value
    #: drives 1:1.
    BRAINCO_FINGERS = (2, 3, 4, 5)

    def __init__(self, hw):
        self._hw = hw
        self._brainco = getattr(hw, "hand", "dex1") == "brainco"
        self._subs: list = []

    def connect(self) -> None:
        from g1d_deploy.runtime.drivers.dds.channel import ChannelSubscriber
        from g1d_deploy.runtime.drivers.dds.idl.unitree_go import MotorStates_

        _init_channels(self._hw)
        for topic in self._hw.topic_hand_state:
            sub = ChannelSubscriber(topic, MotorStates_)
            sub.Init()
            self._subs.append(sub)

    def read(self, out: np.ndarray) -> bool:
        grip = out["grip"]
        block = grip.reshape(schema.N_HANDS, schema.HAND_DIM)
        jaws = []
        for sub in self._subs:
            msg = sub.Read()            # blocks until a sample arrives
            if msg is None:
                return False
            if self._brainco:
                jaws.append(float(np.mean(
                    [msg.states[i].q for i in self.BRAINCO_FINGERS])))
            else:
                jaws.append(float(msg.states[self.JAW].q))
        grip[:] = 0.0
        for hand, jaw in enumerate(jaws):
            block[hand, 0] = jaw
        return True

    def close(self) -> None:
        for sub in self._subs:
            sub.Close()
        self._subs = []


# --- fakes ------------------------------------------------------------------


class _Fake:
    """Shared body of the two fakes.

    `n` is what ends a loop under test: the source reaches EOF and P1 returns,
    the same path a closed subscriber takes. `delay` paces the stream so a test
    can be about *ordering between two topics* rather than about who wins a
    free-running race.
    """

    dtype = LOWSTATE

    def __init__(self, *, n: int | None = None, delay: float = 0.0):
        self.n = n
        self.delay = float(delay)
        self.i = 0
        self.connected = False
        self.closed = False

    def connect(self) -> None:
        self.connected = True

    def _next(self) -> int | None:
        if self.n is not None and self.i >= self.n:
            return None
        if self.delay:
            time.sleep(self.delay)
        i = self.i
        self.i += 1
        return i

    def close(self) -> None:
        self.closed = True


class FakeLowState(_Fake):
    """Deterministic joints: sample `i` is a fixed function of `i` alone.

    Deterministic rather than random so that a test can assert the ring holds
    the FK of *this* sample's joints, recomputing both sides from the index.
    """

    dtype = LOWSTATE

    def read(self, out: np.ndarray) -> bool:
        i = self._next()
        if i is None:
            return False
        phase = 0.01 * i + 0.1 * np.arange(schema.N_JOINTS)
        out["q"] = 0.2 * np.sin(phase)
        out["dq"] = 0.2 * np.cos(phase)
        out["tick"] = 2 * i                 # a 500 Hz robot clock, in ms
        return True


class FakeHandState(_Fake):
    """The gripper's two jaws, opening linearly. Dims 1-5 and 7-11 stay zero,
    as they do on the wire."""

    dtype = HANDSTATE

    def read(self, out: np.ndarray) -> bool:
        i = self._next()
        if i is None:
            return False
        grip = out["grip"]
        grip[:] = 0.0
        grip[0] = 1.2 + 0.01 * i
        grip[schema.HAND_DIM] = 1.2 + 0.02 * i
        return True
