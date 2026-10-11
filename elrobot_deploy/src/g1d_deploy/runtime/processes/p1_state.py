"""P1 — the state reader.

Reads `rt/lowstate`, the hand state topics and the column
(`runtime/drivers/robot_state.py`, `runtime/drivers/column.py`); is the one
writer of the state ring. The record it writes:
`docs/infra_logic/arena_layout_and_reads.md`, "The records, field by field".

Two subscriptions, deliberately independent. The arm stream is the loop; the
end-effector stream is a latch on its own thread and touches nothing else. That
asymmetry *is* the requirement: losing the hand topic must leave `grip_meas`
stale without interrupting the arm stream, which gravity compensation depends
on. A single loop reading both topics would block the arms on a silent gripper.

The five steps in `_loop` are in this order and the order is the whole
point:

1. **`t_recv` first**, before anything is parsed out of the sample. Every
   consumer's notion of "when this measurement happened" is this number, and
   any work done before it is a constant bias nothing reports.
2. `q[29]`, `dq[29]` as measured.
3. FK the 14 arm joints **with the measured waist**. It is computed here rather
   than in P5 or P6 so that the flange pose carries exactly the `t_recv` of the
   joints that produced it, and so the two consumers cannot disagree about it.
   The measured waist must go into the same model slots P6's IK pins the
   *commanded* waist into, which is why both call `g1d_deploy.robot.kinematics`
   rather than each placing the model themselves.
4. Copy the latched grip value **and its arrival time**. The time is what makes
   staleness checkable at all: there is no flag, and a frozen gripper reading is
   indistinguishable from a still one without it.
5. One seqlock write, so a reader sees the whole sample or retries.

Step 2 is the one departure, and it is forced: cyclonedds deserialises the
whole sample inside its blocking take, so by the time Python can run at all the
message is already parsed. `t_recv` is stamped the instant `read()` returns —
the earliest instant this process can observe — and everything ordered
*after* the stamp (the FK, the latch copy, the write) genuinely happens after
it.

There is no staleness timeout, no reconnect and no watchdog anywhere in this
file. A dead topic is invisible by design.
"""

from __future__ import annotations

import threading

import numpy as np

from g1d_deploy.runtime import cpu_pinning as affinity
from g1d_deploy.runtime.shared_memory import Arena, build, open_all, schema
from g1d_deploy.runtime.timing import now
from g1d_deploy.config_loading.resolved import DeployConfig, Hardware
from g1d_deploy.config_loading.derive import derive
from g1d_deploy.runtime.drivers import robot_state as wire


class GripLatch:
    """The second subscription: latch the value and its arrival, and nothing else.

    The sample is stored as one immutable tuple, which is what lets the loop
    read it without a lock: `(value, arrival)` is taken in a single attribute
    load, so no state sample can ever pair a new grip value with the previous
    reading's timestamp. Under CPython that load is atomic; a two-field latch
    would need a lock P1 cannot afford to contend on at 1042 Hz.

    Before the first sample the latch reads `(zeros, 0.0)`. A `t_grip_recv` of
    zero is therefore "never received", distinguishable from any real arrival on
    a monotonic clock.
    """

    def __init__(self, source):
        self.source = source
        self.latest: tuple[np.ndarray, float] = (np.zeros(schema.GRIP_DIM), 0.0)
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._pump, name="p1-grip",
                                        daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        rec = wire.record(self.source.dtype)
        while True:
            ok = self.source.read(rec)
            t = now()                      # stamped on return, as in the loop
            if not ok:
                return                     # EOF: the latch simply stops moving
            self.latest = (rec["grip"].copy(), t)


class ColumnLatch:
    """The third subscription: the lift column's height, latched.

    `GripLatch` with a scalar. Same shape for the same reason — the height is on
    its own topic at ~30 Hz against this loop's 1070, so it must never be read
    inline: a silent or slow `rt/hispeed_state` has to leave the height stale,
    exactly as a silent gripper leaves `grip_meas` stale, and must never stall
    the stream gravity compensation depends on.

    One immutable `(height, arrival)` tuple, taken in a single attribute load,
    so no state sample can pair a new height with the previous reading's stamp.

    A `source` of None is a latch that never moves — which is what a run without
    the column subscriber gets, and it records as "never received" rather than
    as a robot standing at exactly zero.
    """

    def __init__(self, source):
        self.source = source
        self.latest: tuple[float, float] = (0.0, 0.0)
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.source is None:
            return
        self._thread = threading.Thread(target=self._pump, name="p1-column",
                                        daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        while True:
            h = self.source.read()
            t = now()                      # stamped on return, as in the loop
            if h is None:
                return                     # EOF: the latch simply stops moving
            self.latest = (float(h), t)


def _loop(control, ring, source, latch, fk, arm_idx, waist_idx,
          column=None) -> None:
    rec = wire.record(source.dtype)
    q, dq = rec["q"], rec["dq"]
    while control.session < schema.STOPPED:
        if not source.read(rec):
            return
        t_recv = now()                                        # (1)
        flange = fk.flange(q[arm_idx], q[waist_idx])          # (3)
        grip_meas, t_grip_recv = latch.latest                 # (4)
        column_h, t_column_recv = (0.0, 0.0) if column is None else column.latest
        ring.write(t_recv=t_recv, q=q, dq=dq, flange=flange,  # (5)
                   grip_meas=grip_meas, t_grip_recv=t_grip_recv,
                   column_h=column_h, t_column_recv=t_column_recv,
                   tick=rec["tick"])


def run(shm_name: str, cfg: DeployConfig, hw: Hardware, *,
        lowstate=None, handstate=None, fk=None, columnstate=None) -> None:
    """Entrypoint. The sources are injectable so the loop can be driven from
    `drivers/robot_state.py`'s fakes with no robot; a real run constructs them here,
    inside the spawned process, because a forked DDS context is unusable."""
    layout = build(cfg, derive(cfg, hw))
    arena = Arena.attach(shm_name, layout)
    r = open_all(arena, cfg)
    affinity.pin(hw, "p1_state")

    source = lowstate if lowstate is not None else wire.DdsLowState(hw)
    hand = handstate if handstate is not None else wire.DdsHandState(hw)
    column = columnstate
    if fk is None:
        # Imported here rather than at module scope: it pulls in mujoco and the
        # G1 model, which nothing else in this process needs and which a test
        # driving the loop off a fake source should not have to load.
        from g1d_deploy.robot.kinematics import ForwardKinematics
        fk = ForwardKinematics(hw.urdf)

    # Indices, not slices: the arm block is 14 joints in `[left, right]` order
    # and the waist is 3, both named by the robot file rather than assumed.
    arm_idx = np.asarray(hw.joints_arm, dtype=np.intp)
    waist_idx = np.asarray(hw.joints_waist, dtype=np.intp)

    try:
        source.connect()
        hand.connect()
        latch = GripLatch(hand)
        latch.start()
        # The column is OPTIONAL and off unless a source is supplied. Nothing in
        # the control path reads it yet, so a run without it is the run this
        # process has always made — it simply records the height as never
        # received.
        col = ColumnLatch(column)
        col.start()
        _loop(r["control"], r["state"], source, latch, fk, arm_idx, waist_idx,
              column=col)
    finally:
        source.close()
        hand.close()
        if column is not None:
            column.close()
        arena.close()
