"""Typed accessors over the arena regions.

This module owns every ring scan in the system, so the process loops contain scheduling logic
and nothing else. If P6 and the bus process each wrote their own "find the slot for tick k"
they would each get the wrap-around wrong differently.

**Column scans are deliberately unlocked.** A scan for "the slot with the smallest future
deadline" reads the `k` and `deadline` columns without the seqlock: both are naturally aligned
8-byte fields, so a value is either the old one or the new one, never a mix. The *chosen* slot
is then read back under its seqlock and re-checked before its payload is used. That keeps a
250 Hz scan to a few vector loads instead of a locked read per slot.

Each class's docstring names its one writer. The g1d version (chunks, cameras, two hands) is
`legacy_g1d/src/elrobot_deploy/runtime/shared_memory/rings.py`.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from elrobot_deploy.runtime.shared_memory import schema
from elrobot_deploy.runtime.shared_memory.primitives import Seqlock, fence


class Control:
    """The atomic words. One writing process each, except `request` (see there)."""

    def __init__(self, arena):
        self._c = arena.view("control")

    @property
    def session(self) -> int:
        return int(self._c[0]["session"])

    @session.setter
    def session(self, value: int) -> None:          # P0 only
        self._c[0]["session"] = value

    @property
    def k_exec(self) -> int:
        """The tick the bus process is walking toward. The bus process only.

        Never rewound: the bus process owns the serial port for the whole session (restarting
        it would mean releasing torque), so ticks keep counting across rollouts and every
        producer anchors on `k_exec + 2` rather than on zero.
        """
        return int(self._c[0]["k_exec"])

    @k_exec.setter
    def k_exec(self, value: int) -> None:
        self._c[0]["k_exec"] = value

    @property
    def request(self) -> int:
        """The pending request, one of `schema.REQ_*`.

        **Two writers, on purpose: the one place with two.** A requester stores, P0 clears to
        `REQ_NONE` once it has taken it. If a store lands in the instant between P0's read and
        its clear, that request is lost: a button that did nothing, visibly, pressed again. It
        can never walk `session` backwards, which is what the one-writer rule protects.
        """
        return int(self._c[0]["request"])

    @request.setter
    def request(self, value: int) -> None:          # a requester stores, P0 clears
        self._c[0]["request"] = value

    @property
    def intervene(self) -> int:
        """The intervention counter (`schema.CONTROL`). The phone source only."""
        return int(self._c[0]["intervene"])

    @intervene.setter
    def intervene(self, value: int) -> None:
        self._c[0]["intervene"] = value

    @property
    def bus(self) -> int:
        """`schema.BUS_*`. The bus process only."""
        return int(self._c[0]["bus"])

    @bus.setter
    def bus(self, value: int) -> None:
        self._c[0]["bus"] = value

    @property
    def wait(self) -> int:
        """`schema.WAIT_*`: what P0 is waiting for. P0 only."""
        return int(self._c[0]["wait"])

    @wait.setter
    def wait(self, value: int) -> None:
        self._c[0]["wait"] = value

    @property
    def release_torque(self) -> bool:
        """May the bus process release torque at STOPPED? P0 only."""
        return bool(self._c[0]["release_torque"])

    @release_torque.setter
    def release_torque(self, value: bool) -> None:
        self._c[0]["release_torque"] = int(bool(value))

    # --- multi-word text blocks, each under its own counter ------------------------------
    #
    # Odd while the one writer is mid-write, even when consistent; a reader retries until it
    # copies the block between two equal even reads (`primitives.Seqlock`'s rule, inline).

    def _write_block(self, seq: str, **fields) -> int:
        c = self._c
        c[seq][0] += 1                               # -> odd: writing
        fence()
        try:
            for name, value in fields.items():
                c[name][0] = value
        finally:
            fence()
            c[seq][0] += 1                           # -> even: consistent
        return int(c[seq][0])

    def _read_block(self, seq: str, *names: str) -> tuple[int, list]:
        c = self._c
        while True:
            s0 = int(c[seq][0])
            if s0 & 1:
                continue
            fence()
            out = [np.copy(c[name][0]) for name in names]
            fence()
            if int(c[seq][0]) == s0:
                return s0, out

    @staticmethod
    def _text(buf, n) -> str:
        return bytes(np.asarray(buf, dtype=np.uint8)[:int(n)]).decode("utf-8", errors="replace")

    @staticmethod
    def _reason(text: str) -> tuple[np.ndarray, int]:
        """UTF-8 into the fixed box, cut at a character boundary if too long."""
        raw = str(text).encode("utf-8")[:schema.REASON_BYTES]
        raw = raw.decode("utf-8", errors="ignore").encode("utf-8")
        buf = np.zeros(schema.REASON_BYTES, dtype=np.uint8)
        buf[:len(raw)] = np.frombuffer(raw, dtype=np.uint8)
        return buf, len(raw)

    def fault(self) -> tuple[int, str]:
        """`(fault_seq, reason)` from the bus process; seq 0 = never faulted."""
        seq, (n, buf) = self._read_block("fault_seq", "fault_len", "fault")
        return seq, self._text(buf, n)

    def set_fault(self, reason: str) -> int:                  # the bus process only
        buf, n = self._reason(reason)
        return self._write_block("fault_seq", fault_len=n, fault=buf)

    def report(self) -> tuple[int, str]:
        """`(report_seq, json)`: the bus process's hardware check; seq 0 = none yet."""
        seq, (n, buf) = self._read_block("report_seq", "report_len", "report")
        return seq, self._text(buf, n)

    def set_report(self, text: str) -> int:                   # the bus process only
        raw = str(text).encode("utf-8")
        if len(raw) > schema.REPORT_BYTES:
            raise ValueError(f"report is {len(raw)} bytes, the arena holds {schema.REPORT_BYTES}")
        buf = np.zeros(schema.REPORT_BYTES, dtype=np.uint8)
        buf[:len(raw)] = np.frombuffer(raw, dtype=np.uint8)
        return self._write_block("report_seq", report_len=len(raw), report=buf)

    def refusal(self) -> tuple[int, str]:
        """`(refusal_seq, reason)`; seq 0 = nothing was ever refused."""
        seq, (n, buf) = self._read_block("refusal_seq", "refusal_len", "refusal")
        return seq, self._text(buf, n)

    def refuse(self, reason: str) -> int:                     # P0 only
        buf, n = self._reason(reason)
        return self._write_block("refusal_seq", refusal_len=n, refusal=buf)


# --- state ----------------------------------------------------------------------------------


class StateRing:
    """What the bus process measured. Written by the bus process only; anyone may read it.

    A ring rather than a latest-value slot so a reader can ask for the measurement nearest an
    instant (a camera frame's, later) or a window of them (a recorder), not only the newest.
    """

    def __init__(self, arena):
        self._rec = arena.view("state_rec")
        self._lock = Seqlock(arena.view("state_seq"), self._rec)
        self.n = len(self._rec)
        self._cursor = 0
        self._index = 0

    def write(self, *, t, q, q_cmd, t_slow=np.nan, speed=np.nan, load=np.nan, temp=np.nan,
              alarm=0) -> int:
        self._index += 1
        with self._lock.write(self._cursor) as slot:
            slot["index"] = self._index
            slot["t"] = t
            slot["q"] = q
            slot["q_cmd"] = q_cmd
            slot["t_slow"] = t_slow
            slot["speed"] = speed
            slot["load"] = load
            slot["temp"] = temp
            slot["alarm"] = alarm
        self._cursor = (self._cursor + 1) % self.n
        return self._index

    def latest(self) -> np.ndarray | None:
        idx = self._rec["index"]
        if not idx.any():
            return None
        return self._lock.snapshot(int(np.argmax(idx)), schema.STATE)

    def nearest(self, t: float) -> np.ndarray | None:
        """The row whose `t` is closest to `t`."""
        idx = self._rec["index"]
        valid = idx > 0
        if not valid.any():
            return None
        dist = np.where(valid, np.abs(self._rec["t"] - t), np.inf)
        return self._lock.snapshot(int(np.argmin(dist)), schema.STATE)

    def window(self, t0: float, t1: float) -> np.ndarray:
        """Every row with `t0 <= t <= t1`, ascending. Copy-and-verify, vectorised: a slot the
        writer touched during the copy is dropped, never returned torn."""
        seq = self._lock.seq
        s0 = seq.copy()
        fence()
        rec = self._rec.copy()
        fence()
        s1 = seq.copy()
        ok = ((s0 == s1) & (s0 % 2 == 0) & (rec["index"] > 0)
              & (rec["t"] >= t0) & (rec["t"] <= t1))
        out = rec[ok]
        return out[np.argsort(out["index"], kind="stable")]


# --- the operator's stream ------------------------------------------------------------------


class InterveneRow(NamedTuple):
    """One operator sample, as P6 consumes it."""

    index: int
    deadline: float
    delta: np.ndarray       # (7,) TCP motion since the press, base frame
    grip: float             # gripper command, rad


class InterveneRing:
    """Written by the phone source while the clutch is held; read by P6.

    `meta` is the one record beside it carrying what is true of the phone rather than of a row
    (`schema.INTERVENE_META`).
    """

    def __init__(self, arena):
        self._rec = arena.view("intervene_rec")
        self._lock = Seqlock(arena.view("intervene_seq"), self._rec)
        self._meta = arena.view("intervene_meta")
        self.n = len(self._rec)
        self._cursor = 0
        self._index = int(self._meta[0]["newest"])

    # --- the phone source --------------------------------------------------------------

    def write(self, **fields) -> int:
        """One row. `index` is this ring's own and is never passed in."""
        self._index += 1
        i = self._cursor
        with self._lock.write(i) as slot:
            slot["index"] = self._index
            for name, value in fields.items():
                slot[name] = value
        self._cursor = (i + 1) % self.n
        # AFTER the row: it is what a reader takes as permission to look.
        fence()
        self._meta[0]["newest"] = self._index
        return self._index

    def note(self, **fields) -> None:
        """Update the header: independent aligned words, no seqlock."""
        for name, value in fields.items():
            self._meta[0][name] = value

    def meta(self) -> np.ndarray:
        """A copy of the header (its words may be read at different instants)."""
        return self._meta[0].copy()

    # --- P6 ----------------------------------------------------------------------------

    def newest(self, intervene: int, since: int = 0) -> InterveneRow | None:
        """The most recent row of THIS intervention newer than `since`, or None.

        `since` is the caller's last row: when nothing new arrived (most polls) the answer
        costs one scalar read. Deliberately the newest and not the oldest unsolved: every row
        is relative to the same press, so a skipped row leaves nothing to make up, and a slow
        solve costs one dropped target rather than a backlog felt as lag.
        """
        if int(self._meta[0]["newest"]) <= since:
            return None
        idx = np.where(self._rec["intervene"] == intervene, self._rec["index"], 0)
        if not idx.any():
            return None
        snap = self._lock.snapshot(int(np.argmax(idx)), schema.INTERVENE)
        if int(snap["intervene"]) != intervene or int(snap["index"]) <= since:
            return None                 # wrapped under us; try again next poll
        return InterveneRow(int(snap["index"]), float(snap["deadline"]),
                            snap["delta"].copy(), float(snap["grip"]))


# --- targets --------------------------------------------------------------------------------


class TargetEntry(NamedTuple):
    k: int
    deadline: float
    payload: np.ndarray     # (8,) joints


class TargetRing:
    """The target ring. Written by P6 during ACTIVE and by P0 during HOME, a reset and
    EXITING; read by the bus process (`next_after`, `line_after`) and P6 (`latest_below`).

    One writer at a time, not two: P0 writes only while P6 does not exist, and a reset stops
    P6 before P0 writes its ramp.
    """

    DTYPE = schema.TARGET
    #: Stored for a row no IK produced (P0's ramps). Built once: no allocation inside the
    #: seqlock's critical section.
    _NO_POSE = np.full(schema.POSE_DIM, np.nan)

    def __init__(self, arena):
        self._rec = arena.view("target_rec")
        self._seq = arena.view("target_seq")
        self._lock = Seqlock(self._seq, self._rec)
        self.n = len(self._rec)

    def _entry(self, snap: np.ndarray) -> TargetEntry:
        return TargetEntry(int(snap["k"]), float(snap["deadline"]), snap["q"].copy())

    def _written(self) -> np.ndarray:
        """Which slots have ever been written. The arena is zero-filled, so `seq == 0` is
        exactly "never written"; it matters for tick 0, a real tick (the end of homing) that a
        zeroed slot would otherwise imitate."""
        return (self._seq != 0) & (self._rec["k"] != schema.K_UNSET)

    # --- the writer ----------------------------------------------------------------------

    def write(self, k: int, deadline: float, q, *, intervene: int, err: float = np.nan,
              pose=None) -> None:
        """One tick. `intervene` is required and keyword-only: every writer must say which
        world the row belongs to. P0's ramps after a rollout in which somebody drove run on an
        EVEN but non-zero counter, and a row stamped with a default 0 would never execute."""
        with self._lock.write(k % self.n) as slot:
            slot["k"] = k
            slot["deadline"] = deadline
            slot["intervene"] = intervene
            slot["q"] = q
            slot["err"] = err
            slot["pose"] = self._NO_POSE if pose is None else pose

    # --- the readers ---------------------------------------------------------------------

    def get(self, k: int) -> TargetEntry | None:
        snap = self.full(k)
        return None if snap is None else self._entry(snap)

    def full(self, k: int):
        """The whole record of tick `k` in one snapshot, or None."""
        i = k % self.n
        if self._seq[i] == 0:
            return None
        snap = self._lock.snapshot(i, self.DTYPE)
        return None if int(snap["k"]) != k else snap

    def latest_below_full(self, k: int):
        """The whole record of the written target with the largest `k` below this one.

        **Not filtered on `intervene`**, unlike `next_after`. Selection ("may this be
        executed?") is about authority and must not cross a takeover; seeding ("what
        configuration is the arm in?") is about geometry and must, or the IK's null space jumps
        at the seam.
        """
        ks = self._rec["k"]
        valid = (ks < k) & (ks > k - self.n) & self._written()
        if not valid.any():
            return None
        best = int(np.argmax(np.where(valid, ks, np.iinfo(np.int64).min)))
        snap = self._lock.snapshot(best, self.DTYPE)
        if int(snap["k"]) == schema.K_UNSET:
            return None
        return snap

    def latest_below(self, k: int) -> TargetEntry | None:
        snap = self.latest_below_full(k)
        return None if snap is None else self._entry(snap)

    def newest_full(self):
        """The whole record of the written target with the largest `k`, or None."""
        ks = self._rec["k"]
        valid = self._written()
        if not valid.any():
            return None
        best = int(np.argmax(np.where(valid, ks, np.iinfo(np.int64).min)))
        snap = self._lock.snapshot(best, self.DTYPE)
        return None if int(snap["k"]) == schema.K_UNSET else snap

    def line_after(self, t: float, until: float, intervene: int) -> list[TargetEntry]:
        """The targets of world `intervene` with deadline > `t`, in deadline order, up to the
        first whose deadline reaches `until`. The filter's look-ahead (`line_ahead`)."""
        dl = self._rec["deadline"]
        valid = self._written() & (dl > t) & (self._rec["intervene"] == intervene)
        idx = np.flatnonzero(valid)
        out = []
        for i in idx[np.argsort(dl[idx], kind="stable")]:
            snap = self._lock.snapshot(int(i), self.DTYPE)
            if (float(snap["deadline"]) <= t or int(snap["k"]) == schema.K_UNSET
                    or int(snap["intervene"]) != intervene):
                continue
            out.append(self._entry(snap))
            if out[-1].deadline >= until:
                break
        return out

    def next_after(self, t: float, intervene: int) -> TargetEntry | None:
        """The target of world `intervene` with the SMALLEST deadline strictly after `t`.

        The bus process's segment advance. Stale slots have deadlines in the past and are
        excluded by the comparison. Filtering on `intervene` is what makes a takeover safe
        without a wipe: a row of the previous world can never be selected, so a pause in the
        operator's stream holds the arm instead of letting an older, far-future row win.
        """
        dl = self._rec["deadline"]
        valid = self._written() & (dl > t) & (self._rec["intervene"] == intervene)
        if not valid.any():
            return None
        best = int(np.argmin(np.where(valid, dl, np.inf)))
        snap = self._lock.snapshot(best, self.DTYPE)
        # Re-checked on the coherent snapshot: the scan above was unlocked.
        if (float(snap["deadline"]) <= t or int(snap["k"]) == schema.K_UNSET
                or int(snap["intervene"]) != intervene):
            return None
        return self._entry(snap)


def open_all(arena) -> dict:
    """Every accessor a process might want, built once at attach."""
    return {
        "control": Control(arena),
        "state": StateRing(arena),
        "targets": TargetRing(arena),
        "intervene": InterveneRing(arena),
    }


__all__ = ["Control", "InterveneRing", "InterveneRow", "StateRing", "TargetEntry",
           "TargetRing", "open_all"]
