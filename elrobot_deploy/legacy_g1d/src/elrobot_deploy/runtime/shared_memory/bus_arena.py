"""A small arena for the ElRobot executor: control words, an 8-joint target ring, a bus state ring.

**Temporary, until the single-arm schema (PORTING_PLAN.md workstream 1).** `schema.TARGET` is
still g1d's 17-joint body row, so the full arena cannot carry an 8-joint target yet. This
module builds the three regions the executor needs, under the same rules:

  * `control`  is `schema.CONTROL`, read through `rings.Control` (`intervene`, `k_exec`,
    `session`), exactly as P7 reads it;
  * `target_*` is the target ring, read through `rings.TargetRing`'s own `next_after` /
    `line_after` (only the record changes: 8 joints, no hand/pose/waist columns);
  * `bus_state_*` is what the bus process measured, one row per position read.

When workstream 1 makes `schema.TARGET` 8 wide, `BusTargetRing` and this layout go away and
the executor reads `open_all(...)["body"]` instead; `BusServo` does not change.
"""

from __future__ import annotations

import hashlib

import numpy as np

from elrobot_deploy.robot.elrobot_joints import N_JOINTS
from elrobot_deploy.runtime.shared_memory import schema
from elrobot_deploy.runtime.shared_memory.layout import ALIGN, Layout, Region
from elrobot_deploy.runtime.shared_memory.primitives import Seqlock
from elrobot_deploy.runtime.shared_memory.rings import Control, TargetRing

#: One tick of the target line: what the IK (or the sine test) wants the joints to be at
#: `deadline`. The first three fields are `schema.TARGET`'s, in the same order, because they
#: are what `TargetRing`'s unlocked column scans read.
BUS_TARGET = np.dtype([
    ("k", "<i8"),
    ("deadline", "<f8"),
    ("intervene", "<u8"),
    ("q", "<f8", (N_JOINTS,)),          # rad, servo order; [7] is the gripper (0 = open)
    ("err", "<f8"),                     # the IK's residual, NaN when no IK wrote the row
])

#: One position read by the bus process. The slow fields are refreshed every few ticks and
#: carry the time of their own read, so a reader never mistakes an old load for a new one.
BUS_STATE = np.dtype([
    ("index", "<u8"),                   # 0 = never written
    ("t", "<f8"),                       # monotonic, middle of the position read
    ("q", "<f8", (N_JOINTS,)),          # measured, rad
    ("q_cmd", "<f8", (N_JOINTS,)),      # the goal last written before this read, rad
    ("t_slow", "<f8"),                  # when speed / load / temperature were read
    ("speed", "<f8", (N_JOINTS,)),      # rad/s
    ("load", "<f8", (N_JOINTS,)),       # fraction of stall torque, signed
    ("temp", "<f8", (N_JOINTS,)),       # deg C
    ("alarm", "u1", (N_JOINTS,)),       # each servo's status byte from its latest reply
])

N_TARGET = 64        # ~1 s at 60 Hz; the executor only ever looks a lead (~40 ms) ahead
N_STATE = 1024       # ~4 s at 250 Hz


def bus_layout(n_target: int = N_TARGET, n_state: int = N_STATE) -> Layout:
    regions: dict[str, Region] = {}
    offset = 0

    def add(name, dtype, count):
        nonlocal offset
        offset = (offset + ALIGN - 1) // ALIGN * ALIGN
        regions[name] = Region(name, np.dtype(dtype), count, offset)
        offset += np.dtype(dtype).itemsize * count

    add("header", schema.HEADER, 1)
    add("control", schema.CONTROL, 1)
    add("target_seq", np.uint64, n_target)
    add("target_rec", BUS_TARGET, n_target)
    add("bus_state_seq", np.uint64, n_state)
    add("bus_state_rec", BUS_STATE, n_state)

    digest = hashlib.blake2b(digest_size=8)
    for r in regions.values():
        digest.update(f"{r.name}|{r.dtype.str}|{r.dtype.itemsize}|{r.count}|{r.offset}".encode())
    return Layout(regions=regions, size=offset,
                  hash=int.from_bytes(digest.digest(), "little"), cameras=())


class BusTargetRing(TargetRing):
    """`TargetRing` over `BUS_TARGET`: the same readers, a writer for the 8-joint row."""

    def __init__(self, arena, column: str = "q"):
        self.DTYPE = arena.view("target_rec").dtype
        super().__init__(arena, column)

    def _check(self, column: str) -> str:
        if column not in self.DTYPE.names:
            raise ValueError(f"{column!r} is not a column of BUS_TARGET ({self.DTYPE.names})")
        return column

    def write(self, k: int, deadline: float, payload, *, intervene: int,
              err: float = np.nan) -> None:
        with self._lock.write(k % self.n) as slot:
            slot["k"] = k
            slot["deadline"] = deadline
            slot["intervene"] = intervene
            slot["q"] = payload
            slot["err"] = err


class BusStateRing:
    """Written by the bus process only; anything may read it."""

    def __init__(self, arena):
        self._rec = arena.view("bus_state_rec")
        self._lock = Seqlock(arena.view("bus_state_seq"), self._rec)
        self.n = len(self._rec)
        self._cursor = 0
        self._index = 0

    def write(self, *, t, q, q_cmd, t_slow, speed, load, temp, alarm) -> int:
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
        return self._lock.snapshot(int(np.argmax(idx)), BUS_STATE)


def open_bus_rings(arena) -> dict:
    return {"control": Control(arena), "targets": BusTargetRing(arena),
            "state": BusStateRing(arena)}


__all__ = [
    "BUS_STATE",
    "BUS_TARGET",
    "N_STATE",
    "N_TARGET",
    "BusStateRing",
    "BusTargetRing",
    "bus_layout",
    "open_bus_rings",
]
