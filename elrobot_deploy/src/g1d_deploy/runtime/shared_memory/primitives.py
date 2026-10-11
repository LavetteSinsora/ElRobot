"""The seqlock: how one writer and any number of readers share a record.
Used by every ring in `rings.py`; the reader's side of the rule is in
`docs/infra_logic/arena_layout_and_reads.md`, "Read rules: why a read can come
back empty".

Chosen so that **no reader can ever delay a writer**. P7 has 2 ms and cannot
block on a lock held by a preempted reader, which is what disqualifies a mutex
everywhere in this system.

The camera rings (`rings.CameraRing`) follow the same odd/even counter
discipline without this class: their write window spans a blocking decode, so
it cannot be a `with` block.

Memory ordering rests on x86 store ordering plus the fact that every numpy
operation is an opaque C call, which acts as a compiler barrier. This is **not
sound on ARM** — a port needs explicit barriers.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np


class Seqlock:
    """An odd/even counter per slot, in its own `uint64` array.

    The counter is deliberately *not* a field inside the payload record: a
    counter sharing a dtype with a payload is a counter that can be a float, and
    a float counter freezes past its mantissa limit with every subsequent torn
    read undetected.

    The writer is wait-free: two increments and no branches. The reader retries.
    """

    __slots__ = ("seq", "rec")

    def __init__(self, seq: np.ndarray, rec: np.ndarray):
        if seq.dtype != np.uint64:
            raise TypeError(f"seqlock counter must be uint64, got {seq.dtype}")
        if len(seq) != len(rec):
            raise ValueError(f"seq/rec length mismatch: {len(seq)} vs {len(rec)}")
        self.seq = seq
        self.rec = rec

    @contextmanager
    def write(self, i: int):
        """Yield slot `i` for mutation, odd during, even after."""
        self.seq[i] += 1                      # -> odd: a write is in progress
        try:
            yield self.rec[i]
        finally:
            self.seq[i] += 1                  # -> even: consistent again

    def read_into(self, i: int, out: np.ndarray) -> None:
        """Copy slot `i` into `out`, retrying until it is internally consistent.

        Unbounded by design. A writer that died mid-write leaves the counter odd
        forever and this spins — which is exactly the invisible-dead-process
        failure the design accepts, surfacing here as a hang rather than
        as corrupt joint targets.
        """
        rec, seq = self.rec, self.seq
        while True:
            # int() because numpy refuses to coerce a Python int against uint64
            # for the parity test; the counter itself stays uint64 in memory.
            s0 = int(seq[i])
            if s0 & 1:
                continue
            out[()] = rec[i]
            if int(seq[i]) == s0:
                return

    def snapshot(self, i: int, dtype) -> np.ndarray:
        out = np.empty((), dtype=dtype)
        self.read_into(i, out)
        return out

