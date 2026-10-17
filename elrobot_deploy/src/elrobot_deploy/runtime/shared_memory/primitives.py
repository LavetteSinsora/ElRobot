"""The seqlock: how one writer and any number of readers share a record.
Used by every ring in `rings.py`.

Chosen so that **no reader can ever delay a writer**. The bus process has 4 ms per tick and
cannot block on a lock held by a preempted reader, which is what disqualifies a mutex here.

Memory ordering. On x86 the stores of one core are seen by others in program order, and every
numpy operation is an opaque C call (a compiler barrier), so the counter discipline alone is
sound. **On ARM (Apple Silicon) it is not**: a weakly ordered core may make the payload visible
before the odd counter, or the reader may load the payload before the counter it checks. So
`fence()` sits between the counter and the payload on both sides. It is a real hardware barrier
where one is needed (`OSMemoryBarrier` on macOS, about a microsecond through ctypes) and a no-op
on x86, where the hardware already orders these accesses.
"""

from __future__ import annotations

import platform
import sys
from contextlib import contextmanager

import numpy as np


def _make_fence():
    """A full memory barrier callable from Python, or a no-op where none is needed."""
    if platform.machine().lower() in ("x86_64", "amd64", "i386", "i686"):
        return lambda: None
    if sys.platform == "darwin":
        import ctypes
        try:
            barrier = ctypes.CDLL("/usr/lib/libSystem.B.dylib").OSMemoryBarrier
            barrier.restype = None
            barrier.argtypes = []
            return barrier
        except (OSError, AttributeError):
            pass
    # Elsewhere (Linux on ARM, not a target today): a lock round-trip. Its acquire and release
    # stop most reorderings across it but are not formally a full barrier; a port to such a
    # box should call a real one (e.g. through a tiny C extension) here.
    import threading
    lock = threading.Lock()

    def fence():
        lock.acquire()
        lock.release()
    return fence


#: Full memory barrier: no load or store moves across it, on this core, as seen by others.
fence = _make_fence()


class Seqlock:
    """An odd/even counter per slot, in its own `uint64` array.

    The counter is deliberately *not* a field inside the payload record: a
    counter sharing a dtype with a payload is a counter that can be a float, and
    a float counter freezes past its mantissa limit with every subsequent torn
    read undetected.

    The writer is wait-free: two increments and no branches. The reader retries.
    """

    __slots__ = ("rec", "seq")

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
        fence()                               # ... visible before any of the payload
        try:
            yield self.rec[i]
        finally:
            fence()                           # the payload visible before ...
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
            fence()                           # no payload load before the counter's
            out[()] = rec[i]
            fence()                           # every payload load before the re-check
            if int(seq[i]) == s0:
                return

    def snapshot(self, i: int, dtype) -> np.ndarray:
        out = np.empty((), dtype=dtype)
        self.read_into(i, out)
        return out

