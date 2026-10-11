"""The one clock, and the two ways of waiting on it.

`time.monotonic()` is process-independent on Linux and macOS, which is what lets
a deadline written by P6 mean the same thing to P7 in another process.
Nothing in the system uses wall time. Imported by every process; the clocks
are described in `docs/infra_logic/arena_layout_and_reads.md`, "Ticks and
clocks".

Both waits below are hybrid: sleep coarsely until close, then spin. A bare
`time.sleep` overshoots by however long the OS feels like — typically 1-3 ms,
which is most of a servo period and all of P5's scheduling budget.
"""

from __future__ import annotations

import time

now = time.monotonic

# The last stretch before the target is spun, not slept: below this, no sleep is
# accurate enough to be worth the risk. The loops that use this own their cores
# #, so the cost is CPU nobody else wanted.
SPIN_S = 1.5e-3

# Never sleep more than this fraction of the time remaining.
#
# This is the whole trick, and it rests on one measured property: `time.sleep`
# overshoots by a *proportion* of the requested duration, not by a fixed cost.
# Measured here, idle: 27% of a 0.1 ms sleep, 26% of 1 ms, 21% of 10 ms, 15% of
# 33 ms. So sleeping half the remaining slack lands at ~63% of it in the worst
# case — still short of the target, always — and repeating that converges on the
# deadline from below instead of gambling it on one call.
#
# The rule this replaces slept the whole slack in one go and spun the last
# 1.5 ms. That staked the deadline on a single `time.sleep` returning within
# 1.5 ms of its request, which it does not: a 33 ms sleep overshoots by ~5 ms,
# so P5 woke a measured 2.9 ms late EVERY tick — 9% of an action tick, silently.
SLEEP_FRACTION = 0.5


def sleep_until(t_target: float, *, spin_s: float = SPIN_S) -> float:
    """Block until `monotonic() >= t_target`. Returns the actual wake time.

    Waking late is the failure this exists to prevent, so the loop re-reads the
    clock after every sleep and never sleeps into the spin window. What it can
    promise is that it never *chooses* to be late: it does not return early, it
    does not sleep past the target, and the only way out is the clock actually
    reaching it. What no userspace code can promise is that the kernel schedules
    this thread the instant it becomes runnable — hence the core
    reservation and the optional `SCHED_FIFO` (`cpu_pinning.py`).

    The returned value is the honest one: callers that care about scheduling
    error (P5) can compare it against what they asked for.
    """
    while True:
        remaining = t_target - now()
        if remaining <= 0:
            return now()
        if remaining > spin_s:
            time.sleep(min(remaining - spin_s, remaining * SLEEP_FRACTION))
        # Inside the spin window: fall through and re-read the clock. A busy
        # loop is the point — it is the only wait with no scheduler in it.


class Pacer:
    """Fixed-rate loop pacing that does not accumulate drift.

    The next wake is computed from the phase the loop *should* be on, not from
    when this iteration happened to finish, so a single long iteration does not
    permanently shift the loop's phase. After an overrun long enough to have
    missed whole periods, the phase is re-anchored to now rather than firing a
    burst of catch-up iterations — a 500 Hz loop that fell 20 ms behind wants to
    be on time from here, not to publish ten commands as fast as it can.
    """

    def __init__(self, hz: float):
        self.period = 1.0 / float(hz)
        self._next = now() + self.period

    def wait(self) -> float:
        """Sleep until the next tick boundary. Returns the wake time."""
        t = sleep_until(self._next)
        self._next += self.period
        if self._next <= t:
            self._next = t + self.period
        return t
