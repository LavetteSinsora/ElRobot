"""The wait, which is the one piece of timing every loop depends on.

`sleep_until` is what P5 uses to hit its trigger instant and what
paces P7 and P8. Waking late there is not a performance issue — it is the
schedule being wrong, silently, on every tick.

These tests measure real time, so they are written to be decisive about the
failure they exist to catch rather than tight against the machine's noise. The
implementation they replaced woke a median **2739 us** late at the 33 ms period;
the bounds below are orders of magnitude looser than that and still catch it.
"""

from __future__ import annotations

import statistics
import time

import pytest

from elrobot_deploy.runtime import timing as clock


def _lateness(period: float, n: int) -> list[float]:
    """Microseconds late, over `n` consecutive deadlines `period` apart."""
    t = clock.now() + 0.01
    out = []
    for _ in range(n):
        t += period
        out.append((clock.sleep_until(t) - t) * 1e6)
    return out


def test_never_returns_before_the_target():
    """The hard invariant. Everything else here is a quality bound; this one is
    correctness — an early return means a loop acts on a deadline that has not
    arrived."""
    for _ in range(200):
        target = clock.now() + 0.0005
        assert clock.sleep_until(target) >= target


def test_a_past_target_returns_immediately():
    t0 = clock.now()
    assert clock.sleep_until(t0 - 1.0) >= t0
    assert clock.now() - t0 < 0.01


def test_the_long_wait_does_not_overshoot():
    """The regression that motivated this. One `time.sleep` for the whole slack
    overshoots by ~15% of its duration — 5 ms on a 33 ms sleep — so the old
    rule, which spun only the last 1.5 ms, woke late every single tick."""
    late = _lateness(33.3e-3, 40)
    median = statistics.median(late)
    assert median < 500, f"median {median:.0f}us late — the whole-slack sleep is back"
    assert max(late) < 5000, f"max {max(late):.0f}us late"


def test_the_servo_wait_does_not_overshoot():
    late = _lateness(2e-3, 200)
    assert statistics.median(late) < 200
    assert max(late) < 5000


def test_sleep_never_reaches_into_the_spin_window():
    """The mechanism, asserted directly rather than through its symptom: no
    individual sleep may be long enough to land past the target, which is what
    makes the convergence safe for any overshoot ratio below 100%."""
    slept = []
    real_sleep = time.sleep

    def spy(d):
        slept.append(d)
        real_sleep(d)

    time.sleep = spy
    try:
        target = clock.now() + 0.05
        clock.sleep_until(target, spin_s=clock.SPIN_S)
    finally:
        time.sleep = real_sleep

    assert slept, "a 50 ms wait should sleep, not spin the whole way"
    # Each sleep is at most half the slack that remained when it was issued, so
    # even a 100% overshoot could not cross the target.
    assert all(d <= 0.05 * clock.SLEEP_FRACTION + 1e-9 for d in slept)
    assert len(slept) > 1, "one long sleep is exactly the bug this replaced"


def test_pacer_does_not_accumulate_drift():
    """Phase comes from the schedule, not from when the last iteration
    happened to finish, so one slow iteration does not shift the loop."""
    pacer = clock.Pacer(500.0)
    t0 = clock.now()
    for i in range(50):
        if i == 10:
            time.sleep(0.004)              # a deliberately overrunning iteration
        pacer.wait()
    elapsed = clock.now() - t0
    # 50 ticks at 500 Hz is 100 ms; the overrun is absorbed, not accumulated.
    assert elapsed == pytest.approx(0.100, abs=0.020)


def test_pacer_reanchors_after_a_long_overrun():
    """After missing whole periods the loop wants to be on time from here, not
    to fire a burst of catch-up iterations."""
    pacer = clock.Pacer(500.0)
    time.sleep(0.02)                       # ten periods late
    t0 = clock.now()
    pacer.wait()
    pacer.wait()
    assert clock.now() - t0 >= 0.002       # it waited, rather than firing instantly
