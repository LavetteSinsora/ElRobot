"""`dashboards/inspect/clock.py`: state placed where it was taken.

The thing under test is a measurement, so most of these plant a known answer and
ask for it back: a production series, a counter rate, a receive-side hole, and
then whether the estimator recovers the instants the samples were actually taken
at. The one that matters most is `test_stall_does_not_move_the_pose` — it is the
`20260920-161456` failure in miniature, where a 300 ms hole read as 0.1 rad of
tracking error that the robot never had.
"""

from __future__ import annotations

import numpy as np
import pytest

from g1d_deploy.runtime.shared_memory import schema
from g1d_deploy.dashboards.inspect import clock, loader, timeline

from . import rollout_fixtures as fx


#: What the robot actually publishes at, measured on one recorded rollout: 1013 Hz
#: against a 1 kHz counter, which is why ~4% of its samples share a tick.
#:
#: Not 1000. At exactly 1000 the two clocks are synchronous and the counter's
#: rounding residual becomes a ramp with a period of `1/drift` — 30 s at 33 ppm,
#: longer than the record — so it stops being noise the bin minima average away
#: and the fit reads it as rate instead. That is a property of the fixture and
#: not of any robot, and it cost an hour of looking at the estimator for it.
_RATE_HZ = 1013.0


def _state(n=20000, *, drift_ppm=0.0, base_lat=0.0008, jitter=0.0,
           stall=None, t0=1000.0, rate_hz=_RATE_HZ):
    """A state stream with a known device clock. `stall` is `(start_s, span_s)`."""
    st = np.zeros(n, dtype=schema.STATE)
    st["index"] = np.arange(n)
    prod = t0 + np.arange(n) / rate_hz
    st["tick"] = np.floor((prod - t0) * 1000.0
                          / (1.0 + drift_ppm * 1e-6)).astype(np.uint32)
    lat = np.full(n, base_lat)
    if jitter:
        lat = lat + np.abs(np.sin(np.arange(n) * 0.7)) * jitter
    st["t_recv"] = prod + lat
    if stall is not None:
        a, span = stall
        held = (prod - t0 >= a) & (prod - t0 < a + span)
        st["t_recv"][held] = t0 + a + span + base_lat + 2e-5 * np.arange(held.sum())
    return st, prod


# --- The estimate itself ----------------------------------------------------

def test_recovers_production_instants():
    """The planted production series comes back, to well under a millisecond."""
    st, prod = _state(drift_ppm=33.0, jitter=0.002)
    c = clock.estimate(st)
    assert c.model == "affine"
    # The base latency is unobservable and is absorbed into the offset, so the
    # estimate is late by it. What must be right is the SHAPE: every sample's
    # error identical to within the 1 ms quantisation of `tick`.
    err = (c.t - prod) * 1000.0
    assert 0.0 <= err.mean() <= 1.5
    assert np.ptp(err) < 1.0


def test_recovers_the_counter_rate():
    """`drift_ppm` is read back off the fit, not assumed to be zero.

    Checked as TIME, not as ppm. `tick` is quantised to 1 ms, so the rate a 20 s
    fixture can pin down is limited by how much of a millisecond that leaves
    across its span — ~15 ppm here, and ~1 ppm on a 194 s rollout. What has to
    hold at any length is that the rate error costs well under a millisecond
    over the record, which is what the rate is FOR.
    """
    for planted in (-900.0, -50.0, 0.0, 33.0, 400.0):
        st, _ = _state(drift_ppm=planted, jitter=0.002)
        c = clock.estimate(st)
        assert c.model == "affine"
        span = st["t_recv"][-1] - st["t_recv"][0]
        assert abs(c.drift_ppm - planted) * 1e-6 * span < 1e-4


def test_latency_is_non_negative_and_reports_the_stall():
    """No sample arrives before it was taken, and the hole is what it was."""
    st, _ = _state(stall=(8.0, 0.3), jitter=0.002)
    c = clock.estimate(st)
    assert c.latency_ms.min() > -clock._NEG_TOL_MS
    assert c.latency_ms.max() == pytest.approx(300.0, abs=2.0)
    # The 300 ms hole holds `0.3 * _RATE_HZ` samples, and they all arrive at its
    # end, so their lateness ramps from 300 ms down to nothing: the ones taken
    # in its last 50 ms were barely late at all. A quarter-second's worth of
    # them clear 50 ms, not the whole hole's — and getting that wrong is how a
    # stall's size gets overstated.
    assert c.stalls_50ms == pytest.approx(0.25 * _RATE_HZ, rel=0.05)


def test_a_curved_envelope_falls_back_to_rolling():
    """A stream falling progressively behind is not a clock rate.

    `20260920-153043` is this: the minimum latency itself climbs 75 ms over
    115 s, and a straight line through that curve puts a quarter of the samples
    before they were produced. The affine model has to notice and stand down.
    """
    st, prod = _state(n=40000, jitter=0.001)
    # A backlog growing as sqrt(t) — decelerating, so no line fits it. Scaled to
    # the real record: 75 ms accumulated over 115 s, which is ~44 ms over the
    # 40 s here. The magnitude is the point; a steeper ramp would be a different
    # claim about the hardware than the one measurement supports.
    backlog = 0.075 * np.sqrt(np.arange(len(st)) / 1000.0 / 115.0)
    st["t_recv"] += backlog
    c = clock.estimate(st)
    assert c.model == "rolling"
    assert c.drift_ppm is None
    assert c.latency_ms.min() >= 0.0
    # And it SAYS the absolute offset is not to be trusted, because it is not:
    # a backlog that never drains is absorbed into the estimated production
    # time, so by the end of the record `t` sits ~40 ms after the truth. There
    # is no reference in the record to do better with.
    assert c.local_only is True
    assert np.abs(c.t - prod).max() == pytest.approx(backlog[-1], rel=0.2)

    # What the model does guarantee is LOCAL timing, and the separation is the
    # point. Both errors in the same units — seconds of timing error over the
    # span being compared: across the whole record the estimate is out by 44 ms,
    # while over any one second it is out by `rate * 1 s` = ~2 ms. Comparing a
    # pose against a command one tick away therefore costs 20x less than the
    # absolute offset suggests, which is why this model is worth having; and it
    # is a statement about the RESIDUAL ramp, not a claim that the ramp was
    # removed.
    window_s = 1.0
    step = int(window_s * 1000)
    rate = np.abs(np.diff(c.t[::step]) / np.diff(prod[::step]) - 1.0)
    # Past the first window: inside it the minimum is one-sided, so the estimate
    # there tracks the envelope's slope from the start of the record rather than
    # the local one. A real property of a centred window, and not what is being
    # bounded here.
    edge = int(clock._WIN_S)
    local_err = rate[edge:].max() * window_s
    assert local_err < 3e-3
    assert np.abs(c.t - prod).max() > 15 * local_err
    # And it decays with the ramp it is tracking, rather than accumulating.
    q = len(rate) // 4
    assert rate[-q:].mean() < 0.5 * rate[edge:edge + q].mean()


def test_time_is_non_decreasing_on_both_models():
    """A column that steps backwards would resolve frames to the wrong records."""
    for st in (_state(drift_ppm=-900.0, stall=(5.0, 0.4), jitter=0.003)[0],
               _state(n=40000, jitter=0.001)[0]):
        c = clock.estimate(st)
        assert (np.diff(c.t) >= 0.0).all()


def test_counter_wrap_does_not_teleport_the_tail():
    """`tick` is `<u4` ms, so it wraps every 49.7 days of robot uptime."""
    st, prod = _state(n=5000, jitter=0.001)
    st["tick"] = (st["tick"].astype(np.int64) + (1 << 32) - 2000
                  ).astype(np.uint32)      # wraps 2 s in
    c = clock.estimate(st)
    assert (np.diff(c.t) >= 0.0).all()
    assert np.abs(c.t - prod).max() < 0.002


def test_no_device_clock_is_reported_not_guessed():
    """A record with no `tick`, or a zeroed one, has only receive time."""
    st, _ = _state(n=3000)
    st["tick"] = 0
    assert clock.has_device_clock(st) is False
    assert clock.estimate(st) is None

    without = st[["index", "t_recv", "q"]].copy()
    assert clock.has_device_clock(without) is False
    assert clock.estimate(without) is None


def test_describe_is_json_ready():
    c = clock.estimate(_state(stall=(4.0, 0.2), jitter=0.002)[0])
    d = c.describe()
    assert d["model"] == "affine"
    assert d["local_only"] is False
    assert d["stalls_50ms"] > 0
    assert d["latency_median_ms"] < 5.0
    assert d["latency_max_ms"] > 150.0
    for v in d.values():
        assert v is None or isinstance(v, (bool, int, float, str))


# --- through the timeline ---------------------------------------------------

def test_stall_does_not_move_the_pose(tmp_path):
    """The `20260920-161456` failure, in miniature.

    The arm holds still and the telemetry stalls. On receive time the measured
    pose for the whole hole is the pre-hole sample — held, then jumped — so any
    comparison against a moving command reads the hole as tracking error. On the
    device clock the samples land where they were taken and the motion is
    continuous.
    """
    run = fx.write_record(tmp_path / "stalled", tick=True, drift_ppm=33.0,
                          stall=(1.0, 0.3))
    rollout = loader.load(run)
    try:
        dev = timeline.build(rollout, clock="device")
        recv = timeline.build(rollout, clock="recv")
        assert dev.state_clock is not None
        assert recv.state_clock is None
        assert dev.state_clock.stalls_50ms > 200

        # `flange` x rises linearly with the sample index in the fixture, so the
        # measured pose is a ramp and a hold in it is a flat spot. Compare how
        # flat: on receive time the frames inside the hole all resolve to the
        # same state record, and on the device clock they do not.
        def flat_frames(tl):
            col = tl.index["state"]
            inside = (tl.t - tl.t0 >= 1.0) & (tl.t - tl.t0 < 1.3)
            return int(np.count_nonzero(np.diff(col[inside]) == 0))

        assert flat_frames(recv) >= 8
        assert flat_frames(dev) == 0
    finally:
        rollout.close()


def test_device_is_the_default_and_recv_is_unchanged(tmp_path):
    """The old behaviour is still available, and still byte-identical."""
    run = fx.write_record(tmp_path / "plain", tick=True, drift_ppm=-120.0)
    rollout = loader.load(run)
    try:
        assert timeline.build(rollout).state_clock is not None      # default
        recv = timeline.build(rollout, clock="recv")
        assert np.allclose(recv.q, timeline._lerp(
            rollout.state["t_recv"], rollout.state["q"], recv.t))
    finally:
        rollout.close()


def test_a_record_without_tick_still_builds(tmp_path):
    """Every rollout written before 2026-09-16, asked for the device clock."""
    run = fx.write_record(tmp_path / "old")           # tick=False
    rollout = loader.load(run)
    try:
        tl = timeline.build(rollout, clock="device")
        assert tl.state_clock is None
        assert timeline.describe(tl, rollout)["state_clock"] is None
        # And it is the receive-time timeline, not a degraded one.
        assert tl.n == timeline.build(rollout, clock="recv").n
    finally:
        rollout.close()


def test_describe_carries_the_clock(tmp_path):
    run = fx.write_record(tmp_path / "d", tick=True, stall=(1.0, 0.25))
    rollout = loader.load(run)
    try:
        d = timeline.describe(timeline.build(rollout), rollout)
        assert d["state_clock"]["model"] == "affine"
        assert d["state_clock"]["stalls_50ms"] > 150
    finally:
        rollout.close()
