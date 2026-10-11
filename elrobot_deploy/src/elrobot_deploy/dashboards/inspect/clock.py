"""When a state sample was PRODUCED, not when it arrived.

`state.bin` carries two clocks per record: `t_recv`, stamped by P1 the moment the
DDS sample came out of the socket, and `tick`, the robot's own millisecond
counter at the instant the sample was taken (`runtime/drivers/robot_state.py::LOWSTATE`). Every
other stream in a record — the cameras, P6's targets, P7's commands, the session
log — is stamped on the box's `time.monotonic()`, so `t_recv` is the clock that
lets state share a timeline with them. It is also the clock that carries every
hiccup on the wire.

On one recorded rollout that mattered: one 318 ms receive hole at t+91.19 s, with
`tick` advancing 3 ms across it, left the viewer holding a 375 ms-old pose while
the setpoint walked away from it. The left elbow read 0.103 rad of "tracking
error" at display frame 2754. Placed at its production instant instead it is
0.018 rad — the same residue every other tick carries.

**This module maps `tick` into the box's monotonic frame.** Not into robot
milliseconds: the answer has to be comparable with a camera frame and a
commanded deadline, and those only exist on the box clock. So the output is
still `time.monotonic()` seconds, and the only thing that changes is that each
state record is placed where it was *taken* rather than where it was *received*.

**The offset is estimated from the record's own lower envelope.** `t_recv - tick`
is production-offset plus delivery latency, and latency is non-negative, so the
*minimum* of that difference over a stretch of samples is the offset plus the
irreducible transport cost. Taking bin minima and fitting a line through them
recovers offset and rate together. On the records measured here the envelope is
straight to 20-50 µs rms, and the recovered latency has a median of 0.8 ms and a
p99 of 2-4 ms — which is what a 1 kHz DDS topic on loopback should look like,
and is the evidence that the line really is the envelope and not a line through
the middle of the data.

**`tick` is a loop counter, not a calibrated clock.** Its measured rate against
the box ranges from -944 to +403 ppm across records of the reference robot — far more
than two crystals would disagree by, and it differs run to run, so it is the
robot's control loop being counted rather than an oscillator being divided. This
costs nothing here because the rate is fitted per record and never assumed, but
it does mean `tick` is not a time source on its own: a constant 1000 ticks/s
would be wrong by 6 ms over a 194 s rollout even on the best-behaved record, and
by 60 ms on the worst.

**Two models, because one record in four measured had a curved envelope.** On
one of them the minimum latency itself climbs 75 ms over 115 s — P1 falling
progressively behind, not a clock rate — and a straight line through that curve
puts a quarter of the samples *before* they were produced. So the affine fit is
checked against its own residual and rejected when it implies negative latency,
and a centred rolling minimum takes over: strictly below the data by
construction, and therefore incapable of that failure. It cannot separate a
stall longer than its window from the envelope, which is the honest limit —
a backlog that never drains has no observable zero-latency reference.

**The staircase is smoothed out AFTER the envelope, never before.** `tick` is
whole milliseconds while the robot publishes a little above 1 kHz, so ~4% of
samples share a tick and ~1% skip one: taken literally the device clock steps in
1 ms jumps, which is harmless for placing a pose and not harmless for reading a
velocity off consecutive samples. Within a contiguous run — `index` consecutive,
no delivery gap — production is uniform in the sample number, so refitting
`tick` against `index` there removes the steps.

Doing that BEFORE the envelope fit wrecks it, which is worth stating because it
is the natural order to write and the damage is silent. Each run is fitted
independently, so consecutive runs land up to half a millisecond apart at their
seam; that lands squarely in the lower envelope's residual, and on
one record it took the fit from 0.02 ms rms to 0.43 ms, pushed samples
below the line, and tipped a record with a textbook straight envelope into the
rolling fallback. The envelope wants the counter exactly as the robot reported
it: its quantisation is symmetric noise, and a bin minimum over a thousand
samples barely feels it. So the fit runs on the raw counter and the smoothing is
applied to the result.

**This is the one implementation.** `modes/executor/series.robot_time` calls
it: two copies would mean the same record could be drawn on two slightly
different clocks in two panels of the same dashboard.

Used by `timeline.py` and the executor mode; the device clock is described in
`docs/infra_logic/arena_layout_and_reads.md`, "Ticks and clocks".

**What this cannot recover.** The irreducible minimum latency is absorbed into
the offset, so production instants are late by that constant — sub-millisecond
here, and shared by every sample, so it shifts the whole stream rather than
distorting it. And `grip_meas` rides in the state record but comes off a
*different* DDS topic with its own `t_grip_recv` and no device clock of its own;
moving the record moves the gripper reading with it, which is right to within
the two topics' relative latency and is not separately observable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Width of the bins whose minima the affine fit runs on, in device seconds. One
#: second holds ~1000 samples, so its minimum is a tight estimate of the
#: envelope, and a 20 s record still yields 20 points to fit.
_BIN_S = 1.0

#: Width of the rolling-minimum window, in device seconds, for the fallback.
#: Wide enough that the ~200 ms replan-cadence structure and any single stall
#: sits inside it against low-latency neighbours; narrow enough to follow an
#: envelope that drifts at ~1 ms/s.
_WIN_S = 4.0

#: How far below the affine envelope a sample may sit before the fit is judged
#: to have missed, in milliseconds. `tick` is quantised to 1 ms, so a fraction
#: of a millisecond of apparent negative latency is the quantisation and not a
#: bad fit.
_NEG_TOL_MS = 1.0

#: What fraction of samples may exceed `_NEG_TOL_MS` below the envelope before
#: the affine model is rejected. Non-zero because a single outlying bin minimum
#: can drag the line, and one sample in a thousand is not a wrong model.
_NEG_FRAC_MAX = 0.001

#: Shortest run of contiguous samples worth de-quantising. Below this the fit
#: is dominated by the 1 ms quantisation it is trying to remove.
_MIN_RUN = 20

#: A tick step this large ends a contiguous run: the robot skipped a cycle or
#: the delivery gapped, and either way production was not uniform across it.
_RUN_BREAK_S = 3e-3

#: How many points the rolling minimum is evaluated at before being interpolated
#: back onto every sample. An exact per-sample rolling minimum over a 4 s window
#: of a 200k-record stream is 8e8 comparisons; the envelope is smooth at this
#: scale, so 4000 probes resolve it to well under the latency it is measuring.
_PROBES = 4000


@dataclass(frozen=True)
class DeviceClock:
    """Production instants for one state stream, on the box's monotonic clock."""

    #: `(n,)` monotonic seconds: when each state record was TAKEN, in the same
    #: frame as `t_recv`. Non-decreasing, so it can be `searchsorted` exactly
    #: like the column it replaces.
    t: np.ndarray
    #: `(n,)` milliseconds: how late each record arrived. `t_recv - t`, and the
    #: thing the receive-time view was silently folding into the pose.
    latency_ms: np.ndarray
    #: `"affine"` or `"rolling"` — which envelope model produced `t`. See the
    #: module docstring; a reader should know which one they are looking at.
    model: str
    #: Fitted rate of `tick` against the box clock, in ppm, for the affine
    #: model. `None` for the rolling one, which fits no rate.
    drift_ppm: float | None
    #: Spread of the bin minima about the fitted line, in milliseconds — the
    #: affine model's own statement of how straight the envelope was. `None` for
    #: the rolling model.
    env_rms_ms: float | None

    @property
    def local_only(self) -> bool:
        """Is the absolute offset trustworthy, or only the local structure?

        True for the rolling model. Its envelope is a window-wide minimum, so a
        latency that climbs slowly and never returns to the floor is absorbed
        into the estimated production time rather than reported as lateness —
        on one recorded rollout that is ~75 ms accumulated over 115 s. Nothing can
        do better from one record: a backlog that never drains offers no
        zero-latency reference to measure it against. What survives is what the
        viewer mostly needs — the local structure. Timing WITHIN a few seconds,
        which is every comparison against a chunk, a replan or a commanded tick,
        is right to the latency floor either way.
        """
        return self.model == "rolling"

    @property
    def stalls_50ms(self) -> int:
        """Records that arrived more than 50 ms after they were produced."""
        return int((self.latency_ms > 50.0).sum())

    def describe(self) -> dict:
        p = np.percentile(self.latency_ms, [50, 95, 99, 100])
        return {
            "model": self.model,
            "local_only": self.local_only,
            "drift_ppm": None if self.drift_ppm is None
            else round(self.drift_ppm, 1),
            "env_rms_ms": None if self.env_rms_ms is None
            else round(self.env_rms_ms, 3),
            "latency_median_ms": round(float(p[0]), 2),
            "latency_p95_ms": round(float(p[1]), 2),
            "latency_p99_ms": round(float(p[2]), 2),
            "latency_max_ms": round(float(p[3]), 1),
            "stalls_50ms": self.stalls_50ms,
        }


def _tick_seconds(tick: np.ndarray) -> np.ndarray:
    """`tick` as seconds, unwrapped.

    `tick` is `<u4` milliseconds, so it wraps every 49.7 days of robot uptime.
    A rollout is minutes long and will essentially never straddle a wrap, but
    the one that does would otherwise map its whole tail 49.7 days into the past
    — a silent, spectacular corruption, against a two-line guard.
    """
    t = tick.astype(np.int64)
    wraps = np.r_[0, np.cumsum(np.diff(t) < -(1 << 31))]
    return (t + wraps * (1 << 32)) / 1000.0


def _dequantise(x: np.ndarray, index: np.ndarray) -> np.ndarray:
    """`x` with the 1 ms staircase fitted out, run by run.

    Applied to the OUTPUT of the envelope estimate and never to its input — see
    the module docstring, which has the measurement.

    A run is a stretch over which the robot published every cycle and P1 missed
    none, so the sample number is a linear function of time and the residual is
    the quantisation. Refitting recovers a smooth clock without assuming
    anything about the RATE — that is still measured, per run, from the counter
    itself.

    No half-millisecond bias correction, deliberately. Whether the firmware
    floors or rounds its millisecond decides whether one is owed, and a constant
    added here is subtracted again by the envelope fit downstream — it would
    change nothing except by being applied to some runs and not others, which is
    the one way it could do harm.
    """
    out = x.copy()
    brk = np.flatnonzero((np.diff(x) >= _RUN_BREAK_S) | (np.diff(index) != 1)) + 1
    edges = np.r_[0, brk, len(x)]
    idx = index.astype(np.float64)
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a >= _MIN_RUN:
            slope, offset = np.polyfit(idx[a:b], x[a:b], 1)
            out[a:b] = offset + slope * idx[a:b]
    return out


def _bin_minima(x: np.ndarray, r: np.ndarray, bin_s: float) -> tuple:
    """Per-bin `(mean x, min r)`, the points the affine envelope is fitted to."""
    b = ((x - x[0]) // bin_s).astype(np.int64)
    edges = np.r_[0, np.flatnonzero(np.diff(b)) + 1, len(b)]
    xs = np.array([x[a:c].mean() for a, c in zip(edges[:-1], edges[1:])])
    ys = np.array([r[a:c].min() for a, c in zip(edges[:-1], edges[1:])])
    return xs, ys


def _affine(x: np.ndarray, r: np.ndarray) -> tuple | None:
    """`(offset, rate, rms_ms)` of the line under `r`, or `None` if too short.

    Least squares on the bin minima, then two rounds of dropping the upper
    quartile of the residual. The minima are already a lower envelope, so what
    is being trimmed is the bins that contain no low-latency sample at all —
    a stretch where the stream never caught up — and those are exactly the
    points that would tilt the line.
    """
    xs, ys = _bin_minima(x, r, _BIN_S)
    if len(xs) < 4:
        return None
    A = np.c_[np.ones(len(xs)), xs]
    keep = np.ones(len(xs), dtype=bool)
    coef = np.zeros(2)
    for _ in range(3):
        coef = np.linalg.lstsq(A[keep], ys[keep], rcond=None)[0]
        res = ys - A @ coef
        nxt = res <= np.percentile(res[keep], 75)
        if nxt.sum() < 4:
            break
        keep = nxt
    rms = float(np.std((ys - A @ coef)[keep]) * 1000.0)
    return float(coef[0]), float(coef[1]), rms


def _rolling(x: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Centred rolling minimum of `r` over `_WIN_S` of device time.

    Evaluated at `_PROBES` points and interpolated, then clamped under `r`
    itself. The clamp is what guarantees non-negative latency everywhere: the
    interpolation between two probes can rise above a sample that sits between
    them, and a single such sample would otherwise be reported as arriving
    before it was taken.
    """
    lo = np.searchsorted(x, x - _WIN_S / 2.0, side="left")
    hi = np.searchsorted(x, x + _WIN_S / 2.0, side="right")
    probe = np.unique(np.linspace(0, len(x) - 1, min(len(x), _PROBES)
                                  ).astype(np.int64))
    env = np.array([r[lo[i]:hi[i]].min() for i in probe])
    return np.minimum(np.interp(x, x[probe], env), r)


def has_device_clock(state: np.ndarray) -> bool:
    """Can this state stream be placed on its production instants?

    Asked of the array, not of a date or a version: an old record may have
    no `tick` column at all, and a record whose column exists
    but is all zeros came from a source that does not fill it (`FakeLowState`
    aside, which counts a 500 Hz loop). Either way the honest answer is that
    this record only has receive time, and the caller keeps using it.
    """
    if "tick" not in (state.dtype.names or ()):
        return False
    return bool(len(state)) and int(state["tick"].max()) > 0


def estimate(state: np.ndarray) -> DeviceClock | None:
    """Production instants for `state`, or `None` when it has no device clock.

    The returned `t` is non-decreasing by construction (`maximum.accumulate`),
    which the affine model gives for free — `tick` is non-decreasing and the
    rate correction is monotone — and the rolling one does not: its envelope may
    fall faster than `tick` rises. A timeline built on a column that steps
    backwards would resolve frames to the wrong records without failing, so the
    guarantee is enforced here rather than assumed downstream.
    """
    if not has_device_clock(state):
        return None

    raw = _tick_seconds(state["tick"])
    r = state["t_recv"] - raw
    # Smoothed for the OUTPUT only; the envelope below is measured against the
    # counter as reported. See the module docstring for what happens otherwise.
    x = _dequantise(raw, state["index"].astype(np.int64))

    fit = _affine(raw, r)
    if fit is not None:
        offset, rate, rms = fit
        env = offset + rate * raw
        lat = (r - env) * 1000.0
        if float((lat < -_NEG_TOL_MS).mean()) <= _NEG_FRAC_MAX:
            return DeviceClock(t=np.maximum.accumulate(x + env),
                               latency_ms=lat, model="affine",
                               drift_ppm=rate * 1e6, env_rms_ms=rms)

    env = _rolling(raw, r)
    return DeviceClock(t=np.maximum.accumulate(x + env),
                       latency_ms=(r - env) * 1000.0, model="rolling",
                       drift_ppm=None, env_rms_ms=None)
