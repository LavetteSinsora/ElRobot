"""Replan marks: when P5 was woken, and when its chunk reached the ring.

The chunk ring records six instants per replan (`schema.chunk_dtype`). Two
of them are drawn on the timeline:

* `t_trigger`       — the scheduled instant, when P5 wakes.
* `t_chunk_written` — when the ring write for the finished chunk begins.

The other four (`t_anchor`, `t_obs_received`, `t_cpu_processed`,
`t_chunk_received`) are deliberately not surfaced. They subdivide the same span,
and a subdivision of 90 ms is a subdivision of a few pixels at any zoom this
viewer offers. Adding them later is a change to the drawing, not to this module
— they are already on disk.

**Sync and async need nothing different here, and that is structural.** P5
writes the same six timestamps through the same ring in both modes; the mode
only changes what the marks *look like*. In sync the spans sit in the gaps
between chunks and coverage partitions the tick line; in async both overlap by
design. The only concession to async in this module is `lane`, which the drawing
uses to stack overlapping coverage bars rather than paint them over each other.

**Coverage is read off the recorded ticks, never fitted, and that is not
fastidiousness.** A chunk states which ticks it is for —
`[k_first_action, k_first_action + n_actions)` — and turning a tick into a wall
instant looks like it should be `k / control_hz`, because P7 executes on a fixed
period. It is not. On one recorded rollout, `k` jumps by 2 with a 149 ms
hitch at EVERY sync chunk boundary (P6 re-anchors to `k_exec + 2`, and the arm
waits out the inference), and jumps 0 -> 2 across a 59 s READY wait. A
least-squares line through that record returns a 116 ms "control period" and is
33 s out at its worst point; a line through only its policy ticks is still 58 s
out. Every coverage bar would have been drawn somewhere the arm never was.

So the span of a chunk is the span of the ticks that ACTUALLY EXECUTED inside
its range: the first recorded tick at or after `k0`, up to the first at or after
`k0 + n`. That reads as "this chunk was driving the arm over this stretch",
which is the question the bar is on screen to answer, and it makes consecutive
sync chunks share an endpoint exactly — sync chunks tile the tick line. The one
extrapolation is the final chunk's tail, which the run ended in the middle of:
its last tick is given the nominal period, measured from the record's own
single-tick steps rather than from the config.

Built once per Load by the inspection server; drawn by
`dashboards/web/static/marks.js`. For a user: `docs/deploying_a_policy/rollout_inspection.md`, "The view".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: The two chunk fields this module reads. Named once, because a record that
#: lacks them yields no marks rather than a KeyError halfway through a load.
TRIGGER = "t_trigger"
WRITTEN = "t_chunk_written"


@dataclass(frozen=True)
class TickLine:
    """The executed ticks: `k` ascending, and the instant each was due.

    Built from `body_targets`, which is every target P6 and P0 published, so it
    covers homing (negative `k`) as well as the rollout. Ticks that were never
    published are absent, and their absence is information — it is where the
    arm was waiting.
    """

    k: np.ndarray                       # int64, ascending, unique
    deadline: np.ndarray                # float64, the instant of each
    #: The record's own single-tick step, seconds: the median of `deadline`
    #: differences across ticks that are one apart. 0.0333 on a 30 Hz run.
    period: float

    def span(self, k0: int, n: int) -> tuple[float, float] | None:
        """Wall span of the ticks executed within `[k0, k0 + n)`.

        `None` when no tick in that range ever executed.
        """
        lo = int(np.searchsorted(self.k, k0, side="left"))
        # Nothing at or after `k0`, or the nearest such tick is already past the
        # range: no tick of this chunk ever ran. That is a chunk written as the
        # run ended, and a warm-up chunk whose ticks are P0's homing, and both
        # want the same answer — instants worth marking, no stretch to shade.
        # Without the second half of this test the warm-up case draws a
        # zero-width bar pinned to the rollout's first tick.
        if lo >= len(self.k) or self.k[lo] >= k0 + n:
            return None
        hi = int(np.searchsorted(self.k, k0 + n, side="left"))
        start = float(self.deadline[lo])
        # Past the end of the record, the last tick still occupied its own
        # period; stopping the bar on its deadline would draw the final chunk
        # one tick short of what it drove.
        end = (float(self.deadline[hi]) if hi < len(self.k)
               else float(self.deadline[-1]) + self.period)
        return start, end


@dataclass
class Replan:
    """One chunk: the two instants, and the ticks it was produced to cover."""

    chunk_id: int
    k0: int
    n: int
    #: Absolute monotonic, the same clock as `Timeline.t`.
    trigger: float
    written: float
    #: `[first executed tick, first tick past the range)`, absolute, or `None`.
    cover: tuple[float, float] | None
    #: Which row to draw the coverage bar in. 0 for every chunk in sync, where
    #: coverage partitions the tick line; 0/1/... in async, where it does not.
    lane: int = 0

    @property
    def latency(self) -> float:
        return self.written - self.trigger


def tick_line(body_targets: np.ndarray) -> TickLine | None:
    """The executed ticks, deduplicated and sorted. `None` when there are none.

    `k` can repeat in the stream — P6 re-solves ticks after a new chunk lands in
    async, so the same tick is published more than once — and the LAST write
    wins, because that is the target P7 was holding when the deadline arrived.
    The deadline is the same either way (it is what makes the tick that tick),
    so this only has to pick one, not reconcile two.
    """
    if not len(body_targets):
        return None
    k = body_targets["k"].astype(np.int64)
    d = body_targets["deadline"].astype(np.float64)
    order = np.argsort(k, kind="stable")
    k, d = k[order], d[order]
    keep = np.append(np.diff(k) != 0, True)          # last of each run of `k`
    k, d = k[keep], d[keep]

    step = np.diff(d)[np.diff(k) == 1]
    # A record with no two adjacent ticks anywhere cannot state its own period.
    # 30 Hz is the infra's default control rate, and
    # it is only ever used to give the final tick a width.
    period = float(np.median(step)) if len(step) else 1.0 / 30.0
    return TickLine(k, d, period)


@dataclass
class Marks:
    """Every replan that touches the timeline's window, and the ticks behind it."""

    items: list[Replan]
    ticks: TickLine | None

    def as_json(self, timeline) -> dict:
        """What the browser draws: seconds from the window start, and frames.

        Times are relative to `t0` because that is the axis the scrubber is in,
        and they are NOT clamped to it — a chunk triggered before the window
        opens and written inside it is a real thing, and the drawing clips it.
        Frames are clamped, because a frame is an index and there is no honest
        index outside the timeline; the browser only ever asks for the frame of
        a mark it decided to draw.
        """
        out = []
        for r in self.items:
            out.append({
                "id": r.chunk_id,
                "k0": r.k0,
                "n": r.n,
                "lane": r.lane,
                "trigger": round(r.trigger - timeline.t0, 4),
                "written": round(r.written - timeline.t0, 4),
                "trigger_frame": timeline.frame_at(r.trigger),
                "written_frame": timeline.frame_at(r.written),
                "latency_ms": round(r.latency * 1000.0, 2),
                "cover": None if r.cover is None else [
                    round(r.cover[0] - timeline.t0, 4),
                    round(r.cover[1] - timeline.t0, 4)],
            })

        lat = np.array([r.latency for r in self.items]) * 1000.0
        gaps = np.diff([r.trigger for r in self.items])
        return {
            "replans": out,
            "n": len(out),
            "lanes": 1 + max((r.lane for r in self.items), default=0),
            "latency_ms": None if not len(lat) else {
                "min": round(float(lat.min()), 1),
                "med": round(float(np.median(lat)), 1),
                "max": round(float(lat.max()), 1),
            },
            # The observed replan period. Not the config's `P / control_hz`:
            # this is what the rollout DID, and the two part company as soon as
            # inference overruns its budget.
            "period_s": None if not len(gaps) else round(float(np.median(gaps)), 4),
            "tick_period_ms": (None if self.ticks is None
                               else round(self.ticks.period * 1000.0, 3)),
        }


def build(rollout, timeline) -> Marks:
    """Every replan whose marks or coverage touch the timeline's window.

    A chunk is kept when anything about it lands inside `[t0, t1]` — either
    instant, or any part of its coverage. Chunks entirely outside are dropped
    rather than pinned to the edge: the window is cut to ACTIVE, and P5's
    warm-up inferences sit before it. A mark clamped to the start of the bar
    would read as "the first replan happened here", which is the one thing it
    did not do.

    An empty chunk stream — a record that wrote none, or one truncated before it
    reached disk — yields no marks. So does a stream with neither instant in its
    dtype. Both mean the same thing to the UI, which says so rather than showing
    a bar that looks like a rollout with no replans in it.
    """
    ch = rollout.chunks
    names = (ch.dtype.names or ()) if len(ch) else ()
    if TRIGGER not in names or WRITTEN not in names:
        return Marks([], None)

    ticks = tick_line(rollout.body_targets)
    items: list[Replan] = []
    for row in ch:
        trigger, written = float(row[TRIGGER]), float(row[WRITTEN])
        # A slot P9 swept while P5 was mid-write, or a field a record's own
        # generation never filled. `time.monotonic()` is never 0 on a machine
        # that has been up a second, so this tests "no instant here" without
        # needing a sentinel of its own.
        if not (trigger > 0.0 and written > 0.0):
            continue
        k0, n = int(row["k_first_action"]), int(row["n_actions"])
        cover = ticks.span(k0, n) if (ticks is not None and n > 0) else None

        lo = min(trigger, cover[0]) if cover else trigger
        hi = max(written, cover[1]) if cover else written
        if hi < timeline.t0 or lo > timeline.t1:
            continue
        items.append(Replan(int(row["chunk_id"]), k0, n, trigger, written, cover))

    items.sort(key=lambda r: r.trigger)
    _assign_lanes(items)
    return Marks(items, ticks)


def _assign_lanes(items: list[Replan]) -> None:
    """Stack coverage bars that overlap in time. Every lane is 0 in sync.

    Greedy first-fit over chunks already sorted by trigger: a chunk takes the
    lowest lane whose last bar has ended. In sync that is always lane 0, because
    consecutive chunks share an endpoint and nothing more — hence the
    non-strict `>=`, which keeps a chunk starting exactly where the previous one
    ended on the same row instead of opening a second lane for the whole run.
    In async the lanes are the overlap, drawn.
    """
    lane_end: list[float] = []
    for r in items:
        if r.cover is None:
            continue
        start, end = r.cover
        for lane, busy_until in enumerate(lane_end):
            if start >= busy_until:
                r.lane, lane_end[lane] = lane, end
                break
        else:
            r.lane = len(lane_end)
            lane_end.append(end)
