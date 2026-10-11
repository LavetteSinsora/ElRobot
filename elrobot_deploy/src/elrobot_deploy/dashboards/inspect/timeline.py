"""The index table: for each display frame, which record of each stream is current.

One master grid in monotonic seconds. For every frame it stores an integer index
into each stream, so scrubbing is an array lookup and never a search — which is
the whole difference between a scrubber that tracks the mouse and one that lags
behind it.

**There is no clock alignment problem to solve.** Every timestamp in every stream
is `time.monotonic()` from the same machine: state at ~1 kHz, the
cameras at ~30 Hz. No offsets to estimate, no drift to correct. What
remains is resampling, and that is all this module does.

**The grid lands ON the commanded deadlines, and everything else is filler.**
A uniform 30 Hz grid and the tick line are both nominally 30 Hz, and it is very
easy to assume they are therefore the same grid. They are not: sync re-stamps
the epoch once per chunk, so the tick line slips ~158 ms per replan and
the phase between the two wanders. On one recorded rollout that left the commanded
overlay a mean 37 ms stale against a measured pose accurate to half a
millisecond — and the whole point of this viewer is comparing those two. The
apparent gap at the first tick read 8.8 mm at the frame and 4.7 mm at the
deadline itself; half of what was on screen was the grid, not the robot.

So the anchors are the deadlines of the ticks that actually executed, and
`_fill` inserts frames at roughly `1/fps` wherever consecutive anchors are
further apart than that. Commanded poses are then EXACT on every anchor frame,
with no resampling of the sparse authoritative stream onto a grid it does not
live on. Filler exists so the head (before the first tick), the tail, and the
~197 ms replan gaps are still walkable — the cameras and the measured pose keep
moving through them, which is most of what there is to see there.

**Two resampling rules, one per kind of signal.** A body target and a camera
frame are HELD — the target until the next one arrives, the image until the next
is captured — so both use zero-order hold, and blending two camera frames would
invent an image the robot never saw. Joint state is a *continuously sampled*
signal at ~1040 Hz, so it is interpolated to the exact frame instant instead.
The difference is under 0.1 mm at the TCP (hold staleness is 0.49 ms mean, 2.73 ms
worst), which is precisely why it is worth doing: it costs nothing and it makes
"commanded and measured are stated at the same instant" true by construction
rather than true to within a rounding argument.

**The commanded overlay steps, it does not ramp.** Held on `deadline`, so
between two ticks it shows the last pose the policy asked for and then jumps.
P7 does something more elaborate — it freezes for ~132 ms and then ramps over
~66 ms — but that is the *servo's* setpoint, and this overlay is the
policy's intent, which is a step function by construction. The ramp is visible
anyway: it is what the measured pose is doing.

**Two cameras genuinely disagree, and the table records that.** They
free-run at ~30 Hz without a shared trigger, so they drift in and out of phase:
on one recorded rollout roughly a third of display frames want different frame
numbers from the two cameras. Those images really were captured up to ~15 ms
apart. Forcing them onto one index would be a lie about the data.

**The window opens at ACTIVE, not at the first recorded sample.** P9 records the
whole session — CHECK, the homing ramp, P5 warming, the READY prompt — and none
of that is the rollout. On one recorded rollout it is 60 of 80 seconds. The grid is
cut to the ACTIVE edge from `session.bin`, which is P0's own
statement about when the policy took the arm, rather than a
motion threshold that would have to be tuned and would still be wrong on a
rollout whose first action is to hold still. Records with no session log keep the
full window.

**State is placed where it was PRODUCED, not where it arrived.** `state.bin`
carries the robot's own `tick` beside `t_recv`, and `clock.estimate` maps it into
the box's monotonic frame, so a receive-side stall no longer drags the measured
pose across the timeline. On one recorded rollout one 318 ms hole read as 0.103 rad
of left-elbow tracking error at frame 2754; on the device clock it is 0.018 rad,
the same residue as every neighbouring tick. `clock="recv"` places state on
its receive time instead, and a record with no usable `tick` falls back to it on its own. The
CAMERAS keep their `t_recv` either way — they carry no device clock, so there is
nothing to correct them to, and inventing one would be worse than saying so.

**`-1` means "nothing yet", and it is a real state rather than an edge case.**
Before the first record of a stream arrives there is no correct index, and Python
would happily read `-1` as the *last* element — presenting the end of the run as
its beginning. It is rare with the window opening at ACTIVE, but it is
not gone: in sync the first chunk is still inferring when the phase flips, so the
commanded overlay has no answer for the first hundred-odd milliseconds.

Built once per Load (`build`); `scene.py` and `events.py` index through it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from elrobot_deploy.dashboards.inspect import clock as clock_mod

#: Display rate. 30 to match the cameras — a finer grid would repeat frames
#: without adding information, and a coarser one would drop some.
DEFAULT_FPS = 30.0

#: Index value for "this stream has no record at or before this instant".
NOTHING = -1

#: How close a filler frame may come to the anchor that ends its gap, as a
#: fraction of the nominal period. Below this it is not emitted: a frame a
#: millisecond before a deadline shows the same images and very nearly the same
#: pose, and it would shadow the anchor — the one frame in that neighbourhood
#: whose commanded pose is exact.
_MIN_SEPARATION = 0.5


def _hold(times: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Newest index at or before each grid instant; `NOTHING` before the first.

    `times` must be ascending. `searchsorted(..., "right") - 1` is exactly the
    zero-order hold, and it yields -1 below `times[0]` — which is the answer we
    want, so it is kept rather than clipped, and named.
    """
    idx = np.searchsorted(times, grid, side="right") - 1
    return idx.astype(np.int32)


def _lerp(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """`values` sampled at `times`, linearly interpolated onto `grid`.

    For the CONTINUOUSLY SAMPLED streams only — joint state, at ~1040 Hz. A
    held target or a camera frame must never come through here; see the module
    docstring for why the two kinds are treated differently.

    Clamped at both ends rather than extrapolated: outside the samples the
    honest answer is the nearest one, and the window is built from the streams'
    own overlap, so this only ever binds on the final frame.

    Interpolation is only defensible while the samples are dense. On
    a typical record the largest gap in `state` is 4.4 ms and none exceed 5 ms,
    so every bracket is about one millisecond wide. A stream with a real dropout
    would have this drawing a straight line across it, which is why callers pass
    state and nothing else.
    """
    i = np.clip(np.searchsorted(times, grid, side="right") - 1, 0, len(times) - 2)
    span = times[i + 1] - times[i]
    u = np.clip((grid - times[i]) / np.where(span > 0, span, 1.0), 0.0, 1.0)
    lo, hi = values[i], values[i + 1]
    # `u` needs the trailing axes of `values` to broadcast: (n,) -> (n, 1, ...).
    return lo + u.reshape((-1,) + (1,) * (values.ndim - 1)) * (hi - lo)


def _fill(anchors: np.ndarray, t0: float, t1: float, dt: float) -> tuple:
    """The grid: every anchor, plus evenly spread filler in the wide gaps.

    Returns `(t, on_anchor)`. `t` is ascending and starts at `t0`; `on_anchor`
    marks the frames that ARE a commanded deadline, which is what lets the UI
    say "this frame is exact" and lets a test assert it without re-deriving the
    grid.

    Filler is laid at EXACTLY `dt` from the start of its gap, and simply stops
    short of the next anchor, leaving one runt interval per gap. Spreading it
    evenly to hit the anchor exactly was the other candidate and it is worse for
    a reason that only shows up on records with no commands at all: the whole
    window is then one gap, and even spreading gives it a period of `(t1-t0)/n`
    rather than `1/fps`. On a 30 s record that is 18 µs per frame of drift, which
    is enough to walk a camera whose rate IS the nominal rate off its own
    samples — every frame lands a hair before the image it should be showing.
    Exact `dt` keeps a filler instant meaningful (it is `anchor + i/fps`) and
    makes a command-less record reproduce the plain uniform grid exactly.

    The runt is harmless because playback advances a clock and looks the frame
    up, rather than assuming every frame is one period long.
    """
    inner = anchors[(anchors > t0) & (anchors < t1)].tolist()
    edges = [t0, *inner, t1]
    t: list[float] = []
    on_anchor: list[bool] = []
    for j, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        t.append(a)
        # Edge 0 is `t0` and the last is `t1`; everything between is an anchor.
        # Tested by POSITION, never by value — `t0` can coincide with an anchor
        # and two anchors can coincide, and identity on floats answers neither
        # question.
        on_anchor.append(j > 0)
        i = 1
        while a + i * dt < b - _MIN_SEPARATION * dt:
            t.append(a + i * dt)
            on_anchor.append(False)
            i += 1
    # The window's own end, unless the last filler already sits on top of it.
    if not t or edges[-1] - t[-1] >= _MIN_SEPARATION * dt:
        t.append(edges[-1])
        on_anchor.append(False)
    return np.asarray(t, dtype=np.float64), np.asarray(on_anchor, dtype=bool)


@dataclass
class Timeline:
    """The grid, and one index column per stream."""

    fps: float
    t0: float
    t1: float
    #: Monotonic instant of each display frame, absolute — the same clock the
    #: record uses, so a frame can be quoted straight back at the raw streams.
    t: np.ndarray
    #: `{stream name: (n_frames,) int32 index}`. Cameras are keyed by camera
    #: name; the rest by stream name.
    index: dict[str, np.ndarray]
    #: `(n_frames,)` bool: this frame IS a commanded deadline, so the commanded
    #: pose on it is the exact one P6 published rather than a held one. False on
    #: filler. Empty (all False) for a record with no commands.
    on_anchor: np.ndarray
    #: Interpolated joint state, one row per frame — `state["q"]` resampled to
    #: the exact frame instant. Held out of `index` on purpose: an index says
    #: "this record", and there is no record at these instants, which is the
    #: whole point of interpolating.
    q: np.ndarray
    #: The same for `state["grip_meas"]`.
    grip: np.ndarray
    #: How many seconds of recording sit BEFORE `t0` — the setup the viewer
    #: skipped. 0.0 when nothing was trimmed. See `build`.
    skipped: float = 0.0
    #: Why the window opens where it does: `"active"` when it was cut to the
    #: ACTIVE edge, `"streams"` when it is the plain overlap of state and the
    #: cameras. The UI says which, because "this rollout is 20 s long" and "this
    #: rollout is 80 s long and you are seeing the last 20" are different facts.
    window: str = "streams"
    #: The device-clock estimate the state column was placed on, or `None` when
    #: the record has no usable `tick` or the caller asked for receive time.
    #: Carried so the UI can say which clock it is showing and how late the
    #: stream actually was; see `clock.DeviceClock`.
    state_clock: "clock_mod.DeviceClock | None" = None

    @property
    def n(self) -> int:
        return int(len(self.t))

    @property
    def duration(self) -> float:
        return float(self.t1 - self.t0)

    def rel(self, frame: int) -> float:
        """Seconds from the start of the window — what the UI shows."""
        return float(self.t[frame] - self.t0)

    def at(self, stream: str, frame: int) -> int:
        """The index of `stream` at this display frame, or `NOTHING`."""
        return int(self.index[stream][frame])

    def frame_at(self, t: float) -> int:
        """The display frame nearest an absolute instant — the inverse of `t`.

        Nearest rather than zero-order hold, and the difference is the point.
        `_hold` answers "which record was in force", which is what a pane must
        show. This answers "where on the scrubber is this instant", which is
        what a mark on the timeline is asking, and rounding down there would
        land a click up to 33 ms early — reliably showing the frame *before* the
        one the mark refers to.

        A search, not `round((t - t0) * fps)`, because the grid is NOT uniform:
        it lands on the commanded deadlines and fills the gaps, so the arithmetic
        form is wrong by a whole frame near every replan and by more than that
        anywhere after one. `fps` is the filler rate, never a way to convert
        between a frame and an instant.

        Clamped, so an instant outside the window still yields a usable frame.
        Callers that care whether it was inside compare `t` against `t0`/`t1`
        themselves; this never reports a frame that does not exist.
        """
        i = int(np.searchsorted(self.t, t))
        if i <= 0:
            return 0
        if i >= self.n:
            return self.n - 1
        return i if (self.t[i] - t) < (t - self.t[i - 1]) else i - 1


def build(rollout, fps: float = DEFAULT_FPS,
          clock: str = "device") -> Timeline:
    """The table for one rollout.

    `clock` picks the time axis the STATE stream is placed on: `"device"` uses
    each record's production instant (`clock.estimate`), `"recv"` uses `t_recv`.
    Device is the default because a receive-side
    stall is not a fact about the robot and the viewer's whole job is comparing
    the measured pose against the commanded one; `"recv"` remains available
    because "when did P1 see this" is a real question, and it is the only answer
    for a record with no `tick`, which resolves to it silently.

    The window is `[max(first), min(last)]` across state and the cameras — the
    span over which *every* pane has something to show. A union window would
    open on a black camera pane or a robot with no pose, which reads as a broken
    viewer rather than as a stream that had not started.

    **Then it is cut to the ACTIVE edge**, so the viewer shows the rollout and
    not the machine getting ready for it. Everything before ACTIVE is setup:
    CHECK watching the streams settle, P0's homing ramp, P5 warming, and READY
    waiting on a human. On one recorded rollout that is 60 of the 80 seconds
    recorded — three quarters of the scrubber spent on an arm holding still at
    the home pose.

    The cut moves `t0` only; nothing is deleted and no index is rebased. Every
    `t` is still an absolute monotonic instant, so a frame can still be quoted
    straight back at the raw streams, and `skipped` says how much came before.

    It is skipped entirely when the record has no session log, when the run never reached ACTIVE, or when the edge falls
    outside the stream overlap — a run killed during homing has no rollout in it
    to show, and an empty timeline would be a worse answer than the whole
    recording.
    """
    # The state column's own time axis, and the window is built from IT: on the
    # device clock the first record sits ~1 ms before its arrival and the last
    # one likewise, and taking the window from `t_recv` while indexing on
    # `t_state` would leave the final frame extrapolating past the last sample.
    state_clock = clock_mod.estimate(rollout.state) if clock == "device" else None
    t_state = (state_clock.t if state_clock is not None
               else rollout.state["t_recv"])

    starts = [float(t_state[0])]
    ends = [float(t_state[-1])]
    for cam in rollout.cameras.values():
        starts.append(float(cam.index["t_recv"][0]))
        ends.append(float(cam.index["t_recv"][-1]))

    t0, t1 = max(starts), min(ends)
    if not t1 > t0:
        raise ValueError(
            "the state stream and the cameras do not overlap in time; there is "
            "no instant at which every pane has a record to show.")

    full_t0 = t0
    window = "streams"
    active = rollout.active_t
    # `active > t0` guards a record whose ACTIVE edge predates the window (P9
    # attached late); `active < t1` guards one that never got a frame after it.
    # Both keep the full window rather than producing a degenerate one.
    if active is not None and t0 < active < t1:
        t0, window = active, "active"

    # The anchors: the deadlines of the ticks that ACTUALLY executed. `k >= 1`
    # drops homing and the exit ramp, which are P0's joint ramps and carry no
    # commanded pose to be exact about — anchoring on them would pack ~90 frames
    # into the homing second and leave the rollout on filler.
    anchors = np.zeros(0, dtype=np.float64)
    bt = rollout.body_targets
    if len(bt):
        anchors = np.sort(np.asarray(
            bt["deadline"][bt["k"] >= 1], dtype=np.float64))

    grid, on_anchor = _fill(anchors, t0, t1, 1.0 / fps)
    n = len(grid)

    index = {"state": _hold(t_state, grid)}
    for name, cam in rollout.cameras.items():
        index[name] = _hold(cam.index["t_recv"], grid)

    # Within the window every stream has a record by construction, so a NOTHING
    # here would mean the window itself is wrong. Cheap to assert, and it turns
    # a silently-black pane into a load-time error.
    for name, column in index.items():
        if int(column.min()) == NOTHING:
            raise ValueError(
                f"stream {name!r} has no record at the window start — the "
                f"timeline window and the stream disagree.")

    # The commanded target in force at each instant. Held on `deadline`, which
    # is what P7 interpolates toward, so this is the same setpoint the servo was
    # driving at that moment.
    #
    # Exempt from the check above, and that exemption is still the point even
    # though the ACTIVE cut removed the biggest cause of it. `NOTHING` remains
    # correct at the head of the window: P0's last homing target is in force
    # when the phase flips, and P6 publishes nothing until the first chunk has
    # been inferred and blended. It is also correct for a record with no
    # `session.bin`, where the window still opens during homing.
    # A commanded overlay that had an answer everywhere would be inventing one.
    if len(rollout.body_targets):
        index["command"] = _hold(rollout.body_targets["deadline"], grid)

    # The measured pose is interpolated, not held — see the module docstring.
    # One clock for both, because `grip_meas` rides in the same record
    # (`t_grip_recv` says when the HAND last spoke, which is a different
    # question and is not what indexes this array). On the device clock that
    # carries the gripper reading to the arm sample's production instant, which
    # is right to within the two DDS topics' relative latency — the hand topic
    # has no device clock of its own to do better with.
    q = _lerp(t_state, rollout.state["q"], grid)
    grip = _lerp(t_state, rollout.state["grip_meas"], grid)

    return Timeline(fps=fps, t0=t0, t1=t1, t=grid, index=index,
                    on_anchor=on_anchor, q=q, grip=grip,
                    skipped=t0 - full_t0, window=window,
                    state_clock=state_clock)


def describe(timeline: Timeline, rollout) -> dict:
    """What the UI needs to know about the grid it is scrubbing."""
    cams = {}
    for name, cam in rollout.cameras.items():
        col = timeline.index[name]
        used = np.unique(col)
        cams[name] = {
            "frames_recorded": int(len(cam.index)),
            "frames_shown": int(len(used)),
            # How often a display frame repeats the previous camera frame.
            # Non-zero is normal at 30 fps against a 30.01 Hz camera.
            "repeats": int(np.count_nonzero(np.diff(col) == 0)),
            "dropped_reported": cam.dropped,
            # `dropped` in the manifest counts every frame published before P9
            # attached as a loss. The honest number is the interior gaps.
            "dropped_interior": int(
                cam.index["frame_id"][-1] - cam.index["frame_id"][0] + 1
                - len(cam.index)),
        }

    disagree = 0
    # The OBSERVED cameras, not every recorded one. This asks whether the two
    # views the policy was shown stepped together, so a record-only camera is
    # not one of its subjects — and counting it would have silently retired the
    # diagnostic the moment the head started being recorded, because the test
    # was "exactly two cameras" and there are now three.
    observed = [n for n, cam in rollout.cameras.items() if cam.used]
    if len(observed) == 2:
        a, b = (timeline.index[n] for n in observed)
        # Compare progress rather than raw index: the two cameras start at
        # different frame_ids, so equal indices is the meaningful test only
        # after both are referenced to their own first shown frame.
        disagree = int(np.count_nonzero((a - a[0]) != (b - b[0])))

    # How stale the commanded overlay is on the frames that show one. On the
    # anchors this is exactly zero, which is the entire reason the grid is built
    # the way it is; the residue is the filler, where the overlay is honestly
    # holding the last thing the policy asked for.
    stale = None
    col = timeline.index.get("command")
    if col is not None:
        bt = rollout.body_targets
        # `k >= 1` because the hold also resolves to P0's homing targets on the
        # frames before the policy's first tick. Those carry a NaN pose and the
        # overlay hides them, so counting their age would report ~60 s of
        # staleness for frames that never draw a ghost at all.
        shown = (col >= 0) & (bt["k"][np.clip(col, 0, None)] >= 1)
        if shown.any():
            age = (timeline.t[shown] - bt["deadline"][col[shown]]) * 1000.0
            stale = {
                "mean_ms": round(float(age.mean()), 2),
                "max_ms": round(float(age.max()), 2),
                "exact_frames": int((age <= 1e-9).sum()),
            }

    # Which clock the measured pose is on, and what it cost to put it there.
    # The UI states this: "the arm lagged the command by 0.1 rad" and "the
    # telemetry arrived 300 ms late" look identical on screen and are different
    # facts, and a reader who is not told which axis they are on cannot tell
    # them apart. `None` for a record with no `tick`, which is itself the
    # answer — that rollout can only be shown on receive time.
    state_clock = (timeline.state_clock.describe()
                   if timeline.state_clock is not None else None)

    return {
        "state_clock": state_clock,
        # The FILLER rate, not a frame-to-time conversion: the grid lands on the
        # deadlines and is not uniform. `t_rel` is the only way to ask when a
        # frame is, and the browser uses it for exactly that.
        "fps": timeline.fps,
        "t_rel": [round(float(x), 5) for x in (timeline.t - timeline.t0)],
        "on_anchor": [int(b) for b in timeline.on_anchor],
        "anchor_frames": int(timeline.on_anchor.sum()),
        "filler_frames": int(timeline.n - timeline.on_anchor.sum()),
        "command_staleness": stale,
        "n_frames": timeline.n,
        "duration_s": round(timeline.duration, 3),
        "t0": timeline.t0,
        # What the viewer trimmed off the front, and why. The UI shows both:
        # a 20 s scrubber over an 80 s recording is a fact the reader has to be
        # told, or the rollout simply looks shorter than it was.
        "skipped_s": round(timeline.skipped, 3),
        "window": timeline.window,
        "cameras": cams,
        "camera_phase_disagreements": disagree,
    }
