"""The anchor rule, and the only implementation of it in the system.

Called by P5 once per inference (`runtime/processes/p5_policy.py`), and by the
inspection dashboard's replay to rebuild the same observation from a record
(`dashboards/inspect/modes/replay/anchors.py`). Reads the camera rings and the
state ring; writes nothing. Design: `docs/overview.md`, "One chunk's life", and
`docs/deploying_a_policy/policy_contract.md`, "The observation (`obs`)".

*Which instant* an observation depicts is the infra's, never the policy's:
`observation.observe` builds the policy-facing `Observation` out of the
`Anchor` this returns. The rule and the deadline shift (`deadline_shift.py`)
define together what "action `j` is due `(j+1)/control_hz` after `t_anchor`"
means, so neither is pluggable; change one only with the other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --- the anchor ---------------------------------------------------------------


@dataclass(frozen=True)
class Anchor:
    """One instant, by construction.

    `t_anchor` — the `t_img_recv` of the frame chosen from the ANCHOR camera
    (`Policy.anchor_camera`) — is the instant the world was observed, and the
    joints are matched to it so that the image and the flange pose describe the
    same moment. That pairing is the whole reason an anchor camera exists: the
    acting camera is mounted on the arm, and a policy composes its actions onto
    `obs.state.flange`. Skew between the two is a geometric error of (arm speed x skew) — measured on the reference robot
    at half a frame period: p99 5.4 mm, max 13.6 mm, up to 3.3 degrees.

    `t_trigger` is a fact about the SCHEDULE; `t_anchor` is a fact about the
    WORLD. They are usually within a few ms and are never the same thing.
    """

    t_trigger: float
    t_anchor: float                     # the primary frame's t_img_recv
    frames: dict[str, np.ndarray]       # per camera, the whole camera-ring record
    state: np.ndarray                   # the state-ring record nearest t_anchor

    def max_image_latency(self) -> float:
        """How far the worst camera's capture time is from `t_anchor`, either
        way, in seconds. How late (or early) the images are against the instant
        the observation is pinned to.

        **Unsigned, and that is the point.** A signed `max(t_anchor - t_C)`
        would be swallowed by the primary's own zero term whenever every other
        camera ran *ahead* of the anchor — a camera 8 ms out of step in that
        direction would report a spread of exactly 0.0. The question this
        answers is "do these images depict the same instant", and that has no
        preferred direction.

        Nothing is lost by dropping the sign, because the two directions are
        bounded differently. A camera ahead of the anchor is capped at
        `(T_primary + T_C)/2` by the nearest-frame rule; a camera *behind* it is
        unbounded once the anchor's wait cap fires (`acquire_anchor`). So a value above that
        bound can still only mean lagging, which is the fault case.

        This is also the only signal that a camera has stopped producing: the
        wait is capped, so a dead camera no longer hangs the run — it quietly
        contributes the same frozen image forever, and this is what makes that
        visible instead of silent.
        """
        return max(abs(self.t_anchor - float(f["t_recv"]))
                   for f in self.frames.values())


class _FrameTracker:
    """Keeps one camera's frame nearest a key instant.

    Copy-on-change, not copy-on-decide: the tracker reads only the NEWEST frame
    of a camera ring (`peek` / `read_into`), and a ring with no declared
    history holds just three slots — the frame just before the key is
    overwritten two frames after a later one has been published. (A ring with
    history keeps it longer, but the rule must not depend on how deep a ring a
    policy happened to declare, so it is the same code for both.) Over one
    acquisition window — at most a camera period — a 30 Hz camera
    publishes once or twice, so this is one or two copies per inference rather
    than thirty per second.

    Two scratch buffers, alternating, because the best-so-far must survive the
    capture that is being compared against it.
    """

    __slots__ = ("region", "_buf", "_i", "_last_id", "_after", "frame", "t")

    def __init__(self, region):
        self.region = region
        self._buf = [region.empty(), region.empty()]
        self._i = 0
        self._last_id = 0
        self._after = False
        self.frame: np.ndarray | None = None
        self.t = float("nan")

    def capture(self) -> np.ndarray | None:
        """The published frame, if it is one we have not taken yet.

        `peek` is two scalar loads; the megabyte is copied only when the frame
        is new. Metadata is read back out of the *copy* (the camera record keeps
        it inside the buffer) because the published index may have moved on since `peek`.
        """
        _, frame_id = self.region.peek()
        if frame_id == 0 or frame_id == self._last_id:
            return None
        buf = self._buf[self._i]
        if not self.region.read_into(buf):
            return None                 # recycled mid-copy; retry next poll
        self._last_id = int(buf["frame_id"])
        self._i ^= 1
        return buf

    def keep(self, buf: np.ndarray) -> None:
        self.frame, self.t = buf, float(buf["t_recv"])

    def offer(self, buf: np.ndarray, t_key: float) -> None:
        """Keep whichever of (best so far, this one) is nearer `t_key`."""
        t = float(buf["t_recv"])
        if t >= t_key:
            self._after = True
        if self.frame is None or abs(t - t_key) < abs(self.t - t_key):
            self.keep(buf)

    def settled(self, t_key: float, half_period: float) -> bool:
        """Can the search stop? Two ways, and the second is the whole point.

        1. A frame at or after the key has been seen. Frames arrive in
           increasing `t_recv`, so every further frame is further from the key.

        2. **The frame in hand is already inside half a period.** Frames come
           `T` apart, so a held frame `age` before the key means the next one
           lands `T - age` after it; the held one is nearer exactly when
           `age < T/2`. Waiting could then only confirm what is already known.

        Rule 2 is an early exit on a search whose answer does not change, not a
        different answer. Before it existed this blocked for a full period on
        every frame that landed a fraction of a millisecond on the wrong side of
        the key — measured on recorded rollouts: 21 of 36 replans threw
        away a frame less than 2 ms old and waited 34 ms for the next, then
        picked the old one anyway. Correlation between the discarded frame's age
        and the wait it caused: -0.993.
        """
        if self.frame is None:
            return False
        return self._after or (t_key - self.t) <= half_period


def acquire_anchor(cameras: dict, state, primary: str, t_trigger: float,
                   half_period: dict, *, now=None) -> Anchor:
    """The anchor rule, and the only implementation of it in the system.

    Every camera takes the frame whose `t_img_recv` is **nearest `t_trigger`**,
    and the primary's choice becomes `t_anchor`, which the joints are then
    matched to. One rule for every stream, so a rig with an exo camera, a head
    and two wrists needs no special case — and the primary stays a real role,
    because it is the view the action frame is expressed in and the one whose
    instant the flange must pair with.

    **`cameras` is the OBSERVED subset, not the arena's whole camera set.** P5
    passes `cfg.observed`, because everything this function does — the wait, the
    frames it returns, the spread `max_image_latency` reports — is about the
    observation, and a camera the run only records takes no part in one. That is
    also what keeps the last paragraph below from being a liability: a camera
    that never publishes blocks this wait forever, and a record-only camera must
    not be able to do that to a run that does not use it.

    **This does not guarantee the observation is newer than the trigger**, and
    that is a deliberate trade. Blocking until the primary produces a frame
    *after* `t_trigger` costs up to a full period per replan and, with a second
    camera, up to two — measured mean 31 ms, max 66 ms, a third of the whole
    replan budget. Under the nearest rule the observation is at most `T/2`
    older than the trigger (~4 ms of mean observation age) and the cycle is
    that much shorter. In sync that is clearly the right side of the trade:
    cycle time is what stretches the executed trajectory (measured: to 26.2 Hz
    against a 30 Hz policy, the arm frozen 12.6% of the run). In async, where
    chunks overlap and cycle time is decoupled from execution, the trade
    matters less.

    **The cap.** Waiting stops at `t_trigger + max(T/2)` and the frames in hand
    are used. In normal operation it cannot fire: a camera that needs to wait
    waits `T - age < T/2`, so the deadline is always slack. It exists for a
    camera that has stopped, which would otherwise hang the run forever. The
    cost is that a dead camera is now silent instead of obvious, which is why
    `Anchor.max_image_latency()` exists and why P5 prints it every chunk.

    A camera that has produced NO frame at all is not covered by the cap — there
    is no image to fall back on, so that still blocks. CHECK has already proven
    every declared camera produces, so this is the decoder dying mid-run.

    The waits below spin rather than sleep, for the same reason `sleep_until`
    spins its last 1.5 ms: P5 owns its core and the scheduler's millisecond of
    slop is a large fraction of what this whole wait now costs.
    """
    if now is None:
        from g1d_deploy.runtime.timing import now
    trackers = {name: _FrameTracker(region) for name, region in cameras.items()}

    # Snapshot every camera BEFORE waiting on any of them. The tracker reads
    # only the newest frame, so the frame in hand at `t_trigger` is out of its
    # reach the moment a later one lands — and for a camera that is already
    # inside half a period, that is exactly the frame being kept.
    for tr in trackers.values():
        buf = tr.capture()
        if buf is not None:
            tr.offer(buf, t_trigger)

    def done(name, tr):
        return tr.settled(t_trigger, half_period[name])

    deadline = t_trigger + max(half_period.values())
    while not all(done(n, tr) for n, tr in trackers.items()):
        # The cap, but never at the price of an anchor with no image in it.
        if now() >= deadline and all(tr.frame is not None
                                     for tr in trackers.values()):
            break
        for name, tr in trackers.items():
            if not done(name, tr):
                buf = tr.capture()
                if buf is not None:
                    tr.offer(buf, t_trigger)

    t_anchor = trackers[primary].t

    # The state sample *nearest* t_anchor needs both of its neighbours to exist:
    # picking the nearest out of samples that all precede it silently returns
    # whatever the ring happens to end at. P1 runs ~35x faster than a camera, so
    # this is satisfied on the first read whenever the anchor is in the past —
    # which, under the nearest rule, is the common case.
    while True:
        latest = state.latest()
        if latest is not None and float(latest["t_recv"]) >= t_anchor:
            break

    return Anchor(t_trigger=t_trigger, t_anchor=t_anchor,
                  frames={n: tr.frame for n, tr in trackers.items()},
                  state=state.nearest(t_anchor))


__all__ = ["Anchor", "acquire_anchor"]
