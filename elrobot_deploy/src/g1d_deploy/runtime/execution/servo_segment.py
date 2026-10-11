"""The segment walk P7 and P8 share: which target is in flight, and when to
abandon it.

Both servos do the same four things every iteration — notice a takeover, pick
the next target once the current segment has expired, hold frozen when there is
none, and otherwise say how far along the segment they are — so they share
this one implementation. Do not fork it: homing, rollouts, interventions and
the exit ramp are all executed by this walk, which is what makes them safe
(`docs/infra_logic/500hz_execution.md`, "What is not pluggable, and why"). The
parts that genuinely differ are outside this class: what is interpolated (17 joints or a
`(2, 6)` command), what is clamped, and what is published.

`Segment` therefore owns no wire and publishes nothing. It answers one question
per iteration, `phase()`, and the servo acts on the answer.
"""

from __future__ import annotations

import enum

import numpy as np

#: The floor on a segment's span. A zero-length segment is not an error — P6
#: may legitimately write a tick whose deadline has already arrived — and `eps`
#: turns it into "jump to the end, clamped by the slew limit" instead of a
#: division by zero.
EPS_SPAN = 1e-9


class Phase(enum.Enum):
    """What `Segment.phase` found at this instant."""

    #: No selectable target: publish the last command frozen and wait.
    HOLD = "hold"
    #: A new segment was just latched: publish nothing, do not wait, iterate
    #: again immediately against it (no servo frame is dropped).
    ADVANCED = "advanced"
    #: Inside a live segment: interpolate and publish.
    INSIDE = "inside"


class Segment:
    """One servo's view of the target ring, walked one segment at a time.

    `x_start` / `x_end` are whatever the ring's payload column holds for this
    reader; the class never looks inside them. `cursor` says whether latching a
    target publishes `control.k_exec` — P7 does, P8 deliberately does not,
    because it sits in the same segment as P7 by construction and a second
    cursor would be a second way to be wrong about the same fact.
    """

    def __init__(self, ring, control, x0: np.ndarray, t0: float, *,
                 cursor: bool):
        self.ring = ring
        self.control = control
        self.cursor = cursor

        # A servo opens stalled, holding the MEASURED state, with the
        # ring empty. `t_end = -inf` makes the very first iteration take the
        # advance branch unconditionally; `stalled = True` with the segment
        # start already set means that even if a target somehow exists at the
        # first iteration, the "leaving a stall" branch has a real `t_start` to
        # spread the motion from.
        self.stalled = True
        self.t_start = t0
        self.x_start = x0
        self.t_end = -np.inf
        self.x_end = x0

        # The intervention counter as this loop last saw it. Seeded
        # from the LIVE word rather than from zero, so a servo spawned by a
        # reset into a session where somebody has already intervened does not
        # read the value it was born with as a change.
        self._intervene = control.intervene

    def phase(self, t_now: float, x_last: np.ndarray) -> Phase:
        """Advance the walk to `t_now`. `x_last` is what the servo published
        last, which is where any hold or preempt starts from."""
        # THE PREEMPT. The only thing a servo is told about an
        # intervention, and it is deliberately not phrased as one: the counter
        # changing means "every target you have latched belongs to a world that
        # no longer exists", which is a statement about the ring and not about
        # humans. The servo still knows nothing — it never reads the parity and
        # cannot tell a takeover from a handback, because it does not need to.
        #
        # Without it, no ring write of any kind can preempt a segment in flight.
        # The advance branch below is gated on `t_now > t_end`, and those two
        # are LOCAL state: a policy target latched microseconds before the
        # operator squeezed the trigger is driven to completion, so up to a
        # whole action tick of policy motion happens after a human asked for it
        # to stop. That is the one window a human cannot be asked to tolerate.
        #
        # `stalled = True` is the load-bearing line and the reason this is four
        # assignments rather than one. Without it the advance branch takes its
        # else-clause and sets `x_start = x_end` — the very target being
        # cancelled — so the command JUMPS to it and interpolates onward from
        # there. Setting it routes the next iteration through the "leaving a
        # stall" branch, which keeps the `(t_now, x_last)` start written here.
        # That is exactly what `stalled` already means; nothing new is
        # introduced, only a second way to enter it.
        intervene = self.control.intervene
        if intervene != self._intervene:
            self._intervene = intervene
            self.stalled = True
            self.t_start = t_now
            self.x_start = x_last
            self.t_end = -np.inf            # force the advance branch below

        if t_now <= self.t_end:
            return Phase.INSIDE

        nxt = self.ring.next_after(t_now, self._intervene)
        if nxt is None:
            # THE STALL. Three things, and all three are load
            # bearing:
            #
            #   1. `t_start` is pushed forward to now, every iteration. That is
            #      the entire point: when the target finally lands the segment
            #      is [now, deadline] and the motion is spread over the time
            #      that ACTUALLY remains, not over a span that expired while
            #      the policy was thinking.
            #   2. The servo republishes `x_last` FROZEN — not the measurement.
            #      Publishing the measurement would make it the next `x_last`,
            #      and every stalled iteration would absorb a little more
            #      tracking error, ratcheting the command toward wherever the
            #      joint has sagged (or the jaw has slipped).
            #   3. `k_exec` is NOT touched. It names the tick being executed
            #      toward, and during a stall that is still the last target
            #      selected.
            self.stalled = True
            self.t_start = t_now
            self.x_start = x_last
            return Phase.HOLD

        if self.stalled:
            # Leaving a stall: start/end were set by the last stalled
            # iteration, microseconds ago. Overwriting them from `t_end` here
            # would rewind the start to the expired deadline and demand the
            # whole delta in whatever is left.
            self.stalled = False
        else:
            self.t_start = self.t_end
            self.x_start = self.x_end

        if self.cursor:
            self.control.k_exec = nxt.k
        self.t_end = nxt.deadline
        self.x_end = nxt.payload
        return Phase.ADVANCED

    @property
    def world(self) -> int:
        """The intervention counter this walk selects targets from."""
        return self._intervene

    # --- inside a segment ----------------------------------------------------

    @property
    def span(self) -> float:
        return max(EPS_SPAN, self.t_end - self.t_start)

    @property
    def delta(self) -> np.ndarray:
        return self.x_end - self.x_start

    def u_raw(self, t_now: float) -> float:
        """Where `t_now` sits on the segment, UNCLAMPED. Negative on an
        inverted span; the servo decides what that means for its velocity."""
        return (t_now - self.t_start) / self.span

    def at(self, u_raw: float) -> np.ndarray:
        """The command at fraction `u_raw`, clamped to the segment's ends."""
        u = min(1.0, max(0.0, u_raw))
        return self.x_start + u * self.delta


__all__ = ["EPS_SPAN", "Phase", "Segment"]
