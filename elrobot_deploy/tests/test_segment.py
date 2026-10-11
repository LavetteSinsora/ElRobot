"""`processes.segment.Segment` — the walk P7 and P8 share.

The servo tests prove each loop's behaviour end to end; these pin the walk
itself, once, and the one thing that distinguishes the two users of it.
"""

from __future__ import annotations

import numpy as np

from g1d_deploy.runtime.shared_memory import schema
from g1d_deploy.runtime.execution.servo_segment import Phase, Segment

BODY = schema.BODY_DOF
T0 = 100.0


def _seg(rings, cursor=True, x0=None):
    x0 = np.zeros(BODY) if x0 is None else x0
    return Segment(rings["body"], rings["control"], x0, T0, cursor=cursor)


def test_it_opens_stalled_and_holds_on_an_empty_ring(rings):
    seg = _seg(rings)
    assert seg.stalled and seg.t_end == -np.inf
    assert seg.phase(T0 + 0.01, np.zeros(BODY)) is Phase.HOLD
    # The stall pushes the start forward every iteration (the stall).
    assert seg.t_start == T0 + 0.01


def test_a_hold_leaves_the_cursor_alone(rings):
    """`k_exec` monotonicity: `k_exec` names the last target selected, and a stall selects
    nothing."""
    rings["control"].k_exec = 7
    seg = _seg(rings)
    seg.phase(T0 + 0.01, np.zeros(BODY))
    assert rings["control"].k_exec == 7


def test_only_the_cursor_owner_publishes_k_exec(rings):
    rings["targets"].write(3, T0 + 1.0, np.ones(BODY), intervene=0)
    rings["control"].k_exec = 0
    quiet = _seg(rings, cursor=False)
    assert quiet.phase(T0 + 0.1, np.zeros(BODY)) is Phase.ADVANCED
    assert rings["control"].k_exec == 0, "P8 publishes no cursor"
    loud = _seg(rings, cursor=True)
    assert loud.phase(T0 + 0.1, np.zeros(BODY)) is Phase.ADVANCED
    assert rings["control"].k_exec == 3


def test_leaving_a_stall_keeps_the_stalls_start(rings):
    """The motion is spread over the time that ACTUALLY remains: the start is
    the instant the last stalled iteration wrote, not the expired deadline."""
    seg = _seg(rings)
    seg.phase(T0 + 0.5, np.zeros(BODY))               # stalled at 100.5
    rings["targets"].write(1, T0 + 1.0, np.ones(BODY), intervene=0)
    assert seg.phase(T0 + 0.6, np.zeros(BODY)) is Phase.ADVANCED
    assert seg.t_start == T0 + 0.5 and seg.t_end == T0 + 1.0
    assert not seg.stalled
    assert seg.phase(T0 + 0.75, np.zeros(BODY)) is Phase.INSIDE
    assert seg.u_raw(T0 + 0.75) == 0.5
    assert np.allclose(seg.at(0.5), np.full(BODY, 0.5))


def test_the_next_segment_starts_where_the_last_one_ended(rings):
    seg = _seg(rings)
    rings["targets"].write(1, T0 + 1.0, np.ones(BODY), intervene=0)
    rings["targets"].write(2, T0 + 2.0, np.full(BODY, 2.0), intervene=0)
    seg.phase(T0 + 0.1, np.zeros(BODY))
    assert seg.phase(T0 + 1.5, np.ones(BODY)) is Phase.ADVANCED
    assert seg.t_start == T0 + 1.0 and np.all(seg.x_start == 1.0)
    assert seg.t_end == T0 + 2.0


def test_the_counter_moving_abandons_the_segment_in_flight(rings):
    """The preempt is a second way INTO the stall, from `x_last`, so
    the command does not jump to the target being cancelled."""
    seg = _seg(rings)
    rings["targets"].write(1, T0 + 1.0, np.ones(BODY), intervene=0)
    seg.phase(T0 + 0.1, np.zeros(BODY))
    assert seg.phase(T0 + 0.5, np.zeros(BODY)) is Phase.INSIDE
    halfway = np.full(BODY, 0.4)
    rings["control"].intervene = 1
    assert seg.phase(T0 + 0.6, halfway) is Phase.HOLD
    assert seg.stalled and np.all(seg.x_start == halfway)
    assert seg.t_start == T0 + 0.6
    # The old world's target is unreachable now, and a new one is selected.
    rings["targets"].write(2, T0 + 0.7, np.full(BODY, 0.9), intervene=1)
    assert seg.phase(T0 + 0.61, halfway) is Phase.ADVANCED
    assert np.all(seg.x_start == halfway) and seg.t_end == T0 + 0.7


def test_an_inverted_span_collapses_to_eps_not_a_division_by_zero(rings):
    seg = _seg(rings)
    seg.stalled = False
    seg.t_start, seg.t_end = 1.0, 0.5
    seg.x_start, seg.x_end = np.zeros(BODY), np.ones(BODY)
    assert seg.span > 0.0
    assert seg.u_raw(0.4) < 0.0
    assert np.all(seg.at(seg.u_raw(0.4)) == 0.0), "pinned at the start"
