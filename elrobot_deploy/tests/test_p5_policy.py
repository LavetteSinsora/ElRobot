"""P5 and the policy surface: the scheduling loop, the action contract, the
observation and the history reader.

The failures worth catching here are the ones that produce a plausible-looking
run: a window scan that reads the latest tick instead of the earliest and so
schedules an inference that is already late; a `newest_chunk_id` published one
line too early, which P6 would follow into a half-written slot; a sync chunk one
action too long, which silently moves `t_next_inference` into the past; a frame
handed over in BGR; a deadline shift with its sign flipped.
"""

from __future__ import annotations

import os
import uuid
from contextlib import contextmanager

import numpy as np
import pytest

from elrobot_deploy.config_loading import resolved as cfgmod
from elrobot_deploy.config_loading.derive import derive
from elrobot_deploy import policy_contract as policymod
from elrobot_deploy.runtime.processes import p5_policy as p5mod
from elrobot_deploy.runtime.shared_memory import Arena, build, open_all, schema
from elrobot_deploy.runtime.processes.p5_policy import PolicyLoop, apply_policy_env, warmup
from elrobot_deploy.policy_contract import (Anchor, Chunk, ChunkError, EefActions,
                                        JointActions, Limits, Policy, acquire_anchor,
                                        validate_actions, validate_chunk)

T0 = 1000.0                     # an arbitrary epoch on the monotonic clock


# --- doubles ----------------------------------------------------------------


class FakeClock:
    """`now` only moves when something sleeps, so a test can run the whole
    schedule in no time and still assert on what P5 asked to wait for."""

    def __init__(self, t: float = T0):
        self.t = t
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep_until(self, t: float, **_) -> float:
        self.sleeps.append(t)
        self.t = max(self.t, t)
        return self.t


def _actions(n: int, value: float = 1.0) -> EefActions:
    """`n` eef actions whose x reads `value + row/100`, so a row is traceable."""
    pose = np.zeros((n, 2, 7))
    pose[..., 0] = value + np.arange(n)[:, None] / 100.0
    pose[..., 3] = 1.0                          # identity, wxyz
    return EefActions(pose=pose, hand=np.full((n, 2), abs(value)))


def _chunk(n: int, value: float = 1.0) -> Chunk:
    a = _actions(n, value)
    hand = np.zeros((n, 2, 6))
    hand[:, :, 0] = a.hand
    return Chunk(hand=hand, pose=a.pose)


class StubPolicy(Policy):
    """A policy that records what P5 asked of it and returns fixed actions.

    `deadline_shift` is off: the arena these tests build publishes its frames
    far in the future so the anchor never waits, and a shift computed off that
    would be the clipped maximum. The shift has its own tests below.
    """

    action_mode = "eef"
    end_effector = "dex1"
    anchor_camera = "rwrist"
    deadline_shift = False

    def __init__(self, cfg, *, n: int | None = None):
        super().__init__(None, cfg.prompt, None)
        self.cfg = cfg
        self.n = cfg.H if n is None else n
        self.calls: list[tuple] = []
        self.seen: list = []

    def infer(self, obs, reader):
        self.calls.append(("infer", obs.t_trigger))
        self.seen.append(obs)
        return _actions(self.n, 1.0)


class ProbeLoop(PolicyLoop):
    """`write` is the real one, bracketed: the point of several of these tests
    is the order of the two stores it sits between."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.writes: list[dict] = []
        self.during_write: list[tuple[int, bool]] = []

    def write(self, chunk, **meta):
        self.writes.append(dict(meta, n=chunk.n))
        k = meta["k_first_action"]
        # `adopted_through=None` on purpose: this probe exists to observe the
        # raw publish window — payload visible, id not yet stored — which is the
        # one thing a control caller must never do.
        self.during_write.append(
            (self.control.newest_chunk_id,
             self.chunks.covers_any(k, adopted_through=None)))
        super().write(chunk, **meta)
        self.during_write.append(
            (self.control.newest_chunk_id,
             self.chunks.covers_any(k, adopted_through=None)))
        if getattr(self, "on_write", None) is not None:
            self.on_write(self)


@contextmanager
def _arena_for(cfg, hw):
    layout = build(cfg, derive(cfg, hw))
    arena = Arena.create(f"elr-t{os.getpid()}-{uuid.uuid4().hex[:8]}", layout)
    try:
        rings = open_all(arena, cfg)
        # P5 acquires the anchor itself, so every loop test needs the frames
        # and the state sample a live P1/P2 would have published. Dated far in
        # the future so any trigger instant the test picks is satisfied
        # immediately — the wait itself is tested separately, below.
        for region in rings["cameras"].values():
            _publish(region, T0 + 1e6)
        _state(rings["state"], T0 + 1e6)
        yield rings
    finally:
        arena.close()
        arena.unlink()


def _fill_epoch(body, ks, cfg, *, t_base: float = T0, k_base: int = 0):
    """Write the given ticks under one epoch: `deadline(k) = t_base + k·dt`."""
    for k in ks:
        body.write(int(k), t_base + (int(k) - k_base) * cfg.dt,
                   np.zeros(schema.BODY_DOF), intervene=0)


def _loop(cfg, rings, *, clock=None, policy=None, hw=None, **kw):
    policy = policy or StubPolicy(cfg, **kw)
    return ProbeLoop(cfg, rings, policy, hw=hw, clock=clock or FakeClock()), policy


# --- what the policy observes vs. what the rig records ------------


def test_a_recorded_camera_the_policy_does_not_observe_is_not_anchored(cfg, hw):
    """`cameras` is what EXISTS; the observed set is what is USED.

    A camera can be recorded on every rollout and read by nothing. If it
    stayed in the anchor's camera set it would wait the anchor, count toward
    `max_image_latency`, and — the case that matters — hang the run outright if
    it never published, because a camera with no frame at all is deliberately
    outside `acquire_anchor`'s wait cap.
    """
    watched = cfgmod.replace(cfg, observed_cameras=("rwrist",))
    with _arena_for(watched, hw) as rings:
        loop, _ = _loop(watched, rings)
        assert set(rings["cameras"]) == {"rwrist", "head"}
        assert set(loop.cameras) == {"rwrist"}
        assert set(loop.half_period) == {"rwrist"}


def test_uses_cameras_narrows_the_observed_set_over_the_config(cfg, hw):
    """A policy that declares `uses_cameras` is anchored on exactly those,
    whatever the run observes."""
    class Narrow(StubPolicy):
        uses_cameras = ("rwrist",)

    with _arena_for(cfg, hw) as rings:
        loop, _ = _loop(cfg, rings, policy=Narrow(cfg))
        assert set(loop.cameras) == {"rwrist"}


def test_the_anchor_ignores_a_record_only_camera_that_never_published(cfg, hw):
    """The failure the observed set exists to prevent, driven rather than
    described: only the observed camera ever publishes, `head` stays silent
    exactly as a camera whose decoder died would."""
    watched = cfgmod.replace(cfg, observed_cameras=("rwrist",))
    layout = build(watched, derive(watched, hw))
    arena = Arena.create(f"elr-t{os.getpid()}-{uuid.uuid4().hex[:8]}", layout)
    try:
        rings = open_all(arena, watched)
        _publish(rings["cameras"]["rwrist"], T0)
        _state(rings["state"], T0)
        loop, _ = _loop(watched, rings)

        anchor = acquire_anchor(loop.cameras, rings["state"],
                                loop.anchor_camera, T0,
                                loop.half_period, now=lambda: T0 + 10.0)
        assert set(anchor.frames) == {"rwrist"}
        assert anchor.t_anchor == T0
        assert anchor.max_image_latency() == 0.0
    finally:
        arena.close()
        arena.unlink()


def test_an_unset_observed_set_still_means_every_camera(cfg, hw):
    assert cfg.observed_cameras == ()
    assert cfg.observed == ("head", "rwrist")
    with _arena_for(cfg, hw) as rings:
        loop, _ = _loop(cfg, rings)
        assert set(loop.cameras) == {"rwrist", "head"}


# --- The window scan --------------------------------------------


def test_window_scan_takes_the_earliest_written_tick(cfg, hw):
    """P6 writes in increasing k, so the earliest is the first to exist. Taking
    the latest would mean waiting until the chunk is nearly exhausted — by
    which time t_next_inference may already have passed."""
    wide = cfgmod.replace(cfg, replan_period_ticks=6, action_horizon=12)
    with _arena_for(wide, hw) as rings:
        _fill_epoch(rings["body"], range(2, 7), wide)
        loop, _ = _loop(wide, rings)
        assert loop.scan().k == 2


def test_every_hole_in_the_window_yields_the_same_instant(cfg, hw):
    """`deadline` is affine in `k` within one epoch, so any solved tick in
    [k_start_read, k_next_inference] extrapolates to the same wake time."""
    wide = cfgmod.replace(cfg, replan_period_ticks=6, action_horizon=12)
    with _arena_for(wide, hw) as rings:
        body = rings["body"]
        scheduled = []
        for present in ([2, 3, 4, 5, 6], [3, 5], [6], [2], [4, 6], [5]):
            body.clear()
            _fill_epoch(body, present, wide)
            loop, _ = _loop(wide, rings)
            entry = loop.scan()
            assert entry.k == min(present)          # the earliest, every time
            scheduled.append(loop.schedule(entry))
        want = T0 + 6 * wide.dt                     # deadline(k_next_inference)
        assert max(abs(t - want) for t in scheduled) < 1e-12


def test_an_empty_window_retries_and_advances_nothing(cfg, hw):
    """A persistently empty window blocks P5 forever, on purpose."""
    with _arena_for(cfg, hw) as rings:
        loop, policy = _loop(cfg, rings)
        assert loop.step() is None
        assert (loop.k_start_read, loop.k_next_inference) == (2, cfg.P)
        assert rings["control"].newest_chunk_id == 0
        assert policy.calls == []


# --- one iteration ------------------------------------------------


def test_step_sleeps_until_the_extrapolated_instant(cfg, hw):
    with _arena_for(cfg, hw) as rings:
        _fill_epoch(rings["body"], [2], cfg)
        clock = FakeClock()
        loop, policy = _loop(cfg, rings, clock=clock)
        loop.step()
        want = T0 + cfg.P * cfg.dt
        assert clock.sleeps == [pytest.approx(want, abs=1e-12)]
        # The trigger instant P5 slept to is also the one the observation is
        # anchored at and the one stamped into the chunk.
        assert policy.calls[0] == ("infer", pytest.approx(want, abs=1e-12))


def test_k_first_action_is_one_past_the_trigger_tick(cfg, hw):
    """P6 starts at k_exec + 2, so a chunk's action 0 is never
    executed. The +1 is deliberate and is where that price is paid."""
    with _arena_for(cfg, hw) as rings:
        _fill_epoch(rings["body"], [2, 3], cfg)
        loop, _ = _loop(cfg, rings)
        loop.step()
        meta = loop.writes[-1]
        assert meta["k_first_action"] == cfg.P + 1
        assert meta["chunk_id"] == 1
        assert rings["chunks"].covers_any(cfg.P + 1, adopted_through=None)
        assert not rings["chunks"].covers_any(cfg.P, adopted_through=None)


def test_newest_chunk_id_becomes_visible_only_after_the_payload(cfg, hw):
    """P6 reads nothing but this word to learn a chunk exists; if it
    became visible first, P6 would read a slot mid-write."""
    with _arena_for(cfg, hw) as rings:
        _fill_epoch(rings["body"], [2], cfg)
        loop, _ = _loop(cfg, rings)
        loop.step()
        before, after = loop.during_write
        assert before == (0, False)     # no payload, no id
        assert after == (0, True)       # payload readable, id still old
        assert rings["control"].newest_chunk_id == 1


def test_cursors_advance_by_exactly_one_replan_period(cfg, hw):
    with _arena_for(cfg, hw) as rings:
        _fill_epoch(rings["body"], range(2, 3 * cfg.P), cfg)
        loop, _ = _loop(cfg, rings)
        for i in range(1, 3):
            assert loop.step() == i
            assert loop.k_start_read == 2 + i * cfg.P
            assert loop.k_next_inference == (i + 1) * cfg.P


def test_the_chunk_carries_the_scheduled_trigger_not_the_wake_instant(cfg, hw):
    """The chunk timestamps: `t_trigger` is the SCHEDULE, not the wake."""
    with _arena_for(cfg, hw) as rings:
        _fill_epoch(rings["body"], [2], cfg)
        clock = FakeClock()
        loop, _ = _loop(cfg, rings, clock=clock)
        clock.t = T0 - 5.0                          # woke early, or late
        loop.step()
        meta = loop.writes[-1]
        assert meta["t_trigger"] == pytest.approx(T0 + cfg.P * cfg.dt, abs=1e-12)
        # Five of the six are P5 clock readings and must be non-decreasing.
        # `t_anchor` is deliberately NOT in that chain: it is a camera's arrival
        # stamp, and under the nearest-frame rule routinely before `t_trigger`.
        chain = ("t_trigger", "t_obs_received", "t_cpu_processed",
                 "t_chunk_received", "t_chunk_written")
        got = [meta[n] for n in chain]
        assert got == sorted(got)


# --- The sync invariant: the truncation is the infra's now ---------------------------------


def test_a_sync_policy_returns_its_whole_chunk_and_the_infra_cuts_it_to_P(
        cfg, hw):
    """Truncation moved out of the policy: it has to happen AFTER the deadline
    shift, so the shift's extrapolated tail is never commanded."""
    assert cfg.sync
    with _arena_for(cfg, hw) as rings:
        _fill_epoch(rings["body"], [2], cfg)
        loop, _ = _loop(cfg, rings, n=cfg.H)
        assert loop.step() == 1
        assert loop.writes[-1]["n"] == cfg.P
        slot = rings["chunks"]._rec[1 % rings["chunks"].n]
        assert int(slot["n_actions"]) == cfg.P


def test_a_sync_chunk_shorter_than_P_is_refused(cfg, hw):
    """A chunk short of `P` leaves the tick line short, and the
    window extrapolation then spans two epochs."""
    with _arena_for(cfg, hw) as rings:
        _fill_epoch(rings["body"], [2], cfg)
        loop, _ = _loop(cfg, rings, n=cfg.P - 1)
        with pytest.raises(ChunkError, match="at least replan_period_ticks"):
            loop.step()
        assert rings["control"].newest_chunk_id == 0
        assert (loop.k_start_read, loop.k_next_inference) == (2, cfg.P)


def test_async_accepts_the_full_horizon(cfg, hw):
    acfg = cfgmod.replace(cfg, sync=False)
    with _arena_for(acfg, hw) as rings:
        _fill_epoch(rings["body"], [2], acfg)
        loop, _ = _loop(acfg, rings, n=acfg.H)
        assert loop.step() == 1
        assert rings["chunks"].covers_any(acfg.P + acfg.H, adopted_through=None)


def test_more_than_the_horizon_is_refused(cfg):
    with pytest.raises(ChunkError, match="action_horizon"):
        validate_actions(_actions(cfg.H + 1), "eef", cfg)


def test_validate_chunk_rejects_the_empty_and_the_oversized(cfg):
    with pytest.raises(ChunkError):
        validate_chunk(_chunk(cfg.H + 1), cfgmod.replace(cfg, sync=False))
    with pytest.raises(ChunkError):
        validate_chunk(Chunk(hand=np.zeros((0, 2, 6)), pose=np.zeros((0, 2, 7))),
                       cfg)
    with pytest.raises(ChunkError, match="must be exactly"):
        validate_chunk(_chunk(cfg.P + 1), cfg)


def test_the_contract_shapes_are_checked_where_the_policy_hands_them_over():
    with pytest.raises(ChunkError, match="pose"):
        EefActions(pose=np.zeros((4, 7, 2)), hand=np.zeros((4, 2)))
    with pytest.raises(ChunkError, match="hand"):
        EefActions(pose=np.zeros((4, 2, 7)), hand=np.zeros((4,)))
    with pytest.raises(ChunkError, match="arm_q"):
        JointActions(arm_q=np.zeros((4, 14)), waist_q=np.zeros((4, 3)),
                     hand=np.zeros((4, 2)))
    with pytest.raises(ChunkError, match="waist_q"):
        JointActions(arm_q=np.zeros((4, 2, 7)), hand=np.zeros((4, 2)),
                     waist_q=np.zeros((4, 2)))
    assert _chunk(5).truncate(2).n == 2


def test_an_eef_chunk_can_never_carry_a_waist():
    """The eef contract as a property of the type: the IK owns the waist."""
    with pytest.raises(ChunkError, match="no waist"):
        Chunk(hand=np.zeros((2, 2, 6)), pose=np.zeros((2, 2, 7)),
              waist=np.zeros((2, 3)))
    with pytest.raises(ChunkError, match="exactly one"):
        Chunk(hand=np.zeros((2, 2, 6)))


def test_the_wrong_action_type_for_the_mode_is_refused(cfg):
    with pytest.raises(ChunkError, match="JointActions"):
        validate_actions(_actions(cfg.P), "joint", cfg)


# --- The rows truncation cuts (`cfg.plan_lookahead_ticks`) ------------------


def test_truncate_keeps_what_it_cuts_instead_of_dropping_it():
    cut = _chunk(8).truncate(3)
    assert cut.n == 3
    assert cut.plan is not None and cut.plan.n == 5
    assert _chunk(8).truncate(8).plan is None      # nothing was cut
    assert _chunk(3).plan is None                  # never truncated
    assert np.array_equal(cut.pose, _chunk(8).pose[:3])


def test_validate_chunk_also_checks_the_plan_rows(cfg):
    """A NaN here is not inert. The trunk plan is solved OVER these rows and
    the waist it yields is commanded."""
    whole = _chunk(8)
    whole.pose[6, 0, 0] = np.nan                   # a row only the plan holds
    with pytest.raises(ChunkError):
        validate_chunk(whole.truncate(3), cfg)


class _Tailed(StubPolicy):
    """H actions whose rows past P read x = 9.0."""

    def infer(self, obs, reader):
        a = _actions(self.cfg.H, 1.0)
        a.pose[self.cfg.P:, :, 0] = 9.0
        return a


def test_the_knob_publishes_the_plan_rows_beside_the_actions(cfg, hw):
    lcfg = cfgmod.replace(cfg, plan_lookahead_ticks=3)
    with _arena_for(lcfg, hw) as rings:
        _fill_epoch(rings["body"], [2], lcfg)
        loop, _ = _loop(lcfg, rings, policy=_Tailed(lcfg))
        assert loop.step() == 1
        k = loop.writes[0]["k_first_action"]

        tail = rings["chunks"].plan_rows(k, adopted_through=1)
        assert tail is not None
        assert len(tail.pose) == 3                 # the knob, not the horizon
        assert tail.k_first == k + lcfg.P
        assert tail.pose[0, 0, 0] == 9.0
        assert tail.waist is None                  # an eef chunk has none

        # ...and not one of them became executable.
        assert not rings["chunks"].covers_any(k + lcfg.P, adopted_through=1)
        assert rings["chunks"].covers_any(k + lcfg.P - 1, adopted_through=1)


def test_without_the_knob_the_cut_rows_are_not_published(cfg, hw):
    with _arena_for(cfg, hw) as rings:
        _fill_epoch(rings["body"], [2], cfg)
        loop, _ = _loop(cfg, rings, policy=_Tailed(cfg))
        assert loop.step() == 1
        k = loop.writes[0]["k_first_action"]
        assert rings["chunks"].plan_rows(k, adopted_through=1) is None


# --- The first inference ----------------------------------------


def test_first_inference_is_chunk_one_at_tick_one(cfg, hw):
    with _arena_for(cfg, hw) as rings:
        clock = FakeClock()
        loop, _ = _loop(cfg, rings, clock=clock)
        assert loop.first_inference() == 1
        meta = loop.writes[-1]
        assert (meta["chunk_id"], meta["k_first_action"]) == (1, 1)
        assert meta["t_trigger"] == T0
        assert clock.sleeps == []               # nothing to extrapolate from
        assert (loop.k_start_read, loop.k_next_inference) == (2, cfg.P)
        assert rings["control"].newest_chunk_id == 1


# --- The loop and warmup ----------------------------------------------------


def test_run_stops_when_the_session_leaves_active(cfg, hw):
    with _arena_for(cfg, hw) as rings:
        control = rings["control"]
        control.session = schema.ACTIVE
        _fill_epoch(rings["body"], range(2, 4 * cfg.P), cfg)

        loop, _ = _loop(cfg, rings)
        loop.on_write = lambda _loop: setattr(control, "session", schema.EXITING)
        loop.run()
        assert control.newest_chunk_id == 1


class _Warming(StubPolicy):
    """A policy that counts its dummy inferences and starts the run on the Nth."""

    def __init__(self, cfg, control, start_on: int):
        super().__init__(cfg)
        self.control, self.start_on, self.calls_n = control, start_on, 0

    def infer(self, obs, reader):
        self.calls_n += 1
        if self.calls_n == self.start_on:
            self.control.session = schema.ACTIVE
        return obs


def test_warmup_runs_until_the_operator_starts_the_run(cfg, hw):
    """The stopping rule is the session word, which Enter moves on."""
    with _arena_for(cfg, hw) as rings:
        control = rings["control"]
        control.session = schema.WARMUP
        policy = _Warming(cfg, control, start_on=3)
        warmup(control, policy, "dummy", clock=FakeClock())
        assert policy.calls_n == 3


def test_warmup_starts_before_p0_reaches_the_warmup_phase(cfg, hw):
    """The loop gates on `session < ACTIVE`, not on the two named pre-run
    states, so a P5 spawned during CHECK warms through homing."""
    for phase in (schema.INIT, schema.CHECK, schema.HOME, schema.READY):
        with _arena_for(cfg, hw) as rings:
            control = rings["control"]
            control.session = phase
            policy = _Warming(cfg, control, start_on=2)
            warmup(control, policy, "dummy", clock=FakeClock())
            assert policy.calls_n == 2, f"did not warm during {phase}"


class _Timed(StubPolicy):
    """A policy whose dummy inferences take a scripted amount of clock time.

    `seen_warm[i]` is the state of the word ENTERING inference `i`.
    """

    def __init__(self, cfg, control, clock, script, *, limit):
        super().__init__(cfg)
        self.control, self.clock, self.limit = control, clock, limit
        self.script = list(script)
        self.calls_n = 0
        self.seen_warm: list[bool] = []

    def infer(self, obs, reader):
        self.seen_warm.append(bool(self.control.warm))
        self.clock.t += self.script[min(self.calls_n, len(self.script) - 1)]
        self.calls_n += 1
        if self.calls_n >= self.limit:
            self.control.session = schema.ACTIVE        # the operator's Enter
        return obs


def test_warm_is_declared_once_the_dummy_latency_has_settled(cfg, hw):
    """The script is the measured shape of a real run: a ~30 s compile, then
    ~123 ms forever."""
    with _arena_for(cfg, hw) as rings:
        control = rings["control"]
        control.session = schema.WARMUP
        clock = FakeClock()
        policy = _Timed(cfg, control, clock, [29.9, 0.124, 0.123, 0.122], limit=6)
        warmup(control, policy, "dummy", clock=clock)
        assert policy.seen_warm == [False, False, False, True, True, True]
        assert control.warm == 1


def test_two_slow_inferences_that_agree_are_not_mistaken_for_settled(cfg, hw):
    with _arena_for(cfg, hw) as rings:
        control = rings["control"]
        control.session = schema.WARMUP
        clock = FakeClock()
        policy = _Timed(cfg, control, clock, [20.0, 18.0, 0.12, 0.12, 0.12],
                        limit=6)
        warmup(control, policy, "dummy", clock=clock)
        assert policy.seen_warm[:2] == [False, False]
        assert policy.seen_warm.index(True) == 4


def test_warm_is_a_flag_and_never_becomes_a_count(cfg, hw):
    with _arena_for(cfg, hw) as rings:
        control = rings["control"]
        control.session = schema.WARMUP
        clock = FakeClock()
        policy = _Timed(cfg, control, clock, [0.1], limit=25)
        warmup(control, policy, "dummy", clock=clock)
        assert policy.calls_n == 25
        assert control.warm == 1


def test_a_latency_that_never_settles_is_capped_not_waited_on(cfg, hw):
    with _arena_for(cfg, hw) as rings:
        control = rings["control"]
        control.session = schema.WARMUP
        clock = FakeClock()
        policy = _Timed(cfg, control, clock, [1.0, 2.0] * 8,
                        limit=p5mod.WARM_MAX + 2)
        warmup(control, policy, "dummy", clock=clock)
        assert control.warm == 1
        assert policy.seen_warm.index(True) == p5mod.WARM_MAX


def test_a_free_policy_is_paced_rather_than_spun(cfg, hw):
    """A policy with no model infers in microseconds; unpaced, warmup would
    hold P5's core at 100% through homing. It sleeps `_POLL_S` per dummy."""
    with _arena_for(cfg, hw) as rings:
        control = rings["control"]
        control.session = schema.WARMUP
        clock = FakeClock()
        policy = _Warming(cfg, control, start_on=5)
        warmup(control, policy, "dummy", clock=clock)
        assert control.warm == 1
        assert len(clock.sleeps) == 5
        assert clock.t == pytest.approx(T0 + 5 * p5mod._POLL_S)


def test_the_default_warmup_infers_on_the_infra_dummy(cfg, hw):
    """Policies no longer carry their own dummy: the infra builds one from the
    observed cameras' shapes and binds it, and `Policy.warmup` uses it."""
    from elrobot_deploy.policy_contract import Reader, bind_runtime, dummy_observation

    policy = StubPolicy(cfg)
    dummy = dummy_observation(cfg.cameras, cfg.prompt)
    bind_runtime(policy, dummy, Reader())
    policy.warmup()
    (obs,) = policy.seen
    assert obs is dummy
    assert obs.frames["head"].pixels.shape == (8, 16, 3)
    assert obs.frames["rwrist"].pixels.dtype == np.uint8
    assert obs.state.q.shape == (schema.N_JOINTS,)
    assert obs.state.flange.shape == (2, 7)


def test_warmup_without_a_bound_runtime_says_what_is_missing(cfg):
    with pytest.raises(RuntimeError, match="bind_runtime"):
        StubPolicy(cfg).warmup()


# --- The anchor ----------------------------------------------


def _publish(cam, t):
    cam.spare()["pixels"][:] = int(t * 10) % 251
    cam.publish(t_recv=t)


def _state(ring, t):
    ring.write(t_recv=t, q=np.full(schema.N_JOINTS, t),
               dq=np.zeros(schema.N_JOINTS),
               flange=np.array([[t, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]] * schema.N_HANDS),
               grip_meas=np.zeros(schema.GRIP_DIM), t_grip_recv=t)


HALF = {"rwrist": 0.5 / 30.0, "head": 0.5 / 30.0}


def _anchor(rings, cfg, t_trigger=T0, half=None, now=None):
    return acquire_anchor(rings["cameras"], rings["state"], "rwrist",
                          t_trigger, half or HALF, now=now)


def test_the_frame_in_hand_is_taken_when_it_is_inside_half_a_period(rings, cfg):
    """The early exit, and it changes no answer. Frames come `T` apart, so a
    held frame `age` before the trigger means the next lands `T - age` after —
    the held one is nearer exactly when `age < T/2`. Waiting could only confirm
    what is already known, and before this the anchor blocked a FULL period on
    every frame that landed a fraction of a millisecond on the wrong side."""
    _publish(rings["cameras"]["rwrist"], T0 - 0.004)     # 4 ms < 16.7 ms
    _publish(rings["cameras"]["head"], T0 - 0.004)
    _state(rings["state"], T0 - 0.001)
    _state(rings["state"], T0 + 0.020)

    anchor = _anchor(rings, cfg)

    assert anchor.t_anchor == T0 - 0.004         # the frame already in hand
    assert anchor.t_trigger == T0                # and it is NOT the trigger
    assert set(anchor.frames) == set(cfg.cameras)


def test_a_frame_older_than_half_a_period_waits_for_the_next_one(rings, cfg):
    """The other half of the rule: past `T/2` the *next* frame is nearer, so
    the wait is real — but it is bounded by `T - age < T/2`, never a period."""
    cam = rings["cameras"]["rwrist"]
    _publish(cam, T0 - 0.030)                    # 30 ms > 16.7 ms: too old
    _publish(rings["cameras"]["head"], T0 - 0.001)
    _state(rings["state"], T0 + 0.005)

    # The awaited frame lands while the anchor is polling.
    published = []

    def now():
        if not published:
            published.append(True)
            _publish(cam, T0 + 0.003)
        return T0

    anchor = _anchor(rings, cfg, now=now)

    assert anchor.t_anchor == T0 + 0.003


def test_the_state_is_matched_to_the_anchor_not_to_the_trigger(rings, cfg):
    """The reason a primary exists at all. The acting camera is mounted on the
    arm and a policy composes its actions onto `obs.state.flange`, so image
    and flange must describe one instant. Matching the state to `t_trigger`
    instead would let them drift by up to `T/2` — measured on the reference robot as
    p99 5.4 mm, max 13.6 mm of end-effector error."""
    _publish(rings["cameras"]["rwrist"], T0 - 0.010)
    _publish(rings["cameras"]["head"], T0 - 0.010)
    _state(rings["state"], T0 - 0.011)           # nearest the ANCHOR
    _state(rings["state"], T0 + 0.001)           # nearest the TRIGGER

    anchor = _anchor(rings, cfg)

    assert anchor.t_anchor == T0 - 0.010
    assert anchor.state["t_recv"] == T0 - 0.011


def test_every_camera_uses_the_same_rule_not_a_primary_special_case(rings, cfg):
    """One rule for every stream, so a rig with an exo camera, a head and two
    wrists needs no special case. The primary stays a role — it fixes
    `t_anchor` — but not a different selection rule."""
    _publish(rings["cameras"]["rwrist"], T0 - 0.002)
    _publish(rings["cameras"]["head"], T0 - 0.006)
    _state(rings["state"], T0)

    anchor = _anchor(rings, cfg)

    assert anchor.t_anchor == T0 - 0.002                     # primary fixes it
    assert anchor.frames["head"]["t_recv"] == T0 - 0.006      # kept, not waited on


def test_max_image_latency_reports_a_camera_that_lags_the_anchor(rings, cfg):
    """The only symptom a stopped camera now produces, since the wait is
    capped. Zero for the primary by construction."""
    _publish(rings["cameras"]["rwrist"], T0 - 0.002)
    _publish(rings["cameras"]["head"], T0 - 0.009)
    _state(rings["state"], T0)

    anchor = _anchor(rings, cfg)

    assert anchor.max_image_latency() == pytest.approx(0.007)


def test_max_image_latency_also_sees_a_camera_that_LEADS_the_anchor(rings, cfg):
    """Why the magnitude, not the signed difference. A signed
    `max(t_anchor - t_C)` is swallowed by the primary's own zero term whenever
    the other cameras run AHEAD, so a rig 8 ms out of step would report a
    spread of exactly 0.0 — the one number meant to answer "do these images
    depict the same instant" would answer "perfectly" while they did not."""
    _publish(rings["cameras"]["rwrist"], T0 - 0.010)     # primary: the anchor
    _publish(rings["cameras"]["head"], T0 - 0.002)       # 8 ms NEWER
    _state(rings["state"], T0)

    anchor = _anchor(rings, cfg)

    signed = max(anchor.t_anchor - float(f["t_recv"])
                 for f in anchor.frames.values())
    assert signed == pytest.approx(0.0)                  # what it used to say
    assert anchor.max_image_latency() == pytest.approx(0.008)


def test_the_wait_is_capped_so_a_stopped_camera_cannot_hang_the_run(rings, cfg):
    """The invisible dead process's cost, reduced. A dead camera used to block `acquire_anchor`
    forever; now the frames in hand are used and the run continues. That is a
    trade, not a free win — a robot acting on a frozen image is worse than one
    that stopped — which is why `max_image_latency` exists and P5 prints it."""
    _publish(rings["cameras"]["rwrist"], T0 - 0.030)     # too old, and no more
    _publish(rings["cameras"]["head"], T0 - 0.030)
    _state(rings["state"], T0)

    ticks = iter([T0, T0 + 0.001, T0 + 1.0])             # the clock runs past

    anchor = _anchor(rings, cfg, now=lambda: next(ticks))

    assert anchor.t_anchor == T0 - 0.030                 # the stale frame, used
    assert anchor.max_image_latency() == pytest.approx(0.0)  # both equally stale


def test_the_cap_never_yields_an_anchor_with_no_image(rings, cfg):
    """A camera that has published NOTHING is not covered by the cap: there is
    no frame to fall back on, so that still blocks. CHECK has already proven
    every declared camera produces, so this is a decoder dying mid-run."""
    _publish(rings["cameras"]["rwrist"], T0 - 0.002)
    _state(rings["state"], T0)
    late = []

    def now():
        # The clock is far past the cap, yet `head` has never produced.
        if len(late) == 3:
            _publish(rings["cameras"]["head"], T0 - 0.001)
        late.append(True)
        return T0 + 10.0

    anchor = _anchor(rings, cfg, now=now)

    assert anchor.frames["head"] is not None
    assert len(late) > 3, "returned before head had any frame at all"


def test_echo_policy_holds_the_measured_pose(cfg, hw):
    """The built-in policy that does nothing, so the loop runs without a
    checkpoint."""
    with _arena_for(cfg, hw) as rings:
        _state(rings["state"], T0 + 2e6)             # the anchor's match
        policy = policymod.make_policy(cfg, hw)
        assert policy.anchor_camera == cfg.primary_camera
        loop, _ = _loop(cfg, rings, policy=policy, hw=hw)
        chunk, _stamps = loop.produce(T0)
        validate_chunk(chunk, cfg)
        assert chunk.n == cfg.P                 # sync truncation is the infra's
        assert chunk.mode == "eef" and chunk.waist is None


# --- The observation ----------------------------------------------


def test_frames_reach_the_policy_as_RGB(cfg, hw):
    """The wire is BGR; every training pipeline is RGB. Wrong, this produces no
    error and no warning — the policy behaves as though it had never seen the
    scene — so the infra converts once, for every policy."""
    with _arena_for(cfg, hw) as rings:
        cam = rings["cameras"]["rwrist"]
        spare = cam.spare()
        spare["pixels"][:] = 0
        spare["pixels"][..., 0] = 200                # B, on the wire
        spare["pixels"][..., 2] = 10                 # R
        cam.publish(t_recv=T0 + 2e6)
        _state(rings["state"], T0 + 2e6)             # the anchor's state match
        _fill_epoch(rings["body"], [2], cfg)
        loop, policy = _loop(cfg, rings)
        loop.step()
        px = policy.seen[-1].frames["rwrist"].pixels
        assert px.dtype == np.uint8 and px.shape == (8, 8, 3)
        assert (px[..., 0] == 10).all() and (px[..., 2] == 200).all()
        # A private copy: rewriting the arena does not reach the policy's frame.
        cam.spare()["pixels"][:] = 77
        cam.publish(t_recv=T0 + 3e6)
        assert (px[..., 0] == 10).all()


def test_the_observation_carries_the_anchor_and_its_state(cfg, hw):
    with _arena_for(cfg, hw) as rings:
        _fill_epoch(rings["body"], [2], cfg)
        loop, policy = _loop(cfg, rings)
        loop.step()
        obs = policy.seen[-1]
        assert obs.t_anchor == T0 + 1e6
        assert obs.frames["rwrist"].t_recv == T0 + 1e6
        assert obs.state.t_recv == T0 + 1e6
        assert obs.state.q.shape == (schema.N_JOINTS,)
        assert obs.state.flange.shape == (2, 7)
        assert obs.state.hand.shape == (2,)
        assert obs.prompt == cfg.prompt


def test_the_reader_returns_every_state_sample_in_a_window(rings):
    from elrobot_deploy.policy_contract import Reader

    for i in range(10):
        _state(rings["state"], T0 + i * 0.001)
    reader = Reader(rings["state"], {"state": 1.0})
    w = reader.states_between(T0 + 0.0025, T0 + 0.0065)
    assert list(np.round((w.t_recv - T0) * 1e3)) == [3, 4, 5, 6]
    assert w.q.shape == (4, schema.N_JOINTS)
    assert w.flange.shape == (4, 2, 7) and w.hand.shape == (4, 2)


def test_the_reader_refuses_an_undeclared_stream(rings):
    from elrobot_deploy.policy_contract import Reader

    with pytest.raises(LookupError, match="history"):
        Reader(rings["state"], {}).states_between(0.0, 1.0)
    cams = {"rwrist": rings["cameras"]["rwrist"]}
    with pytest.raises(LookupError, match="history"):
        Reader(rings["state"], {"state": 1.0}, cams).frame_ids_between("rwrist", 0.0, 1.0)
    with pytest.raises(LookupError, match="no camera 'lwrist'"):
        Reader(rings["state"], {"lwrist": 1.0}, cams).frames_between("lwrist", 0.0, 1.0)
    assert Reader(rings["state"], {"rwrist": 1.0}, cams).frames_between(
        "rwrist", 0.0, 1.0) == []                        # declared, nothing yet


def test_a_torn_state_slot_is_dropped_never_returned(rings):
    """Copy-and-verify: a slot whose counter is odd — P1 mid-write —
    is left out of the window rather than handed over half-written."""
    for i in range(4):
        _state(rings["state"], T0 + i * 0.001)
    ring = rings["state"]
    ring._lock.seq[1] += 1                          # slot 1 "being written"
    got = ring.window(T0 - 1, T0 + 1)
    assert list(np.round((got["t_recv"] - T0) * 1e3)) == [0, 2, 3]


# --- The deadline shift -----------------------------------------


def test_the_shift_sign_matches_the_worked_example():
    """`t_trigger` 10 ms, `t_anchor` 25 ms, dt 33.33 ms: the first action is due
    at 43.33 ms — 55% of the way from the anchor state to the model's first
    action, which it reads as 58.33 ms. A flipped sign is a half-tick LEAD."""
    from elrobot_deploy.policy_contract import deadline_shift_slots, shift_to_deadline

    dt = 1 / 30
    s = deadline_shift_slots(0.025, 0.010, dt)
    assert s == pytest.approx(-0.45)

    a = _actions(4, 0.0)
    a.pose[:, :, 0] = [[0.1], [0.2], [0.3], [0.4]]
    meas = {"flange": np.array([[0.0, 0, 0, 1, 0, 0, 0]] * 2),
            "hand": np.zeros(2)}
    out = shift_to_deadline(a, meas, s)
    assert out.pose[0, 0, 0] == pytest.approx(0.055)
    assert out.pose[1, 0, 0] == pytest.approx(0.155)


def test_a_later_shift_extrapolates_the_tail_rather_than_freezing():
    from elrobot_deploy.policy_contract import shift_to_deadline

    a = _actions(3, 0.0)
    a.pose[:, :, 0] = [[0.1], [0.2], [0.3]]
    meas = {"flange": np.array([[0.0, 0, 0, 1, 0, 0, 0]] * 2),
            "hand": np.zeros(2)}
    out = shift_to_deadline(a, meas, +0.5)
    assert out.pose[:, 0, 0] == pytest.approx([0.15, 0.25, 0.35])


def test_a_zero_shift_is_bit_identical_to_none():
    from elrobot_deploy.policy_contract import shift_to_deadline

    a = _actions(3)
    assert shift_to_deadline(a, {}, 0.0) is a


def test_the_shift_is_bounded_for_a_camera_that_stopped():
    """A dead camera leaves the anchor arbitrarily old; the honest correction
    would be an arbitrarily long extrapolation. One tick is the most invented."""
    from elrobot_deploy.policy_contract import deadline_shift_slots
    from elrobot_deploy.policy_contract.deadline_shift import MAX_SHIFT_SLOTS

    assert deadline_shift_slots(T0 - 5.0, T0, 1 / 30) == MAX_SHIFT_SLOTS
    assert deadline_shift_slots(T0 + 5.0, T0, 1 / 30) == -MAX_SHIFT_SLOTS


def test_rotations_are_slerped_and_the_point_is_honoured():
    """The rotation is the same at every rigidly attached point, so only the
    translation depends on where the interpolation happens. Interpolating at a
    TCP 0.2 m out moves the flange by the chord-vs-arc deficit, not by zero."""
    from elrobot_deploy.policy_contract import shift_to_deadline

    th = 0.4
    a = EefActions(pose=np.array([[[0, 0, 0, np.cos(th / 2), 0, 0,
                                    np.sin(th / 2)]] * 2]),
                   hand=np.zeros((1, 2)))
    meas = {"flange": np.array([[0.0, 0, 0, 1, 0, 0, 0]] * 2),
            "hand": np.zeros(2)}
    at_flange = shift_to_deadline(a, meas, -0.5)
    q = at_flange.pose[0, 0, 3:]
    assert q == pytest.approx([np.cos(th / 4), 0, 0, np.sin(th / 4)])
    assert at_flange.pose[0, 0, :3] == pytest.approx([0, 0, 0])

    at_tcp = shift_to_deadline(a, meas, -0.5, point=(0.2, 0.0, 0.0))
    # The TCP's chord midpoint, mapped back through the half-way rotation.
    tcp_mid = 0.5 * (np.array([0.2, 0, 0])
                     + 0.2 * np.array([np.cos(th), np.sin(th), 0]))
    back = tcp_mid - 0.2 * np.array([np.cos(th / 2), np.sin(th / 2), 0])
    assert at_tcp.pose[0, 0, :3] == pytest.approx(back)
    assert np.linalg.norm(back) > 1e-3


def test_joint_actions_shift_linearly_from_the_measured_joints():
    from elrobot_deploy.policy_contract import shift_to_deadline

    a = JointActions(arm_q=np.full((2, 2, 7), 1.0), hand=np.full((2, 2), 2.0),
                     waist_q=np.full((2, 3), 0.2))
    meas = {"arm_q": np.zeros((2, 7)), "waist_q": np.zeros(3),
            "hand": np.zeros(2)}
    out = shift_to_deadline(a, meas, -0.25)
    assert out.arm_q[0] == pytest.approx(np.full((2, 7), 0.75))
    assert out.waist_q[0] == pytest.approx(np.full(3, 0.15))
    assert out.hand[0] == pytest.approx([1.5, 1.5])
    assert out.arm_q[1] == pytest.approx(np.full((2, 7), 1.0))


def test_p5_applies_the_shift_to_every_policy(cfg, hw):
    """The policy returns actions timed from `t_anchor`; P5 re-times them. With
    the frame half a tick newer than the trigger, the first action moves half
    way back to the measured flange."""
    class Shifted(StubPolicy):
        deadline_shift = True

    with _arena_for(cfg, hw) as rings:
        t_trigger = T0 + 5.0
        for region in rings["cameras"].values():
            _publish(region, t_trigger + 0.5 * cfg.dt - 1e-9)
        _state(rings["state"], t_trigger + 0.5 * cfg.dt)
        loop, _ = _loop(cfg, rings, policy=Shifted(cfg))
        chunk, _ = loop.produce(t_trigger)
        measured_x = t_trigger + 0.5 * cfg.dt          # `_state` writes x = t
        assert chunk.pose[0, 0, 0] == pytest.approx(
            0.5 * measured_x + 0.5 * 1.00, rel=1e-6)
        assert chunk.pose[1, 0, 0] == pytest.approx(0.5 * 1.00 + 0.5 * 1.01)


def test_the_clip_catches_what_only_the_shift_put_out_of_range(cfg, hw):
    """The measured gripper can read a hair past the top of travel (5.4248 was
    recorded); interpolating toward it must not get the chunk refused."""
    class Shifted(StubPolicy):
        deadline_shift = True

        def infer(self, obs, reader):
            a = _actions(self.cfg.H, 1.0)
            return EefActions(pose=a.pose, hand=np.full((self.cfg.H, 2), 5.4))

    with _arena_for(cfg, hw) as rings:
        t_trigger = T0 + 5.0
        for region in rings["cameras"].values():
            _publish(region, t_trigger + 0.5 * cfg.dt)
        rings["state"].write(
            t_recv=t_trigger + 0.5 * cfg.dt, q=np.zeros(schema.N_JOINTS),
            dq=np.zeros(schema.N_JOINTS),
            flange=np.array([[0.0, 0, 0, 1, 0, 0, 0]] * 2),
            grip_meas=np.full(schema.GRIP_DIM, 5.4248), t_grip_recv=0.0)
        loop, _ = _loop(cfg, rings, policy=Shifted(cfg))
        chunk, _ = loop.produce(t_trigger)
        assert chunk.hand[..., 0].max() <= 5.4


# --- The contract boundary --------------------------------------------------


def test_a_non_finite_action_never_reaches_the_ring(cfg):
    """A NaN survives both velocity clamps — `np.clip(nan, lo, hi)` is `nan` —
    so this is the only place between a diverged policy and `rt/lowcmd`."""
    a = _actions(cfg.P)
    for name in ("pose", "hand"):
        block = getattr(a, name).copy()
        block.reshape(-1)[0] = np.nan
        bad = EefActions(**{"pose": a.pose, "hand": a.hand, name: block})
        with pytest.raises(ChunkError, match="non-finite"):
            validate_actions(bad, "eef", cfg)
    j = JointActions(arm_q=np.zeros((cfg.P, 2, 7)), hand=np.zeros((cfg.P, 2)),
                     waist_q=np.full((cfg.P, 3), np.inf))
    with pytest.raises(ChunkError, match="non-finite"):
        validate_actions(j, "joint", cfg)
    j = JointActions(arm_q=np.zeros((cfg.P, 2, 7)), hand=np.zeros((cfg.P, 2)),
                     waist_q=np.full((cfg.P, 3), np.nan))
    with pytest.raises(ChunkError, match="non-finite"):
        validate_actions(j, "joint", cfg)
    c = _chunk(cfg.P)
    c.pose[0, 0, 0] = np.inf
    with pytest.raises(ChunkError, match="non-finite"):
        validate_chunk(c, cfg)


def test_the_hand_is_held_to_the_dex1s_travel(cfg):
    """0 … 5.4 rad, 0 = closed. A policy once asked for 5.4304; the contract
    now refuses that at the boundary rather than leaving it to P8's clamp."""
    a = _actions(cfg.P)
    over = EefActions(pose=a.pose, hand=np.full((cfg.P, 2), 5.4304))
    with pytest.raises(ChunkError, match="Dex-1"):
        validate_actions(over, "eef", cfg)
    under = EefActions(pose=a.pose, hand=np.full((cfg.P, 2), -0.1))
    with pytest.raises(ChunkError, match="Dex-1"):
        validate_actions(under, "eef", cfg)
    validate_actions(EefActions(pose=a.pose, hand=np.full((cfg.P, 2), 5.4)),
                     "eef", cfg)


def test_a_non_unit_quaternion_is_refused(cfg):
    a = _actions(cfg.P)
    a.pose[0, 1, 3:] = [2.0, 0, 0, 0]
    with pytest.raises(ChunkError, match="unit"):
        validate_actions(a, "eef", cfg)


def test_joint_actions_are_held_to_the_urdf_range(cfg, hw):
    """Joint mode is the one path where an arm joint reaches the wire without
    the QP's joint limits in front of it; the waist never goes through IK in
    any mode. The absent third waist slot's range is exactly [0, 0]."""
    limits = Limits.from_robot(hw)
    assert not limits.waist[2].any()
    ok = JointActions(arm_q=np.zeros((cfg.P, 2, 7)), hand=np.zeros((cfg.P, 2)),
                      waist_q=np.zeros((cfg.P, 3)))
    validate_actions(ok, "joint", cfg, limits)

    arm = ok.arm_q.copy()
    arm[0, 1, 3] = limits.arm[7 + 3, 1] + 0.1        # right elbow past its top
    with pytest.raises(ChunkError, match="arm_q"):
        validate_actions(JointActions(arm_q=arm, waist_q=ok.waist_q,
                                      hand=ok.hand), "joint", cfg, limits)

    waist = ok.waist_q.copy()
    waist[0, 2] = 0.01                               # the axis that is not there
    with pytest.raises(ChunkError, match="waist_q"):
        validate_actions(JointActions(arm_q=ok.arm_q, hand=ok.hand,
                                      waist_q=waist), "joint", cfg, limits)


#: A waist inside the URDF range: slot 2 has no joint, its range is [0, 0].
WAIST = np.array([0.01, 0.02, 0.0])


def test_a_joint_policy_publishes_joints_and_no_pose(cfg, hw):
    class Joints(StubPolicy):
        action_mode = "joint"
        end_effector = "dex1"

        def infer(self, obs, reader):
            q = np.zeros((self.cfg.H, 2, 7))
            q[:, :, 3] = 0.5
            return JointActions(arm_q=q, waist_q=np.tile(WAIST, (self.cfg.H, 1)),
                                hand=np.zeros((self.cfg.H, 2)))

    with _arena_for(cfg, hw) as rings:
        _fill_epoch(rings["body"], [2], cfg)
        loop, _ = _loop(cfg, rings, policy=Joints(cfg), hw=hw)
        assert loop.step() == 1
        slot = rings["chunks"]._rec[1 % rings["chunks"].n]
        assert int(slot["mode"]) == 1
        assert np.all(slot["arm_q"][:cfg.P, :, 3] == 0.5)
        assert np.isnan(slot["pose"]).all()
        # The waist is the policy's, always: joint mode requires it.
        assert np.allclose(slot["waist"][:cfg.P], WAIST)
        (c,) = rings["chunks"].covering(cfg.P + 1)
        assert c.pose is None and c.arm_q.shape == (2, 7)
        assert np.allclose(c.waist, WAIST)


def test_joint_actions_require_the_waist():
    """No default and no None: the infra assumes nothing about the waist in
    joint mode. A policy that wants it upright returns zeros."""
    with pytest.raises(TypeError, match="waist_q"):
        JointActions(arm_q=np.zeros((3, 2, 7)), hand=np.zeros((3, 2)))
    with pytest.raises(ChunkError, match="REQUIRED"):
        JointActions(arm_q=np.zeros((3, 2, 7)), waist_q=None,
                     hand=np.zeros((3, 2)))


def test_a_joint_chunk_without_a_waist_cannot_exist(cfg, hw):
    with pytest.raises(ChunkError, match="waist"):
        Chunk(hand=np.zeros((2, 2, 6)), arm_q=np.zeros((2, 2, 7)))
    with _arena_for(cfg, hw) as rings:
        with pytest.raises(ValueError, match="waist"):
            rings["chunks"].write(chunk_id=1, k_first_action=1, n_actions=2,
                                  arm_q=np.zeros((2, 2, 7)),
                                  hand=np.zeros((2, 2, 6)))


# --- The class declarations -------------------------------------


def test_action_mode_and_anchor_camera_are_required():
    from elrobot_deploy.policy_contract import PolicyError, validate_policy_class

    class NoMode(Policy):
        anchor_camera = "rwrist"

        def infer(self, obs, reader):
            pass

    class BadMode(NoMode):
        action_mode = "cartesian"

    class NoAnchor(Policy):
        action_mode = "eef"
        end_effector = "dex1"

        def infer(self, obs, reader):
            pass

    class NoInfer(Policy):
        action_mode, anchor_camera = "eef", "rwrist"
        end_effector = "dex1"

    class AnchorNotUsed(NoAnchor):
        anchor_camera, uses_cameras = "rwrist", ("head",)

    for cls, match in ((NoMode, "action_mode"), (BadMode, "action_mode"),
                       (NoAnchor, "anchor_camera"), (NoInfer, "infer"),
                       (AnchorNotUsed, "uses_cameras")):
        with pytest.raises(PolicyError, match=match):
            validate_policy_class(cls)
    with pytest.raises(PolicyError, match="subclass"):
        validate_policy_class(object)
    validate_policy_class(StubPolicy)


def test_end_effector_is_required_and_checked_against_the_robot(hw):
    """`end_effector` names what `hand` means; it must be one the contract
    speaks, and the robot file's."""
    from elrobot_deploy.policy_contract import (PolicyError, check_end_effector,
                                            validate_policy_class)

    class NoHand(Policy):
        action_mode, anchor_camera = "eef", "rwrist"

        def infer(self, obs, reader):
            pass

    class Brainco(NoHand):
        end_effector = "brainco"

    with pytest.raises(PolicyError, match="end_effector"):
        validate_policy_class(NoHand)
    with pytest.raises(PolicyError, match="not supported"):
        validate_policy_class(Brainco)

    class Dex1(NoHand):
        end_effector = "dex1"

    validate_policy_class(Dex1)
    check_end_effector(Dex1, hw)
    bc = cfgmod.for_hand(hw, "brainco")
    with pytest.raises(PolicyError, match="does not support"):
        check_end_effector(Dex1, bc)


def test_the_hand_is_one_value_per_side_in_both_modes():
    """`EefActions(pose, hand)` / `JointActions(arm_q, waist_q, hand)`: `hand`
    is (n, 2); the six-wide block is the ring's, packed by the infra."""
    from elrobot_deploy.policy_contract.actions import hand_block

    a = EefActions(pose=np.tile([0, 0, 0, 1.0, 0, 0, 0], (3, 2, 1)),
                   hand=np.full((3, 2), 2.0))
    assert a.hand.shape == (3, 2)
    block = hand_block(a)
    assert block.shape == (3, 2, 6) and (block[..., 0] == 2.0).all()
    assert not block[..., 1:].any()
    with pytest.raises(ChunkError, match="hand"):
        EefActions(pose=a.pose, hand=np.zeros((3, 2, 6)))
    j = JointActions(arm_q=np.zeros((3, 2, 7)), waist_q=np.zeros((3, 3)),
                     hand=np.zeros((3, 2)))
    assert j.waist_q.shape == (3, 3) and j.hand.shape == (3, 2)


def test_the_declarations_are_checked_against_the_run(cfg):
    from elrobot_deploy.policy_contract import PolicyError, check_against

    class Elsewhere(StubPolicy):
        anchor_camera = "exo"

    class DeepHistory(StubPolicy):
        history = {"state": cfg.state_history_s + 1.0}

    # As resolution builds it: `primary_camera` carries the class's anchor
    # unless the deploy file overrides it.
    with pytest.raises(PolicyError, match="anchor_camera"):
        check_against(Elsewhere, cfgmod.replace(cfg, primary_camera="exo"))
    with pytest.raises(PolicyError, match="state ring"):
        check_against(DeepHistory, cfg)
    check_against(StubPolicy, cfg)
    # A deploy-file override to a delivered camera wins over the class's...
    check_against(Elsewhere, cfgmod.replace(cfg, primary_camera="rwrist"))
    # ...and one to a camera that is not delivered is refused, not ignored.
    with pytest.raises(PolicyError, match=r"\[policy\] anchor_camera"):
        check_against(StubPolicy, cfgmod.replace(
            cfg, primary_camera="head", observed_cameras=("rwrist",)))


def test_a_policy_loads_by_module_or_by_path(tmp_path):
    from elrobot_deploy.policy_contract.loading import load_entry

    cls = load_entry("elrobot_deploy.policy_contract.base:EchoPolicy")
    assert cls.action_mode == "eef"

    src = tmp_path / "mine.py"
    src.write_text(
        "from elrobot_deploy.policy_contract import Policy\n"
        "class Mine(Policy):\n"
        "    action_mode = 'joint'\n"
        "    end_effector = 'dex1'\n"
        "    anchor_camera = 'head'\n"
        "    def infer(self, obs, reader):\n"
        "        return None\n")
    assert load_entry(f"{src}:Mine").action_mode == "joint"


@pytest.mark.parametrize("example", ["deploy.toml", "policies/sine_eef.toml",
                                     "policies/sine_joint.toml"])
def test_the_examples_run_through_p5(example, hw):
    """Every shipped example satisfies its own contract against a live-shaped
    arena: its deploy.toml resolves, the policy loads and validates, and it
    produces a chunk P5 will publish."""
    from pathlib import Path

    from elrobot_deploy.config_loading import deploy_file as deployfile

    path = Path(__file__).resolve().parents[1] / "examples" / example
    r = deployfile.resolve(path, record=False)
    ecfg = r.cfg
    policy = policymod.make_policy(ecfg, r.hw)
    with _arena_for(ecfg, r.hw) as rings:
        loop, _ = _loop(ecfg, rings, policy=policy, hw=r.hw)
        chunk, _ = loop.produce(T0 + 1e6)
        validate_chunk(chunk, ecfg)
        assert chunk.mode == policy.action_mode
        assert chunk.n == ecfg.P


# --- policy_env -------------------------------------------------


def test_policy_env_overrides_what_the_shell_exported(cfg):
    env = {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    changed = apply_policy_env(
        cfgmod.replace(cfg, policy_env={"XLA_PYTHON_CLIENT_PREALLOCATE": "true"}),
        env)
    assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "true"
    assert changed == {"XLA_PYTHON_CLIENT_PREALLOCATE": ("false", "true")}


def test_an_empty_policy_env_touches_nothing(cfg):
    env = {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    assert apply_policy_env(cfg, env) == {}
    assert env == {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}


def test_policy_env_is_applied_before_any_import(cfg):
    import inspect

    from elrobot_deploy.runtime.processes import p5_policy

    src = inspect.getsource(p5_policy.run)
    assert src.index("apply_policy_env(cfg)") < src.index("from elrobot_deploy.policy_contract import")


def test_the_toy_joint_example_commands_an_explicit_upright_waist(cfg, hw):
    """`waist_q` is required in joint mode; the toy returns zeros."""
    from pathlib import Path

    from elrobot_deploy.policy_contract.loading import load_entry
    from elrobot_deploy.policy_contract.observation import Observation, State
    src = Path(__file__).resolve().parents[1] / "examples" / "policies" / "sine_joint.py"
    pol = load_entry(f"{src}:Policy")(None, "", hw)
    obs = Observation(t_anchor=1.0, frames={}, prompt="",
                      state=State(1.0, np.zeros(29), np.zeros(29),
                                  np.zeros((2, 7)), np.zeros(2)))
    a = pol.infer(obs, None)
    assert a.waist_q.shape == (a.n, 3) and not a.waist_q.any()
    validate_actions(a, "joint", cfgmod.replace(cfg, action_horizon=a.n),
                     Limits.from_robot(hw))
