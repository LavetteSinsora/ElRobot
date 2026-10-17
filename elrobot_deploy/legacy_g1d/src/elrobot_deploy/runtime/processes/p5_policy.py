"""P5 — the policy process.

Runs under the ADAPTER's interpreter (`[policy] python`), started through
`policy_host.py`. Design: `docs/overview.md`, "One chunk's life";
`docs/deploying_a_policy/policy_contract.md`, "Lifecycle" and "What happens to
the actions".

Turns observations into chunks. Not real-time: it is allowed to be slow, and
its latency is the thing the whole schedule is built around.

**Owns** `k_start_read` and `k_next_inference`. **Reads** the body target ring
(deadlines only), the camera regions and the state ring. **Writes** the chunk
ring and `newest_chunk_id`. Between rollouts (`RolloutLatch`) it also
reads the prompt request and the active block, and writes the prompt verdict
and `settings_applied`.

The loop is four moves — scan a window, sleep precisely, produce a chunk,
publish — and every subtlety is in the first and the last:

* the window scan takes the **earliest** written tick, not the latest;
* `newest_chunk_id` is published **after** the payload it names.

One iteration is `step()`, so the schedule can be tested without a clock, an
inference or a second process.

**Producing a chunk is mostly the infra's.** The policy
owns one call, `infer(obs, reader)`. Around it, in order, this process:

1. acquires the anchor (`policy_contract.acquire_anchor`), on the policy's
   `anchor_camera` over its observed cameras;
2. builds the `Observation` — frames converted BGR -> RGB once, here;
3. calls `infer`;
4. validates the actions against the contract (`validate_actions`) — every
   error there names the policy;
5. applies the deadline shift (`policy_contract/deadline_shift.py`), so every
   policy's actions meet the deadlines they will actually be executed at, not
   the ones the contract's nominal timing implies;
6. clips what the shift may have pushed a hair outside the joint and hand
   ranges, cuts to `P` in sync (the tail kept as lookahead), packs the hand
   into the hand block, and publishes.

Steps 5 and 6 are in that order for a reason: the shift can extrapolate half a
tick past the end of the chunk, and truncating AFTER it means no extrapolated
action is ever commanded in sync.
"""

from __future__ import annotations

from collections import deque

import os

import numpy as np

from elrobot_deploy.runtime.teleop import intervention as intervene
from elrobot_deploy.runtime.shared_memory import schema
from elrobot_deploy.policy_contract import (EEF, Chunk, Limits, Reader, acquire_anchor,
                                        deadline_shift_slots, observation_cameras,
                                        observe, shift_to_deadline, validate_actions,
                                        validate_chunk)
from elrobot_deploy.policy_contract.actions import first_actions, hand_block, hand_range
from elrobot_deploy.policy_contract.base import anchor_camera

#: Where the arm and the waist sit in the 29-wide wire `q` when no robot is
#: supplied (tests driving the loop directly). The real map is the robot file's
#: `[joints]`, read from `hw` whenever there is one.
_ARM_IDX = tuple(range(15, 29))
_WAIST_IDX = (12, 13, 14)

# The poll used while waiting for a session transition. Not a knob: nothing is
# scheduled against it — it only decides how quickly P5 notices that P0 has
# moved the session word on.
_POLL_S = 1e-3

#: Warmup's stopping rule, owned here because only this process can time
#: an inference. Two consecutive dummies within `WARM_TOL` of each other, never
#: fewer than `WARM_MIN`, never more than `WARM_MAX`. Measured on the deploy box:
#: inference 1 is the ~30 s compile, 2 and 3 are ~123 ms and agree — settles at 3.
#:
#: `WARM_MIN` is not decoration. "Two consecutive that agree" is also true of two
#: consecutive SLOW ones, so a compile split across two calls — two XLA
#: executables, say — would satisfy the tolerance while still compiling. Three
#: puts a known-fast pair on the end of any such run.
WARM_TOL = 0.20
WARM_MIN = 3
WARM_MAX = 10

#: How many recent steady-state dummy inferences `Control.latency_s` is taken
#: over, and which of them: the 90th percentile, so one slow outlier neither
#: refuses an async rollout nor hides a latency that is usually too long.
LATENCY_WINDOW = 20
LATENCY_QUANTILE = 0.9


class PolicyLoop:
    """The scheduling loop, with the two cursors it owns.

    Constructed with an explicit clock so a test can drive the schedule without
    waiting 33 ms per tick. Everything else — the rings, the policy — is real.
    """

    def __init__(self, cfg, rings, policy, *, hw=None, clock=None):
        if clock is None:
            from elrobot_deploy.runtime import timing as _clock
            clock = _clock
        self.cfg = cfg
        self.policy = policy
        self.clock = clock
        self.control = rings["control"]
        self.body = rings["body"]
        self.state = rings["state"]
        self.chunks = rings["chunks"]
        self.mode = policy.action_mode
        # THE OBSERVED SET, not every camera in the arena. `cfg.cameras` is
        # what exists — each one has a region, a reader and a place in the
        # record — and the policy's `uses_cameras` (else `cfg.observed`) is what
        # it sees. Handing the full set to `acquire_anchor` would make a
        # record-only camera wait the anchor, contribute to `max_image_latency`,
        # and, if it never published at all, block the run: that case is
        # deliberately outside the wait's cap. Recording is unaffected either
        # way, because P9 sweeps the arena itself and never looks at an anchor.
        observed = observation_cameras(type(policy), cfg)
        self.cameras = {name: rings["cameras"][name] for name in observed}
        #: The class's declaration, as resolved: which camera's instant is
        #: "now" is a fact about training, which a deploy file may
        #: override (`[policy] anchor_camera`) and the record then names.
        self.anchor_camera = anchor_camera(type(policy), cfg)

        # Joint and hand ranges, read once from the robot. None when no
        # hardware is supplied (tests driving the loop directly): the
        # finiteness, shape and hand checks still run (against the Dex-1's
        # travel), only the URDF limit checks are skipped.
        self.limits = Limits.from_robot(hw) if hw is not None else None
        self.arm_idx = np.asarray(hw.joints_arm if hw is not None else _ARM_IDX,
                                  dtype=np.intp)
        self.waist_idx = np.asarray(
            hw.joints_waist if hw is not None else _WAIST_IDX, dtype=np.intp)
        #: History reads, limited to what the policy declared.
        self.reader = history_reader(rings, policy)

        # `k_start_read = 2` is the first tick P6 will ever solve, and it is the
        # same `+2` P6 starts from (`p6_target.py`): neither can be changed
        # alone without breaking the mutual wait.
        self.k_start_read = 2
        self.k_next_inference = cfg.P

        #: The intervention counter as last seen. Seeded from the LIVE
        #: word rather than from zero, so a P5 that outlives a rollout in which
        #: somebody intervened does not read the value it inherits as a change.
        self._intervene = self.control.intervene

        # Half a frame period per camera — the threshold at which the frame in
        # hand is provably nearer to the trigger than any future one.
        # Computed once: `acquire_anchor` runs on the hot path and this is a
        # property of the rig, not of the tick.
        views = cfg.views
        self.half_period = {name: 0.5 * views[name].frame_period
                            for name in observed}

    # --- one iteration, in pieces ------------------------------------------

    def scan(self):
        """The **earliest** written tick in `[k_start_read, k_next_inference]`.

        Not `deadline(k_start_read)`: P6 may legitimately never write that tick
        — an IK solve that overran its budget leaves a hole — and P5 would wait
        on it forever. Not the latest either: P6 writes in increasing `k`, so
        the earliest is the first to exist, and taking the latest would mean
        waiting until the chunk is nearly exhausted, by which time
        `t_next_inference` may already have passed.
        """
        return self.body.earliest_written_in(self.k_start_read,
                                             self.k_next_inference)

    def schedule(self, entry) -> float:
        """Extrapolate the trigger instant off whatever tick the scan found.

        `deadline` is affine in `k` within one epoch, so **every** solved tick
        in the window yields a bit-identical result: this is extrapolation, not
        approximation. Its validity is exactly the claim that the window never
        spans two epochs — free in async, and true in sync only because chunks
        are truncated to `P`.
        """
        return entry.deadline + (self.k_next_inference - entry.k) * self.cfg.dt

    def produce(self, t_trigger: float):
        """Steps 3-4: observe, infer, and turn the actions into a chunk.

        This is the *body* of the loop, not the loop: something that replaces
        it (a test's scripted producer, a replay of recorded chunks) leaves
        the deadline read, the precise sleep, the chunk metadata and the write
        path exercised exactly as in a live run.

        Returns `(chunk, stamps)`. The stamps are taken HERE because these are
        the seams of the loop: the wait, the observation build, the `infer`
        call and the conversion. Recording them is what makes "inference
        latency" a decomposable number instead of one blob — with only the
        endpoints, a third of what looks like model latency can be the anchor
        waiting for a camera.
        """
        # The anchor is acquired HERE, not inside the policy: the infra fixes
        # the instant, the policy only reads what it depicts.
        anchor = acquire_anchor(self.cameras, self.state, self.anchor_camera,
                                t_trigger, self.half_period, now=self.clock.now)
        t_obs_received = self.clock.now()
        obs = observe(anchor, getattr(self.policy, "prompt", self.cfg.prompt))
        t_cpu_processed = self.clock.now()
        actions = self.policy.infer(obs, self.reader)
        # `infer` returns only once the result is on the host: JAX is async and
        # the `np.asarray` a policy ends with is the sync point. So this instant
        # is the device round trip closing, not the dispatch returning.
        t_chunk_received = self.clock.now()
        chunk = self.to_chunk(actions, anchor)
        return chunk, {"t_anchor": anchor.t_anchor,
                       "t_obs_received": t_obs_received,
                       "t_cpu_processed": t_cpu_processed,
                       "t_chunk_received": t_chunk_received,
                       "max_image_latency": anchor.max_image_latency()}

    def measured(self, anchor) -> dict:
        """The anchor's state sample as the deadline shift's row 0."""
        st = anchor.state
        q = np.asarray(st["q"], dtype=np.float64)
        hand = np.asarray(st["grip_meas"], dtype=np.float64).reshape(
            schema.N_HANDS, schema.HAND_DIM)
        return {"flange": np.asarray(st["flange"], dtype=np.float64),
                "arm_q": q[self.arm_idx].reshape(schema.N_HANDS, -1),
                "waist_q": q[self.waist_idx],
                "hand": hand[:, 0]}

    def to_chunk(self, actions, anchor) -> Chunk:
        """The policy's actions -> the chunk P6 will read.

        Validate (the POLICY's output, so every error names the policy), shift
        onto the real deadlines, clip what the shift's own arithmetic may have
        pushed past a range, pack the hand block, and cut to `P` in sync —
        after the shift, so its extrapolated tail is never commanded.
        """
        # Up to H: a policy that outputs more (its own full horizon, run with a
        # shorter `action_horizon`) is cut, not refused.
        actions = first_actions(actions, self.cfg.H)
        validate_actions(actions, self.mode, self.cfg, self.limits)
        if anchor is not None and self.policy.deadline_shift:
            slots = deadline_shift_slots(anchor.t_anchor, anchor.t_trigger,
                                         self.cfg.dt)
            actions = shift_to_deadline(
                actions, self.measured(anchor), slots,
                point=self.policy.deadline_shift_point)

        # THE CLIPS. Row 0 of the shift's grid is a MEASUREMENT, and the robot
        # can stand a hair outside a URDF bound (the bow rests on a stop at
        # its lower limit), so an interpolated action can leave a range the
        # policy's own output respected. Clipping is right for that and only
        # that: anything the policy itself put outside was refused above.
        hand = hand_block(actions)
        hand[:, :, 0] = np.clip(hand[:, :, 0], *hand_range(self.limits))
        if self.mode == EEF:
            chunk = Chunk(hand=hand, pose=actions.pose)
        else:
            arm_q, waist = actions.arm_q, actions.waist_q
            if self.limits is not None and self.limits.arm is not None:
                lim = self.limits.arm.reshape(schema.N_HANDS, -1, 2)
                arm_q = np.clip(arm_q, lim[..., 0], lim[..., 1])
            if self.limits is not None and self.limits.waist is not None:
                waist = np.clip(waist, self.limits.waist[:, 0],
                                self.limits.waist[:, 1])
            chunk = Chunk(hand=hand, arm_q=arm_q, waist=waist)
        # Sync truncates to exactly P; the window extrapolation in `step`
        # depends on it. The cut rows stay on `chunk.plan` for the lookahead.
        return chunk.truncate(self.cfg.P) if self.cfg.sync else chunk

    def publish(self, produced, *, k_first_action: int,
                t_trigger: float) -> int:
        """Step 5. The metadata and the payload layout are both P5's now."""
        chunk, stamps = produced
        stamps = dict(stamps)
        max_image_latency = stamps.pop("max_image_latency", 0.0)
        validate_chunk(chunk, self.cfg)
        chunk_id = self.control.newest_chunk_id + 1
        # Stamped before the write, not after: this value lives INSIDE the slot
        # the write is about to fill, so it can only mean "the write begins".
        # The ~11 KB payload copy and the id store fall outside it.
        t_chunk_written = self.clock.now()
        self.write(chunk,
                   chunk_id=chunk_id,
                   k_first_action=k_first_action,
                   t_trigger=t_trigger,
                   t_chunk_written=t_chunk_written,
                   **stamps)
        # LAST, after the payload: `newest_chunk_id` is P6's only signal that a
        # chunk exists, and a slot it names must already be readable.
        self.control.newest_chunk_id = chunk_id
        self._report(chunk_id, t_trigger, t_chunk_written, stamps,
                     max_image_latency)
        return chunk_id

    def write(self, chunk: Chunk, *, chunk_id: int, k_first_action: int,
              **stamps: float) -> None:
        """Lay the chunk into the ring.

        `n_actions` comes from the chunk, not from the config: it is `H` in
        async and `P` in sync, and P6 cannot test coverage without it.

        `n_plan` is `n_actions` plus `cfg.plan_lookahead_ticks` rows of the tail
        `truncate` kept — published, and never executed. It is a SECOND count
        precisely so that the first one keeps its meaning: nothing that asks
        "will this tick run" gets a different answer because the planner was
        shown more. At the default 0 the tail is not written at all.

        `stamps` passes through unread: the six replan instants, by name, so the
        schema is the one place their names and order are decided.
        """
        blocks = {name: getattr(chunk, name)
                  for name in ("hand", "pose", "arm_q", "waist")}
        n_plan = chunk.n
        extra = int(getattr(self.cfg, "plan_lookahead_ticks", 0))
        if extra and chunk.plan is not None and chunk.plan.n:
            m = min(extra, chunk.plan.n)
            blocks = {name: (None if value is None else np.concatenate(
                          [value, getattr(chunk.plan, name)[:m]]))
                      for name, value in blocks.items()}
            n_plan = chunk.n + m
        self.chunks.write(chunk_id=chunk_id, k_first_action=k_first_action,
                          n_actions=chunk.n, n_plan=n_plan, **blocks, **stamps)

    def _report(self, chunk_id, t_trigger, t_written, stamps, max_image_latency):
        """One line per chunk, on the operator's terminal.

        Here — after the ring write and before the next sleep — because the
        chunk is already published, so P6 has it and this costs the run nothing:
        it lands in the slack that `sleep_until` would otherwise spend idle.

        Four numbers, each answering a question nothing else answers live:

        * **`total`** — `t_chunk_written - t_trigger`. Everything P5 owns: the
          anchor wait, the host preprocessing, the device round trip and the
          decode. This is what the replan period has to fit, so it is the number
          `replan_period_ticks` is sized against.

        * **`inference`** — `t_chunk_received - t_cpu_processed`. The device
          round trip. **Not "GPU":** for a network it is the host->device
          transfer, the model, and the device->host sync, and they cannot be
          separated from outside the adapter (a framework that dispatches
          asynchronously would need a sync between the transfer and the model,
          which costs real time on the hot path to measure time). Calling this
          "GPU" would be a name that quietly includes something else.

        * **`obs`** — `t_anchor - t_trigger`, SIGNED, and the sign is the point.
          Negative means the anchor took a frame that already existed at the
          trigger and waited for nothing; positive means it waited for a fresher
          one. Bounded by +/- T/2 either way. It says which branch of the
          nearest-frame rule fired, which is otherwise invisible.

        * **`image latency`** — how far the worst camera's capture time sits
          from the anchor, in EITHER direction. It answers "do these images
          depict the same instant", which is a question about the rig, where
          `obs` is a question about the schedule. It is also the only symptom a
          stopped camera produces: a dead one contributes the same frozen frame
          forever without hanging anything, so this number simply grows.
        """
        total = (t_written - t_trigger) * 1e3
        inference = (stamps["t_chunk_received"] - stamps["t_cpu_processed"]) * 1e3
        obs = (stamps["t_anchor"] - t_trigger) * 1e3
        print(f"[P5] chunk {chunk_id}: total {total:6.1f} ms  "
              f"inference {inference:6.1f} ms  obs {obs:+6.1f} ms  "
              f"image latency {max_image_latency * 1e3:5.1f} ms", flush=True)

    # --- the loop ----------------------------------------------------------

    def first_inference(self) -> int:
        """The inference that FOUNDS a tick line. Two callers.

        It cannot be an ordinary iteration: the scan window is empty and there is
        no solved tick to extrapolate a trigger instant from, because P6 has
        nothing to solve until this chunk exists. So the trigger instant is `now`
        and the cursors are re-derived from the live cursor afterwards.

        **`k_exec` rather than a literal zero, and that is what makes the same
        method serve the handback**. At the start of a rollout it IS
        zero — homing's ramp ends at tick 0 and P7 has latched it — so this
        produces `chunk_id = 1` with `k_first_action = 1` exactly as it always
        did. After an intervention it is wherever the operator's ticks left it,
        and re-basing on it is precisely what stops the deadlock: the ordinary
        schedule extrapolates off a written body target, there are none on the
        line P6 is about to anchor to, and waiting for one is mutual — P6 cannot
        write until a chunk covers its cursor and no chunk will until this runs.

        It is read AFTER `produce`, not before, and that ordering is the handback
        case being correct: at the instant the operator lets go, P7 is still
        draining the last `intervene_lead_s` of their targets and `k_exec` is
        still moving. An inference takes ~120 ms; by the time this reads it, it
        has been still for most of them.

        Tick `k_base + 1` is never executed: P6's first solve is `k_exec + 2`.
        That is the price of a dt of solve margin and it is paid here.
        """
        t_trigger = self.clock.now()
        produced = self.produce(t_trigger)
        k_base = self.control.k_exec
        self.k_start_read = k_base + 2
        self.k_next_inference = k_base + self.cfg.P
        return self.publish(produced, k_first_action=k_base + 1,
                            t_trigger=t_trigger)

    def step(self) -> int | None:
        """One iteration. Returns the published `chunk_id`, or None if the
        window held nothing yet.

        A None is a retry, not an error and not a timeout: P5 blocking forever
        on a persistently empty window is accepted behaviour.
        """
        counter = self.control.intervene
        if counter != self._intervene:
            self._intervene = counter
            if not intervene.is_intervening(counter):
                # THE HANDBACK. Resume by FOUNDING a tick line, not by
                # rebasing onto one — see `first_inference`.
                #
                # The condition is "the counter moved and is now even", NOT "it
                # was odd and is now even". P5 can miss the odd phase entirely: a
                # short intervention can begin and end inside one precise sleep,
                # so this may see 0 -> 2 with no intervening observation. Both
                # mean the same thing — a human moved the arm and the tick line
                # this loop remembers is dead — and re-founding is the right
                # answer to either.
                return self.first_inference()

        if intervene.is_intervening(counter):
            # No inference at all while a human is driving. The chunks would not
            # be executed — P6 is in its guided branch for as long as the counter
            # is odd — and producing them anyway would leave one in the ring whose
            # observation predates the correction, which P6 would then adopt at
            # the handback. Waiting is both simpler and the only thing that
            # cannot be wrong.
            self._wait_out()
            return None

        entry = self.scan()
        if entry is None:
            return None
        t_next_inference = self.schedule(entry)
        self.clock.sleep_until(t_next_inference)
        produced = self.produce(t_next_inference)
        if self.control.intervene != self._intervene:
            # THE WORLD MOVED UNDER THIS INFERENCE. The observation
            # it was built from predates a human correction, and the tick line it
            # was scheduled against may no longer be the one anybody is walking.
            # Dropped, cursors untouched, because the next iteration re-founds
            # the line from `k_exec` anyway.
            #
            # **The test is "changed", not "is intervening", and that difference
            # is a deadlock.** This sleep can be most of a replan period, so a
            # whole intervention — takeover AND handback — can happen inside one
            # of them: P5 would wake to an even counter, see nothing wrong, and
            # publish a chunk against cursors from before the takeover. P6 anchors
            # at `k_exec + 2`, which that chunk does not cover, so it reports
            # `uncovered` forever, writes nothing, and P5's own window scan then
            # waits for a target P6 will never write. Both processes wait on each
            # other with the arm held.
            #
            # Dropping rather than stamping the chunk: P6 ignores
            # `newest_chunk_id` while the counter is odd, but at the handback it
            # returns to the policy path, sees an id it has not adopted, and would
            # anchor to a chunk inferred before the human corrected anything.
            # Refusing to publish costs one `if`; teaching the chunk ring to carry
            # the counter would cost three call sites for the same guarantee.
            return None
        chunk_id = self.publish(
            produced,
            # +1, not +2: P6 starts at k_exec + 2 and therefore never executes
            # a chunk's action 0. That is the price of a dt of solve
            # margin, and it is paid here.
            k_first_action=self.k_next_inference + 1,
            t_trigger=t_next_inference)
        if not self.cfg.sync:
            self._warn_if_late(chunk_id, produced[0].n, t_next_inference)
        self.k_next_inference += self.cfg.P
        self.k_start_read += self.cfg.P
        return chunk_id

    def _warn_if_late(self, chunk_id: int, n: int, t_trigger: float) -> None:
        """Say so when an async chunk landed too late to be used as planned.

        P0 refuses an async START whose measured latency cannot keep up
        (`validate.check_async_timing`), but the latency can drift during a
        rollout, and what follows is otherwise silent: an inference longer than
        the replan period puts the next trigger in the past, and a chunk whose
        every tick has passed is one P6 cannot use, so the arm simply holds.
        """
        took = self.clock.now() - t_trigger
        period = self.cfg.P * self.cfg.dt
        # By TIME, not by `k_exec`: row j is due at `t_trigger + (j+1)·dt`, and
        # P6 needs two ticks of margin. When the arm has starved, `k_exec`
        # itself has stopped, so a test against it would never fire.
        if took + 2 * self.cfg.dt >= n * self.cfg.dt:
            print(f"[P5] WARNING chunk {chunk_id} landed after all {n} of its "
                  f"ticks had passed ({took * 1e3:.0f} ms after its trigger); "
                  f"nothing in it can run and the arm is holding. Inference is "
                  f"slower than the replan period ({period * 1e3:.0f} ms): "
                  f"stop, and raise replan_period or run sync", flush=True)
        elif took > period:
            print(f"[P5] WARNING chunk {chunk_id} took {took * 1e3:.0f} ms, "
                  f"longer than the replan period ({period * 1e3:.0f} ms); the "
                  f"next one is already late", flush=True)

    # --- intervention -------------------------------------------

    def _wait_out(self) -> None:
        """Idle until the operator hands back, or until the run ends.

        The session test is not decoration: without it a P5 parked here would
        never notice P0 moving the session on, and EXITING would hang on a
        process waiting for a headset. `run`'s own `while session == ACTIVE` is
        one level too far out to help, because this does not return until the
        counter moves.
        """
        while intervene.is_intervening(self.control.intervene):
            if self.control.session != schema.ACTIVE:
                return
            self.clock.sleep_until(self.clock.now() + _POLL_S)

    def run(self) -> None:
        while self.control.session == schema.ACTIVE:
            self.step()


# --- warmup -----------------------------------------------------


def stabilised(took) -> bool:
    """Has the dummy inference settled? P5's own answer, and nobody else's.

    Warmup runs "until latency stabilises", and P0, which owns the gate,
    cannot time an inference. This process can, so the rule lives
    here — which is the whole reason `warm` is a flag rather than a number. A
    count would only move the judgement back into P0.

    The cap bounds WORK, not time. A machine whose latency genuinely wanders
    would never satisfy the tolerance, and a run must not hang at a prompt that
    never prints; but a timeout would have to encode how long a compile takes,
    which is exactly the quantity nobody can know in advance.
    """
    if len(took) >= WARM_MAX:
        return True
    if len(took) < WARM_MIN:
        return False
    a, b = took[-2], took[-1]
    return abs(b - a) <= WARM_TOL * max(a, b)


def warmup(control, policy, dummy, *, clock, reader=None, latch=None):
    """Dummy inferences until this process says it is ready, then until START.

    Not optional: JAX/XLA compiles on the first call (tens of seconds,
    shape-dependent), CUDA kernels load lazily, allocators grow on demand — the
    first inference can be 10-100x steady state, and without this the run's
    first replan blows every deadline in the schedule and the robot lurches.

    **`control.warm` is what makes that a guarantee rather than a hope.** If
    an operator could start before any dummy inference had run, the run's
    first *real* inference would pay the compile — measured once at 29.9 s of
    frozen robot, on an observation 29.9 s stale by the time it executed. P0
    waits on this word before it will accept a Start, so the operator cannot
    start early.

    **The word says "ready", not "how many".** What counts as warm is a property
    of this policy on this machine, so it is decided here and published as
    one bit; P0 waits for it and never interprets it. A count would have put the
    threshold in P0, which cannot know it. P0 still owns every `session` store —
    P5 writing the state word directly would give that word two writers and break
    the one-writer rule for nothing (a lost update against P0's EXITING would walk the session
    *backwards*, and this loop would then never exit).

    The loop runs while the session is anything before ACTIVE — not the two
    named states — so that this process may be spawned at any point after the
    arena exists and warm through CHECK and HOME as well. That is the whole
    reason P5 can now overlap homing: nothing here depends on which
    pre-run phase P0 is in, only on the run not having started.

    Warming continues after the flag is set. The GPU is idle until Enter anyway,
    and the timings collected in that window are the only measurement of steady-
    state inference latency that exists BEFORE a run — which is the number that
    sizes `replan_period_ticks`. Printing it here is why it is collected.

    **`dummy` is the infra's, built once in `run` from the observed cameras'
    shapes and the state layout** (`policy_contract.dummy_observation`), so no policy
    carries its own. It is a parameter rather than something this function
    builds because `run` calls this once per rollout and the policy's own
    `warmup()` — the compile — must happen exactly once, before the first call.

    A policy fast enough to finish a dummy inside `_POLL_S` (the hold and toy
    examples, or any policy with no model) is paced to one per `_POLL_S` rather
    than spun: an unpaced loop of free inferences would hold P5's core at 100%
    through homing, and CPU contention is a measured cause of a starved
    control loop.

    Re-entered after a reset this returns almost immediately: `control.warm` is
    already 1, so the guard below never fires again and the loop simply keeps the
    GPU busy through HOME and READY until the next START moves the session to
    ACTIVE. The timings it collects are fresh each time, and printing them per
    rollout is a small ongoing measurement of whether steady-state latency has
    drifted over a long session.

    `latch` (`RolloutLatch`) is polled between dummies: it judges prompt
    requests and takes a new rollout generation. Returns the dummy
    observation as the latch left it — carrying the current prompt.
    """
    took: list[float] = []
    while control.session < schema.ACTIVE:
        if latch is not None:
            dummy = latch.poll(dummy)
        t0 = clock.now()
        policy.infer(dummy, reader if reader is not None else policy.reader())
        took.append(clock.now() - t0)
        if latch is not None and len(took) > 1:
            # Not the first of each warmup: after a reset it can still carry
            # a lazy allocation, and it is what `steady` below leaves out too.
            latch.observe(took[-1])
        if took[-1] < _POLL_S:
            clock.sleep_until(clock.now() + _POLL_S)
        if not control.warm and stabilised(took):
            # The payload before the flag, as everywhere else in this system:
            # the operator reads the reason, then P0 reads the permission.
            print(f"[P5] warm after {len(took)} inferences "
                  f"(first {took[0] * 1e3:.0f} ms, "
                  f"last {took[-1] * 1e3:.1f} ms)", flush=True)
            control.warm = 1

    if len(took) > 1:
        # Everything but the compile. This is the latency the schedule is built
        # around, reported before the run rather than dug out of
        # `chunks.bin` afterwards.
        steady = sorted(took[1:])
        print(f"[P5] warmup: {len(took)} inferences, steady-state "
              f"p50 {steady[len(steady) // 2] * 1e3:.1f} ms, "
              f"max {steady[-1] * 1e3:.1f} ms", flush=True)
    return dummy


class RolloutLatch:
    """P5's half of the rollout settings: prompts judged, settings taken, one
    per generation (`config_loading/rollout_settings.py`).

    The policy lives in this process, so two jobs that are about the policy
    land here, both run between warmup's dummy inferences (`poll`), i.e.
    whenever the session is before ACTIVE:

    * **Judging prompt requests.** Only the policy owns a tokenizer, so the
      dashboard's request is checked here with `policy.check_prompt_length` and the
      verdict written back (`Control.judge`) — within one dummy inference of
      the submit, which is what the dashboard shows. P0 refuses a START whose
      newest prompt request is not ACCEPTED.
    * **Taking a new generation.** When P0 accepts a START it bumps
      `rollout_generation`; this reads the active block (never the request),
      hands the policy what changed — `set_prompt` for a new prompt,
      `on_settings` for new settings — runs ONE dummy inference so any
      recompile is paid now, and reports the generation applied. P0 moves the
      session to ACTIVE only after that. If the policy raises, the generation is
      reported with `APPLIED_FAILED` and P0 refuses the START.

    `cfg` is the latched rollout config: what the next `PolicyLoop` is built
    from, so `P`, `sync` and `dt` hold for the whole rollout.

    It also keeps `Control.latency_s` current (`observe`): the recent dummy
    inferences' 90th percentile plus `anchor_wait`, the longest the anchor can
    wait for a fresher frame. P0 checks an async rollout against it at START.
    When a START changes the settings of an async rollout, the old samples say
    nothing about the new ones (a different `sample_steps` is a different
    amount of work), so `_take` re-measures before it reports the generation
    applied.
    """

    def __init__(self, control, policy, cfg, *, clock, reader=None,
                 anchor_wait: float = 0.0):
        from elrobot_deploy.config_loading.rollout_settings import RolloutSettings
        self.control = control
        self.policy = policy
        self.launch = cfg
        self.cfg = cfg
        self.clock = clock
        self.reader = reader
        self.generation = 0
        #: What the policy was last told — the launch values until the first
        #: START — so `on_settings` fires on a change, not on every rollout.
        self.settings = RolloutSettings.of(cfg)
        self._judged = 0
        self.anchor_wait = float(anchor_wait)
        self.samples: deque[float] = deque(maxlen=LATENCY_WINDOW)

    def observe(self, took: float) -> None:
        """One steady-state dummy inference, seconds; republishes the estimate."""
        self.samples.append(took)
        ordered = sorted(self.samples)
        q = ordered[int(LATENCY_QUANTILE * (len(ordered) - 1))]
        self.control.latency_s = q + self.anchor_wait

    def _remeasure(self, dummy) -> None:
        """Steady-state latency under settings that just changed: dummies until
        two agree (`stabilised`), none of the old samples kept."""
        self.samples.clear()
        took: list[float] = []
        reader = self.reader if self.reader is not None else self.policy.reader()
        while not took or not stabilised(took):
            t0 = self.clock.now()
            self.policy.infer(dummy, reader)
            took.append(self.clock.now() - t0)
            self.observe(took[-1])

    def poll(self, dummy):
        """Judge a pending prompt, take a new generation. Returns the dummy
        observation to warm with, which carries the policy's current prompt."""
        self._judge()
        if self.control.rollout_generation != self.generation:
            dummy = self._take(dummy)
        return dummy

    def _judge(self) -> None:
        seq, text = self.control.prompt_request()
        if seq == 0 or seq == self._judged:
            return
        try:
            self.policy.check_prompt_length(text)
            verdict, reason = schema.PROMPT_ACCEPTED, ""
        except Exception as exc:                          # noqa: BLE001
            verdict, reason = schema.PROMPT_REFUSED, str(exc) or type(exc).__name__
        self.control.judge(seq, verdict, reason)
        self._judged = seq
        print(f"[P5] prompt {text!r}: "
              f"{schema.PROMPT_VERDICT_NAMES[verdict]}"
              f"{' — ' + reason if reason else ''}", flush=True)

    def _take(self, dummy):
        import dataclasses

        from elrobot_deploy.config_loading.rollout_settings import RolloutSettings, latched
        from elrobot_deploy.policy_contract.base import deliver_settings
        cfg, prompt, gen = latched(self.control, self.launch)
        try:
            if prompt != getattr(self.policy, "prompt", None):
                self.policy.set_prompt(prompt)
            settings = RolloutSettings.of(cfg)
            changed = settings != self.settings
            if changed:
                self.policy.on_settings(settings)
                self.settings = settings
            deliver_settings(self.policy, settings)
            dummy = dataclasses.replace(dummy, prompt=prompt)
            t0 = self.clock.now()
            self.policy.infer(dummy, self.reader if self.reader is not None
                              else self.policy.reader())
            took = self.clock.now() - t0
            if changed and not cfg.sync:
                # That one paid any recompile; these measure what is left.
                # Async only: it is the only mode whose START checks latency,
                # and a sync START should not wait for dummies it never uses.
                self._remeasure(dummy)
            else:
                self.observe(took)
        except Exception:                                 # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"[P5] the policy refused rollout {gen}'s settings/prompt; "
                  f"P0 will refuse the START", flush=True)
            self.generation = gen
            self.control.settings_applied = gen | schema.APPLIED_FAILED
            return dummy
        self.cfg = cfg
        self.generation = gen
        self.control.settings_applied = gen
        print(f"[P5] rollout {gen} settings taken (one dummy inference "
              f"{took * 1e3:.0f} ms)", flush=True)
        return dummy


def apply_policy_env(cfg, environ=None) -> dict:
    """`cfg.policy_env` into the environment. Returns what was actually changed.

    Must run before the adapter's framework is imported in this process: e.g.
    XLA reads its allocator flags once, when the backend is initialised, and a
    value set after that is read by nothing and reported by nothing.

    Overwrites rather than `setdefault`. A shell profile may export these for
    the whole shell, so a `setdefault` here would silently
    lose every time and the deploy file's value would be advisory — which is the
    one thing a config field must never be.
    """
    if environ is None:
        environ = os.environ
    changed = {}
    for key, value in (cfg.policy_env or {}).items():
        before = environ.get(key)
        environ[key] = str(value)
        if before != str(value):
            changed[key] = (before, str(value))
    return changed


def run(shm_name: str, cfg, hw, *, policy=None) -> None:
    """P5's process entry point.

    Every import that costs a CUDA or DDS context happens here, in the child,
    because the processes are spawned and never forked. `policy_env` is
    applied FIRST, ahead of every one of them.
    """
    apply_policy_env(cfg)

    from elrobot_deploy.runtime import timing as clock
    from elrobot_deploy.runtime.cpu_pinning import pin
    from elrobot_deploy.runtime.shared_memory import (Arena, build, open_all,
                                                  open_read_only)
    from elrobot_deploy.config_loading.derive import derive
    from elrobot_deploy.policy_contract import (DummyReader, bind_runtime,
                                            camera_info, dummy_observation,
                                            make_policy)

    arena = Arena.attach(shm_name, build(cfg, derive(cfg, hw)))
    try:
        pin(hw, "p5_policy")
        rings = open_all(arena, cfg)
        # `reader.arena`: every region, read-only, for the policy's own reads.
        rings["read_only"] = open_read_only(arena, cfg)
        control = rings["control"]
        # ONCE, for the life of this process — the construction (config,
        # tokenizer) and the weight restore + compile (its `warmup`) that every
        # rollout after the first must not pay again.
        if policy is None:
            policy = make_policy(cfg, hw)
        cams = observation_cameras(type(policy), cfg)
        views = cfg.views
        dummy = dummy_observation({c: views[c] for c in cams}, cfg.prompt)
        # The warmup reader is a DUMMY too, and a full one: every declared
        # history at its real length and rate, ending at the dummy's t_anchor,
        # so the graph compiled here is the one the run executes. The live
        # reader is `PolicyLoop.reader`, bound to the rings.
        reader = DummyReader(views, dict(policy.history or {}),
                             state_hz=hw.lowstate_hz if hw is not None else 1000.0)
        from elrobot_deploy.config_loading.rollout_settings import RolloutSettings
        bind_runtime(policy, dummy, reader, camera_info(cfg),
                     RolloutSettings.of(cfg))
        t0 = clock.now()
        policy.warmup()
        print(f"[P5] {type(policy).__name__}.warmup() took "
              f"{(clock.now() - t0) * 1e3:.0f} ms", flush=True)

        # One iteration per rollout. `warmup` blocks while the session is
        # anything before ACTIVE, so it is already the shape this loop needs: on
        # a reset P0 moves the session HOME → READY and this simply waits there,
        # warm, until the next START.
        # Prompt verdicts and per-rollout settings, between dummies.
        latch = RolloutLatch(control, policy, cfg, clock=clock, reader=reader,
                             anchor_wait=max(0.5 * views[c].frame_period
                                             for c in cams))
        while True:
            dummy = warmup(control, policy, dummy, clock=clock, reader=reader,
                           latch=latch)
            if control.session != schema.ACTIVE:
                return                  # EXITING or STOPPED: the session is over
            # P0 moves to ACTIVE only once this process has taken the new
            # generation, so this is a no-op on the real path; it is here so a
            # START that somehow skipped the handshake still runs the settings
            # it names rather than the previous rollout's.
            dummy = latch.poll(dummy)

            # P5's own word, zeroed by P5 — P0 zeroes `k_exec` during
            # reset only because P7 is dead for that instant, and no such window
            # exists here. Chunk ids restart at 1 with `first_inference`, and the
            # previous rollout's slots are inert rather than stale: `covering`
            # answers by which ticks a chunk spans, and those span ticks this
            # rollout will never reach.
            control.newest_chunk_id = 0

            # A NEW loop per rollout, not a reused one. `k_start_read = 2` and
            # `k_next_inference = P` are the startup values, and they are
            # correct again because reset put `k_exec` back to 0 — carrying the
            # previous rollout's cursors into this one would ask for a window
            # five figures above the tick the robot is actually executing.
            #
            # Built from the LATCHED config: this rollout's `P`,
            # `sync` and `dt`, held until it ends whatever is requested meanwhile.
            loop = PolicyLoop(latch.cfg, rings, policy, hw=hw, clock=clock)
            loop.first_inference()
            loop.run()
    finally:
        arena.close()


def history_reader(rings, policy) -> Reader:
    """The live reader: the state ring and every camera ring the
    policy declared history for — nothing it did not declare."""
    history = dict(policy.history or {})
    cams = {c: rings["cameras"][c] for c in history if c in rings["cameras"]}
    return Reader(rings["state"], history, cams, arena=rings.get("read_only"))


__all__ = ["PolicyLoop", "RolloutLatch", "history_reader", "run", "stabilised",
           "warmup"]
