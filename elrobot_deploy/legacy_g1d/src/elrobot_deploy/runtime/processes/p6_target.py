"""P6 — chunks in, joint targets out.

One tick per iteration, as fast as the IK solves. P6 owns `last_read_chunk_id`,
`k_to_solve`, and the epoch `(t_base, k_base)` that every deadline in the system
is derived from; it writes the target ring and nothing else. It reads the chunk
ring, the target ring (for the seed) and the state ring (for the trunk the IK
solves against). Spawned per rollout, under elrobot_deploy's own interpreter.

Design: `docs/overview.md`, "One chunk's life" (step 5);
`docs/infra_logic/ik_solvers.md` (the solvers it calls) and
`docs/infra_logic/chunk_blending.md` (the blend it applies).

Four things in this loop are subtle enough to be worth stating before the code.

**P6 owns the waist**. An eef chunk carries no waist at all, so
the commanded trunk is P6's decision: with the pure arm IK it is `OWN_WAIST`
(zero, upright), and with a trunk planner (a waist-mode solver, `runtime/ik/`) it is the plan.
A joint-mode chunk always commands the waist: `waist_q` is required there, and
P6 sends it as given.

**Joint mode skips only the IK**. Its `arm_q` IS the solution: no
solve, no residual (`err` is NaN, as for any tick no IK produced), and every
other step — the cursor, the epoch, the blend, the velocity clamp against the
seed, the write gate, the one ring write — is the same code, so P7 cannot tell
a joint-mode target from an eef one.

**The IK solves against the MEASURED trunk; the wire is commanded the waist P6
owns.** Those are two different questions. The chunk's poses are absolute, in a
base frame that sits BELOW the bow (`robot/g1d_model.py::BASE_BODY`) and
therefore does not move when the trunk does — so
"where must the arm joints be" depends on the trunk the arm is actually hanging
off *at this instant*, not on the trunk anybody asked for. On the reference
robot the two differ permanently: the trunk rests on a physical obstruction
0.039 rad forward of its commanded zero and no torque moves it, so a solve
against the commanded trunk places the arm in a frame tilted 2.25 deg from the
real one — 8-11 mm at the flange, 13-15 mm at the fingertips, always forward
and down, on every tick.

The measurement is read fresh each tick rather than carried on the chunk. The
chunk's own anchor waist would be nearly as good (the bow moves 0.0002 rad
across a whole rollout) but it is the wrong quantity in principle and it goes
stale by up to a replan period — which the yaw, with a measured 0.022 rad
transient, is large enough to notice.

Both trunks are recorded, and they have to be: the target's `q[14:17]` is the
commanded one because that is what goes on the wire, and `waist_ik` beside it is
the one the arm joints mean anything in. A record carrying only the first cannot
be checked at all — `FK(q[:14], q[14:17])` is not the `pose` stored next to it,
and the gap is the droop rather than the residual `err` claims. Nothing in this
loop pays for it: the array is the one `waist_now()` already returned.

**The cursor advance is conditional, and that is what makes one loop work in
both modes.** Naively tracking `k_exec` deadlocks in sync: `k` freezes during
inference, so the tick after the chunk's last one never appears and the loop
waits for a cursor that is waiting for it. Instead the cursor moves only onto a
tick covered by a chunk this loop has already ADOPTED — one whose `chunk_id` it
has read off `newest_chunk_id` — and a new chunk re-anchors it to `k_exec + 2`
outright.

That "adopted" qualifier is the difference between the two branches meaning
different things and meaning the same thing by accident. Only the new-chunk
branch re-stamps the epoch; the advance branch must therefore never be
able to reach a chunk the new-chunk branch has not seen yet
(`ChunkRing.adopted_through`). Coverage goes true when P5's payload lands and
the id arrives after it; an advance that consulted coverage alone would walk
onto each new chunk's first tick one iteration early, with the previous
chunk's epoch still stamped — a deadline ~100 ms in the past (measured on 36
of 37 chunks before the limit existed). Two rules rest on this cursor: a
chunk's action 0 is never executed, and P5's scan window never spans two
epochs.

**`k_exec` is read twice per iteration, on purpose, and the two reads mean
different things.** The one at the top is captured *once* and is what
`k_to_solve` and the sync epoch re-stamp are both derived from — re-reading it
for the stamp would break `deadline(k_exec + 2) = now + 2·dt`, which is the
whole point of the re-stamp. The one at write time is deliberately fresh:
the write gate's question is whether the value could perturb the segment P7 is
*currently* mid-interpolation on.

**There is a second source of targets, and everything after it is shared.**
While the operator is intervening the action comes from the
intervention ring instead of from a blend of chunks, and its deadline comes from
their clock instead of from the epoch. That is the entire difference: `_commit` —
the seed, the solve, the velocity clamp, the write gate and the two ring writes —
is the same code for both, so an operator's target is bounded, gated and recorded
by exactly the machinery a policy target is, and the two are directly comparable
in the record. That comparability is the DAgger pair; a second write path would
have destroyed it.

Two things make that possible with so little. The rows are decimated to the
ACTION rate, so `k` keeps meaning what it meant and every piece of arithmetic
built on it survives untouched. And a takeover CLEARS NOTHING — the slots carry
the counter and become unreachable rather than absent — so `latest_below` still
finds the last policy posture and the first operator solve needs no fallback
seed. Without `--intervene` the counter never leaves 0, `is_intervening` is never
true, and none of this is reachable.

**Abandoning a tick means writing nothing.** Not a partial write, not a
best-effort joint angle: a slot believed unsolved is never published. The tick
becomes a hole, which P7 skips by construction (it only ever selects a future
deadline) and P5's window scan tolerates. This is also why the one ring
write sits at the very end, past every branch that can fail.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from elrobot_deploy.runtime import cpu_pinning as affinity, timing as clock
from elrobot_deploy.runtime.execution import chunk_blend as blend
from elrobot_deploy.runtime.teleop import intervention as intervene
from elrobot_deploy.runtime.shared_memory import Arena, build, open_all, schema
from elrobot_deploy.config_loading.derive import derive


#: How early a guide row may be and still be taken as its slot's (seconds).
#: Intervention only — see `Target._guided`. It is a tie-break for a row landing
#: a hair before its own grid point, not a resampling policy: the reference it
#: is compared against is the GRID, so anything between the PICO's arrival
#: jitter (tens of microseconds) and its frame (~11 ms at the 90 Hz the device
#: delivers) selects the same rows. Half a frame makes the rule "the row nearest
#: this slot", and leaves the most room for the device's clock to wander against
#: ours without a slot being skipped.
#:
#: It is also, therefore, how much SHORTER than `dt` one guided segment may be,
#: which is the span `_commit`'s velocity clamp assumes — so that clamp can be
#: up to ~18% loose for the one segment on which the phase shifts. P7's
#: per-publish slew limit is what actually bounds the wire, and it is unchanged.
GUIDE_SLACK_S = 0.005

#: The waist P6 commands when nobody else does — an eef chunk under the pure arm
#: IK. Upright, in
#: `kinematics.WAIST_JOINT_NAMES` slot order. The eef contract has no waist,
#: and the arm is solved against the MEASURED trunk regardless
#: (`solve_frame`), so this is the command, never the frame.
OWN_WAIST = np.zeros(schema.WAIST_DOF)


class Step(NamedTuple):
    """What one iteration did. Returned so the loop body is testable a tick at
    a time; the process wrapper ignores it."""

    k: int | None           # the tick chosen, None if the iteration chose none
    written: bool
    deadline: float | None
    reason: str             # idle | uncovered | unseeded | abandoned | gated | written


class Target:
    """The body of P6's loop, one `step()` per tick."""

    def __init__(self, rings, cfg, hw, solver, blend_fn=None, *, now=clock.now):
        self.cfg = cfg
        self.dt = float(cfg.dt)
        self.control = rings["control"]
        self.chunks = rings["chunks"]
        # `targets` is the writer's view; `body` is the same region opened as
        # a reader over the `q` column, which is what `latest_below` seeds from.
        self.targets = rings["targets"]
        self.body = rings["body"]
        # Read-only, and only for the trunk the IK is solved against. P6 still
        # writes the body and hand rings and nothing else — the invariant is
        # about WRITES, and this adds no writer to anything.
        self.state = rings["state"]
        # Which columns of the measured `q` are the waist, in the (3,) slot
        # order `kinematics.WAIST_JOINT_NAMES` uses. Resolved once: this is on
        # the hot path and the robot file cannot change mid-run.
        self.waist_cols = np.asarray(hw.joints_waist, dtype=np.intp)
        self.solver = solver
        self.blend = blend.get(cfg.blend if blend_fn is None else blend_fn, cfg)
        self._now = now
        # Only the arm is clamped here: the waist passes through from the blend
        # untouched by the IK, so there is no solver excursion to bound.
        # P7's last-defence clamp covers every published joint.
        self.qdot_max_arm = np.asarray(hw.qdot_max_body[:schema.ARM_DOF],
                                       dtype=np.float64)

        self.last_read_chunk_id = 0     # vs a first chunk_id of 1
        self.k_to_solve: int | None = None
        self.t_base: float | None = None
        self.k_base: int | None = None
        # sync only: the `k_exec` captured for the current chunk, held until
        # that chunk's first solve SUCCEEDS. The epoch is stamped after the
        # first solve succeeds, and a chunk whose first solve is abandoned
        # still needs its stamp — otherwise every deadline it produces is early
        # by the whole inference latency, and the very first chunk would leave
        # no epoch at all.
        self._epoch_pending: int | None = None

        #: How far ahead of `k_exec` this loop may solve. See `_within_lead`.
        #: 0 disables the bound. Read once — it cannot change mid-run.
        self.lead = int(cfg.solve_lead_ticks)

        #: `(k0, traj)` for the chunk being executed, or None. `traj` is one
        #: `(3,)` commanded waist per tick from `k0`, produced by the planner
        #: when the chunk landed. None means every tick takes the measured
        #: trunk, which is the deployed behaviour.
        self._plan: tuple[int, np.ndarray] | None = None

        # --- waist-mode IK (docs/infra_logic/ik_solvers.md) -----------------
        #
        # One boolean, read once, and answered by the config itself
        # (`DeployConfig.wbik`) so this loop and `run`'s solver choice cannot
        # disagree about whether the trunk is planned.
        self.wbik = bool(cfg.wbik)
        #: Does the chunk hook run? The whole-body tiers plan the trunk; a
        #: solver that plans its own thing per chunk (`arm_planner`, a research
        #: `module:factory`) says `plans_chunks`
        #: and returns None, which leaves the trunk measured.
        self.plans_chunks = self.wbik or bool(getattr(solver, "plans_chunks", False))
        #: How many ticks past `k_to_solve` the chunk path will walk looking for
        #: coverage.
        #:
        #: From the HORIZON, not the replan period. In sync a chunk carries `P`
        #: actions and the two agree, but `H` is what bounds what a chunk ring
        #: slot can physically hold, and a bound that can be shorter than the
        #: chunk in front of it silently truncates the plan: the walk stops, the
        #: remaining ticks are never solved, and the cursor lands mid-chunk where
        #: the per-tick advance branch picks them up one at a time against an
        #: epoch that was stamped for the whole. `+2` because the anchor is
        #: `k_exec + 2` while a chunk's own first action is `k_exec + 1`.
        #:
        #: The walk stops on coverage regardless, so an over-long bound costs
        #: nothing but one extra `covers_any` call.
        self._chunk_span = int(cfg.action_horizon) + 2
        #: Show the trunk planner this many ticks past the chunk it solves
        #: (`cfg.plan_lookahead_ticks`, 0 = off and the deployed behaviour).
        #: Read once here so the branch in `_plan_trunk` is a plain int test
        #: rather than a config lookup per chunk, and so a run that asked for
        #: the lookahead cannot have it change underneath it mid-rollout.
        self._plan_lookahead = int(getattr(cfg, "plan_lookahead_ticks", 0))

        # --- intervention -------------------------------------
        # Three fields, and none of them is a mode: the branch is taken on the
        # parity of a word P11 owns, so there is no state here that can disagree
        # with the rest of the system about who is driving.
        self.intervene_ring = rings["intervene"]
        self._intervene = self.control.intervene
        #: The guide row already solved, so a poll that finds no new one costs an
        #: idle iteration rather than a duplicate target.
        self._row_index = 0
        #: The deadline at which the next guide target is DUE — the decimation
        #: grid, not the last deadline published. See `_guided`.
        self._guided_due: float | None = None
        #: Async only: has an intervention voided the run's single epoch? See
        #: `_epoch_deadline`. False in sync, where every chunk re-stamps anyway.
        self._epoch_voided = False

    # --- one tick -----------------------------------------------------------

    def step(self) -> Step:
        # 0. WHICH WORLD. Read once per iteration, because every
        #    decision below depends on it and a value that changed mid-iteration
        #    would mix the two. P11 owns this word; this loop only ever reads it.
        counter = self.control.intervene
        if counter != self._intervene:
            self._intervene = counter
            self._switch()
        if intervene.is_intervening(counter):
            return self._guided(counter)

        # 1. which tick. The only place k_to_solve is chosen.
        newest = self.control.newest_chunk_id
        if newest != self.last_read_chunk_id:
            self.last_read_chunk_id = newest
            k_captured = self.control.k_exec        # captured ONCE (the epoch)
            self.k_to_solve = k_captured + 2
            self._epoch_pending = k_captured
            if self.plans_chunks:
                # PLAN the trunk for this chunk, then fall through and solve
                # ticks one at a time exactly as the per-tick path always has.
                #
                # This is the whole of the whole-body change to this loop. The
                # planner needs the chunk (pass B's reachability tube is a
                # backward sweep, so tick 0's bound depends on tick 24's demand)
                # and costs ~80 ms once; after it, every tick is an ordinary
                # arms-only solve with the trunk placed at the planned angle
                # instead of the measured one. Nothing else here moves: same
                # cursor, same gate, same epoch, same write.
                self._plan_trunk(self.k_to_solve)
        elif (self.k_to_solve is not None
                and self._within_lead(self.k_to_solve + 1)
                and self.chunks.covers_any(
                    self.k_to_solve + 1,
                    adopted_through=self.last_read_chunk_id)):
            # `adopted_through` is what keeps this branch inside the chunk it
            # believes it is walking through. A chunk whose payload has landed
            # but whose id has not is coverable and not yet announced;
            # without the limit this branch would step onto its first tick
            # without re-stamping the epoch, and publish a deadline from the
            # PREVIOUS chunk's clock: ~100 ms in the past, silently discarded
            # by P7.
            self.k_to_solve += 1
        else:
            # Horizon exhausted (or nothing has ever been published). There is
            # no tick to work on and nothing to wait on but the next chunk.
            return Step(None, False, None, "idle")

        k = self.k_to_solve

        # 2. contributors — every ADOPTED chunk covering this tick, ascending
        #    id. An announced-but-unadopted one waits for step 1's re-anchor.
        contrib = self.chunks.covering(k, adopted_through=self.last_read_chunk_id)
        if not contrib:
            # Only reachable through the new-chunk branch, which anchors to
            # k_exec + 2 without testing coverage: in async P6 may have fallen
            # far enough behind that the chunk starts past it. There is
            # nothing to blend, so the tick is a hole like any other.
            return Step(k, False, None, "uncovered")

        # 3. blend to one action. `apply` aligns the hemispheres first;
        #    the hand and the waist come out of the same call as the pose so
        #    they can never be taken from different chunks.
        target = blend.apply(self.blend, contrib)

        # 4-7. seed, solve, clamp, deadline, gate, write. Shared VERBATIM with
        #      the intervention path: only the SOURCE of `target` differs,
        #      so everything a policy target is clamped, gated and recorded by is
        #      what an operator's target is clamped, gated and recorded by too.
        #      `deadline=None` is what selects the epoch over a supplied
        #      one, and it is the only argument the two paths disagree about.
        return self._commit(k, target, self._intervene)

    # --- the shared tail ----------------------------------------------------

    def _commit(self, k: int, target, counter: int, *,
                deadline: float | None = None) -> Step:
        """Solve `target` for tick `k`, clamp it, gate it and write the ring.

        Steps 4 to 7 of the tick, shared by both sources, with the one thing an
        intervention needs to say differently expressed as an argument:
        **`deadline`**, which for an operator comes from their own clock — already
        mapped onto this machine's by `intervention.Pace` — and so has no epoch
        behind it. `None` means "derive it", which is the policy path and the only
        path that stamps the epoch.

        Nothing else needed a parameter, and that is a consequence of stamping the
        counter into the slots rather than wiping them. A takeover leaves the ring
        intact, so `latest_below` still finds the policy's last posture and the
        first operator solve is seeded from it — no fallback seed, no
        deadline-derived velocity span, no special case for an empty ring.
        """
        # The seed is the written target with the largest k below this one —
        # never the measurement, which would make tracking error an input to the
        # next target. NOT filtered on the counter: selection is about
        # authority and must not cross a takeover, seeding is about geometry and
        # must (see `TargetRing.latest_below_full`).
        seed = self.body.latest_below(k)
        if seed is None:
            # Homing leaves k = 0 written, so this is only reachable if
            # the ring has wrapped past every predecessor. Nothing to seed from,
            # therefore nothing to solve and nothing to write.
            return Step(k, False, None, "unseeded")
        seed_arm = seed.payload[:schema.ARM_DOF]
        # The SEED is unchanged and deliberately so: the written target below
        # this tick, never the measurement. Feeding tracking error back
        # in as a seed makes it an input to the next target and it accumulates.
        # The TRUNK is the opposite case — it is not an error to be rejected but
        # a fact about the geometry the arm hangs off, and using anything other
        # than the measurement solves the arm into a frame the robot is not in.
        # Bound to a local, not inlined: this exact array is BOTH the frame the
        # QP is asked about and the frame the answer is only meaningful in, so
        # the value that goes to the solver and the value that goes into the
        # record have to be the same read of the state ring. Two calls a tick
        # apart would be two different trunks with nothing saying so.
        # `planned` is what the trunk is COMMANDED to; `waist_ik` is where the
        # arm is SOLVED. With a stationary trunk they are the same thing and
        # this is the deployed behaviour; with a moving one they are not, and
        # conflating them is a measured 9 mm error. See `solve_frame`.
        planned = self._planned_waist(k)
        # THE COMMANDED WAIST, decided once. Precedence: a trunk plan (P6's own
        # planner, eef only), else the chunk's — a joint-mode `waist_q`, always
        # present there, or an operator's row — else `OWN_WAIST`.
        unplanned = target.waist if target.waist is not None else OWN_WAIST
        commanded = planned if planned is not None else unplanned
        # The frame falls back to the UNPLANNED command when there is no
        # measurement: `solve_frame` adds the plan's increment on top of
        # whatever it measured, so handing it the plan here would count it twice.
        waist_ik = self.solve_frame(k, unplanned, planned)
        if target.arm_q is not None:
            # JOINT MODE: the policy's joints are the solution. Nothing is
            # solved, so there is no residual — NaN, the convention for a tick
            # no IK produced (`schema.TARGET.err`). `waist_ik` is still the
            # measured trunk: it is the frame these joints put the flange in,
            # which is what makes the record's FK checkable.
            q_arm, err = np.asarray(target.arm_q, np.float64).reshape(
                schema.ARM_DOF), float("nan")
        else:
            q_arm, err, ok = self.solver.solve(target.pose, waist_ik,
                                               seed_arm, budget=self.dt)
            if not ok:
                return Step(k, False, None, "abandoned")

        # 5. velocity clamp against that SAME seed. The span is the seed's tick
        #    distance, not one tick: after a hole, or after a new chunk snaps
        #    k_to_solve back, the predecessor may be several ticks away and the
        #    arm has that many ticks to cover the delta.
        span = (k - seed.k) * self.dt
        limit = self.qdot_max_arm * span
        q_arm = np.clip(q_arm, seed_arm - limit, seed_arm + limit)

        # 6. epoch, then deadline — policy only.
        if deadline is None:
            deadline = self._epoch_deadline(k)

        # 7. write the target (body and hand in one row) — or nothing.
        k_exec = self.control.k_exec                 # RE-READ here (the write gate)
        if k == k_exec or k == k_exec - 1:
            return Step(k, False, deadline, "gated")
        q_body = np.empty(schema.BODY_DOF)
        q_body[:schema.ARM_DOF] = q_arm
        # The waist decided above: the plan, the chunk's own, or OWN_WAIST.
        q_body[schema.ARM_DOF:] = commanded
        # `err` and the blended pose ride along with the target they belong to.
        # `err` is the only thing that separates "the arm cannot go there" from
        # "the machine was too busy"; without it a record shows 82 mm of
        # under-reach as a perfectly ordinary-looking joint angle.
        #
        # `target.pose` is the IK's INPUT, and it is stored for a different
        # reason: it cannot be recovered afterwards. It is the blend of every
        # chunk `covering(k)` returned, and which chunks those were depends on
        # ring residency at an instant nothing records. Sync happens to be
        # reconstructible — one contributor per tick — but async is the mode
        # where the blend does real work, and there the answer is genuinely
        # unrecoverable. Recording it here is one store; recomputing it offline
        # is impossible.
        # `waist_ik` rides along for the same reason as the other two, and it is
        # the one that makes them checkable. `q_body[14:17]` is the COMMANDED
        # waist and `waist_ik` is the one the arm joints were solved in; storing
        # only the first left `FK(q_arm, q_body[14:17]) != pose ± err` in every
        # record, with the 8-11 mm of trunk droop looking like IK error that no
        # amount of tuning could remove. It costs one store of an array this
        # method already holds.
        # ONE write, both payloads, under one seqlock (`schema.TARGET`): body
        # and hand share `k` and the deadline by construction. `counter` stamps the row with
        # the world it belongs to, so P7 and P8 select it together or not
        # at all.
        self.targets.write(k, deadline, q_body, target.hand, intervene=counter,
                           err=err, pose=target.pose, waist_ik=waist_ik)
        return Step(k, True, deadline, "written")

    # --- one whole chunk (waist-mode solvers) -------------------------------

    def _plan_trunk(self, k0: int) -> None:
        """Plan the trunk for the chunk that starts at `k0`. Passes A and B.

        Called once when a chunk lands. Everything after it is the ordinary
        per-tick path — the plan simply replaces the measured trunk
        as the frame each tick is solved in, and becomes the commanded waist.

        A failure here is not fatal and not loud: the plan is dropped and the
        chunk runs on the measured trunk, as under an arm-mode solver. A
        whole-body run that quietly becomes an arms-only run is worse than one
        that says so, so it says so — but it does not take the robot down for a
        solver that could not plan one chunk.

        With `cfg.plan_lookahead_ticks` set, the trajectory handed to the solver
        runs PAST the ticks this loop will solve — see the knob's own docstring
        for why. Two things keep that from turning planning input into actions:
        the walk below still collects only covered ticks, so `n_exec` is exactly
        what it was; and `budget` is computed from `n_exec`, not from the length
        of what the planner sees, so a 10-tick lookahead does not silently tell
        the solver it has 1.6 s to cross a chunk that executes in 0.8.
        """
        self._plan = None
        poses = []
        for k in range(k0, k0 + self._chunk_span):
            contrib = self.chunks.covering(k, adopted_through=self.last_read_chunk_id)
            if not contrib:
                break
            target = blend.apply(self.blend, contrib)
            if target.pose is None:
                # Nothing to plan: a joint chunk has no Cartesian targets for
                # a trunk planner to reach, and has decided the trunk itself
                # (`waist_q` is required in joint mode).
                return
            poses.append(target.pose)
        if not poses:
            return
        n_exec = len(poses)
        if self._plan_lookahead:
            tail = self.chunks.plan_rows(
                k0, adopted_through=self.last_read_chunk_id)
            if tail is not None and tail.pose is not None:
                # Poses only: the lookahead is planning input, and an eef chunk
                # carries no trunk opinion to extend.
                poses.extend(tail.pose)
        seed = self.body.latest_below(k0)
        if seed is None:
            return
        # The measured trunk, read ONCE for the chunk and used as the plan's
        # origin. The reason is the module docstring's: the bow rests against a stop
        # 0.043 rad forward of its commanded zero, so a plan that starts
        # anywhere else starts in a frame the robot is not in.
        waist_meas = self.waist_now(OWN_WAIST)
        try:
            traj = self.solver.plan_chunk(
                np.asarray(poses, dtype=np.float64), waist_meas,
                seed.payload[:schema.ARM_DOF],
                budget=max((n_exec - 1) * self.dt, self.dt))
        except Exception as exc:                            # noqa: BLE001
            print(f"p6: trunk plan failed, running on the measured trunk "
                  f"({exc!r})", flush=True)
            return
        if traj is None:
            return                      # a plug-in that plans no trunk
        # Only the executed span is a plan. A solver handed the lookahead may
        # return a trunk for it too; those rows describe ticks no chunk covers
        # and the next replan will re-plan anyway, so they are cut here rather
        # than left for `_planned_waist` to index into by accident.
        traj = np.asarray(traj, dtype=np.float64)[:n_exec]
        self._plan = (k0, traj)

    def _within_lead(self, k: int) -> bool:
        """May tick `k` be solved yet, or is it too far ahead of the servo?

        `cfg.solve_lead_ticks`. P6 solves a tick in ~4 ms against P7's 33 ms
        period, so with no bound it races: the whole chunk is solved within
        ~110 ms of landing and then the loop idles for the remaining 700. Every
        tick is therefore solved against ONE observation of the trunk, taken
        before any of them executed.

        That is exactly what `solve_frame` cannot compensate for. It corrects
        the arm for where the trunk will be, using the measurement plus the
        plan's increment — but if the measurement is 25 ticks stale, the
        increment is the whole chunk and the correction is an open-loop
        prediction. Holding P6 near the servo makes it a short extrapolation
        from a fresh measurement, which is the same quantity the per-tick path
        has always used.

        Nothing is added to make this wait: refusing the advance returns `idle`
        and the caller calls `step()` again, which is what the loop already
        does when the horizon is exhausted.

        **The bound must stay comfortably above the solve time.** P6 has to
        publish tick `k` before its deadline or it becomes a hole; at 33 ms a
        tick, a lead of 5 is ~165 ms of slack against a ~4 ms solve. Below
        about 2 the margin is a single solve and a slow one costs a tick.
        0 disables the bound entirely and restores the racing behaviour.
        """
        if self.lead <= 0:
            return True
        return k <= self.control.k_exec + self.lead

    def solve_frame(self, k: int, fallback, planned):
        """The trunk the arm should be solved in for tick `k`.

        **Not the planned angle.** The trunk does not arrive where it is sent.
        Measured on the reference robot's bow over five clean rates:
        steady-state error is `20.9 + 24.1 * v` milliradians, R^2 = 0.90. At
        `rate_waist_bow = 0.35` that is 29 mrad, and at that robot's measured
        0.28 m of effective lever arm, **8 mm at the flange** — an order above
        the residual the whole tier exists to remove.

        **The static term is 71% of it**, and three independent measurements
        agree on it: 21.2 mrad from position steps, 20.9 from this fit, ~24
        from stepping either side of a bias. It is a constant ~6.3 N.m that
        nothing compensates, and it is NOT stiction — stepping both ways from
        a bias came out short one way and long the other, which a friction
        term cannot do.

        Solving at the planned angle therefore puts the arm in a frame the robot
        is not in, which is precisely the failure `waist_now()` was written to
        prevent, reintroduced one level up and larger.

        So: take the trunk's ACTUAL position from the state ring, and add only
        what the plan says it will move BETWEEN NOW AND THEN.

            solve_frame(k) = measured_now + (plan[k] - plan[k_exec])

        The measurement carries the static offset and whatever the joint is
        currently behind by; the plan contributes only the increment over the
        lead. Both terms are then right for the instant the tick executes:

        * a stationary plan reduces this to `waist_now()` exactly — the
          deployed behaviour, arrived at rather than special-cased;
        * **the static 21 mrad cancels exactly.** If the trunk sits a constant
          d behind its command then `measured = plan[k_exec] - d`, and adding
          the increment gives `plan[k] - d`, which is where the trunk will
          actually be. What survives is only the change in d over the lead
          window — `24.1 * dv`, at most 8 mrad or 2.4 mm during the hardest
          acceleration the rate limit permits, and near zero at constant
          velocity;
        * a lagging trunk stays lagged in the estimate rather than being
          assumed to have caught up;
        * no lag model, no calibration constant. The 21.4 and 32.5 above are
          why this is needed, not numbers this uses.

        What it cannot fix is the trunk being somewhere the arm cannot
        compensate for; it fixes the arm being solved for a trunk that is
        somewhere else.
        """
        measured = self.waist_now(fallback)
        if planned is None:
            return measured
        ref = self._planned_waist(self.control.k_exec)
        if ref is None:
            # `k_exec` is outside the plan — before its first tick on a fresh
            # chunk, or past its last. The chunk's own start is then the right
            # reference: the plan is anchored there and the measurement is of
            # a trunk that has not yet been asked to move.
            ref = self._plan[1][0] if self._plan is not None else planned
        return measured + (np.asarray(planned, np.float64)
                           - np.asarray(ref, np.float64))

    def _planned_waist(self, k: int):
        """The planned waist for tick `k`, or None to use the measurement.

        None for every tick outside the planned chunk, which is what makes a
        short chunk, a hole, or the flag being off all take the deployed path
        without a branch anywhere else.
        """
        if self._plan is None:
            return None
        k0, traj = self._plan
        i = k - k0
        return traj[i] if 0 <= i < len(traj) else None

    def _epoch_deadline(self, k: int) -> float:
        """Stamp the epoch if this tick is the one that owes it, then derive.

        Split out of `step` so the intervention path can skip it without a branch
        inside the shared tail: an operator's deadline is already an instant.

        **The async branch re-stamps too, because of interventions.** Stamping
        once per run is right only while `k` and the clock always advance
        together. An intervention breaks that: real time passes while
        the epoch's tick line does not advance at `dt`, so on the way back
        `t_base + (k - k_base)*dt` lands far in the past and P7 discards EVERY
        policy target thereafter — the arm would hold forever after a handback,
        in async only, with nothing reporting why. `_epoch_pending` is set by the
        new-chunk branch, which is the only way back onto the policy path after a
        takeover, so honouring it here fires exactly at the first chunk and after
        every intervention, and nowhere else.
        """
        if self._epoch_pending is not None and (self.cfg.sync
                                                or self._epoch_voided):
            # SYNC, once per chunk: clock time passed during inference while k
            # did not, so without the re-stamp every subsequent deadline is early
            # by the accumulated inference latency. The CAPTURED k_exec, never a
            # fresh read — a fresh one would break deadline(k_exec + 2) =
            # now + 2·dt, which is the whole point.
            #
            # ASYNC, only after a handback. Async has ONE epoch for the life of
            # a run, `k_base = 0` — chunks overlap by design and their
            # tick line has to stay continuous, so re-stamping per chunk here
            # would jump every deadline, which P5's window scan relies on not
            # happening. An intervention is the one thing that can void it: real
            # time passes while the tick line does not advance at `dt`, so the
            # line has to be re-founded rather than continued, and
            # `_epoch_voided` is `_switch` saying exactly that.
            self.k_base, self.t_base = self._epoch_pending, self._now()
            self._epoch_pending = None
            self._epoch_voided = False
        elif self.t_base is None:
            # Async at the start of a run. k_base = 0 because tick 0 is the end
            # of homing, and that origin is what makes every deadline in the run
            # re-derivable from one pair of numbers.
            self.k_base, self.t_base = 0, self._now()
        return self.t_base + (k - self.k_base) * self.dt

    # --- intervention --------------------------------------------

    def _switch(self) -> None:
        """The intervention counter moved. Drop everything latched against it.

        **Nothing is cleared, and that is the whole economy of stamping the
        counter into the slots.** At a takeover the ring still holds up to `P`
        future-dated policy targets; they are simply unreachable, because P7 and
        P8 now select on the new counter and none of them carries it. The arm
        stops because P7 finds nothing selectable, stalls, and republishes
        `q_cmd_last` frozen — "hold whatever you were last commanded",
        reached through a path P7 already had.

        Leaving them there is not merely cheaper than a wipe, it is better in two
        ways. `latest_below` still finds the policy's last posture, so the first
        operator solve is seeded from the configuration the policy left and the
        IK's null space does not jump at the seam — which is why this path needs
        no fallback seed at all. And a wipe is N seqlock writes against a live P7
        whose column scans are unlocked, which is a window in which the ring looks
        empty; here there is no such window because nothing is written.

        The same holds in reverse at the handback, where the operator's targets
        stay as the seed for the policy's first solve back.

        Three pieces of local state go, and each names something that was true of
        the world just left: the cursor, so the policy path can only resume
        through its new-chunk branch — the one that re-anchors on `k_exec` and
        re-stamps the epoch — and never by walking a cursor from before; the row
        index and the decimation reference, which belong to an intervention that
        is over; and the EPOCH itself, which is what makes async correct across a
        handback (see `_epoch_deadline`).
        """
        self.k_to_solve = None
        self._row_index = 0
        self._guided_due = None
        self.t_base = None
        self._epoch_voided = True
        # A trunk plan belongs to the chunk it was planned for, which is now
        # abandoned. `validate.check_intervene_ik` refuses waist-mode IK with
        # `--intervene`, so this is None on every run that can reach here — and
        # it is dropped anyway, so a planned waist can never be commanded under
        # an operator's targets should that refusal ever be lifted without
        # thinking this through.
        self._plan = None

    def _guided(self, counter: int) -> Step:
        """One tick driven by the operator instead of by the policy.

        The whole of it is: take the newest guide row, put it where a blended
        chunk would have gone, and commit it. A row IS a `chunk_blend.Action` — same
        shapes, same base frame, same units — which is why there is no conversion
        here and no second write path.

        **The rows are DECIMATED to the action rate, and that is what keeps every
        other process working.** P11 publishes at the PICO's ~90 Hz; committing
        one target per row would run `k` at three times the action rate, and `k`
        is not a free counter — `deadline = t_base + (k - k_base)*dt` is affine in
        it, P5 extrapolates its trigger instant off that, a chunk's 25-tick window
        would be consumed in under a third of a second, and `span = (k - seed.k)*dt`
        in the velocity clamp would licence three times the joint speed the policy
        path allows. Publishing only one row per action period keeps `k` on the
        action line, so the seed, the span, the clamp, the write gate and P5's
        arithmetic are all the code they already were.

        **The reference is the GRID, and that is the whole of the correctness
        here.** `_guided_due` advances by exactly `dt` per published row; it is
        never set to the deadline of the row that happened to be taken. Measured
        on one recorded rollout, referencing the row instead put the operator on the
        wire at 22.9-24.2 Hz rather than 30 — every intervention of the rollout,
        with segments alternating 33.4 and 44.8 ms. The PICO delivers at 90 Hz, so
        three frames is 33.333 ms and `dt` is 33.333 ms: the test sat exactly on
        the boundary, dithered between the third row and the fourth, and a row
        reference carried each slip forward for the rest of the intervention
        instead of correcting it. A grid reference cannot accumulate — a row taken
        late leaves the next one due at the same instant it always was — and
        `GUIDE_SLACK_S` keeps the boundary case off the knife edge. P7 turns
        segment length straight into its velocity feedforward (`delta/span`), so
        what the operator felt was not a slower arm but an alternating commanded
        velocity: a 25% judder at every step.

        A row a full `dt` late is a different thing — the stream paused, or P11
        went stale — and there the grid is meaningless: it re-founds on the row
        actually published rather than firing the backlog it appears to owe.

        What it costs is resampling the operator from 90 Hz to 30 Hz, which is
        not a loss worth having machinery for: P7 interpolates linearly between
        targets anyway, a human hand carries nothing above about 5 Hz, and 30 Hz
        is exactly the authority the policy has — so a correction the operator
        makes is one the policy could have made, which is what a DAgger label is
        supposed to mean.

        A row already solved, or one arriving inside the decimation window, is an
        idle iteration rather than a repeat: every row is an absolute pose from a
        fixed anchor, so skipping one leaves no increment to make up and the next
        row already carries the whole displacement.
        """
        row = self.intervene_ring.newest(counter, since=self._row_index)
        if row is None:
            # P11 has flipped the counter but its first row has not landed, or
            # the stream has paused. Either way there is nothing to solve, and P7
            # is already holding: no slot in the ring carries this counter.
            return Step(None, False, None, "idle")
        self._row_index = row.index
        if (self._guided_due is not None
                and row.deadline < self._guided_due - GUIDE_SLACK_S):
            return Step(None, False, None, "decimated")

        # The cursor. `k_exec + 2` on the first row of an intervention, exactly as
        # the new-chunk branch anchors, and then one per committed row. The
        # two-tick gap is P7's write margin, unchanged; it leaves one
        # empty slot at `k_exec + 1` per takeover, which P7 skips the way it skips
        # any hole.
        self.k_to_solve = (self.control.k_exec + 2 if self.k_to_solve is None
                           else self.k_to_solve + 1)
        step = self._commit(self.k_to_solve,
                            blend.Action(row.pose, row.hand, row.waist),
                            counter, deadline=row.deadline)
        if step.written:
            # Advanced only on a WRITE. A gated or abandoned tick must leave the
            # reference where it was, or the row that replaces it would be
            # decimated away too and the operator would feel a stall twice as
            # long as the one that actually happened.
            on_grid = (self._guided_due is not None
                       and row.deadline < self._guided_due + self.dt)
            self._guided_due = ((self._guided_due if on_grid else row.deadline)
                                + self.dt)
        return step

    def waist_now(self, fallback) -> np.ndarray:
        """The measured trunk, `(3,)` in waist-slot order — the frame the IK
        solves in.

        Falls back to the tick's COMMANDED waist (the plan, the chunk's own,
        or `OWN_WAIST` — see `_commit`) when the state ring is empty,
        which is only reachable before P1 has published anything: P0's CHECK
        phase proves the stream is live before the run starts, so in an ACTIVE
        run this branch is unreachable. It exists so a test can drive `step()`
        without a state writer, and so the first tick of a race cannot solve on
        garbage. The commanded value rather than a bare zero, so that a policy
        that drives the waist does not silently get an upright trunk when the
        measurement is missing.

        Non-finite is treated the same way, and that guard is not decoration.
        This is the one path where a measurement becomes a commanded joint
        angle: a NaN here places the whole model at NaN, the QP returns NaN, and
        `validate_chunk` is upstream of this point so nothing between here and
        `rt/lowcmd` would catch it. P6's own velocity clamp does not either —
        `np.clip(nan, lo, hi)` is `nan`.

        The value is NOT clamped to the URDF's joint range. It is a measurement:
        if the encoder says the trunk is somewhere the model thinks impossible,
        the honest response is to solve for where the arm really is, not to
        pretend the robot is somewhere else. (It is worth knowing that the range
        this model declares for the bow — up to 2.36 rad of forward travel — is
        not what the hardware has: the bow rests on a mechanical stop near
        +0.04 rad; see `runtime/ik/waist_planner.py`.)
        """
        snap = self.state.latest()
        if snap is None:
            return fallback
        waist = np.asarray(snap["q"], dtype=np.float64)[self.waist_cols]
        return waist if np.isfinite(waist).all() else fallback

    def deadline(self, k: int) -> float | None:
        """`t_base + (k − k_base)·dt` — the epoch, for anything that wants to
        re-derive a written deadline. None before the first stamp."""
        if self.t_base is None:
            return None
        return self.t_base + (k - self.k_base) * self.dt


# --- process ----------------------------------------------------------------


def run(shm_name: str, cfg, hw, **kwargs) -> None:
    """P6 as a process. Runs while `session == ACTIVE`.

    Spawned once per rollout, at ACTIVE, so it latches the rollout's settings
    once, at birth: `cfg` is the launch config and the arena's
    `active_config` says what this rollout runs with — `P`, sync, `dt`, `ik`.
    The layout is built from the launch config, which is the same layout
    whatever the settings (`derive.derive`).
    """
    from elrobot_deploy.config_loading.rollout_settings import latched

    layout = build(cfg, derive(cfg, hw))
    arena = Arena.attach(shm_name, layout)
    try:
        rings = open_all(arena, cfg)
        affinity.pin(hw, "p6_target")
        cfg, _, _ = latched(rings["control"], cfg)
        # The solver `cfg.ik` names (`ik_solvers`). mujoco/mink are imported
        # only here, by the factory — and `wbik` only by the solvers that use
        # it (`waist_qp`, `waist_planner`), so an `arm_qp` run never loads its
        # model surgery or its cost model.
        from elrobot_deploy.runtime.ik import build as build_solver
        solver = build_solver(cfg, hw)
        loop = Target(rings, cfg, hw, solver)
        control = rings["control"]
        while control.session == schema.ACTIVE:
            loop.step()
    finally:
        arena.close()


__all__ = ["Step", "Target", "run"]
