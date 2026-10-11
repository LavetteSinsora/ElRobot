"""Shapes, dtypes and record layouts — the vocabulary every process shares.

Nothing here allocates or reads anything. It is the single place where the
action contract and every other shared record turn into numpy dtypes, so that a disagreement about what a slot
contains is a diff in this file rather than a torn read at 500 Hz.

Counters are `uint64` in their own array, never a field inside a payload record:
a float counter stops incrementing past its mantissa limit, and
every torn read afterwards goes undetected.

Imported by every process, including P5 under the policy's interpreter (numpy
only). The records field by field, for a reader:
`docs/infra_logic/arena_layout_and_reads.md`, "The records, field by field".
Changing a record here means bumping `VERSION` (see THE RULE below).
"""

from __future__ import annotations

import numpy as np

# --- the contract -------------------------------------------------

N_HANDS = 2
POSE_DIM = 7          # x y z qw qx qy qz
HAND_DIM = 6          # gripper uses dim 0; Brainco all six
WAIST_DOF = 3         # yaw, roll, pitch
ARM_DOF = 14          # 7 left + 7 right
BODY_DOF = ARM_DOF + WAIST_DOF          # 17 — one body target slot

N_JOINTS = 29         # G1_29 lowstate/lowcmd width
GRIP_DIM = N_HANDS * HAND_DIM           # 12 — measured end-effector state

# `k` sentinel for "this slot has never been written". Homing runs at k < 0
# #, so the sentinel cannot be a small negative number.
K_UNSET = np.iinfo(np.int64).min

# --- session states ---------------------------------------------

INIT, CHECK, HOME, WARMUP, READY, ACTIVE, EXITING, STOPPED = range(8)

SESSION_NAMES = {
    INIT: "INIT", CHECK: "CHECK", HOME: "HOME", WARMUP: "WARMUP",
    READY: "READY", ACTIVE: "ACTIVE", EXITING: "EXITING", STOPPED: "STOPPED",
}

# --- operator requests ------------------------------------------
#
# The operator's intent, written by P10 and consumed by P0. Distinct from
# `session`, which is where the run actually IS: a request is what somebody
# asked for and may be refused, ignored or superseded, while `session` is the
# state machine's own answer and has exactly one author.
#
# START begins a rollout from READY. RESET ends the current rollout and
# returns to READY without re-warming — the whole point of the word. EXIT is
# the exit sequence (ramp to the exit pose, then damp), identical to the first
# Ctrl-C.
#
# There is deliberately no DAMP. The second Ctrl-C drops the arm into gravity
# and is "an explicit second decision"; a browser at the far end of an
# SSH tunnel that may be saturated by the camera stream is the wrong instrument
# for the one action taken in an emergency. That one stays on the terminal.
REQ_NONE, REQ_START, REQ_RESET, REQ_EXIT = range(4)

REQUEST_NAMES = {
    REQ_NONE: "NONE", REQ_START: "START", REQ_RESET: "RESET", REQ_EXIT: "EXIT",
}

# --- records ----------------------------------------------------------------

HEADER = np.dtype([
    ("magic", "<u8"),
    ("version", "<u8"),
    ("layout_hash", "<u8"),
])

MAGIC = 0x473144000001  # "G1D" + 1
#: 9: the chunk slot carries either action mode (`mode`, `has_waist`, `arm_q`)
#: — the policy contract's two action modes.
#: 10: cameras are rings of N slots with a per-slot uint64 sequence counter and
#: a published `head` word, one ring per camera VIEW (a
#: side_by_side camera is two); the slot record lost `generation`.
#: 11: the control region carries the rollout settings and the prompt channel
#: (`requested_config` / `active_config`, `rollout_generation`).
#: 12: `SETTINGS.ik` indexes the IK SOLVERS (`IK_OPTIONS`, `runtime/ik` `SOLVERS`)
#: rather than the IK modes, with `IK_LAUNCH` for a launch-named module.
#: 13: `TARGET.err` is measured at the point the IK drives — the robot file's
#: TCP when it gives one, else the flange (it was always the flange).
#: 14: the chunk slot lost `has_waist`: a joint chunk ALWAYS carries the
#: policy's `waist_q` (it is required), an eef chunk never does — `mode` says it.
#: 15: the control region carries `latency_ns`, P5's measured trigger-to-chunk
#: latency, which P0 checks an async rollout against at START.
#: 16: a fisheye view's ring holds the square around its lens circle (fitted
#: at launch, before the arena exists), not the sensor; its shape is that
#: square's.
#: 17: `SETTINGS.blend` — the async blend switches between rollouts
#: (`BLEND_OPTIONS`, `BLEND_LAUNCH` for a launch-named module).
#:
#: THE RULE: bump this whenever a record's fields, a region's name or a ring's
#: read protocol changes — anything a reader written against the old layout
#: would misread rather than fail on. The layout hash already refuses an attach
#: across different SIZES; this is what refuses one across different MEANINGS
#: (`docs/infra_logic/arena_layout_and_reads.md`, "What is stable").
VERSION = 17

# --- rollout settings ---------------------------------------------
#
# What may change BETWEEN rollouts of one launch, as one fixed record. Written
# into the control region twice: `requested_config` (P10, any time) and
# `active_config` (P0 only, copied from the request at the START it accepts).
# `rollout_settings.RolloutSettings` is the Python face of it; this is the bytes.
#
# What is NOT here is as deliberate: robot, policy, checkpoint, H, history and
# the camera set change the arena layout or the loaded model, so they are
# fixed per launch.

#: `SETTINGS.ik` is an index into this — the IK solvers elrobot_deploy ships,
#: `runtime/ik` `SOLVERS` in its order (`rollout_settings.py` asserts the two agree). Was the
#: IK MODE list until VERSION 12.
IK_OPTIONS = ("arm_planner", "arm_qp", "waist_planner", "waist_qp")
#: `SETTINGS.ik` for "the launch's own solver": a research `module:factory`
#: named at launch, which has no index here and is fixed per launch.
IK_LAUNCH = 255

#: `SETTINGS.blend` is an index into this — the blends elrobot_deploy ships
#: (`chunk_blend._BLENDS`, in its order; `rollout_settings` asserts the two
#: agree). Only async uses it: a sync tick has one chunk.
BLEND_OPTIONS = ("temporal_ensemble", "newest")
#: `SETTINGS.blend` for "the launch's own": a `module:function` blend named in
#: the deploy file, which has no index here and is chosen at launch.
BLEND_LAUNCH = 255

SETTINGS = np.dtype([
    ("sync", "<u8"),                 # 1 = sync, 0 = async
    ("replan_period", "<u8"),        # P, ticks
    ("control_hz", "<f8"),           # the action rate
    ("ik", "<u8"),                   # index into IK_OPTIONS, or IK_LAUNCH
    ("sample_steps", "<u8"),         # the policy's, via `Policy.on_settings`
    ("blend", "<u8"),                # index into BLEND_OPTIONS, or BLEND_LAUNCH
    # `resolved.ServoShaping`, field for field.
    ("filter_hz", "<f8"),
    ("filter_lead", "<u8"),
    ("velocity_ff", "<f8"),
    ("torque_ff", "<f8"),
    ("arm_gain_scale", "<f8"),
])

#: Bytes of UTF-8 a prompt may take in the arena. The POLICY's token limit is a
#: separate, usually tighter, question (`Policy.check_prompt_length`); this is only the
#: size of the box, and P10 refuses a longer prompt before storing it.
PROMPT_BYTES = 256

#: Room for one line of "why" — a refused prompt or a refused START.
REASON_BYTES = 256

#: `verdict` values, written by P5.
PROMPT_PENDING, PROMPT_ACCEPTED, PROMPT_REFUSED = range(3)

PROMPT_VERDICT_NAMES = {PROMPT_PENDING: "pending", PROMPT_ACCEPTED: "accepted",
                        PROMPT_REFUSED: "refused"}

#: Set in `settings_applied` beside the generation when the POLICY refused it
#: (`set_prompt` / `on_settings` raised): P0 then refuses the START.
APPLIED_FAILED = 1 << 63

# One record. The single words are independently atomic aligned uint64/int64
# stores; there is no seqlock around them
# because no reader needs two of them to be mutually consistent. The settings
# blocks at the end are multi-word and carry their own seqlock counters.
CONTROL = np.dtype([
    ("session", "<u8"),          # written by P0
    ("k_exec", "<i8"),           # written by P7
    ("newest_chunk_id", "<u8"),  # written by P5, after the payload
    ("warm", "<u8"),             # written by P5: dummy inferences completed
    # P5's estimate of trigger -> chunk written, ns (`Control.latency_s`).
    ("latency_ns", "<u8"),       # written by P5
    ("request", "<u8"),          # written by P10, CLEARED by P0 — see Control
    # THE INTERVENTION COUNTER. Written by P11, which is the process
    # that sees the trigger edge and freezes the operator's anchors in the same
    # instant. Even = the policy is driving, odd = the operator is. Every
    # increment is ALSO "every target stamped with the previous value belongs to
    # a world that no longer exists", because those are the same event and must
    # not be able to disagree.
    #
    # A counter rather than a flag so a transition cannot be MISSED: a flag
    # toggled twice between two of a reader's polls is indistinguishable from no
    # change at all, while a counter incremented twice is not — and one
    # invalidation is the correct response to any number of transitions.
    #
    # Monotonic for the life of the session and never reset. P11 guarantees it
    # leaves ACTIVE on an EVEN value, which is what a rollout boundary needs and
    # is cheaper than a second writer storing a zero.
    ("intervene", "<u8"),        # written by P11 — see Control.intervene

    # --- settings and prompt, switched between rollouts --------------------
    #
    # Multi-word blocks, so unlike the words above each has a SEQLOCK counter
    # beside it (odd while written, even when consistent) — `rings.Control`
    # reads them with `primitives.Seqlock` over one-element views. One writer
    # per block, as everywhere.

    # The operator's wish. P10 writes it any time; ONLY P0 reads it, at START.
    ("requested_seq", "<u8"),
    ("requested_config", SETTINGS),
    # The settings the current (or last) rollout runs with. ONLY P0 writes it,
    # at the START it accepts, after validating the request; every process
    # latches it when a rollout starts and holds it until the rollout ends.
    # `active_prompt` travels in the same block so the two cannot tear apart.
    ("active_seq", "<u8"),
    ("active_config", SETTINGS),
    ("active_prompt_len", "<u8"),
    ("active_prompt", "u1", (PROMPT_BYTES,)),
    # Bumped by P0 AFTER `active_*` is written — the payload before the
    # counter, as with `newest_chunk_id`. 0 = the launch settings,
    # before any START. A process that sees it move re-latches.
    ("rollout_generation", "<u8"),
    # P5's answer: the generation whose settings and prompt the policy has
    # taken (`set_prompt`, `on_settings`, one dummy inference). P0 moves to
    # ACTIVE only once this equals `rollout_generation`, so a recompile for a
    # new `sample_steps` is paid at READY, not by the first real inference.
    ("settings_applied", "<u8"),

    # The prompt request: P10 writes the text, seqlock-style — `prompt_seq`
    # odd while writing, even after, so each request moves it by two.
    ("prompt_seq", "<u8"),
    ("prompt_len", "<u8"),
    ("prompt", "u1", (PROMPT_BYTES,)),
    # P5's verdict on it (only the policy owns a tokenizer), its own seqlocked
    # block. `verdict_of` is the `prompt_seq` judged, so a reader can tell a
    # verdict on ITS request from one on an older request.
    ("verdict_seq", "<u8"),
    ("verdict_of", "<u8"),
    ("verdict", "<u8"),              # PROMPT_*
    ("verdict_reason_len", "<u8"),
    ("verdict_reason", "u1", (REASON_BYTES,)),

    # Why P0 refused the last START (a bad request, a refused prompt). P0 only;
    # seqlocked, so `refusal_seq / 2` is also how many STARTs were refused.
    ("refusal_seq", "<u8"),
    ("refusal_len", "<u8"),
    ("refusal", "u1", (REASON_BYTES,)),
])

#: The single-word fields of `CONTROL` — what a per-phase record row wants
#: (`p9_record.SESSION_REC`). The blocks above are recorded per rollout in the
#: manifest instead; a nested record would not survive a JSON `descr`.
CONTROL_WORDS = np.dtype([(name, CONTROL.fields[name][0]) for name in
                          ("session", "k_exec", "newest_chunk_id", "warm",
                           "request", "intervene", "rollout_generation")])

STATE = np.dtype([
    ("index", "<u8"),                        # absolute sample counter
    ("t_recv", "<f8"),                       # stamped FIRST, before parsing
    ("q", "<f8", (N_JOINTS,)),
    ("dq", "<f8", (N_JOINTS,)),
    ("flange", "<f8", (N_HANDS, POSE_DIM)),  # FK here, so it carries this t_recv
    ("grip_meas", "<f8", (GRIP_DIM,)),
    ("t_grip_recv", "<f8"),
    # The lift column's height, metres, from `rt/hispeed_state.y`.
    #
    # **The one DOF `rt/lowstate` does not report**: the column is not in
    # `rt/lowstate`'s 29 slots — measured, every slot outside arm+waist is
    # bit-identically zero over 65,559 samples — it is on its own topic at
    # ~30 Hz. Without this field no recording could say how high the robot was
    # standing.
    #
    # It costs the deploy loop nothing and never did: the pose frame RIDES the
    # column (`g1d_model.BASE_BODY` sits above it), so every flange pose is correct at
    # any height. This is for the record, and for the controller that will
    # command the column.
    #
    # Its own stamp beside it, exactly as `t_grip_recv` sits beside
    # `grip_meas` and for the same reason: this arrives at ~30 Hz against the
    # loop's 1070, so most samples carry a HELD value and a reader has to be
    # able to tell. Zero means never received.
    ("column_h", "<f8"),
    ("t_column_recv", "<f8"),
    # The robot's own millisecond clock at this sample (`LowState_.tick`, see
    # `runtime/drivers/robot_state.py`). `t_recv - tick` is the delivery
    # delay; a recording made by an older schema rebuilds its dtype from its
    # own manifest and simply lacks it.
    ("tick", "<u4"),
])

#: ONE target record: everything P6 publishes for a tick, in one row.
#:
#: One row rather than a body ring and a hand ring, so that "same `k`, same
#: deadline" and "body and hand, both or neither" are facts of the layout and
#: not rules P6 has to keep: there is one row, written under one seqlock, so a
#: body target cannot exist without its hand target and they cannot carry
#: different deadlines. Do not split it.
#:
#: The two readers each open a view of it: P7 one whose payload column is `q`
#: and P8 one whose payload is `hand` (`rings.TargetRing`).
TARGET = np.dtype([
    ("k", "<i8"),                            # K_UNSET = never written
    ("deadline", "<f8"),
    # WHICH WORLD THIS SLOT BELONGS TO — the value of `control.intervene` when
    # it was written. Read by P7/P8 in the same unlocked column scan
    # as `deadline`, which is why it sits beside it rather than at the end of the
    # record: the three words a segment advance needs are then one cache line.
    #
    # It is what makes a takeover instantaneous without wiping anything. On the
    # counter changing, every slot in the ring still holds a valid, future-dated
    # target — up to P of them, ~830 ms of committed policy motion — and
    # `next_after` selects on the smallest future deadline, so an intervention
    # target 35 ms out beats a policy target 800 ms out only for as long as
    # intervention targets keep arriving. The moment the operator's stream
    # pauses, the orphaned policy target at the far end of the horizon wins and
    # the arm lurches somewhere nobody asked for. Filtering on this word is what
    # makes that unreachable, at the cost of one comparison per scan.
    #
    # One word for BOTH payloads: the body and hand columns share a row, so P7
    # and P8 cannot come to disagree about which side of a takeover it is on.
    ("intervene", "<u8"),
    ("q", "<f8", (BODY_DOF,)),               # 14 arm, then 3 waist — P7's view
    ("hand", "<f8", (N_HANDS, HAND_DIM)),    # P8's view
    # The IK's own residual for this tick: worst POSITION error, metres, at the
    # point the IK drives — the robot file's `[end_effector] tcp` when it gives
    # one, else the flange — at the configuration actually published. Without it
    # "the arm cannot go there" and "the arm went there" are
    # indistinguishable in a record — rollouts have hidden 82 mm of
    # under-reach behind joint angles that looked perfectly ordinary.
    #
    # NaN, not 0.0, when no solve produced the tick: homing and the exit ramp
    # are written by P0 and have no IK behind them, so their tracking error is
    # undefined rather than perfect. Read by nobody in the control path; it
    # exists for `p9_record` and costs one store per tick.
    ("err", "<f8"),
    # The IK's INPUT: the blended absolute flange pose, base frame, that `q`
    # above is this tick's solution for. Same NaN convention as `err`, and for
    # the same reason — P0's ramps are joint-space and have no Cartesian target
    # behind them at all.
    #
    # Recorded because it is otherwise UNRECOVERABLE. It is
    # `chunk_blend.apply(fn, chunks.covering(k))`, and `covering()` answers with the
    # chunks physically resident in the ring at the instant P6 asked — which
    # depends on a `chunk_id % chunk_slots` eviction schedule and on a wall-clock
    # instant that nothing records. In sync that reconstruction happens to be
    # exact (chunks partition the tick line, one contributor each). In
    # async two chunks overlap by design, the answer is a blend of both, and the
    # best a reader can do offline is bound P6's read to a ~35 ms window — the
    # IK budget plus P9's sweep period — which straddles a replan whenever P is
    # small. That is precisely the regime async exists for, so the overlay would
    # be wrong exactly where it is interesting.
    #
    # The other two thirds of the blend are already recorded verbatim: the hand
    # ring carries `target.hand` untouched, and `q[14:17]` IS `target.waist`,
    # which passes through the IK unchanged. Only the pose was consumed. This
    # makes the body target self-describing in the same way the rest of the
    # record already is (cf. `STATE.flange`, P1's FK stored rather than
    # recomputed). Costs one store per tick and nothing on any read path.
    ("pose", "<f8", (N_HANDS, POSE_DIM)),
    # The FRAME that solve was expressed in: the `(3,)` waist P6 pinned the model
    # at before asking the QP, in `kinematics.WAIST_JOINT_NAMES` slot order. Same NaN
    # convention as the two fields above, and for the same reason.
    #
    # This is the third argument of `solve(pose, waist, seed)` and the only one
    # that was not recoverable. `q[14:17]` above is the waist COMMANDED for this
    # tick, which is a different quantity: the IK solves against
    # the trunk the arm is actually hanging off — the MEASURED one — because a
    # chunk's poses are absolute in a base frame that sits below the bow and does
    # not move when the trunk does (see `p6_target`). On the reference robot the trunk
    # rests 0.039 rad forward of the zero it is commanded to and no torque moves
    # it, so the two differ permanently and by 8-11 mm at the flange.
    #
    # Without this the entry does not close. `err` is the residual of a solve
    # whose frame is unstated, so nothing offline can check it — `FK(q[:14],
    # q[14:17])` is not `pose` and never was. With it, it is, to micrometres; and
    # posing the published joints draws a configuration the robot was actually in
    # rather than one that existed nowhere.
    #
    # It is NOT "the measured waist near this tick", and the difference is why it
    # is stored rather than looked up. P6 reads the state once, before the solve,
    # and that instant is recorded nowhere: offline it can only be bounded to the
    # IK budget plus P9's sweep period. Nor is it always a measurement —
    # `Target.waist_now` falls back to the commanded waist when the state ring is
    # empty — which is why the field is named for its ROLE in the solve and not
    # for where the number came from.
    ("waist_ik", "<f8", (WAIST_DOF,)),
])


#: ONE 500 Hz body command, exactly as P7 handed it to the wire.
#:
#: The targets say what P7 was ASKED for at 30 Hz; nothing said what it actually
#: SENT at 500 Hz, which is the only thing the firmware sees and the whole object
#: of any change to how P7 interpolates. Offline it had to be re-derived by
#: replaying the segment walk, which is exact only for the linear servo and
#: only up to when each target became available. This ring is P7's own log of
#: it: written once per publish, read only by P9, and costing P7 one seqlocked
#: store of ~700 B per period.
COMMAND = np.dtype([
    ("index", "<u8"),                        # absolute publish counter
    ("t", "<f8"),                            # P7's clock at the publish
    ("k_exec", "<i8"),                       # the tick being walked toward
    ("q", "<f8", (BODY_DOF,)),               # published position
    ("dq", "<f8", (BODY_DOF,)),              # published velocity feedforward
    ("tau", "<f8", (BODY_DOF,)),             # published torque (gravity + ff)
    ("tau_ff", "<f8", (BODY_DOF,)),          # the model feedforward part of tau
    # Today's command at this instant: the plain linear segment walk. Equal to
    # `q` on a default servo; the difference is what shaping did.
    ("q_lin", "<f8", (BODY_DOF,)),
])


# --- the intervention --------------------------------------------
#
# Written by P11 while an intervention is running; read by P6. One row is ONE
# ACTION — the identical triple a chunk row carries, in the identical frame and
# the identical units — plus the deadline it is due at.
#
# **That sameness is the whole design.** P6's policy path reduces a set of chunk
# rows to a `chunk_blend.Action` and commits it; its intervention path takes a row from
# here and commits it. Everything after that substitution is the code that was
# already there — the seed, the IK, the velocity clamp, the write gate, both ring
# writes — so an operator's target is bounded and recorded by exactly the
# machinery a policy target is, and every intervention target in the record is
# directly comparable to the policy target it replaced. That comparability IS the
# DAgger pair; a second write path would have destroyed it.
#
# Rows are written ONLY between a takeover and a handback. Outside one there is
# nothing for them to say, and the liveness question — "is the headset alive?" —
# is answered by `INTERVENE_META.samples` instead, which is bumped on every poll
# for exactly that reason.
INTERVENE = np.dtype([
    # This ring's own absolute row counter, so "the newest row" is an `argmax`
    # over the column — the same answer `StateRing.latest` gives and for the same
    # reason: the index and its payload go in under ONE seqlock, so they cannot
    # tear against each other the way a separate "newest" word could.
    ("index", "<u8"),
    # The value of `control.intervene` this row was written under. What makes a
    # wipe unnecessary at the handback: a row from a previous intervention
    # carries a different value and is unreachable, so P6 can never solve a pose
    # the operator asked for a takeover ago. Always ODD — rows exist only while
    # intervening.
    ("intervene", "<u8"),
    # The deadline, on THIS machine's monotonic clock, already mapped off the
    # PICO's device clock by `intervention.Pace`. Absolute rather than relative to
    # anything, so P6 does no arithmetic on it at all: it reads a row and writes a
    # target.
    ("deadline", "<f8"),
    # The action. Same shapes, same frame, same units as a chunk row — an
    # absolute FLANGE pose in the base frame, because that is what the IK solves
    # for. P11 reasons at the virtual TCP throughout and converts at the very last
    # step.
    ("pose", "<f8", (N_HANDS, POSE_DIM)),
    ("hand", "<f8", (N_HANDS, HAND_DIM)),
    ("waist", "<f8", (WAIST_DOF,)),
    # Which hands the operator was actually driving when this row was written.
    # NOT read by anything in the control path — a held hand and a driven hand
    # produce the same row, deliberately, so P6 has no case to distinguish. It is
    # here for the record, where "the human was steering this arm at this instant"
    # is the label a DAgger pair needs and nothing else carries.
    ("engaged", "<u8", (N_HANDS,)),
    # The raw operator inputs this row was built from, recorded rather than used.
    # `trigger` is the clutch, `axis_y` the thumbstick, `direction` the latch's
    # three-state answer for it. `direction` is stored beside `axis_y` because the
    # two answer different questions: the stick says what the thumb did, the latch
    # says what the jaw was asked for, and a dead zone means they disagree for
    # most of every push.
    ("trigger", "<f8", (N_HANDS,)),
    ("axis_y", "<f8", (N_HANDS,)),
    ("direction", "<i8", (N_HANDS,)),
    # The PICO's own instant, and when this machine read it. Both recorded, never
    # used to schedule anything — `deadline` above is what scheduling means here.
    # Their difference is the transport latency `intervene_lead_s` has to cover,
    # and it is the only measurement of it there is.
    ("t_device", "<f8"),
    ("t_recv", "<f8"),
])

#: P11's header: one record beside the ring carrying what is true of the PICO
#: rather than of any one row. Independently atomic aligned words, exactly like
#: `CONTROL`, and for the same reason — no reader needs two of them to agree.
#:
#: `samples` is the liveness counter and it is bumped on EVERY poll, intervening
#: or not. It is the only thing separating "nobody has taken over" from "the
#: headset died", because the ring itself cannot say: it is written only during
#: an intervention, so at CHECK time it is empty by design and an empty ring is
#: exactly what a healthy idle rig looks like.
INTERVENE_META = np.dtype([
    ("samples", "<u8"),                        # polls, bumped every iteration
    ("newest", "<u8"),                         # published row index
    ("t_last_sample", "<f8"),                  # when a FRESH sample last landed
    ("fresh", "<u8"),                          # is the stream live right now
    ("active", "<u8"),                         # is an intervention running
    ("engaged", "<u8", (N_HANDS,)),
    ("trigger", "<f8", (N_HANDS,)),
    ("axis_y", "<f8", (N_HANDS,)),
])


def chunk_dtype(horizon: int) -> np.dtype:
    """Chunk slots are horizon-sized, so the dtype is built per run.

    `n_actions` is what makes coverage testable: a chunk covers
    `[k_first_action, k_first_action + n_actions - 1]`, and in sync mode
    `n_actions == P`, not `H`.

    `n_plan >= n_actions` is how many rows are RESIDENT, which is a different
    question from how many will run. The gap is `cfg.plan_lookahead_ticks`:
    rows the policy decoded past the replan period, published so the trunk
    planner can see where the arm is being asked to go next and not commit to a
    branch that is a dead end two ticks later. They are planning input and never
    actions — nothing reads them through `covering`, which still gates on
    `n_actions` alone, so a tick outside `[kf, kf + n_actions)` remains
    uncovered exactly as before. With the knob at its default 0 the two words
    are equal and this field is inert.

    Rows in `[n_plan, horizon)` are whatever the chunk that last used this slot
    left there; `n_plan` is the high-water mark, and a reader that
    ignores it reads another chunk's stale poses.
    """
    return np.dtype([
        ("chunk_id", "<u8"),                 # 0 = empty slot
        ("k_first_action", "<i8"),
        ("n_actions", "<i4"),
        ("n_plan", "<i4"),                   # >= n_actions; see above
        # WHICH HALF OF THE CONTRACT this chunk speaks: 0 = eef
        # (`pose` is the payload, P6 runs the IK), 1 = joint (`arm_q` is the
        # payload, P6 skips only the IK). Per slot rather than per run so the
        # chunk is self-describing — P6 is another process and never sees the
        # policy class, and a record must say it too.
        # A joint chunk's `waist` is the policy's own `waist_q`, always (the
        # contract requires it); an eef chunk's is NaN and the waist is P6's —
        # the eef contract has no waist at all.
        ("mode", "<i4"),
        # The six instants of one replan. Six rather than a start/end pair so
        # that "inference latency" cannot silently include the anchor wait:
        # the SCHEDULED instant and the instant inference began are separate.
        ("t_trigger", "<f8"),                # the scheduled instant; P5 wakes
        ("t_anchor", "<f8"),                 # t_img_recv of the primary frame
        ("t_obs_received", "<f8"),           # every camera + the state settled
        ("t_cpu_processed", "<f8"),          # CPU preprocessing done
        ("t_chunk_received", "<f8"),         # device -> host sync returned
        ("t_chunk_written", "<f8"),          # this ring write begins
        # The payload. Whichever of `pose` / `arm_q` the mode does not use is
        # NaN, as is `waist` in an eef chunk — the same "no value here"
        # convention the target record uses for a tick with no IK behind it.
        ("pose", "<f8", (horizon, N_HANDS, POSE_DIM)),
        ("hand", "<f8", (horizon, N_HANDS, HAND_DIM)),
        ("waist", "<f8", (horizon, WAIST_DOF)),
        ("arm_q", "<f8", (horizon, N_HANDS, ARM_DOF // N_HANDS)),
    ])


def jpeg_capacity(height: int, width: int) -> int:
    """How many bytes of compressed source a camera slot reserves.

    Derived, never configured: a quarter of the uncompressed size. An image that
    will not compress 4:1 at any sane quality is noise, so this is generous, and
    deriving it means one fewer number that can be set wrong for a new camera.
    """
    return (height * width * 3) // 4


def camera_dtype(height: int, width: int,
                 jpeg_capacity_bytes: int | None = None) -> np.dtype:
    """One camera-ring slot.

    Metadata lives INSIDE the slot so it cannot be separated from its pixels —
    and so does `jpeg`, the compressed frame the robot actually sent.

    `image_server` publishes JPEG, so the compressed bytes already exist on the
    wire; P2 would otherwise decode and discard them. Keeping them in the SAME
    slot (rather than a parallel ring) makes the compressed source and the
    decoded pixels the same frame by construction — one publish, one sequence
    counter. A parallel ring would need its own index and a new invariant nobody
    can violate here, guarding a bug this layout cannot express.

    `jpeg_capacity_bytes` defaults to `jpeg_capacity(height, width)`. A view cut
    from a wider wire frame (one eye of the side_by_side head) passes the WIRE
    frame's capacity when it is the view that carries the JPEG, and 0 when it
    is not — the other eye's slots then reserve no compressed bytes at all
    (`resolved.CameraView.carries_jpeg`).

    `jpeg_nbytes == 0` means "not captured": a frame over the cap degrades the
    RECORD, never the run.

    `source_seq` is the PUBLISHER's own counter, carried through from the wire
    and stored beside the frame it belongs to for the same reason the JPEG is:
    separated from its pixels it would be a number about nothing. It is not
    `frame_id`, and the difference is the whole point — `frame_id` counts frames
    this process RECEIVED, `source_seq` counts frames the source SENT, so their
    divergence over a window is exactly the count dropped in transit. ZMQ PUB
    discards silently at its high-water mark by design (newest frame wins), and
    without this that loss is unobservable.

    `-1` means the source has no counter: the vendor `image_server` publishes a
    bare JPEG with no header at all, and `FakeCamera` has none either.

    There is no counter in here. The slot's sequence counter is a `uint64` in
    its own array beside the records (`cam_<name>_seq`), per the rule at the top
    of this file; the triple buffer's in-record `generation` is gone with it.
    """
    cap = jpeg_capacity(height, width) if jpeg_capacity_bytes is None \
        else int(jpeg_capacity_bytes)
    return np.dtype([
        ("frame_id", "<u8"),                 # frames THIS process received; 0 = empty
        ("source_seq", "<i8"),               # frames the SOURCE sent; -1 = no counter
        ("t_recv", "<f8"),
        ("jpeg_nbytes", "<u8"),              # 0 = not captured
        ("jpeg", "u1", (cap,)),
        ("pixels", "u1", (height, width, 3)),
    ])
