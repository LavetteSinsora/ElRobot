"""Typed accessors over the arena regions.

This module owns every ring scan in the system, so the four loops contain
scheduling logic and nothing else. If P5, P6 and P7 each wrote their own "find
the slot for tick k" they would each get the wrap-around wrong differently.

**Column scans are deliberately unlocked.** A scan for "the slot with the
smallest future deadline" reads the `k` and `deadline` columns without the
seqlock: both are naturally-aligned 8-byte fields, so an individual value is
either the old one or the new one, never a mix. The *chosen* slot is then read
back under its seqlock before its payload is used. That keeps a 500 Hz scan over
128 slots to two vector loads instead of 128 locked reads.

Each class's docstring names its one writer. A policy reaches the same classes
read-only through `reader.arena` (`open_read_only`); the regions, clocks and
read rules as a reader sees them are `docs/infra_logic/arena_layout_and_reads.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from g1d_deploy.runtime.shared_memory import schema
from g1d_deploy.runtime.shared_memory.primitives import Seqlock


class Control:
    """The atomic words. One writing process each — except
    `request`, which is the one deliberate exception and says why below."""

    def __init__(self, arena):
        self._c = arena.view("control")

    @property
    def session(self) -> int:
        return int(self._c[0]["session"])

    @session.setter
    def session(self, value: int) -> None:          # P0 only
        self._c[0]["session"] = value

    @property
    def k_exec(self) -> int:
        return int(self._c[0]["k_exec"])

    @k_exec.setter
    def k_exec(self, value: int) -> None:
        """P7 only — except the one window where P7 does not exist.

        `reset` stops P7, stores the zero and spawns a new one, so the
        word still has a single writer at every instant: P7 while there is a P7,
        and P0 precisely while there is not. That is the whole reason reset kills
        the interpolators instead of teaching them to rewind — a cursor that
        names the tick being executed cannot be moved backwards under a process
        that is executing ticks.
        """
        self._c[0]["k_exec"] = value

    @property
    def newest_chunk_id(self) -> int:
        return int(self._c[0]["newest_chunk_id"])

    @newest_chunk_id.setter
    def newest_chunk_id(self, value: int) -> None:  # P5 only, AFTER the payload
        self._c[0]["newest_chunk_id"] = value

    @property
    def warm(self) -> int:
        """Has P5 declared itself ready? 0 = not yet, 1 = warm.

        The one thing P0 knows about P5, and deliberately the only thing. What
        counts as warm is a property of the policy and the machine, so P5
        decides it (`p5_policy.stabilised`) and publishes one bit; P0 waits for
        the bit and never interprets it. Publishing a COUNT instead would put
        the threshold in P0, which is the one process that cannot know it.

        Note which word this is NOT. P5 does not advance `session` — that has
        exactly one writer and a second one racing P0's EXITING could
        walk the state backwards. P5 says "ready"; P0 decides what to do about
        it. Same visible result, one writer per word.
        """
        return int(self._c[0]["warm"])

    @warm.setter
    def warm(self, value: int) -> None:             # P5 only
        self._c[0]["warm"] = value

    @property
    def latency_s(self) -> float:
        """P5's estimate of one inference, trigger to chunk written, seconds;
        0.0 before it has measured any.

        Measured on the dummy observation during warmup (and again at a START
        whose settings changed), plus the worst wait the anchor may add (half
        the slowest observed camera's frame period). P0 checks an async
        rollout against it (`validate.check_async_timing`): an inference slower
        than the replan period starves the arm, silently, for good.
        """
        return int(self._c[0]["latency_ns"]) * 1e-9

    @latency_s.setter
    def latency_s(self, value: float) -> None:      # P5 only
        self._c[0]["latency_ns"] = max(0, int(round(value * 1e9)))

    @property
    def request(self) -> int:
        """The operator's pending intent: one of `schema.REQ_*`.

        **Two writers, on purpose — the one place in this system with two.**
        P10 stores a request; P0 clears it to `REQ_NONE` once it has taken it.
        The alternative was a monotonic `request_seq` that only P10 writes, with
        P0 holding a private cursor, and it was rejected as machinery bought for
        a failure that does not hurt here.

        What the race actually costs: if P10 stores EXIT in the same instant P0
        clears a RESET it just took, one store lands on the other and the EXIT is
        LOST. Compare that to the lost update the one-writer rule is about — a second writer
        racing P0's EXITING walks `session` *backwards*, and P5's
        `while session == ACTIVE` and P9's `while session < STOPPED` then resume
        a run that was shutting down. A dropped `request` is a button that did
        nothing. The state machine never moves anywhere it was not told to go.

        And it is a *visible* button that did nothing, which is what makes it
        tolerable: the dashboard renders this word and `session` by polling the
        arena, never from what it believes it sent, so a dropped request shows up
        as a banner that did not change. The operator presses it again. The
        window is the handful of instructions between P0's read and its clear,
        against a human hand on a mouse.

        Neither path is the emergency path. Ctrl-C is a signal and preempts
        whatever P0's main thread is doing; this word is only seen where P0 polls
        for it, so it cannot interrupt a homing ramp mid-write.
        """
        return int(self._c[0]["request"])

    @request.setter
    def request(self, value: int) -> None:          # P10 stores, P0 clears
        self._c[0]["request"] = value

    @property
    def intervene(self) -> int:
        """The intervention counter. Even = policy, odd = operator.

        **P11 only**, and note which word this is NOT. The obvious-looking home
        for "the operator has taken over" is `request`, beside RESET and EXIT,
        and it is wrong three times over. P0's request handling reads and CLEARS
        anything it finds, so an intervention parked there would survive about a
        millisecond. `request` is an EDGE that P0 consumes once, while this is a
        LEVEL held for as long as an intervention lasts. And `request` is the
        system's only two-writer word, affordable precisely because "a dropped
        request is a button that did nothing" — a dropped takeover, or an EXIT
        lost to a takeover racing it, is not that.

        Readers use it two ways and neither needs a second word: P5, P6, P7 and
        P8 read its PARITY (`intervention.is_intervening`) to know which world they
        are in, and P6 reads any CHANGE in it as "drop everything latched against
        the value we just left". The target rings and the intervention ring both
        stamp it into every row, so a slot from a previous world can never be
        selected.

        Monotonic, and never reset. P11 guarantees it leaves ACTIVE on an EVEN
        value — it hands back on its way out — which is what a rollout boundary
        needs, and it keeps this word to one writer where storing a zero from P0
        would not.
        """
        return int(self._c[0]["intervene"])

    @intervene.setter
    def intervene(self, value: int) -> None:        # P11 only
        self._c[0]["intervene"] = value

    # --- rollout settings and the prompt channel ----------------------------
    #
    # Each block is guarded by its own counter, used exactly as `Seqlock` uses
    # its per-slot counters: odd while the one writer is mid-write, even when
    # the block is consistent, and a reader retries until it copies the block
    # between two equal even reads. Records come back as private copies; the
    # Python face of `SETTINGS` is `rollout_settings.RolloutSettings`, kept out of this
    # module so the arena imports nothing above it.

    def _write_block(self, seq: str, **fields) -> int:
        c = self._c
        c[seq][0] += 1                               # -> odd: writing
        try:
            for name, value in fields.items():
                c[name][0] = value
        finally:
            c[seq][0] += 1                           # -> even: consistent
        return int(c[seq][0])

    def _read_block(self, seq: str, *names: str) -> tuple[int, list]:
        c = self._c
        while True:
            s0 = int(c[seq][0])
            if s0 & 1:
                continue
            out = [np.copy(c[name][0]) for name in names]
            if int(c[seq][0]) == s0:
                return s0, out

    @staticmethod
    def _text(buf, n) -> str:
        return bytes(np.asarray(buf, dtype=np.uint8)[:int(n)]).decode(
            "utf-8", errors="replace")

    @staticmethod
    def _bytes(text: str, capacity: int) -> tuple[np.ndarray, int]:
        """UTF-8 into a fixed box. Raises ValueError rather than truncating: a
        cut prompt is a different prompt, and a cut reason a wrong one is
        cheaper to shorten at the writer."""
        raw = str(text).encode("utf-8")
        if len(raw) > capacity:
            raise ValueError(f"{len(raw)} bytes of UTF-8, the arena holds "
                             f"{capacity}")
        buf = np.zeros(capacity, dtype=np.uint8)
        buf[:len(raw)] = np.frombuffer(raw, dtype=np.uint8)
        return buf, len(raw)

    @classmethod
    def _reason(cls, text: str) -> tuple[np.ndarray, int]:
        raw = str(text).encode("utf-8")[:schema.REASON_BYTES]
        return cls._bytes(raw.decode("utf-8", errors="ignore"),
                          schema.REASON_BYTES)

    # the request: P10 writes, P0 reads (at START, and nobody else ever)

    def requested_config(self) -> np.ndarray:
        """A consistent copy of `requested_config` (a `schema.SETTINGS` record)."""
        return self._read_block("requested_seq", "requested_config")[1][0]

    def request_config(self, record) -> None:                 # P10 only
        self._write_block("requested_seq", requested_config=record)

    # the active block: P0 writes at START, everyone latches

    def active(self) -> tuple[np.ndarray, str]:
        """`(active_config, active_prompt)`, consistent with each other."""
        _, (rec, n, buf) = self._read_block(
            "active_seq", "active_config", "active_prompt_len", "active_prompt")
        return rec, self._text(buf, n)

    def activate(self, record, prompt: str) -> None:          # P0 only
        """Write the active block. Does NOT bump `rollout_generation` — the
        caller does that after, so the counter is published after its payload."""
        buf, n = self._bytes(prompt, schema.PROMPT_BYTES)
        self._write_block("active_seq", active_config=record,
                          active_prompt_len=n, active_prompt=buf)

    @property
    def rollout_generation(self) -> int:
        return int(self._c[0]["rollout_generation"])

    @rollout_generation.setter
    def rollout_generation(self, value: int) -> None:  # P0 only, AFTER `activate`
        self._c[0]["rollout_generation"] = value

    @property
    def settings_applied(self) -> int:
        """The generation P5's policy has taken (see `schema.CONTROL`)."""
        return int(self._c[0]["settings_applied"])

    @settings_applied.setter
    def settings_applied(self, value: int) -> None:    # P5 only
        self._c[0]["settings_applied"] = value

    # the prompt request: P10 writes, P5 judges, P0 takes at START

    def prompt_request(self) -> tuple[int, str]:
        """`(prompt_seq, text)`. seq 0 = no prompt was ever requested."""
        seq, (n, buf) = self._read_block("prompt_seq", "prompt_len", "prompt")
        return seq, self._text(buf, n)

    def request_prompt(self, text: str) -> int:               # P10 only
        """Store a prompt request; returns its seq (always even, never 0)."""
        buf, n = self._bytes(text, schema.PROMPT_BYTES)
        return self._write_block("prompt_seq", prompt_len=n, prompt=buf)

    def verdict(self) -> tuple[int, int, str]:
        """`(verdict_of, verdict, reason)`: P5's judgement of request `verdict_of`."""
        _, (of, v, n, buf) = self._read_block(
            "verdict_seq", "verdict_of", "verdict", "verdict_reason_len",
            "verdict_reason")
        return int(of), int(v), self._text(buf, n)

    def judge(self, of: int, verdict: int, reason: str = "") -> None:  # P5 only
        buf, n = self._reason(reason)
        self._write_block("verdict_seq", verdict_of=of, verdict=verdict,
                          verdict_reason_len=n, verdict_reason=buf)

    # why P0 refused a START

    def refusal(self) -> tuple[int, str]:
        """`(refusal_seq, reason)`; seq 0 = nothing was ever refused."""
        seq, (n, buf) = self._read_block("refusal_seq", "refusal_len", "refusal")
        return seq, self._text(buf, n)

    def refuse(self, reason: str) -> int:                     # P0 only
        buf, n = self._reason(reason)
        return self._write_block("refusal_seq", refusal_len=n, refusal=buf)


# --- state ------------------------------------------------------------------


class StateRing:
    """Written by P1 at the lowstate rate; read by P5 (anchor) and P7 (gravity).

    A ring rather than a latest-value slot because the anchor must be the arm
    pose at the instant the image depicts — a measurement, which only ever
    exists in the past.
    """

    def __init__(self, arena):
        self._lock = Seqlock(arena.view("state_seq"), arena.view("state_rec"))
        self._rec = arena.view("state_rec")
        self.n = len(self._rec)
        self._cursor = 0
        self._index = 0

    def write(self, *, t_recv, q, dq, flange, grip_meas, t_grip_recv,
              column_h: float = 0.0, t_column_recv: float = 0.0, tick: int = 0) -> int:
        self._index += 1
        i = self._cursor
        with self._lock.write(i) as slot:
            slot["index"] = self._index
            slot["t_recv"] = t_recv
            slot["q"] = q
            slot["dq"] = dq
            slot["flange"] = flange
            slot["grip_meas"] = grip_meas
            slot["t_grip_recv"] = t_grip_recv
            # Defaulted, so a run with no column subscriber records an honest
            # "never received" rather than refusing to write a state sample.
            slot["column_h"] = column_h
            slot["t_column_recv"] = t_column_recv
            # The robot's clock at the sample; 0 from a source without one.
            slot["tick"] = tick
        self._cursor = (i + 1) % self.n
        return self._index

    def latest(self) -> np.ndarray | None:
        idx = self._rec["index"]
        if not idx.any():
            return None
        return self._lock.snapshot(int(np.argmax(idx)), schema.STATE)

    def nearest(self, t: float) -> np.ndarray | None:
        """The sample whose `t_recv` is closest to `t` — the anchor rule."""
        idx = self._rec["index"]
        valid = idx > 0
        if not valid.any():
            return None
        dist = np.where(valid, np.abs(self._rec["t_recv"] - t), np.inf)
        return self._lock.snapshot(int(np.argmin(dist)), schema.STATE)

    def window(self, t0: float, t1: float) -> np.ndarray:
        """Every sample with `t0 <= t_recv <= t1`, ascending — the policy
        reader's `states()`. Copy-and-verify, vectorised.

        The whole ring is copied between two reads of the counters, and a slot
        is kept only if its counter was even and unchanged across the copy: the
        per-slot seqlock rule, applied to every slot at once. A slot P1 was
        writing during the copy is DROPPED rather than retried — the writer laps
        one slot per ~1 ms, so at most a sample or two at the leading edge go
        missing, and a torn sample is never returned. Retrying instead would
        make this an unbounded spin on the hottest slot in the arena.

        ~one memcpy of the ring (1 s of history is ~1024 slots of ~700 B).
        """
        seq = self._lock.seq
        s0 = seq.copy()
        rec = self._rec.copy()
        s1 = seq.copy()
        ok = ((s0 == s1) & (s0 % 2 == 0) & (rec["index"] > 0)
              & (rec["t_recv"] >= t0) & (rec["t_recv"] <= t1))
        out = rec[ok]
        return out[np.argsort(out["index"], kind="stable")]



class CommandRing:
    """P7's publish log (`schema.COMMAND`). Written by P7, read by P9 only.

    The state ring's shape exactly — an absolute `index`, one seqlock per slot,
    a writer-local cursor — so P9's sweep is the state sweep again and cannot
    double-count a slot overwritten mid-sweep.
    """

    def __init__(self, arena):
        self._lock = Seqlock(arena.view("command_seq"), arena.view("command_rec"))
        self._rec = arena.view("command_rec")
        self.n = len(self._rec)
        self._cursor = 0
        self._index = 0

    def write(self, *, t, k_exec, q, dq, tau, tau_ff, q_lin) -> int:
        self._index += 1
        i = self._cursor
        with self._lock.write(i) as slot:
            slot["index"] = self._index
            slot["t"] = t
            slot["k_exec"] = k_exec
            slot["q"] = q
            slot["dq"] = dq
            slot["tau"] = tau
            slot["tau_ff"] = tau_ff
            slot["q_lin"] = q_lin
        self._cursor = (i + 1) % self.n
        return self._index

    def latest(self) -> np.ndarray | None:
        idx = self._rec["index"]
        if not idx.any():
            return None
        return self._lock.snapshot(int(np.argmax(idx)), schema.COMMAND)

# --- chunks -----------------------------------------------------------------


class Contribution(NamedTuple):
    """One chunk's opinion about one tick.

    `pose` for an eef chunk, `arm_q` and `waist` (the policy's `waist_q`,
    always given in joint mode) for a joint one; the others None. An eef
    chunk's `waist` is None: the waist is P6's.
    """

    chunk_id: int
    pose: np.ndarray | None         # (2, 7)
    hand: np.ndarray                # (2, 6)
    waist: np.ndarray | None        # (3,)
    arm_q: np.ndarray | None = None  # (2, 7)
    #: Which row of its chunk this is: `k - k_first_action`, so how far into
    #: that chunk's prediction the tick lies — an age-weighted blend's input
    #: (ACT's temporal ensembling weighs a prediction by it). -1 = unknown.
    row: int = -1


class PlanRows(NamedTuple):
    """One chunk's decoded-but-unexecuted tail — planning input, never actions.

    `k_first` is the tick the first of these rows WOULD have been, had the
    chunk run `n_plan` actions instead of `n_actions`. No tick in that range is
    covered and none will be solved; the index is here so the planner can place
    the rows on the same time axis as the ones that do run.
    """

    chunk_id: int
    k_first: int
    pose: np.ndarray | None         # (m, 2, 7), None for a joint chunk
    hand: np.ndarray                # (m, 2, 6)
    waist: np.ndarray | None        # (m, 3), None unless the policy gave one


class ChunkRing:
    """Written by P5, read by P6. Slot index is `chunk_id mod chunk_slots`."""

    def __init__(self, arena, horizon: int):
        self._lock = Seqlock(arena.view("chunk_seq"), arena.view("chunk_rec"))
        self._rec = arena.view("chunk_rec")
        self.n = len(self._rec)
        self.horizon = horizon

    def write(self, *, chunk_id: int, k_first_action: int, n_actions: int,
              hand, pose=None, arm_q=None, waist=None,
              n_plan: int | None = None, **stamps: float) -> None:
        """`stamps` is the six replan instants (`schema.chunk_dtype`), by name. Passed as a mapping rather
        than six positional floats so that the schema is the single place their
        names and order are decided — a transposed pair of timestamps produces
        a plausible negative latency and no error.

        `n_plan` is how many rows to store; `None` means `n_actions`, which is
        every write this system made before `cfg.plan_lookahead_ticks` existed.
        The extra rows are planning input for P6's trunk planner and are NOT
        actions: `n_actions` is unchanged by them, so coverage — the only
        executability predicate — answers exactly as it did.
        """
        n_plan = n_actions if n_plan is None else int(n_plan)
        # The mode is the payload's shape, not a separate argument that could
        # disagree with it. An eef chunk carrying a waist is refused rather than
        # stored: the eef contract has no waist, and a ring that
        # accepted one would let P6 command a trunk nobody is allowed to ask for.
        if (pose is None) == (arm_q is None):
            raise ValueError("a chunk carries exactly one of pose (eef) and "
                             "arm_q (joint)")
        if pose is not None and waist is not None:
            raise ValueError("an eef chunk carries no waist: the IK owns it "
                             "")
        if arm_q is not None and waist is None:
            raise ValueError("a joint chunk always carries the waist: the "
                             "contract requires `waist_q` in joint mode")
        if n_plan < n_actions:
            raise ValueError(
                f"n_plan {n_plan} < n_actions {n_actions}: the plan must "
                f"contain the actions, since P6 reads it as their continuation")
        if n_plan > self.horizon:
            raise ValueError(
                f"chunk of {n_actions} actions + plan to {n_plan} exceeds "
                f"horizon {self.horizon}")
        with self._lock.write(chunk_id % self.n) as slot:
            slot["chunk_id"] = chunk_id
            slot["k_first_action"] = k_first_action
            slot["n_actions"] = n_actions
            # AFTER `n_actions`, so the store order `covers_any` documents and
            # depends on (chunk_id, then k_first_action, then n_actions) is
            # still a prefix of this sequence.
            slot["n_plan"] = n_plan
            for name, value in stamps.items():
                slot[name] = value
            slot["mode"] = 1 if arm_q is not None else 0
            # Absent blocks are NaN over the WHOLE horizon, not just `n_plan`:
            # a reader indexing a stale row of an absent block must see "no
            # value", never the previous occupant's numbers in the other mode.
            for name, value in (("pose", pose), ("arm_q", arm_q),
                                ("waist", waist)):
                if value is None:
                    slot[name] = np.nan
                else:
                    slot[name][:n_plan] = value[:n_plan]
            slot["hand"][:n_plan] = hand[:n_plan]

    def _row(self, i: int, k: int) -> Contribution | None:
        """Seqlock-read only the metadata and the single row for tick `k`.

        Copying the whole horizon here would be ~30 KB per slot per tick; the
        row is 200 bytes. The consistency guarantee is unchanged — the counter
        is checked after the row is copied.
        """
        seq, rec = self._lock.seq, self._lock.rec
        while True:
            s0 = int(seq[i])
            if s0 & 1:
                continue
            slot = rec[i]
            cid = int(slot["chunk_id"])
            kf = int(slot["k_first_action"])
            n = int(slot["n_actions"])
            if cid == 0 or not (kf <= k < kf + n):
                if int(seq[i]) == s0:
                    return None
                continue
            row = k - kf
            joint = int(slot["mode"]) == 1
            out = Contribution(
                cid,
                None if joint else slot["pose"][row].copy(),
                slot["hand"][row].copy(),
                slot["waist"][row].copy() if joint else None,
                slot["arm_q"][row].copy() if joint else None,
                row)
            if int(seq[i]) == s0:
                return out

    def get(self, chunk_id: int) -> np.ndarray | None:
        """A consistent private copy of chunk `chunk_id`'s whole slot (a
        `schema.chunk_dtype(H)` record), or None once its slot holds another
        chunk. Its `k_first_action` and `n_actions` say which ticks it
        covers: row `j` is tick `k_first_action + j`. In async that tick is due
        `(j + 1)·dt` after `t_trigger`; in sync P6 re-stamps the epoch at
        adoption, so read the deadline from the target ring. Slot 0 holds
        `chunk_id == 0` until chunk 1 is written: never ask for id 0."""
        i = int(chunk_id) % self.n
        rec = self._lock.snapshot(i, self._rec.dtype)
        return rec if int(rec["chunk_id"]) == int(chunk_id) else None

    def covering(self, k: int, *,
                 adopted_through: int | None = None) -> list[Contribution]:
        """Every live chunk that covers tick `k`, ascending `chunk_id`.

        Sync always yields exactly one, which is why blending is inert there.

        `adopted_through` means what it means for `covers_any`: slots holding a
        chunk newer than it are left out. P6 passes its `last_read_chunk_id`,
        so a chunk whose payload has landed but whose id has not is
        never blended into a tick one iteration before P6 adopts it — which
        would hand the newest-wins blend an action from a chunk P6 has not
        re-anchored to, and the arm planner a pose its plan does not contain.
        `None` (what is physically in the ring) is for tests and offline tools.
        """
        out = [c for c in (self._row(i, k) for i in range(self.n))
               if c is not None
               and (adopted_through is None or c.chunk_id <= adopted_through)]
        out.sort(key=lambda c: c.chunk_id)
        return out

    def covers_any(self, k: int, *, adopted_through: int | None) -> bool:
        """Is tick `k` covered by a live chunk? No payload copy.

        **`adopted_through` is required, and it is the whole safety of this
        call.** It names the newest `chunk_id` the caller has adopted — for P6,
        `last_read_chunk_id`. Slots holding anything newer are ignored.

        Without it this read races P5's publish — measured on 36 of 37
        chunks of one set of rollouts. P5 writes the payload and only then the id,
        so between those two stores a chunk is *coverable but not yet
        announced*. A cursor advance that consults coverage and not
        the id walks onto the new chunk's first tick one iteration
        early — through the branch that does NOT re-stamp the epoch —
        and stamped that tick with the previous chunk's clock, ~100 ms in the
        past. P7 discarded it as stale, so the robot was unharmed and nothing
        reported anything.

        A seqlock here would only narrow that window, not close it: even a
        perfectly consistent read sees the chunk the moment its write
        *completes*, which is still before the id is stored. Only the id can
        say "announced", which is why the limit is this and not a lock.

        `None` means "any chunk, announced or not". That is a legitimate answer
        for a test or a diagnostic asking what is physically in the ring, and
        never for a caller making a control decision.

        The read order below is load bearing. `write` stores `chunk_id` first,
        then `k_first_action`, then `n_actions`; x86 keeps stores ordered, so a
        reader that evaluates the coverage mask *before* it reads `chunk_id`
        cannot see a fresh `k_first_action` beside a stale id — which is
        precisely the combination that would slip an unannounced chunk past the
        filter. The comparisons, not the view assignments, are the reads.
        """
        kf = self._rec["k_first_action"]
        n = self._rec["n_actions"]
        covered = (kf <= k) & (k < kf + n)               # reads kf and n
        cid = self._rec["chunk_id"].copy()               # reads chunk_id, after
        live = covered & (cid != 0)
        if adopted_through is not None:
            live &= cid <= adopted_through
        return bool(np.any(live))

    def plan_rows(self, k: int, *, adopted_through: int | None) -> PlanRows | None:
        """The rows the chunk covering `k` decoded but will NOT execute.

        `None` unless `cfg.plan_lookahead_ticks` put some there, which is every
        run before that knob existed. Its one caller is `p6_target._plan_trunk`,
        and what it does with these is the only thing they are for: extend the
        trajectory the trunk planner optimizes over, WITHOUT extending the
        trajectory P6 solves and publishes. A row returned here never becomes a
        target — `covering()` is what makes a tick executable and it does not
        look at `n_plan`.

        Why a separate accessor instead of widening `covering`: coverage is the
        executability predicate, and P5's window scan, P6's cursor advance
        and the recorder's chunk partition all read it as "these are the ticks
        this chunk owns". Widening it would hand a stale epoch to P5's scan and
        break the tick-line partition rollout inspection assumes, to say
        something coverage was never asked. So the extra rows get their own
        word, their own reader, and no say in what runs.

        `adopted_through` is required for the same reason `covers_any` requires
        it: P5 stores the payload before the id, so an unannounced chunk is
        briefly readable. A trunk planned against one would be planned against
        poses whose chunk P6 has not adopted.
        """
        seq, rec = self._lock.seq, self._lock.rec
        best = None
        for i in range(self.n):
            while True:
                s0 = int(seq[i])
                if s0 & 1:
                    continue
                slot = rec[i]
                cid = int(slot["chunk_id"])
                kf = int(slot["k_first_action"])
                n = int(slot["n_actions"])
                m = int(slot["n_plan"])
                ok = (cid != 0 and kf <= k < kf + n and m > n
                      and (adopted_through is None or cid <= adopted_through))
                if not ok:
                    if int(seq[i]) == s0:
                        break
                    continue
                joint = int(slot["mode"]) == 1
                out = PlanRows(cid, kf + n,
                               None if joint else slot["pose"][n:m].copy(),
                               slot["hand"][n:m].copy(),
                               (slot["waist"][n:m].copy()
                                if joint else None))
                if int(seq[i]) == s0:
                    if best is None or out.chunk_id > best.chunk_id:
                        best = out
                    break
        return best


# --- targets ----------------------------------------------------------------



# --- the intervention --------------------------------------------


class InterveneRow(NamedTuple):
    """One row of the intervention ring: the action the operator is asking for."""

    index: int
    deadline: float
    pose: np.ndarray        # (2, 7) absolute flange pose, base frame
    hand: np.ndarray        # (2, 6)
    waist: np.ndarray       # (3,)


class InterveneRing:
    """Written by P11 while intervening; read by P6 and swept by P9.

    A ring rather than a latest-value slot for the reason the state ring is one:
    the RECORD wants every row, and P9 sweeps regions rather than being handed
    values. The control path only ever wants the newest, which is `newest()`.

    `meta` is a single record beside it carrying what is true of the PICO rather
    than of one row. Its `samples` counter is bumped on every poll whether or not
    anyone is intervening, and it is the only thing separating "nobody has taken
    over" from "the headset stopped" — the ring itself cannot say, being written
    only during an intervention.
    """

    def __init__(self, arena):
        self._lock = Seqlock(arena.view("intervene_seq"),
                             arena.view("intervene_rec"))
        self._rec = arena.view("intervene_rec")
        self._meta = arena.view("intervene_meta")
        self.n = len(self._rec)
        self._cursor = 0
        self._index = 0

    # --- P11 ---------------------------------------------------------------

    def write(self, **fields) -> int:
        """One row. `index` is this ring's own and is never passed in."""
        self._index += 1
        i = self._cursor
        with self._lock.write(i) as slot:
            slot["index"] = self._index
            for name, value in fields.items():
                slot[name] = value
        self._cursor = (i + 1) % self.n
        # AFTER the row, for the reason P5 stores `newest_chunk_id` after its
        # payload: it is what a reader takes as permission to look.
        self._meta[0]["newest"] = self._index
        return self._index

    def note(self, **fields) -> None:
        """Update the header. P11 only, and deliberately NOT under the ring's
        seqlock: these are independently atomic aligned words, exactly like
        `CONTROL`, and no reader needs two of them to agree."""
        for name, value in fields.items():
            self._meta[0][name] = value

    def meta(self) -> np.ndarray:
        return self._meta[0]

    # --- P6 ----------------------------------------------------------------

    def newest(self, intervene: int, since: int = 0) -> InterveneRow | None:
        """The most recent row belonging to THIS intervention, newer than
        `since`. None if there is none.

        `since` is the caller's own last row, and it makes the common answer cost
        one scalar read. P6 spins as fast as it can whenever it has nothing to
        solve, and at the PICO's rate it has nothing new on the large majority of
        those iterations; scanning two 8 kB columns to discover that is megabytes
        a millisecond through the shared segment, evicting cache from the cores
        P7 and P1 sit on.

        `argmax` over `index` filtered by the counter, which is why this ring
        needs neither a wipe at the handback nor a separate "newest" cursor. A
        wipe would be N seqlock writes and would leave a window in which the ring
        looks empty; a cursor would be a second source of truth that can tear
        against the row it names. Here the index and its payload go in under ONE
        seqlock and cannot disagree.

        **Deliberately not "the oldest unsolved".** P6 solves the newest and
        skips whatever it missed. Every row is an absolute pose computed from a
        fixed anchor, so a skipped row leaves no increment to make up and a slow
        solve costs one dropped target rather than a growing backlog the operator
        would feel as lag.
        """
        if int(self._meta[0]["newest"]) <= since:
            return None
        idx = np.where(self._rec["intervene"] == intervene, self._rec["index"], 0)
        if not idx.any():
            return None
        snap = self._lock.snapshot(int(np.argmax(idx)), schema.INTERVENE)
        if int(snap["intervene"]) != intervene:
            return None                 # wrapped under us; try again next poll
        return InterveneRow(int(snap["index"]), float(snap["deadline"]),
                            snap["pose"].copy(), snap["hand"].copy(),
                            snap["waist"].copy())


class TargetEntry(NamedTuple):
    k: int
    deadline: float
    payload: np.ndarray


class TargetRing:
    """The one target ring. P6 writes it; P7 and P8 read different columns.

    Written by P6 during ACTIVE and by P0 during HOME/EXITING. That is one
    writer at a time, not two: P6 does not exist outside ACTIVE, which is what
    keeps the one-writer rule true for this region.

    `column` picks which field `get`, `latest_below` and `next_after` return as
    the entry's payload — `"q"` for P7's 17 body joints, `"hand"` for P8's
    `(2, 6)` end-effector command. `view()` opens a second handle over the SAME
    memory and the same seqlock state, so the two readers cannot come to
    disagree about a tick, and `write` cannot produce one payload without the
    other. "Same `k`, same deadline" and "body and hand, both or neither" are
    not rules P6 follows; they are properties of there being a single row
    under a single lock.
    """

    DTYPE = schema.TARGET

    #: What `write` stores for a tick that had no IK behind it. Built once: a
    #: fresh `np.full` per homing tick would allocate inside the seqlock's
    #: critical section, which is the one place in this file that must not.
    _NO_POSE = np.full((schema.N_HANDS, schema.POSE_DIM), np.nan)
    #: Likewise for the waist the solve was pinned at. Same reason it is a
    #: constant: no allocation inside the seqlock's critical section.
    _NO_WAIST = np.full(schema.WAIST_DOF, np.nan)
    #: And for a tick with no hand command — P0's ramps write both, so this is
    #: only reachable from a test that writes a body target alone.
    _NO_HAND = np.zeros((schema.N_HANDS, schema.HAND_DIM))

    def __init__(self, arena, column: str = "q"):
        self._rec = arena.view("target_rec")
        self._seq = arena.view("target_seq")
        self._lock = Seqlock(self._seq, self._rec)
        self.n = len(self._rec)
        self.column = self._check(column)

    @staticmethod
    def _check(column: str) -> str:
        # Checked BEFORE anything is read: a typo here is a programming error
        # and should say so, not surface later as a view that reads a column
        # nobody meant.
        if column not in schema.TARGET.names:
            raise ValueError(f"{column!r} is not a column of TARGET "
                             f"({schema.TARGET.names})")
        return column

    def view(self, column: str) -> "TargetRing":
        """A second reader over this ring, returning `column` as the payload.

        Shares the arrays and the lock rather than re-mapping the region, which
        is what makes it a view of THIS ring and not another ring that happens
        to sit at the same offset.
        """
        other = object.__new__(TargetRing)
        other._rec, other._seq, other._lock = self._rec, self._seq, self._lock
        other.n = self.n
        other.column = self._check(column)
        return other

    def _entry(self, snap: np.ndarray) -> TargetEntry:
        return TargetEntry(int(snap["k"]), float(snap["deadline"]),
                           snap[self.column].copy())

    def _written(self) -> np.ndarray:
        """Which slots have ever been written.

        The seqlock counter carries this for free: the arena is zero-filled at
        creation, so `seq == 0` is exactly "never written" and cannot be
        forgotten the way an explicit sentinel pass can. It matters most for
        tick 0, which is a real tick — the end of homing — and would otherwise
        be indistinguishable from a zeroed slot.
        """
        return (self._seq != 0) & (self._rec["k"] != schema.K_UNSET)

    # --- the writer (P6, and P0 for its ramps) -------------------------------

    def write(self, k: int, deadline: float, payload, hand=None, *,
              intervene: int, err: float = np.nan, pose=None,
              waist_ik=None) -> None:
        """One tick, both payloads, under one lock.

        `payload` is the 17 body joints and `hand` the `(2, 6)` command; the
        three that follow describe where the tick came from — the IK's residual,
        the Cartesian pose it was solving for, and the trunk it was solving in.

        `intervene` says which world the row belongs to. **Required,
        and keyword-only, on purpose.** Every writer has to answer the question,
        and the one that would have got a default wrong is P0: its exit ramp runs
        after a rollout in which somebody intervened, so `control.intervene` is
        EVEN but not zero, and a ramp stamped with a default 0 would be filtered
        out by P7 — the arm would stay stiff wherever the last target left it.

        All three default to NaN rather than 0.0 so that P0's ramps — homing and
        the exit ramp, which no IK ever touched — record as *undefined* rather
        than as a perfect solve of a target at the origin, in a frame that
        happens to be upright. A reader can then tell "P0 wrote this" from "P6
        wrote this" by looking, instead of inferring it from `k < 1`.

        **None of the three is on any read path.** `get()`, `latest_below()` and
        `next_after()` still return a three-field `TargetEntry`, so P7's 500 Hz
        segment advance and P6's seed lookup are byte-for-byte the work they
        were — they never touch these fields and never copy them. The recorder
        reads them with its own full-record snapshot (`p9_record._sweep_targets`)
        precisely so that the control path does not have to carry them.
        """
        with self._lock.write(k % self.n) as slot:
            slot["k"] = k
            slot["deadline"] = deadline
            slot["intervene"] = intervene
            slot["q"] = payload
            slot["hand"] = self._NO_HAND if hand is None else hand
            slot["err"] = err
            slot["pose"] = self._NO_POSE if pose is None else pose
            slot["waist_ik"] = self._NO_WAIST if waist_ik is None else waist_ik

    def clear(self) -> None:
        """Make every slot unselectable. P0 between rollouts.

        **Both fields, and the deadline is the load-bearing one.** `k` alone is
        enough while every reader is stopped, which is P0's case; it is not
        enough against a live P7, whose column scans are deliberately unlocked. A
        reader can choose a slot off the `deadline` column and then snapshot it
        after this has passed, coming away with `k = K_UNSET` beside a deadline
        that still looks valid — and P7 would store that `k` as `k_exec`, which
        is `-2**63`, with P6's `k_exec + 2` following it there.

        Writing an expired deadline closes it inside machinery that already
        exists: `next_after` re-checks the deadline after its snapshot, so the
        torn read is rejected by a test it already performs.
        """
        for i in range(self.n):
            with self._lock.write(i) as slot:
                slot["k"] = schema.K_UNSET
                slot["deadline"] = -np.inf

    # --- the readers ---------------------------------------------------------

    def get(self, k: int) -> TargetEntry | None:
        """The entry for exactly tick `k`, or None if it was never written (or
        has since been overwritten by a later wrap)."""
        snap = self.full(k)
        return None if snap is None else self._entry(snap)

    def full(self, k: int):
        """The WHOLE record for tick `k`, not just the payload column.

        `get` answers with the target itself, which is all the control path ever
        wants. P10 also needs the columns beside it — `pose`, the absolute flange
        pose the IK was given for this tick, and `err`, its residual — to draw
        the commanded overlay against the arm that was solved for it; P0's exit
        ramp needs `q` and `hand` from the same tick to seed both its ramps.

        One snapshot, not two reads. `get(k).payload` followed by a separate read
        of `pose` could straddle a write and pair one tick's joints with another
        tick's Cartesian target — an overlay that is individually plausible and
        jointly a lie.
        """
        i = k % self.n
        if self._seq[i] == 0:
            return None
        snap = self._lock.snapshot(i, self.DTYPE)
        return None if int(snap["k"]) != k else snap

    def earliest_written_in(self, k_lo: int, k_hi: int) -> TargetEntry | None:
        """P5's window scan.

        The **earliest** written tick, not the latest: P6 writes in increasing
        `k`, so the earliest is the first to exist, and P5 can compute its wake
        time as soon as the window opens.
        """
        for k in range(k_lo, k_hi + 1):
            entry = self.get(k)
            if entry is not None:
                return entry
        return None

    def latest_below_full(self, k: int):
        """The whole record of the written target with the largest `k` below
        this one, or None. `latest_below` is this with the payload picked out.

        **Deliberately NOT filtered on `intervene`**, and that asymmetry against
        `next_after` is the point. SELECTION asks "may this target be executed",
        which is a question about authority and must never cross a takeover.
        SEEDING asks "what configuration is the arm in", which is a question
        about geometry and does not care who put it there. At a takeover the
        first operator solve seeds from the last policy posture; at a handback
        the first policy solve seeds from the last operator posture. Both are
        what continuity of the IK's null space requires — a seed that jumped at
        the seam would swing the elbow through a different branch of the solution
        while the flange went nowhere.
        """
        ks = self._rec["k"]
        valid = (ks < k) & (ks > k - self.n) & self._written()
        if not valid.any():
            return None
        best = int(np.argmax(np.where(valid, ks, np.iinfo(np.int64).min)))
        snap = self._lock.snapshot(best, self.DTYPE)
        if int(snap["k"]) == schema.K_UNSET:
            return None                 # cleared between the scan and the read
        return snap

    def latest_below(self, k: int) -> TargetEntry | None:
        """P6's seed: the written target with the largest `k` below this one."""
        snap = self.latest_below_full(k)
        return None if snap is None else self._entry(snap)

    def line_after(self, t: float, until: float, intervene: int) -> list[TargetEntry]:
        """The written targets of world `intervene` with deadline > `t`, in
        deadline order, stopping at the first whose deadline reaches `until`.

        P7's look-ahead reads the target line with this (`shaping.line_ahead`):
        a few entries per period, each read under its seqlock and re-checked
        like `next_after`'s, so a slot rewritten mid-scan is skipped, not torn.
        """
        dl = self._rec["deadline"]
        valid = (self._written() & (dl > t)
                 & (self._rec["intervene"] == intervene))
        idx = np.flatnonzero(valid)
        out = []
        for i in idx[np.argsort(dl[idx], kind="stable")]:
            snap = self._lock.snapshot(int(i), self.DTYPE)
            if (float(snap["deadline"]) <= t
                    or int(snap["k"]) == schema.K_UNSET
                    or int(snap["intervene"]) != intervene):
                continue
            out.append(self._entry(snap))
            if out[-1].deadline >= until:
                break
        return out

    def next_after(self, t: float, intervene: int) -> TargetEntry | None:
        """The entry with the SMALLEST deadline strictly greater than `t`, among
        the slots belonging to world `intervene`.

        P7 and P8's segment advance. Stale slots — ticks already executed, or
        ticks P6 wrote below `k_exec` — have deadlines in the past
        and are excluded by the comparison, which is why they are harmless.

        **`intervene` is required, and it is what makes a takeover safe without
        wiping the ring**. At the instant the operator takes over, the
        ring holds up to `P` future-dated policy targets — ~830 ms of committed
        motion at P = 25, 30 Hz. Operator targets, written ~35 ms out, then overwrite
        those slots one at a time as rows arrive, and for as long as they keep
        arriving the smallest future deadline is always one of theirs. The moment
        the stream pauses, the un-overwritten policy target at the far end of the
        horizon becomes the smallest future deadline and the arm lurches to a
        pose nobody asked for. Filtering here makes that unreachable.

        It is also what preempts a segment already in flight, because P7 re-asks
        this question the instant the counter moves rather than waiting for its
        current segment to expire.
        """
        dl = self._rec["deadline"]
        valid = (self._written() & (dl > t)
                 & (self._rec["intervene"] == intervene))
        if not valid.any():
            return None
        best = int(np.argmin(np.where(valid, dl, np.inf)))
        snap = self._lock.snapshot(best, self.DTYPE)
        # Re-checked after the snapshot, all three, because the scan above is
        # unlocked and a slot may have been rewritten under it. The snapshot is
        # internally consistent — that is what the seqlock buys — so these three
        # tests are asked of one coherent record rather than of three columns
        # read at three instants.
        if (float(snap["deadline"]) <= t
                or int(snap["k"]) == schema.K_UNSET
                or int(snap["intervene"]) != intervene):
            return None
        return self._entry(snap)


# --- cameras ----------------------------------------------------------------


class CameraRing:
    """One camera VIEW's ring of N frame slots.

    A ring keeps the last N frames, so a policy can read history
    (`policy_contract.Reader`), and at `N = CAMERA_MIN_SLOTS = 3` — a view
    with no declared history — it behaves exactly as a triple buffer: one slot being written, the
    newest, and the one before it.

    **Three regions** (`layout.build`):

    * `head` `(1,)`: the `frame_id` of the newest PUBLISHED frame. The only
      shared word the writer stores outside a slot, and it stores it LAST.
    * `seq` `(N,) uint64`: one sequence counter per slot. Odd while the writer
      is filling the slot, even when it is consistent. In its own array, never a
      field of the record (the rule at the top of `schema.py`).
    * `rec` `(N,)`: `schema.camera_dtype` — `frame_id`, `source_seq`, `t_recv`,
      the JPEG and the pixels, inside the slot so metadata and pixels cannot
      come apart.

    **Where frame `f` lives:** slot `(f - 1) % N`. `frame_id`s are assigned by
    the one writer, consecutively from 1, so the slot is a function of the id
    and a reader never searches for a frame it was given the id of.

    **The writer** (P2 — `spare()`, then `publish()`): `spare()` makes the next
    slot's counter odd and hands the slot out, so the decoder can write ~1 MB
    straight into it with no copy; `publish()` stamps the metadata, makes the
    counter even, then stores `head`. Wait-free: the writer never looks at a
    reader. While P2 blocks for the next frame, the slot it will decode into is
    odd — that slot held the OLDEST frame, which is one reason N has headroom.

    **The read rule, for every reader** (copy-and-verify): read the
    slot's counter `s0`; if odd, the slot is being written — skip it; check the
    slot's `frame_id` is the frame you mean; copy what you need; read the
    counter again and keep the copy only if it still equals `s0`. A frame that
    fails is DROPPED, never retried and never returned torn: the writer only
    comes back to a slot N frames after writing it, so a failure means the
    frame is gone and waiting would not bring it back. The one exception is
    "the newest frame" (`peek`, `peek_counters`), which retries from `head` —
    there is always a newest frame, and a retry costs a few scalar loads.

    Metadata-only scans (`window`) apply the same rule to every slot at once,
    vectorised, exactly as `StateRing.window` does.

    Two classes of reader with different needs, neither paying for the other's
    bytes: P5 and the policy reader want pixels, never the JPEG; P9 and the
    dashboard want the JPEG and the header, never the pixels. At ~1.15 MB a slot
    the difference is real at 30 Hz.
    """

    def __init__(self, arena, name: str):
        self.name = name
        self._head = arena.view(f"cam_{name}_head")
        self._seq = arena.view(f"cam_{name}_seq")
        self._rec = arena.view(f"cam_{name}_rec")
        self.n = len(self._rec)
        self.dtype = self._rec.dtype
        self.jpeg_capacity = self.dtype["jpeg"].shape[0]
        # Writer-local. Resumed from `head` rather than from zero, so a camera
        # process that is ever restarted over a live ring keeps the frame-id ->
        # slot mapping every reader relies on.
        self._frame_id = int(self._head[0])
        self._open = False

    def _slot(self, frame_id: int) -> int:
        return (int(frame_id) - 1) % self.n

    # --- writer (P2/P3/P4) -------------------------------------------------

    def spare(self) -> np.ndarray:
        """The slot the next frame goes into, marked "being written".

        Idempotent until `publish()`: the loop may ask for it more than once
        per frame. A counter already odd — a writer that died mid-frame — is
        left odd rather than incremented back to even over a half-written slot.
        """
        i = self._frame_id % self.n
        if not self._open:
            if not int(self._seq[i]) & 1:
                self._seq[i] += 1                 # -> odd: being written
            self._open = True
        return self._rec[i]

    def publish(self, t_recv: float, jpeg: bytes | None = None,
                seq: int | None = None) -> int:
        """Publish the spare slot, optionally carrying its compressed source.

        `jpeg` is what the robot sent, stored verbatim so the recorder does no
        encoding and adds no second lossy generation. A frame that does not fit
        the slot's capacity is stored as `jpeg_nbytes = 0` — "not captured" —
        because degrading the RECORD is always preferable to failing the RUN,
        and the recorder's index makes the gap visible rather than silent. A
        view that carries no JPEG (the right eye of a split camera) has
        capacity 0 and always stores 0.

        `seq` is the source's own counter, or None for a source that has none
        (the vendor `image_server`'s bare JPEG, or a fake). It rides in the same
        slot as the frame it numbers; see `schema.camera_dtype`.

        Order: metadata, then the counter even, then `head`. A reader that
        finds the new id in `head` therefore finds its slot already complete.
        """
        slot = self.spare()
        i = self._frame_id % self.n
        fid = self._frame_id + 1
        slot["t_recv"] = t_recv
        slot["frame_id"] = fid
        slot["source_seq"] = -1 if seq is None else int(seq)
        if jpeg is not None and len(jpeg) <= self.jpeg_capacity:
            n = len(jpeg)
            slot["jpeg"][:n] = np.frombuffer(jpeg, dtype=np.uint8)
            slot["jpeg_nbytes"] = n
        else:
            slot["jpeg_nbytes"] = 0
        self._seq[i] += 1                         # -> even: consistent
        self._open = False
        self._head[0] = fid
        self._frame_id = fid
        return fid

    # --- the newest frame --------------------------------------------------

    def _newest(self):
        """`(slot, s0, frame_id)` of the newest published frame — `s0` sampled
        BEFORE the caller copies anything — or None if nothing is published.
        Retries while `head` moves under it."""
        while True:
            fid = int(self._head[0])
            if fid == 0:
                return None
            i = self._slot(fid)
            s0 = int(self._seq[i])
            if not s0 & 1 and int(self._rec[i]["frame_id"]) == fid:
                return i, s0, fid
            # Lapped between reading `head` and reading the slot — only a
            # reader preempted for N-1 frames gets here. The next `head` is
            # newer; try that.

    def _same(self, i: int, s0: int) -> bool:
        return int(self._seq[i]) == s0

    def peek(self) -> tuple[float, int]:
        """`(t_recv, frame_id)` of the newest frame without copying pixels;
        `(0.0, 0)` before the first.

        This is what a poll for "the first frame after t" costs: a few scalar
        loads, not a megabyte.
        """
        while True:
            got = self._newest()
            if got is None:
                return 0.0, 0
            i, s0, fid = got
            t = float(self._rec[i]["t_recv"])
            if self._same(i, s0):
                return t, fid

    def peek_counters(self) -> tuple[int, int] | None:
        """`(frame_id, source_seq)` of the newest frame, or None.

        Two scalars, no pixels and no JPEG — CHECK samples this at both ends of
        its window and the difference between how the two counters advanced is
        the number of frames the transport dropped. Copying a ~1.15 MB slot to
        read sixteen bytes would make the measurement cost more than the thing
        it measures.

        `source_seq` is -1 when the source has no counter.
        """
        while True:
            got = self._newest()
            if got is None:
                return None
            i, s0, fid = got
            seq = int(self._rec[i]["source_seq"])
            if self._same(i, s0):
                return fid, seq

    def read_into(self, out: np.ndarray) -> bool:
        """The newest whole slot into `out` (a `self.empty()`). False if there
        is none yet, or it was overwritten mid-copy (then: poll again)."""
        got = self._newest()
        if got is None:
            return False
        i, s0, _ = got
        out[()] = self._rec[i]
        return self._same(i, s0)

    def read_pixels_into(self, out: np.ndarray) -> tuple[float, int] | None:
        """Just the newest decoded pixels, plus the header. The crop fit's path.

        The copy-and-verify rule does not care which fields were copied between
        its two counter reads, so skipping `jpeg` is free and correct.
        """
        got = self._newest()
        if got is None:
            return None
        i, s0, fid = got
        t_recv = float(self._rec[i]["t_recv"])
        out[...] = self._rec[i]["pixels"]
        return (t_recv, fid) if self._same(i, s0) else None

    def read_jpeg(self) -> tuple[float, int, bytes] | None:
        """The newest compressed source and its header — never the pixels.

        None for a torn read or an uncaptured frame; the caller records the gap
        rather than retrying, because the next frame is 33 ms away and a
        reader that blocks the camera is the one thing this must never be.
        """
        got = self._newest()
        if got is None:
            return None
        return self._jpeg_at(*got)

    # --- history -------------------------------------------------

    def _jpeg_at(self, i: int, s0: int, fid: int):
        b = self._rec[i]
        n = int(b["jpeg_nbytes"])
        if n == 0:
            return None
        t_recv = float(b["t_recv"])
        payload = b["jpeg"][:n].tobytes()
        return (t_recv, fid, payload) if self._same(i, s0) else None

    def jpegs_since(self, last_id: int) -> tuple[list[tuple[float, int, bytes]], int]:
        """Every compressed frame published after `last_id`, oldest first, and
        the newest `frame_id` looked at. P9's sweep.

        With the triple buffer the recorder could only ever take the newest
        frame, so a sweep that ran late lost every frame in between. A ring
        keeps N of them, so a late sweep now catches up instead. A frame already
        lapped, torn, or not captured is simply absent from the list: the caller
        counts the gap in `frame_id`s, which is the record's drop count.
        """
        newest = int(self._head[0])
        out = []
        for fid in range(max(last_id + 1, newest - self.n + 1), newest + 1):
            i = self._slot(fid)
            s0 = int(self._seq[i])
            if s0 & 1 or int(self._rec[i]["frame_id"]) != fid:
                continue
            got = self._jpeg_at(i, s0, fid)
            if got is not None:
                out.append(got)
        return out, max(newest, last_id)

    def window(self, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
        """`(frame_ids, t_recv)` of every frame with `t0 <= t_recv <= t1`,
        ascending. Metadata only — the reader's `frame_ids_between`.

        The copy-and-verify rule applied to every slot at once: the two metadata
        columns are copied between two copies of the counters, and a slot is
        kept only if its counter was even and unchanged. ~N x 16 bytes; no
        pixels move.
        """
        s0 = self._seq.copy()
        fid = self._rec["frame_id"].copy()
        t = self._rec["t_recv"].copy()
        s1 = self._seq.copy()
        ok = (s0 == s1) & (s0 % 2 == 0) & (fid > 0) & (t >= t0) & (t <= t1)
        fid, t = fid[ok], t[ok]
        order = np.argsort(fid, kind="stable")
        return fid[order].astype(np.int64), t[order]

    def read_frame(self, frame_id: int, *, flip_channels: bool = False
                   ) -> tuple[np.ndarray, float] | None:
        """`(pixels, t_recv)` of exactly frame `frame_id`, a private copy — or
        None if that frame is no longer in the ring (lapped), is being written,
        or was overwritten during the copy.

        `flip_channels` reverses the channel axis in the same pass as the copy,
        so the wire's BGR becomes the policy's RGB for one memcpy, not two.
        """
        i = self._slot(frame_id)
        s0 = int(self._seq[i])
        b = self._rec[i]
        if s0 & 1 or int(b["frame_id"]) != int(frame_id):
            return None
        src = b["pixels"]
        px = np.ascontiguousarray(src[..., ::-1]) if flip_channels else src.copy()
        t_recv = float(b["t_recv"])
        return (px, t_recv) if self._same(i, s0) else None

    def empty(self) -> np.ndarray:
        return np.empty((), dtype=self.dtype)

    def empty_pixels(self) -> np.ndarray:
        return np.empty(self.dtype["pixels"].shape, dtype=np.uint8)


#: The triple buffer's accessor name, kept so code written against it still
#: reads. The class is the ring.
CameraRegion = CameraRing


@dataclass(frozen=True)
class ArenaRings:
    """Every region of a live arena, read-only: what `reader.arena` is.

    The same accessors the processes use (`open_all`), over a `ReadOnlyArena`,
    so every read method works and every write raises. The regions, their
    clocks and their read rules: `docs/infra_logic/arena_layout_and_reads.md`.
    """

    control: "Control"               # session, k_exec, newest_chunk_id, ...
    state: "StateRing"               # ~1 kHz robot state, P1
    chunks: "ChunkRing"              # the policy's chunks as P5 published them
    targets: "TargetRing"            # P6's per-tick joint targets
    intervene: "InterveneRing"       # PICO takeover rows, P11
    command: "CommandRing"           # P7's 500 Hz commands
    cameras: dict                    # view name -> CameraRing
    layout: object                   # the region table (`layout.describe()`)


def open_read_only(arena, cfg) -> ArenaRings:
    """`ArenaRings` over `arena`: every ring, none of them writable."""
    from g1d_deploy.runtime.shared_memory.layout import ReadOnlyArena
    ro = ReadOnlyArena(arena)
    r = open_all(ro, cfg)
    return ArenaRings(control=r["control"], state=r["state"], chunks=r["chunks"],
                      targets=r["targets"], intervene=r["intervene"],
                      command=r["command"], cameras=r["cameras"],
                      layout=ro.layout)


def open_all(arena, cfg):
    """Every accessor a process might want, built once at attach."""
    targets = TargetRing(arena)
    views = cfg.views
    cameras = {n: CameraRing(arena, n) for n in views}
    return {
        "control": Control(arena),
        "state": StateRing(arena),
        "chunks": ChunkRing(arena, cfg.H),
        # Three names, ONE region. `targets` is the writer's handle and the only
        # one anything calls `write` on; `body` and `hand` are the two readers',
        # differing only in which column they hand back as a payload. `body` IS
        # `targets` — both read `q` — and `hand` is a view over the same arrays.
        "targets": targets,
        "body": targets,
        "hand": targets.view("hand"),
        "intervene": InterveneRing(arena),
        "command": CommandRing(arena),
        # One ring per camera VIEW — the names a policy uses; a side_by_side
        # camera is two of them (`resolved.camera_views`).
        "cameras": cameras,
        # One per ROBOT-FILE camera: the view whose slots carry that wire
        # stream's JPEG — the whole wire frame, both eyes. What P9 records and
        # the dashboard streams, under the robot file's name, exactly as before
        # the split.
        "camera_sources": {v.source: cameras[v.name]
                           for v in views.values() if v.carries_jpeg},
    }
