"""P0 — launcher, homing and exit.

The process `elrobot_deploy run` becomes (`cli/main.py`). For the operator:
`docs/deploying_a_policy/running_and_live_dashboard.md`, "A launch, start to
finish"; the session states are `shared_memory/schema.py`'s.

**Writes:** the header, `session`, the target ring during HOME and EXITING,
`active_config` / `rollout_generation` at each START it accepts, and
`rt/lowcmd` for the final damp only. **Reads:** `k_exec`, the state ring, and —
the only process that does — `requested_config`.

P0 is not a loop over the rings. It brings the robot to the posture the run
starts from, starts everything, and brings it back afterwards. There is no
orchestrator inside the run: once the session word says ACTIVE, P0 does nothing
but wait for the operator.

Two things here look like they belong elsewhere and do not:

**Homing goes through the target ring**. P0 writes ordinary target slots
at ticks `k < 0` and P7 executes them with ordinary interpolation — P7 never
learns that homing exists. Homing then exercises the same deadline handling,
the same interpolation and the same slew backstop as every normal tick, instead
of being a path that only runs when the arm moves unattended. Tick numbering
never restarts *within a rollout*, so `k_exec` stays monotonic and the startup
state (homing ends on tick 0) falls out with no special case.

Between rollouts it does restart, and only because `reset` stops P7 to do
it: the cursor is P7's word, so the one safe way to rewind it is a window in
which no P7 exists. What comes back up is the boot condition itself rather than
an imitation of it — which is why nothing above needed a second reading.

**The damp is P0's, not P7's**, and so is the exit motion — by the same
mechanism as homing. That is what lets P7 stay a pure interpolator with no
notion of sessions or safety states.
"""

from __future__ import annotations

import os
import signal
import sys
import time
import traceback

import numpy as np

from elrobot_deploy.runtime import cpu_pinning as affinity
from elrobot_deploy.runtime import timing as _clock
from elrobot_deploy.runtime.shared_memory import (Arena, ShmTooSmall, build, check_shm,
                                              open_all, schema)
from elrobot_deploy.config_loading.errors import ConfigError
from elrobot_deploy.config_loading.validate import (check_async_timing,
                                              check_intervene_ik, validate)
from elrobot_deploy.config_loading.derive import derive
from elrobot_deploy.config_loading.rollout_settings import RolloutSettings
from elrobot_deploy.config_loading.rollout_settings import check as check_settings
from elrobot_deploy.runtime.launcher.children import _Children, _spawn_process
from elrobot_deploy.runtime.launcher.ramps import (cosine_ramp, cosine_samples, home_ticks,
                                               write_ramp)
from elrobot_deploy.runtime.launcher.startup_checks import (_POLL_S, _SHUTDOWN_JOIN_S,
                                                        StartupChecks)


class _Refused(Exception):
    """A START that P0 will not accept. The session stays at READY."""


class _ExitNow(KeyboardInterrupt):
    """The operator's first Ctrl-C: run the exit sequence (`Launcher.exit`)."""


class _DampNow(KeyboardInterrupt):
    """The operator's second Ctrl-C: skip to the damp.

    An immediate damp drops the arm into gravity, so it is an explicit second
    decision and never automatic.
    """


# --- the launcher -----------------------------------------------------------


class Launcher(StartupChecks):
    """The session sequence (INIT … STOPPED), and nothing else.

    `spawn`, `event` and `lowcmd` are injectable so the sequence can be
    exercised over a real arena without real processes or a real robot.
    """

    def __init__(self, cfg, hw, *, arena_name: str | None = None, clock=_clock, spawn=None,
                 event=None, lowcmd=None, log=print, record_dir=None,
                 pico: bool = False,
                 provenance: dict | None = None, shm_free=None, lens_frames=None):
        # Before anything is assigned: an intervention on any IK but `arm_qp`,
        # is refused however the launcher was reached
        # (`validate.check_intervene_ik`).
        check_intervene_ik(cfg, intervene=pico)
        self.cfg = cfg
        self.hw = hw
        #: `--intervene`. OFF by default, and off means off: without
        #: it P11 is never spawned, so `control.intervene` stays 0 for the life
        #: of the run, `is_intervening` is never true, and every other process
        #: takes exactly the path it took before any of this existed.
        self.pico = pico
        #: Where P9 writes this rollout, or None to run without a recorder.
        #: Nothing in the control path depends on it either way.
        self.record_dir = record_dir
        #: What the launch was resolved from — robot and machine files, deploy
        #: file, policy entry, checkpoint, code revisions (`__main__._provenance`).
        #: Handed to P9 for every manifest, with INIT's lens decisions added.
        self.provenance = dict(provenance or {})
        #: INIT's per-camera lens decisions (`fisheye_circle_fit.CropDecision`).
        self.crops: list = []
        #: `camera -> [luma frames]` for the lens fit; None reads the camera.
        #: Injectable so INIT can be tested without a camera.
        self._lens_frames = lens_frames
        #: Whether INIT already checked the subnet and started the board's
        #: image server (a fisheye rig needs both before its lens fit).
        self._board_ready = False
        self.name = arena_name or f"elr-{os.getpid()}"
        self.clock = clock
        self.log = log
        #: `() -> (free, total)` bytes of /dev/shm, for `init`'s refusal; None
        #: reads the real tmpfs. Injectable so the refusal can be tested.
        self._shm_free = shm_free
        self.children = _Children(spawn or _spawn_process, event)
        self._lowcmd = lowcmd
        self.arena = None
        self.rings = None
        self.derived = None
        #: The config the current (or last) rollout runs with: `cfg` with the
        #: `active_config` P0 accepted at its START applied. The launch
        #: config until the first START.
        self.rollout_cfg = cfg
        self._wire = None
        self._damped = False
        self._skip_ramp = False
        self._interrupts = 0
        self._failure: BaseException | None = None
        #: Wall clock, for `_robot_diagnosis`: a Cyclone trace older than this
        #: is a previous run's and says nothing about this one.
        self._t_launch = time.time()

    # --- helpers -----------------------------------------------------------

    def _set(self, state: int) -> None:
        self.rings["control"].session = state
        self.log(f"[P0] {schema.SESSION_NAMES[state]}")

    def _measured_body(self) -> np.ndarray:
        """The measured `(17,)` posture: 14 arm joints, then 3 waist."""
        latest = self.rings["state"].latest()
        if latest is None:
            raise ConfigError("no state samples: P1 is not producing")
        idx = list(self.hw.joints_arm) + list(self.hw.joints_waist)
        return np.asarray(latest["q"], dtype=float)[idx]

    def _measured_hand(self) -> np.ndarray:
        """The measured `(2, 6)` end-effector state — where a hand ramp starts.

        The same source P8 seeds `cmd_last` from, so the first hand target
        of a ramp continues the aperture P8 is already publishing rather than
        stepping away from it.
        """
        latest = self.rings["state"].latest()
        if latest is None:
            raise ConfigError("no state samples: P1 is not producing")
        return np.asarray(latest["grip_meas"], dtype=float).reshape(
            schema.N_HANDS, schema.HAND_DIM)

    def _wait(self, predicate) -> None:
        """Poll until true. No timeout anywhere in this file, by decision."""
        while not predicate():
            self.clock.sleep_until(self.clock.now() + _POLL_S)

    def _take_request(self) -> int:
        """The pending request, consumed. `REQ_NONE` when there is none.

        Reading and clearing are one operation because P0 is the request's
        CONSUMER, not a second observer of it. A request left in the word would
        be taken again on the next pass of whichever loop is polling — one press
        of Reset would reset forever.

        This is the store that makes `request` the system's only two-writer word.
        `Control.request` carries the argument for why that is affordable here
        and nowhere else; the short version is that the losing store costs a
        button press, not a state machine.
        """
        control = self.rings["control"]
        request = control.request
        if request != schema.REQ_NONE:
            control.request = schema.REQ_NONE
            self.log(f"[P0] request "
                     f"{schema.REQUEST_NAMES.get(request, request)}")
        return request

    # --- the sequence ------------------------------------------

    def init(self) -> None:
        """Validate, fit the fisheye lenses, derive, print, create the arena,
        write the layout hash.

        The lens fit comes first because a fisheye camera's ring is the square
        around its lens: the arena cannot be sized before it is known. It
        needs the robot on the subnet and the board's image server up, so a
        fisheye rig does those two CHECK steps here.
        """
        validate(self.cfg, self.hw)
        if any(getattr(s, "kind", "normal") == "fisheye180"
               for s in self.cfg.cameras.values()):
            if self._lens_frames is None:
                self._verify_robot_subnet()
                self._ensure_board_imager()
                self._board_ready = True
            self._fit_lenses()
        self.derived = derive(self.cfg, self.hw)
        layout = build(self.cfg, self.derived)
        self.log(f"[P0] derived: {self.derived.describe()}")
        self.log(layout.describe())
        # BEFORE the create, because the create cannot fail: a tmpfs file is
        # sparse, so an arena too big for /dev/shm maps fine and SIGBUSes the
        # first process to touch a page past the limit (`layout.check_shm`).
        # Camera history is what makes this arena large.
        try:
            self.log(f"[P0] /dev/shm: {check_shm(layout, free=self._shm_free)}")
        except ShmTooSmall as exc:
            raise ConfigError(str(exc)) from None
        self.arena = Arena.create(self.name, layout)
        self.rings = open_all(self.arena, self.cfg)
        self._seed_settings()
        self._set(schema.INIT)

    def _seed_settings(self) -> None:
        """The launch settings in both blocks, generation 0: what the
        dashboard shows before anybody asks for anything, and what a START
        with no request accepts."""
        control = self.rings["control"]
        launch = RolloutSettings.of(self.cfg).to_record()
        control.request_config(launch)
        control.activate(launch, self.cfg.prompt)

    def check(self) -> None:
        """Start the producers, confirm they produce, verify the rates.

        P7 and P8 are started last and run from here onward: they connect DDS
        and begin publishing a hold at the *measured* posture, which is what
        makes homing an ordinary ramp rather than a jump.

        Confirming the two servo loops is part of what "confirm they produce"
        means, exactly as it is for P1 and the cameras: a ramp written into a
        ring that no interpolator is reading yet is executed as a jump; see
        `_await_publishing`.
        """
        self._set(schema.CHECK)
        self._report_isolation()
        if not self._board_ready:
            self._verify_robot_subnet()
            self._ensure_board_imager()
        args = (self.name, self.cfg, self.hw)
        self.children.start("p1_state", "elrobot_deploy.runtime.processes.p1_state.run",
                            args)
        for cam in self.cfg.cameras:
            self.children.start(f"cam_{cam}",
                                "elrobot_deploy.runtime.processes.p2_camera.run", args,
                                camera=cam)
        if self.pico:
            # The service BEFORE the process that talks to it: P11's first act
            # is `xrt.init()`, which succeeds against a dead service and then
            # returns zeros forever.
            self._ensure_pico_service()
            # With P1 and the cameras, not later: it is a PRODUCER, and CHECK's
            # whole job is to prove the producers produce before the robot moves.
            self.children.start("p11_pico",
                                "elrobot_deploy.runtime.processes.p11_pico.run", args)

        state = self.rings["state"]
        self._await_producer(
            "p1_state", lambda: state.latest() is not None,
            "its first robot state sample",
            lambda: f"Nothing on {self.hw.topic_lowstate} (domain="
                    f"{self.hw.dds_domain}). {self._robot_diagnosis().explain()}")
        for cam, region in self.rings["cameras"].items():
            self._await_producer(
                f"cam_{cam}", lambda r=region: r.peek()[1] > 0,
                "its first frame", self._camera_hint(cam))

        self._verify_rates()
        self._verify_hand_state()
        if self.pico:
            self._verify_pico()

        self.children.start("p7_body", "elrobot_deploy.runtime.processes.p7_body.run", args,
                            gated=True)
        self.children.start("p8_hand", "elrobot_deploy.runtime.processes.p8_hand.run", args,
                            gated=True)

        # P5 starts HERE, not at WARMUP, and this is the earliest point it may.
        # Loading a checkpoint and compiling an XLA executable takes ~40 s and
        # needs nothing this system produces — no state, no camera, no target —
        # so it is pure dead time if it waits for homing to finish. It is
        # spawned after `_verify_rates` rather than in `init()` for the reason
        # `validate` refuses a missing checkpoint before the arena is mapped: a
        # robot that is not producing lowstate should cost a second to find
        # out, not an XLA warmup. The ~1.5 s that costs is bought back many
        # times over by overlapping the compile with the homing ramp.
        #
        # It is safe here because P5 warms on `session < ACTIVE`, so which
        # pre-run phase P0 is in is not something it can be wrong about, and
        # because its cores are its own — a compile that saturates
        # [8, 12, 14] cannot reach P7 on 2 or P1 on 10.
        #
        # Under the POLICY's interpreter (`[policy] python`; the runtime's own
        # when unset), and always through `processes/policy_host.py`: one path,
        # so the default launch exercises the one a policy venv takes
        # (`docs/deploying_a_policy/policy_contract.md`, "The adapter's own
        # Python").
        self.children.start("p5_policy", "elrobot_deploy.runtime.processes.p5_policy.run",
                            (self.name, self.cfg, self.hw),
                            interpreter=self.cfg.policy_python or sys.executable)

        # P9 last, and started here rather than at ACTIVE so its process spawn,
        # import and arena attach overlap this same dead time — not so it
        # records CHECK/HOME/WARMUP/READY, which it does not. `run()`
        # itself waits out everything before ACTIVE and writes nothing until
        # then: that phase is setup, not the rollout, and recording it would be
        # roughly as much camera JPEG as the run itself for a window nobody
        # reads back. It is spawned rather than left to be run by hand because a
        # recorder you have to remember is a recorder that is missing from the
        # rollout you most wanted.
        if self.record_dir is not None:
            self.children.start("p9_record",
                                "elrobot_deploy.runtime.processes.p9_record.run",
                                (self.name, self.cfg, self.hw),
                                out_dir=str(self.record_dir),
                                provenance=dict(
                                    self.provenance,
                                    crops={d.camera: d.as_dict()
                                           for d in self.crops}))
            self.log(f"[P0] recording to {self.record_dir}")

        # P10 is GATED, and that is the whole reason it is not simply started
        # and forgotten. START arrives through it, so a P0 that reached
        # READY before the socket was listening would be waiting for a press the
        # operator has no way to make — the terminal offers no second path.
        self.children.start("p10_dashboard",
                            "elrobot_deploy.runtime.processes.p10_dashboard.run",
                            (self.name, self.cfg, self.hw),
                            gated=True,
                            step=self._checkpoint_step(),
                            intervene=self.pico)

        # LAST, and after every other spawn, for the same reason P5 is started
        # before the ramp rather than after it: P7's ~2.8 s of DDS and pinocchio
        # is dead time that should overlap something. Waiting here rather than
        # at the top of `home()` puts P5's policy load, P9's arena attach
        # and P7's connect on the same wall clock, so the gate costs a run
        # nothing it was not already paying.
        self._await_publishing("p7_body", "p8_hand", "p10_dashboard")

    def _checkpoint_step(self) -> str | None:
        """Which step is served, for the dashboard's banner: the deploy file's
        `[checkpoint] step` as resolved (CLI `--step` included), else the
        checkpoint directory's own name — a step directory is named by its
        step. None for a policy that loads nothing."""
        tree = (self.provenance.get("deploy") or {}).get("resolved_tree") or {}
        step = (tree.get("checkpoint") or {}).get("step")
        if step is not None:
            return str(step)
        return os.path.basename(os.path.normpath(self.cfg.checkpoint)) \
            if self.cfg.checkpoint else None

    def home(self) -> None:
        """The ramp from the measured posture to `home_q`, ending at `k = 0`.

        Both rings ramp. The arm's velocity limit sets the duration and the hand
        rides the same tick grid — see `cosine_samples` for why the gripper
        cannot derive one of its own, and why it does not need to.

        Every measurement here is taken AFTER `check` has confirmed P7 and P8
        are publishing, so `q_from` is the posture P7 is actually holding rather
        than one sampled ~2.8 s before P7 sampled its own.
        """
        self._set(schema.HOME)
        q_from = self._measured_body()
        q_to = np.concatenate([np.asarray(self.cfg.home_q, dtype=float),
                               np.asarray(self.cfg.home_waist, dtype=float)])
        qs = cosine_ramp(q_from, q_to, dt=self.cfg.dt,
                         qdot_max=self.hw.qdot_max_body,
                         min_s=self.cfg.home_min_s,
                         margin=self.cfg.home_velocity_margin)
        hands = cosine_samples(self._measured_hand(),
                               np.asarray(self.cfg.home_hand, dtype=float),
                               len(qs))
        ks = home_ticks(len(qs))
        # Logged before the write, not after: `write_ramp` paces itself against
        # the ring's capacity and only returns once the last tick is in, which
        # is a second or two into the motion it is announcing.
        self.log(f"[P0] homing over {len(qs)} ticks "
                 f"({len(qs) * self.cfg.dt:.2f} s), k = {ks[0]} … {ks[-1]}")
        written = write_ramp(
            self.rings, ks, qs, hands,
            t_first=self.clock.now() + self.cfg.dt, dt=self.cfg.dt,
            lead_ticks=self.derived.n_target - 2, clock=self.clock)
        self.clock.sleep_until(written[-1][1])
        self._verify_home(q_to)

    def warmup(self) -> None:
        """Wait until P5 is warm. It was spawned in `check()`.

        Without it a run pays the policy's whole load and compile inside its
        *first real inference* (measured once at 29.9 s) whenever the operator
        reaches READY before P5 has finished loading. `control.warm` is the
        word P0 observes P5 through: one atomic slot
        in the arena, written by P5 and by nobody else, exactly as
        `newest_chunk_id` is. Waiting on it is the entire mechanism.

        **P0 reads the word and P0 makes the transition.** P5 could have stored
        the session state itself and saved this wait — and it would have broken
        the one-writer rule, because `session` would then have two writers and a lost update
        against EXITING would walk the state backwards. What P5 owns is the
        judgement (`p5_policy.stabilised`); what P0 owns is every store to
        `session`. P5 prints why it is ready; this method never re-derives it.

        The wait ends on one of two facts, never on a clock: P5 is warm, or P5
        is gone. A timeout would have to encode how long a policy takes to
        load, which is a property of the policy and the machine, and
        guessing it wrong either fails a healthy run or lets a dead one through.
        """
        self._set(schema.WARMUP)
        control = self.rings["control"]
        self.log("[P0] waiting for P5 to load the policy and warm up")
        self._wait(lambda: control.warm or not self.children.alive("p5_policy"))
        if not control.warm:
            # P5 died before declaring itself ready — a bad checkpoint, a
            # missing GPU, an OOM. Its traceback is already on this terminal;
            # what is worth saying here is that there will be no run, because
            # the alternative is a READY prompt for a policy that does not
            # exist. Raising takes the failure path: no exit ramp, damp,
            # STOPPED.
            raise ConfigError(
                "p5_policy exited before it warmed up — see its traceback "
                "above. There is no policy to run.")

    def ready(self) -> None:
        """Wait for the operator's START. Returns when it arrives.

        A poll, not an `input()`: a process asleep inside `read(2)` on stdin
        polls nothing, so a START written into the arena by P10 would never be
        seen. A blocking read is not a state a controllable process may sit in.

        The terminal has Ctrl-C and no Enter. The two interfaces are not the
        same mechanism: Ctrl-C is a signal and
        preempts this loop wherever it is, while START is a word this loop has to
        come around and look at.

        A RESET arriving here is not an error and not a no-op worth reporting —
        it asks for the state this method is already in. `_take_request` clears
        it and the loop goes round again.
        """
        self._set(schema.READY)
        self.log("[P0] READY — press Start on the dashboard, Ctrl-C to exit.")
        while True:
            request = self._take_request()
            if request == schema.REQ_START and self._accept_start():
                return
            if request == schema.REQ_EXIT:
                raise _ExitNow
            self.clock.sleep_until(self.clock.now() + _POLL_S)

    # --- START: the settings and prompt of the next rollout -------

    def _accept_start(self) -> bool:
        """Latch the operator's request as the next rollout's. False = refused.

        **The one moment `requested_config` is read**, and the one moment
        `active_config` is written. In order:

        1. the request is validated against this launch (`rollout_settings.check`:
           `P <= H`, the servo ranges, the IK mode, `--intervene`'s limits);
        2. the prompt: the newest prompt request must carry P5's ACCEPTED
           verdict — only the policy owns a tokenizer, so P5 judged it when it
           was submitted and P0 only reads the answer (waiting for it if P5 has
           not got round to it yet). No request at all keeps the active prompt;
        3. `active_config` + `active_prompt` are written, THEN
           `rollout_generation` is bumped — the payload before its counter;
        4. P0 waits for P5 to report the generation applied (`set_prompt`,
           `on_settings`, one dummy inference), so a recompile is paid here at
           READY and not by the rollout's first real inference.

        A refusal leaves the session at READY and says why twice: on this
        terminal and in the control region's `refusal`, which the dashboard
        shows. A refused request is not cleared — the operator fixes it and
        presses Start again.
        """
        control = self.rings["control"]
        try:
            settings = RolloutSettings.from_record(control.requested_config())
            rollout = check_settings(settings, self.cfg, self.hw,
                                     intervene=self.pico)
            prompt = self._accepted_prompt()
            control.activate(settings.to_record(), prompt)
            gen = control.rollout_generation + 1
            control.rollout_generation = gen
            self._await_p5(lambda: (control.settings_applied
                                    & ~schema.APPLIED_FAILED) == gen,
                           "take the new rollout settings")
            if control.settings_applied & schema.APPLIED_FAILED:
                raise _Refused(
                    f"the policy refused rollout {gen}'s settings or prompt "
                    f"(its traceback is on this terminal, from P5)")
            # After P5 took them: a new `sample_steps` changes the latency,
            # and P5 re-measures it before it reports the generation applied.
            check_async_timing(rollout, control.latency_s)
        except (ConfigError, _Refused) as exc:
            control.refuse(str(exc))
            self.log(f"[P0] START refused: {exc}")
            return False
        self.rollout_cfg = rollout
        s = settings
        self.log(f"[P0] rollout {gen}: {'sync' if s.sync else 'async'}, "
                 f"P={s.replan_period}, {s.control_hz:g} Hz, ik={rollout.ik}, "
                 f"sample_steps={s.sample_steps}, servo {s.servo.describe()}, "
                 f"prompt {prompt!r}")
        return True

    def _accepted_prompt(self) -> str:
        """The prompt the next rollout runs with, or `_Refused`."""
        control = self.rings["control"]
        seq, text = control.prompt_request()
        if seq == 0:
            return control.active()[1]
        self._await_p5(lambda: control.verdict()[0] == seq, "judge the prompt")
        _, verdict, reason = control.verdict()
        if verdict != schema.PROMPT_ACCEPTED:
            raise _Refused(f"the policy refused the prompt {text!r}: "
                           f"{reason or 'no reason given'}. Submit another.")
        return text

    def _await_p5(self, predicate, what: str) -> None:
        """Wait on P5 at READY. Ends on the answer, P5's death, or EXIT.

        No timeout, as everywhere in this file: P5 answers between two dummy
        inferences, and the only way it does not is by being gone. An EXIT from
        the dashboard is still honoured while waiting — the one request that
        must never be swallowed — and anything else pressed meanwhile is
        consumed and ignored, as it would be at any other moment of READY.
        """
        while not predicate():
            if not self.children.alive("p5_policy"):
                raise ConfigError(f"p5_policy exited instead of answering "
                                  f"({what}) — see its traceback above.")
            if self._take_request() == schema.REQ_EXIT:
                raise _ExitNow
            self.clock.sleep_until(self.clock.now() + _POLL_S)

    def active(self) -> None:
        """Spawn P6 and hand the run over. **Returns** on RESET.

        The first inference is triggered by this transition, not by P5's loop:
        it is the `k_next_inference = 0` inference, producing `chunk_id = 1`
        with `k_first_action = 1`. P5 performs it on seeing ACTIVE,
        because P5 is the only process that holds the policy.

        It returns to `run`'s rollout loop on RESET; EXIT leaves by
        raising, so the exit sequence has exactly one implementation and the
        dashboard's Exit and the first Ctrl-C are the same path.
        """
        self._set(schema.ACTIVE)
        # The LAUNCH config, deliberately: P6 latches `active_config` from the
        # arena itself (`rollout_settings.latched`), like every other process, so there
        # is one source for the rollout's settings rather than two to agree.
        self.children.start("p6_target", "elrobot_deploy.runtime.processes.p6_target.run",
                            (self.name, self.cfg, self.hw))
        self.log("[P0] running — Reset on the dashboard to end this rollout, "
                 "Ctrl-C to exit, twice to damp immediately.")
        while True:
            request = self._take_request()
            if request == schema.REQ_RESET:
                return
            if request == schema.REQ_EXIT:
                raise _ExitNow
            self.clock.sleep_until(self.clock.now() + _POLL_S)

    # --- reset -------------------------------------------------

    def reset(self) -> None:
        """End the rollout and come back to READY still warm.

        The whole reason this method exists. Without it a second rollout costs
        a relaunch, and a relaunch costs the policy's load and compile
        that `warmup` exists to pay exactly once — tens of seconds, per rollout,
        for a robot that is already standing at home. Everything below is chosen
        to avoid touching P5.

        **Restarted:** P6, P7, P8. **Kept:** P5 — the entire point — along with
        P1 and the cameras, because P5 goes on warming against live observations
        and `home` needs a measured posture the instant it is called; P9; and
        P10, which must outlive every rollout because START now arrives through
        it.

        P9 is kept and NOT restarted even though each rollout gets its own
        record. It closes one and opens the next on the session word itself,
        which is the only arrangement in which every record gets a
        manifest: `stop` is SIGTERM and runs no `finally`, so a recorder killed
        between rollouts would leave the one it was writing unreadable.

        **Restarting the interpolators is what keeps `k_exec` honest.** It is
        P7's word and P7 sets it to the tick it just took, so it cannot be
        rewound underneath a living P7 — but with P7 stopped there is no writer
        at all, P0 may store the zero, and the P7 spawned next has never seen a
        tick. That reproduces the boot condition rather than simulating it:
        homing writes `k < 0` again, `write_ramp`'s stale-cursor gate reads the
        same zero it reads at startup, and P5's `k_start_read = 2` /
        `k_next_inference = P` are literally correct instead of needing to be
        rebased onto a cursor five figures deep. The alternative — teaching P7
        that tick numbering can restart — puts a state machine inside the one
        process whose value is that it is a pure interpolator.

        **The arm is not unheld while P7 is gone.** The firmware runs its PD law
        on the last command it received, forever (`_damp`), so the posture the
        policy left is simply frozen for the ~2.8 s of DDS and pinocchio bringup,
        and the new P7 seeds its hold from the MEASURED posture. `home`
        then ramps from wherever that is, exactly as it does at startup — which
        is why nothing here needs to know where the rollout ended.

        Not polled anywhere in here: an EXIT that arrives mid-reset is seen when
        `ready` comes round, a few seconds later. Ctrl-C is a signal and does not
        wait, which is the reason it stays.
        """
        self._set(schema.HOME)
        # The word first: P5 and P6 both leave ACTIVE on it. Two ticks is P6's
        # whole iteration budget — the same wait `exit` makes, for
        # the same reason. Terminating P6 mid-write leaves a seqlock odd forever
        # and hangs whoever reads that slot next.
        self.clock.sleep_until(self.clock.now() + 2 * self.rollout_cfg.dt)
        self.children.stop("p6_target", "p7_body", "p8_hand")

        # Only here, with every writer of these regions stopped, is any of this
        # safe. `newest_chunk_id` is NOT zeroed from P0: it is P5's word and P5
        # is still alive, so P5 zeroes it as it enters its next rollout.
        self.rings["control"].k_exec = 0
        # One region, so one clear.
        self.rings["targets"].clear()

        args = (self.name, self.cfg, self.hw)
        self.children.start("p7_body", "elrobot_deploy.runtime.processes.p7_body.run", args,
                            gated=True)
        self.children.start("p8_hand", "elrobot_deploy.runtime.processes.p8_hand.run", args,
                            gated=True)
        # A fresh `ready` latch per spawn (`_Children.start`), so this is the
        # same gate `check` makes and not a stale one that is already set.
        self._await_publishing("p7_body", "p8_hand")
        self.home()

    # --- exit --------------------------------------------------

    def exit(self) -> None:
        """`EXITING` → exit ramp → stop P7/P8 → damp → `STOPPED`. In that order,
        always.

        `STOPPED` is published **before** the surviving children are terminated,
        and that ordering is load bearing rather than tidy. Every remaining
        child — P1, the cameras, P9 — loops on `session < STOPPED`, so the word
        is their exit signal and the only path on which their `finally` blocks
        run. Terminating them first is SIGTERM, which runs none of it: P9 would
        lose every stream still inside its 1 MB write buffer plus
        `manifest.json` itself, so a rollout's `chunks.bin` — the only record of
        what each inference cost — would arrive empty and unreadable.
        """
        if self.rings is None:
            return
        self._set(schema.EXITING)
        try:
            # 1. P5 and P6 stop, so nothing else writes the rings. They stop
            #    themselves on the session word — terminating P6 mid-write
            #    would leave a seqlock odd forever and hang whoever read that
            #    slot next. Two ticks is P6's whole iteration budget,
            #    so it is enough for any solve in flight to land.
            self.clock.sleep_until(self.clock.now() + 2 * self.rollout_cfg.dt)
            self.children.stop("p5_policy", "p6_target")
            if not self._skip_ramp:
                self._exit_ramp()                                    # 2
        except _DampNow:
            self.log("[P0] second interrupt — skipping to the damp")
        except Exception:                                 # noqa: BLE001
            # Whatever the ramp could not do, steps 3 and 4 must still happen:
            # the firmware is holding a setpoint and only the damp ends that.
            self.log(traceback.format_exc())
        self.children.stop("p7_body", "p8_hand")                     # 3
        self._damp()                                                 # 4

        # 5. The word, then the wait. `STOPPED` after the damp so the recorder
        #    captures it, and before the terminate so everyone still running
        #    closes their own files.
        self._set(schema.STOPPED)
        late = self.children.wait(*list(self.children.handles),
                                  timeout=_SHUTDOWN_JOIN_S)
        if late:
            self.log(f"[P0] {', '.join(sorted(late))} did not exit within "
                     f"{_SHUTDOWN_JOIN_S:.0f}s of STOPPED — terminating. Any "
                     f"record they had buffered is lost.")
        self.children.stop_all()

    def _exit_ramp(self) -> list[tuple[int, float]]:
        """A ramp to `exit_q`, forward from `k_exec + 2`, by homing's rule.

        It starts from the current *commanded* posture — the target P7 is
        interpolating toward — and not from the measurement, for the same
        reason P6 seeds from the ring: starting from the measurement
        would fold tracking error into the first exit target.

        `deadline(k_exec + 2) = now + 2·dt` is the same relation P6 re-stamps
        its epoch with: P7 is already committed to the segment ending at
        `deadline(k_exec)`, so the first tick P0 may own is two out.
        """
        k_exec = self.rings["control"].k_exec
        k_first = k_exec + 2
        # ONE snapshot for both ramps. The body and the hand seed are two
        # columns of the same row, so reading the row once is what guarantees
        # the arm and the jaw ramp away from the SAME tick — two scans could
        # straddle a P6 write and seed them from neighbouring ticks.
        seed = self.rings["targets"].latest_below_full(k_first)
        q_from = (seed["q"].copy() if seed is not None
                  else self._measured_body())
        q_to = np.concatenate([np.asarray(self.cfg.exit_q, dtype=float),
                               np.asarray(self.cfg.exit_waist, dtype=float)])
        qs = cosine_ramp(q_from, q_to, dt=self.cfg.dt,
                         qdot_max=self.hw.qdot_max_body,
                         min_s=self.cfg.home_min_s,
                         margin=self.cfg.home_velocity_margin)
        # The end effector goes where the config says, exactly like the arm and
        # the waist, and it RAMPS there over the same ticks. Holding it at
        # `exit_hand` for the whole ramp would make the first exit tick a
        # step, and P8 has no slew clamp to absorb one.
        #
        # Seeded from the last COMMANDED aperture rather than the measured one,
        # for the reason the body is: a run that ends mid-grasp is
        # holding a jaw against an object, so the measurement lags the command
        # by whatever the object is worth, and starting there would step.
        #
        # Note that `exit_hand` is a real decision, not a default: setting it
        # open drops whatever is grasped at the end of every run, and setting it
        # closed ends the run still holding it.
        h_from = (seed["hand"].copy() if seed is not None
                  else self._measured_hand())
        hands = cosine_samples(h_from, np.asarray(self.cfg.exit_hand,
                                                  dtype=float), len(qs))
        self.log(f"[P0] exit ramp over {len(qs)} ticks, k = {k_first} … "
                 f"{k_first + len(qs) - 1}")
        written = write_ramp(
            self.rings, np.arange(k_first, k_first + len(qs)), qs, hands,
            t_first=self.clock.now() + 2 * self.cfg.dt, dt=self.cfg.dt,
            lead_ticks=self.derived.n_target - 2, clock=self.clock,
            cursor_live=True)
        # Step 3's precondition is "k_exec reaches the last exit tick and P7
        # stalls". That instant is the last deadline — a deadline is when the
        # tick is finished executing — so this waits on the clock. Waiting
        # on the cursor would hang forever if P7 skipped the last tick, and
        # there is no timeout to fall back on.
        self.clock.sleep_until(written[-1][1])
        return written

    def _damp(self) -> None:
        """The only real stop, and the last thing that ever touches `rt/lowcmd`.

        The firmware runs `tau = kp·(q*−q) + kd·(dq*−dq) + tau_ff` on the last
        command it received, forever: "stop publishing" is not a stop. `kp = 0,
        kd = 2` on every joint is, and nothing may command the arm afterwards
        — which is why the latch is set before the publish rather than
        after it.
        """
        if self._damped:
            return
        self._damped = True
        try:
            legs = self.rings["state"].latest()
            leg_q = (np.asarray(legs["q"], dtype=float)[list(self.hw.joints_leg)]
                     if legs is not None else np.zeros(len(self.hw.joints_leg)))
            factory = self._lowcmd
            if factory is None:
                from elrobot_deploy.runtime.drivers.arm_command import DdsLowCmd
                factory = DdsLowCmd
            self._wire = factory(self.hw)
            self._wire.connect(leg_q)
            self._wire.damp()
            self.log("[P0] damp latched")
        except Exception:
            self.log(traceback.format_exc())

    # --- lifecycle ---------------------------------------------------------

    def _install_sigint(self) -> None:
        def handler(signum, frame):
            self._interrupts += 1
            if self._interrupts == 1:
                raise _ExitNow
            # The second decision is final: further interrupts must not disturb
            # the damp itself.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            raise _DampNow

        signal.signal(signal.SIGINT, handler)

    def _teardown(self) -> None:
        if self._wire is not None:
            self._wire.close()
        if self.arena is not None:
            self.arena.close()
            self.arena.unlink()

    def run(self) -> None:
        affinity.pin(self.hw, "p0_launcher")
        self.init()
        self._install_sigint()
        try:
            self.check()
            self.home()
            self.warmup()
            # One iteration per rollout. CHECK, the first HOME and
            # WARMUP are above it because they are the things a session pays for
            # once; everything inside is what a rollout costs. The loop has no
            # exit condition of its own — EXIT raises, from a request or from
            # Ctrl-C, and that is the only way a session ends.
            while True:
                self.ready()
                self.active()
                self.reset()
        except _ExitNow:
            pass
        except _DampNow:
            self._skip_ramp = True
        except BaseException as exc:                      # noqa: BLE001
            # Whatever went wrong, the arm is holding a setpoint and only the
            # damp ends that. Do not ramp on a failure nobody understands.
            self._skip_ramp = True
            self._failure = exc
            self.log(traceback.format_exc())
        finally:
            try:
                self.exit()
            finally:
                self._teardown()
        if self._failure is not None:
            raise self._failure


def run(cfg, hw, **kwargs) -> None:
    Launcher(cfg, hw, **kwargs).run()


__all__ = ["Launcher", "run"]
