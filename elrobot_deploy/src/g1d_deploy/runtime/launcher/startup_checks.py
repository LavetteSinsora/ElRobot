"""P0's CHECK phase: every verifier the launcher runs before homing — the
processes are up and producing, the streams run at their rates, the network,
the board image server, the PICO headset, the hands, the homed pose. Mixed
into `Launcher` (`state_machine.py`), whose `check()` calls them in order."""

from __future__ import annotations

import os

import numpy as np

from g1d_deploy.runtime import cpu_pinning as affinity
from g1d_deploy.runtime.teleop import pico_service
from g1d_deploy.runtime.shared_memory import (schema)
from g1d_deploy.config_loading.resolved import replace
from g1d_deploy.config_loading.errors import ConfigError
from g1d_deploy.robot.kinematics import WAIST_AXIS_NAMES as _WAIST_AXIS_NAMES

# How long CHECK watches the streams before believing their rates. One second
# of a 1042 Hz stream is a sub-percent measurement; anything shorter and the
# tolerance below would have to absorb the sampling error instead of the
# machine's actual jitter.
_CHECK_WINDOW_S = 1.0
_RATE_TOLERANCE = 0.10

# How long CHECK waits for the FIRST end-effector sample before refusing. Longer
# than the lowstate window because this stream has no rate to measure — one
# sample is the whole test — and because the gripper's own service may still be
# coming up when P1 subscribes. Two seconds is many hundreds of samples on a
# healthy topic and still a short wait on a dead one.
_HAND_STATE_WINDOW_S = 2.0

# How long the children get to exit on the session word before they are
# terminated. Sized for the only one that has real work to do at STOPPED: P9
# flushes up to `FLUSH_BYTES` per stream and writes its manifest, which is a few
# megabytes of sequential I/O. Generous on purpose — it is paid once per run,
# and only in full when something is genuinely wedged.
_SHUTDOWN_JOIN_S = 5.0

# The poll used while waiting for the operator or for a stream to appear.
# Nothing is scheduled against it.
_POLL_S = 1e-3

# How often the wait for a gated child re-checks that it is still alive. Coarser
# than `_POLL_S` because it is not a poll for the thing itself: `Event.wait`
# blocks in the kernel and returns the instant the child sets it, so this only
# bounds how long a child that DIED goes unnoticed. P7 takes ~2.8 s to bring up
# DDS and pinocchio; 50 ms is 56 wakeups across that, against 2800 at `_POLL_S`.
_READY_POLL_S = 0.05
#: When a CHECK wait first says what it is waiting on, with the full hint. NOT
#: a timeout: the wait still ends only on success or on the producer's process
#: being gone. Long enough that a normal startup (DDS discovery, the board's
#: image server coming up: ~20 s on the fisheye rig) prints nothing.
_FIRST_NOTE_S = 30.0
#: After the first note, how often a one-line "still waiting (N s)" follows, so a
#: live wait is not mistaken for a wedge. The hint is not repeated.
_STILL_WAITING_S = 30.0

#: What to tell the operator when a gated child never set its latch. Keyed by
#: name because the causes have nothing in common, and a wrong hint costs more
#: than no hint: a servo loop that never reached its loop is a DDS or URDF
#: problem, while a dashboard that never listened is almost always a port that
#: is already taken. `None` is the default for the servo loops.
_GATE_HINTS = {
    None: ("It never reached its servo loop, so the arm is under no control at "
           "all and homing would have been written into a ring nobody reads. "
           "Check the DDS interface, and for p7_body the URDF `PinocchioComp` "
           "loads — a bad path raises there by design rather than degrading to "
           "zero gravity comp."),
    "p10_dashboard": (
        "It never started listening, so there is no way to press Start: the "
        "terminal stopped offering Enter when the dashboard took over. "
        "The usual cause is the port already being in use — another dashboard, "
        "or one left running from a previous session. Set the machine file's "
        "[dashboard] port (or G1D_DASHBOARD_PORT) to move it, and Ctrl-C to "
        "leave."),
}

#: Step 1 of `_verify_hand_state`'s refusal, per `--hand`.
_HAND_SERVICE = {
    "dex1": ("the Dex-1 service is running on the robot board, for BOTH "
             "grippers: `sudo ./dex1_1_gripper_server --network eth0 -l -r`"),
    "brainco": ("the BrainCo hand service is running on the robot board and "
                "publishing rt/brainco/{left,right}/state for BOTH hands"),
}


class StartupChecks:
    """The CHECK verifiers, as methods of `Launcher`."""

    def _await_publishing(self, *names: str) -> None:
        """Block until each gated child has a command on the wire.

        **This is what makes homing a ramp.** P7 needs ~2.8 s after spawn to
        import the DDS bindings, complete the `mode_machine` handshake and load
        pinocchio's URDF. `write_ramp` does not pace inside the ring's
        capacity, so a `home()` called right after the spawn would put all 90
        ticks in the ring before P7 ran a single iteration. Its first
        `next_after` would then return a tick near the END of the ramp — 98.9%
        of the way to `home_q` — and it would leave the stall with a segment of
        one tick to cover the whole distance: dead still for 2.77 s, then the
        entire move in ~150 ms against P7's slew clamp, which is meant never
        to bind.

        Nothing downstream could see it. `_verify_home` checks the endpoint, and
        the endpoint was reached — at 5.5 rad/s.

        No timeout, following `warmup`: the only thing that may end this wait
        other than success is the child being gone. `Event.wait` returns the flag
        rather than raising, so the loop is the liveness check and nothing else.
        """
        for name in names:
            ready = self.children.ready[name]
            while not ready.wait(_READY_POLL_S):
                if not self.children.alive(name):
                    raise ConfigError(f"{name} exited before it set its gate. "
                                      + _GATE_HINTS.get(name, _GATE_HINTS[None]))
            self.log(f"[P0] {name}: ready")

    def _await_producer(self, name: str, produced, what: str, hint, *,
                        ok: str = "producing",
                        first_note_s: float = _FIRST_NOTE_S) -> None:
        """Block until a CHECK producer has produced, and SAY so while it has not.

        `_wait` plus the two things it lacked. A `--camera normal` run printed
        the PICO SDK's banner and then nothing at all, arm at rest, with no way
        to tell from the terminal whether P1 or a camera was holding CHECK, or
        whether that process even still existed.

        Still no timeout, for `_await_publishing`'s reason: a server can take
        seconds to come up and DDS discovery longer, so the wait ends on success
        or on the child being gone. A normal startup waits ~20 s here, so the
        full note (what, and the hint) fires once, after `first_note_s`; after
        that only a short elapsed-time line every `_STILL_WAITING_S`, so the
        terminal shows the wait is live without repeating a paragraph.
        `first_note_s` is shorter where the hint is an instruction the operator
        needs up front (the PICO headset), not a diagnosis.

        `ok` names what the wait was for, because one child can be waited on
        twice: P11 is asked to be POLLING and then, separately, to be streaming
        actual poses, and two identical "producing" lines would hide that
        the second wait is a different question about the same process.

        `hint` is a string, or a callable that produces one at the moment the
        note fires: P1's hint is a network diagnosis (`drivers/dds/domain.py::diagnose`) whose
        answer can change while the wait runs — the robot gets switched on, the
        cable gets plugged in — and a hint fixed at the start would keep naming
        a problem that is already gone. It is asked when the note fires, and
        again if the child dies.
        """
        t0 = self.clock.now()
        t_note = t0 + first_note_s
        noted = False
        while not produced():
            if not self.children.alive(name):
                raise ConfigError(
                    f"{name} exited before {what} — its traceback, if any, is "
                    f"above. {hint() if callable(hint) else hint}")
            now = self.clock.now()
            if now >= t_note:
                if not noted:
                    self.log(f"[P0] CHECK still waiting on {name}: {what}. "
                             f"{hint() if callable(hint) else hint}")
                    noted = True
                else:
                    self.log(f"[P0] CHECK still waiting on {name} "
                             f"({now - t0:.0f} s)")
                t_note = now + _STILL_WAITING_S
            self.clock.sleep_until(now + _POLL_S)
        self.log(f"[P0] {name}: {ok}")

    def _camera_hint(self, cam: str) -> str:
        """Where camera `cam`'s frames should be coming from, and what to do.

        Per rig, because the fix is opposite for the two: under `normal` the
        board must be running teleimager, under `fisheye` it must NOT be.
        """
        # `cam` is a ring's name — a VIEW; the wire facts are its source's.
        view = self.cfg.views.get(cam)
        if view is not None:
            cam = view.source
        spec = self.cfg.cameras[cam]
        if spec.source != "zmq":
            return f"source={spec.source} device={spec.device or '<unset>'}"
        host = self.hw.camera_host
        where = f"{host}:{self.hw.camera_ports.get(cam)}"
        if self.hw.image_server != "fisheye":
            return (f"No frame from {where}. This robot file reads the vendor's "
                    f"teleimager: `sudo systemctl start teleimager` on the "
                    f"board. If the fisheye publisher still holds the cameras "
                    f"from an earlier run, release them first with "
                    f"g1d_deploy.runtime.drivers.board.stop('{host}'), which also restarts "
                    f"teleimager.")
        return (f"No frame from {where}. That is the fisheye image server: "
                f"`tail /tmp/fish_eye_image_server.log` on the board. A robot "
                f"with the stock wrists needs a teleimager robot file "
                f"(g1d_normal_dex1).")

    def _robot_diagnosis(self, *, ping: bool = True):
        """`drivers/dds/domain.py::diagnose` for this run: the NIC, the robot board (the
        camera host — Unitree's PC2 is both the image server and the DDS
        participant that publishes lowstate), and only a Cyclone trace written
        since this launcher started."""
        from g1d_deploy.runtime.drivers.dds import domain as dds

        since = self._t_launch
        return dds.diagnose(self.hw.dds_iface, self.hw.camera_host, ping=ping,
                            ownip=lambda: dds.own_ip(since))

    def _verify_robot_subnet(self) -> None:
        """Refuse a run whose NIC cannot hear the robot, BEFORE P1 subscribes.

        The failure this replaces was silent and looked like a quiet robot:
        the card held only an office DHCP lease (or a hand-added robot-subnet
        address had not survived a re-plug), Cyclone bound the address
        it had, and CHECK sat on "waiting on p1_state" forever while ssh and
        ping to the robot — which route by IP — still worked. Nothing about
        that wait can succeed, so it is refused here, in the one place that can
        also say what to do about it (`drivers/dds/domain.py::remedy`). The probe that costs
        time, the ping, is left to the wait's note: a robot that is still
        booting is a reason to wait, not to refuse.
        """
        if not self.hw.dds_iface or os.name != "posix":
            return
        d = self._robot_diagnosis(ping=False)
        if not d.nic or d.bound is None:
            raise ConfigError(d.explain())
        self.log(f"[P0] dds: {self.hw.dds_iface} binds {d.bound}; robot at "
                 f"{self.hw.camera_host}")

    def _report_isolation(self) -> None:
        """Say which scheduling regime this run actually got. Read-only.

        Reserving a sibling core in the machine file keeps *our* processes off
        it; it does not keep the kernel's off. That needs `isolcpus` on the boot
        cmdline, which is a machine-level setting no launcher should be editing
        — so this reports the truth rather than asserting it, and never blocks a
        run. The reservation is an optimisation, not a correctness requirement.

        It is here so that a jitter investigation starts from a recorded fact
        instead of an assumption about how the box was booted.
        """
        reserved = sorted(int(c) for c in self.hw.cpu_reserved)
        if not reserved:
            return
        iso = affinity.isolated()
        if not iso:
            self.log(f"[P0] reserved cores {reserved}: NOT isolated at boot — "
                     f"kernel work may still be scheduled there. To change it: "
                     f"isolcpus={','.join(map(str, reserved))} "
                     f"nohz_full={','.join(map(str, reserved))} on the cmdline.")
        else:
            missing = [c for c in reserved if c not in iso]
            self.log(f"[P0] reserved cores {reserved}, kernel-isolated {sorted(iso)}"
                     + (f" — NOT isolated: {missing}" if missing else " — all isolated"))
        rt = getattr(self.hw, "rt_priority", None)
        self.log(f"[P0] SCHED_FIFO priority: "
                 f"{rt if rt is not None else 'not requested'}")

    def _verify_rates(self) -> None:
        """`lowstate_hz` is MEASURED and re-verified here.

        A lowstate stream at half its declared rate is a startup refusal, not a
        warning: `state_slots` is derived from it, so the state ring would
        silently reach back twice as far as the config asked for and every
        anchor lookup would still look plausible.

        The cameras are only *confirmed producing*: their measured rate is printed and
        judged by the operator.
        """
        state = self.rings["state"]
        i0, t0 = int(state.latest()["index"]), self.clock.now()
        seen0 = {n: r.peek()[1] for n, r in self.rings["cameras"].items()}
        count0 = {n: r.peek_counters() for n, r in self.rings["cameras"].items()}
        self.clock.sleep_until(t0 + _CHECK_WINDOW_S)
        i1, t1 = int(state.latest()["index"]), self.clock.now()

        rate = (i1 - i0) / (t1 - t0)
        for name, region in self.rings["cameras"].items():
            hz = (region.peek()[1] - seen0[name]) / (t1 - t0)
            self.log(f"[P0] camera {name}: {hz:.1f} Hz{self._drop_note(name, count0[name], region)}")
        self.log(f"[P0] lowstate: {rate:.1f} Hz "
                 f"(the robot file says {self.hw.lowstate_hz:.0f})")

        if abs(rate - self.hw.lowstate_hz) > _RATE_TOLERANCE * self.hw.lowstate_hz:
            raise ConfigError(
                f"lowstate measured at {rate:.1f} Hz, the robot file declares "
                f"{self.hw.lowstate_hz:.0f} Hz. state_slots is derived from "
                f"that number.")

    def _fit_lenses(self) -> None:
        """Fit every `fisheye180` camera's lens circle, and so its square.

        At INIT, BEFORE the arena exists: the square is the camera ring's
        shape. P0 opens each fisheye camera itself (the binding P2 will use,
        `p2_camera.make_source`, or the injected `lens_frames`), lets it
        settle, pools frames and fits (`fisheye_circle_fit.decide`). A fit that
        fails or lands far from the robot file's `fallback_lens_circle` falls back to
        that circle with a WARNING; with no `fallback_lens_circle` the launch is
        refused.

        The decision goes into `self.cfg` (each `CameraSpec.crop` / `circle`),
        which every process is handed, and into every rollout manifest via
        P9's provenance. Rectilinear cameras keep `crop = None`.
        """
        from g1d_deploy.runtime.drivers import fisheye_circle_fit as _fit
        from g1d_deploy.runtime.processes.p2_camera import make_source

        cameras = dict(self.cfg.cameras)
        fisheyes = [n for n, s in sorted(cameras.items())
                    if getattr(s, "kind", "normal") == "fisheye180"]
        for name in fisheyes:
            spec = cameras[name]
            if self._lens_frames is not None:
                frames = self._lens_frames(name)
            else:
                frames = _fit.capture_source(make_source(self.cfg, self.hw, name),
                                             (spec.height, spec.width, 3))
            decision = _fit.decide(name, spec, frames)
            self.crops.append(decision)
            self.log(f"[P0] {decision.line()}")
            cameras[name] = replace(spec, crop=decision.crop, lens_circle=decision.lens_circle)
        if fisheyes:
            self.cfg = replace(self.cfg, cameras=cameras)

    def _verify_pico(self) -> None:
        """The PICO's CHECK — two waits and two refusals.

        Everything here reads the intervention ring's HEADER and never its rows:
        rows are written only during an intervention, so at CHECK the ring is
        empty by design and an empty ring is exactly what a healthy idle rig
        looks like. `samples` is bumped on every poll for precisely this reason.

        1. **It polls at all.** Through `_await_producer` rather than a bare
           `_wait`: a P11 that died in `xrt.init()` — a missing
           native library, a service that went away between the autostart and
           the import — refuses the run instead of leaving CHECK spinning
           on a header nobody will ever write again, terminal silent.
        2. **P11 keeps up.** `samples` counts POLLS, and the poll rate decides
           how old a sample is when it is used — which is what
           `intervene_lead_s` has to cover and therefore what the operator feels
           as lag.
        3. **The headset is actually streaming.** A WAIT, not a refusal,
           and that is what makes the flag usable. The headset
           dials in to the PC Service over WiFi — nothing here dials out — so
           it can connect long after this box is up. Refusing would mean the headset
           has to be ON THE HEAD before the command is typed and stay there
           through P5's warmup, or the run dies at CHECK and
           the operator started over. Waiting costs a healthy run nothing: the
           poses are there and the wait falls straight through.
        4. **NO TRIGGER IS ALREADY HELD.** The one that catches a real mistake: a
           controller resting on a bench with something against it. A takeover
           needs a rising edge and the console starts disarmed, so a held trigger
           cannot actually start one — but it would not be NOTICED until the
           operator squeezed and got nothing, and the honest place to say "your
           trigger is down" is before the robot has moved.
        """
        ring = self.rings["intervene"]
        self._await_producer(
            "p11_pico", lambda: int(ring.meta()["samples"]) > 0,
            "its first poll of the XRoboToolkit SDK",
            f"Is the PC Service running on this box? --intervene autostarts "
            f"{pico_service.service_path()}; its output goes to "
            f"{pico_service.LOG_PATH}.",
            ok="polling")

        n0, t0 = int(ring.meta()["samples"]), self.clock.now()
        self.clock.sleep_until(t0 + _CHECK_WINDOW_S)
        n1, t1 = int(ring.meta()["samples"]), self.clock.now()
        rate = (n1 - n0) / (t1 - t0)
        self.log(f"[P0] pico: polled {rate:.1f} Hz "
                 f"(the robot file says {self.hw.pico_poll_hz:.0f})")

        if abs(rate - self.hw.pico_poll_hz) > _RATE_TOLERANCE * self.hw.pico_poll_hz:
            raise ConfigError(
                f"pico polled at {rate:.1f} Hz, the robot file declares "
                f"pico_poll_hz = {self.hw.pico_poll_hz:.0f}. P11 is not keeping "
                f"up, which costs sample age and therefore operator lag "
                f".")

        self._await_producer(
            "p11_pico", lambda: int(ring.meta()["fresh"]),
            "the headset to start streaming poses",
            "The headset connects to THIS box, not the other way round: put it "
            "ON (its tracking runtime does not run off the head), open "
            "XRoboToolkit, point PC Service at this machine's IP on port 63901 "
            "and turn Send on. Ctrl-C to leave, or drop --intervene.",
            ok="streaming poses", first_note_s=5.0)

        meta = ring.meta()
        held = [n for n, h in zip(("left", "right"), meta["trigger"])
                if float(h) >= 1.0 - 1e-6]
        if held:
            raise ConfigError(
                f"the {' and '.join(held)} trigger is fully pressed before the "
                f"run has started. Something is resting on the controller; a "
                f"takeover needs a rising edge, so it would not have started an "
                f"intervention, but it would also not have worked when squeezed "
                f".")

    def _ensure_pico_service(self) -> None:
        """Start the vendor PC Service on this box, if it is not already up.

        Before P11 for `_ensure_board_imager`'s reason, one layer down: P11's
        first action is `xrt.init()`, and a dead service does not fail there —
        `init()` SUCCEEDS and every getter returns zeros, quite happily — so
        the symptom is CHECK waiting on poses that will never come, which reads
        as a headset nobody put on rather than as a service nobody started.

        WARNED, NEVER FATAL, exactly like the board imager. This saves the
        operator a second terminal; it is not the gate. The gate is
        `_verify_pico`, which asks whether poses are ARRIVING — the actual
        requirement, and one no amount of process bookkeeping can substitute
        for. A service that is already running is left strictly alone: it holds
        63901, and a second one would only fight the first for it.
        """
        try:
            was_running = pico_service.ensure_running()
        except pico_service.PicoServiceError as exc:
            self.log(f"[P0] PICO PC Service: autostart FAILED — {exc}")
            self.log("[P0] continuing anyway; CHECK will wait below until the "
                     "headset streams. Start it by hand if it does not.")
            return
        self.log(f"[P0] PICO PC Service: "
                 f"{'already running' if was_running else 'started'}"
                 f" ({pico_service.service_path()})")

    def _ensure_board_imager(self) -> None:
        """Start the fisheye image server on the robot board, if any camera needs it.

        Before P2 rather than after, because P2's first action is to connect a
        SUB socket and block for a frame. A server that is not up does not fail
        there — ZMQ connects to a port nobody is bound to quite happily — so the
        symptom is `check()` waiting forever on `peek()[1] > 0`, which reads as a
        dead camera rather than as a server nobody started.

        Only when a camera actually declares `source="zmq"`. A rig whose cameras
        are all on this machine's own USB has no business touching the robot
        board, and should not need SSH to reach it to be able to run.
        """
        if not any(spec.source == "zmq" for spec in self.cfg.cameras.values()):
            return
        if self.hw.image_server != "fisheye":
            # `[image_server] type = "teleimager"` (or "none"): the vendor's
            # server. The autostart would STOP teleimager — the one server those
            # cameras have — to start ours, which a board that never had the
            # fisheye rig does not even have installed. So no SSH at all. The
            # gate is unchanged: CHECK still waits for every camera's frames.
            self.log(f"[P0] image server = {self.hw.image_server}: leaving the "
                     f"board alone — "
                     f"teleimager on {self.hw.camera_host} must already be "
                     f"serving ports {sorted(self.hw.camera_ports.values())}. "
                     f"If CHECK stalls on a camera: `sudo systemctl start "
                     f"teleimager` on the board.")
            return
        from g1d_deploy.runtime.drivers.board import BoardImagerError, ensure_running

        # WARNED, NEVER FATAL. This is a convenience — it saves an operator an
        # SSH — and it is not the gate. The gate is three lines below, where
        # CHECK blocks until every camera region has published a frame: that
        # asks whether images are ARRIVING, which is the actual requirement,
        # and no amount of SSH bookkeeping can substitute for it.
        #
        # Letting it abort the run inverts that. A server already serving
        # perfectly well would be killed off by a helper that could not log in,
        # or could not parse its own pgrep — both of which have happened.
        # Both are the robot file's: the login ([board]) and where the
        # server lives on the board ([image_server]).
        login = {k: getattr(self.hw, f"board_{k}") for k in
                 ("ssh_user", "ssh_password")
                 if getattr(self.hw, f"board_{k}", None)}
        try:
            was_running = ensure_running(
                self.hw.camera_host, remote_dir=self.hw.image_server_dir,
                conda_root=self.hw.image_server_conda,
                conda_env=self.hw.image_server_conda_env,
                log=lambda msg: self.log(f"[P0] {msg}"), **login)
        except BoardImagerError as exc:
            self.log(f"[P0] board image server: autostart FAILED — {exc}")
            self.log("[P0] continuing anyway; CHECK will refuse below if no "
                     "frames arrive. Start it by hand if they do not.")
            return
        self.log(f"[P0] board image server on {self.hw.camera_host}: "
                 f"{'already running' if was_running else 'started'}")

    @staticmethod
    def _drop_note(name: str, before, region) -> str:
        """How many frames the transport lost over the CHECK window.

        `frame_id` counts what P2 RECEIVED, `source_seq` counts what the server
        SENT. Both advance once per frame, so subtracting one delta from the
        other is the count ZMQ discarded at its high-water mark — the newest
        frame wins, by design, and every earlier one goes without a word.

        Reported, never refused. A handful of drops is what the design asks for
        (a queued frame's arrival stamp is a lie by however long it
        waited), and only the operator can say whether the number in front of
        them is the cost of that trade or a saturated link. What is not
        acceptable is not knowing, which is where this started.

        Silent for a source with no counter — the vendor `image_server` and the
        fakes — because a fabricated zero would read as "measured, and fine".
        """
        after = region.peek_counters()
        if before is None or after is None:
            return ""
        (frames0, seq0), (frames1, seq1) = before, after
        if seq0 < 0 or seq1 < 0:
            return ""                       # this source publishes no counter
        sent, received = seq1 - seq0, frames1 - frames0
        dropped = sent - received
        if dropped <= 0:
            return f"  (no drops, {sent} sent)"
        return f"  ⚠ {dropped}/{sent} frames dropped in transit"

    def _verify_hand_state(self) -> None:
        """The end-effector stream, which nothing downstream can check for you.

        Without this check a silent `rt/dex1/{left,right}/state` is invisible
        for a whole rollout. It is the worst stream to lose silently, for two reasons that compound:

        * `p1_state.GripLatch` opens at `(zeros, 0.0)` and P1 copies that into
          every state sample. A dead topic therefore looks exactly like a
          gripper that is closed and holding still — a *plausible* reading, not
          a missing one, and no consumer can tell the difference.
        * A policy may take the measured gripper as its main (or only)
          proprioceptive input. A frozen reading is then a constant
          observation: the policy stops responding to its own actions,
          converges on whatever it was doing, and the rollout reads as a weak
          policy.

        P8 also seeds `cmd_last` from `grip_meas`, so it opens commanding
        a jaw position nobody measured.

        `t_grip_recv` is the test, and it is exact rather than heuristic: P1
        stamps it on arrival and zero means "never received" on a monotonic
        clock — the one value a real sample cannot have. Refused rather than
        warned, following `_verify_rates`: a run whose proprioception is a
        constant cannot be salvaged afterwards, and the forensics that recover
        that fact cost far more than a startup refusal.

        NOTE this checks the Dex-1's topics specifically, because that is what
        `drivers/robot_state.py::DdsHandState` subscribes to. A Brainco rig publishes on
        `rt/brainco/{left,right}/state` and will need this test taught about it
        rather than deleted.
        """
        state = self.rings["state"]
        topics = " + ".join(self.hw.topic_hand_state)
        deadline = self.clock.now() + _HAND_STATE_WINDOW_S
        while self.clock.now() < deadline:
            latest = state.latest()
            if latest is not None and float(latest["t_grip_recv"]) > 0.0:
                age = self.clock.now() - float(latest["t_grip_recv"])
                self.log(f"[P0] hand state: receiving on "
                         f"{topics} ({age * 1e3:.0f} ms old)")
                return
            self.clock.sleep_until(self.clock.now() + _POLL_S)

        raise ConfigError(
            f"no sample on {topics} after "
            f"{_HAND_STATE_WINDOW_S:.0f}s (every state record still carries "
            f"t_grip_recv = 0, which P1 writes only when the topic has never "
            f"produced).\n\n"
            f"Note P1 pairs the two topics, so ONE silent gripper stops the "
            f"latch: this refusal does not say which.\n\n"
            f"The gripper reads as CLOSED AND STILL for the whole run; a policy "
            f"that reads the gripper state gets a constant — so the rollout "
            f"would look like a bad policy.\n\n"
            f"Robot just switched on? Most often the gripper service is still "
            f"starting: wait ~30 s and launch again (docs/deploying_a_policy/field_notes.md). If "
            f"it persists, check, in order:\n"
            f"  1. {_HAND_SERVICE[getattr(self.hw, 'hand', 'dex1')]}\n"
            f"  2. `ros2 topic hz {self.hw.topic_hand_state[0]}` (or the SDK's "
            f"own subscriber) sees it from THIS machine\n"
            f"  3. iface={self.hw.dds_iface} domain={self.hw.dds_domain} match "
            f"the publisher's — lowstate arriving proves the domain but not "
            f"these topics, since a separate service publishes them")

    def _verify_home(self, q_to: np.ndarray) -> None:
        """Did the robot actually arrive at `home_q` / `home_waist`?

        Sleeping to the last deadline and returning would be commanding a
        posture and never confirming it. Everything downstream assumes the
        robot is at `home_q`: the policy's first observation is supposed to be
        the start it was made for (for a learned policy, in the distribution a
        training episode starts from), and
        the trunk is supposed to be exactly where P6's IK and `PinocchioComp`
        both believe it is.

        Two tolerances, for the reason `home_tol_arm` / `home_tol_waist`
        document: an arm that droops under gravity is honest — the observation
        carries the measured joints — while a trunk that is not where it was
        commanded is a frame error that biases every pose in the system and that
        nothing else in this system can see.

        A refusal, not a warning, for the same reason `_verify_rates` refuses a
        slow lowstate stream: the value is not merely wrong, it is silently
        assumed correct by code that will produce plausible numbers either way.
        This runs before WARMUP, so it costs a homing ramp to find out and not a
        checkpoint load.
        """
        measured = self._measured_body()
        err = np.abs(measured - np.asarray(q_to, dtype=float))
        arm, waist = err[:schema.ARM_DOF], err[schema.ARM_DOF:]
        self.log(f"[P0] homed: arm within {arm.max():.4f} rad "
                 f"(joint {int(np.argmax(arm))}), waist within {waist.max():.4f} "
                 f"rad (joint {int(np.argmax(waist))})")

        bad = []
        if arm.max() > self.cfg.home_tol_arm:
            bad.append(f"arm joint {int(np.argmax(arm))} is {arm.max():.4f} rad "
                       f"from home_q (tolerance {self.cfg.home_tol_arm})")
        if waist.max() > self.cfg.home_tol_waist:
            j = int(np.argmax(waist))
            # Named from the model map, not a literal: the G1-D's second waist
            # axis is the bow, and the third has no motor. A hardcoded
            # ("yaw", "roll", "pitch") would print a joint the G1-D does not have.
            bad.append(f"waist joint {j} ({_WAIST_AXIS_NAMES[j]}) is "
                       f"{waist.max():.4f} rad from home_waist (tolerance "
                       f"{self.cfg.home_tol_waist}) — every flange pose in this "
                       f"run would be biased against a trunk that is not there")
        if bad:
            raise ConfigError(
                "homing did not arrive: " + "; ".join(bad) + ". The robot is "
                "not where the rest of the system assumes it is. Fix the joint, "
                "or raise the tolerance ([home] tol_arm / tol_waist in the deploy "
                "file) if the offset is understood and accepted.")
