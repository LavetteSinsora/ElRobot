"""P11 — the operator's console (only with `--intervene`).

The maths is `runtime/teleop/intervention.py`; the PC service it needs is
`runtime/teleop/pico_service.py`. For the operator:
`docs/deploying_a_policy/running_and_live_dashboard.md`, "`--intervene`: PICO
takeover"; the knobs: `docs/deploying_a_policy/deploy_toml.md`, "`[intervene]`:
the PICO takeover".

Polls the XRoboToolkit SDK at the robot file's `pico_poll_hz` and turns what it
finds into rows of the intervention ring. **Owns** `control.intervene` and the
intervention ring; **reads** the state ring (where the arms are) and the
target ring (what the jaws were last commanded). It writes no target and moves
no joint: every row it publishes still has to pass P6's IK, P6's velocity clamp
and P6's write gate before it reaches a wire.

Three things about this loop are worth stating before the code.

**A poll that finds nothing new is the common case, and it is not an error.**
The SDK republishes at roughly 90 Hz and this polls at 180, so about half of all
iterations find the same device timestamp they saw last. That has to be dropped
rather than written twice — a duplicate row would claim the operator held still
for a sample they were moving through, and at 90 Hz of them it would halve the
apparent speed of every correction.

**The intervention LEVEL is the clutches**, not a separate button: it opens when
the first hand squeezes and closes when the last one lets go. So "somebody is
driving" and "some hand is engaged" are the same proposition by construction,
and there is no way to be in an intervention with nobody holding anything. A
hand inside an intervention that is not engaged HOLDS, which is what makes
single-arm correction work: grab the left, fix it, let go, and the right arm
never moved.

**The counter is stored LAST, after the row it names.** `control.intervene` is
every other process's only signal that the world changed, and the row a reader
switches onto must already be readable — the same rule P5 follows for
`newest_chunk_id`. Storing it first would wake P6 into an intervention
with nothing to solve.

The one thing that is NOT here is any notion of where the operator is standing.
The mapping is tool-frame (`intervention.Anchor`), so the XR world's orientation
cancels between the two anchors and there is no frame calibration, no headset
yaw, and nothing to get wrong by facing the wrong way.

The SDK speaks `[x y z qx qy qz qw]`; every pose in `elrobot_deploy` is
`[x y z qw qx qy qz]`. That conversion happens in `_pose7` and nowhere else,
which is the only way a convention like that does not eventually cost a week.
The import of `xrobotoolkit_sdk` is likewise local to `open()`: it is a
site-installed binding, and a laptop that will never intervene must not need it
to `import elrobot_deploy`.
"""

from __future__ import annotations

import numpy as np

from elrobot_deploy.runtime import cpu_pinning as affinity, timing as clock
from elrobot_deploy.runtime.teleop import intervention as iv
from elrobot_deploy.runtime.shared_memory import Arena, build, open_all, schema
from elrobot_deploy.config_loading.resolved import DEX1_TRAVEL
from elrobot_deploy.config_loading.derive import derive

#: The getters this loop needs. Checked before `init()` so that a PC-Service
#: build without controller input fails at startup with a list of what is
#: missing, rather than as an `AttributeError` mid-rollout with the robot moving.
SDK_GETTERS = (
    "get_time_stamp_ns",
    "get_left_controller_pose", "get_right_controller_pose",
    "get_left_trigger", "get_right_trigger",
    "get_left_axis", "get_right_axis",
)


def build_script() -> str:
    """The script that builds `xrobotoolkit_sdk`, in this checkout."""
    from pathlib import Path
    return str(Path(__file__).resolve().parents[4] / "scripts"
               / "build_xrobotoolkit.sh")


def sdk_missing_message() -> str:
    """What `--intervene` says when `xrobotoolkit_sdk` does not import."""
    import sys
    return (
        "--intervene needs the PICO bindings, and `xrobotoolkit_sdk` does not "
        f"import in {sys.executable}. It is a pybind11 extension over a native "
        "SDK, so no package install can provide it. Build it once on this box "
        "(needs cmake, git and a C++ compiler):\n\n"
        f"    {build_script().replace('build_xrobotoolkit.sh', 'install.sh')} "
        "--intervene\n\n"
        "or, into another interpreter that has pybind11:\n\n"
        f"    PYTHON={sys.executable} bash {build_script()}\n\n"
        "then start the XRoboToolkit PC Service and point the PICO app at this "
        "box (the script's closing notes say how).")


def sdk_available() -> bool:
    try:
        import xrobotoolkit_sdk  # noqa: F401
    except ImportError:
        return False
    return True


class PicoSource:
    """The vendor SDK, behind the smallest surface P11 actually uses."""

    def __init__(self):
        self._x = None
        self._last_ns = 0

    def open(self) -> None:
        try:
            import xrobotoolkit_sdk as xrt
        except ImportError as exc:      # noqa: PERF203
            raise RuntimeError(sdk_missing_message()) from exc

        missing = [n for n in SDK_GETTERS if not callable(getattr(xrt, n, None))]
        if missing:
            raise RuntimeError(
                "xrobotoolkit_sdk is missing controller input functions: "
                + ", ".join(missing) + ". The PC-Service build is too old or is "
                "not the one scripts/build_xrobotoolkit.sh builds.")
        xrt.init()
        self._x = xrt
        self._last_ns = 0

    def close(self) -> None:
        if self._x is not None:
            try:
                self._x.close()
            finally:
                self._x = None

    def read(self) -> dict | None:
        """One coherent NEW sample, or None.

        None means one of two things and the caller must not distinguish them,
        because the correct response is identical: the payload was mid-update, or
        the device timestamp has not advanced since the last read. The second is
        the common case by design — this polls at about twice the publish rate —
        and treating it as a dead stream is the single mistake that would make
        every takeover release itself on the following poll.

        The vendor's getters sit behind separate mutexes, so the timestamp is
        read either side of the batch and the sample is discarded when it moved.
        A dropped sample costs 11 ms and the next one is complete; a torn one
        pairs one hand's pose with the other hand's instant, which composes into
        a plausible target nobody asked for.
        """
        x = self._x
        for _ in range(3):
            before = int(x.get_time_stamp_ns())
            left, right = x.get_left_controller_pose(), x.get_right_controller_pose()
            trigger = (x.get_left_trigger(), x.get_right_trigger())
            axis = (x.get_left_axis(), x.get_right_axis())
            after = int(x.get_time_stamp_ns())
            if before != after or before <= 0:
                continue
            if before == self._last_ns:
                return None
            self._last_ns = before
            return {"t_device": before * 1e-9, "pose": (left, right),
                    "trigger": trigger, "axis": axis}
        return None


def _pose7(raw) -> np.ndarray:
    """SDK `[x y z qx qy qz qw]` -> contract `[x y z qw qx qy qz]`.

    The ONE place the two quaternion conventions meet. Normalised here as well:
    everything in `intervene` builds rotations from these and inverts them as
    conjugates, both of which are exact for a unit quaternion and neither of
    which is for one off by a percent.
    """
    v = np.asarray(raw, dtype=np.float64).reshape(schema.POSE_DIM)
    q = iv.quat_normalise(v[3:])
    return np.array([v[0], v[1], v[2], q[3], q[0], q[1], q[2]])


def _lever(raw) -> float:
    """One analog trigger, clamped to [0, 1]; anything unreadable is 0.0.

    Not an error. A garbled trigger should read as "not pressed", which releases
    the clutch and holds the arm — the safe answer, and the same one a finger
    lifting gives.
    """
    try:
        v = float(raw)
    except Exception:                                    # noqa: BLE001
        return 0.0
    return float(np.clip(v, 0.0, 1.0)) if np.isfinite(v) else 0.0


def _axis_y(raw) -> float:
    """The thumbstick's Y component, clamped to [-1, 1]; unreadable is 0.0.

    The SDK reports `(x, y)`; only Y drives the jaw.
    """
    try:
        v = float(np.asarray(raw, dtype=np.float64).reshape(2)[1])
    except Exception:                                    # noqa: BLE001
        return 0.0
    return float(np.clip(v, -1.0, 1.0)) if np.isfinite(v) else 0.0


class PicoLoop:
    """One poll per `step()`, so the whole state machine is testable without an
    SDK, a clock, an arena writer or a robot."""

    def __init__(self, rings, cfg, hw, source):
        self.cfg = cfg
        self.source = source
        self.control = rings["control"]
        self.state = rings["state"]
        self.hand = rings["hand"]
        self.ring = rings["intervene"]
        self.waist_cols = np.asarray(hw.joints_waist, dtype=np.intp)

        self.console = iv.Console()
        self.pace = iv.Pace(cfg.intervene_lead_s)
        # The gripper knobs are Dex-1 radians. A hand with other units — the
        # BrainCo's 0..1 — gets the same feel: the same fraction of ITS travel
        # per second, and a squeeze margin the same fraction of its travel.
        # Exactly 1.0 on the Dex-1.
        travel = float(hw.hand_max - hw.hand_min) / DEX1_TRAVEL
        self.grippers = [
            iv.Gripper(cfg.intervene_gripper_rate * travel,
                       hw.hand_min, hw.hand_max,
                       squeeze_margin=cfg.intervene_squeeze_margin_rad * travel,
                       up_closes=cfg.intervene_up_closes,
                       close_is_max=hw.hand_close_is_max)
            for _ in range(schema.N_HANDS)]
        self.anchor: list[iv.Anchor | None] = [None] * schema.N_HANDS

        self.scale = float(cfg.intervene_scale)
        self.stale_budget = float(cfg.intervene_stale_budget_s)
        # The robot side is the FLANGE, never the robot file's TCP: a
        # rotation pivots at the flange (`intervention.py`, "THE ROBOT SIDE IS THE
        # FLANGE").
        # PER HAND, `(2, 7)`. The default calibration (`intervene_hand_tcp`)
        # differs between the two by 12.5 degrees of rotation, so a shared value
        # would be wrong for at least one of them by that much.
        self.hand_tcp = np.asarray(cfg.intervene_hand_tcp,
                                   dtype=np.float64).reshape(
                                       schema.N_HANDS, schema.POSE_DIM)

        # The action this loop is asking for. Every row is this, whole: both
        # hands every time, engaged or not, because P6 writes one target per tick
        # for the whole robot and a row that spoke for only one arm would leave
        # the other undefined.
        self.pose = np.zeros((schema.N_HANDS, schema.POSE_DIM))
        self.hand_cmd = np.zeros((schema.N_HANDS, schema.HAND_DIM))
        self.waist = np.zeros(schema.WAIST_DOF)

        self._t_last_sample = -np.inf
        self._t_prev_device: float | None = None
        # A gap longer than this contributes only this much to the jaw
        # integration, so a stall cannot dump a large offset in one step. Derived
        # from the poll rate rather than configured: it is "a few samples", which
        # is a property of the stream and not a choice.
        self._dt_cap = 3.0 / float(hw.pico_poll_hz)

    # --- where things are ---------------------------------------------------

    def _measured(self):
        """Where the arms and jaws actually are, or None before P1 publishes."""
        snap = self.state.latest()
        if snap is None:
            return None
        return (np.asarray(snap["flange"], dtype=np.float64).copy(),
                np.asarray(snap["grip_meas"], dtype=np.float64).reshape(
                    schema.N_HANDS, schema.HAND_DIM).copy(),
                np.asarray(snap["q"], dtype=np.float64)[self.waist_cols].copy())

    def _commanded_jaw(self, measured_grip):
        """The aperture the jaws are being driven to, per hand.

        **Commanded, and for the gripper that is not a matter of millimetres.**
        0 rad is fully closed, so a jaw squeezing an object sits ABOVE its
        command — ask for 1.0, the object stops it at 1.4, and that difference IS
        the grip force through the driver's position loop. Anchoring on the
        measurement commands 1.4, the squeeze goes to zero, and the robot drops
        what it was holding at the very moment a human took over to save it.

        Falls back to the measurement only when there is no command to continue:
        before P6 has written anything, or after the slot naming `k_exec` has
        been evicted by a wrap.
        """
        entry = self.hand.get(self.control.k_exec)
        if entry is None or not np.isfinite(entry.payload).all():
            return measured_grip[:, 0].copy()
        return np.asarray(entry.payload, dtype=np.float64)[:, 0].copy()

    # --- transitions --------------------------------------------------------

    def _take_over(self) -> bool:
        """Freeze what the arms are doing. False if nothing can be read.

        A takeover that cannot establish where the arms are is refused rather
        than guessed at, and refusing leaves the console disarmed — the
        operator's squeeze does nothing visible and they squeeze again. The only
        way here is a state ring with no samples at all, which CHECK has already
        proved impossible before the run started.

        **The ARM anchors on the MEASURED flange and the JAW on the COMMANDED
        aperture**, and the asymmetry is deliberate rather than an oversight. The
        arm's measurement lags its command by the servo's velocity lag plus
        steady-state droop — ~13 mm standing on the reference robot — so anchoring there
        costs one small step down at the takeover, bounded and not cumulative,
        which is the price of an anchor that is unambiguously "where the arm
        actually is". The jaw's gap is not lag at all but grip force, and giving
        it up drops the object.
        """
        measured = self._measured()
        if measured is None:
            return False
        flange, grip, waist = measured
        self.pose[:] = flange
        self.waist[:] = waist
        self.hand_cmd[:] = 0.0
        self.hand_cmd[:, 0] = self._commanded_jaw(grip)
        self.anchor = [None] * schema.N_HANDS
        self.pace.reset()
        return True

    def _engage(self, grabbed, xr_pose) -> None:
        """Re-anchor each hand that just took hold.

        Both halves at once, which is what makes releasing free: the operator's
        controller pose and the arm's held pose are frozen together, so however
        far apart the two have drifted, the next sample commands no motion. It is
        the remedy for every mapping problem there is — let go, put your hand
        where it is comfortable and pointing the way the gripper points, take
        hold again.
        """
        for h in range(schema.N_HANDS):
            if not grabbed[h]:
                continue
            self.anchor[h] = iv.Anchor.freeze(
                xr_pose[h], self.pose[h], hand_tcp=self.hand_tcp[h])
            self.grippers[h].engage(self.hand_cmd[h, 0])

    def _follow(self, xr_pose, axis, dt_dev, grip_meas) -> None:
        """Advance the held action: driven where engaged, frozen where not."""
        for h in range(schema.N_HANDS):
            if self.console.engaged[h] and self.anchor[h] is not None:
                self.pose[h] = self.anchor[h].follow(xr_pose[h], self.scale)
                self.hand_cmd[h, 0] = self.grippers[h].step(
                    axis[h], dt_dev,
                    measured=None if grip_meas is None else grip_meas[h, 0])
            else:
                # The pose is left exactly as it is — that IS the hold, and it is
                # the last thing the operator asked for rather than anything
                # measured, for the reason P7's stall republishes `q_cmd_last`
                # frozen.
                self.hand_cmd[h, 0] = self.grippers[h].hold()

    def release(self) -> None:
        """End any running intervention, from this side.

        **The counter is left EVEN**, which is what lets it stay monotonic and
        single-writer for the life of the session: a P0 that stored a zero into
        it would be a second writer of a word a live P11 also writes, for no gain
        (cf. `newest_chunk_id`, whose producer resets it for the same reason).

        `console.reset()` is the half that matters physically. An operator whose
        finger is still on a trigger when a rollout ends must not have the next
        one begin already intervening: entry needs a fresh release and squeeze,
        the same rule CHECK applies at startup.
        """
        if iv.is_intervening(self.control.intervene):
            self.control.intervene = self.control.intervene + 1
        self.console.reset()
        self.pace.reset()
        self.anchor = [None] * schema.N_HANDS
        # The header too, or the dashboard goes on showing an operator who let go
        # minutes ago: it is only written on the active path, and this is the one
        # exit that does not go through it.
        self.ring.note(active=0, engaged=np.zeros(schema.N_HANDS))

    # --- one poll -----------------------------------------------------------

    def step(self, t_now: float, active: bool = True) -> int:
        """Poll once. Returns `+1` on takeover, `-1` on handback, else `0`.

        `active` is the session gate. When it is false this still polls and still
        records the raw buttons — that is the window CHECK measures the rate in,
        and the one in which an operator would notice an unplugged headset — but
        the console never runs, so a trigger squeezed during HOME or READY does
        nothing.
        """
        try:
            sample = self.source.read()
        except Exception:                                # noqa: BLE001
            sample = None

        meta = self.ring.meta()
        note = {"samples": int(meta["samples"]) + 1}

        if sample is None:
            fresh = t_now - self._t_last_sample
            note["fresh"] = int(fresh <= self._dt_cap)
            # THE STALE HANDBACK. Until the budget runs out the arm
            # simply HOLDS — no rows are written, P6 has nothing new to solve, P7
            # stalls and republishes the last command frozen — which is the
            # correct response and is reached with no code at all. Past it the
            # run is handed back to the policy, deliberately and visibly, rather
            # than left waiting on a headset that is not coming back.
            #
            # The raw buttons are NOT written on this path. They belong to the
            # last real sample, and zeroing them on the polls in between would
            # make them flicker at the poll rate — which is exactly what CHECK
            # reads to decide whether a trigger is being held down.
            if active and self.console.active and fresh > self.stale_budget:
                self.release()
                self.ring.note(**note)
                return -1
            self.ring.note(**note)
            return 0

        try:
            xr_pose = np.stack([_pose7(p) for p in sample["pose"]])
        except Exception:                                # noqa: BLE001
            # A malformed sample is not a dead stream — it arrived, so it counts
            # as liveness. Skip it and take the next one 11 ms later.
            self._t_last_sample = t_now
            self.ring.note(**note, t_last_sample=t_now, fresh=1)
            return 0

        trigger = np.array([_lever(v) for v in sample["trigger"]])
        axis = np.array([_axis_y(v) for v in sample["axis"]])
        t_device = float(sample["t_device"])
        dt_dev = (0.0 if self._t_prev_device is None
                  else min(max(t_device - self._t_prev_device, 0.0), self._dt_cap))
        self._t_prev_device = t_device
        self._t_last_sample = t_now
        note.update(t_last_sample=t_now, fresh=1, trigger=trigger, axis_y=axis)

        if not active:
            self.ring.note(**note)
            return 0

        edge, grabbed = self.console.update(trigger)
        if edge < 0:
            self.release()
            self.ring.note(**note, active=0, engaged=np.zeros(schema.N_HANDS))
            return -1

        counter = self.control.intervene
        if edge > 0:
            if not self._take_over():
                self.console.reset()
                self.ring.note(**note)
                return 0
            # The value this intervention WILL have. Everything below is written
            # under it; the counter itself is stored last.
            counter += 1

        if not self.console.active:
            self.ring.note(**note, active=0, engaged=np.zeros(schema.N_HANDS))
            return 0

        measured = self._measured()
        self._engage(grabbed, xr_pose)
        self._follow(xr_pose, axis, dt_dev,
                     None if measured is None else measured[1])
        self.ring.write(
            intervene=counter,
            deadline=self.pace.due(t_device, t_now),
            pose=self.pose, hand=self.hand_cmd, waist=self.waist,
            engaged=self.console.engaged.astype(np.uint64),
            trigger=trigger, axis_y=axis,
            direction=np.array([g.direction for g in self.grippers]),
            t_device=t_device, t_recv=t_now)
        if edge > 0:
            # LAST, after the row it names. `control.intervene` is every reader's
            # only signal that a world has changed, and the row a reader switches
            # onto must already be readable — the rule P5 follows for
            # `newest_chunk_id`. Otherwise P6 wakes into an intervention
            # with nothing to solve and holds for one poll.
            self.control.intervene = counter
        self.ring.note(**note, active=1,
                       engaged=self.console.engaged.astype(np.uint64))
        return edge


# --- process ----------------------------------------------------------------


def run(shm_name: str, cfg, hw, *, source=None, ready=None) -> None:
    """P11 as a process. Runs from CHECK until STOPPED.

    It outlives ACTIVE on purpose. P0's exit sequence walks the session past
    ACTIVE while the arm is still being ramped down, and an operator whose finger
    is on a trigger at that moment must be handed back rather than left holding a
    counter nobody will ever decrement — `release()` on the way out is what
    guarantees the word is EVEN when the session ends, which is what a reset
    needs and what keeps this the word's only writer.
    """
    arena = Arena.attach(shm_name, build(cfg, derive(cfg, hw)))
    rings = open_all(arena, cfg)
    affinity.pin(hw, "p11_pico")
    control = rings["control"]

    if source is None:
        source = PicoSource()
    try:
        source.open()
        loop = PicoLoop(rings, cfg, hw, source)
        if ready is not None:
            ready.set()
        pacer = clock.Pacer(hw.pico_poll_hz)
        while control.session < schema.STOPPED:
            loop.step(clock.now(), active=control.session == schema.ACTIVE)
            pacer.wait()
        loop.release()
    finally:
        source.close()
        arena.close()


# --- the bench ---------------------------------------------------


def main() -> int:
    """`python -m elrobot_deploy.runtime.processes.p11_pico` — the headset, with no robot.

    The check to run FIRST, before anything is powered. It opens the SDK, polls
    at `pico_poll_hz`, and prints what the operator's hand is doing in the
    ROBOT's terms: the poll rate, the triggers and thumbsticks, and — once a
    trigger is squeezed — the displacement of their virtual gripper expressed in
    its own tool frame.

    **That last column is the whole point.** It is the mapping the arm will
    follow, and it is checkable in thirty seconds without risking anything:
    squeeze a trigger, push your hand straight out along the way the gripper is
    pointing, and `out` must go positive. If `out` stays near zero while
    `across` or `along` moves, `intervene_hand_tcp`'s ROTATION is wrong and the
    operator's tool frame is not the robot's — which on a robot shows up as an
    arm that moves confidently in the wrong direction.

    The robot half is a stand-in identity pose: there is no arena and no state
    ring here, so this measures the OPERATOR's side of the map alone, which is
    the half a rig can get wrong.
    """
    import argparse

    from elrobot_deploy.config_loading import resolved as cfgmod

    ap = argparse.ArgumentParser(description="PICO bench")
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--hand-tcp", default=None, help=(
        "override BOTH hands' controller -> virtual TCP with one pose, as 7 "
        "comma-separated numbers x,y,z,qw,qx,qy,qz. Only for checking a "
        "candidate measurement; the default is the per-hand calibration in "
        "`DeployConfig.intervene_hand_tcp`."))
    args = ap.parse_args()

    hw = cfgmod.load_hardware()
    # Per hand, from the config's own default
    # (`DeployConfig.intervene_hand_tcp`). `--hand-tcp` overrides BOTH hands with one
    # pose, which is only for checking a candidate measurement by hand.
    hand_tcp = np.asarray(cfgmod.DeployConfig.intervene_hand_tcp,
                          dtype=np.float64).reshape(schema.N_HANDS,
                                                    schema.POSE_DIM)
    if args.hand_tcp is not None:
        one = np.array([float(v) for v in args.hand_tcp.split(",")])
        hand_tcp = np.tile(one, (schema.N_HANDS, 1))

    console = iv.Console()
    anchor: list = [None] * schema.N_HANDS
    held = np.tile(iv.IDENTITY_POSE, (schema.N_HANDS, 1))
    xr = held.copy()
    trigger = np.zeros(schema.N_HANDS)
    axis = np.zeros(schema.N_HANDS)

    source = PicoSource()
    source.open()
    n, fresh = 0, 0
    pacer = clock.Pacer(hw.pico_poll_hz)
    t_end = clock.now() + args.seconds
    t_report = clock.now() + 1.0
    print(f"polling at {hw.pico_poll_hz:.0f} Hz for {args.seconds:.0f} s. "
          f"Squeeze a trigger, then push your hand out along the way the "
          f"gripper points: `out` must go positive. Ctrl-C to stop.")
    try:
        while clock.now() < t_end:
            n += 1
            sample = source.read()
            if sample is not None:
                fresh += 1
                try:
                    xr = np.stack([_pose7(p) for p in sample["pose"]])
                except Exception:                        # noqa: BLE001
                    continue
                trigger = np.array([_lever(v) for v in sample["trigger"]])
                axis = np.array([_axis_y(v) for v in sample["axis"]])
                _, grabbed = console.update(trigger)
                for h in range(schema.N_HANDS):
                    if grabbed[h]:
                        anchor[h] = iv.Anchor.freeze(xr[h], held[h],
                                                     hand_tcp=hand_tcp[h])
                        print(f"  [{'LR'[h]}] anchored")
                    elif not console.engaged[h]:
                        anchor[h] = None
            now = clock.now()
            if now >= t_report:
                t_report = now + 1.0
                cols = []
                for h in range(schema.N_HANDS):
                    if anchor[h] is None:
                        cols.append(f"[{'LR'[h]}] {'-' * 34}")
                        continue
                    d = (anchor[h].follow(xr[h])[:3] - held[h][:3]) * 100.0
                    cols.append(f"[{'LR'[h]}] out{d[0]:+6.1f} across{d[1]:+6.1f} "
                                f"along{d[2]:+6.1f} cm")
                print(f"{fresh:4d}/{n:4d} fresh  " + "  ".join(cols)
                      + f"  trig {trigger[0]:.2f}/{trigger[1]:.2f}"
                        f"  stick {axis[0]:+.2f}/{axis[1]:+.2f}")
                n = fresh = 0
            pacer.wait()
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PicoLoop", "PicoSource", "SDK_GETTERS", "main", "run"]
