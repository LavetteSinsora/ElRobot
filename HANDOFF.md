# ElRobot phone teleop: handoff (2026-10-17, end of the second session)

Read this first in a new session. It says where the project stands, what was decided and
measured, and what is left before the arm is teleoperated with the iPhone. The system as it is
now: **`elrobot_deploy/docs/architecture.md`**.

## 1. The goal and the three layers

iPhone teleop of our ElRobot arm (7-DoF arm + gripper, 8 Feetech ST3215 servos on one serial
bus, Waveshare adapter at 1 Mbps, controlled from the Mac).

| layer | what it does | status |
|---|---|---|
| **phone -> target** | iPhone ARKit pose + buttons over UDP (packet v2); P11 turns it into TCP motion with deadlines (clutch, room-frame mapping, gripper, guards) | **built, tested in simulation end to end**; app v2 written but **not yet compiled** (no Xcode in the cloud session) |
| **IK** | TCP pose -> 7 joints per tick (P6), behind one interface | **built and wired** (`tcp_qp`) |
| **executor** | target ring -> 250 Hz goals on the bus (P7) | **built, run on the real arm** (session 1); now the launcher's bus process; per-joint tuning is profile config, values not chosen yet |

## 2. Where things are

- Repo: GitHub `LavetteSinsora/ElRobot`, branch `claude/charming-curie-4748g7` (three commits on
  `fc9c2cc`: the single-arm runtime, the phone side, docs). Not merged.
- `elrobot/`: bring-up tools (bus, calibration, `relax`, `move`, ...). Unchanged.
- `description/elrobot_ours/`: our URDF (calibrated limits, `tcp`). Unchanged.
- `elrobot_deploy/`: the deploy runtime, rebuilt for one arm this session.
  - `docs/architecture.md`: the map (processes, arena, session, teleop, timing, safety, the
    pluggable IK and executor, configuration, commands).
  - `src/elrobot_deploy/robot/profiles/elrobot.toml`: the robot profile, every key commented
    (bus, executor tuning, home/exit poses, IK, teleop). `--set section.key=value` per run.
  - `runtime/`: `launcher/` (P0), `processes/` (P6 `p6_target.py`, P7 `p7_bus.py`, P11
    `p11_phone.py`), `shared_memory/`, `ik/`, `execution/`, `drivers/`, `teleop/`.
  - `legacy_g1d/`: g1d's code not ported (policies, recording, dashboards, cameras, CHECK),
    unchanged, as reference (README there).
  - `PORTING_PLAN.md`: the plan; its status block lists what changed against it and why.
- `phone_teleop/`: the iOS app (v2: world_epoch, the Mac's state banner + haptics, a one-thumb
  pad) and its README (how to hold it and drive, packet format, build steps).
- Logs: `ElRobot/logs/executor/` (executor-test), `ElRobot/logs/sessions/` (each run's bus
  tick log, the last 10 minutes). Git-ignored.

## 3. Decisions already made (do not re-litigate)

- **Calibration**: URDF zero kept; servo offsets centre each joint's range on 2048 (no 4095->0
  crossing in travel); signs 1-7 = +1, gripper = -1; gripper safe range 3.0..139.8 deg (hand
  sweep, not upstream's 126.2); joint 4 zero re-swept to 2012. Rest pose: shoulder -90, elbow 0,
  wrist pitch +90, rest 0, gripper open.
- **Teleop spec**: everything at **60 Hz** (ARKit rate); deadlines = phone capture time mapped to
  the Mac clock + **40 ms lead**, re-pinned at every clutch press (intervention counter bump);
  **15 Hz** critically damped filter with look-ahead; executor writes at **250 Hz**.
- **Teleop mode** = g1d's intervention path with **no policy**: released clutch -> the arm holds
  the last commanded pose.
- **Safety**: torque goes off only after a completed exit ramp to the rest pose; a failure, a bus
  fault or a second Ctrl-C ends the session with the arm HELD (torque on).
- **Mapping (decided 2026-10-17, second session)**: only the phone's motion RELATIVE to the
  clutch press drives the **TCP**, in the **room frame**: move the phone forward and the arm moves
  forward. Grip: screen facing LEFT, back camera RIGHT, **top (Dynamic Island) pointing FORWARD**
  (ARKit: forward = -X, left = +Z, up = +Y). "Forward" is where the phone's top points,
  horizontally, at each press; rotations apply about the same room axes, about the TCP.
- **Gripper**: hold-to-close (the phone pad's forward zone); the jaw command closes/opens at a
  fixed rate on device time; holds while the clutch is released; **squeeze bound** (the command
  may lead the measured jaw toward closed by at most `squeeze_margin`, enforced in the phone
  source and in the bus process every tick; default 5 deg, uncalibrated); anti-windup (the
  bounded command is stored); hardware TORQUE_LIMIT on servo 8.
- Control box: this Mac. Camera exposure: "sticky". Cameras/recording/dashboard come after teleop.

## 4. The executor: what was measured on the arm (joint 6, +-5 deg sine at 0.25 Hz)

Loop: 4.00 ms period (p99 4.02), bus busy 44 %, command follows the 60 Hz target line exactly.
The executor itself is correct; what remains is how the servos follow.

| setting (joint 6) | amp | delay | reversal dead zone | hold err | err rms | mid-travel jolt |
|---|---|---|---|---|---|---|
| P32 (factory) | 97.5 % | 82 ms | 0.97 deg | -0.09 | 0.53 deg | 440 deg/s^2 |
| P48 | 98.9 % | 65 ms | 0.67 deg | -0.09 | 0.40 deg | 440 |
| P16 (LeRobot's) | 93 % | 132 ms | 1.65 deg | -0.09 | 0.92 deg | **220** |
| **P16 + friction 1.0** | **100.3 %** | **-18 ms** | **0.09 deg** | **-0.09** | **0.51 deg** | 440* |
| P16 + friction 1.5 | 103 % | -92 ms | 0.01 | +0.62 (overcompensated) | 0.72 | 220* |
| P48 + friction 0.5 | 100.2 % | -14 ms | 0.12 | +0.09 | 0.16 | 440 |

(*jolt is quantised: encoder 0.088 deg -> 220 deg/s^2 steps over 20 ms windows; treat as coarse.)

Findings:
- **Shaking happens only in motion.** Holding (no traffic, streaming every tick, or goals only on
  change) is calm. Cause: stick-slip of the moving joint (P-only servo vs static friction) and gear
  backlash knocking when the load changes sign at a reversal, shaking the printed arm; the other
  joints move inside their backlash, below one encoder step.
- **Lower P = smoother travel** (with D fixed, damping relative to stiffness rises): P16 felt
  clearly less shaky. Cost: lag, dead zone, sag.
- **Friction compensation** (`BusServo.friction`, goal leads the line by `f*tanh(v/2deg/s)`) fixes
  timing at the turns (P16: turn lag 520 -> 144 ms) but **not the turn-around jolt** (user feels no
  difference at reversals). 1.0 deg is right for P16; 1.5 overcompensates.
- **No effect / worse**: GOAL_SPEED cap near commanded speed (no change); D48 (much worse: 300 ms
  delay). Acceleration 0 works (no servo ramp).
- **Gravity sag**: joint 4 holds 1.8 deg low at P32 (17 % load). P16 would roughly double it.
- **"Sluggish servo" state**: once, after a mixed session, joint 6 tracked with 5x the lag at the
  same load though all registers were normal; **a 12 V power-cycle cleared it**. If a run shows
  amp < ~95 % or delay > ~100 ms unexpectedly, power-cycle before concluding anything.
- LeRobot (SO-101) writes P16 I0 D32 to STS3215s; lerobot issues #1333, #930 report similar
  jitter; no definitive fix exists online.

## 5. Next steps, in order

### A. On the Mac, no arm power yet
1. `cd elrobot_deploy && uv sync --extra runtime && uv run pytest -q` (91 tests, ~20 s). The
   suite was written on Linux; run it on the Mac once (macOS `/dev/shm`-less shared memory, the
   ARM memory fence).
2. `uv run elrobot_deploy run --sim` in one terminal, `uv run elrobot_deploy fake-phone` in
   another: the whole chain without the phone (the fake phone circles 3 cm left/up).
3. **Build the app** (`phone_teleop/README.md`). The Swift changes were written without a
   compiler: expect to fix small errors in Xcode. Then `uv run elrobot_deploy phone-bench`: hold
   the phone as described, hold the pad, push forward -> `fwd` grows; left -> `left` grows; up
   -> `up` grows; tip the top down -> `rot about left` positive.
4. `uv run elrobot_deploy run --sim` driven by the real phone (M2). Read P11's per-engagement
   latency line (`L_n - L_1` p99, "the deadline map slid"): is the 40 ms lead enough on your WiFi?

### B. The real arm
5. Check `[poses] home` (TCP 25 cm out, 10 cm up, pitched 45 deg down: `[0,-30,0,35,0,40,0,3]`
   deg) and `exit` (rest) against the real arm and table; adjust the profile.
6. `uv run elrobot_deploy run --no-phone`: home, hold, Ctrl-C -> exit ramp -> torque off. A
   bring-up of the launcher on the bus without anyone driving.
7. First teleop (M3), small box and slow, hand on the 12 V switch, small motions first:
   ```bash
   uv run elrobot_deploy run --set limits.velocity_scale=0.3 \
       --set 'teleop.workspace_min=[-0.15, 0.15, 0.04]' --set 'teleop.workspace_max=[0.15, 0.32, 0.2]'
   ```
   (Quote values with brackets: zsh would glob them.)

### C. Tuning (whenever; it is configuration now)
8. Per-joint servo settings in `[executor]` (`p_coef`, `d_coef`, `friction`), from
   `executor-test` sine runs (section 4): light joints 5-7 P16 + friction 1.0 deg is the current
   best guess; the gravity joints 2-4 (and 1) still need their own runs.
9. Gravity compensation (sag ~ gravity torque / stiffness), squeeze margin vs force on a real
   object, TORQUE_LIMIT on servo 8.
10. If level, sideways gripper motion is needed: the wrist-branch question (`tcp_qp.py`
    docstring); a new IK is `--set ik.solver=file.py:factory` (`runtime/ik/__init__.py`).

### D. Later: cameras (M4), recording / dashboard / CHECK (M5), policies: from `legacy_g1d/`.

## 6. Minimum checklist before teleoperating with the phone

- [x] Single-arm schema; the executor runs as the launcher's bus process on the real ring
- [x] Phone source with clutch, anchoring, gripper, guards; room-frame mapping for the grip
- [x] P6 driving `tcp_qp` at 60 Hz; launcher teleop mode; end-to-end test in simulation
- [ ] The test suite run on the Mac
- [ ] App v2 built and on the iPhone; `phone-bench` confirms the axis mapping
- [ ] `run --sim` driven by the real phone; latency measured (lead 40 ms enough?)
- [ ] Home and exit poses checked on the real arm; `run --no-phone` on the real arm
- [ ] Executor per-joint P/friction set (or accept the defaults for a first slow session)
- [ ] Gripper squeeze margin and TORQUE_LIMIT checked on a real object
- [ ] First hardware session: small workspace box, low velocity_scale, hand on the 12 V switch

## 7. How to run things (from `ElRobot/elrobot_deploy/`)

```bash
uv sync --extra runtime                                   # deploy env (includes ../elrobot)
uv run pytest -q                                          # deploy tests (no hardware)
uv run elrobot_deploy config                              # print the resolved profile
uv run elrobot_deploy phone-bench                         # the grip check, no robot
uv run elrobot_deploy run --sim                           # simulated servos
uv run elrobot_deploy fake-phone                          # drive a --sim run without the phone
uv run elrobot_deploy run --no-phone                      # real arm: home, hold, exit
uv run elrobot_deploy run                                 # real arm, phone teleop
uv run elrobot_deploy executor-test --joints 6 --p-coef 16 --friction-deg 1.0   # sine test
#   (also: python -m elrobot_deploy.cli.executor_test ...; flags unchanged: --hold, --d-coef,
#   --goal-speed, --rewrite-every, --amp-deg, --freq, --sim, --dry-run)
cd .. && uv run python -m elrobot.calibration --show      # live joint angles (read-only)
uv run python -m elrobot.relax                            # torque off everywhere
uv run pytest -q                                          # root repo tests
```

## 8. Gotchas

- **ST3215 multi-turn offset trap**: never write an offset with torque on; power-cycle after
  writing offsets; no joint may cross 4095->0 while powered (the centring offsets prevent it).
- Torque left on by a crashed/old run, or by a session that ended without its exit ramp (a
  fault, a second Ctrl-C: by design the arm stays held): the bus process refuses to start
  ("torque is already on"); support the arm and run `elrobot.relax`. The bus process never turns
  off torque it didn't turn on.
- One Ctrl-C under `uv run` reaches the process twice; two within 0.5 s count as one. A real
  second Ctrl-C (later) skips the exit ramp and leaves the arm held.
- The clutch only counts if pressed after the rollout starts (and after any forced release):
  the phone shows "RELEASED" until the thumb is lifted.
- The `VIRTUAL_ENV ... will be ignored` warning from uv is harmless.
- `legacy_g1d/` is excluded from ruff and pytest; don't "fix" it, port from it.
- User preferences: explain before acting on hardware; never command motion without the user's go.
