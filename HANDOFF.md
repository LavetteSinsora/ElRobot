# ElRobot phone teleop: handoff (2026-10-17, end of the second session, + dashboard)

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
| **session + dashboard** | hardware check, Home (previewed) and Start gates, a browser view of the URDF (measured / commanded / home), dry runs | **built, tested in simulation**; not yet on the Mac |

The session as it runs now: `run` -> the hardware check (a table on the terminal and the
dashboard) -> waiting for the phone to stream (>= 30 Hz for 1 s, tracking normal) -> the arm held where it was found, the home pose shown as a green ghost at
`http://127.0.0.1:8080` -> **Home** (button or Enter) -> homing -> **Start** -> tap ENGAGE on
the phone to drive -> Reset / Exit (button) or Ctrl-C. `--dry-run` does all of it on the real
arm with torque OFF (only the blue ghost moves).

## 2. Where things are

- Repo: GitHub `LavetteSinsora/ElRobot`, branch `claude/charming-curie-4748g7` (commits on
  `fc9c2cc`: the single-arm runtime, the phone side, docs, the dashboard round). Not merged.
  To get it on the Mac: `git fetch origin claude/charming-curie-4748g7` then check it out (add
  the remote first if the local repo has none).
- `elrobot/`: bring-up tools (bus, calibration, `relax`, `move`, ...). Unchanged.
- `description/elrobot_ours/`: our URDF (calibrated limits, `tcp`). Unchanged.
- `elrobot_deploy/`: the deploy runtime, rebuilt for one arm this session.
  - `docs/architecture.md`: the map (processes, arena, session, teleop, timing, safety, the
    pluggable IK and executor, configuration, commands).
  - `src/elrobot_deploy/robot/profiles/elrobot.toml`: the robot profile, every key commented
    (bus, executor tuning, home/exit poses, IK, teleop). `--set section.key=value` per run.
  - `runtime/`: `launcher/` (P0), `processes/` (P6 `p6_target.py`, P7 `p7_bus.py`, P11
    `p11_phone.py`), `shared_memory/`, `ik/`, `execution/`, `drivers/`, `teleop/`.
  - `dashboard/`: P10, the browser view (stdlib server + a three.js page, vendored).
  - `legacy_g1d/`: g1d's code not ported (policies, recording, inspection, cameras),
    unchanged, as reference (README there).
  - `PORTING_PLAN.md`: the plan; its status block lists what changed against it and why.
- `phone_teleop/`: the iOS app (v2: world_epoch, the Mac's state banner + haptics, ENGAGE
  toggle + GRIP hold) and its README (how to hold it and drive, packet format, build steps).
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
- Control box: this Mac. Camera exposure: the camera's own settings first (`camera_first`),
  ours ("sticky") only when they clearly fail (decided 2026-10-17, section 5D).

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
1. `cd elrobot_deploy && uv sync && uv run pytest -q` (112 tests, ~30 s). The
   suite was written on Linux; run it on the Mac once (macOS shared memory, the ARM memory
   fence).
2. `uv run elrobot_deploy view`, open http://127.0.0.1:8080: the URDF, the home pose (green)
   and the exit pose (orange), sliders per joint. Is home where you want it?
3. `uv run elrobot_deploy run --sim` in one terminal, `uv run elrobot_deploy fake-phone` in
   another, the dashboard in the browser: press Home, then Start; the fake phone then circles
   the simulated arm 3 cm left/up.
4. **Build the app** (`phone_teleop/README.md`). The Swift changes were written without a
   compiler: expect to fix small errors in Xcode. Then `uv run elrobot_deploy phone-bench`: hold
   the phone as described, tap ENGAGE, push forward -> `fwd` grows; left -> `left` grows; up
   -> `up` grows; tip the top down -> `rot about left` positive.
5. `uv run elrobot_deploy run --sim` driven by the real phone (M2). Read P11's per-engagement
   latency line (`L_n - L_1` p99, "the deadline map slid"): is the 40 ms lead enough on your WiFi?

### B. The real arm
6. `uv run elrobot_deploy run --dry-run` (arm powered, torque never on): the hardware check
   reads every servo; the measured arm on the dashboard must match the real one; Home and
   Start move only the blue ghost; driving with the phone moves only the ghost.
7. Check `[poses] home` (TCP 25 cm out, 10 cm up, pitched 45 deg down: `[0,-30,0,35,0,40,0,3]`
   deg) and `exit` (rest) against the real arm and table; adjust the profile.
8. `uv run elrobot_deploy run --no-phone`: check, Home, hold, Exit -> exit ramp -> torque off.
9. First teleop (M3), slow, hand on the 12 V switch, small motions first:
   ```bash
   uv run elrobot_deploy run --set limits.velocity_scale=0.3
   ```
   No workspace box by default; an optional one is `--set 'teleop.workspace_min=[...]'
   --set 'teleop.workspace_max=[...]'` (quote brackets: zsh would glob them).

### C. Tuning (whenever; it is configuration now)
10. Per-joint servo settings in `[executor]` (`p_coef`, `d_coef`, `friction`), from
   `executor-test` sine runs (section 4): light joints 5-7 P16 + friction 1.0 deg is the current
   best guess; the gravity joints 2-4 (and 1) still need their own runs.
11. Gravity compensation (sag ~ gravity torque / stiffness), squeeze margin vs force on a real
   object, TORQUE_LIMIT on servo 8.
12. If level, sideways gripper motion is needed: the wrist-branch question (`tcp_qp.py`
    docstring); a new IK is `--set ik.solver=file.py:factory` (`runtime/ik/__init__.py`).

### D. Cameras (done 2026-10-17, branch `claude/cameras`; `docs/architecture.md`, "Cameras")

The wrist fisheye (HX200S, USB 10bb:2b08, the same unit as g1d's wrists) runs as P2 in every
session, shown live on the dashboard; `elrobot_deploy camera list | probe | test` on the bench.
Measured on this Mac, in the room, with the camera plugged in:
- AVFoundation streams it at 1280x720, 30.1 fps, no drops (its MJPEG modes show as `420v`; raw
  `yuvs` manages 10 fps at 720p). OpenCV's macOS backend is NOT used (medium preset, scaled).
- **The macOS controls spike (PORTING_PLAN D1) is resolved**: UVC requests over libusb, without
  root and without claiming the interface, read and write every control WHILE AVFoundation
  streams (manual exposure at a quarter: frame mean 57 -> 2.9; handed back: 57.3).
- Lens fit at 720p: cx 635, cy 385, r 326 px (residual 1.3-1.9 px); the circle fits the sensor.
- **Exposure, "the camera's own settings first"**: its own auto-exposure held the disc at mean
  157 with 7 % clipped, all of it a lit monitor, with the workspace well exposed. Taking over
  with g1d's law (p99.5 at 245) saved the monitor and darkened the workspace to mean 48 at gain
  97, worse for the arm. So `camera_first` (the default) keeps the camera's settings unless its
  loop clearly fails for 2 s: 15 % of the disc clipped (`takeover_clip_pct`; g1d's rig: 26 %),
  or the disc's mean >= 190, or dark. Then it takes over (sticky) and says why.

Left for the cameras: **re-run the executor sine test with the camera streaming** (the camera
and the servo adapter share the VIA USB 2 hub: 0x0213... and 0x0211...; PORTING_PLAN section
6); `fallback_lens_circle = [635, 385, 326]` in the profile if a fit ever fails at start; the
camera-to-joint time offset (D4); an exocentric camera is another `[cameras.<name>]` table.

### E. Recording and the Astra harness (built 2026-10-17, branch `claude/astra-harness`;
`elrobot_deploy/docs/astra.md`)

- **P9, the recorder**, ported from g1d: one record per rollout under `logs/records/` (bus
  reads, targets, operator rows, every camera frame as JPEG). `run --record` for teleop.
- **The Astra harness**: GPT-6 Astra (Codex + MCP) drives the arm with `send_action`,
  `inspect_observation`, `finish`, the exp_011 design ported to one arm. **P12** replaces the
  phone source on the intervention ring (one action = one engagement), the harness writes the
  session folder from P9's record, `start_codex.sh` starts Codex on it. Decided with the user:
  a P9 like g1d's; no lag / sag in Astra's README; no workspace box (the bus guards are the
  safety); the exo camera is a second iPhone as a Continuity Camera (`[cameras.exo]`,
  `use = "astra"`, never in teleop); gripper 55 mm open (ruler), linear.
- **Tested in simulation end to end** (`tests/test_astra_session.py`); **not yet run with
  Codex (not installed on this Mac), with the iPhone streaming, or on the arm.** Order:
  `astra --sim --fake-cameras` + Codex, then `--dry-run`, then the arm slowly
  (`--set limits.velocity_scale=0.3`). docs/astra.md section 10 lists what is open.

Later: policies, from `legacy_g1d/`. Optional: Start / Reset buttons on the phone.

## 6. Minimum checklist before teleoperating with the phone

- [x] Single-arm schema; the executor runs as the launcher's bus process on the real ring
- [x] Phone source with clutch, anchoring, gripper, guards; room-frame mapping for the grip
- [x] P6 driving `tcp_qp` at 60 Hz; launcher teleop mode; end-to-end test in simulation
- [x] Hardware check at startup; Home / Start gates; dashboard with the URDF; dry-run mode
- [x] The test suite run on the Mac (2026-10-17: 164 passed, with the Astra harness)
- [ ] App v2 built and on the iPhone; `phone-bench` confirms the axis mapping
- [ ] `run --sim` driven by the real phone; latency measured (lead 40 ms enough?)
- [ ] `run --dry-run` on the real arm (check passes, measured arm matches reality)
- [ ] Home and exit poses checked on the real arm; `run --no-phone` on the real arm
- [ ] Executor per-joint P/friction set (or accept the defaults for a first slow session)
- [ ] Gripper squeeze margin and TORQUE_LIMIT checked on a real object
- [ ] First hardware session: low velocity_scale, hand on the 12 V switch

## 7. How to run things (from `ElRobot/elrobot_deploy/`)

```bash
uv sync                                                   # the whole env: IK, bus, cameras, dev tools
uv run pytest -q                                          # deploy tests (no hardware)
uv run elrobot_deploy config                              # print the resolved profile
uv run elrobot_deploy view                                # the URDF + poses in the browser
uv run elrobot_deploy phone-bench                         # the grip check, no robot
uv run elrobot_deploy run --sim                           # simulated servos
uv run elrobot_deploy fake-phone                          # drive a --sim run without the phone
uv run elrobot_deploy run --dry-run                       # real arm read, torque OFF (ghost only)
uv run elrobot_deploy run --record                        # ... and record every rollout (P9)
uv run elrobot_deploy astra --task "..." --sim --fake-cameras   # Astra drives (docs/astra.md)
../logs/astra/current/start_codex.sh                      # Codex on that session, once READY
uv run elrobot_deploy run --no-phone                      # real arm: check, home, hold, exit
uv run elrobot_deploy run                                 # real arm, phone teleop
#   dashboard: http://127.0.0.1:8080 ; run flags: --auto-home --auto-start --no-dashboard
#   cameras: --no-cameras --fake-cameras ; --set cameras.wrist.exposure="camera"
uv run elrobot_deploy camera list                         # what AVFoundation and USB see
uv run elrobot_deploy camera probe                        # UVC controls, before/while streaming
uv run elrobot_deploy camera test --seconds 20            # one camera as P2 runs it, + snapshots
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
- Bus guard stops (session 2 on the arm): the old rules (20 deg off at any instant; 8 deg
  behind for 0.5 s) tripped on a lagging joint 5 and on the gripper opening during homing.
  Replaced by `[safety]` (`runtime/execution/guard.py`): runaway = far off AND moving away;
  blocked = far behind AND not moving for 1.5 s (never the gripper); both recover: straight
  to rest -> torque off (`safety.recover_via_home` to go through home first). Tune with `--set safety.<key>=...`.
- One Ctrl-C under `uv run` reaches the process twice; two within 0.5 s count as one. A real
  second Ctrl-C (later) skips the exit ramp and leaves the arm held.
- The clutch only counts if pressed after the rollout starts (and after any forced release):
  the phone shows "RELEASED" until the thumb is lifted.
- The `VIRTUAL_ENV ... will be ignored` warning from uv is harmless.
- `legacy_g1d/` is excluded from ruff and pytest; don't "fix" it, port from it.
- User preferences: explain before acting on hardware; never command motion without the user's go.
