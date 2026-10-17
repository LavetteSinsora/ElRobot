# Porting g1d_deploy to ElRobot: the plan

`elrobot_deploy/` is a verbatim copy of `g1d_deploy` (see `UPSTREAM.md`). This
page says what has to change to run it on the ElRobot follower: one 7-DoF arm
plus a gripper, all ST3215 servos on one serial bus, iPhone teleoperation, a
210° fisheye wrist camera, and no robot board (everything runs on the Mac).

Paths below are relative to `src/g1d_deploy/` (now `src/elrobot_deploy/`) unless they start
with `elrobot_deploy/` or `ElRobotRepo/`.

**Status (2026-10-17, second session):** workstreams 0, 1 and most of C are done; the system
map of what exists now is `docs/architecture.md`.
- **Done: prune (0) and the single-arm schema (1).** The g1d code with no ElRobot counterpart,
  and the subsystems not ported yet (P5 / policy contract / chunk blending, P9, P10 and
  inspection, P2 and the cameras, CHECK), moved unchanged to `legacy_g1d/`. One 8-joint target
  row (the gripper in it), the bus state ring, an operator ring; `robot/profiles/elrobot.toml`
  with limits from the URDF x `velocity_scale`; `SimDriver` as the simulated bus.
- **Done: the executor as the launcher's bus process (A).** P7 on the real arena, started at
  CHECK and kept for the whole session; `bus_arena.py` is gone. Per-joint P / D / friction /
  torque limit are profile keys (`[executor]`), so the tuning in HANDOFF.md 5A is configuration.
- **Done: the IK behind one interface (B)**, `runtime/ik/__init__.py`: `solve`, `fk`,
  `set_nominal`; `tcp_qp` shipped, any `module:factory` pluggable.
- **Done: C1 (app v2), C2 (phone source P11), C3 (P6), C4 (the launcher's teleop mode), C5
  (tests, including the whole chain end to end over UDP on simulated servos)**, with the changes
  listed below. Not yet run with the real phone or on the real arm (M2/M3).
- **Next:** build the app; `phone-bench` with the real phone; `run --sim` driven by the real
  phone (M2); then the real arm (M3). HANDOFF.md section 6 is the checklist.

**What changed against the plan below, and why (decided 2026-10-17):**
- **Open decision 1 is decided: ROOM frame, and the grip changed.** The phone is held with the
  screen facing left, the back camera right and the **top pointing forward** (not up). Only
  motion relative to the clutch press counts; "forward" is where the phone's top points
  (horizontally) at the press, "up" is gravity. Rotations are applied in the same room axes,
  about the TCP. (`runtime/teleop/intervention.py`, `RoomFrame`.)
- **The operator ring carries motion, not a pose.** P11 writes the TCP's motion since the
  press, in base axes; P6, which owns the model, freezes the TCP pose (`fk` of the command the
  bus process holds) at the same counter change and composes. So the phone source needs no
  model of the arm, and there is no flange anywhere (C2's "anchor on the TCP" is P6's).
- **Ticks never rewind.** A reset does not restart the bus process (that would release torque),
  so `k_exec` counts up across rollouts and P0's ramps anchor on `k_exec + 2`.
- **Torque is released only after a completed exit ramp**; any failure, fault or a second
  Ctrl-C ends the session with the arm held (g1d's damp has no safe ElRobot equivalent: torque
  off drops the arm).
- **One-thumb pad on the phone**: anywhere engages, the forward zone also grips (two separate
  hold buttons cannot be held with one thumb).
- **GUIDE_SLACK is 0.45 of a frame** (not exactly half: a stream at twice the grid rate sits on
  that edge); the velocity clamp spans the deadline difference (C3), at least one grid period.
- **The seqlock has real memory fences** on ARM (Apple Silicon), which g1d's (x86) did not need.

---

## 1. Decisions already made

| topic | decision |
|---|---|
| rate | everything at **60 Hz**: `control_hz = 60`, so `dt = 16.7 ms`, matching ARKit's frame rate |
| servo shaping | the shipped critically damped filter (`CritDamped`) at **`filter_hz = 15`**, with `filter_lead = true` (look-ahead `1/(π·15)` ≈ 21 ms) |
| teleop lead | **`intervene_lead_s = 0.040`** by default |
| deadlines | stamped from the **phone's capture time** (`ARFrame.timestamp`), mapped onto the Mac's clock by `intervention.Pace`. The mapping is pinned when the first sample of an engagement arrives, so transport latency does not accumulate |
| re-pinning | **`Pace` re-pins on every clutch press.** Each press bumps the intervention counter (`control.intervene`), which also preempts P7 and resets P6's grid |
| teleop mode | **reuse the intervention path with no policy at all** (not a "hold" policy). Released: P6 has no chunk, so it idles and P7 holds the last *commanded* pose. Engaged: the operator drives. The policy-plus-intervention mode stays available for DAgger later |
| pose mapping | **the phone's pose drives the robot's virtual TCP**: the midpoint of the far ends of the two gripper fingers. Relative to the clutch anchor, tool-frame as in `intervention.Anchor`. On the robot side the anchor is the TCP, not the flange, so rotating the phone in place rotates the gripper about its fingertips (C2) |
| starting teleop | the arm moves **only while the phone's engage button is held** (the clutch). Nothing moves before the first press, and a button already held when the console starts doesn't count until it has been seen released (`Console`'s arming rule) |
| gripper | **hold-to-close** with a maintained target width: while the close button is held, the target closes at a fixed rate; when released, it opens at the same rate. Integrated on device time, not per packet. Details in C2 |
| exposure | **sticky**: change exposure and gain only when a real, sustained need exists, then in one decisive step (section 6) |
| control box | **this Mac**, for now |

Feasibility numbers from the user's bus tests: sync_read of all positions takes
1.40 ms median; sync_read plus sync_write takes 1.67 ms median (1.75 ms p99).

---

## 2. Open decisions

1. **Phone axes → gripper axes (tool-frame or room-frame: decide later).**
   The grip is settled. Start with the phone upright in portrait, screen
   facing the operator. Then roll it 90° about its own long axis so that:
   - the top still points up;
   - the **screen faces left**;
   - the back camera (the one ARKit tracks with) looks to the operator's
     right.

   In that pose, moving the phone forward moves the TCP forward, left moves it
   left, and so on.

   ARKit's camera frame is fixed to the device:
   - +X runs along the long axis from the top (front camera) toward the
     bottom;
   - +Z points out of the screen;
   - +Y completes a right-handed frame.

   For this grip that gives:

   | room direction | camera axis |
   |---|---|
   | forward | −Y |
   | left | +Z |
   | up | −X |

   `hand_tcp`'s rotation maps these onto the TCP's forward/left/up axes. Read
   those from the calibrated URDF; don't assume them. Its translation is zero.
   The bench tool confirms the signs before anything moves: push forward and
   "forward" must go positive.

   One practical point: ARKit tracks with the back camera, so whatever is to
   the operator's right should be textured and static. A moving robot arm
   filling that view can degrade tracking.

   With the gripper pointing forward and level, tool-frame and room-frame
   give the same result. They differ once the gripper is pitched or yawed:
   - **tool-frame:** "forward" means along the jaws;
   - **room-frame:** "forward" means away from the operator.
2. **Grasp behaviour details** (C2, "Gripper"): the grip-force number, and
   whether holding the button for a long carry is acceptable or a
   latch-on-contact is wanted.
3. **Two-way `Pace`.** Let the mapping slide back slowly after a latency spike,
   instead of only ever sliding later. Optional; re-pinning per clutch already
   bounds how long a spike's extra lag lasts.
4. **Optional phone buttons for START and RESET** (the episode boundaries), so
   recording can be driven from the phone.

---

## 3. Workstreams

### 0. Rename and prune (first; mechanical)

- Rename the package `g1d_deploy` → `elrobot_deploy`. This covers imports, the
  CLI entry point, the `g1d-` arena prefix, `G1D_*` environment variables, the
  docs, and the tests.
- Delete what has no ElRobot counterpart:
  - `runtime/drivers/dds/` and the `cyclonedds` dependency;
  - `runtime/drivers/board/image_server_launch.py`, `paramiko`, and the SSH and
    teleimager handling;
  - `runtime/teleop/pico_service.py` and `scripts/build_xrobotoolkit.sh`;
  - the G1-D and Dex-1 assets and profiles;
  - `runtime/ik/waist_qp.py`, `waist_planner.py` and `arm_planner.py`'s
    dual-arm specifics;
  - the pinocchio gravity compensation (the ST3215 takes position only, so
    there is no torque channel).
- **Keep but don't launch in teleop mode:** P5, `policy_contract/`, chunk
  blending, and real-time chunking. They are needed later for policies and
  DAgger.
- **Keep `fisheye_image_server/photometry.py`** (the exposure loop and the
  lens-circle fit) and the capture logic in `publisher.py`. They move into a
  local camera driver (workstream D).
- Exit criterion: `uv run pytest` is green on the Mac after the deletions,
  with tests for deleted modules removed.

### 1. Single-arm schema and robot profile (unblocks everything else)

- `runtime/shared_memory/schema.py` changes from two arms plus a waist to one
  arm:

  | constant | now | becomes |
  |---|---|---|
  | `N_HANDS` | 2 | 1 |
  | `ARM_DOF` | 14 | 7 |
  | `WAIST_DOF` | 3 | 0 |
  | `BODY_DOF` | 17 | 8 (7 arm joints + gripper) |

  These constants are used in 28 source files and 31 test files.
- **The gripper joins the body target row.** It is servo 8 on the same bus, at
  the same rate. So drop the separate hand ring and the hand servo P8: one row
  carries 8 joints under one deadline, and one segment walk executes them.
  (Revisit only if a different gripper ever needs a different rate.)
- Write a new robot profile, `elrobot.toml`, and make limits come from the
  URDF instead of being hard-coded:
  - position limits from `<limit lower/upper>`;
  - velocity limits from `<limit velocity>` times a `velocity_scale` set in the
    profile.

  The ElRobot URDF says 4.71 rad/s for every joint. That is about the ST3215's
  no-load speed at 12 V, so start the scale around 0.5 and measure. (g1d
  deliberately ignored the URDF here, `g1d_base.toml [limits]`, because the
  G1-D's URDF numbers were 22–37 rad/s motor maxima. For ElRobot the URDF is
  the right source once it is calibrated.)
- Add a **simulated bus**: the robot model integrating commanded positions
  through a first-order lag, with the same interface as the real bus.
  `ElRobotRepo/tests/fake_bus.py` is a starting point. Use it to run the whole
  teleop chain before the real bus process exists.

### A. Low-level execution (deferred), but its *shape* is decided now

**The structural difference from g1d: one half-duplex serial bus.** g1d had
separate DDS topics for state (`rt/lowstate`, read by P1) and commands
(`rt/lowcmd`, written by P7). On ElRobot, reads and writes share one wire, so
**one process must own the port.** P1, P7 and P8 merge into a single **bus
process**:

- **Every tick:** take the segment walk and the filter from P7, unchanged
  (`execution/servo_segment.py`, `servo_filter.py`), then sync_write the 8 goal
  positions.
- **Every Nth tick:** sync_read positions, plus speed and load at a lower rate,
  and write the state ring as P1 did. Forward kinematics for the flange/TCP
  pose can stay in this process or move to a small reader.
- **Rate:** sync_read plus sync_write is 1.67 ms median and 1.75 ms p99,
  inside a 2 ms period but with no margin. Options: write at 500 Hz and read
  every 2nd–4th tick, or run a 400 Hz loop.
- **ST3215 settings:**
  - set the acceleration and goal-speed registers so the servo's own profile
    doesn't lag behind the stream (verify what 0 means on the STS3215);
  - leave `TORQUE_LIMIT` as a hardware bound, especially on the gripper.
- **No `dq` or `tau` on the wire:** drop the velocity and torque feedforward
  from the servo.
- The rest of the system only sees the target ring and the state ring, so
  until this exists the simulated bus (workstream 1) stands in for it.

### B. IK and limits (deferred until the URDF is calibrated)

- One-arm `arm_qp`: a single frame task at the TCP, joint position and velocity
  limits from the URDF, and a self-collision limit only if the arm can actually
  hit its base.
- Add a **weak posture task toward a nominal pose**, not only toward the seed.
  Over long teleop sessions the 7-DoF null space otherwise drifts toward joint
  limits.
- Development can start on the upstream URDF
  (`ElRobotRepo/description/elrobot_upstream/`). Calibration changes accuracy,
  not code.

### C. Teleoperation (now)

**C1. iPhone app (`ElRobotRepo/phone_teleop/ios`)**, packet `v: 2`:

- Keep `t` (the ARFrame capture time; this is the device time `Pace` uses),
  `seq`, `pose`, `track`, and `engage`.
- Add a `world_epoch` counter that increments on "Reset world origin" and when
  ARKit reports relocalization. The receiver force-disengages when it changes.
- Keep `engage` (the clutch) and `grip` (hold-to-close) as hold buttons,
  sent on every packet.
- Keep `t_send` and the echo. Add the receiver's arrival time to the echo so
  the Mac can estimate one-way latency (NTP-style) and log `Lₙ − L₁`.

**C2. Phone source process (replaces P11's PICO source).**

- Keep `PicoLoop`'s structure (`runtime/processes/p11_pico.py`): `Console`,
  `Anchor`, `Gripper`, `Pace`, stamping the counter last, `release()`.
- Replace `PicoSource` with a **blocking UDP receive loop**. Stamp the arrival
  time when each datagram arrives and write the row immediately, so there is no
  180 Hz polling delay.
- Convert the pose: a row-major 4×4 matrix to `[x y z qw qx qy qz]`, at that
  one boundary only, as `_pose7` does.
- `hand_tcp` for the phone is a fixed rotation (open decision 1, for the
  upright, screen-left grip described there) with zero translation. Verify it with the
  bench tool below.
- **Anchor the robot side on the TCP, not the flange.** At the press, freeze
  the TCP pose (`flange ∘ tcp`, with the TCP from the robot file).
  `Anchor.follow` then moves the TCP, and the result is converted back to a
  flange pose (`tcp_pose ∘ tcp⁻¹`) before it goes into the row. P6 and the IK
  keep consuming flange poses, so nothing downstream changes.
- **Gripper (hold-to-close).** Adapt `intervention.Gripper`. The behaviour
  the operator sees:
  - **hold `grip`** to close; keep holding to keep grasping;
  - **release `grip`** to open, at the same rate;
  - **while the clutch is released**, the width holds and the grip button is
    ignored, so repositioning your hand never drops the object.

  **Why a position-commanded jaw needs a force rule.** The ST3215 is a
  position servo. Pressing on an object, its torque is roughly
  `P_COEF × (measured − commanded)`. If the command is integrated all the way
  to the closed stop while a 5 mm object holds the jaw open, the position
  error is the whole remaining travel and the servo pushes as hard as it can.
  That is how things get crushed. So **the grip force is set by how far the
  command may run ahead of the measured jaw, never by the command alone.**
  Four layers, outermost first:

  1. **Hardware torque cap.** Set `TORQUE_LIMIT` (register 48, in 0.1% of
     stall torque) on servo 8 to the most force the gripper should ever apply.
     It holds even if the software is wrong. Set it in CHECK and read it back.
  2. **Squeeze bound: this sets the holding force.** The jaw command may sit
     at most `squeeze_margin` closer to closed than the *measured* jaw:
     `cmd = max(cmd, measured − squeeze_margin)`.
     - In free air the jaw keeps up with the command, so the bound never
       binds and closing is unaffected;
     - on contact the jaw stops, the bound binds, and the servo holds a
       steady force of about `P_COEF × squeeze_margin`, whatever the
       object's width;
     - if the object slips or compresses, the measurement moves and the
       command follows. The grip re-closes onto it at the same force instead
       of letting go.

     **Enforce it in the bus process every tick,** using the freshest
     position read, not only in the phone process at 60 Hz. The bus process
     is the last thing before the wire.
  3. **Anti-windup:** store the bounded command back into the integrator
     (the phone process applies the same rule, using the state ring). Then
     releasing `grip` opens immediately from the bounded command, not from
     a wound-up "fully closed". This also fixes a g1d bug: `Gripper.hold()`
     (`intervention.py:480`) returns the wound-up value *without* the bound,
     so in g1d, letting go of one hand's trigger while the other hand is
     still engaged commands that jaw fully shut. (A one-arm rig doesn't
     reach `hold()`, but fix it at the source.)
  4. **Position limits:** clamp the command to the calibrated travel. The
     lower end is "fingers just touching", not a hard press past it; closing
     on nothing then meets the same squeeze bound when the fingers touch.

  Rates and inputs:
  - the target moves at `gripper_rate` per second of **device time**
    (`ARFrame` timestamps), not a fixed step per packet, so the speed doesn't
    depend on packet rate or WiFi bursts. Cap each step's dt, as g1d does;
  - on each clutch press, re-anchor on the *commanded* width (as g1d does),
    so an existing grasp keeps its force.

  **Watch the servo's overload protection.** The ST3215 cuts its own torque
  to `PROTECTIVE_TORQUE` when the load stays above `OVERLOAD_TORQUE` for
  `PROTECTION_TIME` (registers 34–36). A grasp held near stall for a long
  carry can trip it: the grip suddenly weakens and the object drops. Keep
  the holding force (squeeze margin) well under the overload threshold, and
  watch temperature (`PRESENT_TEMPERATURE`) on long holds.

  **Calibration before first use:**
  - **re-measure the gripper's range.** `calibration/ranges.json` currently
    flags joint 8 as failed: 2.2° of travel against the URDF's 126°;
  - **map squeeze margin to force.** Close on a rigid test object at several
    margins, logging `PRESENT_LOAD` and `PRESENT_CURRENT` (and ideally a
    scale under the fingers). Pick a default margin that holds typical
    objects without marking soft ones.

  Later options, if wanted: a phone setting for two or three grip strengths
  (different margins); latch-on-contact, so a long carry doesn't need the
  button held; and slip detection from the load reading.
- **Guards** (new; the PICO's tracking didn't need them, ARKit's does):
  - `track != "normal"` counts as stale, and the clutch is force-released;
  - a change in `world_epoch` force-releases;
  - per-sample jump guard: a step larger than a few cm (or degrees) in one
    frame force-releases;
  - a workspace box clamps the target pose.

  A forced release always requires a fresh press, with the existing arming
  rule in `Console`.
- **Stale stream in teleop mode:** release, which means hold; there is no
  policy to hand back to.
- Port the bench tool (`p11_pico.main()`): with no robot, print the operator's
  displacement in tool axes, to check `hand_tcp` before anything moves.

**C3. P6 (`runtime/processes/p6_target.py`).**

- At 60 Hz rows on a 60 Hz grid, `_guided`'s decimation selects every row.
  Keep the grid logic, which still handles a phone that drops to 30 fps.
  Revisit `GUIDE_SLACK_S`: it is 5 ms against a 16.7 ms grid, while the rule's
  intent is half a frame (8.3 ms).
- In guided mode, base the velocity clamp's span on the **deadline
  difference** to the seed rather than `(k − seed.k)·dt`. That way a late or
  dropped phone frame can't license too fast or too slow a step.
- `_switch` and seeding are unchanged.

**C4. Teleop launch mode.**

- Add a `--teleop` launch flag in `cli/main.py` and `runtime/launcher/state_machine.py`.
  It skips P5 everywhere P0 waits on it:
  - the spawn at CHECK (`state_machine.py:317`);
  - the warm-up wait (`:432`);
  - the settings and prompt handshake at START (`:504`, `:532`);
  - the first inference at ACTIVE.
- Allow the filter under teleop:
  `config_loading/rollout_settings.py:286` currently refuses any non-linear
  servo with `--intervene`.
- Processes in teleop mode: **P0, the bus process, P6, the phone source**,
  plus P2 (camera), P9 (record) and P10 (dashboard) when wanted.
- START and RESET stay the episode boundaries, so each demonstration is one
  rollout directory.

**C5. Tests.**

- Port `tests/test_intervene*.py` to the new source and packet.
- Add tests for:
  - each guard;
  - re-pinning per clutch;
  - the deadline-based clamp span;
  - an end-to-end run with no hardware: recorded phone packets replayed over
    UDP, then the phone source, then P6, then the simulated bus.

### D. Cameras on the Mac (section 6 has the reasoning)

1. **Spike first: frames *and* exposure/gain control on macOS.**

   Why this needs a spike: the camera is a standard USB video (UVC) device.
   UVC defines two things over one USB connection: a **video stream** (the
   MJPEG frames) and **controls** (exposure, gain, white balance, ...). On
   Linux, the kernel's V4L2 API ("Video4Linux2") exposes both: OpenCV's
   `CAP_V4L2` reads frames through it, and the same device node takes control
   writes. g1d's `V4l2Camera` (`runtime/drivers/cameras.py:172`) relies on
   that, so it doesn't run on macOS. macOS's camera framework, AVFoundation,
   gives you the frames, but it doesn't expose exposure or gain controls for
   external USB cameras. Reading frames is easy; setting exposure needs a
   second route. Candidates, in order of preference:
   - (a) **pyuvc (libuvc)**, the library the board server already uses. It
     talks to the camera directly over USB and does both (raw MJPEG frames
     and controls) in one process, exactly as on the board. On recent macOS,
     libusb usually has to take the camera away from the system's camera
     driver, which tends to require running as root. Check this on the Mac.
   - (b) **AVFoundation for frames** (e.g. OpenCV's `CAP_AVFOUNDATION`, which
     returns decoded frames only), **plus UVC control requests sent over
     IOKit** for exposure and gain, while the stream runs. Tools such as
     `uvc-util` and the CameraController app work this way. Confirm it works
     on this macOS version.

   Both routes stream and set exposure **at the same time, in the same
   process.** Nothing forbids doing both at once; the question is only which
   API can do the control half on macOS.
2. A local camera process combining publisher, exposure loop and crop:
   - capture, then write the JPEG and the cropped frame into the ring;
   - the exposure loop (`photometry.ServoRunner`) runs in the same process,
     every 3rd frame at half scale, as on the board;
   - no ZMQ hop, no header, no SSH.
3. **Lens fit at INIT.** This is about two *processes*, not about frames
   versus controls. A local USB camera's video stream can be opened by only
   one process at a time. On the board, the image server owned the camera
   and any number of processes could subscribe to its ZMQ stream, so g1d's P0
   could fit the lens while P2 was also receiving frames. Here, P0 opens the
   camera, fits, and closes it; then P2 opens it. Alternatively, P2 does the
   fit itself, with the ring sized to the largest plausible square and the
   fitted crop recorded as metadata.
4. Re-measure the camera-to-joint time offset on the Mac.

### E. Recording, check phase, dashboard (after teleop works)

- **P9 recording:** mostly unchanged once the schema is single-arm.
- **CHECK:**
  - keep the phone liveness check (P11's `_verify_pico`, rewritten for UDP)
    and the lens fit;
  - drop the robot subnet, board, DDS, and PICO-service checks;
  - add bus checks: every servo ID responds, the mode is position, and torque
    and acceleration settings read back as configured.
- **P10 dashboard:** strip the G1-D scene and the waist, then render the
  ElRobot URDF.

---

## 4. Milestones

| # | deliverable | depends on |
|---|---|---|
| M0 | rename and prune; tests green on the Mac | — |
| M1 | single-arm schema, `elrobot.toml` with URDF limits, simulated bus | M0 |
| M2 | `--teleop` mode end-to-end in simulation, driven by the real iPhone (C1–C5) | M1 |
| M3 | bus process on the real ST3215s (A); first real teleop | M2 |
| M4 | fisheye camera on the Mac (D) | M0 (independent of M2/M3) |
| M5 | recording, CHECK, dashboard (E) | M2–M4 |
| — | IK accuracy and limits from the calibrated URDF (B) | URDF calibration |

M4 can run in parallel with M2 and M3.

---

## 5. Teleop timing reference (60 Hz, lead 40 ms, filter 15 Hz)

- **Capture to arm:** about `L₁ + 40 ms + servo tracking lag`, where L₁ is
  the first sample's transport time in that engagement. It is fixed within an
  engagement and re-pinned on every clutch press.
- **Margin for sample n:** `40 + L₁ − Lₙ` ms. A sample arriving later than
  that makes `Pace` slide later, adding lag until the next clutch.
- **Filter look-ahead available:** about `40 − (up to 16.7) − (Lₙ − L₁)` ms,
  against the 21 ms a 15 Hz filter needs. That leaves about 2 ms of transport
  jitter before the filter adds a little lag. It stays correct either way.
- **What to log from the first session:** `Lₙ − L₁` (p50/p99) and the filter's
  actual look-ahead. Then set the lead from data.

---

## 6. Cameras without a robot board: impact and improvements

**What goes away.**
- The board, SSH, the vendor's teleimager conflict, keeping code in sync on the
  board, and the ZMQ hop.
- Frames go straight from USB into this machine's memory: one fewer network
  hop of latency and jitter, and one fewer place to drop frames silently.
- The 16-byte sequence-number header can go: a local capture loop knows its
  own frame count.

**What gets harder.**
- **macOS has no V4L2.** Reading frames is easy; setting exposure and gain
  needs pyuvc or IOKit (D1). That is the main risk of this workstream, hence
  the spike.
- **One process per camera stream.** The startup lens fit and the capture
  process can't both open the camera at once, so they take turns (D3).
  Streaming and changing exposure at the same time is fine.
- **USB contention with the servo bus.** A 1280×720 MJPEG stream reserves
  isochronous bandwidth. If the camera and the serial adapter share a USB 2
  hub, the serial adapter's bulk transfers can see extra latency, which is the
  thing the 500 Hz loop is sensitive to. Use separate ports or controllers, and
  **re-run the servo latency benchmark with the camera streaming.**
- **CPU is no longer the constraint.** The board's budget (Orin NX, 2.3 ms per
  half-scale decode) is irrelevant on Apple silicon, so the exposure loop's
  10 Hz rate is not a cost question here. It was chosen because the plant
  settles in 2 frames, so a 100 ms sample measures a settled response. Keep it.

**Exposure: sticky (decided).** The current loop (`photometry.step`, mode
`servo`) holds the 99.5th-percentile brightness inside the lens circle (p99.5)
just under clipping. Its deadband is 235–247. It reacts on the very next 10 Hz
tick and moves at most 1.6× per tick. On a moving wrist camera that is too
eager: a lamp or window crossing the 210° view moves p99.5, the loop cuts
exposure, and the rest of the scene darkens mid-demo, then brightens again
when the lamp leaves.

New mode **`sticky`**, next to `servo`, `auto` and `pinned`. It keeps the
measurement and the existing control law, and adds **when to act**:

1. **Hold by default.** Exposure and gain stay where they are while the image
   is acceptable. "Acceptable" is much wider than `servo`'s deadband: no
   meaningful clipping, and not badly dark.
2. **Act only on a sustained need,** with different patience per direction,
   because overexposure loses information and darkness doesn't:
   - **too bright:** p99.5 ≥ 250 *and* clipped fraction ≥ ~1% for ~0.5 s.
     The 1% condition ignores a small lamp or reflection;
   - **too dark:** p99.5 < ~200 for ~2 s;
   - **panic:** clipped fraction ≥ ~10% for ~0.2 s.

   Any acceptable tick resets the timer, so a lamp swept briefly through the
   view changes nothing.
3. **When it acts, it corrects decisively,** then sticks again. Run the
   existing `servo` law until p99.5 is back in its tight 235–247 band. That
   takes a few ticks and approaches from below, so it can't overshoot into
   clipping. Then return to holding, with a ~1–2 s cooldown before the next
   correction (panic excepted). One correction is one short, visible
   transition, not a continuous drift.
4. **Record exposure and gain with every frame** (or every change, with a
   timestamp), so training data can detect any step.

The thresholds above are first guesses, set from the knee table in
`photometry.py`: clipping rises sharply once p99.5 passes 250. Tune them from
recordings. Two other modes are possible: `pinned` (fixed manual controls)
for no adaptation at all, and freezing `sticky` for the length of an episode
for the strictest consistency.

**Lens circle and crop:** port as-is (`runtime/drivers/fisheye_circle_fit.py`,
`photometry.fit_circle`). The fit is per launch, finds the rim by its
brightness drop (so it doesn't depend on exposure), and falls back to
`fallback_lens_circle`. For the 210° lens, check that the whole image circle
fits on the sensor: the G1-D's circles overflowed the top of a 720-row sensor,
and the crop already allows for that. Re-measure the fallback circle for this
unit.

---

## 7. Real-time on the Mac (the control box for now)

- `cpu_pinning` and `rt_priority` do nothing on macOS (`runtime/cpu_pinning.py`:
  there is no `sched_setaffinity` or `SCHED_FIFO`). Only the hybrid
  sleep-then-spin wait in `runtime/timing.py` remains.
- Expect occasional jitter of a few ms in the bus loop. Measure it: the
  command log records every publish time.
- Children are started with `spawn`, and the arena name `g1d-<pid>` fits
  macOS's 31-character limit on POSIX shared-memory names. Keep the renamed
  prefix short.
- If jitter shows up in tracking, move the bus process (and possibly the whole
  stack) to a small Linux box. That is the setup g1d's pinning and
  `SCHED_FIFO` were built for.
