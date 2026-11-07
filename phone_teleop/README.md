# phone_teleop: the iPhone app (ElRobot), as the controller or as a camera

One app, installed on any number of iPhones. Each phone takes one **role**, chosen on first
launch and remembered (Settings changes it):

- **Controller**, in your hand. ARKit world tracking sends one UDP packet per camera frame
  (about 60 Hz) to the Mac: a timestamp, the 4x4 pose `T_world_phone`, the tracking state, a
  world-frame counter, two buttons and the phone's name. On the Mac, `elrobot_deploy run`
  receives them (its phone source, P11,
  `elrobot_deploy/src/elrobot_deploy/runtime/processes/p11_phone.py`) and echoes back what the
  arm is doing, which the app shows (and buzzes on).
- **Camera**, on a stand. It streams its back camera to the Mac as one of the profile's
  `[cameras.<name>]` (`source = "phone"`,
  `elrobot_deploy/src/elrobot_deploy/runtime/cameras/phone_camera.py`).

Each phone also has a **name** (`hand`, `exo`, `side`...). Nobody types an IP address: the Mac
advertises itself over Bonjour, and the Controller lists Macs by their computer name. The Mac
finds each camera phone by its name. The phones need no Apple ID in common with the Mac.

## How to hold it, and drive (Controller)

Hold the phone in your right hand with the **screen facing left**, the back camera facing
right, and the **top (Dynamic Island) pointing forward**, your thumb on the screen.

- **Hold ENGAGE** (0.4 s: a bar fills, the phone taps your hand) to start. The arm's gripper tip
  then copies the phone's motion **relative to where the phone was at that moment**: move the
  phone forward and the arm moves forward; left, left; up, up. Tilting the phone's top down
  pitches the gripper down; turning it left turns the gripper left.
- **Tap ENGAGE** to stop: the arm holds where it is. Starting takes a deliberate hold, stopping
  one tap. Re-engage anywhere and continue: that is how you move further than your arm reaches,
  or re-orient.
- **Hold GRIP** to close the jaw (about 0.7 s fully); **let go** and it opens again at the
  same speed. While disengaged the jaw holds.
- The app switches ENGAGE off by itself whenever the Mac releases the arm (and buzzes), stops
  answering, or says another phone drives. It never shows ENGAGED while the arm is not
  following. Careful: while engaged, a dropped phone drags the arm until the jump guard catches
  it; disengage before putting the phone down.

Built against mispresses:
- ENGAGE is greyed out, and says why, until the Mac is ready and tracking is normal.
- A touch that slides off ENGAGE before the hold completes does nothing.
- The only other control on the screen is the gear, which opens Settings (the Mac, Reset world
  origin, the role) on a 0.6 s hold. It is locked while engaged.
- Edge swipes (home, Control Centre) need a second swipe, so a stray thumb does not leave the
  app.

Only the heading of the phone at the press matters ("forward" is where its top points,
horizontally); you need not hold it level. ARKit tracks with the back camera, so whatever is
to your right should be textured and static (not the moving arm).

**The Mac releases the arm on its own** (the banner turns red, the phone buzzes, ENGAGE goes
off; hold it again to continue) when:
- tracking is not "normal"
- the world origin changes (Reset world origin, relocalization)
- the phone jumps more than 5 cm or 30 degrees in one frame
- no packet arrives for 250 ms
- the phone's top points straight up or down at the press

## Several phones

- **Several Macs:** the Controller lists them. Tap yours once; it is remembered by name and found
  again even if its IP changes. With only one Mac on the WiFi, it connects without asking.
- **Several Controllers:** only one drives. The Mac takes the first phone that sends. Any other
  Controller shows "ANOTHER PHONE is driving" and its ENGAGE stays off. It can take over only
  once the driving phone has sent nothing for 1 s (closed, locked, out of the WiFi), so two
  phones never drive at once.
- **Several cameras:** give each its own name (`exo`, `side`), and the profile one table each,
  with `device` set to that name:
  ```toml
  [cameras.exo]
  source = "phone"
  device = "exo"
  ...
  [cameras.side]
  source = "phone"
  device = "side"
  ...
  ```
  `device = "auto"` (the default) works only while exactly one camera phone is on.

## Packet v2 (UTF-8 JSON, one datagram)
```json
{"v":2, "seq":1234, "t":5123.456, "t_send":5123.470,
 "pose":[r00,r01,r02,tx, r10,r11,r12,ty, r20,r21,r22,tz, 0,0,0,1],
 "track":"normal", "world_epoch":3, "buttons":{"engage":false, "grip":false},
 "name":"hand"}
```
- `t` is the ARFrame capture time on the phone's clock. The deadlines on the Mac are this
  plus a fixed lead (40 ms), re-pinned at every press, so the motion replays at the pace it was
  made, whatever bursts WiFi delivers it in.
- `pose` is row-major and in metres. The world frame is fixed at app start (or at "Reset world
  origin"), with +Y pointing up against gravity.
- `world_epoch` is bumped whenever the world frame may have changed (reset, relocalization,
  an interruption). The Mac releases the arm when it changes. A v1 app (without it) still works.
- `name` is the phone's name, for the Mac's logs and the busy echo (optional).
- Receivers ignore keys they don't know.

The Mac's echo: `{"seq", "t_send", "t_mac", "state", "why", "intervene"}`. `state` is one of:
- `inactive`: the arm is not ready
- `idle`: ready, hold ENGAGE
- `engaged`
- `released`: a guard let go; `why` says which
- `busy`: another phone drives; `why` names it

## Build and install (needs a Mac with Xcode; a free Apple ID is enough)
1. `brew install xcodegen`, then `cd ios && xcodegen`. Open `PhoneTeleop.xcodeproj` (the
   project; the app on the phone is called **ElRobot**).
2. Xcode → Settings → Accounts: sign in with your Apple ID. `project.yml` names the team
   (`DEVELOPMENT_TEAM`). For another team, change it there, or pick yours under Target →
   Signing & Capabilities. If the bundle ID is taken, change it.
3. Connect each iPhone by USB and tap "Trust". On the iPhone, turn on Settings → Privacy &
   Security → **Developer Mode** (the phone restarts). The phone's own Apple ID does not matter:
   the app is signed with yours.
4. Choose the iPhone as the run destination and press ⌘R. The first time, the iPhone may ask
   you to trust the developer under Settings → General → VPN & Device Management.
5. On first launch, choose the role and a name, and allow **Camera** and **Local Network**
   access.
6. A free-team build expires after 7 days. To renew it, run it again from Xcode.

This app replaces the old PhoneTeleop app in place (same bundle ID): an updated phone opens on
the role picker once.

## Run (from `ElRobot/elrobot_deploy/`)

```bash
uv run elrobot_deploy phone-bench        # 1. no robot: check the grip
uv run elrobot_deploy run --sim          # 2. the whole chain on simulated servos
uv run elrobot_deploy run                # 3. the real arm
```

`run` checks the hardware, then waits. Open the dashboard (http://127.0.0.1:8080), press
**Home** (the arm moves to the green ghost), then **Start**. Then hold ENGAGE on the phone.

The Controller finds the Mac by itself: `[P11] on the LAN as '<name>'` says under which name.
If macOS asks whether Python may accept incoming connections, allow it. The phones and the Mac
must be on the same Wi-Fi network; a guest network with client isolation blocks the traffic. If
the Mac is not listed (Bonjour blocked on that network), enter its IP in Settings (hold the
gear).

In `phone-bench`, hold the phone as above, hold ENGAGE and push the phone forward: `fwd` must
grow (left: `left` grows; up: `up` grows; tip the top down: `rot about left` goes positive).

`receiver/receiver.py` is a dependency-free link test (it only echoes). It does not advertise
itself: enter the Mac's IP in Settings.

## Camera role

Put the phone on its stand in landscape, plugged in, the app in front (iOS stops the camera of
an app in the background).
- The preview shows exactly the frame the Mac receives. If it is upside down there, it is
  upside down on the Mac: turn the phone over, or set `phone_rotate = 180` in the profile.
- `uv run elrobot_deploy camera list` lists the camera phones it can see.
  `uv run elrobot_deploy camera test --camera exo` runs one on its own.
- Every `run` session starts the profile's cameras. If a phone is off, the session warns and
  carries on without it (`required = false`).
- The Mac sends the mode from the profile: resolution, fps, `phone_lens`, `phone_quality`
  (JPEG) and `phone_exposure`. With `lock`, the default, the camera settles for 2 s, then
  exposure, white balance and focus are locked, so every recorded frame is exposed alike.
- Each frame carries its capture time on the phone's clock. The Mac maps it onto its own clock
  by a ping clock sync (the fastest of the last 10 s of round trips). Expect about a millisecond
  over Wi-Fi; that is not yet measured on a real phone. (The simulator shares the Mac's clock,
  so its offset of 0.01 ms only shows that the protocol is right.)
