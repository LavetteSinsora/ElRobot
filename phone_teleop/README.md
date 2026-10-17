# phone_teleop: the iPhone as the teleop controller (packet v2)

The iPhone runs ARKit world tracking and sends one UDP packet per camera frame (about 60 Hz)
to the Mac: a timestamp, the 4x4 pose `T_world_phone`, the tracking state, a world-frame
counter and two buttons. On the Mac, `elrobot_deploy run` receives them (its phone source,
P11, `elrobot_deploy/src/elrobot_deploy/runtime/processes/p11_phone.py`), and echoes back what
the arm is doing, which the app shows (and buzzes on).

## How to hold it, and drive

Hold the phone in your right hand with the **screen facing left**, the back camera facing
right, and the **top (Dynamic Island) pointing forward**, your thumb on the screen.

- **Put your thumb anywhere on the pad to ENGAGE.** While it is down, the arm's gripper tip
  copies the phone's motion **relative to where the phone was when you pressed**: move the phone
  forward and the arm moves forward; left, left; up, up. Tilting the phone's top down pitches
  the gripper down; turning it left turns the gripper left.
- **Slide your thumb into the forward (top) zone of the pad to GRIP.** The jaw closes while
  your thumb is there and opens when it leaves.
- **Lift your thumb to release.** The arm holds where it is (and so does the jaw). Re-grip
  anywhere and continue: that is how you move further than your arm reaches, or re-orient
  yourself.

Only the heading of the phone at the press matters ("forward" is where its top points,
horizontally); you need not hold it level. ARKit tracks with the back camera, so whatever is
to your right should be textured and static (not the moving arm).

**The Mac releases the arm on its own** (the banner turns red and the phone buzzes; lift your
thumb and press again) when: tracking is not "normal", the world origin changes (Reset world
origin, relocalization), the phone jumps more than 5 cm or 30 degrees in one frame, no packet
arrives for 250 ms, or the phone's top points straight up or down at the press.

## Packet v2 (UTF-8 JSON, one datagram)
```json
{"v":2, "seq":1234, "t":5123.456, "t_send":5123.470,
 "pose":[r00,r01,r02,tx, r10,r11,r12,ty, r20,r21,r22,tz, 0,0,0,1],
 "track":"normal", "world_epoch":3, "buttons":{"engage":false, "grip":false}}
```
- `t` is the ARFrame capture time on the phone's clock: the deadlines on the Mac are this
  plus a fixed lead (40 ms), re-pinned at every press, so the motion replays at the pace it was
  made, whatever bursts WiFi delivers it in.
- `pose` is row-major and in metres. The world frame is fixed at app start (or at "Reset world
  origin"), with +Y pointing up against gravity.
- `world_epoch` is bumped whenever the world frame may have changed (reset, relocalization,
  an interruption). The Mac releases the arm when it changes. A v1 app (without it) still works.
- Receivers ignore keys they don't know.

The Mac's echo: `{"seq", "t_send", "t_mac", "state", "why", "intervene"}`; `state` is
`inactive` (the arm is not ready), `idle` (ready: press), `engaged` or `released` (a guard
let go; `why` says which).

## Build and install (needs a Mac with Xcode; a free Apple ID is enough)
1. `brew install xcodegen`, then `cd ios && xcodegen`. Open `PhoneTeleop.xcodeproj`.
   (Without XcodeGen: in Xcode, create a new iOS App (SwiftUI) named
   PhoneTeleop and replace its Swift files with the three in
   `ios/PhoneTeleop/`. Under Target → Info, add *Privacy – Camera Usage
   Description* and *Privacy – Local Network Usage Description*.)
2. Xcode → Settings → Accounts: sign in with your Apple ID. Then open
   Target → Signing & Capabilities and pick your "Personal Team". If the
   bundle ID is taken, change it.
3. Connect the iPhone by USB and tap "Trust". On the iPhone, turn on
   Settings → Privacy & Security → **Developer Mode** (the phone restarts).
4. Choose the iPhone as the run destination and press ⌘R. The first time,
   the iPhone may ask you to trust the developer under Settings → General →
   VPN & Device Management.
5. On first launch, allow **Camera** and **Local Network** access.
6. A free-team build expires after 7 days. To renew it, run it again from Xcode.

## Run (from `ElRobot/elrobot_deploy/`)

```bash
uv run elrobot_deploy phone-bench        # 1. no robot: check the grip (prints this Mac's IP)
uv run elrobot_deploy run --sim          # 2. the whole chain on simulated servos
uv run elrobot_deploy run                # 3. the real arm
```

`run` checks the hardware, then waits: open the dashboard (http://127.0.0.1:8080), press
**Home** (the arm moves to the green ghost), then **Start**; then hold the pad on the phone.

Enter the Mac's IP in the app and tap Connect. If macOS asks whether Python may accept incoming
connections, allow it. The phone and the Mac must be on the same Wi-Fi network; a guest network
with client isolation blocks the traffic.

In `phone-bench`, hold the phone as above, hold the pad and push the phone forward: `fwd` must
grow (left: `left` grows; up: `up` grows; tip the top down: `rot about left` goes positive).

`receiver/receiver.py` is a dependency-free link test (it only echoes).
