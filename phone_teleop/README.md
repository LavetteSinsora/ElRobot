# phone_teleop: iPhone 6-DoF pose streamer (prototype v1)

The iPhone runs ARKit world tracking and sends one UDP packet per camera frame
(about 60 Hz) to the Mac. Each packet carries a timestamp, the 4x4 pose
`T_world_phone` and the state of two hold buttons (ENGAGE, GRIP).
`receiver/receiver.py` receives the packets and echoes each one back, so the
phone can show the round-trip time.

## Packet format (UTF-8 JSON, one datagram)
```json
{"v":1, "seq":1234, "t":5123.456, "t_send":5123.470,
 "pose":[r00,r01,r02,tx, r10,r11,r12,ty, r20,r21,r22,tz, 0,0,0,1],
 "track":"normal", "buttons":{"engage":false, "grip":false}}
```
- `pose` is row-major and in metres. The world frame is fixed at app start
  (or at "Reset world origin"), with +Y pointing up against gravity.
- `track` should be `normal`. Ignore pose data while it reads `limited:*`.
- To extend the format, add keys. Receivers ignore keys they don't know.

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

## Run
```bash
python3 phone_teleop/receiver/receiver.py   # from the repo root; prints the Mac's LAN IP
```
Enter that IP in the app and tap Connect. If macOS asks whether Python may
accept incoming connections, allow it. The phone and the Mac must be on the
same Wi-Fi network, and a guest network with client isolation blocks the
traffic.
