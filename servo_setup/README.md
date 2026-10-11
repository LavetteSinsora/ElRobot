# ElRobot follower: servo ID setup

`servo_id.py` gives each ST3215 its ElRobot bus ID. This is step 4 of NormaCore's
`manual-follower.pdf` ("Set servo IDs using app, repeat 8 times"), done from the
command line instead of the NormaCore Station web UI.

## The ID map (from the NormaCore repo)

| ID | Joint | URDF joint |
|---|---|---|
| 1 | base yaw | rev_motor_01 |
| 2 | shoulder | rev_motor_02 |
| 3 | elbow | rev_motor_03 |
| 4 | wrist pitch | rev_motor_04 |
| 5 | forearm roll | rev_motor_05 |
| 6 | wrist bend | rev_motor_06 |
| 7 | wrist roll | rev_motor_07 |
| 8 | gripper | rev_motor_08 |

- **Baud rate:** 1 Mbps (`SUPPORTED_BAUD_RATES[0]` in `software/drivers/st3215/src/protocol/packet.rs`).
- **Order:** the ElRobot auto-calibration (`auto_calibrate/elrobot.rs`) treats motor 1 as the base
  and motor 8 as the gripper. The manual says to label the servos 1 to 8 with a marker (step 3).

## Setup

1. Move the Waveshare Bus Servo Adapter jumper to **B** (USB mode).
2. Connect the 12 V supply to the board's DC jack, and the board's USB-C port to the Mac with a **data** cable.
3. From `ElRobotRepo/`, run `uv sync` once. This creates `.venv/` with pyserial and pytest, from `pyproject.toml`.

## Usage

Run these from `ElRobotRepo/`:

```bash
uv run servo_setup/servo_id.py ports            # the board shows up as CH343, 1a86:55d3
uv run servo_setup/servo_id.py wizard           # guided: plug in ONE servo, Enter, label it, next...
uv run servo_setup/servo_id.py wizard --start 5 # resume from ID 5
uv run servo_setup/servo_id.py set-id 3         # one-off: whichever single servo is connected becomes ID 3
uv run servo_setup/servo_id.py scan             # after daisy-chaining all 8, expect IDs 1-8
uv run servo_setup/servo_id.py scan --full --all-bauds   # find a "lost" servo at any ID or baud rate
uv run servo_setup/servo_id.py info 4           # dump ID, baud, angle limits, offset, lock, position...
```

**Only connect one servo when assigning IDs.** New servos all come as ID 1. Two of them on
the bus answer at the same time, their replies collide, and both could end up with the
same new ID.

## What `set-id` does

1. Pings IDs 0–253 at 1 Mbps (falling back to the other baud rates if nothing answers) and
   checks that exactly one servo replies.
2. Turns torque off (`0x28 = 0`), so nothing moves.
3. Unlocks the EEPROM (`0x37 = 0`). Without this the new ID only lasts until the next power-off.
4. Writes the new ID to `0x05`, then pings the new ID.
5. If the servo wasn't at 1 Mbps, writes `0x06 = 0` (1 Mbps) and reopens the port.
6. Locks the EEPROM again (`0x37 = 1`) and reads `0x05` back to check it.

No motion commands are sent. Calibration (offset, angle limits) is a separate step, done later.

## Tests (no hardware)

`uv run pytest` (from `ElRobotRepo/`) runs the tool against a simulated ST3215 bus. It covers packet
checksums, keeping the ID through a power cycle, adapters that echo TX back to RX, fixing a
servo at the wrong baud rate, refusing when two servos are connected, and scanning a full chain.

## Other ways to do this (references)

- **NormaCore Station** (official): its web UI's ST3215 motor config page
  (`software/station/clients/station-viewer/src/pages/St3215MotorConfigPage.tsx`) uses the
  same unlock → write `0x05` → lock sequence, and limits IDs to 1–10.
- **LeRobot** `lerobot-setup-motors` (`src/lerobot/motors/feetech/feetech.py`): finds a single motor,
  then writes its ID and baud rate. It's built for SO-100/101 (6 motors), so it doesn't know the ElRobot map.
- **Feetech FD.exe** (Windows GUI) and the **Waveshare ST/SC servo Python/Arduino libraries**
  (Bus Servo Adapter (A) wiki, Resources): `unLockEprom` → write `SMS_STS_ID` → `LockEprom`.
