# Our ElRobot description

`elrobot.urdf` describes the arm we assembled: NormaCore's ElRobot follower (`../elrobot_upstream/`,
unchanged) plus our calibration, our TCP, and our Yoke + Facet wrist-camera mount. It is
**generated**; don't edit it by hand.

```bash
uv run python description/elrobot_ours/make_description.py                     # regenerate after recalibrating
uv run --with matplotlib python description/elrobot_ours/render.py rest -o r.png    # look at it
uv run --with pytest pytest tests/test_description.py                          # check it
```

| file | what |
|---|---|
| `make_description.py` | builds `elrobot.urdf` and `assets/` from the inputs below |
| `elrobot.urdf` | the result. Upstream meshes are referenced in `../elrobot_upstream/assets/` |
| `assets/camera_mount_*.stl` | the printed mount, re-expressed in its own CAD frame (generated) |
| `printed/` | the mount exactly as printed: the two STLs and the parametric CadQuery script, copied from `3D Printing Files/Camera mounts (HX200S)/` (3 Oct, the final Yoke + Facet version) |
| `render.py` | draws the description at a pose, with the `tcp` and camera frames |
| `renders/` | reference renders: `arm_zero.png`, `arm_rest.png`, `gripper_zero.png` |

## What differs from upstream

0. **`base_link` is REP-103**: x forward (where the arm reaches at the zero pose), y left, z up.
   Upstream's `base_link` has the arm along +y (x to the robot's right); it is renamed
   `base_body` and hangs under a new, empty `base_link` by a fixed joint turned -90 deg about z.
   No joint, origin or angle of the arm changes: only the frame poses are written in.
   (`elrobot/kinematics.py` defaults to the UPSTREAM file and keeps upstream's axes.)
1. **Joint limits** are our calibrated safe ranges: the measured ends pulled in by
   `elrobot/calibration.py`'s margins (3 deg; 5 deg on the cable-limited rolls 5 and 7). They come
   from `calibration/calibration.json`, except the gripper's closed end. That file caps it at
   upstream's 126.2 deg; we use the hand-swept travel in `calibration/hand_ranges.json` (142.8 deg
   from the open stop), minus 3 deg.

   | joint | ours (deg) | upstream (deg) |
   |---|---|---|
   | 1 base yaw | -85.2 .. 85.2 | +-88.9 |
   | 2 shoulder | -90.3 .. 90.3 | +-92.4 |
   | 3 elbow | -98.4 .. 98.4 | +-100.9 |
   | 4 wrist pitch | -98.1 .. 98.1 | +-100.5 |
   | 5 forearm roll | -151.8 .. 166.2 | -150.1 .. 186.3 |
   | 6 wrist bend | -75.6 .. 97.8 | -78.9 .. 101.1 |
   | 7 wrist roll | -165.9 .. 150.8 | -183.4 .. 156.6 |
   | 8 gripper | 3.0 .. 139.8 | 0 .. 126.2 |

   Velocity (4.71 rad/s, about the ST3215's no-load speed) and effort are upstream's placeholders;
   the deploy side scales velocity down.
2. **Gripper jaws.** Upstream's jaw joint frames sit ~0.3 m away from the gripper (its meshes cancel
   the offset). They are moved onto the jaws without changing the rendered geometry. The jaw/gear
   ratio is scaled to 10.15 mm/rad per jaw (upstream: 11.5), so the tips meet at our closed stop.
   The open tip gap is 50.6 mm. Either the real gear ratio or our open stop differs from the CAD;
   control doesn't care, because the gripper is commanded and bounded in servo angle.
3. **`tcp`** (fixed, on `Gripper_Base_v1_1`): the midpoint of the jaw tips' far ends,
   (1.8, 103.4, -0.7) mm from the wrist-roll frame. **x out of the jaws, y left, z up** at the
   zero pose. The jaws close symmetrically, so it doesn't move with the gripper. It is the only
   end-effector frame the deploy code uses; there is no flange frame.
4. **`camera_mount`** (fixed, on `Gripper_Base_v1_1`): the Yoke + Facet body and back cover. Its
   foot bolts through the left-right pair (3.5 mm, 14 mm apart) of the four-hole cross on top of
   the gripper base, with the camera looking forward along the jaws. About 19.5 g with the camera
   board (approximate; position control doesn't use it).
5. **`wrist_camera_optical`** (fixed, on `camera_mount`): at the board's front-face centre plus
   11 mm along the view axis (approximate lens centre). OpenCV convention for the upright image:
   z along the view (25 deg below the gripper's axis), x right, y down. It is 61.5 mm above and
   21.4 mm in front of the wrist-roll frame. The raw image may need the 180 deg rotation noted in
   `ELROBOT_CONTEXT.md`; this frame describes the image after that.

## How the mount was placed (and checked)

The STLs are in print orientation, so nothing about their placement is assumed. The generator
finds the mount's own CAD frame from its features:
- the foot's contact face, the only face holding two 3.5 mm holes 14 mm apart;
- the lens window, the 23 mm bore along the view axis.

The script's own numbers are the check. Fitted from the STL alone, the board centre comes out at
(-0.19, 0, 51.04) mm, the view axis at (0.9063, 0, -0.4226), and the holes at (14, +-7, 0), all
exact. The back cover is placed from its screw holes, counterbore and cable notch the same way.

On the arm side, the holes are found on the `Gripper_Base` mesh: radius 1.750 mm, 14.0 mm apart,
on the top face. The front-back pair of the same cross is 2 mm, so it isn't the mount's.

**One assumption to confirm by eye:** the camera faces forward, towards the jaws. Bolting the
same foot rotated 180 deg on the same holes would face it backwards. Compare
`renders/gripper_zero.png` with the arm.

## Not modelled

- The arm's zero pose, joint origins and axes are upstream's (including the 0.5-1 deg axis tilts
  of the CAD export). Calibration maps our servo readings onto them.
- Named poses (`rest`, `zero`) live in `elrobot/poses.py`, not in the URDF.
- Cables, the table clamp and the mount plate.
