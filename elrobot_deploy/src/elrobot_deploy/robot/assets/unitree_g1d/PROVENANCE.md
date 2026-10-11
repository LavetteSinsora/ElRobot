# G1-D description

Downloaded 2026-08-26 from Unitree's own description package:

    https://github.com/unitreerobotics/unitree_ros
    robots/g1_d_description/{g1_d.urdf, meshes/}

`g1_d.urdf` is **byte-identical to upstream**: do not edit it. Everything that
has to change about it (stripping the three-finger hand, mounting the Dex-1) is
done in memory by `robot/g1d_model.py`.

It is the only robot model elrobot_deploy loads. A robot file's `[model] urdf`
points here, and two engines read it: MuJoCo for geometry (P1's FK, P6's IK and
its collision limit, the dashboards' 3-D view) and pinocchio for mass (P7's
gravity feedforward).

## The joints

Its non-fixed joints, in full:

| joint | type | axis | range | what it is |
|---|---|---|---|---|
| `Left_Wheel_Joint` / `Right_Wheel_Joint` | continuous | ∓Y | — | the differential base |
| `LZ_mt_Joint`, `LZ_it_Joint` | prismatic | +Z | 0 … 0.21 m each | the lift column, two telescoping stages |
| `Yaw_Joint` | revolute | **0 1 0** | −0.0436 … 2.3562 rad | **the BOW** (−2.5° … +135°) |
| `torso_Joint` | revolute | **0 0 1** | ±2.7053 rad | **the YAW** about the column (±155°) |
| 14 arm joints | revolute | — | — | the same arm chain as the G1 humanoid |

**`Yaw_Joint` is the bow and `torso_Joint` is the yaw.** The names are backwards
from their own axes. Map by axis or by this table, never by the name.

The base frame elrobot_deploy expresses poses in is placed where the G1 humanoid's
pelvis sits (`robot/g1d_model.py::PELVIS_FROM_TORSO`), so waist-at-zero poses
match the humanoid's pelvis frame.

## What was left out

Upstream's `meshes/` also ships the **three-finger hand** (`*_hand_*.STL`, more
than half of the download). elrobot_deploy mounts a Dex-1 instead
(`../unitree_dex1/`), so those meshes are not shipped here. The URDF still
references them; `robot/g1d_model.py` removes the hand links before MuJoCo
resolves a filename, which makes that strip mandatory: the model does not
compile without it.

The hand links stay in place for **pinocchio**, which reads this same file
unmodified. It never resolves a mesh (it wants kinematics and inertia only),
and the three-finger hand's mass is what the gravity feedforward assumes.

Nothing is decimated: 25 meshes.

## Loading it

MuJoCo's URDF importer prefixes its own `meshdir`, so a mesh referenced as
`meshes/X.STL` is looked up at `meshes/meshes/X.STL`; `robot/g1d_model.py` keys
its assets dict accordingly. Visual geoms come back as `contype == 0` (group 1)
and collision as `contype != 0` (group 0).

**MuJoCo merges a link joined by a FIXED joint into its parent.** `AGV_link` is
the URDF root and becomes the worldbody; `Pitching_Link` disappears into
`LZ_it_Link`; `{side}_dex1_base_link` into `{side}_wrist_yaw_link`. So names in
this file are not all names in the compiled model, which is why
`robot/g1d_model.py::BASE_BODY` says `LZ_it_Link`.
