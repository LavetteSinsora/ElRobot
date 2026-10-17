# Dex-1 gripper meshes

Downloaded 2026-08-26 (visual) and 2026-09-01 (collision) from Unitree's own
description package:

    https://github.com/unitreerobotics/unitree_ros
    robots/g1_description/meshes/{Dex1_base_link,Dex1_finger_link_1,Dex1_finger_link_2}.STL
    robots/g1_description/meshes/{dex1_col_1,dex1_col_2}.stl

They belong to upstream's `g1_29dof_mode_15_with_dex1_1` variant (a G1 with the
Dex-1 mounted). Binary STL, metres, no scale factor, nothing decimated.

## The two `dex1_col_*.stl` are the jaws' collision geometry

Upstream gives each jaw a different mesh for `<visual>` and for `<collision>`,
and `robot/g1d_model.py` reproduces that split. It is not a triangle-count
optimisation: **MuJoCo collides the CONVEX HULL of whatever mesh it is given**,
and the hull of a jaw's visual mesh spans the gap between the fingers, which
would model an open gripper as a solid block. The base link has no such variant
upstream and collides against its own mesh.

A mounted link needs a `<collision>` element at all: one with `<visual>` only
compiles to `contype=0` and silently drops out of the IK's collision avoidance.
`tests/test_kin_g1d.py` asserts the jaws are collidable, against these meshes.

## What reads them

`robot/g1d_model.py` mounts these on both wrists of the model that P1's FK,
P6's IK and the dashboards' 3-D view share. P7's gravity compensation never
sees them: pinocchio reads `g1_d.urdf` as shipped.

Nothing in the control path depends on the jaws being posed. Poses are of the
wrist-yaw frame (the flange); the gripper is present so that the IK's collision
limit can steer around it and the viewer can draw it.

## How it mounts

From the same upstream URDF:

    left_base_joint    fixed  left_wrist_yaw_link  -> left_dex1_base_link
                              xyz="0.0415 0 0"  rpy="0 0 0"
    right_base_joint   fixed  right_wrist_yaw_link -> right_dex1_base_link
                              xyz="0.0415 0 0"  rpy="0 0 0"

The gripper base is 41.5 mm along the flange's +x, unrotated, the same on both
sides.

## The jaws

Two prismatic joints per hand, opposed:

    {side}_dex1_finger_joint_1   axis="0 -1 0"   range=[-0.02, 0.0245]
    {side}_dex1_finger_joint_2   axis="0  1 0"   range=[-0.02, 0.0245]

**The units map is uncalibrated.** The measured grip is in gear radians (about
0 closed to 5.4 open); these joints are in metres. The viewer assumes a linear
relation:

    finger = -0.02 + (grip - hand_min) / (hand_max - hand_min) * 0.0445

so jaw opening in the viewer is indicative, not measured.
