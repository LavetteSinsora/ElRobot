"""Named arm poses, in URDF joint angles (degrees).

The URDF itself only describes geometry and joints; named poses live here.

rest  The folded pose the arm is parked in. Upper arm back on the shoulder stop, the elbow
      straight and the wrist pitch folding the forearm back over the upper arm (chosen
      2026-10-11 for stability: earlier it was elbow +90, wrist pitch 0). Elbow + wrist pitch
      = +90 keeps the forearm flat over the upper arm. Gripper fully open (0 = open).
zero  The URDF's zero pose: upper arm straight up, forearm level and pointing forward.
"""

POSES: dict[str, dict[int, float]] = {
    "rest": {1: 0, 2: -90, 3: 0, 4: 90, 5: 0, 6: 0, 7: 0, 8: 0},
    "zero": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0},
}
