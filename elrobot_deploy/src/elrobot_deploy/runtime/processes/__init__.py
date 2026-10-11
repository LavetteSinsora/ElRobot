"""The processes — one module each, spawned by P0 (`runtime/launcher/`) and
never imported across each other. What P7 and P8 share — the segment walk, the
stall and the takeover preempt — lives in `runtime/execution/servo_segment.py`,
which is not a process, so the two servos cannot drift apart on it.

    P1  state reader           ~1 kHz   p1_state.py
    P2… camera readers          30 Hz   p2_camera.py   (one per camera)
    P5  policy                          p5_policy.py   (policy_host.py starts it)
    P6  target (blend + IK)             p6_target.py
    P7  body control           500 Hz   p7_body.py
    P8  hand control           200 Hz   p8_hand.py
    P9  record                          p9_record.py
    P10 live dashboard                  p10_dashboard.py
    P11 PICO takeover (--intervene)     p11_pico.py

What each does, and one chunk's path through them: `docs/overview.md`, "The
processes" and "One chunk's life".

Deliberately empty of imports. P0 spawns children by dotted path
(`elrobot_deploy.runtime.processes.p1_state.run`) into a fresh interpreter, and P7 must not
acquire a transitive mujoco dependency; re-exporting the modules here
would put every one of them in every child's address space.
"""
