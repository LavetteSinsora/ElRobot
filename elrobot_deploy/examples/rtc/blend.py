"""The blend an RTC policy needs: the newest chunk takes over.

    [control]
    sync = false
    blend = "blend.py:newest"

In async mode a new chunk arrives while the previous one is still running, so
for some ticks two chunks both say what the arm should do. P6 asks the blend
to reduce them to one action. The default blend averages them (ACT's temporal
ensemble). A real-time-chunking policy must not be averaged: it already
generated the new chunk to continue the old one (`policy.py`), and averaging
would pull the arm back toward the older plan. So this blend returns the
newest chunk's action unchanged.

It does exactly what the built-in `blend = "newest"` does, and is written out
here as the smallest example of the blend interface
(`docs/infra_logic/chunk_blending.md`, "What a blend receives and returns"):

* it receives every chunk covering this tick, oldest first, each with the
  action that chunk holds for the tick;
* it returns ONE whole action (pose or joint angles, hand, waist), all from
  the same decision;
* it runs in P6, under elrobot_deploy's own Python, once per tick: numpy only.

`README.md` beside this file explains the whole example.
"""

from __future__ import annotations

from elrobot_deploy.runtime.execution.chunk_blend import Action


def _copy(a):
    return None if a is None else a.copy()


def newest(contributors) -> Action:
    """The last contributor is the newest chunk. The arrays are copied: they
    were read out of shared memory, and what is returned goes on to the IK."""
    c = contributors[-1]
    return Action(_copy(c.pose), c.hand.copy(), _copy(c.waist), _copy(c.arm_q))
