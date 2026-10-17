"""Look at a rollout: its cameras and the robot, on one timeline.

    elrobot_deploy inspect <root-or-record>

For the user: `docs/deploying_a_policy/rollout_inspection.md`. Runs as its own
process, offline: it reads records (`rollout_record_utils/`) and never the
arena.

A local web app. Pick a recording (grouped by its path under the root), press
Load, and scrub. The cameras and a WebGL render of the G1-D all follow one
playhead, so pausing freezes every pane on the same instant.

**Which run a recording is, is answered before it is loaded.** Beside the Load
button sits the record chip: the policy, its checkpoint and step if any, the rollout's
own settings (sync/async, P, control rate, IK, sample steps), and a click opens
the rest — the prompt, the rollout generation, the policy entry and class, the
full checkpoint path, the robot profile, the servo settings and the git
revisions of both the infra and the policy. It is read from `manifest.json` at
discovery (`discover.specifics`), because the question decides whether a
rollout is worth the load rather than following it.

**It opens on the rollout, not on the recording.** P9 captures the whole session
— CHECK, P0's homing ramp, P5 warming, the READY prompt — and none of that is
the thing you came to look at. The timeline is cut to the ACTIVE edge from
`session.bin`, which on one recorded rollout turns 2399 display frames into 581 and
drops 60 s of an arm holding still at the home pose. A record without that
stream keeps the full window, and the UI always says which it is showing.

**Three configurations are drawn, because a miss has two causes.** A rollout
that does not reach its target either could not be solved for — the IK ran out
of arm — or was solved fine and not executed. Those need opposite work and look
identical from the commanded pose and the measured arm alone. So the joint
angles P6 published are posed as well, as a translucent robot between the two:
the gap from the amber gripper to it is the IK's own failure to reach, the gap
from it to the opaque arm is servo lag, gravity sag, a joint on its limit.

**Each of the three is one switch, and a switch carries its source's whole
contribution** — geometry, TCP marker and trajectory together — so what is
plotted is always the path of what is drawn, and comparing two of them means
switching the third off. `?off=measured,ghost` does it from the URL.

**The commanded joints stand on the trunk the IK solved in.** The seventeen
numbers in a body target answer two questions: `q[14:17]` is the waist the tick
COMMANDED, and `waist_ik` beside it is the waist P6 pinned the model at to
solve for the fourteen arm angles — the MEASURED trunk. The overlay is always
posed on the second when the record has it (the arm angles only mean what they
mean in that frame, and the torso then lands on the measured torso, leaving
every gap to the arm); a record without `waist_ik` carries only the first
and is posed on it.

**Joint-mode rollouts have no commanded pose**, except where an operator
intervened (PICO commands are EEF poses). The page greys the switch out on a
joint rollout without an intervention, and draws the pose only on the
intervened ticks otherwise; the commanded joints are drawn either way.

**The timeline carries the replans.** Every chunk P5 produced is marked twice
on the scrubber — a hollow triangle where it was triggered, a solid one where it
reached the ring — with the stretch of ticks that chunk went on to drive shaded
underneath. Clicking any of it jumps there, and `[` / `]` walk the marks. The
90 ms between one chunk's two triangles is under a pixel on the full bar, so it
gets its own band: a zoom to 1.5 s either side of the playhead, where the
inference is 30 px wide, the dotted line from a ring write to the first tick it
drove is legible, and async chunks that overlap are stacked instead of painted
over each other. Sync and async need nothing different — P5 records the same six
instants in both modes, and only the picture changes.

**Nothing is written to disk.** Every artifact — the index table, the mesh
payload, the per-frame transforms — is built in memory when Load is pressed and
dropped when the next rollout is loaded. A rollout costs a second or two to
prepare, which is cheaper than owning a cache that can go stale against a record
someone re-recorded under the same name.

**The clock problem does not exist here, and that is worth knowing before reading
`timeline.py`.** Every timestamp in every stream is `time.monotonic()` from the
same machine — state at ~1 kHz, the cameras at 30, targets at the control rate.
There is no alignment to estimate and no drift to correct. What is left is pure
resampling onto a common grid, which is all `timeline.py` does.

The modules, in dependency order:

* `discover` — what is on disk: every record under the root, grouped.
* `record_robot` — the robot a record ran on, from its manifest.
* `loader`   — one recording's manifest and streams, read through its OWN
               recorded dtypes (never `schema.py`, which may have moved on).
* `timeline` — the index table: for each display frame, which record of each
               stream is current.
* `events`   — the replan marks: when P5 woke, when its chunk hit the ring, and
               which ticks that chunk was for.
* `scene`    — MuJoCo, offline: the meshes once, then a transform per geom per
               frame, for the measured pose and for the solved one. MuJoCo never
               runs in the browser.
"""

from elrobot_deploy.dashboards.inspect.discover import Recording, scan
from elrobot_deploy.dashboards.inspect.loader import Rollout, load
from elrobot_deploy.dashboards.inspect.timeline import Timeline, build as build_timeline
from elrobot_deploy.dashboards.inspect.events import (Replan, TickLine, Marks,
                                                  build as build_marks)

__all__ = ["Recording", "scan",
           "Rollout", "load", "Timeline", "build_timeline",
           "Replan", "TickLine", "Marks", "build_marks"]
