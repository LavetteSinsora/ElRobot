# Chunk blending

In async mode a new chunk arrives while the previous one is still running, so
for a stretch of ticks two (or more) chunks each say what the arm should do.
The **blend** reduces them to one action per tick. The two shipped blends can be
switched between rollouts; it can also be your own function, chosen at launch.

- **Read first:** `docs/overview.md`, and the async part of
  `docs/deploying_a_policy/deploy_toml.md` (`[control] sync`).
- **Code:** `src/elrobot_deploy/runtime/execution/chunk_blend.py` (`apply`,
  `temporal_ensemble`, `newest`, `weighted`, `get`); the caller is
  `src/elrobot_deploy/runtime/processes/p6_target.py::Target`.
- **Related:** `docs/infra_logic/arena_layout_and_reads.md` (the chunk ring),
  `examples/custom_modules/blend.py`.

## When it matters

- **Sync:** chunks are cut to P actions and partition the tick line, so
  the blend is called with exactly one contributor (and must return it).
- **Async:** chunk `n+1` is triggered P ticks after chunk `n` and covers the
  next H ticks from there, so consecutive chunks overlap by `H − P` ticks. When
  it lands, P6 re-solves every tick from the one executing plus two
  (`k_exec + 2`) with both chunks as contributors; ticks before that keep
  what was already written.

## What a blend receives and returns

```python
def blend(contributors: Sequence[Contribution]) -> Action: ...
```

`contributors` — every chunk P6 has adopted that covers this tick, in
ascending `chunk_id` (oldest first, newest last). Each is a
`src/elrobot_deploy/runtime/shared_memory/rings.py::Contribution`:

| field | meaning |
|---|---|
| `chunk_id` | 1, 2, ... in the rollout |
| `row` | which row of its chunk this tick is (`k − k_first_action`): how far into that chunk's prediction the tick lies. The newest chunk has the smallest row. |
| `pose` | `(2, 7)` flange pose, eef chunks; `None` for joint chunks |
| `arm_q`, `waist` | `(2, 7)`, `(3,)`, joint chunks; `None` for eef |
| `hand` | `(2, 6)`; the Dex-1 value is element 0 |

Return a `chunk_blend.Action(pose, hand, waist, arm_q)` — the whole action,
from one decision. Pose, hand and waist are blended together so a jaw is never
taken from one chunk while the arm follows another.

Before the blend is called, `chunk_blend.apply` flips every contributor's
quaternions into the newest one's hemisphere (`q` and `-q` are the same
rotation and naively average to garbage), so a blend never has to.

## The shipped blends

| `blend` | what |
|---|---|
| `"temporal_ensemble"` (**default**) | ACT's temporal ensembling (Zhao et al., 2023): every chunk's prediction of the tick, averaged with `w_i = exp(−m · i)`, `i = 0` the oldest prediction, ages counted in ticks (`row`), so the weights do not depend on P. Older predictions weigh slightly more. `m` is `[control] temporal_ensemble_m` (ACT's 0.01; 0 = a plain mean). Consecutive chunks are blended over their `H − P` overlapping ticks instead of switched at the seam: smoother, at the cost of reacting more slowly to the newest chunk, and of averaging two plans if the model changes its mind |
| `"newest"` | the newest chunk wins outright: each new chunk takes over at once, so the motion switches plans at the seam (a step the IK's velocity clamp bounds). The right choice when the model already makes consecutive chunks agree (real-time chunking), or to react fastest |

Both return whole actions via `chunk_blend.weighted` / `newest`, which a
blend of your own can reuse.

## Writing your own

```toml
[control]
sync = false
blend = "blends.py:temporal_ensemble"    # relative to the deploy file, or "pkg.module:fn"
```

- **It runs in P6, under elrobot_deploy's own Python** (not the adapter's), so
  its module must import there. It is imported at launch, and one that does
  not import is refused before anything moves.
- **Once per tick per re-solve**, inside P6's per-tick budget: pure numpy,
  microseconds, no allocation-heavy work.
- **Return every block the contributors carry** (`pose` or `arm_q`, `hand`,
  `waist` if present); use `None` where they are all `None`.
- **Per launch.** The dashboard switches between the shipped blends, and
  away from yours, but not back to it.

`examples/custom_modules/blend.py` is a **mode-aware ensemble**: it averages
only the chunks that agree with the newest one at this tick (within 3 cm at
every flange), with ACT's weights, and drops the rest. A multimodal policy
(diffusion, flow) may pass an object on the left in one chunk and on the right
in the next; the plain ensemble would average those into a path through the
object, while this one keeps the ensemble's smoothness when the plan holds and
switches cleanly when it changes.

## Real-time chunking is not a blend

Real-time chunking (RTC) makes the *model* generate a chunk whose first rows
agree with what the previous chunk will still be executing while this one is
inferred, so switching to the newest chunk is seamless. That happens inside
the adapter, before the chunk exists; set `blend = "newest"` (averaging would
undo the agreement it built). What the
adapter needs is which ticks of its previous chunk will run before the new one
takes over: `reader.arena` gives `k_exec`, the published chunks and the
measured latency (`docs/infra_logic/arena_layout_and_reads.md`, "Worked example"), and
`examples/rtc/` shows the bookkeeping.

A complete worked example, with the reasons for each piece: `examples/rtc/README.md`.
