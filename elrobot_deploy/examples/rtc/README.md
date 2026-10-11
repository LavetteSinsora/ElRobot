# Example: an async policy with real-time chunking

    g1d_deploy check-policy examples/rtc/deploy.toml     # offline
    g1d_deploy run examples/rtc/deploy.toml              # async; MOVES THE ARM

A complete async deployment in three files. It has no model: the "policy" is
a small sinusoid, so the part worth reading is how the pieces fit.

| file | shows |
|---|---|
| `policy.py` | an adapter that reads shared memory (the arena) to find out which actions will still be executed while it is inferring |
| `blend.py` | a blend function: how overlapping chunks become one action per tick |
| `deploy.toml` | async inference with that blend |

## The problem this example is about

In async mode the arm keeps moving on the previous chunk while the next one
is inferred. When the new chunk arrives, it replaces the old one mid-motion.
If the two chunks disagree at that moment, the arm jumps.

There are two ways to deal with that, and they are opposites:

- **Average the overlap.** Blend the old and the new chunk over the ticks they
  share. This is the infra's default (`temporal_ensemble`). It smooths any
  policy, at the cost of reacting more slowly and of averaging two different
  plans when the policy changes its mind.
- **Make the chunks agree in the first place.** Real-time chunking (RTC)
  tells the policy, before it infers, which actions of the old chunk will
  already have been executed by the time the new one lands, and the policy
  generates the new chunk to continue from them. The switch is then seamless
  and nothing needs averaging.

This example is the second way.

## Why it needs its own blend

With RTC the new chunk already continues the old one. Averaging it with the
old chunk would undo that: the arm would be pulled back toward the older
plan. So the blend must hand over to the newest chunk outright, which is what
`blend.py` does.

The infra ships this blend as `blend = "newest"`; the deploy file here names
`blend.py:newest` instead so that the example also shows how a blend is
written and plugged in. A blend runs in the target process (P6), not in the
policy's Python (`docs/infra_logic/chunk_blending.md`).

## Why the adapter reads the arena

To continue the old chunk, the adapter must know which of its rows will run
before the new chunk takes over. That is not in the observation. It needs
three facts, all in shared memory, read through `reader.arena`:

| fact | where | why |
|---|---|---|
| the previous chunk, as published | `reader.arena.chunks` | "as published" means after the infra re-timed it, so these are exactly the actions that will be executed |
| which tick is executing now | `control.k_exec` | to count how far the old chunk has got |
| how long inference takes | `control.latency_s` | the new chunk lands that much later, so that many more old rows will have run |

From these it computes the **frozen prefix**: the rows of the new chunk that
fall before the takeover and must therefore equal the old chunk's.

The design choices behind those reads:

- **The takeover is two ticks after the tick executing when the chunk
  lands** (`k_exec + 2`). The target process needs one tick to solve, so a
  chunk never affects the tick in progress or the next one.
- **Three cases freeze nothing**, because the previous chunk says nothing
  about what runs next: the first inference of a rollout, a previous chunk
  that has already run out, and the first inference after an operator hands
  back from a takeover.
- **`reader.arena` is `None` outside a live rollout** (warmup,
  `check-policy`, a replay on a recording). The adapter then returns its own
  chunk unchanged. An adapter that reads the arena must always handle this.
- **Reads are read-only and can come back empty.** A record being rewritten
  during the read is dropped rather than returned half-written
  (`docs/infra_logic/arena_layout_and_reads.md`, "Read rules").

A real RTC policy applies the frozen prefix inside its sampler, as a
constraint on what it generates. This example applies it the crude way, by
copying the rows, because the bookkeeping is the reusable part.

## The deploy file

`sync = false` selects async. `replan_period = 10` means a new inference
every 10 ticks; inference must be shorter than that, and the infra refuses a
Start that cannot keep up. `blend` names the function above.

## Further reading

- `docs/infra_logic/arena_layout_and_reads.md`, "Worked example: where is my
  last chunk?"
- `docs/infra_logic/chunk_blending.md`, "Real-time chunking is not a blend"
- `examples/custom_modules/blend.py` for a blend that does average, but only
  chunks that agree
