# The arena: layout, clocks, and reading it yourself

Everything the runtime's processes share lives in one block of shared memory,
the **arena**. `obs` and `reader` (`docs/deploying_a_policy/policy_contract.md`) are built from it
and cover most adapters. This document is for the ones that need more — the
tick being executed, the chunks already published and when each of their
actions is due (real-time chunking), P6's joint targets, P7's 500 Hz commands
— and for anyone writing a tool that reads a live run.

- **Read first:** `docs/overview.md` (the processes), `docs/deploying_a_policy/policy_contract.md`.
- **Code:** `src/elrobot_deploy/runtime/shared_memory/` — `schema.py` (every
  record dtype, `VERSION`), `layout.py` (the region table, the layout hash,
  `ReadOnlyArena`), `primitives.py` (`Seqlock`), `rings.py` (one accessor
  class per region, `ArenaRings`, `open_read_only`).
- **Related:** `examples/rtc/policy.py` (an adapter that reads it),
  `docs/infra_logic/chunk_blending.md`, `docs/infra_logic/500hz_execution.md`.

## Contents

1. [Getting at it from an adapter](#getting-at-it-from-an-adapter)
2. [Ticks and clocks](#ticks-and-clocks)
3. [The regions](#the-regions)
4. [The records, field by field](#the-records-field-by-field)
5. [Read rules: why a read can come back empty](#read-rules-why-a-read-can-come-back-empty)
6. [The accessor classes](#the-accessor-classes)
7. [Worked example: where is my last chunk?](#worked-example-where-is-my-last-chunk)
8. [What is stable](#what-is-stable)
9. [Size, and `/dev/shm`](#size-and-devshm)

## Getting at it from an adapter

`reader.arena` in `infer` is an
`src/elrobot_deploy/runtime/shared_memory/rings.py::ArenaRings`:

```python
def infer(self, obs, reader):
    a = reader.arena
    if a is None:                       # warmup, check-policy, policy replay
        return self._infer_without_arena(obs, reader)
    k_now = a.control.k_exec            # the tick P7 is executing
    last = a.chunks.get(a.control.newest_chunk_id)
```

| field | region | written by |
|---|---|---|
| `control` | single words: session, `k_exec`, newest chunk id, ... | several, one per word |
| `state` | ~1 kHz robot state | P1 |
| `chunks` | the chunks P5 published | P5 |
| `targets` | one joint target per tick | P6 (P0 during homing and exit) |
| `intervene` | the PICO operator's targets | P11 |
| `command` | every 500 Hz command | P7 |
| `cameras` | view name → camera ring | P2 (one per robot-file camera) |
| `layout` | the region table; `layout.describe()` prints it | — |

- **Read-only.** Every view has numpy's `writeable` flag off, so any write —
  including a ring's own `write` method — raises `ValueError: assignment
  destination is read-only`. Each region has exactly one writer, and a
  second one corrupts it silently; this turns that mistake into an error. It
  is a guard, not a security boundary: the adapter runs inside P5.
- **`None` outside a live rollout:** during `warmup`, in `check-policy` and in
  a policy replay. An adapter that uses it must have a path for `None`, and
  must not let it change what gets compiled (warmup compiles the live graph).
- **Not gated by `history`.** Anything in any ring can be read. But a
  camera ring's depth *is* set by `history`: undeclared cameras keep 3 frames.
- **Copy, then use.** Every accessor returns a private copy; do not keep numpy
  views into the arena (a slot can be rewritten under you mid-use).

## Ticks and clocks

**Every time is box-monotonic seconds** (`time.monotonic()` on the deploy
box, `src/elrobot_deploy/runtime/timing.py`), stamped when data reaches the box.
The robot's own clock appears only as `state.tick`.

**A tick** is one action period, `1 / control_hz`. Ticks are counted by `k`
from the end of homing (`k = 0`); homing and the exit ramp run at negative `k`.

- **`control.k_exec`** — the tick P7 is walking toward right now: the target
  it is interpolating to. Published by P7 when it moves on to the next target.
- **A chunk** covers ticks `k_first_action .. k_first_action + n_actions − 1`;
  its row `j` is tick `k_first_action + j`.
- **A deadline** is the instant a tick's target should be reached. P6 derives
  every deadline from one pair `(t_base, k_base)`, the *epoch*:
  `deadline(k) = t_base + (k − k_base) / control_hz`. The epoch is not
  published; recover it from any written target, since
  `targets.get(k)` returns `(k, deadline, payload)`.
  - **Sync:** re-stamped at every chunk, when P6 adopts it:
    `deadline(k_exec + 2) = now + 2/control_hz`. The robot does not advance
    ticks while waiting for inference, so the clock and the tick line
    re-align each chunk.
  - **Async:** one epoch for the whole rollout (`k_base = 0`), re-founded only
    after a PICO intervention hands back.
- **Chunk timing.** P5 triggers an inference at `t_trigger`, the deadline of
  tick `k_first_action − 1` (the first inference of a rollout, or after a
  handback, triggers at "now" on `k_exec`).
  - **Async:** one epoch, so row `j` is due at
    `t_trigger + (j + 1) / control_hz` — the contract's
    `(j + 1) / control_hz` after `t_anchor`, less the sub-tick gap the
    deadline shift corrects.
  - **Sync:** P6 re-stamps the epoch when it adopts the chunk, so every row is
    due about one inference latency later than that; read the real deadline
    from `targets.get(k)`.
- **Adoption.** P6 notices a new chunk (via `control.newest_chunk_id`) and
  starts solving at `k_exec + 2` as read at that instant. Row 0 is therefore
  never executed, and in async every row whose tick is `≤ k_exec + 1` at
  adoption comes from the older chunk.

## The regions

In layout order. `N` is a ring's slot count, computed at launch
(`src/elrobot_deploy/config_loading/derive.py::derive`) and printed at startup as
`derived:`, with each region's size.

| region(s) | record | N | writer | read rules |
|---|---|---|---|---|
| `header` | `HEADER`: magic, version, layout hash | 1 | P0, once | checked on every attach |
| `control` | `CONTROL` | 1 | one writer per word | independent atomic 8-byte words; multi-word blocks carry their own seqlock |
| `state_seq`, `state_rec` | `STATE` | `next_pow2(lowstate_hz × state_history_s)` (2048 at the defaults) | P1, ~1 kHz | seqlock ring, absolute `index` |
| `chunk_seq`, `chunk_rec` | `chunk_dtype(H)` | `ceil(H / 2) + 1` | P5 | seqlock ring, slot `chunk_id % N`; `control.newest_chunk_id` stored after the payload |
| `target_seq`, `target_rec` | `TARGET` | `next_pow2(3 × H)` | P6 (rollouts), P0 (ramps) | seqlock ring by tick `k` |
| `intervene_seq`, `intervene_rec`, `intervene_meta` | `INTERVENE`, `INTERVENE_META` | `next_pow2(pico_poll_hz × 4)` | P11 | seqlock ring + a header of atomic words |
| `command_seq`, `command_rec` | `COMMAND` | `next_pow2(body_hz × 0.5)` (256) | P7, 500 Hz | seqlock ring, absolute `index` |
| `cam_<view>_head`, `_seq`, `_rec` | `u64`, `u64`, `camera_dtype(h, w, cap)` | `ceil(history_s × hz × 1.25) + 3` | P2 | camera ring (below) |

All sizes depend only on per-launch values, so a rollout can change `P`,
`sync` or `control_hz` without the arena changing.

## The records, field by field

**`CONTROL`** — the words worth reading (`rings.Control`):

| word | meaning |
|---|---|
| `session` | `INIT, CHECK, HOME, WARMUP, READY, ACTIVE, EXITING, STOPPED` (0–7, `schema.SESSION_NAMES`) |
| `k_exec` | the tick P7 is executing |
| `newest_chunk_id` | the newest chunk P5 published; 0 before the first of a rollout |
| `latency_s` | P5's measured inference latency, trigger to chunk written (p90 of recent dummies + the anchor's worst wait) |
| `intervene` | even: the policy drives; odd: a PICO operator does |
| `rollout_generation`, `active()` | which rollout, and its settings and prompt |

**`STATE`** (P1, one per `rt/lowstate` message, ~1042 Hz):

| field | shape | meaning |
|---|---|---|
| `index` | | absolute sample counter |
| `t_recv` | | box time the message arrived |
| `q`, `dq` | `(29,)` | joint angles (rad) and velocities, wire order |
| `flange` | `(2, 7)` | FK of this `q`, base frame, `[x y z qw qx qy qz]` |
| `grip_meas`, `t_grip_recv` | `(12,)`, | the hands' measured state (Dex-1: element 0 of each hand's 6) and when it arrived (~200 Hz, so often held) |
| `column_h`, `t_column_recv` | | lift column height (m, ~30 Hz, held) and its arrival time; 0 = never |
| `tick` | u32 | the robot's own millisecond clock |

**Chunk slot** (`schema.chunk_dtype(H)`, P5):

| field | meaning |
|---|---|
| `chunk_id` | 1, 2, ... per rollout; 0 = empty slot |
| `k_first_action`, `n_actions` | the ticks it covers |
| `n_plan` | rows resident (≥ `n_actions`; extra rows are trunk-planner look-ahead, never executed) |
| `mode` | 0 = eef (`pose`), 1 = joint (`arm_q`, `waist`) |
| `t_trigger` | scheduled trigger |
| `t_anchor` | the anchor frame's receive time |
| `t_obs_received` | every camera and the state settled |
| `t_cpu_processed` | preprocessing done |
| `t_chunk_received` | model output back on the host |
| `t_chunk_written` | this write began |
| `pose (H, 2, 7)`, `hand (H, 2, 6)`, `waist (H, 3)`, `arm_q (H, 2, 7)` | the actions **after** the deadline shift; unused blocks NaN; `hand[..., 0]` is the Dex-1 value |

**`TARGET`** (P6, one row per tick):

| field | meaning |
|---|---|
| `k`, `deadline` | the tick and when it should be reached |
| `intervene` | the intervention counter it was written under |
| `q (17,)` | 14 arm joints then 3 waist: what P7 follows |
| `hand (2, 6)` | what P8 follows |
| `err` | the IK's worst position residual (m) at the TCP (or flange); NaN if no IK produced it |
| `pose (2, 7)` | the blended flange target the IK was given; NaN in joint mode and for ramps |
| `waist_ik (3,)` | the (measured) trunk the IK solved in |

**`COMMAND`** (P7, one row per 500 Hz publish): `index`, `t`, `k_exec`, and
`(17,)` each of `q` (published position), `dq` (velocity feedforward), `tau`
(gravity + model feedforward), `tau_ff` (the feedforward part), `q_lin` (the
plain linear walk at that instant; equals `q` when no smoother is on).

**`INTERVENE`** (P11, only during a takeover): `index`, `intervene`,
`deadline`, `pose (2, 7)`, `hand (2, 6)`, `waist (3,)`, the operator's raw
inputs (`engaged`, `trigger`, `axis_y`, `direction`), `t_device`, `t_recv`.
`INTERVENE_META`: `samples` (bumped every poll: liveness), `newest`,
`t_last_sample`, `fresh`, `active`, `engaged`.

**Camera slot** (`schema.camera_dtype`, P2): `frame_id` (frames this box
received, from 1), `source_seq` (frames the source sent, −1 = no counter; the
gap is transport loss), `t_recv`, `jpeg_nbytes`/`jpeg` (the wire JPEG, on the
one view per wire camera that carries it), `pixels (h, w, 3)` **BGR**.
`CameraRing.read_frame(id, flip_channels=True)` returns RGB, as the reader
does.

**A fisheye view is the square around its lens circle**, never the sensor
(`src/elrobot_deploy/runtime/processes/p2_camera.py::_loop_crop`): P2 decodes the whole
frame and cuts the square into the slot, black where it runs off the sensor.
The square is fitted at INIT, before the arena is created (P0 reads the camera
itself), so it is the ring's shape and costs the arena nothing else. The JPEG
in the slot stays the whole wire frame: the recorder stores what the robot
sent, and the replay cuts the same square from it.

## Read rules: why a read can come back empty

No reader can ever delay a writer: P7 has 2 ms and cannot wait on a lock, so
there is no mutex anywhere. Instead, every ring slot has a **sequence
counter** (a `uint64` in its own array): the writer makes it odd, writes,
makes it even. A reader:

1. reads the counter; odd means "being written" (skip or retry);
2. copies the record;
3. reads the counter again; if it changed, the copy may be torn — discard it.

- **One slot** (`Seqlock.read_into`, e.g. `chunks.get`, `targets.get`):
  retried until consistent; the write window is microseconds.
- **A window** (`state.window`, `CameraRing.window`): copy all counters, all
  records, all counters again; keep only slots even and unchanged in both
  copies. A slot being written is **dropped, not retried**, because the writer
  is always somewhere in the ring and retrying would chase it.
- **Camera frames:** frame `f` lives in slot `(f − 1) % N`. The newest is
  `cam_<view>_head`. `read_frame(f)` checks the slot's counter and that it
  still holds `frame_id == f`; if not, the frame is **gone** (overwritten N
  frames later) and it returns `None` — waiting would not bring it back.
- **Memory ordering** relies on x86 store order; this is not sound on ARM.

If you read raw regions rather than through a class, follow exactly that:
counter before, copy, counter after, odd is unreadable, and for a camera also
check `frame_id`. Never hard-code an offset: views come from
`layout.build`, which every process computes identically from the config.

## The accessor classes

Read methods, all returning private copies (`rings.py`):

| class | read methods |
|---|---|
| `Control` | properties `session`, `k_exec`, `newest_chunk_id`, `warm`, `latency_s`, `intervene`, `rollout_generation`; `active()`, `refusal()`, `verdict()` |
| `StateRing` | `latest()`, `nearest(t)`, `window(t0, t1)` → `STATE` records |
| `ChunkRing` | `get(chunk_id)` → whole slot or None; `covering(k, adopted_through=None)` → one `Contribution` (`chunk_id`, `pose`, `hand`, `waist`, `arm_q`, `row`) per chunk covering `k`, ascending; `covers_any(k, adopted_through=...)` |
| `TargetRing` | `get(k)` → `(k, deadline, q)`; `full(k)` → the whole record; `latest_below(k)`; `next_after(t, intervene)`; `line_after(t, until, intervene)` |
| `CommandRing` | `latest()` |
| `InterveneRing` | `meta()`, `newest(intervene, since=0)` |
| `CameraRing` | `peek()` → `(t_recv, frame_id)` of the newest; `window(t0, t1)` → ids and times; `read_frame(id, flip_channels=True)` → `(pixels, t_recv)` or None; `read_jpeg()` |

## Worked example: where is my last chunk?

Real-time chunking generates the next chunk so that it agrees with the part of
the previous one that will still run while the new one is being inferred. In
async with `blend = "newest"` that is:

- the adapter's previous chunk was published as `newest_chunk_id` and covers
  ticks from its `k_first_action`;
- this inference's chunk will cover ticks from `k_first_action + P` (steady
  state: triggers are P ticks apart), so its row `j` is the previous chunk's
  row `j + P`;
- it will land about `latency_s` from now, and P6 takes it over at the tick
  executing then, plus two. Rows before that are the previous chunk's and
  will run no matter what.

```python
a = reader.arena
prev = None
if a.control.newest_chunk_id != 0:                        # 0: a rollout's first
    prev = a.chunks.get(a.control.newest_chunk_id)        # None once overwritten
if prev is not None and prev["k_first_action"] + prev["n_actions"] > a.control.k_exec + 2:
    P = self.settings.replan_period                      # from on_settings
    dt = 1.0 / self.settings.control_hz
    k_new_first = int(prev["k_first_action"]) + P
    k_takeover = a.control.k_exec + 2 + int(np.ceil(a.control.latency_s / dt))
    frozen = max(0, k_takeover - k_new_first)            # rows to keep equal to prev
    prefix = prev["pose"][P:P + frozen]                  # what those rows will be
```

Two more cases freeze nothing: a previous chunk already past (the condition
above), and the first inference after a PICO operator hands back — P5 then
re-founds the tick line on `k_exec`, not on the previous chunk plus P; detect
it as a change of `control.intervene` since the last inference.

The published rows are the ones after the deadline shift, and P5 shifts the
new chunk again by its own sub-tick gap, so the frozen rows match what runs to
within a fraction of a tick: condition on them as a soft prefix.
`examples/rtc/policy.py` is this, runnable, with all three cases.

## What is stable

| layer | promise |
|---|---|
| `obs`, `reader.*` | stable; arena changes do not change them |
| `reader.arena`, the ring classes and the rules above | documented, and versioned with `schema.VERSION`; a process built against another version refuses to attach (`LayoutMismatch`), so a mismatch fails at startup, not as a wrong value |
| byte offsets, padding, region order | internal: never hard-code them |

`schema.VERSION` is bumped whenever a record's fields, a region's name or a
read protocol changes — whenever an old reader would misread rather than fail.
The layout hash already catches size and offset differences.

## Size, and `/dev/shm`

The arena is one segment, `/dev/shm/g1d-<pid>`, created by P0 and never
resized. P0 checks `/dev/shm` before creating it, because a tmpfs is sparse:
an arena too big is created without error and the first process to touch a
page past the limit dies of SIGBUS.

For the `g1d_fisheye_dex1` robot (two 720×1280 fisheye wrists, held as ~690 px
squares; the head split into two 480×640 eyes):

| declared camera history | arena |
|---|---|
| none | ~28 MB |
| 4 s of both head eyes | ~375 MB |
| 4 s of every camera | ~1.0 GB |

`head_left` is larger than `head_right` because its slots also carry the wire
JPEG of the whole binocular frame.
