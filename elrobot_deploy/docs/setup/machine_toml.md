# `machine.toml`: this box's overrides (optional)

The machine file is **optional**. Everything about the computer the runtime
runs on is detected or has a default: which network card DDS uses, the CPU
pinning, the dashboard port, where recordings go, environment for the policy
process. Write a machine file only to change one of them, most often to put
recordings on a big disk. Nothing about the robot is here (its address and
login are the robot file's `[board]`, `docs/setup/robot_toml.md`), and nothing
about the task.

- **Code:** `src/elrobot_deploy/config_loading/machine_file.py`
  (`load_machine`, `machine_path`, `records_root`),
  `src/elrobot_deploy/runtime/cpu_pinning.py`.
- **Template:** `machine.example.toml` at the checkout's root.
- **Related:** `docs/setup/robot_toml.md`, `docs/deploying_a_policy/running_and_live_dashboard.md`.

## Where it is

First found wins: `--machine PATH`, then `$ELROBOT_MACHINE`, then
`<checkout>/machine.toml`, then no file (every default applies). Nothing is
created for you and nothing is written to your home directory. To have one:

    cp machine.example.toml machine.toml     # keep only what you change

It is in `.gitignore` (it describes one computer). Each scalar can be
overridden for one shell by the `ELROBOT_*` variable named below. `elrobot_deploy
check` prints what was resolved and from where; every rollout's manifest
records it.

**When one is needed.** Only when the automatic CPU table refuses this
computer (too few cores, or no hyperthreading): then `[cpu.pin]` must be
stated by hand, below.

## Fields

| key | default | env | what it is for |
|---|---|---|---|
| `[network] iface` | `"auto"` | `ELROBOT_IFACE` | the network card DDS binds to. `"auto"` = the first interface with an address on `subnet` |
| `[network] subnet` | `"192.168.123."` | | the robot's DDS network. The G1-D's factory addresses are on 192.168.123.x, so this box needs a static address there on the card cabled to the robot; without one, `auto` finds nothing and the launch says so |
| `[dashboard] port` | 8080 | `ELROBOT_DASHBOARD_PORT` | the live dashboard, on loopback |
| `[paths] records` | unset | `ELROBOT_RECORDS` | this box's record root: a run whose deploy file has no `[record] out` records to `<records>/<task>/`, and `elrobot_deploy inspect` with no argument opens it. Relative to this file; put it on a big disk (~1 GB per rollout). Unset: each deploy file records beside itself |
| `[env]` | | | environment exported into P5 (the policy process) before it imports anything, e.g. `XLA_PYTHON_CLIENT_PREALLOCATE = "false"` on a shared GPU. A deploy file's `[policy.env]` wins over it |
| `[cpu] reserved`, `rt_priority`, `[cpu.pin]` | derived | | CPU pinning, below |
| `[inspect]` | | | defaults for `elrobot_deploy inspect`: `port`, `host`, `modes`, `executor_producer`, `runs` (`docs/deploying_a_policy/rollout_inspection.md`) |

## CPU pinning

Every process is pinned to cores. The 500 Hz servos (P7, P8) cannot miss a
2 ms deadline because a camera decoder or the recorder landed on their core.

- **Without `[cpu.pin]`** the table is derived from this CPU's `/sys`
  topology. `python -m elrobot_deploy.runtime.cpu_pinning` prints the table it
  would derive, to paste if you would rather state one.
- **A stated table is checked, not believed:** each core's real siblings are
  read at startup, and a table that does not match this machine (one copied
  from another box) is refused.
- **Derivation refuses rather than degrading:** a CPU that cannot give P7 and
  P8 a physical core each is one this system should not run on.

| process | gets | why |
|---|---|---|
| `p7_body`, `p8_hand` | a whole physical performance core each, the sibling thread reserved (`[cpu] reserved`); the favoured (highest-clocked) cores when the CPU has them | the 500/200 Hz deadlines |
| `p6_target`, `p5_policy`, `p1_state` | one performance thread each (`p5_policy` may take a list) | latency-sensitive |
| cameras (`p2..p4`, `exo`) | efficiency cores, one per L2 cluster | bulk JPEG decode; decoders sharing a cluster evict each other's cache |
| `p9_record` | the far efficiency cluster | streams megabytes; kept away from the decoders |
| `p0`, `p10`, `p11` | spare efficiency cores | light, but unpinned they could land on a latency core |
| physical core 0 | nothing | left to interrupts |

`[cpu] rt_priority` (e.g. 80) asks for SCHED_FIFO on P7 and P8; it needs root
or `CAP_SYS_NICE`, and a refused request stops the run rather than silently
continuing at normal priority. `tests/machine.toml` is a worked, stated table
for an i9-14900KF.
