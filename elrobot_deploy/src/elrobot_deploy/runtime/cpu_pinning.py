"""CPU pinning, and the check that the pinning actually means what it says.

P7 owns a physical core, and its sibling logical core is left idle so nothing
shares that core with the 500 Hz loop. A table that places another process on
P7's sibling costs exactly the jitter the reservation exists to prevent.

`validate_pinning` is what makes the reservation real. It is a startup refusal, not a
watchdog: it runs once, before anything is spawned, and it inspects the
*configuration* plus the machine's own topology. Nothing polls, and nothing
reacts at run time.

Off Linux there is no `sched_setaffinity` and no `/sys` topology, so pinning is
a no-op and the checks that need real cores are skipped. That is what lets the
whole system be exercised on a laptop.

Every process calls `pin(hw, "<its name>")` first thing; P0 runs
`validate_pinning` at resolution. The table comes from the machine file's
`[cpu.pin]` or from `derive` (this machine's topology). Design:
`docs/setup/machine_toml.md`, "CPU pinning".
"""

from __future__ import annotations

import os
from pathlib import Path

_SIBLINGS = Path("/sys/devices/system/cpu")

# The processes that must not share a physical core with anything else.
ISOLATED = ("p7_body", "p8_hand")

#: Placement rules. An E core is roughly half a P core's
#: single-thread throughput: a good trade for work with a generous deadline, a
#: bad one for anything on a latency path.
#:
#: `p5_policy` is listed with a share > 1 because its preprocessing sits INSIDE
#: the inference latency, and inference latency is what sets the replan period.
#: It is the most placement-sensitive non-real-time process in the system.
PERFORMANCE = ("p6_target", "p5_policy", "p1_state")
EFFICIENCY = ("p2_camera", "p3_camera", "p4_camera")
#: The recorder is an efficiency-core process too, but allocated from the FAR
#: END: E cores are clustered (four per shared L2 on Intel hybrid parts), and the one
#: process that streams megabytes should not share L2 with the three that decode
#: them.
BULK_FAR = "p9_record"
#: The light processes: a 180 Hz network poll, a web server that renders a
#: MuJoCo scene when polled, and the launcher itself. Not latency-path work,
#: but NOT unpinned either — an unpinned process can land on `p1_state`'s or
#: `p7_body`'s core, and pinning restricts only the pinned. One recorded run
#: had 29 stalls of ~90 ms in P1's reception with the dashboard open; whether
#: that was a stray process on P1's core or the link is what this rules out. They
#: take an E core each, spread one per cluster like the decoders, beside a
#: decoder rather than beside the recorder.
LIGHT = ("p11_pico", "p10_dashboard", "p0_launcher")
#: Extra performance threads handed to P5 once every other process has one.
GREEDY = "p5_policy"


def available() -> bool:
    """True where pinning is enforceable at all."""
    return hasattr(os, "sched_setaffinity")


def cores_for(hw, name: str) -> set[int]:
    """The logical CPUs assigned to `name`, as a set.

    `cpu.pin` accepts an int or a list: everything holds one thread, except P5,
    which is allowed several because OpenCV and numpy will use them and its
    latency is the number that sets `P`.
    """
    value = hw.cpu.get(name)
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {int(v) for v in value}
    return {int(value)}


def pin(hw, name: str) -> bool:
    """Pin the calling process to the cores the machine file names for it."""
    cores = cores_for(hw, name)
    if not cores or not available():
        return False
    os.sched_setaffinity(0, cores)
    return True


def set_realtime(hw, name: str, *, error=RuntimeError) -> bool:
    """Put a real-time loop on `SCHED_FIFO`, if the site file asks for it.

    This is the only isolation mechanism available to a process about itself.
    Everything that genuinely keeps the kernel off a core — `isolcpus`,
    `nohz_full`, a cgroup v2 isolated partition, interrupt affinity — is either
    a boot parameter or a machine-wide mutation that would outlive a crashed
    run, so none of it belongs in a launcher.

    What this buys is narrower than isolation and closer to what actually
    matters: it does not stop other tasks being scheduled on the core, it stops
    them *delaying* this loop. A normal-priority task there is preempted within
    scheduler latency.

    Denied is a refusal, not a warning (`rt_priority` unset is how you ask for
    nothing). A run that was configured for real-time priority and quietly did
    not get it is indistinguishable from one that did, right up until the
    jitter — and the whole point of the setting is that you know which regime
    you are in.
    """
    prio = getattr(hw, "rt_priority", None)
    if prio is None:
        return False
    if not hasattr(os, "sched_setscheduler"):
        raise error(
            f"machine.toml [cpu] rt_priority = {prio} but this platform has no "
            f"sched_setscheduler. Unset it to run here.")
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(int(prio)))
    except (PermissionError, OSError) as exc:
        raise error(
            f"{name}: SCHED_FIFO priority {prio} was refused ({exc}). Real-time "
            f"scheduling needs root or CAP_SYS_NICE. Either grant it, or unset "
            f"[cpu] rt_priority — running at normal priority while the site "
            f"file says otherwise is the one outcome worth refusing.") from exc
    return True


def isolated() -> set[int]:
    """The CPUs the kernel was booted to keep off the general scheduler.

    `/sys/devices/system/cpu/isolated` is the kernel's own account of
    `isolcpus=`, so this reports what the machine is actually doing rather than
    what the boot config was meant to say.
    """
    spec = _read(_SIBLINGS / "isolated")
    return set(_parse_list(spec)) if spec else set()


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _parse_list(spec: str) -> list[int]:
    """Parse the kernel's "0-7,16,18-19" CPU list format."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def _max_freq(cpus: list[int]) -> dict[int, int]:
    """Each CPU's HARDWARE maximum frequency in kHz, from cpufreq.

    `cpuinfo_max_freq` is the silicon's ceiling — it does not move with the
    governor, the current thermal state, or what is running. On a part with
    Intel's favoured cores (Turbo Boost Max 3.0 / TVB) two P cores are binned
    higher than the rest and this is where that shows up: on the deploy box's
    i9-14900KF, cpu8-11 read 6000000 and every other P thread reads 5700000.

    Empty where cpufreq is not exposed (no sysfs, some VMs, macOS). Callers
    must treat an empty map as "no preference", never as "all equal at zero".
    """
    out: dict[int, int] = {}
    for c in cpus:
        raw = _read(_SIBLINGS / f"cpu{c}" / "cpufreq" / "cpuinfo_max_freq")
        if raw and raw.isdigit():
            out[c] = int(raw)
    return out


def _l2_group(cpu: int) -> tuple[int, ...] | None:
    """The logical CPUs sharing an L2 with `cpu`, or None if unreadable.

    Found by walking `cache/index*` for the one whose `level` is 2 rather than
    assuming `index2`: the index numbering is per-CPU and an L1i/L1d/L2 ordering
    is a convention, not a guarantee.
    """
    base = _SIBLINGS / f"cpu{cpu}" / "cache"
    try:
        indices = sorted(base.glob("index*"))
    except OSError:
        return None
    for idx in indices:
        if _read(idx / "level") == "2":
            spec = _read(idx / "shared_cpu_list")
            if spec:
                return tuple(sorted(_parse_list(spec)))
    return None


def _l2_groups(cpus: list[int]) -> list[tuple[int, ...]]:
    """The machine's L2 domains, in ascending order, deduplicated.

    On Intel hybrid parts a P core has an L2 to itself (shared only with its SMT
    sibling) and E cores come in CLUSTERS of four sharing one. That grouping is
    the whole basis of the `BULK_FAR` rule and, on a part with enough clusters,
    of spreading the camera decoders — so it is read from the machine rather
    than inferred from `cpu // 4`, which is an enumeration habit that happens to
    hold on the two parts we have owned.

    Empty when any CPU cannot be read, because a partial grouping would place
    processes against a map that is only sometimes true.
    """
    groups: list[tuple[int, ...]] = []
    for c in cpus:
        g = _l2_group(c)
        if g is None:
            return []
        if g not in groups:
            groups.append(g)
    return sorted(groups)


def topology() -> dict:
    """The machine's own account of itself: which logical CPUs are P, which are
    E, which share a physical core, which share an L2, and how fast each can go.

    P/E come from the hybrid PMUs (`/sys/devices/cpu_core/cpus` and
    `cpu_atom/cpus`), which is how Intel exposes the split on Linux — not from
    max frequency, and not from "the low-numbered ones are P", which is an
    enumeration habit. On a non-hybrid machine there is no `cpu_atom` and every
    CPU is reported as performance.

    `max_freq` and `l2_groups` are ADVISORY: `derive` uses them to break ties it
    would otherwise break by cpu id, and falls back to id order when either is
    empty. That is what keeps a synthetic topology — a test's, or a machine
    without cpufreq — deriving the same table it always did.

    Returns `{}` where none of this is readable (macOS, containers without
    /sys), which is what the callers treat as "cannot verify".
    """
    online = _read(_SIBLINGS / "online")
    if online is None:
        return {}
    cpus = _parse_list(online)

    p_spec = _read(Path("/sys/devices/cpu_core/cpus"))
    e_spec = _read(Path("/sys/devices/cpu_atom/cpus"))
    p_cpus = set(_parse_list(p_spec)) if p_spec else set(cpus)
    e_cpus = set(_parse_list(e_spec)) if e_spec else set()

    cores: dict[tuple[int, ...], list[int]] = {}
    for c in cpus:
        key = tuple(sorted(siblings(c)))
        cores.setdefault(key, []).append(c)

    return {"cpus": cpus, "performance": sorted(p_cpus), "efficiency": sorted(e_cpus),
            "physical_cores": sorted(cores),
            "max_freq": _max_freq(cpus), "l2_groups": _l2_groups(cpus)}


def derive(topo: dict | None = None, *, error=RuntimeError) -> dict:
    """The placement rules applied to THIS machine's real topology.

    Returns `{"pin": {name: [cpus]}, "reserved": [cpus]}`. Order of allocation
    is the order of scarcity: the real-time loops take whole physical
    performance cores first, then the latency-path processes take one thread
    each, then the bulk processes take efficiency cores, and whatever
    performance threads remain go to P5 — the one process that can use more
    than one and whose latency sets the replan period.

    Physical core 0 is never allocated: it carries the bulk of the interrupt
    load, and it is the one core nothing of ours should sit on.

    **Which** cores, where the machine reports enough to choose (both measured
    on an i9-14900KF):

    * The isolated loops take the FASTEST whole cores, not the lowest-numbered.
      Intel bins two P cores higher than the rest (6.0 GHz vs 5.7 on that part)
      and by cpu id those land on `p1_state`, a process with roughly 6% duty.
      Ties, and any machine with no cpufreq, fall back to id order.
    * The camera decoders take one core from each of SEPARATE L2 clusters when
      there are more clusters than cameras. E cores share an L2 four at a time;
      three JPEG decoders in one cluster evict each other for no reason on a
      part with four clusters and eight idle E cores. With only two clusters
      (an 8-E-core part) there is no spare and they pack together.

    Neither is a claim about achieved latency — `p7_body` spins on
    `time.monotonic()`, and the TSC is invariant, so a faster core does not make
    the CLOCK faster. What it makes faster is the work between deadline checks.

    Refuses rather than doubling up. A machine that cannot give P7 and P8 a
    whole core each is a machine this system should not start on, and saying so
    is far better than quietly sharing a core with the 500 Hz loop.
    """
    topo = topology() if topo is None else topo
    if not topo:
        raise error("CPU topology unreadable — cannot derive a pin table. Run "
                    "on the deploy PC, or state [cpu.pin] explicitly.")

    perf, eff = set(topo["performance"]), list(topo["efficiency"])
    freq = topo.get("max_freq") or {}
    whole = [c for c in topo["physical_cores"]
             if set(c) <= perf and 0 not in c and len(c) > 1]
    # Fastest first, id order within a speed. With no cpufreq every core scores
    # 0 and this is a no-op re-sort of an already-sorted list.
    whole.sort(key=lambda core: (-max(freq.get(c, 0) for c in core), core[0]))
    if len(whole) < len(ISOLATED):
        raise error(
            f"{len(whole)} hyperthreaded performance core(s) available, but "
            f"{len(ISOLATED)} are needed so {', '.join(ISOLATED)} can each own "
            f"one with its sibling idle. Refusing rather than "
            f"sharing a core with a real-time loop.")

    pin: dict[str, list[int]] = {}
    reserved: list[int] = []
    for name, core in zip(ISOLATED, whole):
        pin[name] = [core[0]]
        reserved.extend(core[1:])

    taken = {c for pair in whole[:len(ISOLATED)] for c in pair}
    free_perf = [c[0] for c in topo["physical_cores"]
                 if set(c) <= perf and 0 not in c and not set(c) & taken]
    if len(free_perf) < len(PERFORMANCE):
        raise error(
            f"{len(free_perf)} performance thread(s) left for "
            f"{len(PERFORMANCE)} latency-path processes ({', '.join(PERFORMANCE)}). "
            f"Putting any of them on an efficiency core roughly halves its "
            f"single-thread throughput.")
    for name, cpu in zip(PERFORMANCE, free_perf):
        pin[name] = [cpu]

    spare_perf = list(free_perf[len(PERFORMANCE):])
    bulk = list(EFFICIENCY) + [BULK_FAR]
    notes: list[str] = []

    if eff:
        # One camera per L2 cluster when there is a cluster to spare for the
        # recorder as well; otherwise the first free E cores, as before.
        clusters = [g for g in (topo.get("l2_groups") or []) if set(g) <= set(eff)]
        spread = ([g[0] for g in clusters[:len(EFFICIENCY)]]
                  if len(clusters) > len(EFFICIENCY) else [])
        for i, name in enumerate(EFFICIENCY):
            if spread:
                pin[name] = [spread[i]]
            elif i < len(eff):
                pin[name] = [eff[i]]
            else:
                notes.append(f"{name}: no efficiency core left")
        far = [c for c in reversed(eff)
               if c not in {v for u in pin.values() for v in u}]
        if far:
            pin[BULK_FAR] = [far[0]]
        else:
            notes.append(f"{BULK_FAR}: no efficiency core left")
    else:
        # A NON-HYBRID machine: `topology()` reports every CPU as performance
        # when there is no `cpu_atom` PMU. The bulk processes then have no E
        # core to go to; leaving them UNPINNED would be four processes with no
        # placement and nothing said about it.
        #
        # Place them on spare performance threads instead and say so. There is
        # no L2-cluster argument to make here (the BULK_FAR rule is about E
        # cores sharing an L2 in fours), so they are simply spread one per
        # thread, and `p5_policy` gets whatever is left after them rather than
        # before.
        if len(spare_perf) < len(bulk):
            raise error(
                f"no efficiency cores on this machine, and only "
                f"{len(spare_perf)} spare performance thread(s) for the "
                f"{len(bulk)} bulk processes ({', '.join(bulk)}). This CPU is "
                f"too small to run the deploy system: it needs "
                f"{1 + len(ISOLATED) + len(PERFORMANCE) + len(bulk)} physical "
                f"cores at minimum, counting core 0 for interrupts.")
        for name in bulk:
            pin[name] = [spare_perf.pop(0)]
        notes.append(
            "non-hybrid CPU: no efficiency cores, so "
            f"{', '.join(bulk)} were placed on performance threads")

    # The light processes: one free E core per cluster (the third of each, the
    # second is left for an exo camera), else any free E core, else a spare
    # performance thread, else unpinned with a note — never a refusal.
    used = {v for u in pin.values() for v in u}
    free_eff = [c for c in eff if c not in used]
    clusters = [g for g in (topo.get("l2_groups") or []) if set(g) <= set(eff)]
    for i, name in enumerate(LIGHT):
        pick = None
        if i < len(clusters) and len(clusters[i]) > 2 and clusters[i][2] in free_eff:
            pick = clusters[i][2]
        elif free_eff:
            pick = free_eff[0]
        elif spare_perf:
            pick = spare_perf.pop(0)
        if pick is None:
            notes.append(f"{name}: no core left, runs unpinned")
            continue
        pin[name] = [pick]
        if pick in free_eff:
            free_eff.remove(pick)

    # Whatever performance threads are left over go to the one process that can
    # use them. Nothing else in the system is multi-threaded on the CPU side.
    #
    # `setdefault` rather than `pin[GREEDY].append`: GREEDY happens to be in
    # PERFORMANCE today, so the key exists, but that is an undeclared coupling
    # between two module constants and a KeyError for whoever edits one.
    for cpu in spare_perf:
        pin.setdefault(GREEDY, []).append(cpu)

    return {"pin": {k: sorted(v) for k, v in pin.items()},
            "reserved": sorted(reserved),
            "notes": notes}


def suggest(topo: dict | None = None) -> str:
    """`derive()` rendered as a `[cpu]` block to paste into the machine file."""
    try:
        table = derive(topo)
    except RuntimeError as exc:
        return f"# {exc}"
    lines = ["[cpu]", f"reserved = {table['reserved']}", "", "[cpu.pin]"]
    for name in ("p1_state", "p2_camera", "p3_camera", "p4_camera",
                 "p5_policy", "p6_target", "p7_body", "p8_hand", "p9_record"):
        cpus = table["pin"].get(name)
        if cpus:
            lines.append(f"{name:<10}= {cpus[0] if len(cpus) == 1 else cpus}")
    return "\n".join(lines)


def siblings(core: int) -> set[int]:
    """The logical cores sharing a physical core with `core`, `core` included.

    Read from the kernel rather than computed: the common "core N and N + n/2
    are siblings" rule is an x86 enumeration convention, not a guarantee, and
    guessing it wrong silently reserves the wrong core.
    """
    path = _SIBLINGS / f"cpu{int(core)}" / "topology" / "thread_siblings_list"
    try:
        raw = path.read_text().strip()
    except OSError:
        return {int(core)}
    out: set[int] = set()
    for part in raw.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


def validate_pinning(hw, *, error=RuntimeError) -> None:
    """Refuse a pin table that cannot deliver the core reservation.

    Four ways it can fail, in the order they are worth catching:

    1. Two processes on one logical core — always wrong, and the only check
       that does not need Linux.
    2. A process pinned to a core reserved as somebody's idle sibling, which
       reads as "reserved" in the file and "shared" on the machine.
    3. An isolated process whose sibling is not reserved. This is the one the
       reservation exists for, and the one nobody notices: P7 gets its own
       *logical* core, another process gets the other half of the same
       *physical* core, and the 500 Hz loop is contended by a decoder.
    4. An isolated process with no pin at all, where pinning is enforceable.
    """
    pins: dict[str, set[int]] = {k: cores_for(hw, k) for k in hw.cpu}
    reserved = {int(c) for c in hw.cpu_reserved}

    seen: dict[int, str] = {}
    for name, cores in sorted(pins.items()):
        for core in sorted(cores):
            if core in seen:
                raise error(
                    f"machine.toml [cpu.pin]: {name} and {seen[core]} are both "
                    f"pinned to core {core}.")
            seen[core] = name

    for name, cores in sorted(pins.items()):
        for core in sorted(cores):
            if core in reserved:
                raise error(
                    f"machine.toml: {name} is pinned to core {core}, which "
                    f"[cpu] reserved lists as an idle sibling.")

    # Config-only, so it runs everywhere: a real-time loop owning several cores
    # would defeat the sibling reservation it depends on, and that is true
    # regardless of whether this machine can enforce pinning at all.
    for name in ISOLATED:
        if len(pins.get(name, ())) > 1:
            raise error(
                f"machine.toml [cpu.pin]: {name} holds {sorted(pins[name])}. A "
                f"real-time loop owns ONE logical core with its sibling idle; "
                f"several cores would defeat the reservation it depends on.")

    if not available():
        return

    for name in ISOLATED:
        if not pins.get(name):
            raise error(
                f"machine.toml [cpu.pin] has no core for {name}. On Linux the "
                f"real-time loops must be pinned; pick a core and "
                f"reserve its sibling (see `lscpu -e` or "
                f"/sys/devices/system/cpu/cpuN/topology/thread_siblings_list).")
        core = next(iter(pins[name]))
        others = siblings(core) - {core}
        unreserved = others - reserved
        if unreserved:
            raise error(
                f"machine.toml: {name} is pinned to core {core}, whose "
                f"sibling logical core(s) {sorted(unreserved)} are not in "
                f"[cpu] reserved. Anything scheduled there shares the physical "
                f"core with a real-time loop.")


def main() -> None:
    """`python -m elrobot_deploy.runtime.cpu_pinning` on the deploy PC.

    Prints what the machine reports and the `[cpu]` block that follows from it,
    so the site file records a measurement rather than an assumption about how
    Linux numbered the cores.
    """
    topo = topology()
    if not topo:
        print("CPU topology unreadable (not Linux?). Run this on the deploy PC.")
        return
    perf, eff = topo["performance"], topo["efficiency"]
    print(f"{len(topo['cpus'])} logical CPUs, "
          f"{len(topo['physical_cores'])} physical cores")
    print(f"  performance: {perf}")
    print(f"  efficiency : {eff if eff else '(none — not a hybrid CPU)'}")
    freq = topo.get("max_freq") or {}
    print("  physical cores (logical siblings):")
    for c in topo["physical_cores"]:
        kind = "P" if set(c) <= set(perf) else "E"
        mhz = max((freq.get(x, 0) for x in c), default=0) // 1000
        fastest = mhz and mhz == max(v for v in freq.values()) // 1000
        print(f"    {kind} {list(c)}"
              + (f"  {mhz} MHz{'  <- favoured' if fastest else ''}" if mhz else ""))
    clusters = topo.get("l2_groups") or []
    if clusters:
        print("  L2 domains (cores that evict each other):")
        for g in clusters:
            if len(g) > 2:
                print(f"    {list(g)}")
    print()
    print("# --- paste into machine.toml (the project root's) ---")
    print(suggest(topo))
    print()
    print("# Pinning keeps OUR processes off the reserved siblings. To keep the")
    print("# KERNEL's other work off them too, add to the boot cmdline:")
    print("#   isolcpus=<reserved> nohz_full=<reserved> rcu_nocbs=<reserved>")


if __name__ == "__main__":
    main()
