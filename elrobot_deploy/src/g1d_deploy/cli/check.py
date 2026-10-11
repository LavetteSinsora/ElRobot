"""`g1d_deploy check` — the one-time setup's two questions, in order: is THIS box
ready to deploy (`check_machine`), and does the robot answer (`check_robot`)?

    g1d_deploy check [--robot NAME|PATH] [--machine PATH] [--no-machine] [--no-robot]

**Why this exists.** The only other entry point is a live run, and a live run
finds its problems in a bad order: a missing `pyrealsense2` surfaces when P2
opens a RealSense, a CPU table from another machine after the arena is mapped, a
dead camera as a CHECK that waits forever. Bringing up a new deploy box would
mean discovering those one at a time, each behind the last, some of them with
an arm already live.

So `check` walks the preconditions in dependency order and reports ALL of
them. They reuse the real checks — `machine_file.load_machine`,
`robot_file.load_robot`, `resolved.load_hardware`,
`cpu_pinning.validate_pinning`, `drivers/dds/domain.py::diagnose`, the
driver sources P1/P2 run — rather than restating them, so they cannot drift into
approving something the run would refuse.

**It never moves the robot or publishes.** The robot half subscribes and
reads; it opens every declared camera exactly as P2 would and reads one frame.

Exit status is 0 only if nothing FAILED. SKIP is not failure: a check that
cannot run here says so and says why.

For a user: `docs/setup/install.md`, "Check the computer and the robot".
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]   # the package root; robot/assets/ ship inside it

_PASS, _FAIL, _SKIP, _WARN = "PASS", "FAIL", "SKIP", "WARN"
_MARK = {_PASS: "  ok  ", _FAIL: " FAIL ", _SKIP: " skip ", _WARN: " warn "}

#: The arena is one mmap in /dev/shm and never grows. Below this there is not
#: room for three cameras' triple buffers plus the rings; the run's own layout
#: prints its exact size at INIT.
_SHM_MIN_MB = 256


class Report:
    """Accumulates results so every check runs, rather than stopping at the
    first failure — the whole point is to see the list."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, str]] = []
        self._tier = ""

    def tier(self, name: str) -> None:
        self._tier = name
        print(f"\n--- {name}")

    def add(self, status: str, what: str, detail: str = "", remedy: str = "") -> None:
        self.rows.append((self._tier, status, what, remedy))
        line = f"[{_MARK[status]}] {what}"
        if detail:
            line += f"  —  {detail}"
        print(line)
        if status in (_FAIL, _WARN) and remedy:
            print(f"           -> {remedy}")

    def check(self, what: str, fn, *, remedy: str = "", skip_if: str = "") -> bool:
        """Run `fn`, which returns a detail string or raises."""
        if skip_if:
            self.add(_SKIP, what, skip_if)
            return False
        try:
            detail = fn()
        except Exception as exc:                       # noqa: BLE001 — report, don't raise
            self.add(_FAIL, what, f"{type(exc).__name__}: {exc}", remedy)
            return False
        self.add(_PASS, what, detail or "")
        return True

    @property
    def failed(self) -> int:
        return sum(1 for _, s, _, _ in self.rows if s == _FAIL)

    def summary(self) -> int:
        n = {k: sum(1 for _, s, _, _ in self.rows if s == k) for k in _MARK}
        print(f"\n{n[_PASS]} ok, {n[_FAIL]} failed, {n[_WARN]} warned, {n[_SKIP]} skipped")
        if self.failed:
            print("\nFAILED:")
            for tier, s, what, remedy in self.rows:
                if s == _FAIL:
                    print(f"  {tier}: {what}")
                    if remedy:
                        print(f"      {remedy}")
        return 1 if self.failed else 0


def _have(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def _raise(msg: str):
    raise RuntimeError(msg)


#: (module, what breaks without it). The RUNTIME's imports only — a learning
#: framework and its friends are a policy's dependencies, never g1d_deploy's, and a policy
#: that needs them finds out when it is loaded. Checked by import NAME, because
#: the distribution name differs for half of these (`pin` -> `pinocchio`,
#: `opencv-python` -> `cv2`, `pyzmq` -> `zmq`) and the import name is what
#: actually fails at runtime.
_PACKAGES = [
    ("numpy",          "everything"),
    ("mujoco",         "IK and FK"),
    ("mink",           "the IK QP"),
    ("qpsolvers",      "the IK QP's solver"),
    ("cyclonedds",     "DDS: every command and every state read"),
    ("cv2",            "JPEG decode in P2"),
    ("zmq",            "the board's cameras"),
    ("pinocchio",      "P7's gravity feedforward"),
    ("paramiko",       "starting the fisheye image server on the board"),
]


# --- the machine ---------------------------------------------------------------


#: (module, extra, what it is for): installed only on request.
_OPTIONAL = [
    ("pyrealsense2", "realsense", "a RealSense camera (the exo profile)"),
    ("pybind11",     "intervene", "building the PICO bindings"),
]


def _packages(rep: Report) -> None:
    rep.tier("PACKAGES")
    for asset in ("robot/assets/unitree_g1d/g1_d.urdf",
                  "robot/assets/unitree_dex1/Dex1_base_link.STL"):
        rep.check(f"asset {asset}",
                  lambda a=asset: "present" if (REPO_ROOT / a).is_file()
                  else _raise("missing"),
                  remedy="incomplete install — the assets ship inside the package")
    for mod, why in _PACKAGES:
        if _have(mod):
            rep.add(_PASS, f"import {mod}")
        else:
            rep.add(_FAIL, f"import {mod}", f"without it: {why}",
                    "run scripts/install.sh (it builds CycloneDDS, then uv sync)")
    for mod, extra, why in _OPTIONAL:
        if _have(mod):
            rep.add(_PASS, f"import {mod}", f"extra `{extra}`")
        else:
            rep.add(_SKIP, f"import {mod}",
                    f"extra `{extra}` not installed; needed only for {why} "
                    f"(uv sync --extra runtime --extra {extra})")


def _load_machine(rep: Report, path):
    from g1d_deploy.config_loading.machine_file import load_machine, machine_path

    machine = None
    if not rep.check("machine file",
                     lambda: str(machine_path(path) or "none (optional): defaults apply")):
        return None

    def _load():
        nonlocal machine
        machine = load_machine(path, resolve_iface=False)
        return machine.describe()

    rep.check("machine file parses", _load)
    return machine


def _network(rep: Report, machine) -> None:
    rep.tier("NETWORK")
    if machine is None:
        rep.add(_SKIP, "network", "the machine file did not load")
        return

    def _iface():
        if not machine.iface:
            _raise(f"no interface holds a {machine.subnet}x address")
        return f"{machine.iface} ({machine.sources.get('iface')})"

    rep.check("DDS interface resolved", _iface,
              remedy="plug the robot cable in and give this box its static "
                     f"{machine.subnet}x address, or set [network] iface")

    def _up():
        state = pathlib.Path(f"/sys/class/net/{machine.iface}/operstate")
        if not state.is_file():
            _raise(f"/sys has no interface {machine.iface!r}")
        s = state.read_text().strip()
        if s not in ("up", "unknown"):
            _raise(f"{machine.iface} is {s}")
        return f"{machine.iface} is {s}"

    rep.check("interface is up", _up,
              skip_if="" if sys.platform == "linux" and machine.iface
              else "needs Linux and a resolved interface",
              remedy="check the cable; `ip link` lists the state")


def _cpu(rep: Report, machine) -> None:
    rep.tier("CPU")
    from g1d_deploy.runtime import cpu_pinning as affinity

    def _count():
        topo = affinity.topology()
        n_logical = len(topo.get("cpus", [])) or os.cpu_count()
        physical = len(topo.get("physical_cores", [])) or None
        p, e = len(topo.get("performance", [])), len(topo.get("efficiency", []))
        return (f"{n_logical} logical CPUs"
                + (f", {physical} physical cores" if physical else "")
                + (f" ({p} P threads, {e} E cores)" if e else ""))

    rep.check("core count", _count,
              skip_if="" if sys.platform == "linux" else f"not Linux ({sys.platform})")

    if machine is None or sys.platform != "linux":
        rep.add(_SKIP, "CPU pin table", "needs Linux and a machine file")
        return

    def _pin():
        from dataclasses import replace

        from g1d_deploy.config_loading.resolved import load_hardware

        m = replace(machine, iface=machine.iface or "lo")
        hw = load_hardware(None, m)
        affinity.validate_pinning(hw)
        return (f"{hw.cpu_source}: {len(hw.cpu)} processes pinned, "
                f"reserved {hw.cpu_reserved}")

    rep.check("CPU pin table matches this machine", _pin,
              remedy="regenerate with `python -m g1d_deploy.runtime.cpu_pinning`, or delete "
                     "[cpu.pin] from the machine file and let it be derived")


def _shm(rep: Report) -> None:
    rep.tier("MEMORY")

    def _free():
        st = os.statvfs("/dev/shm")
        mb = st.f_bavail * st.f_frsize / 1e6
        if mb < _SHM_MIN_MB:
            _raise(f"only {mb:.0f} MB free")
        return f"{mb:.0f} MB free"

    rep.check("/dev/shm has room for the arena", _free,
              skip_if="" if os.path.isdir("/dev/shm") else "no /dev/shm here",
              remedy="sudo mount -o remount,size=8G /dev/shm")


def _gpu(rep: Report, machine) -> None:
    rep.tier("GPU")

    def _smi():
        exe = shutil.which("nvidia-smi")
        if exe is None:
            _raise("nvidia-smi not found: no NVIDIA driver")
        out = subprocess.run((exe, "-L"), capture_output=True, text=True, timeout=10)
        lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
        if out.returncode != 0 or not lines:
            _raise(out.stderr.strip() or "no GPU listed")
        return "; ".join(lines)

    rep.check("GPU visible", _smi,
              remedy="needed only by a policy that uses the GPU; install the "
                     "NVIDIA driver (jax/torch bundle their own CUDA)")


def check_machine(machine=None) -> int:
    rep = Report()
    _packages(rep)
    rep.tier("MACHINE FILE")
    m = _load_machine(rep, machine)
    _network(rep, m)
    _cpu(rep, m)
    _shm(rep)
    _gpu(rep, m)
    return rep.summary()


# --- the robot ----------------------------------------------------------------


def _within(fn, timeout_s: float):
    """`fn()` on a daemon thread, or TimeoutError. The wire reads block with no
    timeout of their own (P1/P2 want exactly that); a check must not."""
    box: dict = {}

    def target():
        try:
            box["value"] = fn()
        except BaseException as exc:            # noqa: BLE001
            box["error"] = exc

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        raise TimeoutError(f"nothing within {timeout_s:.0f} s")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _camera_source(name, spec, hw):
    from g1d_deploy.runtime.drivers.cameras import (RealSenseCamera, V4l2Camera,
                                                    ZmqCamera)

    shape = (spec.height, spec.width, 3)
    if spec.source == "realsense":
        return RealSenseCamera(shape, serial=spec.device)
    if spec.source == "v4l2":
        return V4l2Camera(spec.device, shape)
    return ZmqCamera(spec.host or hw.camera_host, hw.camera_ports[name], shape)


def _cameras(rep: Report, robot, hw, window_s: float) -> None:
    import numpy as np

    from g1d_deploy.runtime.drivers import fisheye_circle_fit as _fit

    rep.tier("CAMERAS")
    if not robot.cameras:
        rep.add(_WARN, "cameras", "the robot file declares none")
    from g1d_deploy.config_loading.resolved import missing_camera_driver

    for name, spec in sorted(robot.cameras.items()):
        def _probe(name=name, spec=spec):
            missing = missing_camera_driver(name, spec.source)
            if missing:
                _raise(missing)
            src = _camera_source(name, spec, hw)
            buf = np.empty((spec.height, spec.width, 3), np.uint8)
            try:
                _within(src.connect, window_s + 5.0)
                t0 = time.monotonic()
                n = 0
                fisheye = spec.kind == "fisheye180"
                pooled = []
                while time.monotonic() - t0 < window_s:
                    if not _within(lambda: src.read_into(buf), window_s + 1.0):
                        break
                    n += 1
                    if fisheye and len(pooled) < _fit.REFIT_FRAMES:
                        pooled.append(_fit.luma(buf).copy())
                if n == 0:
                    _raise("connected, but no frame")
                hz = n / (time.monotonic() - t0)
                line = (f"{spec.source} {getattr(src, 'device', '')}  "
                        f"{spec.height}x{spec.width}  {hz:.1f} Hz")
                if fisheye:
                    # A preview of what a launch would fit: the lens circle and
                    # the square the policy would see.
                    try:
                        line += "; " + _fit.decide(name, spec, pooled).line()
                    except Exception as exc:     # noqa: BLE001 - reported, not raised
                        line += f"; lens NOT found: {exc}"
                return line
            finally:
                try:
                    src.close()
                except Exception:               # noqa: BLE001
                    pass

        if spec.source == "zmq" and hw.image_server == "teleimager":
            remedy = "teleimager is not serving on the board: `sudo systemctl start teleimager`"
        elif spec.source == "zmq":
            remedy = ("the fisheye image server is not running on the board (a run "
                      "starts it over SSH; see src/g1d_deploy/runtime/drivers/"
                      "board/fisheye_image_server/README.md), or "
                      "teleimager is holding the cameras")
        else:
            remedy = f"is it plugged in? source={spec.source} device={spec.device or 'auto'}"
        rep.check(f"camera {name}", _probe, remedy=remedy)


def _streams(rep: Report, hw, window_s: float) -> None:
    rep.tier("STATE")
    if not _have("cyclonedds"):
        rep.add(_SKIP, "state streams", "cyclonedds not installed")
        return
    from g1d_deploy.runtime.drivers import robot_state as ls

    def _low():
        src = ls.DdsLowState(hw)
        _within(src.connect, 10.0)
        out = ls.record(src.dtype)
        t0, n = time.monotonic(), 0
        try:
            while time.monotonic() - t0 < window_s:
                if not src.read(out, timeout=1.0):
                    break
                n += 1
        finally:
            src.close()
        if n == 0:
            _raise(f"no sample on {hw.topic_lowstate}")
        rate = n / (time.monotonic() - t0)
        ok = abs(rate - hw.lowstate_hz) <= 0.10 * hw.lowstate_hz
        return (f"{rate:.0f} Hz (robot file says {hw.lowstate_hz:.0f})"
                + ("" if ok else " — OUTSIDE 10%: a run would refuse"))

    rep.check(f"lowstate {hw.topic_lowstate}", _low,
              remedy="is the robot on, and in the mode that publishes lowstate?")

    def _hand():
        src = ls.DdsHandState(hw)
        _within(src.connect, 10.0)
        out = ls.record(src.dtype)
        try:
            if not _within(lambda: src.read(out), 5.0):
                _raise("no sample")
        finally:
            src.close()
        grip = out["grip"].reshape(2, -1)[:, 0]
        return f"{hw.hand}: grips {grip.round(3).tolist()}"

    rep.check(f"hand state {list(hw.topic_hand_state)}", _hand,
              remedy=("the Dex-1 service on the board: `sudo ./dex1_1_gripper_server "
                      "--network eth0 -l -r`" if hw.hand == "dex1"
                      else "the BrainCo hand service on the board"))


def check_robot(robot=None, machine=None, window_s: float = 2.0) -> int:
    from g1d_deploy.config_loading.resolved import load_hardware
    from g1d_deploy.config_loading.machine_file import load_machine
    from g1d_deploy.config_loading.robot_file import DEFAULT_ROBOT, load_robot

    rep = Report()
    rep.tier("FILES")
    loaded: dict = {}

    def _robot():
        loaded["robot"] = load_robot(robot or DEFAULT_ROBOT)
        return loaded["robot"].describe()

    def _machine():
        loaded["machine"] = load_machine(machine)
        return loaded["machine"].describe()

    rep.check("robot file", _robot)
    rep.check("machine file", _machine,
              remedy="see the machine checks; the NIC must be resolvable to talk DDS")
    if "robot" not in loaded or "machine" not in loaded:
        return rep.summary()

    def _hw():
        loaded["hw"] = load_hardware(loaded["robot"], loaded["machine"])
        return f"hand={loaded['hw'].hand} cameras={sorted(loaded['robot'].cameras)}"

    if not rep.check("robot on this machine", _hw):
        return rep.summary()
    hw = loaded["hw"]

    rep.tier("NETWORK")
    from g1d_deploy.runtime.drivers.dds import domain as dds

    linux = "" if sys.platform == "linux" else "needs `ip` and `ping -I` (Linux)"

    def _bind():
        d = dds.diagnose(hw.dds_iface, hw.camera_host, ping=False, ownip=None)
        if not d.nic or d.bound is None:
            _raise(d.explain())
        return f"{hw.dds_iface} binds {d.bound}"

    def _reach():
        d = dds.diagnose(hw.dds_iface, hw.camera_host, ownip=None)
        if d.bound is None or not d.reach:
            _raise(d.explain())
        return f"{hw.camera_host} answers from {d.bound}"

    rep.check("DDS interface", _bind, skip_if=linux)
    rep.check("robot board", _reach, skip_if=linux,
              remedy="power the robot; check the cable from the NIC above")

    _streams(rep, hw, window_s)
    _cameras(rep, loaded["robot"], hw, window_s)
    return rep.summary()


def check(robot=None, machine=None, window_s: float = 2.0, *,
          do_machine: bool = True, do_robot: bool = True) -> int:
    """`g1d_deploy check`: the machine's checks, then the robot's. Both halves always
    run when asked for — a machine failure does not hide what the robot would
    have said. Exit status 0 only if neither failed."""
    rc = 0
    if do_machine:
        print("=== MACHINE")
        rc = max(rc, check_machine(machine))
    if do_robot:
        print("\n=== ROBOT")
        rc = max(rc, check_robot(robot=robot, machine=machine, window_s=window_s))
    return rc
