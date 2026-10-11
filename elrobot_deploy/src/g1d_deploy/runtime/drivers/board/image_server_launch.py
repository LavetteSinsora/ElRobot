"""Start the fisheye image server on the robot board, and stop the vendor's.

Called by P0 at CHECK when the robot file says `[image_server] type =
"fisheye"` (`runtime/launcher/startup_checks.py`); `docs/setup/robot_toml.md`,
"`[image_server]`". The server itself is `fisheye_image_server/` beside this
file (its README.md describes the board).

The wrists are on the board, not on this machine, so SOME process over there has
to read the physical devices and forward frames. What this removes is the
operator having to remember (paramiko, `bash -lc` so conda's `.bashrc` hook
actually runs). One directory per robot (the robot file's `[image_server]
dir`), synced from this package and restarted whenever its code differs.

**Why this stops teleimager.** The vendor
server's `reload_uvc_driver()` runs `sudo modprobe -r uvcvideo` at startup, and
its systemd unit is `Restart=always`. Exactly one server can own the cameras, and
leaving the vendor's running does not produce a port conflict you would notice —
it produces a `uvcvideo` removal at some unpredictable later moment, which reads
as cameras dying mid-run. So the stop is part of starting, not a courtesy.

Stopping it takes the vendor's HEAD stream (port 55555) down with the wrists,
which breaks anything else that reads it (the vendor's teleop). `stop()` puts
it all back.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

#: The server's source in THIS package — what `ensure_running` syncs to the board. Where
#: it lives on the board (`dir`), and the conda it runs in, are the robot
#: file's `[image_server]`: one path per robot, not a search.
LOCAL_DIR = Path(__file__).resolve().parent / "fisheye_image_server"
#: The vendor systemd unit that must not be running at the same time.
VENDOR_UNIT = "teleimager"

DEFAULT_SSH_USER = os.environ.get("G1D_BOARD_SSH_USER", "unitree")
DEFAULT_SSH_PASSWORD = os.environ.get("G1D_BOARD_SSH_PASSWORD", "123")
#: Set to skip all of this — for driving a board you started by hand, or for
#: running against the vendor server deliberately.
SKIP_ENV = "G1D_NO_BOARD_AUTOSTART"

REMOTE_LOG = "/tmp/fish_eye_image_server.log"

#: The files the board must run byte-for-byte: everything the publisher imports
#: or reads. README.md and __pycache__ are not code the board executes.
SYNCED_SUFFIXES = (".py", ".yaml")

#: `<publisher pid> <digest of the code it was started from>`, written by
#: `ensure_running` once a launch is confirmed. A file on disk that matches is
#: NOT enough: python has the old module in memory until the process restarts
#: — a board can run a stale exposure law for a week that way. /tmp, so a
#: reboot (which also ends the publisher) takes the stamp with it.
STAMP = "/tmp/fish_eye_image_server.stamp"

#: The publisher's own command line, ANCHORED AT THE START — and the anchor is
#: the whole point.
#:
#: `pgrep -f` and `pkill -f` match against the full command line of every
#: process, **including the shell that is running the pgrep**. We invoke through
#: `bash -lc '...'`, so a bare substring pattern appears verbatim in that
#: shell's own argv and matches it. Measured on the board: an
#: unanchored `pgrep -f fish_eye_image_server/publisher.py` returned the PID of
#: the `bash -c` wrapper asking the question, which would have made
#: `_is_running()` answer True forever — so `ensure_running` would decide the
#: server was already up and never start it, and `stop()` would kill its own
#: shell instead of the publisher.
#:
#: The launcher runs `python publisher.py` from the remote dir, so that process's
#: command line BEGINS with `python`, while every shell wrapping it begins with
#: `bash`. Anchoring on `^` is what separates them.
#: NO BACKSLASHES, and that is load-bearing. This pattern is interpolated with
#: `repr()` into a shell command, and `repr()` escapes a backslash — so `\.`
#: reached `pgrep` as `\\.`, an escaped BACKSLASH followed by `py`, which
#: matches nothing. `_is_running` therefore answered False forever: the
#: autostart launched a SECOND publisher on top of the running one, which died
#: on "Resource busy" holding the cameras hostage. `[.]` is the same character
#: class with no escape to mangle.
PROCESS_PATTERN = r"^python[0-9.]* publisher[.]py"


class BoardImagerError(RuntimeError):
    """The board's image server could not be confirmed running."""


def _connect(host: str, ssh_user: str, ssh_password: str, timeout: float):
    import paramiko          # lazy: a v4l2/realsense-only rig never pays for it

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # ALL THREE timeouts, not just the TCP one. `timeout` bounds the connect;
    # paramiko's banner and auth phases have their own, and left at their
    # defaults an unresponsive sshd stalls here with no ceiling. This runs
    # inside CHECK, before a single camera process exists, so a stall is
    # indistinguishable from a hung startup.
    client.connect(host, username=ssh_user, password=ssh_password,
                   timeout=timeout, banner_timeout=timeout,
                   auth_timeout=timeout)
    return client


#: Ceiling on any single remote command. Generous for a pgrep or a systemctl,
#: short enough that a wedged one surfaces as an error inside CHECK rather than
#: as a run that never starts.
COMMAND_TIMEOUT = 20.0


def _run(client, command: str, timeout: float = COMMAND_TIMEOUT) -> str:
    """One command over an existing connection; stripped stdout.

    `bash -lc` rather than a bare exec: conda's activation hook lives in the
    profile, which a non-interactive SSH channel does not source, so
    `conda activate` is a no-op without it.

    **Timed out, always.** `exec_command` without one returns a channel whose
    `read()` blocks forever, so any remote command that never closes its stdout
    — a login profile waiting on a tty, an sshd under load — hangs P0 in CHECK
    with no output and no ceiling (observed).
    """
    _stdin, stdout, _stderr = client.exec_command(
        f"bash -lc {command!r}", timeout=timeout)
    try:
        return stdout.read().decode().strip()
    except Exception as exc:            # noqa: BLE001 - socket.timeout and kin
        raise BoardImagerError(
            f"the board did not answer {command!r} within {timeout:.0f}s "
            f"({exc}). The image server may still be fine — start it by hand "
            f"and re-run with G1D_NO_BOARD_AUTOSTART=1.") from exc


def _sudo(client, command: str, password: str) -> str:
    """`sudo -S` reads the password from stdin, which is the only way to get one
    in over a non-interactive channel."""
    return _run(client, f"echo {password} | sudo -S {command} 2>/dev/null")


def _is_running(client) -> bool:
    return bool(_run(client, f"pgrep -f {PROCESS_PATTERN!r} || true"))


#: What `systemctl is-active` reports for a unit that is genuinely not holding
#: anything. Everything else — `active`, `activating`, `deactivating`,
#: `reloading` — means it has, or is about to have, the cameras.
VENDOR_GONE = ("inactive", "failed", "unknown", "")


def _vendor_active(client) -> bool:
    """Is the vendor unit holding, or about to hold, the cameras?

    **Not `== "active"`, and the difference is a race we lose.** `teleimager` is
    `Restart=always` with `RestartSec=5`, and on the reference board it crash-loops:
    sampled, it went `active` -> `activating` -> `active` on a five
    second cycle, because systemd reports the RestartSec wait as `activating`.

    An equality test against `"active"` therefore answers False for several
    seconds out of every cycle. Land in that window and `ensure_running` skips
    the stop, starts our publisher, and teleimager comes back moments later and
    takes the devices — running `modprobe -r uvcvideo` on its way in, which is
    the "cameras died mid-run" failure the module docstring describes. The
    window is wide (~5 s in ~10) and nothing about hitting it is unlikely.

    So the question is inverted: gone is a short, closed list, and anything not
    on it counts as present. Being wrong in this direction costs one redundant
    stop; being wrong in the other costs the run.
    """
    state = _run(client, f"systemctl is-active {VENDOR_UNIT} || true")
    return state not in VENDOR_GONE


def _stop_vendor(client, password: str, timeout: float) -> None:
    """Stop the vendor unit without waiting out systemd's stop timeout.

    **A plain `systemctl stop teleimager` blocks for ninety seconds**, so a
    shorter timeout on it can never succeed. Measured on the board, straight
    out of the journal::

        10:00:10  Stopping Teleimager Image Server...
        10:01:40  teleimager.service: Failed with result 'timeout'.

    Ninety seconds exactly, which is the unit's `TimeoutStopSec` default. The
    unit's `ExecStart` is a COMPOUND command — `source … && conda activate tv &&
    teleimager-server` — so bash cannot `exec` into the server and the MainPID
    is bash, not the server. systemd SIGTERMs the cgroup, `teleimager-server`
    does not exit on it (it is tearing down usbfs and the uvcvideo module, which
    is the whole reason it must not run beside ours), and systemd waits out the
    timeout before the SIGKILL that was always going to be needed.

    So: do not wait for it. The two commands must go in THIS ORDER.

    `stop --no-block` first, because a manual stop is what cancels
    `Restart=always`. The unit also carries `RestartSec=5`, and the same journal
    shows it exercising that — stopped at 10:00:05, back up by 10:00:10 — so a
    kill on its own buys five seconds and then loses the camera again.

    `kill -s KILL` second, which reaches the whole control group immediately and
    takes the child the SIGTERM never persuaded. With the stop already in
    flight, nothing restarts it.

    Polling rather than one shot because `is-active` is the only report that
    matters and the kill is idempotent. The vendor is gone in a second or two,
    against ninety.
    """
    _sudo(client, f"systemctl stop --no-block {VENDOR_UNIT}", password)
    deadline = time.monotonic() + timeout
    while True:
        if not _vendor_active(client):
            return
        if time.monotonic() >= deadline:
            raise BoardImagerError(
                f"{VENDOR_UNIT} is still active {timeout:.0f}s after `systemctl "
                f"stop --no-block` and repeated `systemctl kill -s KILL`. It "
                f"holds the cameras through usbfs, so ours cannot claim them "
                f"while it lives. SSH in and look: `systemctl status "
                f"{VENDOR_UNIT}`.")
        _sudo(client, f"systemctl kill -s KILL {VENDOR_UNIT}", password)
        time.sleep(0.5)


def local_manifest(local_dir: Path = LOCAL_DIR) -> dict[str, str]:
    """`{file name: sha256}` of the server this package ships."""
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(local_dir.iterdir())
            if f.is_file() and f.suffix in SYNCED_SUFFIXES}


def manifest_digest(manifest: dict[str, str]) -> str:
    """One short name for a whole manifest, for the stamp and the log."""
    lines = "".join(f"{h}  {n}\n" for n, h in sorted(manifest.items()))
    return hashlib.sha256(lines.encode()).hexdigest()[:16]


def _remote_manifest(client, remote_dir: str, names) -> dict[str, str]:
    """`{file name: sha256}` for those of `names` present in `remote_dir`.
    A missing file (or directory) is simply absent from the result."""
    out = _run(client, f"cd {remote_dir} 2>/dev/null && "
                       f"sha256sum {' '.join(sorted(names))} 2>/dev/null; true")
    found = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2:
            found[parts[1].lstrip("*")] = parts[0]
    return found


def _sync(client, remote_dir: str, local: dict[str, str],
          local_dir: Path = LOCAL_DIR) -> list[str]:
    """Make the board's copy byte-identical to ours; returns the names rewritten.

    Each file goes up as `<name>.g1d-tmp` and is RENAMED into place, so a
    publisher starting mid-sync never imports half a file. Verified by hashing
    again on the board: a copy that did not land is an error, not a board
    quietly running something else.
    """
    before = _remote_manifest(client, remote_dir, local)
    stale = sorted(n for n, h in local.items() if before.get(n) != h)
    if not stale:
        return []
    _run(client, f"mkdir -p {remote_dir}")
    sftp = client.open_sftp()
    try:
        for name in stale:
            tmp = f"{remote_dir}/{name}.g1d-tmp"
            sftp.put(str(local_dir / name), tmp)
            sftp.posix_rename(tmp, f"{remote_dir}/{name}")
    finally:
        sftp.close()
    after = _remote_manifest(client, remote_dir, local)
    still = sorted(n for n, h in local.items() if after.get(n) != h)
    if still:
        raise BoardImagerError(
            f"copied {', '.join(still)} to {remote_dir} but the board's sha256 "
            f"still differs from this package's. Check the disk on the board.")
    return stale


def _publisher_pids(client) -> list[str]:
    return _run(client, f"pgrep -f {PROCESS_PATTERN!r} || true").split()


def _stamp_matches(client, digest: str) -> bool:
    """Was the RUNNING publisher started from code `digest`?"""
    stamp = _run(client, f"cat {STAMP} 2>/dev/null || true").split()
    return (len(stamp) == 2 and stamp[1] == digest
            and stamp[0] in _publisher_pids(client))


def _write_stamp(client, digest: str) -> None:
    pids = _publisher_pids(client)
    if pids:
        _run(client, f"echo {pids[0]} {digest} > {STAMP}")


def _kill_publisher(client, timeout: float) -> None:
    """Stop our publisher and wait until it has let go of the cameras."""
    for pid in _publisher_pids(client):
        _run(client, f"kill {pid} || true")
    deadline = time.monotonic() + timeout
    while _is_running(client) and time.monotonic() < deadline:
        time.sleep(0.5)
    if _is_running(client):
        raise BoardImagerError(
            f"the image server did not exit within {timeout:.0f}s of a kill; "
            f"it still holds the cameras.")


def _launch(client, remote_dir: str, conda_root: str, conda_env: str) -> None:
    """Detach it properly: `nohup ... & disown`, stdin closed.

    Without the redirect and the disown the process dies with the SSH channel,
    which looks exactly like a server that started and crashed.
    """
    # FIRED, NOT READ. `stdout.read()` waits for the channel to reach EOF, and a
    # backgrounded, disowned process need never produce one — so reading here
    # blocks on a command whose entire purpose is to return nothing. There is
    # also nothing to read: stdout and stderr both go to REMOTE_LOG.
    #
    # Whether it actually started is answered by the polling loop below, which
    # asks the only question that matters.
    script = (f"source {conda_root}/etc/profile.d/conda.sh && "
              f"conda activate {conda_env} && cd {remote_dir} && "
              f"nohup python publisher.py > {REMOTE_LOG} 2>&1 < /dev/null "
              "& disown")
    client.exec_command(f"bash -lc {script!r}")


def ensure_running(host: str, *, remote_dir: str | None, conda_root: str | None,
                   conda_env: str | None,
                   ssh_user: str = DEFAULT_SSH_USER,
                   ssh_password: str = DEFAULT_SSH_PASSWORD,
                   connect_timeout: float = 10.0,
                   start_timeout: float = 20.0,
                   vendor_stop_timeout: float = 30.0,
                   log: Callable[[str], None] | None = None) -> bool:
    """Make our image server — THIS package's copy of it — the one running on
    `host`.

    Returns True if it was ALREADY running exactly this code (nothing done),
    False if this call started or restarted it. Raises `BoardImagerError` on
    anything else — SSH failure, a copy that did not land, or a process that
    never appears.

    **The board's copy is synced, never trusted.** Every `SYNCED_SUFFIXES` file
    is hashed on both sides and any that differ are copied up (`_sync`); a
    running publisher is then kept only if its `STAMP` says it was started from
    exactly this code, and restarted otherwise. A fix to the server reaches the
    robot on the next launch, not whenever someone remembers to rsync.

    `log`, when given, hears about every copy and restart (P0 passes its own
    console); they are also logged at WARNING.

    `vendor_stop_timeout` bounds the wait for the vendor unit to let go of the
    cameras. It is a ceiling on `_stop_vendor`'s own polling, deliberately not
    on a blocking `systemctl stop` — see that function for why the difference
    is the whole bug.

    `remote_dir`, `conda_root` and `conda_env` are the robot file's
    `[image_server]` (`dir`, `conda`, `conda_env`); none has a default.

    Set `G1D_NO_BOARD_AUTOSTART=1` to skip entirely and use whatever is
    already up.
    """
    # `"0"` IS A NON-EMPTY STRING, and therefore truthy. A bare
    # `if os.environ.get(SKIP_ENV)` made `G1D_NO_BOARD_AUTOSTART=0` mean
    # "skip", which is the exact opposite of what anyone typing 0 intends — and
    # it then returned True, so the caller logged "already running" about a
    # board it had not looked at. Both halves of that were wrong.
    raw = os.environ.get(SKIP_ENV, "")
    if raw.strip().lower() not in ("", "0", "false", "no", "off"):
        logger.info("%s=%s — leaving the board's cameras alone; whatever is "
                    "serving now is what this run will read", SKIP_ENV, raw)
        return True

    if not (remote_dir and conda_root and conda_env):
        raise BoardImagerError(
            "the robot file's [image_server] must give dir, conda and conda_env "
            "for type = \"fisheye\": where the server is installed on the board")
    try:
        client = _connect(host, ssh_user, ssh_password, connect_timeout)
    except Exception as exc:            # noqa: BLE001 - one clear error type out
        raise BoardImagerError(
            f"could not SSH to {ssh_user}@{host}: {exc}. The wrists are on the "
            f"board, so there is no way to read them without it. Check the "
            f"address (192.168.123.164, not .124) and that a route exists — "
            f"with a VPN/TUN up, an unrouted private subnet can accept the "
            f"handshake and then close, which looks like a dead robot.") from exc

    try:
        # THE VENDOR GOES FIRST, AND UNCONDITIONALLY — also when this server is
        # already up, or the two run side by side. That is not tidiness:
        # teleimager crash-loops here
        # (`Restart=always`, `RestartSec=5`) and runs `modprobe -r uvcvideo` on
        # every attempt to come up, which pulls the devices out from under a
        # publisher that has been serving happily for minutes (observed: this
        # server serving, teleimager `activating` beside it).
        if _vendor_active(client):
            logger.warning(
                "stopping %s on %s — it holds the cameras through usbfs and "
                "reloads the uvcvideo module on restart. This also takes the "
                "HEAD camera (port 55555) down, so v1 perception tooling will "
                "not work until it is started again.", VENDOR_UNIT, host)
            _stop_vendor(client, ssh_password, vendor_stop_timeout)

        def say(msg: str) -> None:
            logger.warning(msg)
            if log is not None:
                log(msg)

        local = local_manifest()
        digest = manifest_digest(local)
        copied = _sync(client, remote_dir, local)
        if copied:
            say(f"board image server: copied {', '.join(copied)} to "
                f"{host}:{remote_dir} (differed from this package; code "
                f"{digest})")

        if _is_running(client):
            if not copied and _stamp_matches(client, digest):
                logger.info("fish_eye_image_server already running on %s "
                            "(code %s)", host, digest)
                return True
            say(f"board image server: restarting it on {host} — the running "
                f"publisher was not started from this package's code "
                f"({digest}) and would keep serving whatever it loaded then")
            _kill_publisher(client, start_timeout)

        logger.info("starting fish_eye_image_server on %s (conda env %r)",
                    host, conda_env)
        _launch(client, remote_dir, conda_root, conda_env)

        deadline = time.monotonic() + start_timeout
        while time.monotonic() < deadline:
            time.sleep(1.0)
            if _is_running(client):
                _write_stamp(client, digest)
                logger.info("fish_eye_image_server started on %s", host)
                return False
        tail = _run(client, f"tail -n 20 {REMOTE_LOG} 2>/dev/null || true")
        raise BoardImagerError(
            f"started publisher.py on {host} but it never appeared in the "
            f"process list within {start_timeout:.0f}s.\n\n"
            f"--- {REMOTE_LOG} ---\n{tail or '(empty)'}\n\n"
            f"Most likely: a camera's physical_path in cameras.yaml does not "
            f"match what is plugged in (run `python publisher.py "
            f"--list-devices`), the unit does not offer the configured MJPG "
            f"mode, or a control was declined.")
    finally:
        client.close()


def stop(host: str, *, ssh_user: str = DEFAULT_SSH_USER,
         ssh_password: str = DEFAULT_SSH_PASSWORD,
         restore_vendor: bool = True, timeout: float = 90.0) -> None:
    """Stop our server, and by default hand the cameras back to the vendor's.

    The counterpart to `ensure_running`, so that a rollout on the fisheye rig
    does not silently leave the head camera dark for whoever next runs teleop.

    **The vendor unit is started, never restarted, and only once ours is
    confirmed dead.** Both halves are learned from doing it wrong on the live
    board: `teleimager` is `Restart=always`, so while it cannot get the cameras
    it sits in an auto-restart loop, and a `systemctl restart` issued into that
    loop wedged it in `deactivating` — active by name, publishing nothing, for
    minutes. Killing our publisher first lets its own next restart simply
    succeed.
    """
    client = _connect(host, ssh_user, ssh_password, 10.0)
    try:
        for pid in _publisher_pids(client):
            _run(client, f"kill {pid} || true")

        deadline = time.monotonic() + timeout
        while _is_running(client) and time.monotonic() < deadline:
            time.sleep(1.0)
        if _is_running(client):
            raise BoardImagerError(
                f"could not stop the image server on {host}; it still holds the "
                f"cameras, so {VENDOR_UNIT} cannot have them back.")

        if restore_vendor:
            _sudo(client, f"systemctl start {VENDOR_UNIT} --no-block",
                  ssh_password)
            while (time.monotonic() < deadline
                   and not _vendor_active(client)):
                time.sleep(2.0)
            logger.info("%s on %s: %s", VENDOR_UNIT, host,
                        "active" if _vendor_active(client) else "NOT active")
    finally:
        client.close()
