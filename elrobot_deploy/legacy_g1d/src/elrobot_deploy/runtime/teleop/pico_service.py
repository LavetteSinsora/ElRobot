"""The XRoboToolkit PC Service — what the PICO headset connects TO.

Started by P0 at CHECK when `--intervene` is given
(`runtime/launcher/startup_checks.py`). The local counterpart of
`runtime/drivers/board/image_server_launch.py`: a vendor service this package
does not own, whose absence is invisible at exactly the point it matters.

The failure this exists to prevent: with the service dead, `xrobotoolkit_sdk`
still imports, `init()` still SUCCEEDS, and every getter returns zeros forever.
Nothing raises. P11 polls happily and `samples` climbs at the full poll rate,
so the one symptom is that no pose ever arrives — which reads as a broken
build, or as a headset nobody put on, rather than as a service nobody started.
`scripts/build_xrobotoolkit.sh` builds the binding and cannot act on this,
because starting a system service is not something a build script should do.

Two ports, and they are not interchangeable:

* **63901** is where the HEADSET connects, over WiFi, to this box's LAN
  address. The headset dials in; nothing here ever dials out to it, which is
  why the headset may be connected long after this box is up.
* **60061** is loopback only, between `xrobotoolkit_sdk` and this service.

Neither is opened by this package — both live inside the vendor's `.so` — so
there is nothing here to bind and nothing to configure. This module answers
one question: is that process running, and if not, start it.
"""

from __future__ import annotations

import os
import subprocess
import time

#: The vendor's launcher, as its own installer lays it down. Overridable
#: because it is an absolute path OUTSIDE this package, on a box it does
#: not provision: `scripts/build_xrobotoolkit.sh` builds the *binding*, while the
#: service arrives from the vendor's installer and can land anywhere.
SERVICE_ENV = "ELROBOT_PICO_SERVICE"
DEFAULT_SERVICE = "/opt/apps/roboticsservice/runService.sh"

#: What to look for in the process table. NOT the launcher script: it exports
#: three Qt paths, backgrounds this binary with `&` and exits immediately, so
#: `runService.sh` is never what is still running a second later.
PROCESS_PATTERN = "RoboticsServiceProcess"

#: Where the service's own stdout goes. It is a Qt application and chatty on
#: startup; named in the failure message so the operator has somewhere to look.
LOG_PATH = "/tmp/g1d_pico_service.log"

#: How long the service gets to appear in the process table after being
#: started. A bounded wait, unlike CHECK's: this is not waiting on a human or
#: on a network, it is waiting on `fork` and a Qt constructor on this machine.
#: If it has not appeared in ten seconds it is not coming.
_START_TIMEOUT_S = 10.0
_POLL_S = 0.1


class PicoServiceError(RuntimeError):
    """The PC Service is not running and could not be started."""


def service_path() -> str:
    """The launcher this box should use."""
    return os.environ.get(SERVICE_ENV, "").strip() or DEFAULT_SERVICE


def is_running() -> bool:
    """Is the vendor service in the process table?

    `pgrep -f` against an argv list with NO shell in it, deliberately. `-f`
    matches whole command lines, so running it through `sh -c` gives the shell
    a command line that contains the pattern and makes it a match for itself —
    a trap `runtime/drivers/board/image_server_launch.py` documents.
    An argv list has no shell to match.

    `-f` rather than `-x` because `RoboticsServiceProcess` is 22 characters and
    a process's `comm` is truncated at 15, so an exact-name match can never
    succeed. The full command line is `./RoboticsServiceProcess`.
    """
    try:
        found = subprocess.run(["pgrep", "-f", PROCESS_PATTERN],
                               capture_output=True, text=True, timeout=5.0,
                               check=False)
    except (OSError, subprocess.SubprocessError):
        # No pgrep, or it hung. Report "not running" and let the caller try to
        # start one: a second service is refused by the vendor's own port bind,
        # which is a better outcome than refusing a run over a missing tool.
        return False
    return found.returncode == 0 and bool(found.stdout.strip())


def ensure_running(*, timeout_s: float = _START_TIMEOUT_S) -> bool:
    """Start the PC Service if it is not up. True if it already was.

    Idempotent, so it is safe on every run: a box where the operator already
    started the service by hand, or left one from an earlier session, is left
    entirely alone — the service holds 63901, and a second one would only
    fight the first for it.

    Detached with `start_new_session`, so the service outlives the deploy run
    that started it. That is deliberate and matches how an operator uses it:
    the headset connects to the SERVICE, not to us, and killing it between
    rollouts would drop the headset's link every time.

    Raises `PicoServiceError` if the launcher is missing or nothing appeared.
    """
    if is_running():
        return True

    path = service_path()
    if not os.path.isfile(path):
        raise PicoServiceError(
            f"no XRoboToolkit PC Service launcher at {path}. It is a vendor "
            f"package with its own installer, not part of elrobot_deploy — install "
            f"it, or point {SERVICE_ENV} at its runService.sh.")

    try:
        with open(LOG_PATH, "ab", buffering=0) as log:
            subprocess.Popen(["bash", path], stdout=log,
                             stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL,
                             start_new_session=True)
    except OSError as exc:
        raise PicoServiceError(f"could not start {path} with its output to "
                               f"{LOG_PATH}: {exc}") from exc

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if is_running():
            return False
        time.sleep(_POLL_S)
    raise PicoServiceError(
        f"started {path} but no {PROCESS_PATTERN} appeared within "
        f"{timeout_s:.0f}s. Its output is in {LOG_PATH}.")
