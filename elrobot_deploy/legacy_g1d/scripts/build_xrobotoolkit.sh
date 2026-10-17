#!/usr/bin/env bash
# scripts/build_xrobotoolkit.sh — one-time build of the XRoboToolkit PICO
# bindings, INSIDE this checkout (.xrobotoolkit/), for a deploy box
# that will run `elrobot_deploy run ... --intervene`.
#
# Usually run for you by `scripts/install.sh --intervene`. By hand (from
# anywhere — paths are computed from this script's own location):
#   uv sync --extra intervene && bash scripts/build_xrobotoolkit.sh
# PYTHON defaults to the project's .venv/bin/python if it exists, else python3.
# That interpreter needs pybind11 (the `intervene` extra).
#
# Safe to re-run — exits early if the wrapper already imports. Delete
# .xrobotoolkit/ to force a from-scratch rebuild.
#
# WHY THIS IS A SCRIPT AND NOT A LINE IN pyproject.toml
#
# `xrobotoolkit_sdk` is a pybind11 CMake extension that links against
# PXREARobotSDK — a native library that lives in a DIFFERENT repository
# (XRoboToolkit-PC-Service) and must be built and staged into the wrapper's own
# `lib/` and `include/` before its `setup.py` will configure at all. So there is
# no git or path dependency that `uv sync` could resolve on a clean box: uv
# builds in an isolated checkout, and that checkout would have neither
# directory. This is the same shape of problem as CycloneDDS, and it gets the
# same shape of answer: a script that does the native half.
#
# Two things this script CANNOT do, and they are not Python packaging problems:
#
#   1. The XRoboToolkit PC Service must be RUNNING on this box for the SDK to
#      return anything. Getters on a dead service succeed and return nothing,
#      which reads as a broken build rather than as a service nobody started —
#      P0's CHECK refuses the run for exactly this (`_verify_pico`).
#   2. The XRoboToolkit app must be installed on the PICO, opened, pointed at
#      THIS box's IP, with the head and controller pose boxes ticked. The pose
#      stream is over WiFi — the headset and this box must be on the same
#      network — and no USB or ADB connection is involved at run time.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$ROOT/.xrobotoolkit"
SERVICE_SRC="$WORK/XRoboToolkit-PC-Service"
PYBIND_SRC="$WORK/XRoboToolkit-PC-Service-Pybind"

SERVICE_URL="https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git"
PYBIND_URL="https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind.git"

PY="${PYTHON:-python3}"
if [ -z "${PYTHON:-}" ] && [ -x "$ROOT/.venv/bin/python" ]; then
    PY="$ROOT/.venv/bin/python"
fi

if "$PY" -c "import xrobotoolkit_sdk" 2>/dev/null; then
    echo "xrobotoolkit_sdk already imports in $PY — nothing to do"
    echo "(delete $WORK and re-run to force a rebuild)"
    exit 0
fi

for tool in cmake git; do
    command -v "$tool" >/dev/null 2>&1 || {
        echo "ERROR: $tool not found. Install a C++ toolchain first, e.g.:" >&2
        echo "  sudo apt-get install -y build-essential cmake git" >&2
        exit 1
    }
done
command -v c++ >/dev/null 2>&1 || command -v g++ >/dev/null 2>&1 || {
    echo "ERROR: no C++ compiler found." >&2
    echo "  sudo apt-get install -y build-essential" >&2
    exit 1
}

mkdir -p "$WORK"

# --- 1. the native SDK, from its own repository ----------------------------
if [ ! -d "$SERVICE_SRC" ]; then
    echo "cloning XRoboToolkit-PC-Service ..."
    git clone --depth 1 "$SERVICE_URL" "$SERVICE_SRC"
fi

SDK_DIR="$SERVICE_SRC/RoboticsService/PXREARobotSDK"
if [ ! -d "$SDK_DIR" ]; then
    echo "ERROR: $SDK_DIR is missing — upstream has moved the SDK." >&2
    echo "Check the layout of $SERVICE_URL and update this script." >&2
    exit 1
fi
echo "building PXREARobotSDK ..."
( cd "$SDK_DIR" && bash build.sh )

# --- 2. the pybind wrapper, with the native artifacts staged into it --------
if [ ! -d "$PYBIND_SRC" ]; then
    echo "cloning XRoboToolkit-PC-Service-Pybind ..."
    git clone --depth 1 "$PYBIND_URL" "$PYBIND_SRC"
fi

mkdir -p "$PYBIND_SRC/lib" "$PYBIND_SRC/include"
# `setup.py` runs CMake against the wrapper's OWN tree, and CMakeLists.txt looks
# for the SDK in ./include and ./lib. Staging rather than pointing at the build
# tree, because the wrapper is what gets installed and the build tree is not.
found_lib=0
while IFS= read -r so; do
    cp -f "$so" "$PYBIND_SRC/lib/"
    found_lib=1
done < <(find "$SDK_DIR" -name 'libPXREARobotSDK*' -type f 2>/dev/null)
[ "$found_lib" -eq 1 ] || {
    echo "ERROR: PXREARobotSDK built but no libPXREARobotSDK* was produced." >&2
    echo "Look in $SDK_DIR for what build.sh actually emitted." >&2
    exit 1
}
while IFS= read -r hdr; do
    cp -f "$hdr" "$PYBIND_SRC/include/"
done < <(find "$SDK_DIR" -name '*.h' -type f 2>/dev/null)

# --- 3. the Python extension, into the synced venv -------------------------
echo "building xrobotoolkit_sdk into $PY ..."
"$PY" -c "import pybind11" 2>/dev/null || {
    echo "ERROR: pybind11 is not in $PY." >&2
    echo "  uv sync --extra intervene   # then re-run this" >&2
    exit 1
}
# A uv venv has no pip; install with uv into that interpreter when it is there.
if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$PY" --no-build-isolation "$PYBIND_SRC"
else
    "$PY" -m pip install --no-build-isolation "$PYBIND_SRC"
fi

"$PY" - <<'EOF'
import xrobotoolkit_sdk as x

need = ("get_time_stamp_ns", "get_left_controller_pose",
        "get_right_controller_pose", "get_left_trigger", "get_right_trigger",
        "get_left_axis", "get_right_axis")
missing = [n for n in need if not callable(getattr(x, n, None))]
if missing:
    raise SystemExit(
        "built, but the module is missing: " + ", ".join(missing)
        + "\nThis build is too old for elrobot_deploy's P11.")
print("xrobotoolkit_sdk OK — every getter elrobot_deploy needs is present")
EOF

cat <<'EOF'

Built. Two things this script cannot do for you:

  1. Start the XRoboToolkit PC Service on this box. Without it every getter
     succeeds and returns nothing, and P0's CHECK will refuse the run.
  2. Open the XRoboToolkit app on the PICO, point it at this box's IP, and tick
     the head and controller pose boxes. That stream is over WiFi: put the
     headset and this box on the same network.

Then check the headset with no robot powered:

    python -m elrobot_deploy.runtime.processes.p11_pico
EOF
