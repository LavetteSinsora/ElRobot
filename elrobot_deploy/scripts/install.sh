#!/usr/bin/env bash
# scripts/install.sh — set up elrobot_deploy on a new box, from a fresh clone.
#
#   scripts/install.sh [--realsense] [--intervene] [--no-dev]
#
#   --realsense   also install pyrealsense2 (a RealSense camera on this box:
#                 the g1d_fisheye_dex1_exo profile)
#   --intervene   also install pybind11 and build the PICO bindings
#                 (scripts/build_xrobotoolkit.sh), for `elrobot_deploy run --intervene`
#   --no-dev      leave out pytest/ruff/scipy
#
# What it does, each step skipped when already done (safe to re-run):
#
#   1. Builds the CycloneDDS C library into .cyclonedds/install, INSIDE the
#      checkout. The Python `cyclonedds` binding ships no wheels, so every box
#      compiles it, and compiling it needs the C library and CYCLONEDDS_HOME.
#      The C library's release line is read from uv.lock's `cyclonedds`
#      version, so the two cannot drift apart.
#   2. `uv sync --locked --extra runtime` into .venv, with CYCLONEDDS_HOME exported.
#   3. With --intervene, builds xrobotoolkit_sdk into that venv.
#
# Delete .cyclonedds/ to force a from-scratch CycloneDDS build.
#
# THE RPATH TRAP. The compiled `cyclonedds` extension records the ABSOLUTE path
# of .cyclonedds/install. Move or copy the checkout afterwards and it fails to
# load. Clone where it will run, then install there; never copy .venv or
# .cyclonedds between machines or directories.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CYC="$ROOT/.cyclonedds"
SRC="$CYC/src"
BUILD="$CYC/build"
INSTALL="$CYC/install"
CYC_URL="https://github.com/eclipse-cyclonedds/cyclonedds"

extras=()
dev=1
intervene=0
for arg in "$@"; do
    case "$arg" in
        --realsense) extras+=(--extra realsense) ;;
        --intervene) extras+=(--extra intervene); intervene=1 ;;
        --no-dev)    dev=0 ;;
        -h|--help)   sed -n '2,27p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "ERROR: unknown option $arg (see --help)" >&2; exit 2 ;;
    esac
done

die() { echo "ERROR: $*" >&2; exit 1; }

need() {   # need <tool> <what to install>
    command -v "$1" >/dev/null 2>&1 || die "$1 not found. Install it first:
  $2"
}

need uv  "curl -LsSf https://astral.sh/uv/install.sh | sh   # then open a new shell"
need git "sudo apt-get install -y git"
[ -f "$ROOT/uv.lock" ] || die "$ROOT/uv.lock is missing — this checkout is incomplete"

# --- 1. CycloneDDS, the C library -------------------------------------------

version="$(awk '/^name = "cyclonedds"$/ {getline; gsub(/[^0-9.]/, ""); print; exit}' "$ROOT/uv.lock")"
[ -n "$version" ] || die "uv.lock has no cyclonedds entry; cannot tell which C library to build"
branch="releases/${version%.*}.x"      # 0.10.2 -> releases/0.10.x

if [ -f "$INSTALL/lib/cmake/CycloneDDS/CycloneDDSConfig.cmake" ]; then
    echo "==> CycloneDDS already built at $INSTALL"
else
    echo "==> building CycloneDDS ($branch, for the cyclonedds==$version binding) into $INSTALL"
    need cmake "sudo apt-get install -y build-essential cmake"
    command -v cc >/dev/null 2>&1 || command -v gcc >/dev/null 2>&1 \
        || die "no C compiler found. Install one first:
  sudo apt-get install -y build-essential"
    if [ ! -d "$SRC/.git" ]; then
        rm -rf "$SRC"
        git clone --depth 1 -b "$branch" "$CYC_URL" "$SRC"
    else
        echo "    source already at $SRC"
    fi
    cmake -B "$BUILD" -S "$SRC" -DCMAKE_INSTALL_PREFIX="$INSTALL" \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF
    cmake --build "$BUILD" --target install --parallel
    [ -f "$INSTALL/lib/cmake/CycloneDDS/CycloneDDSConfig.cmake" ] \
        || die "the CycloneDDS build finished but $INSTALL has no CycloneDDSConfig.cmake"
fi
export CYCLONEDDS_HOME="$INSTALL"

# --- 2. the Python environment ------------------------------------------------

sync=(uv sync --locked --extra runtime "${extras[@]}")
[ "$dev" -eq 1 ] || sync+=(--no-dev)
echo "==> CYCLONEDDS_HOME=$CYCLONEDDS_HOME ${sync[*]}"
( cd "$ROOT" && "${sync[@]}" )

venv="${UV_PROJECT_ENVIRONMENT:-$ROOT/.venv}"
"$venv/bin/python" -c "import cyclonedds.domain, elrobot_deploy" \
    || die "the environment synced but cyclonedds / elrobot_deploy do not import"

# --- 3. the PICO bindings -----------------------------------------------------

if [ "$intervene" -eq 1 ]; then
    echo "==> building the PICO bindings"
    PYTHON="$venv/bin/python" bash "$ROOT/scripts/build_xrobotoolkit.sh"
fi

cat <<EOF

Installed. The \`elrobot_deploy\` command lives in this project's venv, so it is
NOT on your PATH until you activate that venv (in every new shell):

  source $venv/bin/activate

or run it without activating: \`uv run elrobot_deploy ...\` from $ROOT.

Next:
  elrobot_deploy check --no-robot                     # this box
  elrobot_deploy check --robot g1d_fisheye_dex1       # and the robot, once it is cabled

A later \`uv sync\` that has to rebuild cyclonedds needs the same variable:
  CYCLONEDDS_HOME=$INSTALL uv sync --extra runtime
EOF
