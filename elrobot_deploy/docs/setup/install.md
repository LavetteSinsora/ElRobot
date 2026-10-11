# Installing and first checks

What to do once on a new computer before any policy can run.

- **Read first:** `docs/overview.md`.
- **Code:** `scripts/install.sh`, `src/g1d_deploy/cli/check.py`.
- **Related:** `docs/setup/machine_toml.md`, `docs/setup/robot_toml.md`.

## Install

Needs Linux with git, cmake, a C compiler and
[uv](https://docs.astral.sh/uv/). Clone the repository where it will run (the
install records absolute paths; do not move or copy the checkout afterwards),
then from its root:

    scripts/install.sh                 # builds CycloneDDS, creates .venv
    source .venv/bin/activate          # puts g1d_deploy on PATH; every new shell

Options: `--realsense` (a RealSense camera plugged into this computer),
`--intervene` (the PICO headset takeover), `--no-dev` (leave out the test
tools). The script is safe to re-run; `scripts/install.sh --help` explains
each step.

## The network

The G1-D's onboard computers sit on `192.168.123.x`. Cable one of this
computer's network cards to the robot and give that card a **permanent static
address on that subnet** (any free one). With NetworkManager:

    nmcli -t -f NAME,DEVICE con show --active        # the connection on the cabled card
    sudo nmcli con mod "<connection>" ipv4.method auto ipv4.addresses 192.168.123.<free number>/24
    sudo nmcli con up "<connection>"

`g1d_deploy check` reports a missing address. The card is then found
automatically.

## On the robot, for a fisheye rig

g1d_deploy copies its own camera server to the robot's onboard computer and
starts it there over SSH. It runs in a conda environment that must already
exist on that computer, with `pyuvc`, `pyzmq` and `pyyaml`
(`docs/setup/robot_toml.md`, `[image_server]`).

## The machine file (optional)

Nothing needs writing: the network card and the CPU table are detected, and
the rest has defaults. To change a default (most often where recordings go):

    cp machine.example.toml machine.toml     # keep only what you change

Every field: `docs/setup/machine_toml.md`. The robot's own address and login
are the robot file's (`docs/setup/robot_toml.md`, `[board]`), factory values
by default.

## Check the computer and the robot

    g1d_deploy check              # this computer, then the robot; read-only
    g1d_deploy check --no-robot   # this computer only

It reports packages, the GPU, shared memory, the network
card and the CPU table, then whether every camera of the robot file and the
state streams answer. It never publishes a command.

## Try the shipped examples

    g1d_deploy check-policy examples/policies/sine_eef.toml   # offline
    g1d_deploy run examples/deploy.toml                    # MOVES THE ARM: homes it and holds

`docs/deploying_a_policy/running_and_live_dashboard.md` describes what a
launch does.

## Connect a policy's Python environment

A policy adapter runs under its own Python (the one with its model's
libraries), which only needs g1d_deploy's numpy-only core:

    /path/to/policy/.venv/bin/python -m pip install -e /path/to/this/checkout

or a wrapper script that puts this checkout's source directory on
`PYTHONPATH`. Details: `docs/deploying_a_policy/policy_contract.md`, "The
adapter's own Python".
