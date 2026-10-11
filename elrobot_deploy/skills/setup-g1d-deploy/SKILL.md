---
name: setup-g1d-deploy
description: Guide for setting up the elrobot_deploy infra on a computer and connecting it to a Unitree G1-D for the first time. Use it whenever the user wants to install elrobot_deploy, get started with it, prepare a new deploy computer or a new robot, choose or write the robot file (robot.toml), decide whether a machine file (machine.toml) is needed, or when `elrobot_deploy check` reports a problem with the computer, the network or the robot's cameras. After setup, hand over to the deploy-a-policy skill.
---

# Set up elrobot_deploy

This skill takes a user from a fresh clone of elrobot_deploy to a computer that
can talk to their Unitree G1-D: installed, on the robot's network, with the
right robot file, and passing `elrobot_deploy check`. Deploying a policy on top of
that is the `deploy-a-policy` skill (`skills/deploy-a-policy/SKILL.md`).

Setup is done when `elrobot_deploy check` passes for the user's robot. Nothing in
this skill moves the robot. Do not run `elrobot_deploy run`; where a launch is
needed to look at something, the user runs it.

Contents: [what elrobot_deploy is](#what-elrobot_deploy-is) ·
[how to work with the user](#how-to-work-with-the-user) ·
[1 install](#1-install) · [2 network](#2-put-the-computer-on-the-robots-network) ·
[3 robot file](#3-the-robot-file) · [4 machine file](#4-the-machine-file-usually-not-needed) ·
[5 check](#5-check) · [6 first look](#6-a-first-look-on-the-dashboard) ·
[next](#next-deploying-a-policy)

## What elrobot_deploy is

elrobot_deploy runs a policy on a Unitree G1-D. It is a set of processes on a
**deploy computer** (a Linux PC, usually with a GPU for the policy) that is
cabled to the robot. It reads the robot's cameras and joint state, runs the
policy, and sends motor commands. Three files configure it:

| file | describes | who writes it |
|---|---|---|
| a **robot file** (`robot.toml`) | the robot: its cameras, its gripper, the address and login of its onboard computer | usually nobody: a shipped profile fits a standard robot |
| a **machine file** (`machine.toml`) | the deploy computer's overrides of the defaults | optional |
| a **deploy file** (`deploy.toml`) with a policy adapter | one deployment of one policy | the `deploy-a-policy` skill |

This skill covers the first two and the installation. Paths such as
`docs/setup/install.md` are relative to the root of the elrobot_deploy checkout
(the directory holding `README.md`, `docs/`, `scripts/` and `src/`), which is
two directories above this file. `docs/overview.md` maps all the documentation.

## How to work with the user

- **Look before asking.** Most facts can be read from the machine: which tools
  are installed, the CPU, the network cards, whether the robot answers. Ask
  the user only for what the machine cannot tell you, such as which cameras
  are mounted on the robot.
- **Say why, whenever you ask for something.** Several steps need information
  or permission that sounds intrusive without its reason (a login for the
  robot, a `sudo` command, a static network address). Each step below states
  the reason; pass it on in plain words, so the user can judge the request.
- **Do not run `sudo` yourself.** Show the exact command and what it changes,
  and let the user run it.
- **Never commit `machine.toml`**, and do not create files in the user's home
  directory.

## 1. Install

Read `docs/setup/install.md` first. In short:

1. **Check the prerequisites**: `uv`, `git`, `cmake` and a C compiler
   (`command -v uv git cmake cc`). The install script checks them too and
   prints the install command for whichever is missing; those commands need
   `sudo` or change the user's shell setup, so the user runs them.
2. **Check the checkout is where it will stay.** The build records absolute
   paths: a checkout moved or copied after installing has to be reinstalled.
3. **Run `scripts/install.sh`** from the checkout root. It builds the
   CycloneDDS library (what elrobot_deploy uses to talk to the robot; it has to be
   compiled on every computer) and creates `.venv` with everything else. It
   takes a few minutes and is safe to re-run. Options: `--realsense` if a
   RealSense camera is plugged into this computer, `--intervene` for the PICO
   headset takeover.
4. **Activate the environment**: `source .venv/bin/activate`, in every new
   terminal. The script cannot do this for the user's shell. In your own
   commands, call `.venv/bin/elrobot_deploy` directly.

Then look at the computer itself:

```bash
.venv/bin/elrobot_deploy check --no-robot
```

It reports the installed packages, the GPU, shared memory, the network card
and the CPU table. Work through whatever it marks as failed; each line prints
its own remedy. Two results need explaining to the user:

- **The CPU table is refused.** elrobot_deploy pins each process to its own cores
  so that the 500 Hz motor loop is never delayed by camera decoding or the
  policy. It derives that assignment automatically, and refuses a CPU it
  cannot lay out (roughly: fewer than 10 physical cores, or no
  hyperthreading). The computer can still be used with an assignment written
  by hand. Work that out yourself:
  `skills/setup-g1d-deploy/references/cpu_pinning.md` explains what each
  process needs, the rules a table must satisfy, a procedure, and the
  trade-offs when cores are short. Do not question the user about cores;
  tell them the outcome, and ask only if a compromise would remove a camera
  or recording.
- **`/dev/shm` is too small.** The processes share camera frames through
  shared memory, which lives in `/dev/shm`. The remedy is a `sudo mount`
  command the check prints; the user runs it.

## 2. Put the computer on the robot's network

The G1-D's onboard computers have fixed factory addresses on the
`192.168.123.x` network, and the robot speaks DDS on that network only. The
deploy computer therefore needs:

- a network cable from one of its network cards to the robot's switch;
- **a static address on `192.168.123.x`** on that card, permanently. Any
  address not used by the robot or another computer works. Without it nothing
  reaches the robot, and an address added by hand is lost when the cable is
  re-plugged.

See what is there with `ip -4 -o addr`. If no card holds a `192.168.123.x`
address, `elrobot_deploy check` reports it. On a computer using NetworkManager,
this adds one beside the card's existing configuration:

```bash
nmcli -t -f NAME,DEVICE con show --active        # the connection on the cabled card
sudo nmcli con mod "<connection>" ipv4.method auto ipv4.addresses 192.168.123.<free number>/24
sudo nmcli con up "<connection>"
```

It needs `sudo` and changes the network settings, so explain it and let the
user run it. elrobot_deploy then finds the card by itself; there is nothing to
configure.

## 3. The robot file

A robot file says what is mounted on this particular robot. The deploy file
later names it (`robot = "g1d_fisheye_dex1"`). Read `docs/setup/robot_toml.md`.

**Pick a shipped profile.** Ask the user which cameras and which gripper their
robot has, then match:

| profile | cameras | gripper |
|---|---|---|
| `g1d_fisheye_dex1` | two fisheye wrist cameras and the head camera | Dex-1 |
| `g1d_fisheye_dex1_exo` | the same, plus a RealSense plugged into the deploy computer | Dex-1 |
| `g1d_normal_dex1` | the head camera only, served by Unitree's own image service | Dex-1 |

If one fits, nothing needs writing. Only the Dex-1 gripper is supported for
policies today.

**Write a robot file only when the robot differs.** It extends a profile and
overrides what is different, so it is usually a few lines
(`examples/robot_custom.toml` is a working one). Typical reasons:

- **The onboard computer's address or login was changed** from the factory
  values. elrobot_deploy reads the cameras from that computer, and for the fisheye
  rig it logs into it over SSH to start its camera server, because the
  robot's stock service cannot serve those cameras. That is the only thing
  the login is used for. The factory values (`192.168.123.164`, user
  `unitree`) are the default; ask for a login only if the default is refused,
  and say what it is for. A changed password is better given through the
  `ELROBOT_BOARD_SSH_PASSWORD` environment variable than written into a file,
  since robot files are copied into every recording.
- **A camera is different or additional**: another port, resolution or a
  camera plugged into the deploy computer.
- **The two wrist cameras are crossed** ([step 6](#6-a-first-look-on-the-dashboard)).

**The fisheye rig needs one thing on the robot's onboard computer.** The
camera server elrobot_deploy starts there runs in a conda environment that must
already exist on that computer, with `pyuvc`, `pyzmq` and `pyyaml` (the
profile names it: `[image_server] conda`, `conda_env`). elrobot_deploy copies and
starts the server itself, but does not create the environment. The server's
own notes are in
`src/elrobot_deploy/runtime/drivers/board/fisheye_image_server/README.md`,
including the per-robot values in its `cameras.yaml`: which USB socket each
wrist camera is plugged into differs between robots, and a robot other than
the reference one needs its own values there.

## 4. The machine file (usually not needed)

`machine.toml` is optional. Without it, the network card and the CPU table
are detected and everything else has a default. Read
`docs/setup/machine_toml.md`. To create one, copy `machine.example.toml` to
`machine.toml` at the checkout root and keep only the lines that change.

**Ask the user one question here: where recordings should be stored.** They
are unlikely to know what recordings are yet, so explain first:

- elrobot_deploy records every rollout automatically: all camera frames, the
  robot's state, the actions the policy produced and the commands sent.
  Recordings are ordinary folders, one per rollout; ones that are no longer
  needed can be deleted at any time to free space.
- Recordings are what make a rollout reviewable afterwards.
  `elrobot_deploy inspect` opens a page where a rollout is replayed on a
  timeline with the cameras and a 3-D robot, and where it can be compared
  against alternatives computed offline on the same recording: what a
  different checkpoint would have done given the same observations, or how a
  different IK solver would have solved the same targets
  (`docs/deploying_a_policy/rollout_inspection.md`).
- By default each rollout is stored beside the deploy file that produced it,
  in a `rollout_record/` folder. Setting `[paths] records` in `machine.toml`
  puts all of them under one directory instead, which keeps them off a small
  system disk and lets `elrobot_deploy inspect` open them without a path.

Then ask whether they want one such directory and where (look at `df -h`
first, and suggest a disk with space). If they have no preference, leave the
default and write no machine file for this.

The other reasons to write a machine file need no question:

- **the dashboard port is taken** (`[dashboard] port`, default 8080);
- **the CPU table was refused in step 1**: write `[cpu]` and `[cpu.pin]`
  following `skills/setup-g1d-deploy/references/cpu_pinning.md`.

## 5. Check

With the robot powered on and cabled:

```bash
.venv/bin/elrobot_deploy check --robot <profile name or path to the robot file>
```

It is read-only: it listens to every camera the robot file lists and to the
robot's state streams, and never sends a command. Each failing line prints
what to do. Common cases:

- **Right after power-on, streams are silent.** The robot's services take
  about a minute to start. Wait and run it again before investigating
  (`docs/deploying_a_policy/field_notes.md`).
- **No interface on the robot's subnet**: step 2.
- **A camera does not answer**: the wrong profile for this robot, the camera
  server on the robot is not running (the fisheye one is started at the first
  launch, not by `check`), or the SSH login was refused.

Setup is complete when this passes.

## 6. A first look on the dashboard

One thing no check can verify: identical cameras cannot be told apart in
software, so whether the camera named `lwrist` really is the left wrist
depends on how this robot is cabled. A policy fed crossed wrists moves
wrongly.

Ask the user to launch the shipped hold policy, which moves the arms to the
home posture and holds them there, and to look at the dashboard:

```bash
elrobot_deploy run examples/deploy.toml      # the USER runs this: the arms move to home
```

Explain before they run it that the arms move during startup, and that the
area around the robot must be clear. On `localhost:8080`, the pane labelled
`lwrist` must show the left arm's view and `rwrist` the right. If they are
crossed, write a robot file that extends the profile and exchanges the two
cameras' ports (`docs/setup/robot_toml.md`, "Check the wrists by eye"), and
use that file from then on. Exit from the dashboard ends the session.

## Next: deploying a policy

Report to the user what was installed, which robot file to use (a profile
name or the path of the file you wrote), whether a machine file exists and
why, and the result of `elrobot_deploy check`. Then continue with the
`deploy-a-policy` skill (`skills/deploy-a-policy/SKILL.md`) to write the
policy adapter and the deploy file.
