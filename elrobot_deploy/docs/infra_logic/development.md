# Developing g1d_deploy

For someone changing the infra itself: the project layout, the tests, and
where things are.

- **Read first:** `docs/overview.md` (the processes and the source layout).
- **Code:** `pyproject.toml`, `scripts/install.sh`, `tests/`.

## The project

- A uv project; `uv.lock` is committed. The runtime is the `runtime` extra
  (`uv sync --extra runtime`, which `scripts/install.sh` runs); the bare
  package is the numpy-only core a policy's own Python installs; `replay` adds
  what a policy replay needs. `uv sync` also installs the `dev` group (pytest,
  ruff, scipy). The IK stack is held at tested versions by
  `[tool.uv] constraint-dependencies` (`pyproject.toml` says why).
- The model assets are inside the package:
  `src/g1d_deploy/robot/assets/unitree_g1d` (the G1-D URDF and meshes) and
  `src/g1d_deploy/robot/assets/unitree_dex1` (the Dex-1 meshes). Each has a
  `PROVENANCE.md`.
- `.gitignore` keeps out `machine.toml`, the venv, `.cyclonedds/`,
  `.xrobotoolkit/`, rollout records and checkpoints.

## Tests

    uv run pytest tests -q        # from the checkout root; about 90 s

- They read `tests/machine.toml`, never the checkout's own `machine.toml`.
- They never talk to a robot. A few load a small policy and are refused while
  a deployment is running on the same computer; run the suite when none is.
- Three tests guard the documentation and the package's independence:
  `tests/test_doc_references.py` (every path a doc, comment or message names
  exists), `tests/test_selfcontained.py` (nothing names anything outside this
  package) and `tests/test_skill_templates.py` (the skill's templates run).

## Where to start reading

Every module's docstring says what it is responsible for, which process runs
it, and which document explains its design. `docs/overview.md` has the process
table and the source layout; the other documents in this folder explain the
parts most often changed.

## Known limits

`docs/deploying_a_policy/running_and_live_dashboard.md`, "Known limits".
