# Adapter patterns

Situations that come up when serving real checkpoints, and a way to handle
each inside the adapter and the deploy file, without changing elrobot_deploy. Read
the section that matches the situation.

- [Several training runs, one adapter](#several-training-runs-one-adapter)
- [A switch the user sets at launch](#a-switch-the-user-sets-at-launch)
- [Serving without one of the cameras](#serving-without-one-of-the-cameras)
- [A longer prompt than training used](#a-longer-prompt-than-training-used)
- [A checkpoint that describes itself](#a-checkpoint-that-describes-itself)
- [A model whose code is not installed as a package](#a-model-whose-code-is-not-installed-as-a-package)

## Several training runs, one adapter

Runs that share a model and a data format (a fine-tune of an earlier run, the
same recipe on a different mixture) differ only in their training
configuration and statistics. Write one adapter and put everything that
differs behind one method; each sibling is then a subclass of a few lines with
its own deploy file.

```python
class BasePolicy(g1d.Policy):
    def run_config(self):
        from .run import RUN
        return RUN

    def __init__(self, checkpoint, prompt, robot):
        super().__init__(checkpoint, prompt, robot)
        self.run = self.run_config()      # everything below reads self.run
```

```python
class SiblingPolicy(BasePolicy):
    def run_config(self):
        from .run import RUN              # this sibling's own configuration
        return RUN
```

`run_config` may look at `self.checkpoint` (for example, which run directory
it sits in) when one adapter serves two runs.

## A switch the user sets at launch

For behaviour that belongs to one model and should not become a elrobot_deploy
feature, read an environment variable in `__init__`:

```python
def _skip_self_check() -> bool:
    return os.environ.get("MYMODEL_SKIP_SELF_CHECK", "") not in ("", "0")
```

The user sets it in the deploy file's `[policy.env]`, or for one launch:

```bash
elrobot_deploy run deploy.toml --set policy.env.MYMODEL_SKIP_SELF_CHECK=1
```

Two rules:

- **Read it in `__init__` or later, not at class level.** The infra reads the
  class's declarations before `[policy.env]` is applied, so a declaration
  cannot depend on such a variable. To make a declared value depend on it, set
  the attribute on the instance in `__init__` (`self.max_prompt_tokens = ...`).
- **Write the flag as a comment in the deploy file**, next to the setting it
  affects, so the user can find it.

## Serving without one of the cameras

The deploy file can narrow the cameras the adapter receives, as long as the
anchor camera stays:

```bash
elrobot_deploy run deploy.toml --set 'policy.uses_cameras=["lwrist", "rwrist"]'
```

`obs.frames` then has no entry for the camera left out. An adapter that
supports this checks for the camera and fills the model's image slot the way
training represented a missing view (commonly a black image with the slot's
mask set to false):

```python
if "head_right" in obs.frames:
    image, present = prepare(obs.frames["head_right"].pixels), True
else:
    image, present = np.zeros(shape, np.uint8), False
```

Tell the user whether the model saw missing views in training. A model that
always had the camera can behave unpredictably without it.

## A longer prompt than training used

`max_prompt_tokens` is the length the adapter pads to and compiles at; it can
exceed the training length. If the adapter also verifies the checkpoint
against its training configuration, keep two values: the training length for
that verification, and the serving length for the tokenizer padding and the
model's input shape. Build the model configuration for serving as a copy of
the training configuration with only the prompt length replaced, and check
that the serving length is not shorter than the training one.

## A checkpoint that describes itself

When a checkpoint directory carries a record of how it was trained (a config
dump, a hash over configuration and statistics), compare it in `__init__` with
what the adapter is about to assume. A mismatch otherwise produces a policy
that runs and is wrong.

Make the refusal useful to someone who did not write the adapter:

- say which fields differ, not only that a hash differs;
- say the likely cause (the training code changed after this checkpoint was
  written, or the wrong run is selected);
- name the launch switch that turns the refusal into a warning, for the case
  where the user knows the difference does not affect serving.

Better still, when the checkpoint's record is complete enough, build the
adapter's configuration from the record instead of from the current code; then
later changes to the training code cannot make old checkpoints unservable.

## A model whose code is not installed as a package

If the model's code is used from a source checkout through `PYTHONPATH`, or
needs environment variables at import time, point `[policy] python` at a
wrapper script instead of a Python binary:

```bash
#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$here/model_repo:/path/to/elrobot_deploy/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$here/model_repo/.venv/bin/python" "$@"
```

Putting elrobot_deploy's `src` directory on `PYTHONPATH` this way also replaces
installing it into the model's environment. Variables that code reads while
being imported belong here: the infra imports the adapter once without
`[policy.env]`, to read its declarations, before the policy process applies it.
