# Examples: policy adapters

Four small adapters, one concept each. None loads a model, so each runs as
shipped. The contract they follow: `docs/deploying_a_policy/policy_contract.md`.

| adapter | deploy file | shows |
|---|---|---|
| `hold.py` | `examples/deploy.toml` | the smallest useful adapter: end-effector actions that hold the arm where it is. The safe first launch on a new robot |
| `sine_eef.py` | `sine_eef.toml` | end-effector actions: both wrists trace a 2 cm sinusoid |
| `sine_joint.py` | `sine_joint.toml` | joint actions: both elbows bend on a sinusoid, with the waist commanded upright |
| `history.py` | `history.toml` | past observations: declaring `history` and reading earlier frames and a window of states through `reader` |

    elrobot_deploy check-policy examples/policies/sine_eef.toml     # offline
    elrobot_deploy run examples/policies/sine_eef.toml              # MOVES THE ARM

`examples/deploy.toml` is the fully commented deploy file; the ones here set
only what they need.

To start an adapter of your own, copy the templates in
`skills/deploy-a-policy/templates/` instead: they carry the TODOs.
