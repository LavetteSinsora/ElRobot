"""Code for the rollout-record format — no records live here.

Records are written by P9 to `[record] out` (default `rollout_record/` beside
the deploy file). This package holds what knows their format: `manifest`
builds the provenance block P9 writes into each `manifest.json`, and `reader`
reads a record back. Records are inspected with `elrobot_deploy inspect`
(`docs/deploying_a_policy/rollout_inspection.md`).
"""
