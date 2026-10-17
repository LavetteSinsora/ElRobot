# legacy_g1d: the parts of g1d_deploy not ported yet

`elrobot_deploy/` started as a verbatim copy of g1d_deploy (`../UPSTREAM.md`), a runtime for a
two-arm Unitree G1-D with a waist, DDS, a robot board and a PICO headset. The single-arm rebuild
(PORTING_PLAN.md workstreams 0 and 1) kept the parts that carry over and moved everything else
here, **unchanged, at the same relative paths**, so the reasoning in it stays one directory away.

Nothing here is imported, packaged, linted or tested. It is reference material, to be ported
subsystem by subsystem when its milestone comes:

| subsystem | where it is here | when it comes back |
|---|---|---|
| policy runtime: P5, `policy_contract/`, chunk blending, real-time chunking, `examples/`, `skills/` | `src/elrobot_deploy/{policy_contract,runtime/processes/p5_policy.py,runtime/execution/chunk_blend.py}` | policies / DAgger (after M3) |
| recording: P9, `rollout_record_utils/` | `src/elrobot_deploy/runtime/processes/p9_record.py` | M5 |
| dashboards: P10 live, rollout inspection, web | `src/elrobot_deploy/dashboards/` | M5 |
| cameras: P2, the fisheye lens fit, the board's image server and exposure loop | `src/elrobot_deploy/runtime/drivers/{cameras.py,fisheye_circle_fit.py,board/}` | **ported 2026-10-17** to `runtime/cameras/` + `processes/p2_camera.py` (the lens fit and the exposure law unchanged; no board, ZMQ or SSH); kept here as reference |
| CHECK verifiers, `elrobot_deploy check`, machine file, CPU pinning | `src/elrobot_deploy/{cli/check.py,runtime/launcher/startup_checks.py,config_loading/machine_file.py,runtime/cpu_pinning.py}` | M5 |
| deploy.toml resolution (`config_loading/`) | `src/elrobot_deploy/config_loading/` | with the policy runtime |
| G1-only, never coming back: DDS, `rt/lowstate`/`rt/lowcmd`, P1/P7_body/P8, the waist and dual-arm IK, the lift column, Dex-1/BrainCo, the G1-D model and assets, PICO service | `src/elrobot_deploy/{runtime/drivers/dds,robot/assets,robot/g1d_model.py,runtime/ik/waist_*}` ... | never |

Files the rebuild rewrote in place (the arena schema and rings, P0, P6, the CLI, the intervention
maths, the IK registry) also have their g1d version copied here, so a diff against the new one
shows what changed. The docs under `docs/` (and `README_g1d.md`) describe g1d's runtime, not the current one; the
current one is `../docs/architecture.md`.

The whole tree as it was before the move is commit `fc9c2cc`.
