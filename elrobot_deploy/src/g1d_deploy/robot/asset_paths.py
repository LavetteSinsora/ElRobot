"""Where the model files are: `robot/assets/unitree_g1d` (the G1-D URDF and meshes)
and `robot/assets/unitree_dex1` (the Dex-1's meshes), inside the package.
Imported by `g1d_model.py` and `mujoco_model.py`; no dependencies."""

import os
import pathlib

#: g1d_deploy/robot/asset_paths.py -> g1d_deploy/ (the installed package; assets ship inside it)
PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]


def assets_dir() -> pathlib.Path:
    """Robot models and gripper meshes, shipped inside the package."""
    return pathlib.Path(os.environ.get("G1D_ASSETS", PACKAGE_ROOT / "robot" / "assets"))
