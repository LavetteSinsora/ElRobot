"""Render the description at a pose, to compare with the real arm by eye.

    uv run --with matplotlib python description/elrobot_ours/render.py rest -o rest.png
    uv run --with matplotlib python description/elrobot_ours/render.py --deg 0 0 0 0 0 0 0 60 --zoom gripper

Draws every visual mesh, plus the tcp and wrist_camera_optical frames as RGB axes (x red,
y green, z blue). `--zoom gripper` frames the gripper and camera mount only.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))

from elrobot.kinematics import ARM_JOINTS, Robot, rpy_matrix  # noqa: E402
from elrobot.poses import POSES  # noqa: E402
from make_description import OUT_URDF, read_stl  # noqa: E402

GRIPPER_LINKS = {"Gripper_Base_v1_1", "ST3215_8_v1_1", "Gripper_Gear_v1_1", "Gripper_Jaw_01_v1_1",
                 "Gripper_Jaw_02_v1_1", "camera_mount", "tcp", "wrist_camera_optical"}


def visuals(urdf: Path):
    """(link, 4x4 visual origin, mesh path, scale) for every mesh visual."""
    for link in ET.parse(urdf).getroot().findall("link"):
        for vis in link.findall("visual"):
            mesh = vis.find("geometry/mesh")
            if mesh is None:
                continue
            o = vis.find("origin")
            T = np.eye(4)
            if o is not None:
                T[:3, :3] = rpy_matrix(*[float(v) for v in o.get("rpy", "0 0 0").split()])
                T[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
            scale = float(mesh.get("scale", "1 1 1").split()[0])
            yield link.get("name"), T, (urdf.parent / mesh.get("filename")).resolve(), scale


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pose", nargs="?", default="zero", choices=sorted(POSES))
    ap.add_argument("--deg", type=float, nargs=8, metavar="Q", help="joint angles 1-8 in degrees")
    ap.add_argument("--zoom", choices=["arm", "gripper"], default="arm")
    ap.add_argument("-o", "--out", type=Path, default=Path("render.png"))
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    deg = args.deg if args.deg else [POSES[args.pose][j] for j in range(1, 9)]
    q = dict(zip(ARM_JOINTS, np.radians(deg)))
    robot = Robot(OUT_URDF)

    fig = plt.figure(figsize=(13, 6.5))
    views = [(20, -60, "front-left"), (10, 0, "from the side (+x)"), (80, -90, "from above")]
    pts_all = []
    meshes = []
    for link, T_vis, path, scale in visuals(OUT_URDF):
        if args.zoom == "gripper" and link not in GRIPPER_LINKS:
            continue
        T = robot.fk(q, link) @ T_vis
        tris = read_stl(path) * scale
        tris = tris @ T[:3, :3].T + T[:3, 3]
        meshes.append((link, tris))
        pts_all.append(tris.reshape(-1, 3))
    pts = np.vstack(pts_all)
    centre, half = (pts.min(0) + pts.max(0)) / 2, (pts.max(0) - pts.min(0)).max() / 2 * 1.05
    for i, (elev, azim, title) in enumerate(views):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        for link, tris in meshes:
            colour = "tab:orange" if link == "camera_mount" else ("0.55" if "Jaw" in link else "0.8")
            ax.add_collection3d(Poly3DCollection(tris, facecolor=colour, edgecolor="none",
                                                 alpha=0.9 if link == "camera_mount" else 0.5))
        for name in ("tcp", "wrist_camera_optical"):
            T = robot.fk(q, name)
            for k, c in enumerate("rgb"):
                ax.quiver(*T[:3, 3], *(T[:3, k] * half * 0.25), color=c, linewidth=2)
            ax.text(*T[:3, 3], f" {name}", fontsize=7)
        ax.set_xlim(centre[0] - half, centre[0] + half)
        ax.set_ylim(centre[1] - half, centre[1] + half)
        ax.set_zlim(centre[2] - half, centre[2] + half)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    fig.suptitle(f"elrobot_ours at {' '.join(f'{d:g}' for d in deg)} deg  (x red, y green, z blue)")
    plt.tight_layout()
    fig.savefig(args.out, dpi=90)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
