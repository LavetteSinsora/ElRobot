"""Our URDF as the dashboard's browser draws it: the kinematic tree and the meshes, as JSON.

The page builds one three.js object tree per arm it shows (measured, commanded, the home
preview) from this, and poses it from joint values itself, so the server never does kinematics
and the drawing is the URDF's own geometry. No MuJoCo: the dashboard must not import the IK
stack.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from elrobot_deploy.robot.elrobot_joints import JOINT_NAMES


def _floats(text: str | None, n: int, default: float = 0.0) -> list[float]:
    if not text:
        return [default] * n
    return [float(v) for v in text.split()]


def describe(urdf: str | Path) -> tuple[dict, list[Path]]:
    """`(robot JSON, mesh files)`. Mesh `i` is served at `/mesh/i`; only these files are."""
    urdf = Path(urdf).resolve()
    root = ET.parse(urdf).getroot()
    colors = {m.get("name"): _floats(m.find("color").get("rgba"), 4)
              for m in root.findall("material") if m.find("color") is not None}
    meshes: list[Path] = []
    links = {}
    for link in root.findall("link"):
        visuals = []
        for v in link.findall("visual"):
            mesh = v.find("geometry/mesh")
            if mesh is None:
                continue
            path = (urdf.parent / mesh.get("filename")).resolve()
            if path not in meshes:
                meshes.append(path)
            origin = v.find("origin")
            mat = v.find("material")
            rgba = None
            if mat is not None:
                c = mat.find("color")
                rgba = _floats(c.get("rgba"), 4) if c is not None else colors.get(mat.get("name"))
            visuals.append({
                "mesh": f"/mesh/{meshes.index(path)}",
                "xyz": _floats(origin.get("xyz") if origin is not None else None, 3),
                "rpy": _floats(origin.get("rpy") if origin is not None else None, 3),
                "scale": _floats(mesh.get("scale"), 3, 1.0),
                "rgba": rgba,
            })
        links[link.get("name")] = {"visuals": visuals}
    joints = []
    children = set()
    for j in root.findall("joint"):
        origin, axis, limit, mimic = (j.find("origin"), j.find("axis"), j.find("limit"),
                                      j.find("mimic"))
        child = j.find("child").get("link")
        children.add(child)
        joints.append({
            "name": j.get("name"), "type": j.get("type"),
            "parent": j.find("parent").get("link"), "child": child,
            "xyz": _floats(origin.get("xyz") if origin is not None else None, 3),
            "rpy": _floats(origin.get("rpy") if origin is not None else None, 3),
            "axis": _floats(axis.get("xyz") if axis is not None else None, 3) if axis is not None
            else [1.0, 0.0, 0.0],
            "lower": float(limit.get("lower")) if limit is not None and limit.get("lower") else None,
            "upper": float(limit.get("upper")) if limit is not None and limit.get("upper") else None,
            "mimic": None if mimic is None else {
                "joint": mimic.get("joint"),
                "multiplier": float(mimic.get("multiplier", 1.0)),
                "offset": float(mimic.get("offset", 0.0))},
        })
    roots = [name for name in links if name not in children]
    return {"name": root.get("name"), "root": roots[0], "links": links, "joints": joints,
            "arm_joints": list(JOINT_NAMES), "tcp": "tcp"}, meshes


__all__ = ["describe"]
