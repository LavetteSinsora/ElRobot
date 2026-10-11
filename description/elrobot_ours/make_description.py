"""Build our ElRobot description: the upstream URDF plus what we actually assembled.

    uv run python description/elrobot_ours/make_description.py     # writes elrobot.urdf and assets/

Inputs (re-run after any recalibration; nothing generated is edited by hand):
  ../elrobot_upstream/elrobot_follower.urdf   geometry, joint origins and axes (kept as-is)
  ../../calibration/calibration.json          safe joint limits, and the gripper's open end
  ../../calibration/hand_ranges.json          the gripper's full hand-swept travel (its closed end)
  printed/camera_mount_yoke_facet_*.stl        the printed camera mount, in print orientation

What changes against upstream:
  1. joint limits         the calibrated safe ranges (measured ends minus elrobot/calibration.py's margins)
  2. gripper jaws         frames moved onto the jaws (upstream's sit ~0.3 m away), and the jaw/gear
                          ratio scaled so the jaw tips meet at our measured closed stop
  3. tcp                  fixed frame on Gripper_Base: midpoint of the jaw tips' far ends; x out of the
                          jaws, y left, z up (at the zero pose). The only end-effector frame we use.
  4. camera_mount         the Yoke + Facet mount, bolted through the base's two 3.5 mm top holes
  5. wrist_camera_optical board front-face centre + LENS_OFFSET_MM along the view axis; z forward,
                          x right, y down (OpenCV), for the upright (de-rotated) image

README.md beside this file describes the frames and how each step was checked.
"""

from __future__ import annotations

import json
import math
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
UPSTREAM = HERE.parent / "elrobot_upstream" / "elrobot_follower.urdf"
UPSTREAM_ASSETS = "../elrobot_upstream/assets"
OUT_URDF = HERE / "elrobot.urdf"
PRINTED = HERE / "printed"
ASSETS = HERE / "assets"

RAD_PER_STEP = 2 * math.pi / 4096
GRIPPER_MARGIN_DEG = 3.0          # elrobot/calibration.py's margin for joint 8

# The camera mount, from printed/camera_mounts_parametric.py. Its own CAD frame is called S there:
# the foot's contact face is z = 0, x points from the mount towards its two foot holes.
FOOT_HOLES_S = np.array([[14.0, 7.0, 0.0], [14.0, -7.0, 0.0]])      # mm
VIEW_TILT_DEG = 25.0              # the view axis is (cos, 0, -sin) in S: 25 deg below horizontal
BOARD_FRONT_S = np.array([-0.19, 0.0, 51.04])                       # mm, board front-face centre
FRONT_MM = 4.5                    # housing front plate: board front face to the housing's face
COVER_OUTER_Z = -7.6              # camera-frame z of the back cover's outer face (BACK_Z - COVER_T)
#: The fisheye's projection centre in front of the board face, mm. Approximate (the script's lens
#: model has its dome centre 11 mm out); a camera calibration can refine it.
LENS_OFFSET_MM = 11.0
#: Printed PLA (1.24 g/cm3) at roughly half fill, kg/m3. The arm is position-controlled, so the
#: inertial only matters for simulation.
MOUNT_DENSITY = 1240.0 * 0.5
#: HX200S board and fisheye lens, kg, at the board centre. Rough.
CAMERA_MASS = 0.012

STL_DTYPE = np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")])


# --- meshes ---------------------------------------------------------------------------------

def read_stl(path: Path) -> np.ndarray:
    """Binary STL -> (n, 3, 3) triangles."""
    d = path.read_bytes()
    n = struct.unpack("<I", d[80:84])[0]
    return np.frombuffer(d[84:84 + 50 * n], dtype=STL_DTYPE)["v"].reshape(-1, 3, 3).astype(np.float64)


def write_stl(path: Path, tris: np.ndarray, name: str) -> None:
    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(tris), dtype=STL_DTYPE)
    rec["n"], rec["v"] = nrm, tris.reshape(-1, 9)
    header = f"{name} (make_description.py)".encode()[:80].ljust(80, b" ")
    path.write_bytes(header + struct.pack("<I", len(tris)) + rec.tobytes())


def circle_fit(p: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Least-squares circle through 2-D points -> (centre, radius, worst residual)."""
    A = np.c_[2 * p, np.ones(len(p))]
    c = np.linalg.lstsq(A, (p ** 2).sum(1), rcond=None)[0]
    r = math.sqrt(c[2] + c[0] ** 2 + c[1] ** 2)
    return c[:2], r, float(np.abs(np.linalg.norm(p - c[:2], axis=1) - r).max())


def mass_properties(tris: np.ndarray, density: float):
    """(mass, centre of mass, inertia about the COM) of a closed mesh, by signed tetrahedra."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    v = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    com = (v[:, None] * (a + b + c) / 4.0).sum(0) / v.sum()
    S = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            S[i, j] = (v * (2 * (a[:, i] * a[:, j] + b[:, i] * b[:, j] + c[:, i] * c[:, j])
                            + a[:, i] * b[:, j] + a[:, j] * b[:, i] + a[:, i] * c[:, j]
                            + a[:, j] * c[:, i] + b[:, i] * c[:, j] + b[:, j] * c[:, i])).sum() / 20.0
    m = density * v.sum()
    S = density * S - m * np.outer(com, com)
    return m, com, np.trace(S) * np.eye(3) - S


# --- small geometry helpers -------------------------------------------------------------------

def frame(x_axis, y_hint) -> np.ndarray:
    """Rotation with x along `x_axis` and y as close to `y_hint` as orthogonality allows."""
    x = np.asarray(x_axis, float) / np.linalg.norm(x_axis)
    y = np.asarray(y_hint, float) - (np.asarray(y_hint, float) @ x) * x
    y /= np.linalg.norm(y)
    return np.c_[x, y, np.cross(x, y)]


def rpy_from_matrix(R: np.ndarray) -> tuple[float, float, float]:
    """URDF rpy (rotate about fixed X, then Y, then Z) of a rotation matrix."""
    pitch = math.asin(-max(-1.0, min(1.0, R[2, 0])))
    if abs(math.cos(pitch)) < 1e-9:
        return math.atan2(-R[1, 2], R[1, 1]), pitch, 0.0
    return math.atan2(R[2, 1], R[2, 2]), pitch, math.atan2(R[1, 0], R[0, 0])


def fmt(v) -> str:
    return " ".join(f"{float(x):.6g}" for x in v)


def _xyz(el) -> np.ndarray:
    return np.array([float(v) for v in el.get("xyz", "0 0 0").split()])


def _axis(joint) -> np.ndarray:
    a = np.array([float(v) for v in joint.find("axis").get("xyz").split()])
    return a / np.linalg.norm(a)


# --- the camera mount: print orientation -> the S frame ----------------------------------------

def body_print_to_S(tris: np.ndarray):
    """Find the printed body's S frame from three features, so nothing about how it was laid out
    for printing is assumed: the foot's contact face and its two 3.5 mm holes, and the lens
    window (a 23 mm bore along the view axis). Returns (R, origin, checks) with
    p_S = R.T @ (p_print - origin)."""
    V = tris.reshape(-1, 3)
    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    area = np.linalg.norm(nrm, axis=1)
    nrm /= np.maximum(area[:, None], 1e-12)

    # The lens window: the z level whose ring of vertices is a circle of radius 11.5.
    bore = None
    for z in np.unique(np.round(V[:, 2], 2)):
        P = V[np.abs(V[:, 2] - z) < 0.01][:, :2]
        for centre_guess in (P.mean(0),) if len(P) else ():
            ring = P[np.abs(np.linalg.norm(P - centre_guess, axis=1) - 11.5) < 0.3]
            if len(ring) >= 24:
                ctr, r, err = circle_fit(ring)
                if abs(r - 11.5) < 0.02 and err < 0.02:
                    bore = ctr
                    break
        if bore is not None:
            break
    if bore is None:
        raise RuntimeError("lens window (r = 11.5 mm) not found in the mount body")
    # Printed lens-up: the view axis is +z and the housing's front face is its top.
    view = np.array([0.0, 0.0, 1.0])
    top = V[np.linalg.norm(V[:, :2] - bore, axis=1) < 16][:, 2].max()
    board = np.array([bore[0], bore[1], top - FRONT_MM])

    # The foot's contact face: the plane that holds two 3.5 mm holes 14 mm apart and faces away
    # from the camera. Every flat face is a candidate; the holes decide.
    found = None
    planes = {(tuple(np.round(n_, 3)), round(float(t_[0] @ n_), 2))
              for n_, t_, a_ in zip(nrm, tris, area) if a_ > 1.0}
    for n_, d in sorted(planes):
        nf = np.array(n_) / np.linalg.norm(n_)
        face = V[np.abs(V @ nf - d) < 0.02]
        if len(face) < 40 or (face.mean(0) - board) @ nf <= 0:
            continue
        # refit the plane exactly from its own vertices (the key was rounded)
        fit = np.linalg.svd(face - face.mean(0))[2][-1]
        nf = fit if fit @ nf > 0 else -fit
        d = float(face.mean(0) @ nf)
        face = V[np.abs(V @ nf - d) < 0.005]
        if len(face) < 40:
            continue
        holes = _hole_pair(face, nf, d)
        if holes is not None:
            found = (nf, holes)
            break
    if found is None:
        raise RuntimeError("no face with two 3.5 mm holes 14 mm apart: is this the Yoke + Facet body?")
    nf, holes = found

    # S: z from the contact face into the part; x from the board's footprint towards the holes.
    Sz = -nf
    mid = holes.mean(0)
    Sx = mid - (board - ((board - mid) @ nf) * nf)
    Sx -= (Sx @ Sz) * Sz
    Sx /= np.linalg.norm(Sx)
    R = np.c_[Sx, np.cross(Sz, Sx), Sz]
    origin = mid - FOOT_HOLES_S[0, 0] * Sx
    to_S = lambda p: R.T @ (np.asarray(p) - origin)
    checks = {"board_front_S": to_S(board), "view_S": R.T @ view,
              "holes_S": np.array(sorted((to_S(h) for h in holes), key=lambda p: -p[1]))}
    return R, origin, checks


def _hole_pair(face: np.ndarray, nf: np.ndarray, d: float):
    """Two r = 1.75 mm circles 14 mm apart among a plane's vertices, as 3-D centres, or None.
    Rims are dense rings of vertices (~0.35 mm apart); link points closer than 1 mm and fit each ring."""
    e1 = np.cross(nf, [0.0, 0.0, 1.0]) if abs(nf[2]) < 0.9 else np.cross(nf, [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(nf, e1)
    uv = np.unique(np.round(np.c_[face @ e1, face @ e2], 4), axis=0)
    label = -np.ones(len(uv), dtype=int)
    for i in range(len(uv)):
        if label[i] < 0:
            label[i] = label.max() + 1
            frontier = [i]
            while frontier:
                k = frontier.pop()
                near = np.where((label < 0) & (np.linalg.norm(uv - uv[k], axis=1) < 1.0))[0]
                label[near] = label[i]
                frontier.extend(near)
    circles = []
    for g in range(label.max() + 1):
        pts = uv[label == g]
        if len(pts) >= 12:
            ctr, r, err = circle_fit(pts)
            if abs(r - 1.75) < 0.02 and err < 0.02:
                circles.append(ctr)
    if len(circles) != 2 or abs(np.linalg.norm(circles[0] - circles[1]) - 14.0) > 0.05:
        return None
    return np.array([c[0] * e1 + c[1] * e2 + d * nf for c in circles])


def cover_print_to_camera(tris: np.ndarray) -> np.ndarray:
    """The back cover, print orientation -> the camera frame (x lateral, y up the face, z view).
    Fitted from its features: the two holes along its long side, the counterbore on its outer
    face, and the 14 mm cable notch at x = +4 on its bottom edge."""
    V = tris.reshape(-1, 3)
    lo, hi = V.min(0), V.max(0)
    c = (lo + hi) / 2
    long_ax = int(np.argmax((hi - lo)[:2]))
    short_ax = 1 - long_ax
    # The notched edge has two outline vertices the other edge lacks: the notch's sides.
    edges = {}
    for side, v in (("lo", lo[short_ax]), ("hi", hi[short_ax])):
        edges[side] = set(np.round(V[np.abs(V[:, short_ax] - v) < 0.01][:, long_ax], 2))
    notched = max(edges, key=lambda s: len(edges[s]))
    sides = np.array(sorted(edges[notched] - edges["hi" if notched == "lo" else "lo"]))
    if len(sides) != 2 or abs(sides[1] - sides[0] - 14.0) > 0.1:
        raise RuntimeError(f"cover notch sides {sides}, expected two 14 mm apart")
    notch_x = sides.mean() - c[long_ax]
    if abs(abs(notch_x) - 4.0) > 0.1:
        raise RuntimeError(f"cover notch at {notch_x:.2f} mm, expected +-4")
    # the counterbore (r 2.1) is on the outer face
    outer = None
    for z in (lo[2], hi[2]):
        P = V[np.abs(V[:, 2] - z) < 0.01][:, :2]
        hole = c[:2] + np.eye(2)[long_ax] * 17.95
        if (np.abs(np.linalg.norm(P - hole, axis=1) - 2.1) < 0.05).sum() > 6:
            outer = z
    if outer is None:
        raise RuntimeError("cover counterbore not found")
    X = np.eye(3)[long_ax] * np.sign(notch_x)
    Y = -np.eye(3)[short_ax] * (1.0 if notched == "hi" else -1.0)        # the notch is at -Y
    Z = np.array([0.0, 0.0, 1.0 if outer == lo[2] else -1.0])
    if abs(np.cross(X, Y) @ Z - 1.0) > 1e-9:
        raise RuntimeError("cover features do not give a right-handed frame")
    p = V - np.array([c[0], c[1], outer])
    return (np.c_[p @ X, p @ Y, p @ Z] + [0.0, 0.0, COVER_OUTER_Z]).reshape(-1, 3, 3)


def camera_to_S(p: np.ndarray) -> np.ndarray:
    """The mount script's `to_S`: camera frame -> S frame (mm)."""
    a = math.radians(VIEW_TILT_DEG)
    U = np.array([0.0, 1.0, 0.0])
    Vup = np.array([math.sin(a), 0.0, math.cos(a)])
    N = np.array([math.cos(a), 0.0, -math.sin(a)])
    return BOARD_FRONT_S + p[..., :1] * U + p[..., 1:2] * Vup + p[..., 2:3] * N


# --- the gripper base's mount holes -> pose of S in Gripper_Base ------------------------------

def base_holes(root) -> tuple[np.ndarray, np.ndarray]:
    """The two 3.5 mm holes in Gripper_Base's top face, Gripper_Base link frame (metres), left
    one (smaller x) first, and the face's upward normal. They are the left-right pair of the
    four-hole cross on top; the front-back pair is 2 mm and is not the mount's."""
    vis = root.find("link[@name='Gripper_Base_v1_1']/visual")
    off = _xyz(vis.find("origin"))
    V = read_stl(UPSTREAM.parent / "assets" / Path(vis.find("geometry/mesh").get("filename")).name)
    V = V.reshape(-1, 3)
    top = V[V[:, 2] > V[:, 2].max() - 12]
    holes, rims = [], []
    for guess in ((-3.4, 284.9), (10.6, 284.7)):            # mm, the mesh's assembly frame
        P = top[np.linalg.norm(top[:, :2] - guess, axis=1) < 2.2]
        ctr, r, err = circle_fit(P[:, :2])
        if abs(r - 1.75) > 0.02 or err > 0.05:
            raise RuntimeError(f"Gripper_Base hole near {guess}: r = {r:.3f}, expected 1.75")
        z = P[:, 2].max()
        holes.append([ctr[0], ctr[1], z])
        rims.append(P[np.abs(P[:, 2] - z) < 0.02])
    rim = np.vstack(rims)
    n = np.linalg.svd(rim - rim.mean(0))[2][-1]
    return np.array(holes) * 0.001 + off, (n if n[2] > 0 else -n)


def mount_pose(root) -> np.ndarray:
    """4x4 pose of S in the Gripper_Base link frame: foot on the top face, holes on holes, the
    camera looking forward along the jaws (S y = +7 is the left hole)."""
    (left, right), Sz = base_holes(root)
    Sy = left - right
    Sy -= (Sy @ Sz) * Sz
    Sy /= np.linalg.norm(Sy)
    Sx = np.cross(Sy, Sz)
    T = np.eye(4)
    T[:3, :3] = np.c_[Sx, Sy, Sz]
    T[:3, 3] = (left + right) / 2 - FOOT_HOLES_S[0, 0] * 0.001 * Sx
    return T


# --- the gripper ------------------------------------------------------------------------------

def jaw_points(root, joint_name: str) -> np.ndarray:
    """A jaw's mesh vertices in the Gripper_Base link frame at q8 = 0 (fully open), metres."""
    j = root.find(f"joint[@name='{joint_name}']")
    vis = root.find(f"link[@name='{j.find('child').get('link')}']/visual")
    mesh = UPSTREAM.parent / "assets" / Path(vis.find("geometry/mesh").get("filename")).name
    return read_stl(mesh).reshape(-1, 3) * 0.001 + _xyz(vis.find("origin")) + _xyz(j.find("origin"))


def tcp_pose(root) -> np.ndarray:
    """4x4 TCP pose in Gripper_Base. x out of the jaws (the wrist-roll axis, pointing away from
    the arm), y left (the jaw-closing axis), z up; origin at the midpoint of the jaw tips' far
    ends. The jaws close symmetrically, so the point does not move with the gripper."""
    roll = _axis(root.find("joint[@name='rev_motor_07']"))
    jaw = _axis(root.find("joint[@name='rev_motor_08_1']"))
    R = frame(roll if roll[1] > 0 else -roll, jaw if jaw[0] < 0 else -jaw)
    tips = []
    for jn in ("rev_motor_08_1", "rev_motor_08_2"):
        P = jaw_points(root, jn)
        reach = P @ R[:, 0]
        tip = P[reach > reach.max() - 0.002].mean(0)
        tips.append(tip + (reach.max() - tip @ R[:, 0]) * R[:, 0])     # on the far end itself
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.mean(tips, axis=0)
    return T


def jaw_gap(root) -> float:
    """Inner gap between the jaw tips at q8 = 0 (fully open), metres."""
    a, b = jaw_points(root, "rev_motor_08_1"), jaw_points(root, "rev_motor_08_2")
    ta = a[a[:, 1] > a[:, 1].max() - 0.005]
    tb = b[b[:, 1] > b[:, 1].max() - 0.005]
    return float(tb[:, 0].min() - ta[:, 0].max())


# --- limits -----------------------------------------------------------------------------------

def calibrated_limits() -> tuple[dict[int, tuple[float, float]], float]:
    """Safe limits per joint (radians), and the gripper's measured closed stop.

    Joints 1-7 and the gripper's open end come from calibration.json. Its gripper closed end is
    capped at upstream's 126.2 deg, while the hand sweep (hand_ranges.json) measured the full
    travel from the open stop; that is used instead, minus the same 3 deg margin."""
    cal = json.loads((REPO / "calibration" / "calibration.json").read_text())["joints"]
    hand = json.loads((REPO / "calibration" / "hand_ranges.json").read_text())["joints"]
    limits = {}
    for k, c in cal.items():
        to_rad = lambda r, c=c: c["sign"] * (r - c["zero_reading"]) * RAD_PER_STEP
        limits[int(k)] = tuple(sorted((to_rad(c["min_reading"]), to_rad(c["max_reading"]))))
    closed = hand["8"]["span_steps"] * RAD_PER_STEP
    limits[8] = (limits[8][0], closed - math.radians(GRIPPER_MARGIN_DEG))
    return limits, closed


# --- build ------------------------------------------------------------------------------------

def _fixed_child(root, name: str, parent: str, T: np.ndarray, inertial=None):
    link = ET.SubElement(root, "link", name=name)
    if inertial is not None:
        m, com, I = inertial
        el = ET.SubElement(link, "inertial")
        ET.SubElement(el, "origin", xyz=fmt(com), rpy="0 0 0")
        ET.SubElement(el, "mass", value=f"{m:.6g}")
        ET.SubElement(el, "inertia", ixx=f"{I[0, 0]:.6g}", ixy=f"{I[0, 1]:.6g}", ixz=f"{I[0, 2]:.6g}",
                      iyy=f"{I[1, 1]:.6g}", iyz=f"{I[1, 2]:.6g}", izz=f"{I[2, 2]:.6g}")
    joint = ET.SubElement(root, "joint", name=f"{name}_joint", type="fixed")
    ET.SubElement(joint, "origin", xyz=fmt(T[:3, 3]), rpy=fmt(rpy_from_matrix(T[:3, :3])))
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=name)
    return link


def build() -> dict:
    tree = ET.parse(UPSTREAM)
    root = tree.getroot()
    root.set("name", "elrobot_ours")
    report: dict = {}

    # Measurements on upstream geometry, before anything is moved.
    T_tcp = tcp_pose(root)
    T_mount = mount_pose(root)
    gap = jaw_gap(root)

    for mesh in root.iter("mesh"):
        mesh.set("filename", f"{UPSTREAM_ASSETS}/{Path(mesh.get('filename')).name}")

    # 1. limits
    limits, closed = calibrated_limits()
    for k, (lo, hi) in limits.items():
        el = root.find(f"joint[@name='rev_motor_0{k}']/limit")
        el.set("lower", f"{lo:.4f}")
        el.set("upper", f"{hi:.4f}")
    report["limits_deg"] = {k: (round(math.degrees(a), 1), round(math.degrees(b), 1))
                            for k, (a, b) in limits.items()}

    # 2. gripper jaws: frames onto the jaws, ratio so the tips meet at our closed stop
    ratio = gap / 2 / closed
    report.update(jaw_mm_per_rad=ratio * 1000, open_tip_gap_mm=gap * 1000,
                  gripper_closed_stop_deg=math.degrees(closed))
    for jn, sign in (("rev_motor_08_1", -1.0), ("rev_motor_08_2", 1.0)):
        j = root.find(f"joint[@name='{jn}']")
        old = _xyz(j.find("origin"))
        P = jaw_points(root, jn)
        new = np.round((P.min(0) + P.max(0)) / 2, 6)            # the jaw's bounding-box centre
        j.find("origin").set("xyz", fmt(new))
        link = root.find(f"link[@name='{j.find('child').get('link')}']")
        for tag in ("visual", "collision", "inertial"):
            for el in link.findall(tag):
                el.find("origin").set("xyz", fmt(_xyz(el.find("origin")) + old - new))
        j.find("mimic").set("multiplier", f"{sign * ratio:.6f}")
        lo, hi = sorted((0.0, sign * ratio * limits[8][1]))
        j.find("limit").set("lower", f"{lo:.5f}")
        j.find("limit").set("upper", f"{hi:.5f}")

    # 3. tcp
    _fixed_child(root, "tcp", "Gripper_Base_v1_1", T_tcp)
    report["tcp_in_gripper_base_mm"] = T_tcp[:3, 3] * 1000

    # 4. camera mount, meshes re-expressed in its own S frame
    ASSETS.mkdir(exist_ok=True)
    body = read_stl(PRINTED / "camera_mount_yoke_facet_body.stl")
    R, origin, checks = body_print_to_S(body)
    body_S = (body - origin) @ R
    cover_S = camera_to_S(cover_print_to_camera(read_stl(PRINTED / "camera_mount_yoke_facet_back_cover.stl")))
    write_stl(ASSETS / "camera_mount_body.stl", body_S, "camera mount body, S frame, mm")
    write_stl(ASSETS / "camera_mount_back_cover.stl", cover_S, "camera mount back cover, S frame, mm")
    report["mount_fit"] = checks
    m, com, I = mass_properties(np.vstack([body_S, cover_S]) * 0.001, MOUNT_DENSITY)
    a = math.radians(VIEW_TILT_DEG)
    N = np.array([math.cos(a), 0.0, -math.sin(a)])
    board_c = (BOARD_FRONT_S - 0.8 * N) * 0.001
    total = m + CAMERA_MASS
    com_t = (m * com + CAMERA_MASS * board_c) / total
    shift = lambda mass, d: mass * (d @ d * np.eye(3) - np.outer(d, d))
    I = I + shift(m, com - com_t) + shift(CAMERA_MASS, board_c - com_t)
    report["mount_mass_g"] = total * 1000
    mount = _fixed_child(root, "camera_mount", "Gripper_Base_v1_1", T_mount, inertial=(total, com_t, I))
    for name, tags in (("camera_mount_body.stl", ("visual", "collision")),
                       ("camera_mount_back_cover.stl", ("visual",))):
        for tag in tags:
            el = ET.SubElement(mount, tag)
            ET.SubElement(el, "origin", xyz="0 0 0", rpy="0 0 0")
            ET.SubElement(ET.SubElement(el, "geometry"), "mesh", filename=f"assets/{name}",
                          scale="0.001 0.001 0.001")
            if tag == "visual":
                ET.SubElement(el, "material", name="silver")

    # 5. the camera's optical frame: z along the view axis, y down the board face, x right
    Vup = np.array([math.sin(a), 0.0, math.cos(a)])
    T_opt = np.eye(4)
    T_opt[:3, :3] = np.c_[np.cross(-Vup, N), -Vup, N]
    T_opt[:3, 3] = (BOARD_FRONT_S + LENS_OFFSET_MM * N) * 0.001
    _fixed_child(root, "wrist_camera_optical", "camera_mount", T_opt)
    T_cam = T_mount @ T_opt
    report["camera_in_gripper_base_mm"] = T_cam[:3, 3] * 1000
    report["camera_axes_in_gripper_base"] = T_cam[:3, :3]

    ET.indent(tree, "  ")
    OUT_URDF.write_text(
        '<?xml version="1.0"?>\n'
        "<!-- GENERATED by description/elrobot_ours/make_description.py from the upstream URDF,\n"
        "     calibration/calibration.json, calibration/hand_ranges.json and the printed camera\n"
        "     mount. Do not edit by hand: change the inputs and re-run the script. -->\n"
        + ET.tostring(root, encoding="unicode") + "\n")
    return report


def main() -> None:
    r = build()
    np.set_printoptions(suppress=True)
    print(f"wrote {OUT_URDF.relative_to(REPO)} and {ASSETS.relative_to(REPO)}/")
    print("limits (deg):")
    for k, (lo, hi) in r["limits_deg"].items():
        print(f"  {k}: {lo:7.1f} .. {hi:6.1f}")
    print(f"gripper: closed stop {r['gripper_closed_stop_deg']:.1f} deg from open, open tip gap "
          f"{r['open_tip_gap_mm']:.1f} mm, jaw travel {r['jaw_mm_per_rad']:.2f} mm/rad each")
    print("tcp in Gripper_Base (mm):", np.round(r["tcp_in_gripper_base_mm"], 2))
    f = r["mount_fit"]
    print("mount fit, S frame (expect board -0.19 0 51.04, view 0.9063 0 -0.4226, holes 14 +-7 0):")
    print("  board", np.round(f["board_front_S"], 3), " view", np.round(f["view_S"], 4),
          " holes", np.round(f["holes_S"], 3).tolist())
    print(f"mount + camera: {r['mount_mass_g']:.1f} g")
    print("camera in Gripper_Base (mm):", np.round(r["camera_in_gripper_base_mm"], 1))
    print("  its x (right), y (down), z (view) in Gripper_Base axes:\n",
          np.round(r["camera_axes_in_gripper_base"], 3))


if __name__ == "__main__":
    main()
