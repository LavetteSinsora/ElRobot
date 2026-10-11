"""Parametric wrist-camera mounts for the ElRobot follower (HX200S board).

All geometry is built in the frame of NormaCore's CameraMount_square_27mm
("S frame": the part's own CAD frame as placed in the arm assembly).
The mounting foot (2 mm plate, two 3.5 mm holes 14 mm apart) is rebuilt
exactly from the original, so every design bolts on the same way.

Camera frame: X = lateral, Y = up the face, Z = viewing direction.
Board front face sits at Z = 0, centred where the original frame held
its 32 mm board.
"""
import cadquery as cq, numpy as np, math

# ---------------- camera board (HX200S, measured from photo; CHECK) --------
BW, BH, BT = 30.0, 25.0, 1.6       # HX200S board, measured tip to tip by Chris (30 x 25 mm)
CLR = 0.35                         # clearance per side in the pocket (ruler reading, so a little extra)
LENS_D = 23.0                      # front window, fits the fisheye barrel (~19 mm)
CONN_W = 15.0
CONN_X = 4.0                       # JST connector centre, seen from the front (+ = right); at the bottom edge (measured on the print)
REAR = 4.0                         # room behind the board for components
LEDGE = 1.0                        # board rests on this rim; inside it the lip is relieved
RELIEF = 1.4                       # relief depth for the mic / pads on the front face

PW, PH = BW + 2 * CLR, BH + 2 * CLR   # pocket
# Board screws: four M2 x 6 self-tapping screws through the board's U-notches (semicircles cut into the
# two long edges, open toward the edge). Measured by Chris: from the left edge, the lower-left notch spans
# 1.5-4.0 mm (centre 2.75, width 2.5) and the lower-right one 25.5-28.0 mm (centre 26.75); the board is
# symmetric top/bottom. Centre-to-centre 24.0 mm, i.e. +/-12.0 mm about the board centre.
NOTCH_X = 12.0
NOTCH_Y = BH / 2
NOTCH_D = 2.5
PLUG_POCKET = 9.0      # plug pocket depth below the board edge (inside the slim crossbar, open at the back)
CB_H = 6.5             # slim yoke crossbar, same look as the original Yoke + Facet
SCREW_PILOT = 1.7      # M2 self-tapping pilot, runs right through the front plate
HEAD_CLR = 4.8         # room for the screw head and a screwdriver behind the board
FRONT = 4.5                        # front plate thickness (Z 0..FRONT); thick enough for the M2 board screws to bite
BACK_Z = -(BT + REAR)              # rear rim of the housing (-5.6)
COVER_T = 2.0

# ---------------- placement (from the original part) ----------------------
_a = math.radians(25.0)
N = np.array([math.cos(_a), 0.0, -math.sin(_a)])   # viewing direction (25 deg below horizontal)
U = np.array([0.0, 1.0, 0.0])             # lateral
V = np.cross(N, U) * -1                   # up the face
V = np.array([math.sin(_a), 0.0, math.cos(_a)])
O = np.array([-0.19, 0.0, 51.04])          # board front-face centre

def to_S(p):
    p = np.asarray(p, float)
    return O + p[0] * U + p[1] * V + p[2] * N

def place(shape):
    pl = cq.Plane(origin=tuple(O), xDir=tuple(U), normal=tuple(N))
    return shape.moved(cq.Location(pl))

# ---------------- the original foot, rebuilt --------------------------------
HOLES = [(14.0, 7.0), (14.0, -7.0)]
def foot():
    pts = [(4.25, -7.29), (4.74, -7.89), (14.71, -9.82), (16.67, -8.2),
           (16.67, 8.2), (14.71, 9.82), (4.74, 7.89), (4.25, 7.29)]
    f = cq.Workplane("XY").polyline(pts).close().extrude(2.0)
    for x, y in HOLES:
        f = f.cut(cq.Workplane("XY").center(x, y).circle(1.75).extrude(2.0))
    return f.val()

def head_clearance():
    # keep screwdriver access to both foot screws
    s = None
    for x, y in HOLES:
        c = cq.Workplane("XY").workplane(offset=2.0).center(x, y).circle(3.3).extrude(90).val()
        s = c if s is None else s.fuse(c)
    return s

# ---------------- shared functional core (in camera frame) ----------------
def core_cuts(outer_h):
    """Pocket, window, relief, connector notch. Returns a solid to subtract."""
    wp = cq.Workplane("XY")
    pocket = wp.workplane(offset=BACK_Z - 5).rect(PW, PH).extrude(5 - BACK_Z).val()   # Z BACK_Z-5 .. 0
    relief = wp.rect(PW - 2 * LEDGE, PH - 2 * LEDGE).extrude(RELIEF).val()            # Z 0..RELIEF
    window = wp.circle(LENS_D / 2).extrude(FRONT + 5).val()
    # The JST connector sits on the BACK of the board at one long edge, and the plug goes in parallel
    # to the board. The board is installed with that edge DOWN, so the plug and cable drop out of the
    # bottom of the housing (into the yoke crossbar, where there is one) and the cable leaves backwards,
    # the same way NormaCore's manual routes it down to the wiring bracket under the gripper.
    y_top, y_bot = -PH / 2 + 7.0, -BH / 2 - PLUG_POCKET          # plug pocket: plug + a cable bend, inside the crossbar, open at the back
    notch = (cq.Workplane("XY").workplane(offset=BACK_Z - 0.01)
             .center(CONN_X, (y_top + y_bot) / 2).rect(CONN_W, y_top - y_bot)
             .extrude(-BACK_Z - BT + 0.6).val())
    cuts = pocket.fuse(relief).fuse(window).fuse(notch)
    for sx in (-1, 1):
        for sy in (-1, 1):
            x, y = sx * NOTCH_X, sy * NOTCH_Y
            pilot = cq.Workplane("XY").workplane(offset=-0.01).center(x, y).circle(SCREW_PILOT / 2).extrude(FRONT + 0.2).val()
            head = cq.Workplane("XY").workplane(offset=BACK_Z - 6).center(x, y).circle(HEAD_CLR / 2).extrude(6 - BACK_Z - BT).val()
            cuts = cuts.fuse(pilot).fuse(head)
    return cuts

SCREW_X = PW / 2 + 2.6            # M2 self-tapping screws, left and right of the board
def screw_holes(z0, z1, d):
    s = None
    for sx in (-SCREW_X, SCREW_X):
        c = cq.Workplane("XY").workplane(offset=z0).center(sx, 0).circle(d / 2).extrude(z1 - z0).val()
        s = c if s is None else s.fuse(c)
    return s

def cover(outline_fn):
    """Back cover: outline matches the housing. A plain dust cover: the board is screwed down on its own."""
    c = outline_fn(BACK_Z - COVER_T, COVER_T)
    # no clamp posts any more: the board is screwed down through its notches; the cover is a dust cover
    # cable relief at the bottom and screw clearance + counterbore from the back
    c = c.cut(cq.Workplane("XY").workplane(offset=BACK_Z - COVER_T - 1)
              .center(CONN_X, (-40.0 + (-BH / 2 + 6.5)) / 2).rect(14.0, 40.0 - BH / 2 + 6.5)
              .extrude(COVER_T + REAR + 2).val())   # open notch over the connector + plug, from the cover's bottom edge
    c = c.cut(screw_holes(BACK_Z - COVER_T - 1, BACK_Z + 1, 2.4))
    c = c.cut(screw_holes(BACK_Z - COVER_T - 1, BACK_Z - COVER_T + 1.0, 4.2))
    return c

# ---------------- stem helpers (S frame) ------------------------------------
BASE_TOP = 6.0
def base_block(x0=4.25, x1=10.6, hw=8.2):
    return cq.Workplane("XY").workplane(offset=2.0).center((x0 + x1) / 2, 0).rect(x1 - x0, 2 * hw).extrude(BASE_TOP - 2.0).val()

def loft_rect(c0, w0, h0, c1, w1, h1, ax0, ax1):
    """Loft between two rectangles given centre, size and in-plane axes (S frame)."""
    def wire(c, w, h, ax):
        a, b = np.array(ax[0]), np.array(ax[1])
        pts = [c + sa * w / 2 * a + sb * h / 2 * b for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        return cq.Wire.makePolygon([cq.Vector(*p) for p in pts], close=True)
    return cq.Solid.makeLoft([wire(np.array(c0), w0, h0, ax0), wire(np.array(c1), w1, h1, ax1)], True)

# =====================================================================================
# Design 1  FACET: chamfered octagon housing in the arm's shield language, forked stem
# =====================================================================================
def d1_outline(z0, t, W=PW + 11.0, H=PH + 5.0, ch=5.5):
    pts = [(-W / 2 + ch, -H / 2), (W / 2 - ch, -H / 2), (W / 2, -H / 2 + ch), (W / 2, H / 2 - ch),
           (W / 2 - ch, H / 2), (-W / 2 + ch, H / 2), (-W / 2, H / 2 - ch), (-W / 2, -H / 2 + ch)]
    return cq.Workplane("XY").workplane(offset=z0).polyline(pts).close().extrude(t).val()

def design1():
    H = PH + 5.0
    body = d1_outline(BACK_Z, FRONT - BACK_Z)
    body = cq.Workplane().add(body).faces(">Z").edges().chamfer(1.6).val()
    # chamfer bezel around the lens
    cone = cq.Solid.makeCone(LENS_D / 2, LENS_D / 2 + 2.2, 1.21, cq.Vector(0, 0, FRONT - 1.2))
    body = body.cut(core_cuts(H / 2 - PH / 2)).cut(cone)
    body = body.cut(screw_holes(BACK_Z - 0.1, FRONT - 0.6, 1.6))
    housing = place(body)
    # forked stem: two faceted legs from the base to the bottom corners of the housing
    base = base_block()
    legs = None
    for s in (-1, 1):
        top_c = to_S((s * (PW / 2 + 1.5), -H / 2 + 1.0, -2.4))
        leg = loft_rect([7.4, s * 4.1, BASE_TOP - 0.01], 6.3, 8.2, top_c, 4.4, 8.5,
                        ([1, 0, 0], [0, 1, 0]), (N, U))
        legs = leg if legs is None else legs.fuse(leg)
    part = housing.fuse(base).fuse(legs).fuse(foot())
    part = part.cut(head_clearance())
    part = part.cut(place(core_cuts(H / 2 - PH / 2)))  # keep the pocket open where legs meet
    cov = place(cover(lambda z0, t: d1_outline(z0, t)))
    return part.clean(), cov.clean()

# =====================================================================================
# Design 2  POD: soft rounded capsule, one swan-neck stem into the back of the pod
# =====================================================================================
def d2_outline(z0, t, W=PW + 11.0, H=PH + 6.0, r=8.0):
    return cq.Workplane("XY").workplane(offset=z0).rect(W, H).extrude(t).edges("|Z").fillet(r).val()

def design2():
    H = PH + 6.0
    body = d2_outline(BACK_Z, FRONT - BACK_Z)
    body = cq.Workplane().add(body).faces(">Z").edges().fillet(2.2).val()
    body = body.cut(core_cuts(H / 2 - PH / 2))
    body = body.cut(screw_holes(BACK_Z - 0.1, FRONT - 0.6, 1.6))
    housing = place(body)
    # swan neck: sweep a rounded section along an arc from the base to the pod's lower back
    p0 = np.array([7.4, 0.0, BASE_TOP - 0.5])
    p2 = to_S((0, -H / 2 + 3.0, BACK_Z + 1.0))
    p1 = np.array([p0[0] - 1.0, 0.0, (p0[2] + p2[2]) / 2 + 2.0]) * 0.5 + (p0 + p2) / 2 * 0.5 + np.array([4.0, 0, 0])
    path = cq.Workplane("XZ").spline([(p0[0], p0[2]), (p1[0], p1[2]), (p2[0], p2[2])],
                                     tangents=[(0, 1), (-0.72, 0.69)], includeCurrent=False)
    prof = (cq.Workplane(cq.Plane(origin=tuple(p0), xDir=(1, 0, 0), normal=(0, 0, 1)))
            .rect(6.2, 12.0).val())
    neck = cq.Workplane(cq.Plane(origin=tuple(p0), xDir=(1, 0, 0), normal=(0, 0, 1))).rect(6.2, 12.0)
    neck = neck.sweep(path, transition="round").val()
    neck = cq.Workplane().add(neck).edges().fillet(1.2).val() if False else neck
    base = cq.Workplane("XY").workplane(offset=2.0).center(7.4, 0).rect(6.35, 16.4).extrude(BASE_TOP - 2.0) \
        .edges("|Z").fillet(2.0).val()
    part = housing.fuse(neck).fuse(base).fuse(foot())
    part = part.cut(head_clearance())
    part = part.cut(place(core_cuts(H / 2 - PH / 2)))
    cov = cq.Workplane().add(d2_outline(BACK_Z - COVER_T, COVER_T)).faces("<Z").edges().fillet(1.2).val()
    cov = place(cover(lambda z0, t: cq.Workplane().add(d2_outline(z0, t)).faces("<Z").edges().fillet(1.2).val()))
    return part.clean(), cov.clean()

# =====================================================================================
# Design 3  BLADE: a single thin fin from foot to a flat, sharp-edged housing
# =====================================================================================
def d3_outline(z0, t, W=PW + 10.0, H=PH + 4.4, r=2.0):
    return cq.Workplane("XY").workplane(offset=z0).rect(W, H).extrude(t).edges("|Z").fillet(r).val()

def design3():   # flat housing in the yoke (same fork as the other yoke designs)
    W, H = PW + 10.0, PH + 4.4
    return yoke_design(lambda z0, t: d3_outline(z0, t), W, H, "chamfer", 0.8)

# ---------------- board model for the viewer ---------------------------------
def board_model(fisheye=True):
    b = cq.Workplane("XY").workplane(offset=-BT).rect(BW, BH).extrude(BT).val()
    holder = cq.Workplane("XY").rect(15.0, 15.0).extrude(4.5).val()
    if fisheye:
        barrel = cq.Solid.makeCylinder(9.5, 9.0, cq.Vector(0, 0, 4.5))
        dome = cq.Solid.makeSphere(7.0, cq.Vector(0, 0, 11.0), angleDegrees1=0, angleDegrees2=90)
        lens = barrel.fuse(dome)
    else:
        lens = cq.Solid.makeCylinder(8.0, 7.5, cq.Vector(0, 0, 4.5))
    conn = cq.Workplane("XY").workplane(offset=-BT - 5.0).center(CONN_X, -BH / 2 + 1.0).rect(10.7, 6.0).extrude(5.0).val()
    plug = cq.Workplane("XY").workplane(offset=-BT - 4.6).center(CONN_X, -BH / 2 - 6.0).rect(10.0, 8.0).extrude(4.4).val()
    # cable: short drop below the plug, then a turn straight back out of the open back of the pocket
    drop = cq.Solid.makeCylinder(1.6, 3.0, cq.Vector(CONN_X, -BH / 2 - 10.0, -BT - 2.4), cq.Vector(0, -1, 0))
    back = cq.Solid.makeCylinder(1.6, 14.0, cq.Vector(CONN_X, -BH / 2 - 11.4, -BT - 2.4), cq.Vector(0, 0, -1))
    cable = drop.fuse(back)
    plug = plug.fuse(cable)
    return place(b), place(holder.fuse(lens)), place(conn.fuse(plug))


# =====================================================================================
# Yoke family: the gimbal fork + trapezoid stem, with different housings
# =====================================================================================
def shield_outline(z0, t, W=PW + 10.0, H=PH + 5.0, top_ch=3.0, bot_ch=8.0):
    """ElRobot-shield silhouette: small chamfers on top, long 45-degree cuts at the bottom."""
    pts = [(-W / 2 + bot_ch, -H / 2), (W / 2 - bot_ch, -H / 2), (W / 2, -H / 2 + bot_ch * 0.7),
           (W / 2, H / 2 - top_ch), (W / 2 - top_ch, H / 2), (-W / 2 + top_ch, H / 2),
           (-W / 2, H / 2 - top_ch), (-W / 2, -H / 2 + bot_ch * 0.7)]
    return cq.Workplane("XY").workplane(offset=z0).polyline(pts).close().extrude(t).val()

def yoke_design(outline_fn, W, H, face="chamfer", face_amt=1.6, disc_r=5.5):
    body = outline_fn(BACK_Z, FRONT - BACK_Z)
    wp = cq.Workplane().add(body).faces(">Z").edges()
    body = (wp.chamfer(face_amt) if face == "chamfer" else wp.fillet(face_amt)).val()
    cone = cq.Solid.makeCone(LENS_D / 2, LENS_D / 2 + 2.0, 1.21, cq.Vector(0, 0, FRONT - 1.2))
    body = body.cut(core_cuts(H / 2 - PH / 2)).cut(cone)
    body = body.cut(screw_holes(BACK_Z - 0.1, FRONT - 0.6, 1.6))
    housing = place(body)
    yb = -H / 2 - 1.0 - CB_H
    yoke = (cq.Workplane("XY").workplane(offset=BACK_Z).center(0, yb + CB_H / 2)
            .rect(W + 6.8, CB_H).extrude(-BACK_Z - 1.0).edges("|Z").fillet(1.5).val())
    for sx in (-1, 1):
        cheek = (cq.Workplane("XY").workplane(offset=BACK_Z).center(sx * (W / 2 + 1.7), yb / 2)
                 .rect(3.4, -yb).extrude(-BACK_Z - 1.0).val())
        disc = cq.Solid.makeCylinder(disc_r, 3.4, cq.Vector(sx * (W / 2), 0, BACK_Z + disc_r), cq.Vector(sx, 0, 0))
        yoke = yoke.fuse(cheek).fuse(disc)
    # cable-tie slots through the crossbar, one each side of the plug channel (strain relief for the cable)
    for xc in (CONN_X - CONN_W / 2 - 2.5, CONN_X + CONN_W / 2 + 2.5):
        slot = cq.Workplane("XY").workplane(offset=BACK_Z - 1).center(xc, yb + CB_H / 2).rect(2.0, 3.6).extrude(-BACK_Z + 1).val()
        yoke = yoke.cut(slot)
    # one trapezoid stem under the middle of the crossbar; the cable leaves the plug pocket out of the back
    top_c = to_S((0, yb, (BACK_Z - 1.0) / 2))
    stem = loft_rect([7.4, 0, BASE_TOP - 0.01], 6.3, 16.0, top_c, 4.4, 20.0, ([1, 0, 0], [0, 1, 0]), (N, U))
    part = housing.fuse(place(yoke)).fuse(stem).fuse(base_block()).fuse(foot())
    part = part.cut(head_clearance()).cut(place(core_cuts(H / 2 - PH / 2)))
    cov = place(cover(outline_fn))
    return part.clean(), cov.clean()

def design4():   # yoke + facet housing
    W, H = PW + 11.0, PH + 5.0
    return yoke_design(lambda z0, t: d1_outline(z0, t), W, H, "chamfer", 1.6)

def design5():   # yoke + pod housing
    W, H = PW + 11.0, PH + 6.0
    return yoke_design(lambda z0, t: d2_outline(z0, t), W, H, "fillet", 2.2)

def design6():   # yoke + ElRobot-shield housing
    W, H = PW + 10.0, PH + 5.0
    return yoke_design(lambda z0, t: shield_outline(z0, t), W, H, "chamfer", 1.8)
