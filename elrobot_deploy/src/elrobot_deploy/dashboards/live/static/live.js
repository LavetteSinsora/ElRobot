/* The stage: the robot and the cameras.
 *
 * Two loops, deliberately independent:
 *
 *   1. The cameras are `<img>` elements pointed at an MJPEG stream. The browser
 *      drives them; there is no JavaScript in that path at all.
 *   2. The robot is polled. `/api/frame` returns one JSON header and the
 *      float32 arrays it maps, and the next poll is scheduled only once the last
 *      has been drawn — so a slow link lowers the frame rate instead of building
 *      a backlog of stale robots. The header carries the banner too, which is
 *      painted through the shell (`hooks.paintBanner`).
 *
 * The renderer is the shared `Viewer` (`dashboards/web/static/gl.js`), the one
 * rollout inspection draws with. A live
 * frame is a one-frame scene: `setTransforms(..., nFrames = 1)` and `setFrame(0)`.
 */

import { Viewer } from "/static/gl.js";
import { buildCameraPanes, normalizeCameras } from "/static/cameras.js";

const $ = (id) => document.getElementById(id);

/* How often the robot is re-read, when the link can keep up. The cameras
 * publish at 30 Hz; the operator is monitoring rather than measuring, and half
 * that is plenty for a picture a human is watching. Lower is also politer to a
 * tunnel the control POST has to share. */
const POLL_MS = 66;

/* Phases in which the overlays can exist at all. Outside a rollout the server
 * sends no overlay arrays, and this is only used to grey the toggles so the
 * absence reads as "nothing is commanding" rather than "the overlay is broken". */
const ACTIVE = 5;

let hooks = null;
let viewer = null;        // built on first activation, kept for the page's life
let meta = null;          // the static model description, fetched once
let panes = {};           // camera name -> {img, cap}

const say = (msg, bad = false) => hooks && hooks.say(msg, bad);
const paintBanner = (b) => hooks && hooks.paintBanner(b);

/* --- the model, once ----------------------------------------------------- */

async function loadModel() {
  const [m, sceneRes] = await Promise.all([
    fetch("/api/meta").then((r) => r.json()),
    fetch("/api/scene.bin"),
  ]);
  const layout = JSON.parse(sceneRes.headers.get("X-Layout"));
  const buf = await sceneRes.arrayBuffer();
  const vertices = new Float32Array(buf, layout.vertices[0], layout.vertices[1] / 4);
  const normals = new Float32Array(buf, layout.normals[0], layout.normals[1] / 4);
  const indices = new Uint32Array(buf, layout.indices[0], layout.indices[1] / 4);
  viewer.setGeometry(m, vertices, normals, indices);
  meta = m;
  panes = buildCameraPanes($("cameras"), normalizeCameras(m.cameras), { live: true });
}

/* The panes are `cameras.js`'s, and only the wiring is this page's: an `<img>`
 * pointed at an MJPEG stream, which the browser then drives on its own. There
 * is no JavaScript in the camera path at all and no decode on either side — P2
 * keeps the bytes the robot sent and the server passes them through. A stream
 * that drops takes its own `<img>` down and leaves the rest of the page, and
 * the robot, alone. */
function streams() {
  for (const [name, { img, cap }] of Object.entries(panes)) {
    cap.classList.remove("bad");
    img.onerror = () => cap.classList.add("bad");
    img.src = `/api/stream/${encodeURIComponent(name)}`;
  }
}

/* --- one live frame ------------------------------------------------------ */

function view(buf, base, span) {
  /* `span` is [offset, nbytes] from the header, relative to the PAYLOAD — so
   * `base` (where the payload starts) has to be added. A missing span means the
   * server sent no such overlay for this frame, which is a fact about the robot
   * rather than a gap to fill with zeros.
   *
   * `base` is 4-aligned because the server pads the header to a 4-byte
   * boundary, and `_pack` aligns each section within the payload. Both are
   * required: `Float32Array(buffer, byteOffset, ...)` throws outright on a
   * byteOffset that is not a multiple of four. */
  return span ? new Float32Array(buf, base + span[0], span[1] / 4) : null;
}

function paintFrame(head, buf, base) {
  const b = head.banner;
  paintBanner(b);
  if (!head.ready) {
    say("waiting for the state stream — P1 has not published yet.");
    return;
  }

  const L = head.layout;
  const transforms = view(buf, base, L.transforms);
  const tcp = view(buf, base, L.tcp);
  viewer.setTransforms(transforms, tcp, 1, meta.n_geoms);

  /* One byte of validity per frame, and this frame is valid exactly when the
   * server sent the arrays. `Viewer` breaks its trails on these, which for a
   * one-frame scene simply means "draw it or do not". */
  const on = new Uint8Array([1]);
  const off = new Uint8Array([0]);

  if (head.solved && L.solved) {
    viewer.setSolved(view(buf, base, L.solved), view(buf, base, L.solved_tcp),
                     on, meta.solved_geoms);
  } else {
    viewer.setSolved(new Float32Array(meta.solved_geoms.length * 12),
                     new Float32Array(6), off, meta.solved_geoms);
  }

  const ghostLabel = $("chk-ghost").parentElement;
  ghostLabel.dataset.title ??= ghostLabel.title;
  ghostLabel.title = head.ghost_absent
    ? `Nothing to draw: ${head.ghost_absent}. The commanded joints are the command.`
    : ghostLabel.dataset.title;
  if (head.ghost && L.ghost) {
    viewer.setGhost(view(buf, base, L.ghost), view(buf, base, L.ghost_tcp), on,
                    meta.ghost_geoms);
  } else {
    viewer.setGhost(new Float32Array(meta.ghost_geoms.length * 12),
                    new Float32Array(6), off, meta.ghost_geoms);
  }

  viewer.setFrame(0);

  const commanding = b.session === ACTIVE;
  $("chk-solved").parentElement.classList.toggle("inactive", !commanding);
  /* Joint-mode actions carry no Cartesian target (pose is NaN by contract), so
   * the commanded-pose overlay has nothing to draw even while commanding. */
  $("chk-ghost").parentElement.classList.toggle(
    "inactive", !commanding || !!head.ghost_absent);
  if (commanding && head.solved) {
    const err = head.solved.err;
    say(head.ghost_absent
      ? `tick ${head.solved.k} · joint targets (no Cartesian target)`
      : err === null || err === undefined
        ? `tick ${head.solved.k}`
        : `tick ${head.solved.k} · IK residual ${(err * 1000).toFixed(1)} mm`);
  } else if (commanding) {
    say(`tick ${b.k_exec} · no target in the ring yet`);
  } else {
    say("measured only — nothing is commanding the arm.");
  }
}

async function poll() {
  if (hooks.stopped()) return;
  try {
    const res = await fetch("/api/frame");
    const buf = await res.arrayBuffer();
    /* uint32 length, then that many bytes of JSON, then the float32 sections.
     * One response so the header and the arrays it describes are the same
     * instant — two requests would pair a layout with a robot that has moved. */
    const n = new DataView(buf).getUint32(0, true);
    const head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, n)));
    /* The header is padded to a 4-byte boundary before the payload starts, so
     * the typed-array views below land on legal offsets. Same rule the server
     * applies; computed rather than sent, because it follows from `n`. */
    const base = (4 + n + 3) & ~3;
    paintFrame(head, buf, base);
  } catch (e) {
    say(`lost the robot: ${e.message}`, true);
  }
  /* Scheduled AFTER the draw, never on an interval: an interval would keep
   * firing into a link that cannot keep up and the queue would grow. */
  setTimeout(poll, POLL_MS);
}

/* --- the view switches --------------------------------------------------- */

const PARTS = ["body", "gripper", "wheel", "column"];
function applyParts() {
  viewer.setVisibleParts($("chk-base").checked ? PARTS : ["body", "gripper"]);
}

function wireSwitches() {
  /* The flag names are the Viewer's own — each gates its source's geometry,
   * TCP marker and trail together, which is why there is nothing to keep in
   * step here. */
  for (const [id, fn] of [
    ["chk-measured", (v) => { viewer.measured = v; }],
    ["chk-ghost", (v) => { viewer.ghosts = v; }],
    ["chk-solved", (v) => { viewer.solvedOn = v; }],
    ["chk-tcp", (v) => { viewer.tcpMarkers = v; }],
  ]) {
    const el = $(id);
    el.onchange = () => { fn(el.checked); viewer.draw(); };
    fn(el.checked);
  }
  $("chk-base").onchange = applyParts;
  applyParts();
}

/* --- start ------------------------------------------------------------- */

export async function start(shellHooks) {
  hooks = shellHooks;
  viewer = new Viewer($("gl"));
  wireSwitches();
  try {
    await loadModel();
  } catch (e) {
    say(`could not load the model: ${e.message}`, true);
    return;
  }
  streams();
  poll();
}
