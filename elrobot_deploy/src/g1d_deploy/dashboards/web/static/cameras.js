// The camera panes, shared by the live dashboard and the rollout viewer.
//
// Both pages answer the same question — which views does this rig have, and
// what was each one FOR — so they share this one file: a pane captioned one way
// live and another way on replay is a pane you cannot trust either time. It
// lives beside `gl.js` and `marks.js` (`dashboards/web/static/`).
//
// THE LAYOUT: scene views on top, wrist views below.
//
// The split is by WHERE A CAMERA IS, not by whether the policy reads it. The
// wrists are bolted to the arms and swing with them; the head and the exo look
// at the scene from a fixed place. That is the distinction a reader is actually
// using when they glance at the panel, and it does not move when a camera
// changes role: serve the exo to the policy or merely record it and it is a
// scene view either way, sitting beside the head. Grouping on `used` instead
// would put the exo in a different band depending on a config flag — the same
// picture rearranging itself for a reason that has nothing to do with what is
// in it. A wrist view is recognised by "wrist" in its name (`isWristView`).
//
// EVERY PANE IS LETTERBOXED, never cropped and never stretched. A view is worth
// looking at because of what is in it, and a pane that fills its box by
// throwing away the edges of the frame is a pane that can hide the thing you
// opened it to see. So each image keeps its captured aspect ratio and the space
// around it stays black.

/** `/api/meta` cameras, as objects, whatever shape the server sent.
 *
 * The server sends objects; bare NAMES are accepted too, because these
 * viewers serve their JS from disk and their Python from memory: a server left
 * running across an update hands the new script the old payload, and the
 * failure mode was a pane captioned "undefined camera", which names nothing. A
 * string means "an older server", which only ever described cameras the policy
 * read — hence `used: true`.
 */
export function normalizeCameras(cams) {
  return (cams || []).map((c) =>
    typeof c === "string"
      ? { name: c, slot: null, used: true, primary: false, binocular: false }
      : c);
}

/** Is this a view from an arm, rather than of the scene?
 *
 * Name first, slot second, because the name is what every rig has and the slot
 * only exists for a camera the policy reads — the head has none. Both carry the
 * word on the shipped rigs (`lwrist`/`rwrist`), and a policy's slot names
 * usually do too (e.g. `left_wrist_0_rgb`), so a rig that renames one still
 * matches on the other.
 */
function isWristView(cam) {
  return /wrist/i.test(cam.name) || /wrist/i.test(cam.slot || "");
}

/** The aspect ratio the pane should reserve, as `w / h`, or null if unknown.
 *
 * A binocular frame is TWO eyes side by side and only the left one is shown, so
 * the box to letterbox into is half the frame's width — reserving the whole
 * 1280x480 would leave the eye floating in a letterbox twice as wide as it
 * needs. An older record dumps no camera shapes at all, and null there means
 * "fall back to plain `object-fit: contain`", which is right for every camera
 * except a binocular one.
 */
function paneAspect(cam) {
  const h = Number(cam.height);
  const w = Number(cam.width);
  if (!(h > 0 && w > 0)) return null;
  return (cam.binocular ? w / 2 : w) / h;
}

/** What this view was for, as a caption a reader can act on.
 *
 * `live` only picks the tense: the dashboard is watching a run happen and the
 * viewer is reading one back. The claims are otherwise identical, and they are
 * the reason this is worth captioning at all — a camera the policy read and a
 * camera the run merely recorded look the same on screen, and mistaking the
 * second for the first misreads the rollout.
 */
function roleOf(cam, live) {
  if (!cam.used) {
    return {
      text: "recorded only",
      title: live
        ? "Recorded for review. The policy is not reading this view, and it " +
          "takes no part in choosing the anchor instant."
        : "Recorded for review. The policy never saw this view, and it took " +
          "no part in choosing the anchor instant.",
    };
  }
  if (cam.primary) {
    return {
      text: `${cam.slot || "used"} · anchor`,
      title: live
        ? "Feeding the policy, and its chosen frame IS t_anchor — the instant " +
          "the measured joints are matched to."
        : "Fed the policy, and its chosen frame IS t_anchor — the instant " +
          "every other stream in this record is matched to.",
    };
  }
  return {
    text: cam.slot || "used",
    title: live
      ? "Feeding the policy in this input slot."
      : "Fed the policy in this input slot.",
  };
}

/** A fisheye's lens square and circle, outlined over its pane, or null.
 *
 * The pane shows the WHOLE sensor frame (the stream is what the robot sent);
 * the policy sees only the square. Drawing it lets a person check at a glance
 * that the fit found the lens, and that this lens looks like the one the data
 * was collected with. The SVG's viewBox is the frame's own pixels and it is
 * letterboxed exactly as the image is (same box, `meet`), so the two coincide.
 */
function lensOverlay(cam) {
  const h = Number(cam.height);
  const w = Number(cam.width);
  if (!cam.crop || !(h > 0 && w > 0) || cam.binocular) return null;
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("class", "lens");
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
  const [x0, y0, x1, y1] = cam.crop.map(Number);
  const rect = document.createElementNS(NS, "rect");
  rect.setAttribute("x", x0);
  rect.setAttribute("y", y0);
  rect.setAttribute("width", x1 - x0);
  rect.setAttribute("height", y1 - y0);
  svg.append(rect);
  if (cam.lens_circle) {
    const [cx, cy, r] = cam.lens_circle.map(Number);
    const c = document.createElementNS(NS, "circle");
    c.setAttribute("cx", cx);
    c.setAttribute("cy", cy);
    c.setAttribute("r", r);
    svg.append(c);
  }
  const t = document.createElementNS(NS, "title");
  t.textContent = `the square the policy sees: [${cam.crop.map(Math.round).join(", ")}]`;
  svg.append(t);
  return svg;
}

function paneFor(cam, live) {
  const fig = document.createElement("figure");
  fig.className = "cam" + (cam.binocular ? " binocular" : "") +
                  (cam.used ? "" : " record-only");

  const who = document.createElement("span");
  who.className = "who";
  who.textContent = cam.name;

  const role = document.createElement("span");
  role.className = "role";
  const { text, title } = roleOf(cam, live);
  role.textContent = text;
  role.title = title;
  who.append(" ", role);

  // The camera frame this display frame resolved to. The viewer fills it; the
  // dashboard leaves it empty, because a live MJPEG stream carries no frame
  // number the page is told about.
  const idx = document.createElement("em");
  const cap = document.createElement("figcaption");
  cap.append(who, idx);

  const img = document.createElement("img");
  img.alt = `${cam.name} camera`;
  if (cam.binocular) {
    img.title = live
      ? "Binocular frame — the left eye is shown; the robot sends both."
      : "Binocular frame — the left eye is shown; both are on disk.";
  }

  // The shape the pane should SHOW, which for a binocular frame is half the
  // shape that arrives. The stylesheet fits a box of exactly this ratio inside
  // the pane and centres it, so "do not crop, do not stretch" is enforced by
  // the layout rather than left to whatever the source happens to be. Unset on
  // a record that does not state its camera shapes, where the plain
  // `object-fit: contain` fallback is already right.
  const aspect = paneAspect(cam);
  if (aspect) img.style.setProperty("--aspect", String(aspect));

  const shot = document.createElement("div");
  shot.className = "shot";
  shot.append(img);
  const lens = lensOverlay(cam);
  if (lens) shot.append(lens);

  fig.append(cap, shot);
  return { fig, img, idx, cap };
}

/** One pane per camera, in two stacked bands. Returns `{name: {img, idx, cap}}`.
 *
 * The bands are the scene views then the wrist views, and an empty one is not
 * rendered at all — so a record from before the head was declared fills the
 * pane with its two wrists exactly as it always did.
 */
export function buildCameraPanes(host, cams, { live = false } = {}) {
  host.replaceChildren();
  const out = {};
  for (const group of [cams.filter((c) => !isWristView(c)),
                       cams.filter((c) => isWristView(c))]) {
    if (!group.length) continue;
    const row = document.createElement("div");
    row.className = "cam-row";
    for (const cam of group) {
      const { fig, img, idx, cap } = paneFor(cam, live);
      row.append(fig);
      out[cam.name] = { img, idx, cap };
    }
    host.append(row);
  }
  return out;
}
