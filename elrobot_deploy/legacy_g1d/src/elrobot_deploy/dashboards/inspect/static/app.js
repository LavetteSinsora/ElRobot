// The dashboard: a recording dropdown, a Load button, and one playhead that
// every pane follows.
//
// The single source of truth is `frame` — an index into the master timeline the
// server built. Cameras and the 3D view are both pure functions of it, which is
// what makes "pause and everything freezes on the same instant" true by
// construction rather than by synchronisation.
//
// Camera frames are fetched as ordinary images and kept in an LRU, with a
// look-ahead prefetch. A wrist frame is ~70 KB (1280x720 fisheye) and the
// browser runs ~6 requests to one host at a time, so the fetching is bounded:
// one request per camera PANE at a time, always for the newest playhead
// (latest wins — a fast scrub costs one round trip per camera per step, not
// one per slider position); a frame already on its way is never asked for
// twice; and the prefetch waits until the playhead rests (or is playing) and
// stays within a small in-flight budget, so it never queues ahead of the frame
// on screen. The cache makes scrubbing back over covered ground instant.

import { Viewer } from "/static/gl.js";
import { Marks } from "/static/marks.js";
import { MetricStrip } from "/static/metric.js";
import { buildCameraPanes, normalizeCameras } from "/static/cameras.js";

const $ = (id) => document.getElementById(id);

const el = {
  rec: $("sel-rec"), recRoot: $("rec-root"), load: $("btn-load"),
  status: $("status"), play: $("btn-play"), reset: $("btn-reset"), seek: $("seek"),
  tNow: $("t-now"), tEnd: $("t-end"), fNow: $("f-now"), splash: $("splash"),
  cams: $("cams"),
  chip: $("rec-chip"), headline: $("rec-headline"), subline: $("rec-subline"),
  card: $("rec-card"),
};

// Per camera name -> the <img> and the frame-number <em> in its pane. Rebuilt
// by `buildCameraPanes` on every load, because the camera set belongs to the
// record and not to the page.
let camEl = {};

const state = {
  // `/api/tree`: {root, single, groups: [{group, records: [...]}]}.
  tree: { root: "", single: false, groups: [] },
  loaded: null,       // the server's describe() payload
  frame: 0,
  playing: false,
  cameras: [],
  cache: new Map(),   // "cam:frame" -> {url, idx}, oldest first (LRU)
  inflight: new Map(), // "cam:frame" -> Promise<entry|null>, not yet cached
  cacheGen: 0,        // bumped by dropCache: a fetch from before it is dropped
  lastRAF: 0,
  // Playback position in seconds from the window start. The grid is not
  // uniform, so time is the thing that advances and the frame is looked up.
  clock: 0,
};

// Entries are shared across cameras, so this is a per-camera depth divided by
// however many the record holds: 1350 keeps ~15 s each at three cameras, which
// is what two cameras had at 900. Scrubbing back further re-fetches from a
// local server, so the cost of being wrong here is latency, not correctness.
const CACHE_MAX = 1350;
const PREFETCH = 12;
//: At most this many prefetch requests in flight, so a prefetch burst never
//: sits in the browser's per-host queue ahead of the frame being shown.
const PREFETCH_INFLIGHT = 6;
//: A paused playhead that has not moved for this long is "resting": prefetch.
const PREFETCH_IDLE_MS = 120;

const viewer = new Viewer($("gl"));

// --- extension hooks --------------------------------------------------------
//
// A MODE draws on top of the rollout this file loaded (`modes/*.py`,
// `mode_<name>.js`): solved checkpoints as ghosts, the IK bench's arms, an
// executor's joint traces. Rather than fork 800 lines of controller, this
// exports the viewer and two callback lists — `onLoaded(describe)` once a
// rollout is in, `onFrame(f)` on every playhead move — plus what a mode needs
// to read: the loaded description, the current frame, the whole-body arrays
// the transform buffer carried, and `dependsOn` for greying a switch out. A
// mode removes its callbacks when it is switched off (`off`).
export const hooks = {
  onLoaded: [], onFrame: [], viewer, goto: (f) => goto(f),
  state: () => state.loaded, frame: () => state.frame,
  // Read the loaded recording again, for a mode that has just changed what is
  // on disk beside it. Fires `onLoaded` like any other load.
  reload: () => loadCurrent(state.loaded ? state.loaded.key : null),
  wbik: null,                 // {tiers, err, offErr, valid, meta} once loaded
  // "eef" | "joint" | "mixed", once loaded: a mode that needs a commanded pose
  // (the IK bench) reads it to say why it has nothing to do.
  actionMode: "eef",
  dependsOn: (id, on) => dependsOn(id, on),
  off: (list, fn) => { const i = list.indexOf(fn); if (i >= 0) list.splice(i, 1); },
  css: (c) => `rgb(${c.map((v) => Math.round(v * 255)).join(", ")})`,
  // The strip under the scrubber (`metric.js`): a mode's per-tick metric over
  // the whole rollout. `show({title, unit, help, series, empty, tools})`,
  // `hide()`. Its playhead is the main one; clicking it seeks the view.
  strip: null,
};

// The replan marks own the strip above the scrubber and the zoom band, and hand
// back frames. They never touch `state` — every seek goes through `goto`, so
// clicking a mark and dragging the slider are the same act as far as the rest
// of the app is concerned.
const marks = new Marks({
  strip: $("marks"),
  band: $("band"),
  tip: $("mark-tip"),
  onSeekFrame: (frame) => { pause(); goto(frame); },
  onSeekTime: (t) => {
    if (!state.loaded) return;
    pause();
    goto(frameAt(t));
  },
});

const metric = new MetricStrip({
  row: $("metric-row"),
  canvas: $("metric"),
  title: $("metric-title"),
  tools: $("metric-tools"),
  legend: $("metric-legend"),
  tip: $("metric-tip"),
  axis: $("marks"),              // the axis it shares, pixel for pixel
  onSeekTime: (t) => {
    if (!state.loaded) return;
    pause();
    goto(frameAt(t));
  },
});
hooks.strip = {
  show: (spec) => metric.show(spec),
  hide: () => metric.hide(),
  clear: (msg) => metric.clear(msg),
};

// ------------------------------------------------------------- time <-> frame
//
// The grid is NOT uniform. It lands on every commanded deadline and fills the
// gaps at roughly `fps`, so the tick line slipping ~158 ms per replan (sync
// re-stamps the epoch once per chunk) shows up as frames that are 31.8 to
// 34.0 ms apart rather than exactly 33.3. `timeline.t_rel` is therefore the
// only way to convert, and `fps` is the filler rate and nothing else —
// `frame / fps` was right when the grid was uniform and is wrong now by a whole
// frame near the first replan and by more after every one that follows.

/** Seconds from the window start, for a display frame. */
function timeAt(frame) {
  const tr = state.loaded.timeline.t_rel;
  return tr[Math.max(0, Math.min(tr.length - 1, frame))];
}

/** The frame nearest an instant — binary search, the inverse of `timeAt`. */
function frameAt(t) {
  const tr = state.loaded.timeline.t_rel;
  let lo = 0, hi = tr.length - 1;
  if (t <= tr[0]) return 0;
  if (t >= tr[hi]) return hi;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (tr[mid] <= t) lo = mid; else hi = mid;
  }
  return (t - tr[lo]) <= (tr[hi] - t) ? lo : hi;
}

// ------------------------------------------------------------------- status

function status(html) { el.status.innerHTML = html; }

function statusForLoaded(d) {
  // One short line: what the rollout was, and only the warnings that change how
  // the screen reads. The record card (the chip) holds everything else.
  const s = d.summary, t = d.timeline, ev = d.events;
  const kv = (k, v) => `<span class="k">${k}</span> <span class="v">${v}</span>`;
  const warn = (w) => `<span class="warn">${w}</span>`;
  const bits = [];
  if (s.robot) bits.push(kv("robot", esc(s.robot)));
  if (s.action_mode) bits.push(kv("actions", s.action_mode));
  bits.push(kv("dur", `${t.duration_s.toFixed(1)} s`) +
            (t.window === "active" ? `<span class="dim"> from Start</span>` : ""));
  if (ev.n) {
    bits.push(kv("replans", ev.n));
    bits.push(kv("infer", `${ev.latency_ms.med} ms`));
  }
  // A stall folds late telemetry into the arm's apparent motion; say so.
  if (t.state_clock && t.state_clock.stalls_50ms) {
    bits.push(warn(`${t.state_clock.stalls_50ms} state samples &gt;50 ms late`));
  }
  if (s.warm === false) bits.push(warn("not warm at start"));
  if (!s.model_from_record) bits.push(warn("robot assumed (default profile)"));
  if (s.tcp_source === "default") bits.push(warn("TCP assumed"));
  if (s.command_source === "chunks") bits.push(warn("commanded pose reconstructed"));
  // Each item unbroken: when the bar is narrow the line breaks BETWEEN items.
  status(bits.map((b) => `<span class="bit">${b}</span>`).join(" &middot; "));
}

// ------------------------------------------------------- the selected record
//
// Which run a recording IS — checkpoint above all, then the settings that
// rollout actually ran with (`rollouts[0].active_config`: settings can change
// between the rollouts of one launch) — read from its manifest by the server
// and shipped with the listing, so this panel is filled in the moment a
// recording is picked and not when it is loaded.

const esc = (s) => String(s ?? "").replace(/[&<>"]/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/** `2026-09-03T11:32:56` -> `2026-09-03 11:32:56`. Local time on the machine
 *  that recorded; handing it to `Date` would re-interpret it as UTC. */
const stamped = (s) => String(s || "").replace("T", " ").split(".")[0];

function allRecords() {
  return state.tree.groups.flatMap((g) => g.records);
}

function currentRec() {
  return allRecords().find((r) => r.key === el.rec.value) || null;
}

/** The last component of a dotted class path, for the chip. */
const shortClass = (c) => String(c || "").split(".").pop();

/** The settings bits every row of the chip and the card share. */
function settingsBits(d) {
  const bits = [];
  if (d.sync !== null && d.sync !== undefined) bits.push(d.sync ? "sync" : "async");
  if (d.P) bits.push(`P${d.P}`);
  if (d.H) bits.push(`H${d.H}`);
  if (d.control_hz) bits.push(`${d.control_hz} Hz`);
  if (d.action_mode === "joint") bits.push(`<b>joint actions</b>`);
  else if (d.ik) bits.push(`<b>ik ${esc(d.ik)}</b>`);
  if (d.sample_steps) bits.push(`${d.sample_steps} steps`);
  return bits;
}

/** The two lines on the chip. */
function chipText(rec) {
  const d = (rec && rec.details) || {};
  if (!rec) return ["&mdash;", ""];
  if (!rec.loadable) return [`<span class="dim">unreadable</span>`, esc(rec.name)];

  let head;
  if (d.checkpoint_step) {
    head = `<b>${esc(d.checkpoint_run || "checkpoint")}</b>` +
           `<span class="dim"> &middot; step </span><b>${esc(d.checkpoint_step)}</b>`;
  } else if (d.checkpoint_run) {
    head = `<b>${esc(d.checkpoint_run)}</b>`;
  } else {
    head = `<span class="dim">no checkpoint recorded</span>`;
  }
  if (d.policy_class) head += `<span class="dim"> &middot; ${esc(shortClass(d.policy_class))}</span>`;

  const bits = settingsBits(d);
  bits.push(`${rec.size_mb} MB`);
  // The facts that make a rollout mean something different from its
  // neighbours, on the chip rather than only in the card.
  if (d.warm === false) bits.push(`<span class="warn">cold</span>`);
  if (d.dirty || d.policy_dirty) bits.push(`<span class="warn">+dirty</span>`);
  return [head, bits.join(" &middot; ")];
}

/** Everything the manifest says, for the card. */
function cardRows(rec) {
  const d = (rec && rec.details) || {};
  const rows = [];
  const add = (k, v, cls) => rows.push(
    `<dt>${k}</dt><dd${cls ? ` class="${cls}"` : ""}>${v}</dd>`);
  const rule = () => rows.push(`<div class="rule"></div>`);
  const sha = (h, dirty) => h
    ? `${esc(h)}${dirty ? ` <span class="warn">+dirty</span>` : ""}`
    : `<span class="dim">not recorded</span>`;

  add("recording", esc(rec.key));
  if (d.created) add("recorded", esc(stamped(d.started || d.created)));
  if (!rec.loadable) {
    add("state", `<span class="warn">${esc(rec.reason)}</span>`);
    return rows.join("");
  }
  if (d.prompt !== null && d.prompt !== undefined) add("prompt", esc(d.prompt));
  if (d.generation !== null && d.generation !== undefined) {
    add("rollout", `generation ${esc(d.generation)}` +
        (d.rollout_index ? `<span class="dim"> &middot; #${esc(d.rollout_index)} of its launch</span>` : ""));
  }
  rule();
  add("policy", d.policy_class ? `<b>${esc(d.policy_class)}</b>` : `<span class="dim">not recorded</span>`);
  if (d.policy_entry) add("entry", esc(d.policy_entry), "path");
  add("checkpoint", esc(d.checkpoint) || `<span class="dim">none</span>`, "path");
  if (d.checkpoint_step) add("step", `<b>${esc(d.checkpoint_step)}</b>`);
  if (d.anchor_camera) add("anchor", esc(d.anchor_camera) +
      (d.uses_cameras && d.uses_cameras.length ? `<span class="dim"> &middot; uses ${esc(d.uses_cameras.join(", "))}</span>` : ""));
  rule();
  if (d.sync !== null && d.sync !== undefined) {
    add("mode", d.sync
      ? `sync <span class="dim">&mdash; the tick line waits for the chunk</span>`
      : `async <span class="dim">&mdash; the tick line free-runs</span>`);
  }
  // P and H in ticks AND in seconds: quoted in ticks, felt in seconds.
  const dt = d.control_hz ? 1 / d.control_hz : 0;
  if (d.P) add("replan", `every ${d.P} ticks` +
              (dt ? `<span class="dim"> &middot; ${(d.P * dt).toFixed(2)} s</span>` : ""));
  if (d.H) add("horizon", `${d.H} actions` +
              (dt ? `<span class="dim"> &middot; ${(d.H * dt).toFixed(2)} s</span>` : ""));
  if (d.control_hz) add("control", `${d.control_hz} Hz`);
  if (d.sample_steps) add("sample steps", esc(d.sample_steps));
  if (d.blend) add("blend", esc(d.blend));
  if (d.action_mode === "joint") add("ik", "not used (joint actions)");
  else if (d.ik) add("ik", `<b>${esc(d.ik)}</b>`);
  if (d.servo) {
    const sv = d.servo;
    add("servo", `filter ${esc(sv.filter_hz)} Hz${sv.filter_lead ? " + lead" : ""}` +
        ` &middot; dq ff &times;${esc(sv.velocity_ff)} &middot; torque ff &times;${esc(sv.torque_ff)}` +
        ` &middot; arm kp &times;${esc(sv.arm_gain_scale)}`);
  }
  if (d.warm !== null && d.warm !== undefined) {
    add("warm", d.warm ? "yes"
      : `<span class="warn">no</span><span class="dim"> &mdash; the first chunk paid the compile</span>`);
  }
  rule();
  add("robot", esc(d.robot) || `<span class="dim">not recorded (drawn as the default)</span>`);
  if (d.robot_files && d.robot_files.length) add("robot files", esc(d.robot_files.join(" < ")), "path");
  add("infra", sha(d.infra_sha || d.sha, d.infra_sha ? d.infra_dirty : d.dirty) +
      (d.infra_version ? `<span class="dim"> &middot; elrobot_deploy ${esc(d.infra_version)}</span>` : ""));
  add("policy code", sha(d.policy_sha, d.policy_dirty));
  add("on disk", `${rec.size_mb} MB`);
  if (rec.dir) add("directory", esc(rec.dir), "path");
  return rows.join("");
}

function renderRecInfo() {
  const rec = currentRec();
  const [head, sub] = chipText(rec);
  el.headline.innerHTML = head;
  el.subline.innerHTML = sub;
  el.chip.disabled = !rec;
  el.chip.title = rec
    ? "What this recording was run with. Click for the full manifest reading."
    : "";
  el.card.innerHTML = rec ? `<dl>${cardRows(rec)}</dl>` : "";
  if (!rec) showCard(false);
}

function showCard(on) {
  el.card.hidden = !on;
  el.chip.setAttribute("aria-expanded", String(!!on));
}

el.chip.addEventListener("click", (e) => {
  e.stopPropagation();
  showCard(el.card.hidden);
});
// Anywhere else dismisses it, including the render pane underneath — the card
// overlaps the stage, and having to find the chip again to close it would be a
// small annoyance on every use.
document.addEventListener("click", (e) => {
  if (!el.card.hidden && !el.card.contains(e.target)) showCard(false);
});

// ---------------------------------------------------------------- dropdowns

/** The one dropdown: an <optgroup> per relative directory, newest first. */
function fillRecordings() {
  const sel = el.rec;
  sel.innerHTML = "";
  const add = (parent, r) => {
    const o = document.createElement("option");
    const d = r.details || {};
    o.value = r.key;
    // The IK and the step are what tell two rollouts of one run apart.
    const tag = d.action_mode === "joint" ? "joint" : d.ik;
    o.textContent = `${r.label}   ${tag ? `[${tag}]` : ""}` +
      `   ${d.checkpoint_step ? "step " + d.checkpoint_step : ""}   ${r.size_mb} MB` +
      `${r.loadable ? "" : "   (unreadable)"}`;
    if (!r.loadable) o.disabled = true;
    parent.appendChild(o);
  };
  for (const g of state.tree.groups) {
    if (!g.group) { g.records.forEach((r) => add(sel, r)); continue; }
    const og = document.createElement("optgroup");
    og.label = g.group;
    g.records.forEach((r) => add(og, r));
    sel.appendChild(og);
  }
  const all = allRecords();
  sel.disabled = all.length === 0;
  const first = all.find((r) => r.loadable);
  if (first) sel.value = first.key;
  updateLoadButton();
}

function updateLoadButton() {
  const r = currentRec();
  el.load.disabled = !(r && r.loadable);
  if (r && !r.loadable) status(`<span class="warn">${esc(r.reason)}</span>`);
  // The chip follows the selection, not the load: it is what you read to decide
  // whether to press Load. A card left open over a different recording's
  // details would be worse than no card, so a change closes it.
  showCard(false);
  renderRecInfo();
}

async function loadTree() {
  const tree = await (await fetch("/api/tree")).json();
  state.tree = tree;
  el.recRoot.textContent = "recording";
  el.recRoot.title = tree.root;
  fillRecordings();
  if (!allRecords().length) {
    status(`<span class="warn">no record (a directory with P9's manifest.json) under ${esc(tree.root)}</span>`);
  }
}

el.rec.addEventListener("change", updateLoadButton);

// --------------------------------------------------------------------- load

/** Load the recording the dropdown names, and rebuild everything from it.
 *
 *  A function rather than only a click handler because a MODE can need the
 *  rollout read again: the IK mode writes new methods into the recording's
 *  sidecar, and those live in the transform buffer, which is built once per
 *  load. `hooks.reload` is this, and going through the same path as the button
 *  is the point — there is one way a rollout reaches the screen. */
async function loadCurrent(key) {
  // `key` names a recording explicitly — a reload asks for the one that is
  // LOADED, which is not necessarily the one the dropdown now names: changing
  // the selection and not pressing Load is how you browse, and a mode's reload
  // must not turn that into a load.
  key = key || el.rec.value;
  pause();
  el.load.disabled = true;
  el.load.classList.add("busy");
  el.load.textContent = "Loading";
  status(`<span class="k">reading</span> <span class="v">${esc(key)}</span>`);

  try {
    const res = await fetch("/api/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        key,
        clock: "device",
      }),
    });
    const d = await res.json();
    if (!res.ok) throw new Error(d.error || res.statusText);

    status(`<span class="k">building geometry</span>`);
    const [sceneBuf, xformBuf] = await Promise.all([
      fetch("/api/scene.bin").then((r) => r.arrayBuffer()),
      fetch("/api/transforms.bin").then((r) => r.arrayBuffer()),
    ]);

    const L = d.layout;
    const vertices = new Float32Array(sceneBuf, L.scene_bin.vertices[0],
                                      L.scene_bin.vertices[1] / 4);
    const normals = new Float32Array(sceneBuf, L.scene_bin.normals[0],
                                     L.scene_bin.normals[1] / 4);
    const indices = new Uint32Array(sceneBuf, L.scene_bin.indices[0],
                                    L.scene_bin.indices[1] / 4);
    const T = L.transforms_bin;
    const transforms = new Float32Array(xformBuf, T.transforms[0], T.transforms[1] / 4);
    const tcp = new Float32Array(xformBuf, T.tcp[0], T.tcp[1] / 4);
    const ghost = new Float32Array(xformBuf, T.ghost[0], T.ghost[1] / 4);
    const ghostTcp = new Float32Array(xformBuf, T.ghost_tcp[0], T.ghost_tcp[1] / 4);
    const ghostValid = new Uint8Array(xformBuf, T.ghost_valid[0], T.ghost_valid[1]);
    const solved = new Float32Array(xformBuf, T.solved[0], T.solved[1] / 4);
    const solvedTcp = new Float32Array(xformBuf, T.solved_tcp[0], T.solved_tcp[1] / 4);
    const solvedValid = new Uint8Array(xformBuf, T.solved_valid[0], T.solved_valid[1]);
    // The same overlay on the trunk the IK was pinned at. Both spans are zero
    // bytes long on a record that predates `waist_ik`, which makes these empty
    // views rather than a missing key — one shape for every record, and the
    // emptiness is what the switch is disabled on.
    const pinned = new Float32Array(xformBuf, T.solved_pinned[0],
                                    T.solved_pinned[1] / 4);
    const pinnedTcp = new Float32Array(xformBuf, T.solved_pinned_tcp[0],
                                       T.solved_pinned_tcp[1] / 4);

    viewer.setGeometry(d.scene, vertices, normals, indices);
    viewer.setTransforms(transforms, tcp, d.timeline.n_frames, d.scene.n_geoms);
    viewer.setGhost(ghost, ghostTcp, ghostValid, d.scene.ghost_geoms);
    viewer.setSolved(solved, solvedTcp, solvedValid, d.scene.solved_geoms,
                     pinned, pinnedTcp);
    // Re-driven from the CHECKBOX, not from whatever the viewer was left
    // holding. Loading a record with no pinned trunk forces the viewer's own
    // flag false, and without this a modern record loaded after a legacy one
    // would come up on the commanded waist under a switch that says otherwise.
    // The whole-body overlays. Sections are named by tier, so a recording
    // solved for a subset lines up by name rather than by position, and one
    // solved for none simply has no sections and no switches.
    const wbTiers = d.scene.wbik_tiers || [];
    const wbik = {};
    let wbValid = null, wbOffErr = null;
    const wbErr = {};
    if (wbTiers.length) {
      wbValid = new Uint8Array(xformBuf, T.wbik_valid[0], T.wbik_valid[1]);
      wbOffErr = new Float32Array(xformBuf, T.wbik_off_err[0],
                                  T.wbik_off_err[1] / 4);
      for (const tier of wbTiers) {
        wbik[tier] = [
          new Float32Array(xformBuf, T[`wbik_${tier}`][0],
                           T[`wbik_${tier}`][1] / 4),
          new Float32Array(xformBuf, T[`wbik_${tier}_tcp`][0],
                           T[`wbik_${tier}_tcp`][1] / 4),
        ];
        wbErr[tier] = new Float32Array(xformBuf, T[`wbik_${tier}_err`][0],
                                       T[`wbik_${tier}_err`][1] / 4);
      }
    }
    viewer.setWbik(wbik, d.scene.wbik_geoms || [], wbValid);
    for (const tier of wbTiers) viewer.setWbikOn(tier, false);
    // What the IK mode builds its switches and readout from. The residuals
    // are held here in METRES at the virtual TCP rather than read back out of
    // the viewer: the readout needs them on every frame, including the ones
    // where no tier is switched on, and the viewer only keeps what it draws.
    hooks.wbik = { tiers: wbTiers, err: wbErr, offErr: wbOffErr, valid: wbValid,
                   meta: d.scene.wbik_meta || {} };

    // The commanded joints always stand on the trunk the IK solved in when
    // the record has it (`waist_ik`); the viewer falls back to the commanded
    // waist on a record that predates it.
    viewer.setSolvedWaist(true);
    applyActionMode(d);

    dropCache();
    state.loaded = d;
    state.cameras = normalizeCameras(d.cameras);
    camEl = buildCameraPanes(el.cams, state.cameras);
    el.seek.max = String(d.timeline.n_frames - 1);
    el.seek.value = "0";
    el.seek.disabled = false;
    el.play.disabled = false;
    el.reset.disabled = false;
    el.tEnd.textContent = d.timeline.duration_s.toFixed(2);
    marks.setData(d.events, d.timeline);
    // Another recording: the mode's strip is stale until it redraws it.
    metric.clear("loading…");
    metric.setDuration(d.timeline.duration_s);
    el.splash.classList.add("hidden");

    await goto(0);
    statusForLoaded(d);
    for (const fn of hooks.onLoaded) await fn(d);
    // `?on=wb-arm_pull_home,wb-waist` lights a mode's switches, which exist
    // only once the recording is loaded and the mode has built its panel —
    // the counterpart of `?off=` for the sources that always exist, and the
    // only way a link can carry a comparison.
    applyOn(new URLSearchParams(location.search));
  } catch (err) {
    status(`<span class="err">${err.message}</span>`);
  } finally {
    el.load.classList.remove("busy");
    el.load.textContent = "Load";
    updateLoadButton();
  }
}

// Wrapped, not passed: the handler's argument is a MouseEvent, and `loadCurrent`
// takes a recording key.
el.load.addEventListener("click", () => loadCurrent());

// -------------------------------------------------------------- frame cache

function dropCache() {
  for (const e of state.cache.values()) URL.revokeObjectURL(e.url);
  state.cache.clear();
  // Requests still out belong to the old recording: they finish, but land
  // nowhere (`frameURL` checks the generation), and are not waited on.
  state.inflight.clear();
  state.cacheGen += 1;
}

function trimCache() {
  while (state.cache.size > CACHE_MAX) {
    const k = state.cache.keys().next().value;
    URL.revokeObjectURL(state.cache.get(k).url);
    state.cache.delete(k);
  }
}

/** The cached entry for (cam, frame), fetching it once if need be. Concurrent
 *  callers for the same key share one request. */
function frameURL(cam, frame) {
  const key = `${cam}:${frame}`;
  const hit = state.cache.get(key);
  if (hit) {
    // Refresh its recency: a Map iterates oldest-first, so the LRU trims the
    // entries least recently LOOKED AT, not least recently fetched.
    state.cache.delete(key);
    state.cache.set(key, hit);
    return Promise.resolve(hit);
  }
  const pending = state.inflight.get(key);
  if (pending) return pending;
  const gen = state.cacheGen;
  const p = (async () => {
    try {
      const res = await fetch(`/api/img/${cam}/${frame}`);
      if (!res.ok) return null;
      // `X-Frame-Index` is which CAMERA frame this display frame resolved to.
      // The cameras land on different frame numbers at the same instant — a
      // real property of the record — and this header is the only place that
      // says which.
      const blob = await res.blob();
      if (gen !== state.cacheGen) return null;      // another recording since
      const entry = { url: URL.createObjectURL(blob),
                      idx: res.headers.get("X-Frame-Index") };
      state.cache.set(key, entry);
      trimCache();
      return entry;
    } catch {
      return null;
    } finally {
      if (gen === state.cacheGen) state.inflight.delete(key);
    }
  })();
  state.inflight.set(key, p);
  return p;
}

function prefetch(from) {
  if (!state.loaded) return;
  const n = state.loaded.timeline.n_frames;
  for (let i = 1; i <= PREFETCH; i++) {
    const f = from + i;
    if (f >= n) break;
    for (const cam of state.cameras) {
      if (state.inflight.size >= PREFETCH_INFLIGHT) return;
      const key = `${cam.name}:${f}`;
      if (!state.cache.has(key) && !state.inflight.has(key)) frameURL(cam.name, f);
    }
  }
}

// One loop per camera pane: while it runs, a newer playhead is picked up when
// the current image arrives rather than queued behind it.
const paneBusy = new Map();   // camera name -> Promise of its running loop

function paint(cam, got) {
  const dst = camEl[cam];
  if (!dst) return;
  dst.img.src = got.url;
  dst.idx.textContent = got.idx ? `#${got.idx}` : "";
}

/** Bring camera `cam`'s pane to the current playhead. Returns when the pane
 *  shows `state.frame` (or the frame could not be had). */
function showCamera(cam) {
  const running = paneBusy.get(cam);
  if (running) return running;              // it will pick up state.frame
  const loop = (async () => {
    try {
      let f;
      do {
        f = state.frame;
        const got = await frameURL(cam, f);
        // Painted even if the playhead has moved on meanwhile: this loop's
        // arrivals are in request order, so each is newer than what the pane
        // shows, and a pane that waited for an exact match would stay frozen
        // for as long as a drag outpaces one round trip. The loop then goes
        // round for the current frame, so it always ends on it.
        if (got) paint(cam, got);
      } while (state.loaded && state.frame !== f);
    } finally {
      paneBusy.delete(cam);
    }
  })();
  paneBusy.set(cam, loop);
  return loop;
}

let prefetchTimer = 0;
function schedulePrefetch(frame) {
  clearTimeout(prefetchTimer);
  if (state.playing) { prefetch(frame); return; }
  prefetchTimer = setTimeout(() => {
    if (state.frame === frame) prefetch(frame);
  }, PREFETCH_IDLE_MS);
}

// ------------------------------------------------------------------ playhead

async function goto(frame) {
  if (!state.loaded) return;
  const n = state.loaded.timeline.n_frames;
  frame = Math.max(0, Math.min(n - 1, Math.round(frame)));
  state.frame = frame;

  viewer.setFrame(frame);

  const t = timeAt(frame);
  el.seek.value = String(frame);
  // The slider is positioned by TIME, not by frame index. The two part company
  // wherever filler is denser or sparser than the anchors, and a handle that
  // tracked the index would drift away from the marks strip beside it, which is
  // drawn in seconds.
  const dur = Math.max(1e-9, state.loaded.timeline.duration_s);
  el.seek.style.setProperty("--pct", `${(t / dur) * 100}%`);
  el.fNow.textContent = String(frame);
  el.tNow.textContent = t.toFixed(2);
  // Exact frames — the ones that ARE a commanded deadline — are worth marking,
  // because the commanded/measured comparison only means what it looks like it
  // means on those. On filler the overlay is holding.
  el.fNow.classList.toggle("anchor", !!state.loaded.timeline.on_anchor?.[frame]);
  marks.setFrame(frame, t);
  metric.setCursor(t);
  for (const fn of hooks.onFrame) fn(frame);

  schedulePrefetch(frame);
  await Promise.all(state.cameras.map((cam) => showCamera(cam.name)));
}

function setPlaying(on) {
  state.playing = on;
  el.play.innerHTML = on ? "&#10074;&#10074;" : "&#9654;";
  el.play.setAttribute("aria-label", on ? "pause" : "play");
  if (on) {
    state.lastRAF = performance.now();
    // Resume from where the playhead IS, in seconds. Playback advances a clock
    // and looks up the frame, so this is the state it has to carry.
    state.clock = timeAt(state.frame);
    requestAnimationFrame(tick);
  }
}
const pause = () => setPlaying(false);

function tick(now) {
  if (!state.playing || !state.loaded) return;
  state.clock += (now - state.lastRAF) / 1000;
  state.lastRAF = now;

  // Advance a WALL CLOCK and resolve it to a frame, rather than stepping the
  // frame index by elapsed frames. On a uniform grid the two agree; on this one
  // they do not, and stepping the index would run the 197 ms replan gaps — five
  // filler frames — at the same rate as five ordinary ticks, i.e. play the
  // moment the arm is frozen six times too fast. Advancing time keeps the
  // rollout at 1x everywhere, and a rendering stall still catches up rather
  // than replaying in slow motion.
  const end = state.loaded.timeline.duration_s;
  if (state.clock >= end) {
    goto(state.loaded.timeline.n_frames - 1);
    return pause();
  }
  const next = frameAt(state.clock);
  if (next !== state.frame) goto(next);
  requestAnimationFrame(tick);
}

$("chk-base").addEventListener("change", (e) => {
  const parts = ["body", "gripper"];
  if (e.target.checked) parts.push("wheel", "column");
  viewer.setVisibleParts(parts);
});
/** Grey out a switch that has nothing left to control, and say why by making it
 *  unclickable. A checkbox that stays lit while being quietly ignored is how a
 *  working feature comes to look broken on the one click most likely to be
 *  tried. */
function dependsOn(childId, on) {
  const child = $(childId);
  if (!child) return;                 // a mode's switch that is not built yet
  child.disabled = !on;
  child.parentElement.classList.toggle("off", !on);
}

$("chk-tcp").addEventListener("change", (e) => {
  viewer.tcpMarkers = e.target.checked;
  // A trail is its marker's own path and cannot outlive it.
  dependsOn("chk-trail", e.target.checked);
  viewer.draw();
});
$("chk-trail").addEventListener("change", (e) => {
  viewer.tcpTrails = e.target.checked;
  viewer.draw();
});

/** Push every source switch's state into the viewer, once, at startup.
 *
 *  The viewer's flags have their own defaults (`gl.js`) and the checkboxes have
 *  theirs (`index.html`), and nothing made the two agree until the first click:
 *  `tcpTrails` starts true in the viewer while `chk-trail` starts unchecked, so
 *  a fresh page drew the trajectories under a switch that said they were off,
 *  and clearing them meant toggling the switch on and off again. Driving the
 *  flags from the DOM here makes the markup the single answer to what is on —
 *  every future switch included, since one added to `index.html` checked or
 *  unchecked now means what it says. */
function syncSwitches() {
  viewer.measured = $("chk-measured").checked;
  viewer.ghosts = $("chk-ghost").checked;
  viewer.solvedOn = $("chk-solved").checked;
  viewer.tcpMarkers = $("chk-tcp").checked;
  viewer.tcpTrails = $("chk-trail").checked;
  const parts = ["body", "gripper"];
  if ($("chk-base").checked) parts.push("wheel", "column");
  viewer.setVisibleParts(parts);
  dependsOn("chk-trail", $("chk-tcp").checked);
}

// The three sources. Each flag gates that source's geometry, its TCP marker and
// its trail together inside the viewer, so there is nothing to keep in step
// here: one checkbox, one flag, and the trajectory follows the marker by
// construction rather than by a rule written twice.
for (const [id, flag] of [["chk-measured", "measured"],
                          ["chk-ghost", "ghosts"],
                          ["chk-solved", "solvedOn"]]) {
  $(id).addEventListener("change", (e) => {
    viewer[flag] = e.target.checked;
    // The wheels and the lift column belong to the measured robot and to
    // nothing else: no body target commands them, so `scene.py` leaves them out
    // of the solved overlay entirely. With the measured robot hidden there is
    // no base left for that switch to show or hide.
    if (flag === "measured") dependsOn("chk-base", e.target.checked);
    viewer.draw();
  });
}

/** The commanded POSE on a joint-mode rollout.
 *
 *  A joint policy commands angles, so its ticks carry no pose and the ghost
 *  (and its TCP marker and trail, which are part of it) has nothing to draw —
 *  EXCEPT where an operator intervened over PICO, whose commands are EEF
 *  poses. With no intervention the switch is greyed out and says why; with
 *  one it stays live and says it draws only those stretches. The per-frame
 *  validity the server sends already hides the ghost everywhere else. */
function applyActionMode(d) {
  const mode = d.scene.action_mode || "eef";
  const joint = mode === "joint";
  const pico = d.scene.intervention_frames || 0;
  const note = $("ghost-note");
  note.textContent = !joint ? "" : pico ? "(PICO only)" : "(joint policy)";
  dependsOn("chk-ghost", !joint || pico > 0);
  const box = $("chk-ghost");
  viewer.ghosts = box.checked && !box.disabled;
  hooks.actionMode = mode;
  // A robot whose file defines no TCP (a BrainCo today): the markers would
  // sit on the flange and read as a TCP, so they are not drawn at all.
  const hasTcp = d.scene.tcp !== false;
  const tcpBox = $("chk-tcp");
  dependsOn("chk-tcp", hasTcp);
  viewer.tcpMarkers = hasTcp && tcpBox.checked;
  dependsOn("chk-trail", hasTcp && tcpBox.checked);
  $("tcp-note").textContent = hasTcp ? "" : "(robot defines none)";
  viewer.draw();
}

// Every switch is wired; now make the viewer agree with all of them.
syncSwitches();

el.play.addEventListener("click", () => setPlaying(!state.playing));
el.reset.addEventListener("click", () => { pause(); goto(0); });
el.seek.addEventListener("input", () => { pause(); goto(Number(el.seek.value)); });

window.addEventListener("keydown", (e) => {
  // Escape closes the details card, and does so whether or not a rollout is
  // loaded — the card is readable before the first Load, which is most of when
  // it is read.
  if (e.code === "Escape" && !el.card.hidden) { showCard(false); return; }
  if (!state.loaded || e.target.tagName === "SELECT") return;
  if (e.code === "Space") { e.preventDefault(); setPlaying(!state.playing); }
  else if (e.code === "ArrowRight") { pause(); goto(state.frame + (e.shiftKey ? 10 : 1)); }
  else if (e.code === "ArrowLeft") { pause(); goto(state.frame - (e.shiftKey ? 10 : 1)); }
  else if (e.code === "Home") { pause(); goto(0); }
  // Walk the replans. At 30 fps a chunk boundary is three or four arrow presses
  // away and easy to overshoot; these land on the mark itself.
  else if (e.code === "BracketRight" || e.code === "BracketLeft") {
    const f = marks.step(e.code === "BracketRight" ? 1 : -1);
    if (f !== null) { pause(); goto(f); }
  }
});

// `?rec=<key>&load=1` selects a recording by its path under the root, and
// `load=1` presses the button. Enough to bookmark a rollout or paste one to
// someone serving the same root — and the only way to drive the page from a
// headless browser, which is how the renderer gets tested.
//
// `?off=measured,base` unchecks switches by name — the id minus `chk-`, and
// `?on=wb-<method>` checks a re-solved overlay's switch once it exists. A view
// with two of the three sources hidden is a real thing to want to send someone
// ("look at the right arm's commanded pose against what the IK made of it"),
// and it is the only way a screenshot can check that a switch takes its trail
// with it. Dispatching the event rather than setting the flag directly means
// the query drives the app through exactly the path a click does.
function applySwitches(q) {
  for (const name of (q.get("off") || "").split(",")) {
    const box = name && $(`chk-${name.trim()}`);
    if (!box) continue;
    box.checked = false;
    box.dispatchEvent(new Event("change"));
  }
}

function applyOn(q) {
  for (const name of (q.get("on") || "").split(",")) {
    const box = name && $(`chk-${name.trim()}`);
    if (!box || box.checked) continue;
    box.checked = true;
    box.dispatchEvent(new Event("change"));
  }
}

// --- modes ------------------------------------------------------------------
//
// The third panel is a MODE (`/api/modes`): none, or one of the server's.
// Choosing one imports its `mode_<name>.js` and calls `activate(ctx)` with the
// slots it may fill — the banner strip, the panel, the readout, the legend —
// and `hooks`. Switching away calls `deactivate()`, which is expected to take
// its overlays and its callbacks with it. `?mode=<name>` carries the choice.
const modes = { list: [], active: "", module: null };
const modeEl = {
  select: $("sel-mode"), banner: $("mode-banner"), panel: $("mode-panel"),
  title: $("mode-title"), note: $("mode-note"), body: $("mode-body"),
  readout: $("mode-readout"), readoutHead: $("mode-readout-head"),
  rows: $("mode-rows"), legend: $("mode-legend"),
};

async function loadModes() {
  const r = await fetch("/api/modes");
  modes.list = await r.json();
  for (const m of modes.list) {
    const o = document.createElement("option");
    o.value = m.name;
    o.textContent = m.title;
    o.title = m.blurb;
    modeEl.select.append(o);
  }
}

async function setMode(name) {
  if (name === modes.active) return;
  if (modes.module?.deactivate) {
    try { modes.module.deactivate(); } catch (err) { console.error(err); }
  }
  modes.module = null;
  modes.active = "";
  metric.hide();
  for (const box of [modeEl.banner, modeEl.body, modeEl.rows, modeEl.legend]) box.innerHTML = "";
  modeEl.title.textContent = "";
  modeEl.note.textContent = "";
  modeEl.panel.hidden = true;
  modeEl.readout.hidden = true;
  viewer.draw();
  const info = modes.list.find((m) => m.name === name);
  const url = new URL(location.href);
  if (info) url.searchParams.set("mode", name); else url.searchParams.delete("mode");
  history.replaceState(null, "", url);
  modeEl.select.value = info ? name : "";
  if (!info) return;
  try {
    const mod = await import(info.script);
    modeEl.title.textContent = info.title;
    modeEl.panel.hidden = false;
    modes.active = name;
    modes.module = mod;
    await mod.activate({ hooks, info, ...modeEl });
    if (state.loaded) applyOn(new URLSearchParams(location.search));
  } catch (err) {
    console.error(err);
    status(`<span class="err">mode ${name}: ${err.message}</span>`);
    modeEl.panel.hidden = true;
    modes.active = "";
    modes.module = null;
  }
}

modeEl.select.addEventListener("change", () => setMode(modeEl.select.value));

async function applyQuery() {
  const q = new URLSearchParams(location.search);
  if (q.has("mode")) await setMode(q.get("mode"));
  if (q.has("rec")) { el.rec.value = q.get("rec"); updateLoadButton(); }
  applySwitches(q);
  if (q.get("load") && !el.load.disabled) el.load.click();
  if (q.has("frame")) {
    const f = Number(q.get("frame"));
    const once = setInterval(() => {
      if (state.loaded) { clearInterval(once); goto(f); }
    }, 100);
    setTimeout(() => clearInterval(once), 30000);
  }
}

Promise.all([loadTree(), loadModes()])
  .then(applyQuery)
  .catch((err) => status(`<span class="err">${err.message}</span>`));
