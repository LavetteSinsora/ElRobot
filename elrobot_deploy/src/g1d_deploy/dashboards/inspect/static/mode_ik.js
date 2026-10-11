// Mode `ik`: which IK solver would have reached this rollout's poses.
//
// The re-solved overlays the loaded recording carries — every method in
// `ik_bench.npz` — arrive inside
// the transform buffer (`app.js` hands them over as `hooks.wbik`). This builds
// one switch, one legend swatch and one strip line per name. Nothing about
// the names is written down here: a scheme registered with the bench gets a
// switch by arriving.
//
// What it also does is PUT them there. Opening this mode on a recording nobody
// has benched spawns the bench for the infra's own solvers (`producers/ik_bench.py`:
// arm_planner, arm_qp, waist_planner, waist_qp, whichever this install's
// `runtime/ik` registers), merges the answers into the recording's sidecar and
// reloads the rollout so the switches appear.
//
// A joint-mode rollout has no pose to re-solve except where an operator
// intervened over PICO; with no intervention the panel says so and Solve stays
// off (`applicable` from the server).
//
// The strip under the scrubber (`hooks.strip`) plots the residual per tick over
// the whole rollout (`/api/ik/strip`): the deployed solution's line, plus one
// line per method whose switch is on, in that switch's colour. Millimetres at
// the virtual TCP (the flange when the robot file names none) or, toggled,
// degrees of flange orientation — never summed.
//
// The reload is the price of the design and worth naming: a method is drawn
// from the transform buffer, which is built once per load, so a method that
// did not exist at load time cannot appear without one. It is a few seconds
// and it keeps the drawing path single — there is no second way for an arm to
// reach the screen.

import { wbikColour } from "/static/gl.js";

const $ = (id) => document.getElementById(id);

//: The three tiers `g1d_deploy.runtime.ik.waist_qp` ships have long labels worth keeping;
//: everything else is labelled by its own name.
const TIER_LABELS = {
  waist: "+ waist", waist_column: "+ waist, column",
  full: "+ waist, column, base",
};

//: The strip's quantity, "mm" or "deg"; remembered in this browser.
const UNIT_KEY = "ik.stripUnit";
//: The deployed solution's line: the commanded joints' own purple (`.dot.solved`).
const DEPLOYED = "#c17dff";

const BANNER = `
  <label class="picker"><span>solve</span><select id="ik-method" disabled></select></label>
  <button id="ik-solve" disabled>Solve</button>
  <span id="ik-job" class="ci-job"></span>`;

const PANEL_TIP = "Every IK scheme solved against this recording, one switch each: the arm each method would have published, posed on the trunk it solved on, drawn as a ghost over the measured robot. Each switched-on method also draws its residual on the strip under the scrubber. Solving is a CPU job on this box (tens of seconds a method) that writes into the recording's own ik_bench.npz and reloads the rollout when it lands.";

let ctx = null;
let hooks = null;
let tiers = [];                     // {name, chk, label, colour, meta}

const state = {
  methods: null,                    // /api/ik/methods
  on: new Set(),                    // names switched on, kept across a reload
  key: "",                          // the loaded recording, to notice a change
  poll: null,
  solving: [],                      // what the running job is solving
  strip: null,                      // /api/ik/strip, for the loaded recording
  autoTried: "",                    // the recording the auto-solve has fired for
  unit: "mm",
};

const onLoaded = () => {
  build();
  loadStrip();
  refreshMethods().then(maybeAuto);
};

// --- the switches -----------------------------------------------------------

function build() {
  const { body, legend, readout } = ctx;
  const css = hooks.css;
  const loaded = hooks.state();
  const key = loaded ? JSON.stringify(loaded.key || {}) : "";
  if (key !== state.key) {          // another recording: its own switches
    state.key = key;
    state.on.clear();
  }
  body.innerHTML = "";
  legend.innerHTML = "";
  const wb = hooks.wbik;
  readout.hidden = true;             // the strip under the scrubber says it
  if (!loaded || !wb) { tiers = []; return; }
  const meta = wb.meta || {};
  let idx = 0;
  tiers = wb.tiers.map((name) => {
    const info = (meta.names && meta.names[name]) || {};
    const colour = wbikColour(name, TIER_LABELS[name] ? 0 : idx++);
    const label = TIER_LABELS[name] || name;
    const chk = `chk-wb-${name}`;
    const lab = document.createElement("label");
    lab.id = `lbl-wb-${name}`;
    const tip = [];
    if (info.source === "bench") tip.push(`IK bench method ‘${name}’ (${info.mode})`);
    if (info.describe) tip.push(info.describe);
    if (info.tcp_mean_mm != null) tip.push(`mean TCP miss ${info.tcp_mean_mm.toFixed(3)} mm over this recording`);
    if (info.source === "wbik") tip.push("a wbik_overlay.npz tier");
    lab.title = tip.join("\n");
    lab.innerHTML = `<input type="checkbox" id="${chk}"> `
      + `<i class="dot" style="background:${css(colour)}"></i>${label}`;
    body.appendChild(lab);
    const sw = document.createElement("span");
    sw.className = "wb";
    sw.innerHTML = `<i class="dot" style="background:${css(colour)}"></i>${label}`;
    legend.appendChild(sw);
    $(chk).addEventListener("change", (e) => {
      if (e.target.checked) state.on.add(name); else state.on.delete(name);
      hooks.viewer.setWbikOn(name, e.target.checked);
      paintStrip();
    });
    // A solve that landed while these switches were on comes back through a
    // reload, which rebuilds them unchecked and resets the viewer's flags.
    // Re-asserting here is what makes the reload invisible.
    if (state.on.has(name)) {
      $(chk).checked = true;
      hooks.viewer.setWbikOn(name, true);
    }
    return { name, chk, label, colour, meta: info };
  });
  note();
}

/** The line beside the panel's title: what is drawn, and what is missing. */
function note() {
  const meta = (hooks.wbik || {}).meta || {};
  const warnings = (meta.notes || []).join(" ");
  const m = state.methods;
  const missing = m ? m.missing.length : 0;
  const parts = [];
  if (tiers.length) parts.push(`(${tiers.length})`);
  else if (!missing) {
    parts.push(m && m.error
      ? "not solved, and the bench is unavailable — see the picker"
      : "not solved yet");
  }
  if (missing && !state.poll) parts.push(`${missing} of the default set not solved`);
  ctx.note.textContent = [...parts, warnings].filter(Boolean).join(" · ");
}

// --- the strip --------------------------------------------------------------

async function loadStrip() {
  state.strip = null;
  if (!hooks || !hooks.state()) return;
  paintStrip();
  try {
    state.strip = await getJSON("/api/ik/strip");
  } catch (err) {
    state.strip = { error: err.message };
  }
  paintStrip();
}

function unitTools() {
  const box = document.createElement("span");
  for (const [u, label, tip] of [
    ["mm", "mm", "translation error at the TCP, mm"],
    ["deg", "deg", "orientation error of the flange, degrees"]]) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = label;
    b.title = tip;
    b.className = state.unit === u ? "on" : "";
    b.addEventListener("click", () => {
      state.unit = u;
      try { localStorage.setItem(UNIT_KEY, u); } catch { /* private mode */ }
      paintStrip();
    });
    box.append(b);
  }
  return box;
}

/** The deployed line and one per switched-on method, in the chosen unit. */
function paintStrip() {
  if (!hooks || !hooks.strip) return;
  const d = state.strip;
  const deg = state.unit === "deg";
  const point = d?.point || "TCP";
  const base = {
    title: deg ? "IK rot. error" : "IK residual",
    unit: deg ? "deg" : "mm",
    help: deg
      ? "Per commanded tick, worst hand: the angle between the commanded and the solved flange orientation, degrees."
      : `Per commanded tick, worst hand: the distance between the commanded and the solved ${point}, mm.`,
    tools: unitTools(),
  };
  if (!d) { hooks.strip.show({ ...base, series: [], empty: "loading…" }); return; }
  if (d.error) { hooks.strip.show({ ...base, series: [], empty: d.error }); return; }
  if (!d.applicable) { hooks.strip.show({ ...base, series: [], empty: d.why }); return; }
  const key = deg ? "rot_deg" : "tcp_mm";
  const colours = new Map(tiers.map((t) => [t.name, hooks.css(t.colour)]));
  const labels = new Map(tiers.map((t) => [t.name, t.label]));
  const series = [];
  for (const s of d.series) {
    if (s.name === "deployed") {
      series.push({ label: `deployed (${point})`, color: DEPLOYED, t: d.t, y: s[key], width: 1.6 });
    } else if (colours.has(s.name) && $(`chk-wb-${s.name}`)?.checked) {
      series.push({ label: labels.get(s.name), color: colours.get(s.name), t: d.t, y: s[key] });
    }
  }
  hooks.strip.show({ ...base, series });
}

// --- solving ----------------------------------------------------------------

async function getJSON(url, opts) {
  const r = await fetch(url, opts);
  const d = await r.json().catch(() => ({ error: r.statusText }));
  if (!r.ok) throw new Error(d.error || r.statusText);
  return d;
}

function job(text, bad = false) {
  const el = $("ik-job");
  if (!el) return;
  el.textContent = text;
  el.classList.toggle("bad", Boolean(bad));
}

/** The picker: the default set first, then every registered method by mode. */
async function refreshMethods() {
  if (!ctx) return;                 // deactivated while a poll was in flight
  let d;
  try {
    d = await getJSON("/api/ik/methods");
  } catch (err) {
    job(err.message, true);
    return;
  }
  state.methods = d;
  const sel = $("ik-method");
  sel.innerHTML = "";
  if (d.missing.length) {
    const o = document.createElement("option");
    o.value = "__missing__";
    o.textContent = `the default set (${d.missing.length} missing)`;
    o.title = d.missing.join(", ");
    sel.append(o);
  }
  const byMode = new Map();
  for (const m of d.methods) {
    if (!byMode.has(m.mode)) byMode.set(m.mode, []);
    byMode.get(m.mode).push(m);
  }
  for (const [mode, rows] of byMode) {
    const g = document.createElement("optgroup");
    g.label = mode ? `${mode} methods` : "in the file";
    for (const m of rows) {
      const o = document.createElement("option");
      o.value = m.name;
      // A solved method stays selectable: re-solving is how a method changed
      // since the sidecar was written gets its new answer in.
      o.textContent = (m.solved ? "✓ " : "") + m.name
                      + (m.name === d.deployed ? "  (flew)" : "")
                      + (m.available ? "" : "  (needs setup)");
      o.disabled = !m.available;
      o.title = [m.describe, m.why].filter(Boolean).join("\n");
      g.append(o);
    }
    sel.append(g);
  }
  sel.disabled = !sel.options.length || Boolean(d.error);
  const canSolve = d.applicable && Boolean(d.record) && !d.busy;
  $("ik-solve").disabled = !canSolve || sel.disabled;
  if (d.error) job(d.error, true);
  else if (d.record && !d.applicable) job(d.why || "nothing to re-solve", true);
  else if (!state.poll) job("");
  note();
}

/** Solve the default set once per recording, unprompted: the infra's own
 *  solvers minus the one the rollout flew (the server leaves it out of
 *  `missing`). Once — a failed or cancelled run is re-asked with Solve. */
async function maybeAuto() {
  const d = state.methods;
  if (!d || state.poll || d.busy) return;
  if (!d.missing.length || !d.applicable || !d.record) return;
  if (state.autoTried === d.record) return;
  state.autoTried = d.record;
  await solve(d.missing);
}

async function solve(names) {
  if (!names.length) return;
  $("ik-solve").disabled = true;
  state.solving = names;
  // What you asked for is what you want to look at: switched on when it lands.
  for (const n of names) state.on.add(n);
  if (state.methods?.record) state.autoTried = state.methods.record;
  try {
    await getJSON("/api/ik/solve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ methods: names }),
    });
    job(`solving ${names.length}…`);
    startPolling();
  } catch (err) {
    job(err.message, true);
    $("ik-solve").disabled = false;
  }
}

function onSolveClick() {
  const v = $("ik-method").value;
  solve(v === "__missing__" ? (state.methods?.missing || []) : [v]);
}

function startPolling() {
  if (state.poll) return;
  state.poll = setInterval(async () => {
    let d;
    try { d = await getJSON("/api/ik/jobs"); } catch { return; }
    const running = d.jobs.filter((j) => ["queued", "running"].includes(j.status));
    if (running.length) {
      const j = running[running.length - 1];
      const p = j.progress || {};
      const where = p.total ? `${p.done || 0}/${p.total}` : "";
      job(`${p.method || j.status} ${where} · ${j.elapsed_s}s`);
      return;
    }
    clearInterval(state.poll);
    state.poll = null;
    const last = d.jobs[d.jobs.length - 1];
    if (last && last.status === "failed") {
      job(`failed — see ${last.log}`, true);
    } else if (last && last.status === "cancelled") {
      job("cancelled");
    } else if (last) {
      const errs = Object.keys((last.progress || {}).errors || {}).length;
      job(`solved in ${last.elapsed_s}s` + (errs ? ` · ${errs} failed` : ""),
          Boolean(errs));
    }
    // The methods are in the sidecar now; the transform buffer is from before
    // it. Reload to draw them, keeping the playhead and the switches that
    // were on — `onLoaded` fires inside this and rebuilds the panel.
    await reload();
  }, 1000);
}

async function reload() {
  if (!ctx) return;                 // ditto: the mode was switched away
  const frame = hooks.frame();
  try {
    await hooks.reload();
    hooks.goto(frame);
  } catch (err) {
    job(`reload failed: ${err.message}`, true);
    await refreshMethods();
  }
}

// --- wiring -----------------------------------------------------------------

export async function activate(c) {
  ctx = c;
  hooks = c.hooks;
  ctx.banner.innerHTML = BANNER;
  ctx.body.parentElement.querySelector("#mode-head").title = PANEL_TIP;
  try { state.unit = localStorage.getItem(UNIT_KEY) === "deg" ? "deg" : "mm"; } catch { /* private mode */ }
  $("ik-solve").addEventListener("click", onSolveClick);
  hooks.onLoaded.push(onLoaded);
  build();
  loadStrip();
  await refreshMethods();
  // A bench run may already be in flight — the panel can be closed and
  // reopened while the CPU works, and the job outlives this module. It has to
  // be an IK job: `busy` is the whole box (a checkpoint solve makes it true
  // too), and polling for one of those would end in a pointless reload.
  let mine = [];
  try {
    const d = await getJSON("/api/ik/jobs");
    mine = d.jobs.filter((j) => ["queued", "running"].includes(j.status));
  } catch { /* the panel still works off the sidecar */ }
  if (mine.length) startPolling(); else await maybeAuto();
}

export function deactivate() {
  if (state.poll) { clearInterval(state.poll); state.poll = null; }
  for (const t of tiers) hooks.viewer.setWbikOn(t.name, false);
  hooks.off(hooks.onLoaded, onLoaded);
  tiers = [];
  state.methods = null;
  state.strip = null;
  state.on.clear();
  ctx = null;
  hooks = null;
}
