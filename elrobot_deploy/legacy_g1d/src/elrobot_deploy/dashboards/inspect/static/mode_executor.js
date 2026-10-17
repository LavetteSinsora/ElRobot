// Mode `executor`: how another 500 Hz executor would have driven the joints.
//
// One switch per replay of the loaded rollout (robot runs solid, sim runs
// dashed), a joint picker, and a strip chart of that joint around the
// playhead: the measured angle against the command each executor sent, the
// source's knots as ticks. The playhead lands on the replays' common clock
// through the server's `offset_s` (chunk 1's first executed tick is t = 0 in
// every run of one source). "Replay" spawns the configured producer
// (`--executor-producer`, `[inspect] executor_producer`) for the chosen
// variants; with none configured the existing replays are still listed and the
// panel says how to plug one in.
//
// The strip under the scrubber (`hooks.strip`) plots each shown run's servo
// tracking error per tick over the whole rollout (`/api/executor/strip`): the
// mean over the 14 arm joints of |measured - commanded|, degrees — this
// rollout's own (when it recorded P7's command) and one line per replay
// switched on, in its colour.

import { drawStrip } from "/static/strip.js";

const $ = (id) => document.getElementById(id);

const PALETTE = [
  [0.98, 0.36, 0.32], [0.99, 0.80, 0.24], [0.25, 0.85, 0.62], [0.98, 0.52, 0.85],
  [0.55, 0.74, 1.00], [0.72, 0.95, 0.35], [1.00, 0.60, 0.35], [0.60, 0.85, 0.90],
];
const WINDOW_S = 1.5;            // half-width of the chart, seconds

let ctx = null;
let hooks = null;
const state = {
  replays: [], joints: [], variants: [], offset: 0, producer: "", note: "",
  colors: new Map(), on: new Set(), series: new Map(),   // name -> {joint, data}
  joint: 3, poll: null, frame: 0,
  strips: new Map(),              // run name ("" = this rollout) -> /api/executor/strip
};

//: This rollout's own line on the strip.
const OWN = "#e6edf3";

async function getJSON(url, opts) {
  const r = await fetch(url, opts);
  const d = await r.json().catch(() => ({ error: r.statusText }));
  if (!r.ok) throw new Error(d.error || r.statusText);
  return d;
}

// --- the panel --------------------------------------------------------------

function buildPanel() {
  const { body } = ctx;
  const css = hooks.css;
  body.innerHTML = `
    <div class="ex-row"><span>joint</span><select id="ex-joint"></select></div>
    <div id="ex-list"></div>
    <canvas id="ex-chart" class="ex-chart" width="600" height="150"></canvas>
    <div class="ex-row"><span>replay</span><select id="ex-variants" multiple size="3"></select>
      <button id="ex-run" disabled>Run</button><span id="ex-job" class="ci-job"></span></div>`;
  const joint = $("ex-joint");
  state.joints.forEach((n, i) => {
    const o = document.createElement("option");
    o.value = String(i); o.textContent = n; if (i === state.joint) o.selected = true;
    joint.append(o);
  });
  joint.addEventListener("change", async () => {
    state.joint = parseInt(joint.value, 10);
    state.series.clear();
    for (const name of state.on) await fetchSeries(name);
    draw();
  });
  const vs = $("ex-variants");
  for (const v of state.variants) {
    const o = document.createElement("option");
    o.value = v.name; o.textContent = v.name; o.title = v.note; o.selected = Boolean(v.default);
    vs.append(o);
  }
  $("ex-run").disabled = !state.producer || !state.variants.length || !hooks.state();
  if (state.note) job(state.note, true);
  $("ex-run").addEventListener("click", runSim);
  const list = $("ex-list");
  list.innerHTML = "";
  let nth = 0;
  for (const r of state.replays) {
    if (!state.colors.has(r.name)) state.colors.set(r.name, PALETTE[nth % PALETTE.length]);
    nth += 1;
    const lab = document.createElement("label");
    lab.className = "ci-solve";
    if (r.error) lab.classList.add("ci-broken");
    const box = document.createElement("input");
    box.type = "checkbox"; box.disabled = Boolean(r.error); box.checked = state.on.has(r.name);
    box.id = `chk-ex-${r.name}`;
    box.addEventListener("change", () => toggle(r.name, box.checked));
    const dot = document.createElement("i");
    dot.className = "dot"; dot.style.background = css(state.colors.get(r.name));
    const text = document.createElement("span");
    text.className = "ci-label";
    text.textContent = `${r.variant || "?"} · ${r.kind === "sim" ? "sim" : "robot"} · ${r.name.slice(9, 15)}`;
    const right = document.createElement("span");
    right.className = "ci-err";
    right.textContent = r.error ? "unreadable" : (r.has_command ? "" : "no command stream");
    lab.title = r.error || [r.variant_note, r.ik ? `ik ${r.ik}` : "",
                            `created ${r.created || "?"}`].filter(Boolean).join(" · ");
    lab.append(box, dot, text, right);
    list.append(lab);
  }
  ctx.note.textContent = state.replays.length ? `(${state.replays.length})` : "(no replays of this rollout yet)";
}

async function fetchSeries(name) {
  const d = await getJSON(`/api/executor/series?run=${encodeURIComponent(name)}&joint=${state.joint}`);
  state.series.set(name, d);
  return d;
}

async function toggle(name, on) {
  if (!on) { state.on.delete(name); draw(); paintStrip(); return; }
  try {
    await fetchSeries(name);
    state.on.add(name);
  } catch (err) {
    job(err.message, true);
  }
  draw();
  await fetchStrip(name);
  paintStrip();
}

// --- the strip --------------------------------------------------------------

async function fetchStrip(name) {
  if (state.strips.has(name)) return;
  try {
    state.strips.set(name, await getJSON(`/api/executor/strip?run=${encodeURIComponent(name)}`));
  } catch (err) {
    state.strips.set(name, { t: [], err_deg: [], error: err.message });
  }
}

function paintStrip() {
  if (!hooks || !hooks.strip || !hooks.state()) return;
  const series = [];
  const own = state.strips.get("");
  if (own && own.err_deg.length) {
    series.push({ label: "this rollout", color: OWN, t: own.t, y: own.err_deg, width: 1.6 });
  }
  for (const n of state.on) {
    const d = state.strips.get(n);
    if (!d || !d.err_deg.length) continue;
    const r = state.replays.find((x) => x.name === n) || {};
    series.push({ label: `${r.variant || "?"} · ${r.kind === "sim" ? "sim" : "robot"}`,
                  color: hooks.css(state.colors.get(n)), t: d.t, y: d.err_deg,
                  dash: r.kind === "sim" ? [3, 2] : [] });
  }
  hooks.strip.show({
    title: "servo tracking", unit: "deg",
    help: "Per tick: the mean over the 14 arm joints of |measured − commanded| angle, degrees (commanded = the 500 Hz command P7 sent), averaged from the tick's deadline to the next.",
    series,
    empty: own && own.err_deg.length === 0 && !state.on.size
      ? "this rollout recorded no P7 command; switch a replay on"
      : "switch a replay on",
  });
}

// --- the chart --------------------------------------------------------------

// The drawing is the shared strip chart (`/static/strip.js`), which the
// timeline strip under the scrubber (`metric.js`) uses too.
function draw() {
  const canvas = $("ex-chart");
  if (!canvas) return;
  const active = [...state.on].filter((n) => state.series.has(n));
  const tc = cursorTime();
  if (!active.length || tc === null) {
    drawStrip(canvas, { x0: 0, x1: 1, series: [],
      empty: active.length ? "no playhead mapping for this rollout" : "switch a replay on" });
    return;
  }
  const series = [];
  for (const n of active) {
    const s = state.series.get(n);
    const r = state.replays.find((x) => x.name === n) || {};
    const color = hooks.css(state.colors.get(n));
    const sim = r.kind === "sim";
    // the command each sent, then what the joint did
    if (s.q_cmd) series.push({ x: s.t, y: s.q_cmd, color, dash: sim ? [2, 3] : [6, 3], alpha: 0.55, width: 1.4 });
    if (s.q_meas) series.push({ x: s.t, y: s.q_meas, color, dash: sim ? [2, 3] : [], width: 1.4 });
  }
  // knots of the first active run (the source's own deadlines, shared)
  drawStrip(canvas, {
    x0: tc - WINDOW_S, x1: tc + WINDOW_S, series, cursor: tc, minSpan: 0.04,
    marks: state.series.get(active[0]).knot_t,
    label: (lo, hi) => `${state.joints[state.joint]}  ${lo.toFixed(2)} … ${hi.toFixed(2)} rad  ·  solid measured, dashed command`,
  });
}

/** The playhead on the replays' clock: seconds since chunk 1 started executing. */
function cursorTime() {
  const d = hooks.state();
  if (!d || state.offset === null || state.offset === undefined) return null;
  // The display grid is not uniform (it lands on the deadlines), so the
  // frame's time is looked up, never `frame / (n - 1) * duration`.
  const tr = d.timeline.t_rel;
  const t = tr[Math.max(0, Math.min(tr.length - 1, state.frame))] ?? 0;
  return t - state.offset;
}

// --- replaying in sim -------------------------------------------------------

function job(text, bad = false) {
  const el = $("ex-job");
  if (!el) return;
  el.textContent = text;
  el.classList.toggle("bad", Boolean(bad));
}

async function runSim() {
  const variants = [...$("ex-variants").selectedOptions].map((o) => o.value);
  $("ex-run").disabled = true;
  try {
    const d = await getJSON("/api/executor/replay", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ variants }),
    });
    job(`queued ${d.job.id}…`);
    startPolling();
  } catch (err) {
    job(err.message, true);
    $("ex-run").disabled = false;
  }
}

function startPolling() {
  if (state.poll) return;
  state.poll = setInterval(async () => {
    let d;
    try { d = await getJSON("/api/executor/jobs"); } catch { return; }
    const running = d.jobs.filter((j) => ["queued", "running"].includes(j.status));
    if (running.length) { job(`${running[0].status} · ${running[0].elapsed_s}s`); return; }
    clearInterval(state.poll); state.poll = null;
    const last = d.jobs[d.jobs.length - 1];
    if (last && last.status === "failed") job(`failed — see ${last.log}`, true);
    else if (last) job(`done in ${last.elapsed_s}s`);
    await refresh();
  }, 2000);
}

async function refresh() {
  const d = await getJSON("/api/executor/replays");
  state.replays = d.replays; state.joints = d.joints; state.variants = d.variants || [];
  state.producer = d.producer || ""; state.note = d.note || "";
  state.offset = d.offset_s;
  for (const n of [...state.on]) if (!state.replays.some((r) => r.name === n)) state.on.delete(n);
  buildPanel();
  draw();
  await fetchStrip("");
  for (const n of state.on) await fetchStrip(n);
  paintStrip();
}

// --- wiring -----------------------------------------------------------------

const onLoaded = async () => {
  state.on.clear(); state.series.clear(); state.colors.clear(); state.strips.clear();
  job("");
  try { await refresh(); } catch (err) { job(err.message, true); }
};
const onFrame = (f) => { state.frame = f; draw(); };

export async function activate(c) {
  ctx = c; hooks = c.hooks;
  ctx.readout.hidden = true;
  hooks.onLoaded.push(onLoaded);
  hooks.onFrame.push(onFrame);
  state.frame = hooks.frame();
  state.strips.clear();
  await refresh();
  getJSON("/api/executor/jobs").then((d) => { if (d.busy) startPolling(); }).catch(() => {});
}

export function deactivate() {
  if (state.poll) { clearInterval(state.poll); state.poll = null; }
  hooks.off(hooks.onLoaded, onLoaded);
  hooks.off(hooks.onFrame, onFrame);
  state.on.clear(); state.series.clear(); state.strips.clear();
  ctx = null; hooks = null;
}
