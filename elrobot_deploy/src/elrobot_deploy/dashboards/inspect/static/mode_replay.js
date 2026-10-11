// Mode `replay`: what a policy would have commanded on this rollout's own
// observations.
//
// The banner holds "New replay", which opens a card with everything a replay
// can vary — the policy entry, the checkpoint (a run directory browsed on the
// server, a step in it, or a pasted step directory), the prompt, the sample
// steps, how many repeats — every field defaulting to what the rollout itself
// ran with (`/api/replay/defaults`, from its manifest). The panel lists the
// recording's own chunks ("live", the reference) and every replay beside it,
// one switch each, drawn through `Viewer.setOverlay` so they share this
// rollout's timeline and frame index.
//
// Overlay data is fetched LAZILY, once per replay per rollout: a replay's
// arrays are ~1.5 MB and building them costs the server a pass over the display
// frames, so six replays toggled on is six fetches, not six on load.

const $ = (id) => document.getElementById(id);

// Distinct from the viewer's own cyan / amber / violet. `LIVE_COLOUR` is
// near-white and belongs to the recording's own chunks alone: it is the
// reference, and a replay that took its colour would be read as it.
const LIVE_COLOUR = [0.93, 0.93, 0.93];
const PALETTE = [
  [0.98, 0.36, 0.32], [0.99, 0.80, 0.24], [0.25, 0.85, 0.62], [0.98, 0.52, 0.85],
  [0.55, 0.74, 1.00], [0.72, 0.95, 0.35], [1.00, 0.60, 0.35], [0.60, 0.85, 0.90],
  [0.80, 0.55, 0.98], [0.35, 0.78, 0.78], [0.95, 0.45, 0.60], [0.62, 0.72, 0.42],
];

const css = (c) => `rgb(${c.map((x) => Math.round(x * 255)).join(",")})`;
const esc = (s) => String(s ?? "").replace(/[&<>"]/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

const BANNER = `
  <div class="rec-info">
    <button id="rp-open" type="button" class="rec-chip rp-open" aria-expanded="false">
      <span class="rec-headline"><b>New replay</b></span>
      <span id="rp-sub" class="rec-subline">the rollout's own settings</span>
    </button>
    <div id="rp-card" class="rec-card rp-card" hidden>
      <div class="rp-sec">deploy file</div>
      <div class="rp-row">
        <input id="rp-deploy" class="rp-wide" spellcheck="false"
               placeholder="/path/to/deploy.toml"
               title="The deploy file names the policy adapter and the Python it runs under. Default: the one this rollout ran from.">
        <button id="rp-deploy-browse" type="button">Browse</button>
        <button id="rp-entry-reset" type="button" title="back to what the rollout ran with">&#8634;</button>
      </div>
      <div id="rp-class" class="rp-note"></div>

      <div class="rp-sec">checkpoint folder</div>
      <div class="rp-row">
        <input id="rp-ckpt" class="rp-wide" spellcheck="false"
               placeholder="/path/to/a/checkpoint/folder"
               title="The folder the policy loads its weights from. Default: the one this rollout ran with.">
        <button id="rp-ckpt-browse" type="button">Browse</button>
      </div>

      <div id="rp-picker" class="rp-picker" hidden>
        <div class="rp-row">
          <span id="rp-picker-what" class="rp-k"></span>
          <input id="rp-dir" class="rp-wide" spellcheck="false" placeholder="/absolute/path">
          <button id="rp-go" type="button">Open</button>
          <button id="rp-use" type="button">Use this folder</button>
        </div>
        <div id="rp-starts" class="rp-starts"></div>
        <div id="rp-browse" class="rp-browse"></div>
      </div>

      <div class="rp-sec">prompt</div>
      <textarea id="rp-prompt" rows="2" spellcheck="false"
        title="Checked by the policy itself (check_prompt_length) when the replay starts, before any inference; a prompt it cannot fit fails the replay with the dashboard's own sentence. Check prompt asks the same question now, without replaying."></textarea>
      <div class="rp-row">
        <button id="rp-check" type="button"
          title="Construct the policy with this checkpoint (never warmup, never infer) and judge the prompt with its own tokenizer. Runs without the GPU, so it works while a deployment is live.">Check prompt</button>
        <span id="rp-verdict" class="rp-note" role="status"></span>
      </div>

      <div class="rp-sec">settings</div>
      <div class="rp-row">
        <span class="rp-k" title="Handed to the policy through on_settings, as P5 does at START.">sample steps</span>
        <input id="rp-steps" type="number" min="1" step="1" class="rp-num">
        <span class="rp-k" title="Each repeat is its own replay and its own switch: the spread between repeats of one setting is the sampler's noise floor.">repeats</span>
        <input id="rp-repeats" type="number" min="1" max="20" step="1" value="1" class="rp-num">
      </div>
      <div id="rp-fixed" class="rp-note"></div>

      <div class="rp-row rp-go">
        <button id="rp-start" type="button" disabled>Replay</button>
        <span id="rp-gpu" class="rp-note rp-warn"></span>
      </div>
    </div>
  </div>
  <span id="rp-job" class="ci-job"></span>`;

const PANEL_TIP = "The recording's own chunks (live, the reference) and every replay of this rollout, one switch each — any number on at once, each in its own colour. A replay is the SAME observations the robot saw, re-run through a policy: a coloured gripper ghost at the chunk row in force, its TCP trail, and the whole chunk it is inside drawn ahead of the playhead — solid for the rows that would have executed, faint for the tail. The millimetres are that replay's median and p99 TCP distance from what the flown policy commanded. Re-running the flown checkpoint is not zero: the sampler's noise differs, so repeats of it are this rollout's sampling floor.";

let ctx = null;
let hooks = null;

const state = {
  defaults: null,             // /api/replay/defaults
  described: null,            // /api/replay/describe of the entry in the field
  replays: [],                // ReplayInfo[]
  colors: new Map(),          // id -> rgb triple
  loaded: new Map(),          // id -> {meta, arrays} once fetched
  on: new Set(),              // ids currently drawn
  pending: new Set(),         // ids the URL asked for before they were listed
  listing: null,              // the directory being browsed
  poll: null,
  strips: new Map(),          // id -> /api/replay/strip
};

async function getJSON(url, opts) {
  const r = await fetch(url, opts);
  const d = await r.json().catch(() => ({ error: r.statusText }));
  if (!r.ok) throw new Error(d.error || r.statusText);
  return d;
}

const post = (url, body, signal) => getJSON(url, {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body), signal,
});

function job(text, bad = false) {
  const el = $("rp-job");
  if (!el) return;
  el.textContent = text;
  el.classList.toggle("bad", Boolean(bad));
}

// --- overlays ---------------------------------------------------------------

async function fetchOverlay(id) {
  if (state.loaded.has(id)) return state.loaded.get(id);
  const q = `id=${encodeURIComponent(id)}`;
  const meta = await getJSON(`/api/replay/overlay?${q}`);
  const buf = await (await fetch(`/api/replay/overlay.bin?${q}`)).arrayBuffer();
  const view = (name, Type) => {
    const [off, nbytes] = meta.layout[name];
    return new Type(buf, off, nbytes / Type.BYTES_PER_ELEMENT);
  };
  const arrays = {
    ghost: view("ghost", Float32Array), ghostTcp: view("ghost_tcp", Float32Array),
    valid: view("valid", Uint8Array), chunkOf: view("chunk_of", Int32Array),
    rowOf: view("row_of", Int32Array), arc: view("arc", Float32Array),
    arcN: view("arc_n", Int32Array), arcExec: view("arc_exec", Int32Array),
    errMm: view("err_mm", Float32Array),
    nFrames: meta.n_frames, nChunks: meta.n_chunks, H: meta.H,
    slots: meta.ghost_geoms,
  };
  const entry = { meta, arrays };
  state.loaded.set(id, entry);
  return entry;
}

async function toggle(id, on, checkbox) {
  if (!on) {
    state.on.delete(id);
    hooks.viewer.setOverlayOn(id, false);
    paintReadout(lastFrame);
    paintStrip();
    return;
  }
  try {
    if (checkbox) checkbox.disabled = true;
    const { arrays } = await fetchOverlay(id);
    hooks.viewer.setOverlay(id, { ...arrays, color: state.colors.get(id), on: true });
    state.on.add(id);
    paintReadout(lastFrame);
    await fetchStrip(id);
    paintStrip();
  } catch (err) {
    if (checkbox) checkbox.checked = false;
    job(`overlay failed: ${err.message}`, true);
  } finally {
    if (checkbox) checkbox.disabled = false;
  }
}

// --- the strip --------------------------------------------------------------
//
// Under the scrubber (`hooks.strip`): per anchor, how far each switched-on
// replay's actions are from the recorded ones (`/api/replay/strip`) — the mean
// over the executed rows of the worst hand's flange translation, mm.

async function fetchStrip(id) {
  if (id === "live" || state.strips.has(id)) return;
  try {
    state.strips.set(id, await getJSON(`/api/replay/strip?id=${encodeURIComponent(id)}`));
  } catch (err) {
    state.strips.set(id, { t: [], diff_mm: [], error: err.message });
  }
}

function paintStrip() {
  if (!hooks || !hooks.strip || !hooks.state()) return;
  const series = [];
  for (const id of state.on) {
    const d = state.strips.get(id);
    if (!d || !d.t.length) continue;
    const r = state.replays.find((x) => x.id === id);
    series.push({ label: r ? r.label : id, color: css(state.colors.get(id)),
                  t: d.t, y: d.diff_mm, dots: true });
  }
  hooks.strip.show({
    title: "vs recorded", unit: "mm",
    help: "Per anchor (replan): the mean, over the rows the recorded chunk executed, of the worst hand's flange translation between the replay's action and the recorded one, mm. Plotted at the anchor instant.",
    series, empty: "switch a replay on",
  });
}

// --- the list ---------------------------------------------------------------

function mm(summary, arm) {
  const s = summary?.tcp_mm?.[arm];
  if (!s || s.p50 == null) return "–";
  return `${s.p50.toFixed(0)}/${s.p99.toFixed(0)}`;
}

function tooltip(r) {
  const s = r.settings || {};
  const lines = [
    r.kind === "live" ? r.note : `replayed ${r.created || "?"}`,
    `policy ${r.policy_class || r.entry || "?"}`,
    `checkpoint ${r.checkpoint || "(none)"}`,
    `prompt “${r.prompt}”`,
    `sample steps ${s.sample_steps ?? "?"} · ${s.sync ? "sync" : "async"} P${s.replan_period ?? "?"} · ${s.control_hz ?? "?"} Hz`,
    r.seed != null ? `seed ${r.seed}` : "",
    `${r.n_chunks} chunks`,
    r.kind === "live" ? "" : "L/R: median/p99 TCP mm from what flew",
    r.state === "broken" ? r.note : "",
  ];
  return lines.filter(Boolean).join("\n");
}

function renderList() {
  const box = $("rp-list");
  if (!box) return;
  box.innerHTML = "";
  // Colours are assigned once per rollout and kept, so a replay that finishes
  // while you read does not renumber the ones already on screen.
  let nth = 0;
  for (const r of state.replays) {
    const live = r.kind === "live";
    if (!state.colors.has(r.id)) {
      state.colors.set(r.id, live ? LIVE_COLOUR : PALETTE[nth % PALETTE.length]);
    }
    if (!live) nth += 1;
    const label = document.createElement("label");
    label.className = "ci-solve";
    if (r.state !== "ready") label.classList.add("ci-" + r.state);
    const box_ = document.createElement("input");
    box_.type = "checkbox";
    box_.id = `chk-rp-${r.id}`;
    box_.disabled = r.state !== "ready";
    box_.checked = state.on.has(r.id);
    box_.addEventListener("change", () => toggle(r.id, box_.checked, box_));
    const dot = document.createElement("i");
    dot.className = "dot";
    dot.style.background = css(state.colors.get(r.id));
    const text = document.createElement("span");
    text.className = "ci-label";
    text.textContent = r.label;
    const right = document.createElement("span");
    right.className = "ci-err";
    right.textContent = r.state === "running" ? "replaying…"
      : r.state === "broken" ? "failed"
      : live ? "reference"
      : `${mm(r.summary, "left")} · ${mm(r.summary, "right")} mm`;
    label.append(box_, dot, text, right);
    label.title = tooltip(r);
    box.append(label);
  }
  const n = state.replays.filter((r) => r.kind !== "live").length;
  ctx.note.textContent = n ? `(${n})` : "(none yet — New replay)";
}

async function refreshList() {
  const d = await getJSON("/api/replay/replays");
  state.replays = d.replays;
  for (const id of [...state.on]) {
    if (!state.replays.some((r) => r.id === id)) {
      state.on.delete(id);
      hooks.viewer.removeOverlay(id);
      state.loaded.delete(id);
    }
  }
  renderList();
  for (const id of [...state.pending]) {
    if (state.replays.some((r) => r.id === id && r.state === "ready")) {
      state.pending.delete(id);
      await toggle(id, true, null);
      renderList();
    }
  }
}

// --- the readout ------------------------------------------------------------

let lastFrame = 0;

function paintReadout(frame) {
  lastFrame = frame;
  if (!ctx) return;
  const ids = [...state.on].filter((id) => state.loaded.has(id) && id !== "live");
  if (!ids.length) { ctx.readout.hidden = true; return; }
  ctx.readout.hidden = false;
  ctx.rows.innerHTML = "";
  for (const id of ids) {
    const { arrays } = state.loaded.get(id);
    const r = state.replays.find((x) => x.id === id);
    const row = document.createElement("div");
    row.className = "tcp-row";
    const valid = arrays.valid[frame];
    const l = arrays.errMm[frame * 2], rr = arrays.errMm[frame * 2 + 1];
    row.innerHTML =
      `<i class="dot" style="background:${css(state.colors.get(id))}"></i>` +
      `<span class="tcp-name">${esc(r ? r.label : id)}</span>` +
      (valid
        ? `<span class="tcp-val">${l.toFixed(1)} / ${rr.toFixed(1)} mm</span>` +
          `<span class="dim">chunk ${arrays.chunkOf[frame]} row ${arrays.rowOf[frame]}</span>`
        : `<span class="dim">no chunk here</span>`);
    ctx.rows.append(row);
  }
}

// --- the form ---------------------------------------------------------------

// Browse opens the DESKTOP's own file dialog, on the box the dashboard runs on
// (`/api/replay/pick`). Where that box has no display (the page is viewed over
// an SSH tunnel) it falls back to the folder list below.
async function browseFor(what) {
  state.picking = what;
  const field = $(what === "deploy" ? "rp-deploy" : "rp-ckpt").value.trim();
  const start = field || state.defaults?.starts?.[0]?.path || "";
  let d;
  try {
    d = await getJSON(`/api/replay/pick?what=${what}&start=${encodeURIComponent(start)}`);
  } catch { d = { unavailable: "no answer" }; }
  if (d.path) pick(d.path);
  else if (d.unavailable) openPicker(what);
}

// The fallback: one folder list, shared by the two fields. `state.picking`
// says which one it is filling ("deploy": click a .toml; "ckpt": click a
// step, or use the folder that is open).
function openPicker(what) {
  const box = $("rp-picker");
  state.picking = what;
  box.hidden = false;
  $("rp-picker-what").textContent = what === "deploy" ? "pick a deploy.toml" : "pick a checkpoint folder";
  $("rp-use").hidden = what !== "ckpt";
  renderStarts();
  const field = $(what === "deploy" ? "rp-deploy" : "rp-ckpt").value.trim();
  const from = field ? field.replace(/\/[^/]*$/, "") : state.defaults?.starts?.[0]?.path;
  if (from) browseTo(from);
}

function pick(path) {
  $("rp-picker").hidden = true;
  if (state.picking === "deploy") {
    $("rp-deploy").value = path;
    resolveDeploy(true);
  } else {
    $("rp-ckpt").value = path;
  }
  staleVerdict();
  summary();
}

function renderStarts() {
  const box = $("rp-starts");
  box.innerHTML = "";
  for (const s of state.defaults?.starts || []) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "rp-chip";
    b.textContent = s.path.split("/").filter(Boolean).slice(-2).join("/") || s.path;
    b.title = `${s.path}\n${s.why}`;
    b.addEventListener("click", () => browseTo(s.path));
    box.append(b);
  }
}

async function browseTo(path) {
  if (!path) return;
  try {
    const d = await getJSON(`/api/replay/browse?path=${encodeURIComponent(path)}`);
    state.listing = d;
    $("rp-dir").value = d.path;
    renderListing();
  } catch (err) {
    $("rp-browse").innerHTML = `<div class="rp-warn">${esc(err.message)}</div>`;
  }
}

function renderListing() {
  const d = state.listing;
  const box = $("rp-browse");
  box.innerHTML = "";
  if (!d) return;
  const row = (text, title, cls, fn) => {
    const el = document.createElement("div");
    el.className = cls;
    el.textContent = text;
    el.title = title;
    el.addEventListener("click", fn);
    box.append(el);
  };
  if (d.parent) row("..", d.parent, "rp-dir", () => browseTo(d.parent));
  if (state.picking === "deploy") {
    for (const f of d.files || []) row(f.name, f.path, "rp-dir rp-pick", () => pick(f.path));
  } else {
    // Numbered folders are a run's steps: newest first, one click selects.
    for (const s of [...d.steps].reverse()) {
      row(`step ${s.step}`, s.path, "rp-dir rp-pick", () => pick(s.path));
    }
  }
  for (const c of d.dirs) row(c.name + "/", c.path, "rp-dir", () => browseTo(c.path));
  if (state.picking === "deploy") {
    for (const s of d.steps) row(`${s.step}/`, s.path, "rp-dir", () => browseTo(s.path));
  }
  if (d.truncated) box.insertAdjacentHTML("beforeend", `<div class="rp-note">(more not shown)</div>`);
}

/** The deploy file in the field -> the policy entry and its Python. An empty
 * field means what the rollout itself ran with. */
async function resolveDeploy(takeCheckpoint = false) {
  const d = state.defaults;
  const path = $("rp-deploy").value.trim();
  const note = $("rp-class");
  state.entry = d?.entry || "";
  state.python = "";
  if (path) {
    try {
      const dep = await getJSON(`/api/replay/deploy?path=${encodeURIComponent(path)}`);
      state.entry = dep.entry;
      state.python = dep.python || "";
      if (takeCheckpoint && dep.checkpoint) $("rp-ckpt").value = dep.checkpoint;
    } catch (err) {
      state.entry = "";
      note.innerHTML = `<span class="rp-warn">${esc(err.message)}</span>`;
      $("rp-start").disabled = true;
      summary();
      return;
    }
  }
  await describeEntry();
  summary();
}

async function describeEntry() {
  const entry = state.entry;
  const note = $("rp-class");
  state.described = null;
  $("rp-start").disabled = Boolean(state.defaults?.gpu) || !entry;
  if (!entry) { note.textContent = "no policy named: pick a deploy file"; return; }
  note.textContent = "loading the policy adapter…";
  try {
    const q = `entry=${encodeURIComponent(entry)}&python=${encodeURIComponent(state.python || "")}`;
    const d = await getJSON(`/api/replay/describe?${q}`);
    state.described = d;
    note.title = d.class;
    note.innerHTML = `<b>${esc(String(d.class).split(".").pop())}</b> · ${esc(d.action_mode)} actions` +
      ` · anchor ${esc(d.anchor_camera)}` +
      (d.max_prompt_tokens ? ` · prompt ≤ ${d.max_prompt_tokens} tokens` : "");
  } catch (err) {
    note.innerHTML = `<span class="rp-warn">${esc(err.message)}</span>`;
    $("rp-start").disabled = true;
  }
}

function fixedSettings(s) {
  if (!s) return "";
  const sv = s.servo || {};
  return `fixed by the recording: ${s.sync ? "sync" : "async"} · P ${s.replan_period}` +
    ` · ${s.control_hz} Hz · ik ${esc(s.ik)} · servo filter ${esc(sv.filter_hz)} Hz.` +
    ` A replay runs the policy at the moments the rollout replanned, so these cannot change.`;
}

/** The request the form describes. */
function request() {
  const body = {
    entry: state.entry || "",
    prompt: $("rp-prompt").value,
    repeats: parseInt($("rp-repeats").value, 10) || 1,
    checkpoint: $("rp-ckpt").value.trim(),
    settings: {},
  };
  if (state.python) body.python = state.python;
  const steps = parseInt($("rp-steps").value, 10);
  if (Number.isFinite(steps)) body.settings.sample_steps = steps;
  return body;
}

/** The chip's second line: what differs from the rollout. */
function summary() {
  const d = state.defaults;
  if (!d) return;
  const b = request();
  const bits = [];
  if (b.entry && b.entry !== d.entry) bits.push("another policy");
  if (b.checkpoint !== (d.checkpoint || "")) {
    bits.push(b.checkpoint.split("/").filter(Boolean).slice(-2).join("/") || "no checkpoint");
  }
  if (b.prompt !== d.prompt) bits.push("another prompt");
  if (b.settings.sample_steps !== undefined && b.settings.sample_steps !== d.settings.sample_steps) {
    bits.push(`${b.settings.sample_steps} steps`);
  }
  if (b.repeats > 1) bits.push(`×${b.repeats}`);
  $("rp-sub").textContent = bits.length ? bits.join(" · ") : "the rollout's own settings";
}

function fillForm() {
  const d = state.defaults;
  $("rp-deploy").value = d.deploy || "";
  $("rp-ckpt").value = d.checkpoint || "";
  $("rp-prompt").value = d.prompt || "";
  $("rp-steps").value = d.settings?.sample_steps ?? "";
  $("rp-repeats").max = String(d.max_repeats || 20);
  $("rp-fixed").innerHTML = fixedSettings(d.settings);
  $("rp-gpu").textContent = d.gpu || "";
  $("rp-start").disabled = Boolean(d.gpu) || !d.entry;
  state.listing = null;
  state.entry = d.entry || "";
  state.python = "";
  $("rp-picker").hidden = true;
  $("rp-browse").innerHTML = "";
  staleVerdict();
  summary();
}

// --- the prompt check -------------------------------------------------------

// A little past the server's own limit (CHECK_PROMPT_TIMEOUT_S), so the
// server's timeout message, which says why, normally arrives first.
const CHECK_TIMEOUT_MS = 190_000;
let checkSeq = 0;

function verdict(text, kind = "") {
  const el = $("rp-verdict");
  if (!el) return;
  el.textContent = text;
  el.classList.toggle("rp-ok", kind === "ok");
  el.classList.toggle("rp-warn", kind === "bad");
  el.classList.toggle("rp-busy", kind === "busy");
}

/** Anything the verdict depends on changed: it no longer describes the form. */
function staleVerdict() {
  checkSeq += 1;
  $("rp-check").disabled = false;
  verdict("");
}

async function checkPrompt() {
  const b = request();
  const seq = ++checkSeq;
  const btn = $("rp-check");
  btn.disabled = true;
  verdict("checking…", "busy");
  const abort = new AbortController();
  const timer = setTimeout(() => abort.abort(), CHECK_TIMEOUT_MS);
  try {
    const d = await post("/api/replay/check-prompt", {
      entry: b.entry, prompt: b.prompt, python: b.python, checkpoint: b.checkpoint,
    }, abort.signal);
    if (seq !== checkSeq) return;
    verdict(d.message, d.ok ? "ok" : "bad");
  } catch (err) {
    if (seq !== checkSeq) return;
    verdict(err.name === "AbortError"
      ? `no answer in ${CHECK_TIMEOUT_MS / 1000} s — the policy took too long to construct`
      : err.message, "bad");
  } finally {
    clearTimeout(timer);
    if (seq === checkSeq) btn.disabled = false;
  }
}

function showCard(on) {
  $("rp-card").hidden = !on;
  $("rp-open").setAttribute("aria-expanded", String(Boolean(on)));
}

async function start() {
  const body = request();
  $("rp-start").disabled = true;
  try {
    // Settings first, so a bad value is refused at once with the live
    // dashboard's own check rather than a minute later by the replay.
    await post("/api/replay/settings", body.settings);
    const d = await post("/api/replay/start", body);
    job(`queued ${d.jobs.length} replay${d.jobs.length > 1 ? "s" : ""}…`);
    showCard(false);
    startPolling();
  } catch (err) {
    job(err.message, true);
  } finally {
    $("rp-start").disabled = Boolean(state.defaults?.gpu) || !state.entry;
  }
}

function startPolling() {
  if (state.poll) return;
  state.poll = setInterval(async () => {
    let d;
    try { d = await getJSON("/api/replay/jobs"); } catch { return; }
    const running = d.jobs.filter((j) => ["queued", "running"].includes(j.status));
    if (running.length) {
      const j = running.find((x) => x.status === "running") || running[0];
      const p = j.progress || {};
      const where = p.n_chunks ? ` ${p.chunk || 0}/${p.n_chunks}` : "";
      job(`${p.phase || j.status}${where} · ${j.elapsed_s}s` +
          (running.length > 1 ? ` · ${running.length - 1} queued` : ""));
      await refreshList();
      return;
    }
    clearInterval(state.poll);
    state.poll = null;
    const last = d.jobs[d.jobs.length - 1];
    if (last && last.status === "failed") {
      // The producer's last line: a refused prompt reads as the live one does.
      const why = (last.error || "").split("\n").filter((l) => l.startsWith("REFUSED:")).pop();
      job(why ? why.replace("REFUSED: ", "") : `failed — see ${last.log}`, true);
    } else if (last) {
      job(`done in ${last.elapsed_s}s`);
    }
    await refreshList();
  }, 1000);
}

// --- wiring -----------------------------------------------------------------

const onLoaded = async () => {
  state.loaded.clear();
  state.on.clear();
  state.colors.clear();
  state.strips.clear();
  paintStrip();
  hooks.viewer.clearOverlays();
  const url = new URLSearchParams(location.search);
  state.pending = new Set((url.get("replay") || "").split(",").filter(Boolean));
  job("");
  try {
    state.defaults = await getJSON("/api/replay/defaults");
    fillForm();
    await refreshList();
    resolveDeploy();
  } catch (err) {
    job(err.message, true);
  }
};
const onFrame = (f) => paintReadout(f);

export async function activate(c) {
  ctx = c;
  hooks = c.hooks;
  ctx.banner.innerHTML = BANNER;
  ctx.body.innerHTML = '<div id="rp-list" class="ci-solves"></div>';
  ctx.body.parentElement.querySelector("#mode-head").title = PANEL_TIP;
  ctx.readoutHead.textContent = "vs the flown chunk, this frame";
  $("rp-open").addEventListener("click", (e) => {
    e.stopPropagation();
    showCard($("rp-card").hidden);
  });
  $("rp-card").addEventListener("click", (e) => e.stopPropagation());
  document.addEventListener("click", closeCard);
  $("rp-deploy").addEventListener("change", () => resolveDeploy(true));
  $("rp-deploy-browse").addEventListener("click", () => browseFor("deploy"));
  $("rp-ckpt-browse").addEventListener("click", () => browseFor("ckpt"));
  $("rp-entry-reset").addEventListener("click", () => {
    fillForm();
    resolveDeploy();
  });
  $("rp-go").addEventListener("click", () => browseTo($("rp-dir").value.trim()));
  $("rp-dir").addEventListener("keydown", (e) => {
    if (e.key === "Enter") browseTo($("rp-dir").value.trim());
  });
  $("rp-use").addEventListener("click", () => {
    if (state.listing) pick(state.listing.path);
  });
  for (const id of ["rp-prompt", "rp-steps", "rp-repeats", "rp-ckpt"]) {
    $(id).addEventListener("input", summary);
    $(id).addEventListener("change", summary);
  }
  // What the prompt verdict depends on: the prompt, the policy, the checkpoint.
  for (const id of ["rp-prompt", "rp-deploy", "rp-ckpt"]) {
    $(id).addEventListener("input", staleVerdict);
    $(id).addEventListener("change", staleVerdict);
  }
  $("rp-check").addEventListener("click", checkPrompt);
  $("rp-start").addEventListener("click", start);
  hooks.onLoaded.push(onLoaded);
  hooks.onFrame.push(onFrame);
  // A replay may already be running when the mode is opened: the dashboard can
  // be reloaded while the GPU works.
  getJSON("/api/replay/jobs").then((d) => { if (d.busy) startPolling(); }).catch(() => {});
  if (hooks.state()) await onLoaded();
}

function closeCard() {
  const card = $("rp-card");
  if (card && !card.hidden) showCard(false);
}

export function deactivate() {
  if (state.poll) { clearInterval(state.poll); state.poll = null; }
  document.removeEventListener("click", closeCard);
  hooks.viewer.clearOverlays();
  state.loaded.clear();
  state.on.clear();
  state.colors.clear();
  state.strips.clear();
  state.replays = [];
  state.defaults = null;
  hooks.off(hooks.onLoaded, onLoaded);
  hooks.off(hooks.onFrame, onFrame);
  ctx = null;
  hooks = null;
}
