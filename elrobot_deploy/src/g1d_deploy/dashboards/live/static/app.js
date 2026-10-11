/* The live dashboard (docs/deploying_a_policy/running_and_live_dashboard.md,
 * "The live dashboard"); the server side is `dashboards/live/server.py`.
 *
 * This file is the page's controls: the banner, the commands, the prompt,
 * the Deploy settings and the status line.
 *
 *   1. Commands are POSTs. Nothing here tracks what it sent: the banner always
 *      renders the arena's own words, so a request that lost its race is visible
 *      as a chip that never lit rather than hidden behind an optimistic label.
 *   2. The banner and the status line are POLLED from /api/banner (4 Hz): the
 *      phase, the pending request, and the settings the current (or last)
 *      rollout runs with — solver, sync/async, P, rate, prompt.
 *   3. The prompt and the Deploy settings are POSTs too, and their
 *      state — requested vs active, P5's verdict on the prompt, P0's reason
 *      for refusing a Start — is POLLED from /api/settings, slower than the
 *      robot. Both take effect only at the next accepted Start.
 *   4. The stage — the cameras and the 3-D scene — is `live.js`'s. It gets
 *      `hooks` for the status message and the banner it paints from its own,
 *      faster, frame poll.
 */

import { start as startLive } from "/static/live.js";

const $ = (id) => document.getElementById(id);

let stopped = false;      // the session reached STOPPED; stop polling
let phase = "";           // the session word's name, from the last banner
let lastBanner = null;    // the last /api/banner (or /api/frame) banner

/* What the stage (`live.js`) may use of the shell. */
export const hooks = {
  say: (msg, bad = false) => say(msg, bad),
  banner: () => lastBanner,
  stopped: () => stopped,
  // The frame poll carries a fresher banner; it paints it through here, so
  // the chips keep its rate.
  paintBanner: (b) => paintBanner(b),
};

/* --- the banner ---------------------------------------------------------- */

function paintBanner(b) {
  $("phase").textContent = b.session_name;
  $("request").textContent = b.request_name;
  $("k-exec").textContent = b.k_exec;

  /* The phase drives the whole control surface. Start is only meaningful at
   * READY, Reset only during a rollout, and Exit whenever the session is still
   * alive — which is exactly what P0 will accept, so the buttons cannot offer a
   * press that would be consumed and ignored. */
  const name = b.session_name;
  phase = name;
  $("btn-start").disabled = name !== "READY";
  $("btn-reset").disabled = name !== "ACTIVE";
  $("btn-exit").disabled = name === "EXITING" || name === "STOPPED";

  /* The prompt and the settings change only BETWEEN rollouts: disabled while
   * ACTIVE, and once the session is over there is nothing left to set. P10
   * refuses them in those phases anyway (409); this only stops the press. */
  const locked = name === "ACTIVE" || name === "EXITING" || name === "STOPPED";
  $("prompt-input").disabled = locked;
  $("btn-settings").disabled = locked;
  for (const el of $("settings-form").elements)
    el.disabled = locked || (el.name === "ik" && !!settingsState?.joint_actions);

  document.body.dataset.phase = name;
  $("request").parentElement.classList.toggle("lit", b.request !== 0);

  $("ident-name").textContent = b.deployment || "—";
  $("ident-step").textContent = b.step ? `step ${b.step}` : "";
  const ck = $("ident-ckpt");
  ck.textContent = b.checkpoint ? b.checkpoint.split("/").slice(-2).join("/") : "";
  ck.title = b.checkpoint || "";

  paintStatusLine(b);
  lastBanner = b;
  bannerAt = performance.now();

  if (name === "STOPPED") {
    stopped = true;
    say("session stopped — this dashboard is now a still picture.");
  }
}

function say(msg, bad = false) {
  const el = $("status");
  el.textContent = msg;
  el.classList.toggle("bad", bad);
}

/* --- commands ------------------------------------------------------------ */

async function send(request) {
  try {
    const r = await fetch("/api/command", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ request }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || r.statusText);
    /* Deliberately NOT "resetting…". The banner says what the arena says, and
     * the next poll is 66 ms away — if the word did not land, the operator sees
     * that instead of a label claiming otherwise. */
    say(`sent ${request}`);
  } catch (e) {
    say(`could not send ${request}: ${e.message}`, true);
  }
}

/* --- the prompt --------------------------------------------------
 *
 * Enter submits. The server stores it and returns its sequence number; the
 * policy judges it in P5 (only it owns a tokenizer) and the verdict comes back
 * through /api/settings naming the request it judged. So the page waits for a
 * verdict whose `of` is ITS seq, and never mistakes the verdict on an older
 * prompt for this one's. */

let promptSeq = 0;        // the seq of the prompt this page last submitted

$("prompt-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const text = $("prompt-input").value;
  try {
    const r = await fetch("/api/prompt", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: text }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || r.statusText);
    promptSeq = d.seq;
    paintPromptState("checking", "the policy is checking it…");
  } catch (e) {
    paintPromptState("refused", e.message);
  }
});

function paintPromptState(kind, text) {
  const el = $("prompt-state");
  el.dataset.kind = kind;
  el.textContent = kind === "accepted" ? "accepted — used from the next Start"
    : kind === "refused" ? `refused: ${text}` : text;
}

/* --- the Deploy settings panel ----------------------------------- */

let settingsState = null; // the last /api/settings answer

// The IK solvers g1d_deploy ships (`runtime/ik` `SOLVERS`), as the panel names
// them. The server's `options.ik` decides what is offered; a name missing here
// is the launch's own `module:factory`.
const IK_LABELS = {
  arm_planner: "arm_planner — arm, branch planner (default)",
  arm_qp: "arm_qp — arm, per-tick QP",
  waist_planner: "waist_planner — + waist, planner (sync)",
  waist_qp: "waist_qp — + waist, whole-body QP (sync)",
};
// The async blends g1d_deploy ships (`chunk_blend`), as the panel names them;
// a name missing here is the launch's own `module:function`.
const BLEND_LABELS = {
  temporal_ensemble: "temporal_ensemble — ACT's weighted mean (default)",
  newest: "newest — the newest chunk only",
};
let refusalSeen = null;   // P0's refusal counter when this page first looked

/* The blend only matters in async (a sync tick has one chunk), so its row
 * shows only while the form says async. */
function showBlendRow() {
  $("blend-row").hidden = $("settings-form").elements["sync"].value !== "false";
}

function fmt(v) {
  if (typeof v === "boolean") return v ? "on" : "off";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
  return String(v);
}

function valueAt(obj, key) {
  return key.split(".").reduce((o, k) => (o == null ? undefined : o[k]), obj);
}

function show(key, v) {
  if (key === "sync") return v ? "sync" : "async";
  return fmt(v);
}

/* Fill the "new" column from the REQUEST — on opening the panel, after a
 * submit, and on "Revert" — never on a poll, which would overwrite what the
 * operator is typing. */
function fillForm(req) {
  const f = $("settings-form");
  f.elements["sync"].value = String(req.sync);
  for (const key of ["replan_period", "control_hz", "sample_steps",
                     "servo.filter_hz", "servo.velocity_ff", "servo.torque_ff",
                     "servo.arm_gain_scale"]) {
    f.elements[key].value = valueAt(req, key);
  }
  f.elements["ik"].value = req.ik;
  f.elements["blend"].value = req.blend;
  f.elements["servo.filter_lead"].checked = !!req.servo.filter_lead;
  showBlendRow();
}

function readForm() {
  const f = $("settings-form");
  const num = (k) => Number(f.elements[k].value);
  return {
    sync: f.elements["sync"].value === "true",
    replan_period: num("replan_period"),
    control_hz: num("control_hz"),
    ik: f.elements["ik"].value,
    blend: f.elements["blend"].value,
    sample_steps: num("sample_steps"),
    servo: {
      filter_hz: num("servo.filter_hz"),
      filter_lead: f.elements["servo.filter_lead"].checked,
      velocity_ff: num("servo.velocity_ff"),
      torque_ff: num("servo.torque_ff"),
      arm_gain_scale: num("servo.arm_gain_scale"),
    },
  };
}

function paintSettings(st) {
  const first = settingsState === null;
  settingsState = st;
  if (first) {
    const sel = $("settings-form").elements["ik"];
    sel.replaceChildren(...st.options.ik.map((m) => {
      const o = document.createElement("option");
      o.value = m; o.textContent = st.joint_actions ? "not used (joint actions)"
        : IK_LABELS[m] || `${m} (this launch's)`;
      return o;
    }));
    $("settings-form").elements["blend"].replaceChildren(...st.options.blend.map((m) => {
      const o = document.createElement("option");
      o.value = m; o.textContent = BLEND_LABELS[m] || `${m} (this launch's)`;
      return o;
    }));
    // Joint actions: P6 runs no IK. The value stays the request's (it must
    // still carry a valid one); only the choice is withdrawn.
    sel.disabled = !!st.joint_actions;
    const [lo, hi] = st.limits.replan_period;
    $("p-range").textContent = `${lo}…${hi}`;
    $("settings-form").elements["replan_period"].max = hi;
    $("prompt-input").maxLength = st.limits.prompt_bytes;
    fillForm(st.requested);
    refusalSeen = st.refusal.seq;
  }
  for (const row of document.querySelectorAll(".settings-table tr[data-key]")) {
    const key = row.dataset.key;
    const req = valueAt(st.requested, key);
    const act = valueAt(st.active, key);
    row.querySelector(".req").textContent = show(key, req);
    row.querySelector(".act").textContent = show(key, act);
    row.classList.toggle("pending", JSON.stringify(req) !== JSON.stringify(act));
  }
  $("btn-settings").classList.toggle(
    "pending", JSON.stringify(st.requested) !== JSON.stringify(st.active));

  /* The prompt's verdict: only once P5 has judged THIS page's request. */
  const v = st.prompt.verdict;
  if (promptSeq && v.of === promptSeq && v.name !== "pending") {
    paintPromptState(v.name, v.reason);
    promptSeq = 0;
  }

  /* A Start P0 refused: say why, once per refusal. */
  if (refusalSeen !== null && st.refusal.seq > refusalSeen) {
    refusalSeen = st.refusal.seq;
    say(`Start refused: ${st.refusal.text}`, true);
  }
}

async function pollSettings() {
  if (stopped) return;
  try {
    const r = await fetch("/api/settings");
    if (r.ok) paintSettings(await r.json());
  } catch (e) {
    /* The robot poll reports a lost link; this one stays quiet. */
  }
  setTimeout(pollSettings, SETTINGS_POLL_MS);
}

/* Slower than the robot: this is a form, not a picture. */
const SETTINGS_POLL_MS = 400;

$("btn-settings").onclick = () => {
  const panel = $("settings-panel");
  panel.hidden = !panel.hidden;
  $("btn-settings").setAttribute("aria-expanded", String(!panel.hidden));
  if (!panel.hidden && settingsState) fillForm(settingsState.requested);
};

$("settings-form").elements["sync"].addEventListener("change", showBlendRow);

$("settings-reload").onclick = () => {
  if (settingsState) fillForm(settingsState.requested);
  $("settings-msg").textContent = "";
};

$("settings-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const msg = $("settings-msg");
  try {
    const r = await fetch("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(readForm()),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || r.statusText);
    paintSettings(d);
    fillForm(d.requested);
    msg.dataset.kind = "ok";
    msg.textContent = "requested — used from the next Start";
    // Accepted: close the panel. The Deploy settings button stays marked
    // (`pending`) until a Start makes the request active.
    $("settings-panel").hidden = true;
    $("btn-settings").setAttribute("aria-expanded", "false");
    say("Deploy settings requested — used from the next Start.");
  } catch (e) {
    msg.dataset.kind = "bad";
    msg.textContent = `refused: ${e.message}`;
  }
});

$("btn-start").onclick = () => send("start");
$("btn-reset").onclick = () => send("reset");
$("btn-exit").onclick = () => send("exit");

/* --- the status line --------------------------------------------------------
 *
 * The ACTIVE block's settings (`banner.rollout`), never the request: what the
 * rollout on screen is actually running with. */

function paintStatusLine(b) {
  const r = b.rollout || {};
  $("sl-phase").textContent = b.session_name;
  $("sl-ik").textContent = r.ik ? `${r.ik} (${r.ik_mode})` : "—";
  $("sl-sync").textContent = r.sync === undefined ? "—" : r.sync ? "sync" : `async · ${r.blend}`;
  $("sl-p").textContent = r.replan_period ?? "—";
  $("sl-rate").textContent = r.control_hz ? `${fmt(r.control_hz)} Hz` : "—";
  $("sl-prompt").textContent = b.prompt || "—";
  $("sl-prompt").parentElement.title = `The prompt in use: ${b.prompt || "none"}`;
}

const BANNER_POLL_MS = 250;
let bannerAt = 0;          // when the last banner was painted, ms

async function pollBanner() {
  if (stopped) return;
  // Skipped when the frame poll has just painted one.
  if (performance.now() - bannerAt > BANNER_POLL_MS * 0.8) {
    try {
      const r = await fetch("/api/banner");
      if (r.ok) paintBanner(await r.json());
    } catch (e) {
      say(`lost the dashboard: ${e.message}`, true);
    }
  }
  setTimeout(pollBanner, BANNER_POLL_MS);
}

/* --- go ------------------------------------------------------------------ */

pollSettings();
pollBanner();
startLive(hooks);
