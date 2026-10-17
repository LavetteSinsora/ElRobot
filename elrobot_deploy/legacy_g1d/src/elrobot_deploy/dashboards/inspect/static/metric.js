// The second timeline: a thin strip under the scrubber that plots the active
// mode's per-tick metric over the WHOLE rollout (IK residual, servo tracking
// error, a replay's distance from the recorded actions).
//
// It is drawn on the main timeline's own axis — seconds from the window start,
// inset exactly like the replan marks above the slider (`#marks`), so a spike
// here sits directly under the instant the slider reaches it. Its playhead is
// the main one: `setCursor` is called on every frame, and a click or a drag
// here seeks the whole view through `onSeekTime`, the same path the marks use.
//
// A mode owns the content (`hooks.strip.show({...})`); the app hides it when
// the mode goes away and empties it when another recording loads. The drawing
// is the shared strip chart (`/static/strip.js`).

import { drawStrip } from "/static/strip.js";

const PAD = 13 / 2;               // half the slider thumb, as in marks.js

export class MetricStrip {
  constructor({ row, canvas, title, tools, legend, tip, axis, onSeekTime }) {
    Object.assign(this, { row, canvas, title, tools, legend, tip, axis, onSeekTime });
    this.spec = null;             // what the mode showed
    this.duration = 1;
    this.cursor = null;
    this.dragging = false;
    canvas.addEventListener("pointerdown", (e) => {
      if (!this.spec) return;
      this.dragging = true;
      canvas.setPointerCapture?.(e.pointerId);
      this.onSeekTime(this._timeAt(e));
    });
    canvas.addEventListener("pointermove", (e) => {
      if (!this.spec) return;
      if (this.dragging) this.onSeekTime(this._timeAt(e));
      this._hover(e);
    });
    const stop = () => { this.dragging = false; };
    canvas.addEventListener("pointerup", stop);
    canvas.addEventListener("pointercancel", stop);
    canvas.addEventListener("mouseleave", () => { this.tip.hidden = true; });
    new ResizeObserver(() => this.draw()).observe(canvas);
  }

  get visible() { return !this.row.hidden; }

  /** The rollout the axis spans. */
  setDuration(s) { this.duration = Math.max(1e-9, s); this.draw(); }

  setCursor(t) { this.cursor = t; this.draw(); }

  /** `{title, unit, series: [{label, color, t, y, dash, dots}], empty, tools}`.
   *  `t` in seconds from the window start, `y` with null for "no value". */
  show(spec) {
    this.spec = spec;
    this.title.textContent = spec.title || "";
    this.title.title = spec.help || "";
    this.tools.replaceChildren(...(spec.tools ? [spec.tools] : []));
    this.legend.innerHTML = (spec.series || []).map((s) =>
      `<span title="${esc(s.label)}"><i style="background:${s.color}"></i>${esc(s.label)}</span>`)
      .join("") || `<span class="dim">${esc(spec.empty || "")}</span>`;
    if (this.row.hidden) {
      this.row.hidden = false;
      document.body.classList.add("has-metric");
    }
    this.draw();
  }

  /** Keep the strip, drop its content (another recording is loading). */
  clear(msg = "") {
    if (!this.spec) return;
    this.show({ ...this.spec, series: [], empty: msg });
  }

  hide() {
    this.spec = null;
    this.tip.hidden = true;
    this.row.hidden = true;
    document.body.classList.remove("has-metric");
  }

  /** The canvas pixels the axis runs between, matched to the marks strip. */
  _span() {
    const W = this.canvas.clientWidth || 1;
    const c = this.canvas.getBoundingClientRect();
    const a = this.axis.getBoundingClientRect();
    const left = c.left + this.canvas.clientLeft;
    const x0 = a.left + PAD - left;
    const x1 = a.right - PAD - left;
    return { W, x0, x1: x1 > x0 + 1 ? x1 : W };
  }

  _timeAt(e) {
    const { x0, x1 } = this._span();
    const c = this.canvas.getBoundingClientRect();
    const x = e.clientX - c.left - this.canvas.clientLeft;
    return Math.max(0, Math.min(this.duration, ((x - x0) / (x1 - x0)) * this.duration));
  }

  draw() {
    if (!this.spec || this.row.hidden) return;
    const { W, x0, x1 } = this._span();
    // drawStrip maps [x0, x1] onto the whole canvas; widen the window so that
    // [0, duration] lands on the pixels under the slider's travel.
    const perPx = this.duration / (x1 - x0);
    const t0 = -x0 * perPx;
    const t1 = t0 + W * perPx;
    const series = (this.spec.series || []).map((s) => ({
      x: s.t, y: s.y, color: s.color, dash: s.dash || [], width: s.width ?? 1.3,
      dots: s.dots, alpha: s.alpha ?? 1,
    }));
    let hi = 0;
    for (const s of series) for (const v of s.y) if (v != null && v > hi) hi = v;
    const unit = this.spec.unit || "";
    drawStrip(this.canvas, {
      x0: t0, x1: t1, series, cursor: this.cursor,
      y: series.length && hi > 0 ? [0, hi * 1.12] : undefined,
      empty: this.spec.empty || "nothing to plot",
      label: series.length && hi > 0 ? `0 … ${fmt(hi)} ${unit}` : "",
    });
  }

  _hover(e) {
    const spec = this.spec;
    if (!spec || !(spec.series || []).length) { this.tip.hidden = true; return; }
    const t = this._timeAt(e);
    const rows = [];
    for (const s of spec.series) {
      const i = nearest(s.t, t);
      if (i < 0) continue;
      const v = s.y[i];
      rows.push(`<tr><td><i class="dot" style="background:${s.color}"></i>${esc(s.label)}</td>`
        + `<td>${v == null ? "–" : `${fmt(v)} ${esc(spec.unit || "")}`}</td></tr>`);
    }
    this.tip.innerHTML = `<b>${esc(spec.title || "")}</b> <span class="dim">+${t.toFixed(2)} s</span>`
      + `<table>${rows.join("")}</table>`;
    this.tip.hidden = false;
    const box = this.tip.getBoundingClientRect();
    const x = Math.min(window.innerWidth - box.width - 8, Math.max(8, e.clientX - box.width / 2));
    this.tip.style.left = `${x}px`;
    this.tip.style.top = `${Math.max(8, e.clientY - box.height - 14)}px`;
  }
}

/** Index of the sample nearest `t` in an ascending array (nulls skipped by the
 *  caller's data never being null in `t`), or -1 for an empty one. */
function nearest(ts, t) {
  let lo = 0, hi = ts.length - 1;
  if (hi < 0) return -1;
  if (t <= ts[0]) return 0;
  if (t >= ts[hi]) return hi;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (ts[mid] <= t) lo = mid; else hi = mid;
  }
  return (t - ts[lo]) <= (ts[hi] - t) ? lo : hi;
}

function fmt(v) {
  const a = Math.abs(v);
  return a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : a >= 1 ? v.toFixed(2) : v.toFixed(3);
}

function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
