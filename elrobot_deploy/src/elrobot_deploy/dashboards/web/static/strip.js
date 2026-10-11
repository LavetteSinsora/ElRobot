// A strip chart on a canvas: a few series over a time window, shared by the
// rollout viewer's executor chart and its timeline strip (`dashboards/inspect/static/metric.js`).
//
// One y-axis per chart, always: two quantities in different units are two
// charts. The caller hands arrays (`x`, `y`, with null for "no value", which
// breaks the line rather than bridging it); this fits the y-range to what is
// inside the x-window unless one is given, draws recessive grid lines, the
// series, vertical marks (knots, events), an optional cursor and a label.

/** Draw. Returns `{lo, hi}`, the y-range used.
 *
 *   x0, x1        the x-window
 *   series        [{x, y, color, dash = [], width = 1.6, alpha = 1}]
 *   marks         x positions of thin vertical rules (knots)
 *   spans         [{x0, x1, color}] shaded x-intervals (flagged stretches)
 *   hlines        [{y, color, dash}] horizontal rules (a threshold, zero)
 *   cursor        an x for a bright vertical rule (the playhead, "now")
 *   y             [lo, hi] to fix the range; else fitted, padded 10 %
 *   minSpan       the smallest fitted range (so noise is not magnified)
 *   label         text, bottom-left, or `(lo, hi) => text`
 *   empty         text when there is nothing to draw
 */
export function drawStrip(canvas, o) {
  const g = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth || 600;
  const H = canvas.clientHeight || 150;
  if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
  }
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  const { x0, x1 } = o;
  const series = (o.series || []).filter((s) => s && s.x && s.y);

  let lo = Infinity, hi = -Infinity;
  if (o.y) {
    [lo, hi] = o.y;
  } else {
    for (const s of series) {
      for (let i = 0; i < s.x.length; i++) {
        const v = s.y[i];
        if (v == null || s.x[i] < x0 || s.x[i] > x1) continue;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
    for (const h of o.hlines || []) { lo = Math.min(lo, h.y); hi = Math.max(hi, h.y); }
    if (!isFinite(lo)) {
      g.fillStyle = "rgba(255,255,255,.35)";
      g.font = "12px system-ui";
      g.fillText(o.empty || "nothing to draw", 10, 20);
      return { lo: null, hi: null };
    }
    const span = Math.max(hi - lo, o.minSpan || 0);
    const mid = (hi + lo) / 2;
    lo = mid - span / 2; hi = mid + span / 2;
    const pad = Math.max(span * 0.1, 1e-9);
    lo -= pad; hi += pad;
  }
  const X = (x) => ((x - x0) / (x1 - x0)) * W;
  const Y = (y) => H - ((y - lo) / (hi - lo)) * H;

  for (const sp of o.spans || []) {
    g.fillStyle = sp.color || "rgba(240,160,75,.18)";
    g.fillRect(X(sp.x0), 0, Math.max(1, X(sp.x1) - X(sp.x0)), H);
  }
  g.strokeStyle = "rgba(255,255,255,.12)";
  g.lineWidth = 1;
  for (const m of o.marks || []) {
    if (m < x0 || m > x1) continue;
    g.beginPath(); g.moveTo(X(m), 0); g.lineTo(X(m), H); g.stroke();
  }
  for (const h of o.hlines || []) {
    g.strokeStyle = h.color || "rgba(255,255,255,.25)";
    g.setLineDash(h.dash || [4, 4]);
    g.beginPath(); g.moveTo(0, Y(h.y)); g.lineTo(W, Y(h.y)); g.stroke();
    g.setLineDash([]);
  }
  for (const s of series) {
    g.setLineDash(s.dash || []);
    g.strokeStyle = s.color;
    g.globalAlpha = s.alpha ?? 1;
    g.lineWidth = s.width ?? 1.6;
    g.beginPath();
    let started = false;
    for (let i = 0; i < s.x.length; i++) {
      const x = s.x[i], v = s.y[i];
      if (x < x0 || x > x1 || v == null) { started = false; continue; }
      if (!started) { g.moveTo(X(x), Y(v)); started = true; } else g.lineTo(X(x), Y(v));
    }
    g.stroke();
    if (s.dots) {
      g.fillStyle = s.color;
      for (let i = 0; i < s.x.length; i++) {
        const x = s.x[i], v = s.y[i];
        if (x < x0 || x > x1 || v == null) continue;
        g.beginPath(); g.arc(X(x), Y(v), 2, 0, 2 * Math.PI); g.fill();
      }
    }
    g.setLineDash([]);
    g.globalAlpha = 1;
  }
  if (o.cursor != null) {
    g.strokeStyle = "rgba(255,255,255,.7)";
    g.beginPath(); g.moveTo(X(o.cursor), 0); g.lineTo(X(o.cursor), H); g.stroke();
  }
  const label = typeof o.label === "function" ? o.label(lo, hi) : o.label;
  if (label) {
    g.fillStyle = "rgba(255,255,255,.55)";
    g.font = "11px system-ui";
    g.fillText(label, 6, H - 6);
  }
  return { lo, hi };
}
