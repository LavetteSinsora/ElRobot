// The replan marks: two glyphs per chunk on the scrubber, the ticks that chunk
// drove shaded underneath, and a zoomed band that makes the 90 ms between the
// two glyphs something you can actually look at.
//
// Two canvases, one model. The STRIP sits directly above the seek slider and
// spans the whole rollout; the BAND shows +/- 1.5 s around the playhead. Both
// draw the same list and share the hit-testing, so a mark means the same thing
// and clicks the same way in either.
//
// **Canvas rather than positioned elements, and the reason is async.** A sync
// rollout puts ~20 marks on the bar and would be perfectly happy as DOM. Async
// at P=5 puts one every 170 ms — several hundred glyphs, most of them a pixel
// or two apart, all wanting hover and click. As a canvas that is one element
// and one nearest-x search whatever the density; as DOM it is hundreds of nodes
// whose 6 px hitboxes overlap so badly that the one you click is decided by
// z-order. So: draw everything, hit-test by distance, no minimum separation
// anywhere — where marks collide, the collision IS the reading.
//
// **The strip is inset by half a thumb.** A range input's travel is not its
// width: the thumb's centre only reaches from `thumb/2` to `W - thumb/2`, so a
// mark drawn at a plain fraction of the width drifts away from the playhead
// towards both ends — worst exactly where the first and last replans are.

const THUMB = 13;                 // #seek's thumb, from style.css
const PAD = THUMB / 2;            // ...so the strip's usable travel matches it

//: How much of the rollout the zoom band shows, each side of the playhead. At
//: 1.5 s a 90 ms inference is ~30 px wide on a 1000 px band: the two glyphs
//: separate, the span between them reads as a duration, and roughly three sync
//: replans stay in view for context.
const BAND_HALF_S = 1.5;

const HIT_PX = 7;                 // how near a click must land to take a glyph

export class Marks {
  constructor({ strip, band, tip, onSeekFrame, onSeekTime }) {
    this.strip = strip;
    this.band = band;
    this.tip = tip;
    this.onSeekFrame = onSeekFrame;
    this.onSeekTime = onSeekTime;

    this.events = null;             // the server's `events` payload
    this.timeline = null;
    this.frame = 0;
    this.time = 0;
    // Hit targets, rebuilt on every draw: {x, y, t, frame, replan, kind}.
    this.hits = { strip: [], band: [] };
    // While dragging the band, its window is frozen at the instant the drag
    // started. Otherwise the band re-centres on the playhead the drag is
    // moving, the content slides out from under the cursor, and a drag to the
    // left runs away from you.
    this.dragLeft = null;

    const css = getComputedStyle(document.documentElement);
    const v = (n, fallback) => (css.getPropertyValue(n).trim() || fallback);
    this.col = {
      trigger: v("--warn", "#f0a04b"),
      written: v("--accent", "#4cc2ff"),
      ink: v("--ink", "#e6edf3"),
      dim: v("--ink-dim", "#8b98a8"),
      faint: v("--ink-faint", "#5b6673"),
      edge: v("--edge", "#232c38"),
      soft: v("--edge-soft", "#1b2330"),
      bg: v("--bg", "#0d1117"),
    };

    for (const [name, canvas] of [["strip", strip], ["band", band]]) {
      canvas.addEventListener("mousemove", (e) => this._hover(name, e));
      canvas.addEventListener("mouseleave", () => this._hideTip());
      canvas.addEventListener("pointerdown", (e) => this._down(name, e));
    }
    window.addEventListener("pointerup", () => { this.dragLeft = null; });
    window.addEventListener("pointermove", (e) => this._drag(e));

    this._ro = new ResizeObserver(() => this.draw());
    this._ro.observe(strip);
    this._ro.observe(band);
  }

  // ----------------------------------------------------------------- model

  setData(events, timeline) {
    this.events = events;
    this.timeline = timeline;
    this.draw();
  }

  setFrame(frame, time) {
    this.frame = frame;
    // `time` is passed in rather than derived: the display grid lands on the
    // commanded deadlines and fills the gaps, so it is not uniform and
    // `frame / fps` names the wrong instant everywhere after the first replan.
    // Everything drawn in this strip is in seconds, so a frame-derived time
    // would slide the playhead away from the marks it is supposed to line up
    // with.
    this.time = time !== undefined ? time
              : (this.timeline ? this.timeline.t_rel[frame] ?? 0 : 0);
    this.draw();
  }

  get replans() { return this.events ? this.events.replans : []; }
  get duration() { return this.timeline ? this.timeline.duration_s : 1; }

  /** The chunk whose coverage contains `t`, or null. Async can have several;
   *  the newest wins, which is the one whose actions P6 was weighting most. */
  covering(t) {
    let best = null;
    for (const r of this.replans) {
      if (r.cover && t >= r.cover[0] && t < r.cover[1]) {
        if (!best || r.id > best.id) best = r;
      }
    }
    return best;
  }

  /** Nearest mark before/after the playhead, for the `[` and `]` keys.
   *
   *  Stepped in FRAMES, not in seconds. Landing on a mark puts the playhead on
   *  the frame nearest it, which is up to half a frame away from the instant
   *  itself — so in seconds the mark you are standing on can still read as
   *  "before you", and `[` refuses to leave it. Frames are what the playhead
   *  actually has, and two marks inside one frame are one stop for the same
   *  reason they are one pixel. */
  step(dir) {
    let best = null;
    for (const r of this.replans) {
      for (const [kind, key] of [["trigger", "trigger_frame"], ["written", "written_frame"]]) {
        if (r[kind] < 0 || r[kind] > this.duration) continue;   // not drawn, not a stop
        const df = r[key] - this.frame;
        if (dir > 0 ? df > 0 : df < 0) {
          if (best === null || Math.abs(df) < Math.abs(best - this.frame)) best = r[key];
        }
      }
    }
    return best;
  }

  // ---------------------------------------------------------------- canvas

  _ctx(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx, w, h };
  }

  /** A triangle pointing down, apex on `y`. The two marks differ only in
   *  colour and fill: same shape, because they are the same kind of event. */
  _glyph(ctx, x, y, color, filled) {
    const w = 4.5, h = 6;
    ctx.beginPath();
    ctx.moveTo(x - w, y - h);
    ctx.lineTo(x + w, y - h);
    ctx.lineTo(x, y);
    ctx.closePath();
    if (filled) {
      ctx.fillStyle = color;
      ctx.fill();
    } else {
      ctx.fillStyle = this.col.bg;
      ctx.fill();
      ctx.lineWidth = 1.4;
      ctx.strokeStyle = color;
      ctx.stroke();
    }
  }

  draw() {
    this._drawStrip();
    this._drawBand();
  }

  // --- the strip: the whole rollout, aligned to the seek slider ------------

  _drawStrip() {
    const { ctx, w, h } = this._ctx(this.strip);
    this.hits.strip = [];
    if (!this.events) return;

    const span = Math.max(1, w - 2 * PAD);
    const X = (t) => PAD + (t / this.duration) * span;
    const lanes = Math.max(1, this.events.lanes);
    const laneH = Math.max(2, Math.min(5, Math.floor((h - 11) / lanes)));
    const barTop = h - lanes * laneH - 1;
    const apex = barTop - 2;

    // Coverage first, so glyphs sit on top of it rather than under.
    const now = this.covering(this.time);
    for (const r of this.replans) {
      if (!r.cover) continue;
      const x0 = Math.max(PAD, X(r.cover[0]));
      const x1 = Math.min(w - PAD, X(r.cover[1]));
      if (x1 <= x0) continue;
      const live = now && now.id === r.id;
      ctx.fillStyle = live ? this.col.written
        : (r.id % 2 ? "rgba(76,194,255,0.46)" : "rgba(76,194,255,0.26)");
      // A hairline gap so two chunks that share an endpoint still read as two.
      ctx.fillRect(x0, barTop + r.lane * laneH, Math.max(1, x1 - x0 - 0.5), laneH - 1);
    }

    for (const r of this.replans) {
      for (const kind of ["trigger", "written"]) {
        const t = r[kind];
        if (t < 0 || t > this.duration) continue;   // outside the window: clipped, not pinned
        const x = X(t);
        this._glyph(ctx, x, apex, kind === "trigger" ? this.col.trigger : this.col.written,
                    kind === "written");
        this.hits.strip.push({ x, y: apex - 3, t, kind, replan: r,
                               frame: r[kind === "trigger" ? "trigger_frame" : "written_frame"] });
      }
    }
  }

  // --- the band: +/- 1.5 s, where 90 ms is 30 px --------------------------

  _drawBand() {
    const { ctx, w, h } = this._ctx(this.band);
    this.hits.band = [];

    const left = this.dragLeft !== null ? this.dragLeft : this.time - BAND_HALF_S;
    const width = 2 * BAND_HALF_S;
    const X = (t) => ((t - left) / width) * w;
    const T = (x) => left + (x / w) * width;
    this._bandLeft = left;

    ctx.fillStyle = this.col.bg;
    ctx.fillRect(0, 0, w, h);

    if (!this.events) {
      ctx.fillStyle = this.col.faint;
      ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("load a rollout to see its replans", w / 2, h / 2 + 4);
      return;
    }

    // Anything outside the rollout is not empty timeline, it is not timeline.
    ctx.fillStyle = "rgba(0,0,0,0.35)";
    if (left < 0) ctx.fillRect(0, 0, X(0), h);
    if (left + width > this.duration) {
      const x = X(this.duration);
      ctx.fillRect(x, 0, w - x, h);
    }

    // Ruler: 100 ms ticks, labelled every 500 ms, in rollout time.
    ctx.strokeStyle = this.col.soft;
    ctx.fillStyle = this.col.faint;
    ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.textAlign = "center";
    ctx.lineWidth = 1;
    const first = Math.ceil(left * 10) / 10;
    for (let t = first; t < left + width; t += 0.1) {
      const x = Math.round(X(t)) + 0.5;
      const major = Math.abs(t * 2 - Math.round(t * 2)) < 1e-6;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, major ? 8 : 4);
      ctx.stroke();
      if (major) ctx.fillText(t.toFixed(1), x, 18);
    }

    const lanes = Math.max(1, this.events.lanes);
    const barTop = 24;
    const spanY = h - 11;           // where the trigger -> written spans live
    const laneH = Math.max(5, Math.min(13, Math.floor((spanY - 14 - barTop) / lanes)));

    const now = this.covering(this.time);
    for (const r of this.replans) {
      if (!r.cover) continue;
      const x0 = X(r.cover[0]), x1 = X(r.cover[1]);
      if (x1 < -20 || x0 > w + 20) continue;
      const live = now && now.id === r.id;
      const y = barTop + r.lane * laneH;
      ctx.fillStyle = live ? "rgba(76,194,255,0.42)"
        : (r.id % 2 ? "rgba(76,194,255,0.22)" : "rgba(76,194,255,0.13)");
      ctx.fillRect(x0, y, Math.max(1, x1 - x0 - 1), laneH - 2);
      ctx.strokeStyle = live ? this.col.written : "rgba(76,194,255,0.35)";
      ctx.strokeRect(Math.round(x0) + 0.5, Math.round(y) + 0.5,
                     Math.max(1, Math.round(x1 - x0 - 1)), laneH - 3);
      if (x1 - x0 > 46) {
        ctx.fillStyle = live ? this.col.ink : this.col.dim;
        ctx.textAlign = "left";
        ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
        ctx.fillText(`#${r.id}  k${r.k0}+${r.n}`, x0 + 4, y + laneH - 4);
      }

      // From the ring write to the first tick that write went on to drive. The
      // gap it spans is the thing the two glyphs exist to make visible: in sync
      // it is short and the arm is waiting inside it; in async it is a chunk
      // landing while an older one is still driving, which is the whole point
      // of the mode and is otherwise invisible.
      const xw = X(r.written);
      if (Math.abs(x0 - xw) < w) {
        ctx.save();
        ctx.setLineDash([2, 3]);
        ctx.strokeStyle = "rgba(76,194,255,0.45)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(xw, spanY - 8);
        ctx.lineTo(xw, y + laneH - 1);
        ctx.lineTo(x0, y + laneH - 1);
        ctx.stroke();
        ctx.restore();
      }
    }

    for (const r of this.replans) {
      const xa = X(r.trigger), xb = X(r.written);
      if (xb < -40 || xa > w + 40) continue;
      // The connector IS the latency here — at this zoom its width is the
      // 90 ms, drawn to scale rather than suggested.
      ctx.strokeStyle = "rgba(240,160,75,0.55)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(xa, spanY - 3);
      ctx.lineTo(xb, spanY - 3);
      ctx.stroke();
      this._glyph(ctx, xa, spanY, this.col.trigger, false);
      this._glyph(ctx, xb, spanY, this.col.written, true);
      this.hits.band.push({ x: xa, y: spanY - 3, t: r.trigger, kind: "trigger",
                            replan: r, frame: r.trigger_frame });
      this.hits.band.push({ x: xb, y: spanY - 3, t: r.written, kind: "written",
                            replan: r, frame: r.written_frame });
      if (xb - xa > 34) {
        ctx.fillStyle = this.col.dim;
        ctx.textAlign = "center";
        ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
        ctx.fillText(`${r.latency_ms.toFixed(0)} ms`, (xa + xb) / 2, spanY - 8);
      }
    }

    // The playhead last, over everything, because it is the one line whose
    // position the whole band is about.
    const px = Math.round(X(this.time)) + 0.5;
    ctx.strokeStyle = this.col.ink;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(px, 0);
    ctx.lineTo(px, h);
    ctx.stroke();
    ctx.fillStyle = this.col.ink;
    ctx.beginPath();
    ctx.moveTo(px - 4, 0);
    ctx.lineTo(px + 4, 0);
    ctx.lineTo(px, 5);
    ctx.closePath();
    ctx.fill();
  }

  // ------------------------------------------------------------ interaction

  _timeAt(name, e) {
    const canvas = name === "strip" ? this.strip : this.band;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    if (name === "strip") {
      const span = Math.max(1, rect.width - 2 * PAD);
      return { x, t: ((x - PAD) / span) * this.duration };
    }
    return { x, t: this._bandLeft + (x / rect.width) * 2 * BAND_HALF_S };
  }

  _nearest(name, x, y) {
    let best = null;
    for (const hit of this.hits[name]) {
      const d = Math.abs(hit.x - x) + Math.max(0, Math.abs(hit.y - y) - 8) * 0.5;
      if (d <= HIT_PX && (!best || d < best.d)) best = { ...hit, d };
    }
    return best;
  }

  _down(name, e) {
    if (!this.events) return;
    const canvas = name === "strip" ? this.strip : this.band;
    const rect = canvas.getBoundingClientRect();
    const hit = this._nearest(name, e.clientX - rect.left, e.clientY - rect.top);
    if (hit) { this.onSeekFrame(hit.frame); return; }
    if (name === "band") {
      // Freeze the window before the first seek, so the drag reads the band as
      // it looked when it started.
      this.dragLeft = this._bandLeft;
      canvas.setPointerCapture?.(e.pointerId);
    }
    const { t } = this._timeAt(name, e);
    this.onSeekTime(t);
  }

  _drag(e) {
    if (this.dragLeft === null || !this.events) return;
    const rect = this.band.getBoundingClientRect();
    const t = this.dragLeft + ((e.clientX - rect.left) / rect.width) * 2 * BAND_HALF_S;
    this.onSeekTime(t);
  }

  _hover(name, e) {
    if (!this.events) return;
    const canvas = name === "strip" ? this.strip : this.band;
    const rect = canvas.getBoundingClientRect();
    const hit = this._nearest(name, e.clientX - rect.left, e.clientY - rect.top);
    canvas.style.cursor = hit ? "pointer" : (name === "band" ? "ew-resize" : "pointer");
    if (!hit) return this._hideTip();
    this._showTip(hit, e);
  }

  _showTip(hit, e) {
    const r = hit.replan;
    const cover = r.cover
      ? `${r.cover[0].toFixed(2)}&ndash;${r.cover[1].toFixed(2)} s`
      : "never executed";
    this.tip.innerHTML = `
      <b>chunk ${r.id}</b> <span class="kind ${hit.kind}">${hit.kind}</span>
      <table>
        <tr><td>trigger</td><td>+${r.trigger.toFixed(3)} s</td><td>f${r.trigger_frame}</td></tr>
        <tr><td>written</td><td>+${r.written.toFixed(3)} s</td><td>f${r.written_frame}</td></tr>
        <tr><td>latency</td><td colspan="2">${r.latency_ms.toFixed(1)} ms</td></tr>
        <tr><td>ticks</td><td colspan="2">k${r.k0}&ndash;${r.k0 + r.n - 1} &middot; ${cover}</td></tr>
      </table>`;
    this.tip.hidden = false;
    const box = this.tip.getBoundingClientRect();
    const x = Math.min(window.innerWidth - box.width - 8, Math.max(8, e.clientX - box.width / 2));
    this.tip.style.left = `${x}px`;
    this.tip.style.top = `${Math.max(8, e.clientY - box.height - 14)}px`;
  }

  _hideTip() { this.tip.hidden = true; }
}
