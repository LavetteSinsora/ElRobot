// The latency mode's client: one fetch per loaded recording, two lines on the
// strip under the scrubber (`hooks.strip`, shared with the timeline's axis).

let hooks = null;

const BASE = {
  title: "inference latency",
  unit: "ms",
  help: "Per replan: trigger to chunk written (what the replan period must fit), and trigger to the anchor frame (the anchor's wait).",
};

async function draw() {
  if (!hooks) return;
  const r = await fetch("/api/latency/strip");
  if (!r.ok) {
    hooks.strip.show({ ...BASE, series: [], empty: r.status === 409 ? "load a recording" : `error ${r.status}` });
    return;
  }
  const d = await r.json();
  hooks.strip.show({
    ...BASE,
    series: [
      { label: "trigger → chunk written", color: "#e07b39", t: d.t, y: d.total_ms, width: 1.6 },
      { label: "trigger → anchor", color: "#3b82c4", t: d.t, y: d.anchor_ms },
    ],
  });
}

export async function activate(ctx) {
  hooks = ctx.hooks;
  hooks.onLoaded.push(draw);
  await draw();
}

export function deactivate() {
  if (hooks) {
    const i = hooks.onLoaded.indexOf(draw);
    if (i >= 0) hooks.onLoaded.splice(i, 1);
    hooks.strip.hide();
  }
  hooks = null;
}
