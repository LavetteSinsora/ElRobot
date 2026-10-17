// The ElRobot dashboard: our URDF drawn three times (measured, commanded, a pose preview) and
// the session's state, polled from /api/state. Buttons only store a request; P0 decides.
import * as THREE from 'three';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const POLL_MS = 50;
const DEG = 180 / Math.PI;
const $ = (id) => document.getElementById(id);

// --- the scene (URDF frames are z-up; so is this scene) ------------------------------------

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
const canvas = $('scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0f1216);
const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 10);
camera.position.set(0.55, -0.35, 0.45);
const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 0.17, 0.12);
controls.update();
// Render only when something changed (a new state, the camera moved, a toggle): the Mac running
// the arm should not spend a core redrawing an unchanged scene 60 times a second.
let dirty = true;
const redraw = () => { dirty = true; };
controls.addEventListener('change', redraw);

scene.add(new THREE.HemisphereLight(0xffffff, 0x223344, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(0.5, -0.4, 1.0);
scene.add(sun);
const grid = new THREE.GridHelper(1.0, 20, 0x3a4652, 0x232c35);
grid.rotation.x = Math.PI / 2;                 // into the table plane (z = 0)
scene.add(grid);
const axes = new THREE.AxesHelper(0.06);       // base frame: x red, y green (forward), z blue
scene.add(axes);

function resize() {
  const r = canvas.parentElement.getBoundingClientRect();
  renderer.setSize(r.width, r.height, false);
  camera.aspect = r.width / Math.max(1, r.height);
  camera.updateProjectionMatrix();
  redraw();
}
window.addEventListener('resize', resize);

// --- the arm, from /api/robot ---------------------------------------------------------------

const euler = (rpy) => new THREE.Euler(rpy[0], rpy[1], rpy[2], 'ZYX');   // URDF fixed-axis rpy

class Arm {
  constructor(robot, geoms, material) {
    this.robot = robot;
    this.joints = {};
    const links = {};
    for (const [name, link] of Object.entries(robot.links)) {
      const g = new THREE.Group();
      for (const v of link.visuals) {
        const mat = typeof material === 'function' ? material(v.rgba) : material;
        const mesh = new THREE.Mesh(geoms[v.mesh], mat);
        mesh.position.fromArray(v.xyz);
        mesh.rotation.copy(euler(v.rpy));
        mesh.scale.fromArray(v.scale);
        g.add(mesh);
      }
      links[name] = g;
    }
    for (const j of robot.joints) {
      const frame = new THREE.Group();
      frame.position.fromArray(j.xyz);
      frame.rotation.copy(euler(j.rpy));
      const motion = new THREE.Group();
      frame.add(motion);
      motion.add(links[j.child]);
      links[j.parent].add(frame);
      this.joints[j.name] = { ...j, motion, axisV: new THREE.Vector3().fromArray(j.axis).normalize() };
    }
    this.root = links[robot.root];
    this.tcp = links[robot.tcp];
    scene.add(this.root);
  }

  set visible(v) { this.root.visible = v; }

  setQ(q) {                                    // q: 8 joint angles, servo order
    if (!q) return;
    const val = {};
    this.robot.arm_joints.forEach((name, i) => { val[name] = q[i] ?? 0; });
    for (const j of Object.values(this.joints)) {
      let x = val[j.name];
      if (j.mimic) x = j.mimic.multiplier * (val[j.mimic.joint] ?? 0) + j.mimic.offset;
      if (x === undefined) continue;
      if (j.type === 'revolute' || j.type === 'continuous') j.motion.quaternion.setFromAxisAngle(j.axisV, x);
      else if (j.type === 'prismatic') j.motion.position.copy(j.axisV).multiplyScalar(x);
    }
  }
}

function ghost(color, opacity) {
  return new THREE.MeshStandardMaterial({ color, transparent: true, opacity, depthWrite: false,
                                         roughness: 0.6, metalness: 0.0 });
}

const solidCache = new Map();
function solid(rgba) {
  // URDF colours: silver links, black servos (lifted a little so they read on a dark page).
  const key = rgba ? rgba.join(',') : 'none';
  if (!solidCache.has(key)) {
    const c = rgba ? new THREE.Color(Math.max(rgba[0], 0.16), Math.max(rgba[1], 0.17), Math.max(rgba[2], 0.19))
                   : new THREE.Color(0.75, 0.77, 0.8);
    solidCache.set(key, new THREE.MeshStandardMaterial({ color: c, roughness: 0.45, metalness: 0.25 }));
  }
  return solidCache.get(key);
}

const layers = {};
let robot = null;
let target = null;
let workspace = null;

async function loadArm() {
  robot = await (await fetch('/api/robot')).json();
  const loader = new STLLoader();
  const urls = [...new Set(Object.values(robot.links).flatMap((l) => l.visuals.map((v) => v.mesh)))];
  const geoms = {};
  await Promise.all(urls.map(async (u) => {
    const g = loader.parse(await (await fetch(u)).arrayBuffer());
    g.computeVertexNormals();
    geoms[u] = g;
  }));
  layers.measured = new Arm(robot, geoms, solid);
  layers.command = new Arm(robot, geoms, ghost(0x4c8dff, 0.38));
  layers.home = new Arm(robot, geoms, ghost(0x2fbf71, 0.30));
  layers.exit = new Arm(robot, geoms, ghost(0xf0a03c, 0.30));
  target = new THREE.Group();
  target.add(new THREE.Mesh(new THREE.SphereGeometry(0.006, 16, 12),
                            new THREE.MeshBasicMaterial({ color: 0xff4d6a })));
  target.add(new THREE.AxesHelper(0.03));
  scene.add(target);
  layers.target = { set visible(v) { target.visible = v && target.userData.valid; } };
  $('loading').remove();
  for (const box of document.querySelectorAll('#legend input')) {
    box.addEventListener('change', applyLayers);
  }
}

function applyLayers() {
  for (const box of document.querySelectorAll('#legend input')) {
    const layer = layers[box.dataset.layer];
    if (layer) layer.visible = box.checked;
    if (box.dataset.layer === 'workspace' && workspace) workspace.visible = box.checked;
  }
  redraw();
}

// --- the panel ------------------------------------------------------------------------------

let state = null;
let previewQ = null;
const TRACK = ['normal', 'limited', 'not available', 'unknown'];

function fmt(x, d = 1) { return x === null || x === undefined ? '—' : x.toFixed(d); }

function banner() {
  const s = state;
  if (!s.live) return ['wait', 'Preview: no robot connected.',
    'Green ghost: home, orange: exit. Drag the sliders to pose the solid arm.'];
  if (s.bus === 'HOLDING') return ['bad', `Bus fault: the arm is HELD where it is.`, s.fault];
  if (s.session === 'STOPPED') return ['', 'Session ended.', s.dry_run ? 'Dry run: torque was never on.'
    : 'If the exit ramp completed, torque is off; otherwise the arm is still held (see the terminal).'];
  if (s.session === 'EXITING') return ['wait', 'Exiting: returning to the exit pose…', ''];
  if (s.wait === 'HOME') return ['wait', 'The arm is held where it was found.',
    'The green ghost is the home pose. Check the path is clear, then press Home.'];
  if (s.session === 'HOME') return ['wait', 'Homing…', 'The arm ramps to the home pose.'];
  if (s.wait === 'START') return ['wait', 'Homed: the arm holds.', 'Press Start, then hold the pad on the phone to drive.'];
  if (s.session === 'ACTIVE') {
    if (s.phone && s.phone.engaged) return ['go', 'Driving: the arm follows the phone.',
      'Lift your thumb to hold. Slide it forward to grip.'];
    return ['go', 'Active: hold the pad on the phone to drive.', s.phone && !s.phone.fresh
      ? 'The phone is not streaming.' : 'The arm holds until you press.'];
  }
  if (s.session === 'CHECK' || s.session === 'INIT') return ['wait', 'Checking the hardware…',
    s.fault || 'Connecting to the servos, then waiting for the phone.'];
  return ['', s.session, ''];
}

function updatePanel() {
  const s = state;
  const mode = $('mode');
  if (!s.live) { mode.textContent = 'PREVIEW'; mode.className = 'badge'; }
  else if (s.dry_run) { mode.textContent = 'DRY RUN'; mode.className = 'badge dry'; }
  else if (s.driver === 'sim') { mode.textContent = 'SIM'; mode.className = 'badge sim'; }
  else { mode.textContent = 'LIVE ARM'; mode.className = 'badge live'; }
  $('session').textContent = s.live ? `${s.session}${s.bus ? ' · bus ' + s.bus : ''}` : 'no session';

  const [cls, title, sub] = banner();
  $('banner').className = `banner ${cls}`;
  $('banner').innerHTML = `${title}${sub ? `<small>${sub}</small>` : ''}`;

  const can = {
    HOME: s.live && s.wait === 'HOME',
    START: s.live && s.wait === 'START',
    RESET: s.live && s.session === 'ACTIVE',
    EXIT: s.live && !['EXITING', 'STOPPED'].includes(s.session),
  };
  for (const b of document.querySelectorAll('#buttons button')) {
    b.disabled = !can[b.dataset.request];
    b.classList.toggle('primary', b.dataset.request === 'HOME' ? can.HOME : b.dataset.request === 'START' && can.START);
  }

  // phone
  const p = s.phone;
  $('phone-card').hidden = !s.live;
  if (p) {
    const streaming = p.samples === 0 ? '<span class="badc">nothing received</span>'
      : p.fresh ? '<span class="ok">streaming</span>' : `<span class="warnc">stale (${fmt(p.age_s)} s)</span>`;
    $('phone').innerHTML = `
      <dt>link</dt><dd>${streaming}</dd>
      <dt>clutch</dt><dd>${p.engaged ? '<span class="ok">engaged</span>' : 'released'}</dd>
      <dt>tracking</dt><dd>${p.samples ? (p.track === 0 ? 'normal' : `<span class="warnc">${TRACK[p.track]}</span>`) : '—'}</dd>
      <dt>latency</dt><dd>${p.latency_ms === null ? '—' : `+${fmt(p.latency_ms)} ms over the best recent`}</dd>
      <dt>forced releases</dt><dd>${p.released}</dd>`;
  }

  // joints
  const rows = [];
  const m = s.measured, c = s.command;
  for (let i = 0; i < 8; i++) {
    const name = i === 7 ? 'grip' : `${i + 1}`;
    if (!s.live) {
      rows.push(`<tr><td>${name}</td><td>${fmt(previewQ[i] * DEG)}</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>`);
      continue;
    }
    const err = m && c ? (c[i] - m[i]) * DEG : null;
    const temp = s.temp ? s.temp[i] : null;
    const load = s.load ? s.load[i] : null;
    // In a dry run the arm is not driven, so a large error is the point, not a fault.
    const errCls = err === null || s.dry_run ? '' : Math.abs(err) > 5 ? 'bad' : Math.abs(err) > 2 ? 'warn' : '';
    const tCls = temp === null ? '' : temp >= 60 ? 'bad' : temp >= 50 ? 'warn' : '';
    rows.push(`<tr><td>${name}</td><td>${m ? fmt(m[i] * DEG) : '—'}</td><td>${c ? fmt(c[i] * DEG) : '—'}</td>
      <td class="${errCls}">${fmt(err)}</td><td class="${tCls}">${fmt(temp, 0)}</td>
      <td>${load === null ? '—' : fmt(load * 100, 0) + '%'}</td></tr>`);
  }
  $('joints').querySelector('tbody').innerHTML = rows.join('');
  $('ik').textContent = s.target_err_mm !== undefined && s.target_err_mm !== null
    ? `IK residual at the newest target: ${fmt(s.target_err_mm, 2)} mm` : '';

  // hardware check
  const rep = s.report;
  $('check-card').hidden = !s.live;
  if (rep) {
    const bad = rep.problems.length, warn = rep.warnings.length;
    $('check-summary').innerHTML = bad ? `<span class="badc">${bad} problem${bad > 1 ? 's' : ''}</span>`
      : `<span class="ok">passed</span>${warn ? `, <span class="warnc">${warn} warning(s)</span>` : ''}`;
    $('check').querySelector('tbody').innerHTML = rep.servos.map((r) => r.ok
      ? `<tr><td>${r.id}</td><td>${fmt(r.deg)}</td><td>${fmt(r.volts)} V</td><td>${fmt(r.temp, 0)}</td>
         <td>${r.p}/${r.d}</td><td class="${r.status.length ? 'bad' : ''}">${r.status.join(', ') || 'ok'}</td></tr>`
      : `<tr><td>${r.id}</td><td colspan="5" class="bad">no reply</td></tr>`).join('');
    $('check-notes').innerHTML = rep.problems.map((x) => `<li class="bad">${x}</li>`).join('')
      + rep.warnings.map((x) => `<li class="warn">${x}</li>`).join('');
    if (bad) $('check-card').open = true;
  } else {
    $('check-summary').textContent = s.live ? 'pending' : '';
  }
}

function updateScene() {
  const s = state;
  if (!layers.measured) return;
  if (!s.live) {
    layers.measured.setQ(previewQ);
    layers.command.visible = false;
  } else {
    layers.measured.setQ(s.measured);
    layers.command.setQ(s.command);
  }
  layers.home.setQ(s.home);
  layers.exit.setQ(s.exit);
  const tp = s.target_pose;
  target.userData.valid = !!(tp && tp[0] !== null && s.session === 'ACTIVE');
  if (target.userData.valid) {
    target.position.set(tp[0], tp[1], tp[2]);
    target.quaternion.set(tp[4], tp[5], tp[6], tp[3]);   // wxyz -> three's xyzw
  }
  if (!workspace && s.workspace) {
    const box = new THREE.Box3(new THREE.Vector3(...s.workspace[0]), new THREE.Vector3(...s.workspace[1]));
    workspace = new THREE.Box3Helper(box, 0x8b96a3);
    scene.add(workspace);
  }
  applyLayers();
  if (!s.live) layers.command.visible = false;
  redraw();
}

// --- the preview sliders (no robot) -----------------------------------------------------------

function buildSliders() {
  const box = $('sliders');
  box.innerHTML = '';
  for (let i = 0; i < 8; i++) {
    const lo = state.lower[i] * DEG, hi = state.upper[i] * DEG;
    const label = document.createElement('label');
    label.textContent = i === 7 ? 'g' : `${i + 1}`;
    const input = Object.assign(document.createElement('input'),
      { type: 'range', min: lo, max: hi, step: 0.5, value: previewQ[i] * DEG });
    const out = document.createElement('span');
    out.textContent = fmt(previewQ[i] * DEG);
    input.addEventListener('input', () => {
      previewQ[i] = input.value / DEG;
      out.textContent = fmt(+input.value);
      updateScene();
      updatePanel();
    });
    box.append(label, input, out);
  }
}

for (const b of document.querySelectorAll('#preview button[data-pose]')) {
  b.addEventListener('click', () => {
    const pose = b.dataset.pose;
    previewQ = pose === 'zero' ? state.lower.map((lo, i) => Math.min(Math.max(0, lo), state.upper[i]))
      : [...state[pose]];
    buildSliders();
    updateScene();
    updatePanel();
  });
}

// --- requests -----------------------------------------------------------------------------------

for (const b of document.querySelectorAll('#buttons button')) {
  b.addEventListener('click', async () => {
    const req = b.dataset.request;
    if (req === 'EXIT' && !confirm('End the session? The arm ramps to the exit pose.')) return;
    if (req === 'HOME' && !confirm('Move the arm to the home pose (the green ghost)?')) return;
    b.disabled = true;
    await fetch('/api/request', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                                   body: JSON.stringify({ request: req }) });
  });
}

// --- the loop -------------------------------------------------------------------------------------

let lastKey = '';
async function poll() {
  try {
    state = await (await fetch('/api/state', { cache: 'no-store' })).json();
    const key = JSON.stringify([state.measured, state.command, state.target_pose, state.session,
                                state.wait, state.bus, state.phone, state.report && state.report.problems]);
    const changed = key !== lastKey;
    lastKey = key;
    if (!state.live && previewQ === null) {
      previewQ = [...state.home];
      $('preview').hidden = false;
      $('legend').querySelector('[data-layer="command"]').parentElement.hidden = true;
      $('legend').querySelector('[data-layer="target"]').parentElement.hidden = true;
      $('legend').querySelector('[data-layer="exit"]').checked = true;
      buildSliders();
    }
    if (changed) {
      updatePanel();
      updateScene();
    }
  } catch (e) {
    $('banner').className = 'banner bad';
    $('banner').innerHTML = 'Lost the dashboard server.<small>The session may have ended.</small>';
  }
  setTimeout(poll, state && !state.live ? 500 : POLL_MS);
}

function frame() {
  if (dirty) {
    dirty = false;
    renderer.render(scene, camera);
  }
  requestAnimationFrame(frame);
}

resize();
frame();
loadArm().then(poll).catch((e) => { $('loading').textContent = `could not load the arm: ${e}`; });
