// The same URDF tree and local three.js assets as the live dashboard, with no live controls.
import * as THREE from 'three';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const euler = r => new THREE.Euler(...r, 'ZYX');
class Arm {
  constructor(robot, geometries, material) {
    this.robot = robot;
    this.joints = {};
    const links = {};
    for (const [name, link] of Object.entries(robot.links)) {
      const group = new THREE.Group();
      for (const v of link.visuals) {
        const mesh = new THREE.Mesh(geometries[v.mesh], material);
        mesh.position.fromArray(v.xyz); mesh.rotation.copy(euler(v.rpy)); mesh.scale.fromArray(v.scale);
        group.add(mesh);
      }
      links[name] = group;
    }
    for (const j of robot.joints) {
      const frame = new THREE.Group(), motion = new THREE.Group();
      frame.position.fromArray(j.xyz); frame.rotation.copy(euler(j.rpy));
      frame.add(motion); motion.add(links[j.child]); links[j.parent].add(frame);
      this.joints[j.name] = {...j, motion, axisV: new THREE.Vector3().fromArray(j.axis).normalize()};
    }
    this.root = links[robot.root];
  }
  pose(q) {
    if (!q || q.some(x => x === null)) { this.root.visible = false; return; }
    const values = Object.fromEntries(this.robot.arm_joints.map((n, i) => [n, q[i] ?? 0]));
    for (const j of Object.values(this.joints)) {
      const x = j.mimic ? j.mimic.multiplier * (values[j.mimic.joint] ?? 0) + j.mimic.offset : values[j.name];
      if (x === undefined) continue;
      if (j.type === 'revolute' || j.type === 'continuous') j.motion.quaternion.setFromAxisAngle(j.axisV, x);
      else if (j.type === 'prismatic') j.motion.position.copy(j.axisV).multiplyScalar(x);
    }
  }
}

export function createScene(canvas, message) {
  THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
  const renderer = new THREE.WebGLRenderer({canvas, antialias: true});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  const scene = new THREE.Scene(); scene.background = new THREE.Color(0x0c151c);
  const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 10);
  camera.position.set(.57, -.65, .42);
  const controls = new OrbitControls(camera, canvas); controls.target.set(.18, 0, .12); controls.update();
  scene.add(new THREE.HemisphereLight(0xffffff, 0x344452, 2));
  const sun = new THREE.DirectionalLight(0xffffff, 2); sun.position.set(.1, -.5, 1); scene.add(sun);
  const grid = new THREE.GridHelper(1, 20, 0x465761, 0x24343e); grid.rotation.x = Math.PI / 2; scene.add(grid);
  scene.add(new THREE.AxesHelper(.07));
  const marker = new THREE.Group(); marker.add(new THREE.AxesHelper(.055));
  marker.add(new THREE.Mesh(new THREE.SphereGeometry(.005), new THREE.MeshBasicMaterial({color:0xec7a9d})));
  scene.add(marker); marker.visible = false;
  let layers = {}, geometries = {}, generation = 0, dirty = true, sample = null;
  const material = (color, opacity) => new THREE.MeshStandardMaterial({color, roughness:.55,
    transparent:opacity < 1, opacity, depthWrite:opacity === 1});
  const materials = {measured:material(0xefad5e, 1), command:material(0x6da8ff,.35), target:material(0xb99afb,.25)};
  controls.addEventListener('change', () => { dirty = true; });
  new ResizeObserver(() => {
    const r = canvas.parentElement.getBoundingClientRect();
    renderer.setSize(r.width, r.height, false); camera.aspect = r.width / Math.max(r.height, 1);
    camera.updateProjectionMatrix(); dirty = true;
  }).observe(canvas.parentElement);
  function apply() {
    for (const box of document.querySelectorAll('[data-layer]')) {
      if (box.dataset.layer === 'pose') marker.visible = box.checked && !!sample?.pose;
      else if (layers[box.dataset.layer]) layers[box.dataset.layer].root.visible = box.checked && !!sample?.[box.dataset.layer];
    }
    dirty = true;
  }
  document.querySelectorAll('[data-layer]').forEach(b => b.addEventListener('change', apply));
  function update(s) {
    sample = s;
    for (const [name, arm] of Object.entries(layers)) arm.pose(s?.[name]);
    if (s?.pose && s.pose.every(x => x !== null)) {
      marker.position.fromArray(s.pose); marker.quaternion.set(s.pose[4],s.pose[5],s.pose[6],s.pose[3]);
    }
    apply();
  }
  function draw() { requestAnimationFrame(draw); if (dirty) { renderer.render(scene,camera); dirty = false; } }
  draw();
  return {update, async load(robot, id) {
    const token = ++generation;
    for (const arm of Object.values(layers)) scene.remove(arm.root);
    for (const g of Object.values(geometries)) g.dispose();
    layers = {}; geometries = {}; sample = null; marker.visible = false; dirty = true;
    if (!robot) { message.textContent = '3D model unavailable · camera playback remains available'; return; }
    message.textContent = 'Loading the recorded robot…';
    const loader = new STLLoader(), loaded = {};
    const urls = [...new Set(Object.values(robot.links).flatMap(l => l.visuals.map(v => v.mesh)))];
    try {
      await Promise.all(urls.map(async u => {
        const response = await fetch(`${u}?id=${encodeURIComponent(id)}`);
        if (!response.ok) throw new Error('A recorded mesh is missing');
        loaded[u] = loader.parse(await response.arrayBuffer()); loaded[u].computeVertexNormals();
      }));
      if (token !== generation) { Object.values(loaded).forEach(g => g.dispose()); return; }
      geometries = loaded;
      for (const name of Object.keys(materials)) {
        layers[name] = new Arm(robot,geometries,materials[name]); scene.add(layers[name].root);
      }
      message.textContent = ''; update(sample);
    } catch (e) { if (token === generation) message.textContent = e.message; }
  }};
}
