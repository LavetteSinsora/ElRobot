// The 3D pane: raw WebGL2, no dependency.
//
// The server has already done everything expensive. Each geom arrives as a
// static vertex block plus one 12-float transform per frame (3 position, 9
// rotation), so drawing frame N is: upload nothing, bind each geom's buffer
// range, push its matrix, draw. That is why scrubbing is free — there is no
// per-frame work proportional to anything but the geom count, which is 39.
//
// Deliberately plain shading: one key light, one fill, a rim term and a soft
// floor fade. Enough to read the robot's shape from any angle, and nothing that
// draws attention to itself.

const VERT = `#version 300 es
precision highp float;
layout(location=0) in vec3 a_pos;
layout(location=1) in vec3 a_normal;
uniform mat4 u_viewProj;
uniform mat3 u_rot;
uniform vec3 u_pos;
out vec3 v_normal;
out vec3 v_world;
void main() {
  vec3 world = u_rot * a_pos + u_pos;
  v_world = world;
  v_normal = normalize(u_rot * a_normal);
  gl_Position = u_viewProj * vec4(world, 1.0);
}`;

const FRAG = `#version 300 es
precision highp float;
in vec3 v_normal;
in vec3 v_world;
uniform vec3 u_color;
uniform vec3 u_eye;
uniform float u_alpha;
uniform vec3 u_rim;
out vec4 outColor;

const vec3 KEY  = normalize(vec3( 0.45,  0.75, 0.90));
const vec3 FILL = normalize(vec3(-0.70, -0.35, 0.35));

void main() {
  vec3 n = normalize(v_normal);
  vec3 view = normalize(u_eye - v_world);
  // Two-sided: STL winding is not always consistent and a flipped triangle
  // reading as unlit black is far more distracting than a slightly flat one.
  if (dot(n, view) < 0.0) n = -n;

  float key  = max(dot(n, KEY), 0.0);
  float fill = max(dot(n, FILL), 0.0) * 0.28;
  float rim  = pow(1.0 - max(dot(n, view), 0.0), 3.0) * 0.30;

  vec3 base = u_color;
  vec3 lit = base * (0.26 + key * 0.85 + fill) + vec3(0.55, 0.72, 0.85) * rim;

  // Very slight desaturation with distance, so far geometry recedes instead of
  // competing with the near arm.
  float d = clamp(length(u_eye - v_world) / 4.0, 0.0, 1.0);
  lit = mix(lit, vec3(0.09, 0.12, 0.16), d * 0.30);

  // An overlay is drawn at alpha < 1 AND with a rim boost, because a plain
  // translucent solid reads as a rendering glitch next to an opaque one. The
  // fresnel edge is what makes it look deliberately immaterial. The tint is a
  // uniform rather than a constant because there are two overlays now and they
  // are different claims — an edge that glowed amber on both would undo the
  // colour that tells them apart.
  if (u_alpha < 1.0) lit += u_rim * rim * 1.6;
  outColor = vec4(lit, u_alpha);
}`;

const GRID_VERT = `#version 300 es
precision highp float;
layout(location=0) in vec3 a_pos;
uniform mat4 u_viewProj;
out float v_fade;
void main() {
  v_fade = 1.0 - clamp(length(a_pos.xy) / 1.7, 0.0, 1.0);
  gl_Position = u_viewProj * vec4(a_pos, 1.0);
}`;

const GRID_FRAG = `#version 300 es
precision highp float;
in float v_fade;
uniform vec3 u_color;
out vec4 outColor;
void main() { outColor = vec4(u_color, v_fade * 0.5); }`;

// The TCP trails get their own program rather than borrowing one of the two
// above. The mesh shader wants normals a polyline does not have, and the grid's
// radial fade would dim a path simply for being far from the world origin,
// which is a statement about the grid and not about the trajectory.
const TRAIL_VERT = `#version 300 es
precision highp float;
layout(location=0) in vec3 a_pos;
layout(location=1) in vec3 a_tangent;
layout(location=2) in float a_side;
uniform mat4 u_viewProj;
uniform vec2 u_half;        // half the viewport, in device pixels
uniform float u_width;      // half the ribbon's width, in device pixels
void main() {
  vec4 clip = u_viewProj * vec4(a_pos, 1.0);
  vec4 ahead = u_viewProj * vec4(a_pos + a_tangent, 1.0);
  // Widened in SCREEN space rather than in world space. A trail is an
  // annotation on the scene, not a tube in it: it should read at the same
  // weight whether the hand is 0.3 m from the eye or 3 m, and a world-space
  // radius would make the far half of a reach thinner than the near half.
  vec2 s0 = clip.xy / clip.w * u_half;
  vec2 s1 = ahead.xy / ahead.w * u_half;
  vec2 d = s1 - s0;
  float len = length(d);
  // A step the projection has collapsed to a point has no direction to be
  // perpendicular to. Any fixed one will do; normalize(0) would hand back NaN
  // and take the whole strip with it.
  vec2 n = len > 1e-6 ? d / len : vec2(1.0, 0.0);
  clip.xy += vec2(-n.y, n.x) * a_side * u_width / u_half * clip.w;
  gl_Position = clip;
}`;

const TRAIL_FRAG = `#version 300 es
precision highp float;
uniform vec3 u_color;
uniform float u_alpha;
out vec4 outColor;
void main() { outColor = vec4(u_color, u_alpha); }`;

// ---------------------------------------------------------------- mat4 bits

function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0,
  ]);
}

function lookAt(eye, target, up) {
  const z = norm(sub(eye, target));
  const x = norm(cross(up, z));
  const y = cross(z, x);
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
  ]);
}

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1],
                         a[2] * b[0] - a[0] * b[2],
                         a[0] * b[1] - a[1] * b[0]];
function norm(a) {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
}
function mul(a, b) {                       // a * b, both column-major
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] +
                     a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
    }
  }
  return o;
}

function compile(gl, vsrc, fsrc) {
  const make = (type, src) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(s));
    }
    return s;
  };
  const p = gl.createProgram();
  gl.attachShader(p, make(gl.VERTEX_SHADER, vsrc));
  gl.attachShader(p, make(gl.FRAGMENT_SHADER, fsrc));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(p));
  }
  return p;
}

// ------------------------------------------------------------------ sphere

function sphere(radius, bands = 16) {
  const pos = [], nrm = [], idx = [];
  for (let y = 0; y <= bands; y++) {
    const phi = (y / bands) * Math.PI;
    for (let x = 0; x <= bands * 2; x++) {
      const theta = (x / (bands * 2)) * Math.PI * 2;
      const n = [Math.sin(phi) * Math.cos(theta), Math.sin(phi) * Math.sin(theta),
                 Math.cos(phi)];
      nrm.push(...n);
      pos.push(n[0] * radius, n[1] * radius, n[2] * radius);
    }
  }
  const row = bands * 2 + 1;
  for (let y = 0; y < bands; y++) {
    for (let x = 0; x < bands * 2; x++) {
      const a = y * row + x, b = a + row;
      idx.push(a, b, a + 1, b, b + 1, a + 1);
    }
  }
  return { pos: new Float32Array(pos), nrm: new Float32Array(nrm),
           idx: new Uint32Array(idx) };
}

// ------------------------------------------------------------------ colours

// Measured cyan, commanded amber — the same two hues the legend's dots use. The
// two HANDS are told apart by where they are rather than by colour, because
// four hues would need a legend of their own; a trail is the same colour as the
// marker whose path it is, which is what makes it readable without one.
const MEASURED_COLOR = [0.30, 0.86, 0.95];
const COMMANDED_COLOR = [1.0, 0.66, 0.18];
//: The solved configuration — the IK's OUTPUT, where the joint angles P6
//: published actually put the arm. Violet, because it has to be told apart from
//: both of the above at a glance: its distance from the amber ghost is the IK's
//: own error, and its distance from the cyan robot is everything after the IK.
const SOLVED_COLOR = [0.76, 0.49, 1.0];

//: Rim tints for the two translucent overlays, each a brightened form of the
//: overlay's own colour rather than one shared highlight.
const GHOST_RIM = [0.90, 0.55, 0.20];
const SOLVED_RIM = [0.62, 0.42, 1.0];

//: The whole-body overlays, one colour per tier. Deliberately far from amber,
//: violet and cyan — a tier has to be told apart from the commanded pose it is
//: chasing, from the deployed IK it is being compared against, and from the
//: robot that actually moved, all at once and often overlapping.
//:
//: The tiers nest (waist is inside waist+column is inside full), so two of them
//: on at once usually differ by millimetres and read as one thicker ghost. That
//: is a true picture and not a bug: where they separate is exactly where the
//: extra joint did something.
//: The three whole-body tiers keep the colours they have always had; any
//: other overlay name — the IK bench's methods — is coloured from `PALETTE`
//: in the order it arrives, by `wbikColour`. The legend and the readout in
//: `app.js` ask the same function, so a swatch and a robot cannot disagree.
const WBIK_COLORS = {
  waist: [0.36, 0.89, 0.50],
  waist_column: [1.00, 0.45, 0.72],
  full: [0.40, 0.62, 1.00],
};
//: Distinct against the measured grey, the amber ghost and the violet solved
//: overlay, and against each other at 30% alpha over a robot. Nothing amber or
//: violet: the first bench method is usually the baseline, and drawing it in
//: the commanded ghost's own hue made the two read as one object.
const PALETTE = [
  [0.25, 0.85, 0.90], [0.95, 0.35, 0.35], [0.80, 0.85, 0.30],
  [0.60, 0.80, 1.00], [0.30, 0.75, 0.55], [0.95, 0.55, 0.85],
  [0.99, 0.95, 0.85], [0.45, 0.55, 0.95],
];

/** The colour every drawer of overlay `name` uses. `index` is the overlay's
 *  position among the non-tier names, so the palette is walked in load order. */
export function wbikColour(name, index) {
  return WBIK_COLORS[name] || PALETTE[index % PALETTE.length];
}
function rimOf(c) { return c.map((v) => v * 0.8); }

//: The solved overlay's alpha. Lower than the ghost's: it covers the whole arm
//: rather than one gripper, and at the ghost's 0.42 a perfectly tracked rollout
//: would look like a robot with a film over it.
const SOLVED_ALPHA = 0.30;

//: The trail's width, in CSS pixels. `gl.lineWidth` is clamped to 1.0 on every
//: desktop driver and fails SILENTLY, so the path is expanded into a
//: screen-space ribbon by the shader above instead of asked for here.
const TRAIL_WIDTH_PX = 3.0;

//: How far along the tangent the vertex shader looks to find the screen
//: direction to widen across, in metres. Small enough that it cannot cross the
//: near plane while the point itself is on screen, large enough that the
//: subtraction does not vanish into float cancellation.
const TRAIL_TANGENT = 0.01;

//: One alpha, because only the travelled stretch is ever drawn — there is no
//: faint half left for it to be told apart from. High, because a trail this
//: thin still has to read against the lit geometry behind it.
const TRAIL_ALPHA = 0.92;

/** `[start, end)` spans of consecutive nonzero entries. */
function contiguous(flags, n) {
  const runs = [];
  let start = -1;
  for (let i = 0; i < n; i++) {
    if (flags[i]) { if (start < 0) start = i; }
    else if (start >= 0) { runs.push([start, i]); start = -1; }
  }
  if (start >= 0) runs.push([start, n]);
  return runs;
}

// ------------------------------------------------------------------ viewer

export class Viewer {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = canvas.getContext("webgl2", { antialias: true, alpha: false });
    if (!gl) throw new Error("WebGL2 is not available in this browser.");
    this.gl = gl;

    this.prog = compile(gl, VERT, FRAG);
    this.gridProg = compile(gl, GRID_VERT, GRID_FRAG);
    this.trailProg = compile(gl, TRAIL_VERT, TRAIL_FRAG);
    this.u = {
      viewProj: gl.getUniformLocation(this.prog, "u_viewProj"),
      rot: gl.getUniformLocation(this.prog, "u_rot"),
      pos: gl.getUniformLocation(this.prog, "u_pos"),
      color: gl.getUniformLocation(this.prog, "u_color"),
      eye: gl.getUniformLocation(this.prog, "u_eye"),
      alpha: gl.getUniformLocation(this.prog, "u_alpha"),
      rim: gl.getUniformLocation(this.prog, "u_rim"),
    };
    this.gu = {
      viewProj: gl.getUniformLocation(this.gridProg, "u_viewProj"),
      color: gl.getUniformLocation(this.gridProg, "u_color"),
    };
    this.tu = {
      viewProj: gl.getUniformLocation(this.trailProg, "u_viewProj"),
      color: gl.getUniformLocation(this.trailProg, "u_color"),
      alpha: gl.getUniformLocation(this.trailProg, "u_alpha"),
      half: gl.getUniformLocation(this.trailProg, "u_half"),
      width: gl.getUniformLocation(this.trailProg, "u_width"),
    };

    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);
    gl.clearColor(0.02, 0.031, 0.047, 1);

    // Orbit state. The G1-D stands ~1.2 m tall with its base on the floor at
    // z = 0; the target sits at torso height, which is where the work is.
    this.cam = { yaw: -0.9, pitch: 0.10, dist: 1.9, target: [0.10, 0, 0.78] };
    this.scene = null;
    this.frame = 0;
    this.tcpMarkers = true;
    this.tcpTrails = true;
    // One flag per SOURCE, and each of the three gates everything that source
    // draws — its geometry, its TCP marker and its trail together. That is what
    // makes "the trajectory follows whatever is on screen" a property of the
    // code rather than three lists someone has to keep in step: there is one
    // place to ask whether a source is showing, and every pass asks it.
    this.measured = true;
    this.ghosts = true;
    //: id -> one solved checkpoint's overlay, for `checkpoint_inspection`.
    //: Empty in every other viewer, and empty is the no-op.
    this.overlays = new Map();
    this.solvedOn = true;
    //: tier -> [transforms, tcp]; tier -> bool. Empty until a rollout with a
    //: `wbik_overlay` sidecar is loaded.
    this.wbik = {};
    this.wbikOn = {};
    this.wbikSlots = [];
    this.wbikValid = null;
    // The solved overlay's trunk. Defaults to the IK's own, which is the frame
    // the published arm angles were computed in and therefore the only one they
    // describe a real configuration in; `setSolved` drops back to the commanded
    // trunk on a record that does not carry the pinned one.
    this.solvedPinned = true;
    this._solvedAt = null;
    // Two polylines per source, one per hand. Built at load, drawn from static
    // buffers; nothing here is rebuilt on a scrub.
    this.trails = { measured: null, commanded: null, solved: null, wbik: {} };
    this.visible = new Set(["body", "gripper", "wheel", "column"]);

    this._buildGrid();
    this._buildSphere();
    this._bindInput();

    this._resize = this._resize.bind(this);
    window.addEventListener("resize", this._resize);
    new ResizeObserver(this._resize).observe(canvas);
    this._resize();
  }

  // --- static helpers ---------------------------------------------------

  _buildGrid() {
    const gl = this.gl, lines = [], n = 12, step = 0.25, half = n * step / 2;
    for (let i = 0; i <= n; i++) {
      const p = -half + i * step;
      lines.push(p, -half, 0, p, half, 0, -half, p, 0, half, p, 0);
    }
    this.grid = { vao: gl.createVertexArray(), count: lines.length / 3 };
    gl.bindVertexArray(this.grid.vao);
    const b = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(lines), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
  }

  _buildSphere() {
    // 8 mm radius. Small enough to read as a point rather than a part — it
    // marks a frame, and a fat ball would hide the jaws it sits between.
    const gl = this.gl, s = sphere(0.008);
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const pb = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, pb);
    gl.bufferData(gl.ARRAY_BUFFER, s.pos, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    const nb = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, nb);
    gl.bufferData(gl.ARRAY_BUFFER, s.nrm, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
    const ib = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, s.idx, gl.STATIC_DRAW);
    gl.bindVertexArray(null);
    this.sphere = { vao, count: s.idx.length };
  }

  /** One hand's TCP path, out of an `(n, 2, 3)` array, as its own buffer.
   *
   *  Vertex `i` IS frame `i` — the array is de-interleaved but never compacted,
   *  even where `runs` excludes a stretch. That identity is what lets the draw
   *  clip a run at the playhead with arithmetic instead of a search, and it is
   *  why the gaps are carried as ranges rather than by dropping vertices.
   */
  _makeTrail(src, hand, runs) {
    const gl = this.gl;
    const n = src.length / 6;
    // A run of one point is not a line. Dropped here so neither the tangents
    // below nor the draw loop has to know the case exists.
    runs = runs.filter(([a, b]) => b - a >= 2);

    // TWO vertices per point, offset to either side of the path by the vertex
    // shader. `a_tangent` is the direction to widen across, differenced
    // centrally inside a run and one-sided at its ends — the neighbours are
    // clamped to the RUN and not to the array, so a break is a real end and
    // the last point of a chunk is not widened across the gap after it.
    const V = new Float32Array(n * 2 * 7);
    const at = (f, k) => src[f * 6 + hand * 3 + k];
    for (const [a, b] of runs) {
      for (let f = a; f < b; f++) {
        const lo = Math.max(a, f - 1), hi = Math.min(b - 1, f + 1);
        const t = [at(hi, 0) - at(lo, 0), at(hi, 1) - at(lo, 1), at(hi, 2) - at(lo, 2)];
        const s = TRAIL_TANGENT / (Math.hypot(t[0], t[1], t[2]) || 1);
        for (let j = 0; j < 2; j++) {
          const o = (f * 2 + j) * 7;
          V[o] = at(f, 0); V[o + 1] = at(f, 1); V[o + 2] = at(f, 2);
          V[o + 3] = t[0] * s; V[o + 4] = t[1] * s; V[o + 5] = t[2] * s;
          V[o + 6] = j ? 1 : -1;
        }
      }
    }

    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, V, gl.STATIC_DRAW);
    for (const [loc, size, offset] of [[0, 3, 0], [1, 3, 12], [2, 1, 24]]) {
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 28, offset);
    }
    gl.bindVertexArray(null);
    // A run of one vertex is not a line, and LINE_STRIP would draw nothing for
    // it anyway. Dropping it here keeps the draw loop from having to know.
    return { vao, buf, runs };
  }

  /** Free one source's trails. Called before every rebuild, so a second Load
   *  does not leak the first rollout's buffers. */
  _dropTrails(which) {
    const gl = this.gl;
    for (const t of this.trails[which] || []) {
      gl.deleteVertexArray(t.vao);
      gl.deleteBuffer(t.buf);
    }
    this.trails[which] = null;
  }

  // --- data -------------------------------------------------------------

  /** Static geometry. `meta.geoms` slices the three concatenated buffers. */
  setGeometry(meta, vertices, normals, indices) {
    const gl = this.gl;
    if (this.scene) {
      for (const g of this.scene.geoms) gl.deleteVertexArray(g.vao);
    }
    const geoms = meta.geoms.map((g) => {
      const vao = gl.createVertexArray();
      gl.bindVertexArray(vao);

      const vb = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, vb);
      gl.bufferData(gl.ARRAY_BUFFER,
        vertices.subarray(g.vertex_offset * 3,
                          (g.vertex_offset + g.vertex_count) * 3),
        gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

      const nb = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, nb);
      gl.bufferData(gl.ARRAY_BUFFER,
        normals.subarray(g.vertex_offset * 3,
                         (g.vertex_offset + g.vertex_count) * 3),
        gl.STATIC_DRAW);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);

      const ib = gl.createBuffer();
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
      // Indices are local to each mesh's vertex block, which is exactly what a
      // per-geom VAO wants — no rebasing.
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,
        indices.subarray(g.index_offset, g.index_offset + g.index_count),
        gl.STATIC_DRAW);

      gl.bindVertexArray(null);
      const [r, gg, b] = g.rgba;
      return { vao, count: g.index_count, color: [r, gg, b], name: g.name,
               part: g.part };
    });
    this.scene = { geoms, nGeoms: meta.n_geoms };
  }

  /** Which `part` classes to draw. The legs are off by default — the G1-D has
   *  none, and the MJCF's are a stand-in posed at a hard zero. */
  setVisibleParts(parts) { this.visible = new Set(parts); this.draw(); }

  /** `(nFrames, nGeoms, 12)` transforms and `(nFrames, 2, 3)` TCP positions. */
  setTransforms(transforms, tcp, nFrames, nGeoms) {
    this.transforms = transforms;
    this.tcp = tcp;
    this.nFrames = nFrames;
    this.nGeoms = nGeoms;
    this.frame = 0;
    // The measured TCP exists at every frame — `state` is what the timeline is
    // built from — so its path is one unbroken run.
    this._dropTrails("measured");
    this.trails.measured = [0, 1].map((h) =>
      this._makeTrail(tcp, h, [[0, nFrames]]));
  }

  /** The commanded overlay: `(nFrames, 6, 12)` transforms for the geoms named
   *  by `slots`, `(nFrames, 2, 3)` TCPs, and a per-frame validity byte. */
  setGhost(transforms, tcp, valid, slots) {
    this.ghost = transforms;
    this.ghostTcp = tcp;
    this.ghostValid = valid;
    this.ghostSlots = slots;
    // Broken on `ghostValid`, and that is not cosmetic: `scene.py` leaves
    // `ghost_tcp` as ZEROS wherever no target was in force, so an unbroken
    // strip would run out to the world origin and back across every homing
    // ramp, every exit ramp and every async gap.
    this._dropTrails("commanded");
    this.trails.commanded = [0, 1].map((h) =>
      this._makeTrail(tcp, h, contiguous(valid, valid.length)));
  }

  /** The solved overlay: the robot posed at the joint angles P6 published.
   *
   *  `(nFrames, slots.length, 12)` transforms over a SUBSET of the same geom
   *  list `setGeometry` was given — every part a body target can move — plus
   *  the virtual TCP that configuration puts the wrists at, and a validity byte
   *  per frame.
   *
   *  It gets a trail like the other two. Where the IK is exact that path lies
   *  on the commanded one and the pair reads as a single thicker line, which is
   *  the correct picture — nothing was lost in the solve. Where they separate,
   *  the separation is the IK giving up, and on one recorded rollout the right arm
   *  parts from its own command by up to 100 mm.
   *
   *  Broken on `solvedValid` for the same reason the commanded trail is broken
   *  on `ghostValid`: `scene.py` leaves `solved_tcp` as ZEROS where no target
   *  was in force, and an unbroken strip would run out to the world origin.
   */
  setSolved(transforms, tcp, valid, slots, pinned = null, pinnedTcp = null) {
    this.solvedValid = valid;
    this.solvedSlots = slots;
    // The same overlay posed on two different trunks, over the same geoms in
    // the same order. `pinned` is empty on a record written before P6 stored
    // the waist it pinned the IK at, and null here is what greys the switch.
    this._solvedAt = {
      commanded: [transforms, tcp],
      pinned: pinned && pinned.length ? [pinned, pinnedTcp] : null,
    };
    this._useSolvedWaist(this.solvedPinned);
  }

  /** Which trunk the solved overlay stands on: the one the IK was PINNED at
   *  (`true`) or the one the tick COMMANDED (`false`).
   *
   *  The arm angles are identical either way — this moves the waist under them,
   *  and the whole upper body with it. Pinned is the configuration the robot was
   *  actually in when those angles were computed, so the overlay's torso lands
   *  on the measured torso and every remaining gap belongs to the arm. Commanded
   *  is what P7 was told to drive all seventeen joints to, waist included, which
   *  is the right picture when the question is about the command rather than
   *  about the solve — and the one a policy that drives the waist has to be read
   *  against, because there the two differ by the trunk's own tracking error
   *  rather than by a constant droop.
   *
   *  Falls back to commanded when there is no pinned array, so a stale `true`
   *  from a previous rollout cannot leave the overlay drawing nothing.
   */
  setSolvedWaist(pinned) {
    this._useSolvedWaist(pinned);
    this.draw();
  }

  _useSolvedWaist(pinned) {
    const src = this._solvedAt;
    this.solvedPinned = Boolean(pinned && src && src.pinned);
    if (!src) return;
    const [transforms, tcp] = this.solvedPinned ? src.pinned : src.commanded;
    this.solved = transforms;
    this.solvedTcp = tcp;
    // The trail is its marker's own path, so it is rebuilt rather than kept: the
    // pinned wrists travel a different curve from the commanded ones, and a
    // trail left over from the other variant would be a line under a marker that
    // never went there.
    this._dropTrails("solved");
    this.trails.solved = [0, 1].map((h) =>
      this._makeTrail(tcp, h,
                      contiguous(this.solvedValid, this.solvedValid.length)));
  }

  /** `trails.wbik` is a dict of tier -> [trail, trail], not the flat array
   *  `_dropTrails` walks, so it gets its own teardown. Sharing that helper
   *  would iterate the object's keys and delete nothing, leaking a VAO and a
   *  buffer per tier per Load.
   */
  _dropWbikTrails() {
    const gl = this.gl;
    for (const pair of Object.values(this.trails.wbik || {})) {
      for (const t of pair || []) { gl.deleteVertexArray(t.vao); gl.deleteBuffer(t.buf); }
    }
    this.trails.wbik = {};
  }

  /** The whole-body overlays: `{tier: [transforms, tcp]}` over `slots`.
   *
   *  Over EVERY geom, unlike the solved overlay's arm-and-waist subset, because
   *  two of these tiers move the lift column and one moves the base. The part
   *  filter still applies, so switching "base & column" off hides those pieces
   *  of these overlays too rather than leaving a coloured copy behind.
   */
  /** Colour of overlay `tier`, from its place among the loaded names. */
  _wbColour(tier) {
    const names = Object.keys(this.wbik).filter((n) => !WBIK_COLORS[n]);
    return wbikColour(tier, Math.max(0, names.indexOf(tier)));
  }

  setWbik(tiers, slots, valid) {
    this.wbik = tiers || {};
    this.wbikSlots = slots || [];
    this.wbikValid = valid || null;
    // Any tier the new rollout does not have stops being on. A stale `true`
    // here would draw nothing while its switch stayed lit.
    for (const k of Object.keys(this.wbikOn)) {
      if (!(k in this.wbik)) delete this.wbikOn[k];
    }
    this._dropWbikTrails();
    this.trails.wbik = {};
    for (const [tier, pair] of Object.entries(this.wbik)) {
      this.trails.wbik[tier] = [0, 1].map((h) =>
        this._makeTrail(pair[1], h,
                        contiguous(this.wbikValid, this.wbikValid.length)));
    }
    this.draw();
  }

  setWbikOn(tier, on) {
    if (on) this.wbikOn[tier] = true; else delete this.wbikOn[tier];
    this.draw();
  }

  setFrame(f) { this.frame = f; this.draw(); }

  // --- replay overlays (inspection mode `replay`) ------------------------
  //
  // N more commanded-pose ghosts, one per solved checkpoint, each in its own
  // colour with its own TCP trail — plus the part the live viewer has no reason
  // to draw: the WHOLE chunk in force at the playhead, as an arc leaving the
  // gripper. Purely additive. `elrobot_deploy/dashboards/live` shares this file and never
  // calls any of it, and an empty `overlays` map costs one `.size` test a frame.

  /** Add or replace one overlay; `spec`'s arrays come from `/api/overlay.bin`. */
  setOverlay(id, spec) {
    this.removeOverlay(id);
    const nc = spec.nChunks | 0, H = spec.H | 0;
    // Broken where the overlay is invalid, for the reason the commanded trail
    // is — and ALSO where the chunk changes, so two chunks either side of a
    // replan do not join into one line across the seam between them.
    const runs = [];
    let a = -1;
    for (let f = 0; f < spec.nFrames; f++) {
      const ok = spec.valid[f] && spec.chunkOf[f] >= 0;
      const same = a >= 0 && ok && spec.chunkOf[f] === spec.chunkOf[a];
      if (ok && a < 0) a = f;
      else if (a >= 0 && !same) { runs.push([a, f]); a = ok ? f : -1; }
    }
    if (a >= 0) runs.push([a, spec.nFrames]);

    // One buffer of `nChunks * H` points per hand: chunk `c` owns
    // `[c*H, c*H + arcN[c])`, split into the rows that would have executed and
    // the tail past them. The two overlap by a vertex so the ribbon joins.
    const arcRuns = [];
    const head = [], tail = [];
    for (let c = 0; c < nc; c++) {
      const n = spec.arcN[c] | 0;
      const e = Math.min(spec.arcExec[c] | 0, n);
      head.push([c * H, c * H + e]);
      tail.push([c * H + Math.max(0, e - 1), c * H + n]);
      arcRuns.push(head[c], tail[c]);
    }

    this.overlays.set(id, {
      color: spec.color || [1, 1, 1],
      ghost: spec.ghost, ghostTcp: spec.ghostTcp, valid: spec.valid,
      chunkOf: spec.chunkOf, rowOf: spec.rowOf, errMm: spec.errMm,
      slots: spec.slots, nFrames: spec.nFrames, H, nChunks: nc,
      head, tail, on: spec.on !== false,
      trails: [0, 1].map((hnd) => this._makeTrail(spec.ghostTcp, hnd, runs)),
      arcs: [0, 1].map((hnd) => this._makeTrail(spec.arc, hnd, arcRuns)),
    });
    this.draw();
  }

  removeOverlay(id) {
    const gl = this.gl;
    const e = this.overlays.get(id);
    if (!e) return;
    for (const t of [...e.trails, ...e.arcs]) {
      gl.deleteVertexArray(t.vao);
      gl.deleteBuffer(t.buf);
    }
    this.overlays.delete(id);
  }

  clearOverlays() {
    for (const id of [...this.overlays.keys()]) this.removeOverlay(id);
    this.draw();
  }

  setOverlayOn(id, on) {
    const e = this.overlays.get(id);
    if (!e) return;
    e.on = Boolean(on);
    this.draw();
  }

  /** One run, whole and unclipped — an arc is the playhead's FUTURE. */
  _drawRun(trail, color, alpha, span) {
    const gl = this.gl;
    if (!span || span[1] - span[0] < 2) return;
    gl.uniform3f(this.tu.color, color[0], color[1], color[2]);
    gl.uniform1f(this.tu.alpha, alpha);
    gl.bindVertexArray(trail.vao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 2 * span[0], 2 * (span[1] - span[0]));
  }

  /** Every enabled overlay: ghost meshes, then trails and arcs, then markers.
   *
   *  Drawn after everything else so its alpha blends against a finished frame,
   *  and in the same three passes the built-in overlays use so the two read as
   *  one picture rather than as a layer on top of one.
   */
  _drawOverlays(f, viewProj, w, h) {
    const gl = this.gl;
    const live = [...this.overlays.values()].filter((e) => e.on);
    if (!live.length) return;
    const rot = new Float32Array(9);

    gl.useProgram(this.prog);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.depthMask(false);
    gl.uniform1f(this.u.alpha, 0.42);
    for (const e of live) {
      if (!e.ghost || !e.valid[f]) continue;
      gl.uniform3f(this.u.rim, e.color[0], e.color[1], e.color[2]);
      gl.uniform3f(this.u.color, e.color[0], e.color[1], e.color[2]);
      const stride = e.slots.length * 12;
      e.slots.forEach((slot, j) => {
        const o = f * stride + j * 12;
        rot[0] = e.ghost[o + 3]; rot[1] = e.ghost[o + 6]; rot[2] = e.ghost[o + 9];
        rot[3] = e.ghost[o + 4]; rot[4] = e.ghost[o + 7]; rot[5] = e.ghost[o + 10];
        rot[6] = e.ghost[o + 5]; rot[7] = e.ghost[o + 8]; rot[8] = e.ghost[o + 11];
        gl.uniformMatrix3fv(this.u.rot, false, rot);
        gl.uniform3f(this.u.pos, e.ghost[o], e.ghost[o + 1], e.ghost[o + 2]);
        const g = this.scene.geoms[slot];
        gl.bindVertexArray(g.vao);
        gl.drawElements(gl.TRIANGLES, g.count, gl.UNSIGNED_INT, 0);
      });
    }
    gl.depthMask(true);
    gl.disable(gl.BLEND);
    gl.uniform1f(this.u.alpha, 1.0);

    if (this.tcpTrails && this.tcpMarkers) {
      gl.useProgram(this.trailProg);
      gl.uniformMatrix4fv(this.tu.viewProj, false, viewProj);
      const dpr = this.canvas.clientWidth ? w / this.canvas.clientWidth : 1;
      gl.uniform2f(this.tu.half, w / 2, h / 2);
      gl.uniform1f(this.tu.width, TRAIL_WIDTH_PX * dpr / 2);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.depthMask(false);
      gl.disable(gl.CULL_FACE);
      for (const e of live) {
        for (const t of e.trails) this._drawTrail(t, e.color, f);
        const c = e.chunkOf ? e.chunkOf[f] : -1;
        if (c < 0) continue;
        for (const t of e.arcs) {
          this._drawRun(t, e.color, 0.95, e.head[c]);
          this._drawRun(t, e.color, 0.30, e.tail[c]);
        }
      }
      gl.enable(gl.CULL_FACE);
      gl.depthMask(true);
      gl.disable(gl.BLEND);
      gl.useProgram(this.prog);
    }

    if (this.tcpMarkers && this.sphere) {
      const identity = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
      gl.disable(gl.DEPTH_TEST);
      gl.uniformMatrix3fv(this.u.rot, false, identity);
      gl.bindVertexArray(this.sphere.vao);
      for (const e of live) {
        if (!e.ghostTcp || !e.valid[f]) continue;
        gl.uniform3f(this.u.color, e.color[0], e.color[1], e.color[2]);
        for (let hnd = 0; hnd < 2; hnd++) {
          const o = f * 6 + hnd * 3;
          gl.uniform3f(this.u.pos, e.ghostTcp[o], e.ghostTcp[o + 1],
                       e.ghostTcp[o + 2]);
          gl.drawElements(gl.TRIANGLES, this.sphere.count, gl.UNSIGNED_INT, 0);
        }
      }
      gl.enable(gl.DEPTH_TEST);
    }
  }

  clear() {
    const gl = this.gl;
    if (this.scene) for (const g of this.scene.geoms) gl.deleteVertexArray(g.vao);
    this.scene = null;
    this.transforms = null;
    this.solved = null;
    this._solvedAt = null;
    this.wbik = {};
    this.wbikValid = null;
    this._dropWbikTrails();
    this._dropTrails("measured");
    this._dropTrails("commanded");
    this._dropTrails("solved");
    this.clearOverlays();
    this.draw();
  }

  // --- input ------------------------------------------------------------

  _bindInput() {
    const c = this.canvas;
    let dragging = false, panning = false, lx = 0, ly = 0;

    c.addEventListener("pointerdown", (e) => {
      dragging = true;
      panning = e.shiftKey || e.button === 1;
      lx = e.clientX; ly = e.clientY;
      c.setPointerCapture(e.pointerId);
    });
    c.addEventListener("pointerup", (e) => {
      dragging = false;
      c.releasePointerCapture(e.pointerId);
    });
    c.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      const dx = e.clientX - lx, dy = e.clientY - ly;
      lx = e.clientX; ly = e.clientY;
      if (panning) {
        // Pan in the camera's own screen plane, scaled by distance so the grab
        // point tracks the cursor at any zoom.
        const s = this.cam.dist * 0.0016;
        const yaw = this.cam.yaw;
        this.cam.target[0] += (-Math.cos(yaw) * dx + 0) * s;
        this.cam.target[1] += (-Math.sin(yaw) * dx) * s;
        this.cam.target[2] += dy * s;
      } else {
        this.cam.yaw -= dx * 0.007;
        this.cam.pitch = Math.max(-1.45, Math.min(1.45,
                                  this.cam.pitch + dy * 0.007));
      }
      this.draw();
    });
    c.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.cam.dist = Math.max(0.35, Math.min(8,
                        this.cam.dist * Math.exp(e.deltaY * 0.0012)));
      this.draw();
    }, { passive: false });
  }

  _resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    this.draw();
  }

  /** One polyline, clipped at the playhead. Assumes `trailProg` is bound.
   *
   *  Only the stretch already travelled is drawn. Carrying the rest at a low
   *  alpha showed where the hand was GOING — which is a fact about the
   *  recording and not about the instant you are parked on, and it read as the
   *  path being already decided. Clipped, a trail is the history behind the
   *  marker: it grows as you play and every frame shows only what had happened
   *  by then.
   */
  _drawTrail(trail, color, f) {
    const gl = this.gl;
    gl.uniform3f(this.tu.color, color[0], color[1], color[2]);
    gl.uniform1f(this.tu.alpha, TRAIL_ALPHA);
    gl.bindVertexArray(trail.vao);
    // Ranges double: point `i` owns ribbon vertices `2i` and `2i + 1`, so the
    // clip survives the widening as an index scale.
    for (const [a, b] of trail.runs) {
      // `f + 1`, because the playhead's own point belongs to the past: the
      // trail has to REACH the marker rather than stop one tick behind it. A
      // run still entirely ahead of the playhead gives `cut < a` and is
      // skipped by the same test that drops a one-point run.
      const cut = Math.min(b, f + 1);
      if (cut - a >= 2) gl.drawArrays(gl.TRIANGLE_STRIP, 2 * a, 2 * (cut - a));
    }
  }

  // --- draw -------------------------------------------------------------

  draw() {
    const gl = this.gl;
    const w = this.canvas.width, h = this.canvas.height;
    gl.viewport(0, 0, w, h);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    const { yaw, pitch, dist, target } = this.cam;
    // Z-up, matching MuJoCo's world.
    const eye = [
      target[0] + dist * Math.cos(pitch) * Math.cos(yaw),
      target[1] + dist * Math.cos(pitch) * Math.sin(yaw),
      target[2] + dist * Math.sin(pitch),
    ];
    const proj = perspective(0.85, w / h, 0.02, 60);
    const viewProj = mul(proj, lookAt(eye, target, [0, 0, 1]));

    // Floor grid first, so the robot always wins the depth test against it.
    gl.useProgram(this.gridProg);
    gl.uniformMatrix4fv(this.gu.viewProj, false, viewProj);
    gl.uniform3f(this.gu.color, 0.16, 0.23, 0.31);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.bindVertexArray(this.grid.vao);
    gl.drawArrays(gl.LINES, 0, this.grid.count);
    gl.disable(gl.BLEND);

    if (!this.scene || !this.transforms) { gl.bindVertexArray(null); return; }

    gl.useProgram(this.prog);
    gl.uniformMatrix4fv(this.u.viewProj, false, viewProj);
    gl.uniform3f(this.u.eye, eye[0], eye[1], eye[2]);

    const f = Math.max(0, Math.min(this.nFrames - 1, this.frame));
    const stride = this.nGeoms * 12;
    const rot = new Float32Array(9);

    gl.uniform1f(this.u.alpha, 1.0);
    // The measured robot is a source like the other two, and switchable like
    // them: with the arm hidden, the commanded pose and the configuration
    // solved for it can be compared against each other with nothing in the way.
    if (this.measured) this.scene.geoms.forEach((g, i) => {
      if (this.visible && !this.visible.has(g.part)) return;
      const o = f * stride + i * 12;
      // geom_xmat is row-major; the shader's mat3 is column-major, so this is
      // the transpose, written out rather than done in the shader.
      rot[0] = this.transforms[o + 3]; rot[1] = this.transforms[o + 6]; rot[2] = this.transforms[o + 9];
      rot[3] = this.transforms[o + 4]; rot[4] = this.transforms[o + 7]; rot[5] = this.transforms[o + 10];
      rot[6] = this.transforms[o + 5]; rot[7] = this.transforms[o + 8]; rot[8] = this.transforms[o + 11];
      gl.uniformMatrix3fv(this.u.rot, false, rot);
      gl.uniform3f(this.u.pos, this.transforms[o], this.transforms[o + 1],
                   this.transforms[o + 2]);
      gl.uniform3f(this.u.color, g.color[0], g.color[1], g.color[2]);
      gl.bindVertexArray(g.vao);
      gl.drawElements(gl.TRIANGLES, g.count, gl.UNSIGNED_INT, 0);
    });

    // The solved configuration — the whole robot at the joint angles P6
    // published. Where it coincides with the opaque arm beneath it, the servo
    // is doing what it was told; where it pulls away, it is not, and the visible
    // offset IS the tracking error.
    //
    // `polygonOffset` is what makes the coincident case readable. Two surfaces
    // at the same depth under the default `LESS` test resolve per fragment on
    // float noise, so a perfectly tracked arm would come out as speckle rather
    // than as a clean tint. Pulling the overlay a hair towards the eye makes it
    // win consistently, and the reading becomes "an even violet wash = tracking,
    // a violet arm beside a cyan one = not".
    if (this.solvedOn && this.solved && this.solvedValid[f]) {
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.enable(gl.POLYGON_OFFSET_FILL);
      gl.polygonOffset(-1.0, -1.0);
      gl.depthMask(false);
      gl.uniform1f(this.u.alpha, SOLVED_ALPHA);
      gl.uniform3f(this.u.rim, ...SOLVED_RIM);
      gl.uniform3f(this.u.color, ...SOLVED_COLOR);
      const sstride = this.solvedSlots.length * 12;
      this.solvedSlots.forEach((slot, j) => {
        const g = this.scene.geoms[slot];
        // The same part filter the opaque pass uses, so switching the base off
        // cannot leave a violet copy of it behind.
        if (this.visible && !this.visible.has(g.part)) return;
        const o = f * sstride + j * 12;
        rot[0] = this.solved[o + 3]; rot[1] = this.solved[o + 6]; rot[2] = this.solved[o + 9];
        rot[3] = this.solved[o + 4]; rot[4] = this.solved[o + 7]; rot[5] = this.solved[o + 10];
        rot[6] = this.solved[o + 5]; rot[7] = this.solved[o + 8]; rot[8] = this.solved[o + 11];
        gl.uniformMatrix3fv(this.u.rot, false, rot);
        gl.uniform3f(this.u.pos, this.solved[o], this.solved[o + 1], this.solved[o + 2]);
        gl.bindVertexArray(g.vao);
        gl.drawElements(gl.TRIANGLES, g.count, gl.UNSIGNED_INT, 0);
      });
      gl.depthMask(true);
      gl.disable(gl.POLYGON_OFFSET_FILL);
      gl.disable(gl.BLEND);
      gl.uniform1f(this.u.alpha, 1.0);
    }

    // The whole-body overlays, drawn exactly as the solved one is and after it,
    // so a tier that agrees with the deployed IK sits on top of it rather than
    // under it. Each is a full robot: same blend, same polygon offset, same
    // part filter.
    if (this.wbikValid && this.wbikValid[f]) {
      for (const [tier, pair] of Object.entries(this.wbik)) {
        if (!this.wbikOn[tier]) continue;
        const [xf] = pair;
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.enable(gl.POLYGON_OFFSET_FILL);
        gl.polygonOffset(-1.0, -1.0);
        gl.depthMask(false);
        gl.uniform1f(this.u.alpha, SOLVED_ALPHA);
        gl.uniform3f(this.u.rim, ...rimOf(this._wbColour(tier)));
        gl.uniform3f(this.u.color, ...this._wbColour(tier));
        const wstride = this.wbikSlots.length * 12;
        this.wbikSlots.forEach((slot, j) => {
          const g = this.scene.geoms[slot];
          if (this.visible && !this.visible.has(g.part)) return;
          const o = f * wstride + j * 12;
          rot[0] = xf[o + 3]; rot[1] = xf[o + 6]; rot[2] = xf[o + 9];
          rot[3] = xf[o + 4]; rot[4] = xf[o + 7]; rot[5] = xf[o + 10];
          rot[6] = xf[o + 5]; rot[7] = xf[o + 8]; rot[8] = xf[o + 11];
          gl.uniformMatrix3fv(this.u.rot, false, rot);
          gl.uniform3f(this.u.pos, xf[o], xf[o + 1], xf[o + 2]);
          gl.bindVertexArray(g.vao);
          gl.drawElements(gl.TRIANGLES, g.count, gl.UNSIGNED_INT, 0);
        });
        gl.depthMask(true);
        gl.disable(gl.POLYGON_OFFSET_FILL);
        gl.disable(gl.BLEND);
        gl.uniform1f(this.u.alpha, 1.0);
      }
    }

    // The commanded gripper — a Dex-1 with no robot behind it, at the pose the
    // policy asked for. Drawn AFTER the opaque pass so alpha blends against a
    // finished frame, and with depth writes off so its own three parts do not
    // occlude each other into a solid.
    //
    // `ghostValid[f] === 0` means no target was in force at this instant —
    // homing, warmup, the wait at READY, the exit ramp. It hides rather than
    // holding the last one, because a stale target drawn as a live one is
    // indistinguishable from a policy that stopped moving.
    if (this.ghosts && this.ghost && this.ghostValid[f]) {
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.depthMask(false);
      gl.uniform1f(this.u.alpha, 0.42);
      gl.uniform3f(this.u.rim, ...GHOST_RIM);
      gl.uniform3f(this.u.color, 0.95, 0.62, 0.22);
      const gstride = this.ghostSlots.length * 12;
      this.ghostSlots.forEach((slot, j) => {
        const o = f * gstride + j * 12;
        rot[0] = this.ghost[o + 3]; rot[1] = this.ghost[o + 6]; rot[2] = this.ghost[o + 9];
        rot[3] = this.ghost[o + 4]; rot[4] = this.ghost[o + 7]; rot[5] = this.ghost[o + 10];
        rot[6] = this.ghost[o + 5]; rot[7] = this.ghost[o + 8]; rot[8] = this.ghost[o + 11];
        gl.uniformMatrix3fv(this.u.rot, false, rot);
        gl.uniform3f(this.u.pos, this.ghost[o], this.ghost[o + 1], this.ghost[o + 2]);
        const g = this.scene.geoms[slot];
        gl.bindVertexArray(g.vao);
        gl.drawElements(gl.TRIANGLES, g.count, gl.UNSIGNED_INT, 0);
      });
      gl.depthMask(true);
      gl.disable(gl.BLEND);
      gl.uniform1f(this.u.alpha, 1.0);
    }

    // The paths those markers travel. One trail per marker that gets drawn —
    // the gating below is deliberately the same condition as the sphere pass
    // beneath it, so "the trajectory follows whatever gripper is shown" is a
    // property of the code and not of two lists kept in step by hand.
    //
    // Unlike the markers these ARE depth-tested, with writes off. A trail is a
    // path through space rather than a gizmo naming a frame, and one that
    // vanishes behind the torso it passed behind is telling you where it went;
    // drawn over everything, four full-length paths become a wireframe cage in
    // front of the robot.
    if (this.tcpTrails && this.tcpMarkers
        && (this.trails.measured || this.trails.commanded || this.trails.solved)) {
      gl.useProgram(this.trailProg);
      gl.uniformMatrix4fv(this.tu.viewProj, false, viewProj);
      // The ribbon is sized in DEVICE pixels, so it keeps the width the CSS
      // constant asks for on a retina panel instead of half of it.
      const dpr = this.canvas.clientWidth ? w / this.canvas.clientWidth : 1;
      gl.uniform2f(this.tu.half, w / 2, h / 2);
      gl.uniform1f(this.tu.width, TRAIL_WIDTH_PX * dpr / 2);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.depthMask(false);
      // A ribbon has no consistent winding — it flips as the path turns away
      // from the eye — so culling would eat half of every trail.
      gl.disable(gl.CULL_FACE);
      // One list per source, each behind that source's own flag — so a trail is
      // shown exactly when the marker it is the path of is. Turning the
      // commanded pose off takes its path with it, and turning everything but
      // the solved joints off leaves that one path alone on screen.
      //
      // A trail stays up on frames where its own validity byte is 0 and its
      // gripper is therefore hidden. Where a source went either side of a gap
      // is exactly what a gap makes you want to know, and the path is a
      // property of the rollout rather than of the instant you are parked on.
      for (const [on, list, colour] of [
        [this.measured, this.trails.measured, MEASURED_COLOR],
        [this.ghosts, this.trails.commanded, COMMANDED_COLOR],
        [this.solvedOn, this.trails.solved, SOLVED_COLOR],
        ...Object.entries(this.wbik).map(([tier]) =>
          [Boolean(this.wbikOn[tier]),
           (this.trails.wbik || {})[tier],
           this._wbColour(tier)]),
      ]) {
        if (!on || !list) continue;
        for (const t of list) this._drawTrail(t, colour, f);
      }
      gl.enable(gl.CULL_FACE);
      gl.depthMask(true);
      gl.disable(gl.BLEND);
      // Uniforms live on the program, so nothing set on `prog` above needs
      // re-uploading here.
      gl.useProgram(this.prog);
    }

    // The two virtual-TCP markers: the midpoint of each gripper's jaw tips at
    // full extent, 184.759 mm along the wrist frame's +X (the robot file's
    // `[end_effector] tcp`). Marking the flange instead would put the sphere
    // 18.5 cm behind the point the gripper actually acts at.
    //
    // Drawn with the depth test OFF, deliberately. The TCP sits between the
    // jaws and, from most angles, behind the gripper body — so a depth-tested
    // marker is occluded and the feature silently does nothing. These are
    // gizmos rather than geometry: their job is to say where a frame is, which
    // is a question about the model and not about what the robot looks like.
    if (this.tcpMarkers && this.tcp) {
      const identity = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
      gl.disable(gl.DEPTH_TEST);
      gl.uniformMatrix3fv(this.u.rot, false, identity);
      gl.bindVertexArray(this.sphere.vao);
      for (let hnd = 0; hnd < 2; hnd++) {
        const o = f * 6 + hnd * 3;
        if (this.measured) {
          gl.uniform3f(this.u.color, ...MEASURED_COLOR);
          gl.uniform3f(this.u.pos, this.tcp[o], this.tcp[o + 1], this.tcp[o + 2]);
          gl.drawElements(gl.TRIANGLES, this.sphere.count, gl.UNSIGNED_INT, 0);
        }

        if (this.ghosts && this.ghostTcp && this.ghostValid[f]) {
          gl.uniform3f(this.u.color, ...COMMANDED_COLOR);
          gl.uniform3f(this.u.pos, this.ghostTcp[o], this.ghostTcp[o + 1],
                       this.ghostTcp[o + 2]);
          gl.drawElements(gl.TRIANGLES, this.sphere.count, gl.UNSIGNED_INT, 0);
        }
        // The solved TCP sits between the other two, and the two gaps either
        // side of it are the whole diagnosis: amber-to-violet is how far the IK
        // fell short of the pose it was given, violet-to-cyan is how far the
        // arm then fell short of the angles it was given.
        if (this.solvedOn && this.solvedTcp && this.solvedValid[f]) {
          gl.uniform3f(this.u.color, ...SOLVED_COLOR);
          gl.uniform3f(this.u.pos, this.solvedTcp[o], this.solvedTcp[o + 1],
                       this.solvedTcp[o + 2]);
          gl.drawElements(gl.TRIANGLES, this.sphere.count, gl.UNSIGNED_INT, 0);
        }
        // One TCP per enabled tier. This is the marker the comparison is
        // actually about: the readout under the canvas is the distance from the
        // amber commanded TCP to each of these, never a flange distance.
        if (this.wbikValid && this.wbikValid[f]) {
          for (const [tier, pair] of Object.entries(this.wbik)) {
            if (!this.wbikOn[tier]) continue;
            const t = pair[1];
            gl.uniform3f(this.u.color, ...this._wbColour(tier));
            gl.uniform3f(this.u.pos, t[o], t[o + 1], t[o + 2]);
            gl.drawElements(gl.TRIANGLES, this.sphere.count, gl.UNSIGNED_INT, 0);
          }
        }
      }
      gl.enable(gl.DEPTH_TEST);
    }

    this._drawOverlays(f, viewProj, w, h);

    gl.bindVertexArray(null);
  }
}
