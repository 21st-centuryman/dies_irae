import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const HEIGHT_SERVER = 'http://localhost:8000';

// Downsample the 4096×4096 source to this resolution for the mesh.
// 512 → 511×511 quads (~520k triangles), smooth and fast.
const GRID = 512;

// Vertical exaggeration — makes elevation differences easier to read.
const HEIGHT_SCALE = 55;

// World-space size of the terrain plane (arbitrary units).
const WORLD_SIZE = 1000;

// ─── Colors ──────────────────────────────────────────────────────────────────
const _css = getComputedStyle(document.documentElement);
const COLOR_BG = _css.getPropertyValue('--container').trim();  // scene background, terrain rim fade
const COLOR_AMBIENT = 0xd8cfc4;  // ambient light
const COLOR_SUN = 0xfffaf0;  // directional sun light
const COLOR_FILL = 0xaabbdd;  // fill light (cool blue from opposite side)
const COLOR_CONTOUR = 0xffffff;  // contour line color
const COLOR_GROUND = 0xC2C1C5;

// Convert a hex color (number or CSS string like "#0D1117") to a GLSL vec3 literal.
function hexToVec3(hex) {
  const n = typeof hex === 'string' ? parseInt(hex.replace('#', ''), 16) : hex;
  const r = ((n >> 16) & 0xff) / 255;
  const g = ((n >> 8) & 0xff) / 255;
  const b = (n & 0xff) / 255;
  return `vec3(${r.toFixed(3)}, ${g.toFixed(3)}, ${b.toFixed(3)})`;
}

// ─── Renderer ────────────────────────────────────────────────────────────────

const renderer = new THREE.WebGLRenderer({
  antialias: true,
  canvas: document.getElementById('bg-canvas'),
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.9;

// ─── Scene ───────────────────────────────────────────────────────────────────

const scene = new THREE.Scene();
scene.background = new THREE.Color(COLOR_BG);

// ─── Camera + controls ───────────────────────────────────────────────────────

const camera = new THREE.PerspectiveCamera(
  55,
  window.innerWidth / window.innerHeight,
  0.5,
  8000
);
camera.position.set(0, 300, 700);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.minDistance = 10;
controls.maxDistance = 4000;
controls.maxPolarAngle = Math.PI * 0.52; // prevent going below ground

// ─── Lighting ────────────────────────────────────────────────────────────────

const ambient = new THREE.AmbientLight(COLOR_AMBIENT, 2.0);
scene.add(ambient);

const sun = new THREE.DirectionalLight(COLOR_SUN, 3.2);
sun.position.set(400, 700, 300);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.near = 1;
sun.shadow.camera.far = 3000;
sun.shadow.camera.left = -800;
sun.shadow.camera.right = 800;
sun.shadow.camera.top = 800;
sun.shadow.camera.bottom = -800;
sun.shadow.bias = -0.0005;
scene.add(sun);

const fill = new THREE.DirectionalLight(COLOR_FILL, 0.8);
fill.position.set(-300, 200, -400);
scene.add(fill);

// ─── Resize ──────────────────────────────────────────────────────────────────

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ─── Render loop ─────────────────────────────────────────────────────────────

(function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
})();

// ─── Float16 decode ──────────────────────────────────────────────────────────
// JavaScript has no native float16, so we unpack manually from the bit pattern.

function float16ToFloat32(h) {
  const s = (h >>> 15) & 1;
  const e = (h >>> 10) & 0x1f;
  const m = h & 0x3ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (m / 1024);
  if (e === 31) return m ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + m / 1024);
}

function decodeFloat16Buffer(buffer) {
  const u16 = new Uint16Array(buffer);
  const f32 = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) f32[i] = float16ToFloat32(u16[i]);
  return f32;
}

// ─── Contour shader ──────────────────────────────────────────────────────────
// Contour spacing in world-Y units. HEIGHT_SCALE/14 gives ~14 bands.
const CONTOUR_SPACING = HEIGHT_SCALE / 14;

const terrainVertexShader = /* glsl */`
  varying float vHeight;
  varying float vFade;
  varying vec3  vWorldNormal;

  void main() {
    vHeight      = position.y;
    vWorldNormal = normalize(mat3(modelMatrix) * normal);

    // Normalised XZ coords for edge-fade (same formula as CPU side).
    float nx   = (position.x / ${WORLD_SIZE.toFixed(1)}) * 2.0;
    float nz   = (position.z / ${WORLD_SIZE.toFixed(1)}) * 2.0;
    float dist = sqrt(nx * nx + nz * nz);
    float t    = clamp((dist - 0.82) / 0.20, 0.0, 1.0);
    vFade      = 1.0 - t * t * (3.0 - 2.0 * t);

    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const terrainFragmentShader = /* glsl */`
  varying float vHeight;
  varying float vFade;
  varying vec3  vWorldNormal;

  uniform float uContourSpacing;

  void main() {
    // Simple diffuse shading so the terrain still reads as 3-D.
    vec3  lightDir = normalize(vec3(0.45, 0.80, 0.35));
    float diffuse  = max(dot(normalize(vWorldNormal), lightDir), 0.0);
    float light    = 0.52 + diffuse * 0.48;

    // Flat grey terrain, shaded.
    vec3 terrainColor = ${hexToVec3(COLOR_GROUND)};

    // Contour lines: darken wherever height is near a multiple of spacing.
    float contour   = mod(vHeight, uContourSpacing);
    float lineWidth = fwidth(vHeight) * 1.5;
    float line      = 1.0 - smoothstep(0.0, lineWidth,
                        min(contour, uContourSpacing - contour));
    line *= vFade; // fade lines out at the rim too

    vec3 color = mix(terrainColor, ${hexToVec3(COLOR_CONTOUR)}, line);

    // Blend toward the scene background at the rim.
    vec3 bgColor = ${hexToVec3(COLOR_BG)};
    color = mix(bgColor, color, vFade);

    gl_FragColor = vec4(color, 1.0);
  }
`;

// ─── Terrain mesh ────────────────────────────────────────────────────────────

let terrain = null;

// Realistic Earth elevation bounds.  Anything outside these is a nodata
// sentinel (e.g. -9999, -32768, 0-fill) and must be discarded.
const ELEV_MIN = -500;   // below Dead Sea
const ELEV_MAX = 9000;   // above Everest

function isValidElev(v) {
  return isFinite(v) && !isNaN(v) && v > ELEV_MIN && v < ELEV_MAX;
}

function smoothstep(edge0, edge1, x) {
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

function buildTerrain(heights, srcW, srcH) {
  const stepX = srcW / GRID;
  const stepY = srcH / GRID;
  const samples = new Float32Array(GRID * GRID);

  // Box-filter downsample: average a small window around each grid point.
  // This removes the aliasing spikes that single-point sampling creates.
  const half = Math.max(1, Math.round(stepX / 4));

  let minH = Infinity, maxH = -Infinity;
  let validCount = 0;

  for (let gy = 0; gy < GRID; gy++) {
    for (let gx = 0; gx < GRID; gx++) {
      const cx = Math.round(gx * stepX);
      const cy = Math.round(gy * stepY);
      let sum = 0, n = 0;
      for (let dy = -half; dy <= half; dy++) {
        for (let dx = -half; dx <= half; dx++) {
          const px = Math.min(srcW - 1, Math.max(0, cx + dx));
          const py = Math.min(srcH - 1, Math.max(0, cy + dy));
          const v = heights[py * srcW + px];
          if (isValidElev(v)) { sum += v; n++; }
        }
      }
      const val = n > 0 ? sum / n : NaN;
      samples[gy * GRID + gx] = n > 0 ? val : NaN;
      if (n > 0) {
        validCount++;
        if (val < minH) minH = val;
        if (val > maxH) maxH = val;
      }
    }
  }

  const coverage = ((validCount / (GRID * GRID)) * 100).toFixed(1);
  console.log(
    `[terrain] ${GRID}×${GRID} grid: ${validCount} valid cells (${coverage}% coverage), ` +
    `elevation range: ${minH.toFixed(1)} – ${maxH.toFixed(1)} m`
  );

  for (let i = 0; i < samples.length; i++) {
    if (isNaN(samples[i])) samples[i] = minH;
  }

  const hRange = (maxH - minH) || 1;

  const geo = new THREE.PlaneGeometry(WORLD_SIZE, WORLD_SIZE, GRID - 1, GRID - 1);
  geo.rotateX(-Math.PI / 2);

  const pos = geo.attributes.position;

  for (let i = 0; i < pos.count; i++) {
    const gy = Math.floor(i / GRID);
    const gx = i % GRID;

    // Normalised coords: -1 → +1 in each axis.
    const nx = (gx / (GRID - 1)) * 2 - 1;
    const ny = (gy / (GRID - 1)) * 2 - 1;
    const dist = Math.sqrt(nx * nx + ny * ny);

    // fade = 1 inside the circle, smoothly falls to 0 at the edge.
    const fade = 1 - smoothstep(0.82, 1.02, dist);

    const raw = samples[i];
    pos.setY(i, ((raw - minH) / hRange) * HEIGHT_SCALE * fade);
  }

  geo.computeVertexNormals();

  const mat = new THREE.ShaderMaterial({
    uniforms: {
      uContourSpacing: { value: CONTOUR_SPACING },
    },
    vertexShader: terrainVertexShader,
    fragmentShader: terrainFragmentShader,
  });

  if (terrain) {
    scene.remove(terrain);
    terrain.geometry.dispose();
    terrain.material.dispose();
  }

  terrain = new THREE.Mesh(geo, mat);
  terrain.receiveShadow = true;
  terrain.castShadow = false;
  scene.add(terrain);

  // Reposition camera to frame the loaded terrain.
  const midY = HEIGHT_SCALE * 0.35;
  controls.target.set(0, midY, 0);
  camera.position.set(0, HEIGHT_SCALE * 2.2, WORLD_SIZE * 0.68);
  controls.update();

  return { coverage };
}

// ─── Fetch ───────────────────────────────────────────────────────────────────

async function fetchTerrain() {
  const lat = parseFloat(document.getElementById('lat').value);
  const lon = parseFloat(document.getElementById('lon').value);
  const btn = document.getElementById('sendMap');
  const status = document.getElementById('terrainStatus');

  btn.disabled = true;
  status.className = 'loading';
  status.textContent = 'fetching…';

  try {
    const res = await fetch(`${HEIGHT_SERVER}/fetch?lat=${lat}&lon=${lon}`);
    if (!res.ok) {
      const msg = await res.text();
      throw new Error(`server ${res.status}: ${msg}`);
    }

    const w = parseInt(res.headers.get('X-Width') || '4096', 10);
    const h = parseInt(res.headers.get('X-Height') || '4096', 10);

    status.textContent = 'decoding float16…';
    const buffer = await res.arrayBuffer();
    const heights = decodeFloat16Buffer(buffer);

    // Scan the raw decoded data so we can see sentinel values in the console.
    let rawValid = 0, rawMin = Infinity, rawMax = -Infinity;
    for (let i = 0; i < heights.length; i++) {
      const v = heights[i];
      if (isFinite(v) && !isNaN(v)) {
        rawValid++;
        if (v < rawMin) rawMin = v;
        if (v > rawMax) rawMax = v;
      }
    }
    console.log(
      `[terrain] raw ${w}×${h}: ${rawValid}/${w * h} finite cells ` +
      `(${(100 * rawValid / (w * h)).toFixed(1)}%), ` +
      `range: ${rawMin.toFixed(1)} – ${rawMax.toFixed(1)} m`
    );

    status.textContent = 'building mesh…';
    await new Promise(r => setTimeout(r, 0));
    const { coverage } = buildTerrain(heights, w, h);

    status.className = coverage < 50 ? 'loading' : 'ok';
    status.textContent = `${w}×${h} → ${GRID}×${GRID}  (${coverage}% coverage)`;
  } catch (err) {
    status.className = 'error';
    status.textContent = err.message;
    console.error('[terrain]', err);
  } finally {
    btn.disabled = false;
  }
}

document.addEventListener('fetchTerrain', fetchTerrain);
