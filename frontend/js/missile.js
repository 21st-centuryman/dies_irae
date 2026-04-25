import * as THREE from "three";
import { Line2 } from "three/addons/lines/Line2.js";
import { LineGeometry } from "three/addons/lines/LineGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { scene, WORLD_SIZE, HEIGHT_SCALE, getTerrainElevRange } from "./3d.js";
import { makeMissileTexture } from "./svg.js";

// ─── Config ──────────────────────────────────────────────────────────────────

const WS_URL = `ws://${window.location.hostname}:4000/missiles/stream`;

const METRES_PER_UNIT = (5 * 2 * 1000) / WORLD_SIZE;
const SCALE    = 1 / METRES_PER_UNIT;
const TRAIL_LEN = 80;
const LINE_WIDTH = 2;

const MISSILE_W = 8;
const MISSILE_H = MISSILE_W * (158 / 174);

// ─── Binary wire format  (identical to drones.js) ────────────────────────────

const HEADER_SIZE = 16;
const RECORD_SIZE = 30;
const LE = true;

function decodeFrame(buf) {
  const view  = new DataView(buf);
  const count = view.getUint16(12, LE);
  const out   = new Array(count);
  for (let i = 0; i < count; i++) {
    const off = HEADER_SIZE + i * RECORD_SIZE;
    out[i] = {
      id: view.getUint16(off, LE),
      x:  view.getFloat32(off + 6,  LE),
      y:  view.getFloat32(off + 10, LE),
      z:  view.getFloat32(off + 14, LE),
    };
  }
  return out;
}

// ─── Scene root ──────────────────────────────────────────────────────────────

const missileRoot = new THREE.Group();
scene.add(missileRoot);

// ─── Missile store ───────────────────────────────────────────────────────────

const missiles = new Map();

// ─── Trail helpers ───────────────────────────────────────────────────────────

const resolution = new THREE.Vector2(window.innerWidth, window.innerHeight);
window.addEventListener("resize", () => resolution.set(window.innerWidth, window.innerHeight));

function makeTrailMaterial() {
  return new LineMaterial({
    color: 0x00aaff,   // blue — distinct from drone red trails
    linewidth: LINE_WIDTH,
    resolution,
  });
}

function buildTrailGeo(history) {
  const positions = new Float32Array(history.length * 3);
  for (let i = 0; i < history.length; i++) {
    positions[i * 3]     = history[i].x;
    positions[i * 3 + 1] = history[i].y;
    positions[i * 3 + 2] = history[i].z;
  }
  const geo = new LineGeometry();
  geo.setPositions(positions);
  return geo;
}

// ─── Missile lifecycle ───────────────────────────────────────────────────────

function toWorld(x, y, z) {
  const { minH, hRange } = getTerrainElevRange();
  return new THREE.Vector3(
    x * SCALE,
    ((z - minH) / hRange) * HEIGHT_SCALE,
    -y * SCALE,
  );
}

function createMissile(data) {
  const pos = toWorld(data.x, data.y, data.z);

  const tex    = makeMissileTexture(data.id);
  const mat    = new THREE.SpriteMaterial({ map: tex, depthTest: true });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(MISSILE_W, MISSILE_H, 1);
  sprite.position.copy(pos);
  missileRoot.add(sprite);

  const trailMat = makeTrailMaterial();
  const trail    = new Line2(buildTrailGeo([pos.clone(), pos.clone()]), trailMat);
  trail.computeLineDistances();
  missileRoot.add(trail);

  return { sprite, mat, tex, trail, trailMat, history: [pos.clone()] };
}

function updateMissile(missile, data) {
  const pos = toWorld(data.x, data.y, data.z);
  missile.sprite.position.copy(pos);

  missile.history.push(pos.clone());
  if (missile.history.length > TRAIL_LEN) missile.history.shift();

  if (missile.history.length >= 2) {
    const old          = missile.trail.geometry;
    missile.trail.geometry = buildTrailGeo(missile.history);
    missile.trail.computeLineDistances();
    old.dispose();
  }
}

function removeMissile(id) {
  const m = missiles.get(id);
  if (!m) return;
  missileRoot.remove(m.sprite);
  missileRoot.remove(m.trail);
  m.trail.geometry.dispose();
  m.trailMat.dispose();
  m.mat.dispose();
  m.tex.dispose();
  missiles.delete(id);
}

export function clearMissiles() {
  for (const id of [...missiles.keys()]) removeMissile(id);
}

// ─── WebSocket ───────────────────────────────────────────────────────────────

function handleMessage(ev) {
  if (typeof ev.data === "string") return;

  const list = decodeFrame(ev.data);
  const seen = new Set();

  for (const data of list) {
    seen.add(data.id);
    if (missiles.has(data.id)) {
      updateMissile(missiles.get(data.id), data);
    } else {
      missiles.set(data.id, createMissile(data));
    }
  }

  for (const id of missiles.keys()) {
    if (!seen.has(id)) removeMissile(id);
  }
}

function connect() {
  const ws = new WebSocket(WS_URL);
  ws.binaryType = "arraybuffer";

  ws.addEventListener("open",    () => console.log("[missiles] connected"));
  ws.addEventListener("message", handleMessage);
  ws.addEventListener("close",   () => setTimeout(connect, 2000));
  ws.addEventListener("error",   () => ws.close());
}

connect();
