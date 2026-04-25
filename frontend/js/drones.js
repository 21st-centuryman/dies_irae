import * as THREE from "three";
import { Line2 } from "three/addons/lines/Line2.js";
import { LineGeometry } from "three/addons/lines/LineGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { scene, WORLD_SIZE, HEIGHT_SCALE, getTerrainElevRange } from "./3d.js";

// ─── Config ──────────────────────────────────────────────────────────────────

const WS_URL = `ws://localhost:3000/simulation/fake/stream`;

const METRES_PER_UNIT = (5 * 2 * 1000) / WORLD_SIZE; // 10 km diameter map → 10 m per world unit
const SCALE = 1 / METRES_PER_UNIT;
const TRAIL_LEN = 600;
const LINE_WIDTH = 2; // px

// ─── Binary wire format ───────────────────────────────────────────────────────

const HEADER_SIZE = 16;
const RECORD_SIZE = 30;
const LE = true;

function decodeFrame(buf) {
  const view = new DataView(buf);
  const count = view.getUint16(12, LE);
  const out = new Array(count);
  for (let i = 0; i < count; i++) {
    const off = HEADER_SIZE + i * RECORD_SIZE;
    out[i] = {
      id: view.getUint16(off, LE),
      x: view.getFloat32(off + 6, LE),
      y: view.getFloat32(off + 10, LE),
      z: view.getFloat32(off + 14, LE),
    };
  }
  return out;
}

// ─── Drone root ──────────────────────────────────────────────────────────────

const droneRoot = new THREE.Group();
scene.add(droneRoot);

// ─── Drone store ─────────────────────────────────────────────────────────────

const drones = new Map();

// ─── Sprite helpers ──────────────────────────────────────────────────────────

const DRONE_W = 10;
const DRONE_H = DRONE_W * (158 / 174);

const SVG_DRONE = (top, bottom) =>
  `<svg xmlns="http://www.w3.org/2000/svg" version="1.2" baseProfile="tiny" width="348" height="316" viewBox="41 -4 174 158"><path d="M 45,150 L45,70 100,20 155,70 155,150" stroke-width="4" stroke="black" fill="rgb(220,40,40)" fill-opacity="1"></path><path d="m 60,84 40,20 40,-20 0,8 -40,25 -40,-25 z" stroke-width="3" stroke="none" fill="black"></path><text x="100" y="71" text-anchor="middle" font-size="25" font-family="Arial" font-weight="bold" dominant-baseline="middle" fill="black">${top}</text><text x="100" y="134" text-anchor="middle" font-size="28" font-family="Arial" font-weight="bold" dominant-baseline="middle" fill="black">${bottom}</text><text x="175" y="40" text-anchor="start" font-size="40" font-family="Arial" font-weight="bold" stroke-width="8" stroke="white" fill="black" paint-order="stroke">__ID__</text></svg>`;

function makeDroneTexture(type, id) {
  const svg = SVG_DRONE(
    type == 0 ? "ISR" : "A",
    type == 2 ? "LR" : "SR",
  ).replace("__ID__", id);
  const blob = new Blob([svg], { type: "image/svg+xml" });
  const url = URL.createObjectURL(blob);
  const tex = new THREE.TextureLoader().load(url, () =>
    URL.revokeObjectURL(url),
  );
  return tex;
}

// ─── Trail helpers ───────────────────────────────────────────────────────────

const resolution = new THREE.Vector2(window.innerWidth, window.innerHeight);
window.addEventListener("resize", () =>
  resolution.set(window.innerWidth, window.innerHeight),
);

function makeTrailMaterial() {
  return new LineMaterial({
    color: 0xff3333,
    linewidth: LINE_WIDTH,
    resolution,
  });
}

function buildTrailGeo(history) {
  const positions = new Float32Array(history.length * 3);
  for (let i = 0; i < history.length; i++) {
    positions[i * 3] = history[i].x;
    positions[i * 3 + 1] = history[i].y;
    positions[i * 3 + 2] = history[i].z;
  }
  const geo = new LineGeometry();
  geo.setPositions(positions);
  return geo;
}

// ─── Drone lifecycle ─────────────────────────────────────────────────────────

function toWorld(x, y, z) {
  const { minH, hRange } = getTerrainElevRange();
  return new THREE.Vector3(
    x * SCALE,
    ((z - minH) / hRange) * HEIGHT_SCALE,
    -y * SCALE,
  );
}

function createDrone(data) {
  const pos = toWorld(data.x, data.y, data.z);

  const tex = makeDroneTexture(data.id);
  const mat = new THREE.SpriteMaterial({ map: tex, depthTest: true });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(DRONE_W, DRONE_H, 1);
  sprite.position.copy(pos);
  droneRoot.add(sprite);

  const trailMat = makeTrailMaterial();
  // Need ≥ 2 points — seed with two identical points.
  const trail = new Line2(buildTrailGeo([pos.clone(), pos.clone()]), trailMat);
  trail.computeLineDistances();
  droneRoot.add(trail);

  return {
    sprite,
    mat,
    tex,
    trail,
    trailMat,
    history: [pos.clone()],
    droneId: data.id,
  };
}

function updateDrone(drone, data) {
  const pos = toWorld(data.x, data.y, data.z);
  drone.sprite.position.copy(pos);

  drone.history.push(pos.clone());
  if (drone.history.length > TRAIL_LEN) drone.history.shift();

  if (drone.history.length >= 2) {
    const old = drone.trail.geometry;
    drone.trail.geometry = buildTrailGeo(drone.history);
    drone.trail.computeLineDistances();
    old.dispose();
  }
}

function removeDrone(id) {
  const drone = drones.get(id);
  if (!drone) return;
  droneRoot.remove(drone.sprite);
  droneRoot.remove(drone.trail);
  drone.trail.geometry.dispose();
  drone.trailMat.dispose();
  drone.mat.dispose();
  drone.tex.dispose();
  drones.delete(id);
}

export function clearDrones() {
  for (const id of [...drones.keys()]) removeDrone(id);
}

// ─── WebSocket ───────────────────────────────────────────────────────────────

function handleMessage(ev) {
  if (typeof ev.data === "string") return;

  const list = decodeFrame(ev.data);
  const seen = new Set();

  for (const data of list) {
    seen.add(data.id);
    if (drones.has(data.id)) {
      updateDrone(drones.get(data.id), data);
    } else {
      drones.set(data.id, createDrone(data));
    }
  }

  for (const id of drones.keys()) {
    if (!seen.has(id)) removeDrone(id);
  }
}

function connect() {
  const ws = new WebSocket(WS_URL);
  ws.binaryType = "arraybuffer";

  ws.addEventListener("open", () => console.log("[drones] connected"));
  ws.addEventListener("message", handleMessage);
  ws.addEventListener("close", () => {
    // Drop all sprites/trails so a reconnect (e.g. after a new
    // /scenario/start replaces the run) doesn't carry old drones over.
    clearDrones();
    setTimeout(connect, 2000);
  });
  ws.addEventListener("error", () => ws.close());
}

connect();
