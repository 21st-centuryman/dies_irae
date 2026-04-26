import * as THREE from "three";
import { Line2 } from "three/addons/lines/Line2.js";
import { LineGeometry } from "three/addons/lines/LineGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { scene, WORLD_SIZE, HEIGHT_SCALE, getTerrainElevRange } from "./3d.js";
import { makeDroneTexture } from "./svg.js";

// ─── Config ──────────────────────────────────────────────────────────────────

const WS_URL      = `ws://${window.location.hostname}:3000/simulation/fake/stream`;
const PREDICT_URL = `ws://${window.location.hostname}:3001/ws`;

const METRES_PER_UNIT = (5 * 2 * 1000) / WORLD_SIZE; // 10 km diameter map → 10 m per world unit
const SCALE       = 1 / METRES_PER_UNIT;
const TRAIL_LEN   = 600;
const RAW_MAX     = 600;   // positions sent to missile server
const LINE_WIDTH  = 2; // px

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

  const tex = makeDroneTexture(data.type, data.id);
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
    history:    [pos.clone()],
    rawHistory: [[data.x, data.y, data.z]],
    droneId:    data.id,
  };
}

function updateDrone(drone, data) {
  const pos = toWorld(data.x, data.y, data.z);
  drone.sprite.position.copy(pos);

  drone.history.push(pos.clone());
  if (drone.history.length > TRAIL_LEN) drone.history.shift();

  drone.rawHistory.push([data.x, data.y, data.z]);
  if (drone.rawHistory.length > RAW_MAX) drone.rawHistory.shift();

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
  hideImpactMarker();
}

// ─── Impact marker ────────────────────────────────────────────────────────────
// Ring on the ground + a short vertical stem — marks the predicted strike point.

let _impactGroup = null;

function _buildImpactMarker() {
  const group = new THREE.Group();

  const ringGeo = new THREE.RingGeometry(6, 8, 48);
  ringGeo.rotateX(-Math.PI / 2);
  const ringMat = new THREE.MeshBasicMaterial({
    color: 0xff4400, side: THREE.DoubleSide, depthTest: false,
  });
  group.add(new THREE.Mesh(ringGeo, ringMat));

  const dotGeo = new THREE.CircleGeometry(2.5, 24);
  dotGeo.rotateX(-Math.PI / 2);
  const dotMat = new THREE.MeshBasicMaterial({
    color: 0xffaa00, side: THREE.DoubleSide, depthTest: false,
  });
  group.add(new THREE.Mesh(dotGeo, dotMat));

  const stemGeo = new THREE.CylinderGeometry(0.5, 0.5, 20, 8);
  stemGeo.translate(0, 10, 0);
  const stemMat = new THREE.MeshBasicMaterial({ color: 0xff4400, depthTest: false });
  group.add(new THREE.Mesh(stemGeo, stemMat));

  group.visible     = false;
  group.renderOrder = 999;
  scene.add(group);
  return group;
}

function showImpactMarker(x, y, z) {
  if (!_impactGroup) _impactGroup = _buildImpactMarker();
  _impactGroup.position.copy(toWorld(x, y, z));
  _impactGroup.visible = true;
}

function hideImpactMarker() {
  if (_impactGroup) _impactGroup.visible = false;
}

// ─── Missile prediction WebSocket ─────────────────────────────────────────────

let _predictWs = null;

function sendPrediction() {
  if (!_predictWs || _predictWs.readyState !== WebSocket.OPEN) return;

  // Use the first active drone's raw history.
  const first = drones.values().next().value;
  if (!first || first.rawHistory.length < 5) return;

  const lat = parseFloat(document.getElementById('lat')?.value  || '0');
  const lon = parseFloat(document.getElementById('lon')?.value  || '0');

  _predictWs.send(JSON.stringify({ lat, lon, positions: first.rawHistory }));
}

function connectPredict() {
  _predictWs = new WebSocket(PREDICT_URL);

  _predictWs.addEventListener('open', () => console.log('[missile] prediction server connected'));

  _predictWs.addEventListener('message', (ev) => {
    try {
      const { impact } = JSON.parse(ev.data);
      if (impact) {
        showImpactMarker(impact[0], impact[1], impact[2]);
      } else {
        hideImpactMarker();
      }
    } catch (_) { /* ignore malformed */ }
  });

  _predictWs.addEventListener('close', () => setTimeout(connectPredict, 2000));
  _predictWs.addEventListener('error', () => _predictWs.close());
}

connectPredict();

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

  // Push latest positions to missile server on every frame.
  sendPrediction();
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
