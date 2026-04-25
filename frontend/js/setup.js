import * as THREE from "three";
import { scene, camera, renderer, WORLD_SIZE, getTerrain } from "./3d.js";

// ─── State ───────────────────────────────────────────────────────────────────

let _active = false;
let _type = "radar"; // 'radar' | 'sam'

const _sprites = [];
const _radarPositions = [];
let _radarCount = 0;
let _samCount = 0;
const MAX_RADARS = 3;

// ─── Radar SVG template ──────────────────────────────────────────────────────
// Original viewBox: "-20 36 321.33 136"  (width≈642, height≈272)
// We replace __ID__ with the assigned number and keep text red.

const RADAR_SVG_TEMPLATE = `<svg xmlns="http://www.w3.org/2000/svg" version="1.2" baseProfile="tiny" width="642.6666666666666" height="272" viewBox="-20 36 321.3333333333333 136"><circle cx="100" cy="100" r="60" stroke-width="4" stroke="black" fill="rgb(0,160,255)" fill-opacity="1" ></circle><g transform="translate(0,0)" ><g transform="scale(1)" ><path d="M72,95 l30,-25 0,25 30,-25 M70,70 c0,35 15,50 50,50" stroke-width="3" stroke="black" fill="none" ></path></g></g><text x="20" y="160" text-anchor="end" font-size="40" font-family="Arial" font-weight="bold" stroke-width="8" stroke="white" fill="red" paint-order="stroke" >__ID__</text><text x="180" y="120" text-anchor="start" font-size="40" font-family="Arial" font-weight="bold" stroke-width="8" stroke="white" fill="red" paint-order="stroke" >target</text></svg>`;

function makeRadarTexture(id) {
  const svg = RADAR_SVG_TEMPLATE.replace("__ID__", String(id));
  const blob = new Blob([svg], { type: "image/svg+xml" });
  const url = URL.createObjectURL(blob);
  return new THREE.TextureLoader().load(url, () => URL.revokeObjectURL(url));
}

// ─── SAM SVG templates ───────────────────────────────────────────────────────

const SAM_SVG_TEMPLATE = (range) =>
  `<svg xmlns="http://www.w3.org/2000/svg" version="1.2" baseProfile="tiny" width="428" height="252" viewBox="-35 46 214 126"><path d="M25,50 l150,0 0,100 -150,0 z" stroke-width="4" stroke="black" fill="rgb(0,160,255)" fill-opacity="1" ></path><path d="M25,150 C25,110 175,110 175,150" stroke-width="3" stroke="black" fill="none" ></path><path d="M 100,82.62 V 120  M 90,120 V 90 c 0,-10 20,-10 20,0 v 30" stroke-width="3" stroke="black" fill="none" ></path><text x="100" y="134" text-anchor="middle" font-size="28" font-family="Arial" font-weight="bold" dominant-baseline="middle" stroke-width="3" stroke="none" fill="black" >${range}</text><text x="5" y="160" text-anchor="end" font-size="40" font-family="Arial" font-weight="bold" stroke-width="8" stroke="white" fill="black" paint-order="stroke" >__ID__</text></svg>`;

function makeSamTexture(template, id) {
  const svg = template.replace("__ID__", String(id));
  const blob = new Blob([svg], { type: "image/svg+xml" });
  const url = URL.createObjectURL(blob);
  return new THREE.TextureLoader().load(url, () => URL.revokeObjectURL(url));
}

// ─── Sprite sizes (world units) ──────────────────────────────────────────────

const SPRITE_SIZE = {
  radar: { w: 64, h: 27 },
  srsam: { w: 214, h: 126 },
  lrsam: { w: 214, h: 126 },
};

// ─── Click vs drag detection ─────────────────────────────────────────────────

let _mouseDown = null;
const DRAG_THRESHOLD = 5; // pixels

// ─── Raycaster ───────────────────────────────────────────────────────────────

const _raycaster = new THREE.Raycaster();

function onPointerDown(e) {
  _mouseDown = { x: e.clientX, y: e.clientY };
}

function onPointerUp(e) {
  if (!_active || !_mouseDown) return;

  const dx = e.clientX - _mouseDown.x;
  const dy = e.clientY - _mouseDown.y;
  if (Math.sqrt(dx * dx + dy * dy) > DRAG_THRESHOLD) return; // was a drag

  if (_type === "radar" && _radarCount >= MAX_RADARS) return;

  const terrain = getTerrain();
  if (!terrain) return;

  const rect = renderer.domElement.getBoundingClientRect();
  const ndc = new THREE.Vector2(
    ((e.clientX - rect.left) / rect.width) * 2 - 1,
    -((e.clientY - rect.top) / rect.height) * 2 + 1,
  );

  _raycaster.setFromCamera(ndc, camera);
  const hits = _raycaster.intersectObject(terrain);
  if (!hits.length) return;

  const pt = hits[0].point;

  const size = SPRITE_SIZE[_type] ?? SPRITE_SIZE.radar;
  const scale = (WORLD_SIZE / 1000) * 8;
  const aspect = size.w / size.h;

  let tex;
  if (_type === "radar") {
    tex = makeRadarTexture(_radarCount);
    _radarPositions.push(pt.clone());
    _radarCount++;
  } else if (_type === "lrsam") {
    tex = makeSamTexture(SAM_SVG_TEMPLATE("LR"), _samCount);
    _samCount++;
  } else {
    tex = makeSamTexture(SAM_SVG_TEMPLATE("SR"), _samCount);
    _samCount++;
  }

  const mat = new THREE.SpriteMaterial({
    map: tex,
    depthTest: true,
    sizeAttenuation: true,
  });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(scale * aspect, scale, 1);
  sprite.position.copy(pt).add(new THREE.Vector3(0, scale * 0.5, 0));
  scene.add(sprite);
  _sprites.push(sprite);
}

// ─── Public API ──────────────────────────────────────────────────────────────

export function getRadarPositions() {
  return _radarPositions.slice();
}
export function isSetupActive() {
  return _active;
}
export function setSetupActive(val) {
  _active = val;
}
export function setSetupType(type) {
  _type = type;
}

export function clearSetup() {
  for (const s of _sprites) {
    scene.remove(s);
    s.material.dispose();
  }
  _sprites.length = 0;
  _radarPositions.length = 0;
  _radarCount = 0;
  _samCount = 0;
}

export function initSetup() {
  const el = renderer.domElement;
  el.addEventListener("pointerdown", onPointerDown);
  el.addEventListener("pointerup", onPointerUp);
}
