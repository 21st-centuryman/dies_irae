import * as THREE from "three";
import { scene, camera, renderer, WORLD_SIZE, getTerrain } from "./3d.js";
import { makeRadarTexture, makeSamTexture } from "./svg.js";

// ─── State ───────────────────────────────────────────────────────────────────

let _active = false;
let _type = "radar"; // 'radar' | 'sam'
let _simRunning = false;

const _sprites = [];
const _radarPositions = [];
const _samPositions = [];
let _radarCount = 0;
let _samCount = 0;
const MAX_RADARS = 3;


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
  if (_simRunning) return; // placement locked during active scenario

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
    tex = makeSamTexture("LR", _samCount);
    _samPositions.push(pt.clone());
    _samCount++;
  } else {
    tex = makeSamTexture("SR", _samCount);
    _samPositions.push(pt.clone());
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

export function getRadarPositions() { return _radarPositions.slice(); }
export function getSamPositions()   { return _samPositions.slice(); }
export function setSimRunning(val)  { _simRunning = val; }
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
  _samPositions.length = 0;
  _radarCount = 0;
  _samCount = 0;
}

export function initSetup() {
  const el = renderer.domElement;
  el.addEventListener("pointerdown", onPointerDown);
  el.addEventListener("pointerup", onPointerUp);
}
