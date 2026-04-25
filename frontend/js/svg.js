import * as THREE from "three";

// ─── Helpers ─────────────────────────────────────────────────────────────────

function toTexture(svg) {
  const blob = new Blob([svg], { type: "image/svg+xml" });
  const url = URL.createObjectURL(blob);
  return new THREE.TextureLoader().load(url, () => URL.revokeObjectURL(url));
}

// ─── Radar ───────────────────────────────────────────────────────────────────
// viewBox: "-20 36 321.33 136"  (width≈642, height≈272)

const RADAR_SVG = (id) =>
  `<svg xmlns="http://www.w3.org/2000/svg" version="1.2" baseProfile="tiny" width="642.6666666666666" height="272" viewBox="-20 36 321.3333333333333 136"><circle cx="100" cy="100" r="60" stroke-width="4" stroke="black" fill="rgb(0,160,255)" fill-opacity="1" ></circle><g transform="translate(0,0)" ><g transform="scale(1)" ><path d="M72,95 l30,-25 0,25 30,-25 M70,70 c0,35 15,50 50,50" stroke-width="3" stroke="black" fill="none" ></path></g></g><text x="20" y="160" text-anchor="end" font-size="40" font-family="Arial" font-weight="bold" stroke-width="8" stroke="white" fill="red" paint-order="stroke" >${id}</text><text x="180" y="120" text-anchor="start" font-size="40" font-family="Arial" font-weight="bold" stroke-width="8" stroke="white" fill="red" paint-order="stroke" >target</text></svg>`;

export function makeRadarTexture(id) {
  return toTexture(RADAR_SVG(id));
}

// ─── SAM ─────────────────────────────────────────────────────────────────────
// viewBox: "-35 46 214 126"  (width≈428, height≈252)
// range: "SR" | "LR"

const SAM_SVG = (range, id) =>
  `<svg xmlns="http://www.w3.org/2000/svg" version="1.2" baseProfile="tiny" width="428" height="252" viewBox="-35 46 214 126"><path d="M25,50 l150,0 0,100 -150,0 z" stroke-width="4" stroke="black" fill="rgb(0,160,255)" fill-opacity="1" ></path><path d="M25,150 C25,110 175,110 175,150" stroke-width="3" stroke="black" fill="none" ></path><path d="M 100,82.62 V 120  M 90,120 V 90 c 0,-10 20,-10 20,0 v 30" stroke-width="3" stroke="black" fill="none" ></path><text x="100" y="134" text-anchor="middle" font-size="28" font-family="Arial" font-weight="bold" dominant-baseline="middle" stroke-width="3" stroke="none" fill="black" >${range}</text><text x="5" y="160" text-anchor="end" font-size="40" font-family="Arial" font-weight="bold" stroke-width="8" stroke="white" fill="black" paint-order="stroke" >${id}</text></svg>`;

export function makeSamTexture(range, id) {
  return toTexture(SAM_SVG(range, id));
}

// ─── Drone ───────────────────────────────────────────────────────────────────
// viewBox: "41 -4 174 158"  (width≈348, height≈316)
// type: 0 = ISR/recon, 1 = attack SR, 2 = attack LR

const DRONE_SVG = (top, bottom, id) =>
  `<svg xmlns="http://www.w3.org/2000/svg" version="1.2" baseProfile="tiny" width="348" height="316" viewBox="41 -4 174 158"><path d="M 45,150 L45,70 100,20 155,70 155,150" stroke-width="4" stroke="black" fill="rgb(220,40,40)" fill-opacity="1"></path><path d="m 60,84 40,20 40,-20 0,8 -40,25 -40,-25 z" stroke-width="3" stroke="none" fill="black"></path><text x="100" y="71" text-anchor="middle" font-size="25" font-family="Arial" font-weight="bold" dominant-baseline="middle" fill="black">${top}</text><text x="100" y="134" text-anchor="middle" font-size="28" font-family="Arial" font-weight="bold" dominant-baseline="middle" fill="black">${bottom}</text><text x="175" y="40" text-anchor="start" font-size="40" font-family="Arial" font-weight="bold" stroke-width="8" stroke="white" fill="black" paint-order="stroke">${id}</text></svg>`;

export function makeDroneTexture(type, id) {
  const top = type === 0 ? "ISR" : "A";
  const bottom = type === 2 ? "LR" : "SR";
  return toTexture(DRONE_SVG(top, bottom, id));
}

// ─── Missile ─────────────────────────────────────────────────────────────────
// viewBox: "41 -4 174 158"  (width≈348, height≈316)

const MISSILE_SVG = (id) =>
  `<svg xmlns="http://www.w3.org/2000/svg" version="1.2" baseProfile="tiny" width="348" height="316" viewBox="41 -4 174 158"><path d="M 155,150 C 155,50 115,30 100,30 85,30 45,50 45,150" stroke-width="4" stroke="black" fill="rgb(0,160,255)" fill-opacity="1"/><path d="m 87,135 v -11 l 6,-5 V 65 l 7,-10 7,10 v 54 l 6,5 v 11 l -13,-10 z" stroke-width="3" stroke="black" fill="rgb(255,255,128)"/><text x="68" y="100" text-anchor="middle" font-size="30" font-family="Arial" font-weight="bold" dominant-baseline="middle" stroke-width="3" stroke="none" fill="black">S</text><text x="132" y="100" text-anchor="middle" font-size="30" font-family="Arial" font-weight="bold" dominant-baseline="middle" stroke-width="3" stroke="none" fill="black">A</text><text x="175" y="40" text-anchor="start" font-size="40" font-family="Arial" font-weight="bold" stroke-width="8" stroke="white" fill="black" paint-order="stroke">${id}</text></svg>`;

export function makeMissileTexture(id) {
  return toTexture(MISSILE_SVG(id));
}
