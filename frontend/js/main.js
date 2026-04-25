import {
  initSetup,
  isSetupActive,
  setSetupActive,
  setSetupType,
  clearSetup,
  getRadarPositions,
  getSamPositions,
  setSimRunning,
} from "./setup.js";
import { WORLD_SIZE, HEIGHT_SCALE, getTerrainElevRange } from "./3d.js";
import { clearDrones } from "./drones.js";
import { clearMissiles } from "./missile.js";

// ===========================================================================
// DARK / LIGHT MODE
// ===========================================================================
document.getElementById("darkMode").addEventListener("click", () => {
  document.documentElement.classList.replace("light", "dark");
  document.dispatchEvent(new CustomEvent("themeChange"));
});
document.getElementById("lightMode").addEventListener("click", () => {
  document.documentElement.classList.replace("dark", "light");
  document.dispatchEvent(new CustomEvent("themeChange"));
});

// ===========================================================================
// PANELS
// ===========================================================================
// Scenario panel
const scenarioPanel = document.getElementById("scenarioTable");
const closeScenario = document.getElementById("closeScenario");
const openScenario = document.getElementById("openScenario");
// Map panel
const mapPanel = document.getElementById("mapTable");
const closeMap = document.getElementById("closeMap");
const openMap = document.getElementById("openMap");
const expandBtn = document.getElementById("expandMap");
// Setup panel
const setupPanel = document.getElementById("setupTable");
const closeSetup = document.getElementById("closeSetup");
const openSetup = document.getElementById("openSetup");

// ===========================================================================
// MAP (Leaflet — Sweden)
// ===========================================================================
function readMapColors() {
  const v = getComputedStyle(document.documentElement);
  return { outline: v.getPropertyValue("--text").trim() };
}

const swedenMap = L.map("sweden-map", {
  zoomControl: false,
  attributionControl: false,
}).setView([62.5, 16.5], 5);

swedenMap.setMaxBounds([
  [54.0, 9.5],
  [70.0, 25.0],
]);

let swedenGeoJSON = null;
let swedenLayer = null;

fetch(
  "https://raw.githubusercontent.com/okfse/sweden-geojson/master/swedish_regions.geojson",
)
  .then((r) => r.json())
  .then((data) => {
    swedenGeoJSON = data;
    const { outline } = readMapColors();
    swedenLayer = L.geoJSON(data, {
      style: {
        color: outline,
        weight: 1,
        opacity: 0.6,
        fillColor: outline,
        fillOpacity: 0.04,
      },
    }).addTo(swedenMap);
  });

document.addEventListener("themeChange", () => {
  if (!swedenLayer) return;
  const { outline } = readMapColors();
  swedenLayer.setStyle({ color: outline, fillColor: outline });
});

let pin = null;
const coordInput = document.getElementById("coordInput");
const latInput = document.getElementById("lat");
const lonInput = document.getElementById("lon");

function setCoordinate(lat, lng) {
  if (pin) {
    pin.setLatLng([lat, lng]);
  } else {
    pin = L.marker([lat, lng]).addTo(swedenMap);
  }
  coordInput.value = `${lat.toFixed(5)}, ${lng.toFixed(5)}`;
  latInput.value = lat;
  lonInput.value = lng;
  if (simRunning) stopScenario();
  clearSetup();
  clearDrones();
  clearMissiles();
}

coordInput.addEventListener("keydown", (e) => {
  if (e.key !== "Enter") return;
  const [latStr, lngStr] = coordInput.value.split(",").map((s) => s.trim());
  const lat = parseFloat(latStr);
  const lng = parseFloat(lngStr);
  if (isNaN(lat) || isNaN(lng)) return;
  setCoordinate(lat, lng);
  swedenMap.setView([lat, lng]);
  coordInput.blur();
});

swedenMap.on("click", (e) => {
  const { lat, lng } = e.latlng;

  if (swedenGeoJSON) {
    const point = turf.point([lng, lat]);
    const inside = swedenGeoJSON.features.some((f) =>
      turf.booleanPointInPolygon(point, f),
    );
    if (!inside) return;
  }

  setCoordinate(lat, lng);
});

initSetup();

document.getElementById("sendMap").addEventListener("click", () => {
  if (!latInput.value || !lonInput.value) return;
  document.dispatchEvent(new CustomEvent("fetchTerrain"));
});

// ===========================================================================
// PANELS
// ===========================================================================
// Init
scenarioPanel.style.display = "none";
closeScenario.addEventListener("click", () => {
  scenarioPanel.style.display = "none";
});
openScenario.addEventListener("click", () => {
  scenarioPanel.style.display = "flex"; // matches your existing display:flex
});

mapPanel.style.display = "none";
closeMap.addEventListener("click", () => {
  mapPanel.style.display = "none";
  mapPanel.classList.remove("expanded");
  expandBtn.textContent = "Expand Map";
});
openMap.addEventListener("click", () => {
  mapPanel.style.display = "flex";
  setTimeout(() => swedenMap.invalidateSize(), 50);
});

expandBtn.addEventListener("click", () => {
  const isExpanded = mapPanel.classList.toggle("expanded");
  expandBtn.textContent = isExpanded ? "Collapse Map" : "Expand Map";
  setTimeout(() => swedenMap.invalidateSize(), 510);
});

setupPanel.style.display = "none";
closeSetup.addEventListener("click", () => {
  setupPanel.style.display = "none";
  setSetupActive(false);
});
openSetup.addEventListener("click", () => {
  setupPanel.style.display = "flex";
  setSetupActive(true);
});

document.getElementById("setupItemType").addEventListener("click", (e) => {
  const btn = e.target.closest(".toggle");
  if (!btn) return;
  document
    .querySelectorAll("#setupItemType .toggle")
    .forEach((b) => b.classList.remove("active"));
  btn.classList.add("active");
  setSetupType(btn.dataset.value);
});

document.getElementById("clearSetup").addEventListener("click", async () => {
  if (simRunning) await stopScenario();
  clearSetup();
  clearDrones();
  clearMissiles();
});

// ===========================================================================
// SPEED
// ===========================================================================
const speedSteps = [1, 2, 4, 10];

async function setSpeed(multiplier) {
  try {
    await fetch(`${SCENARIO_SERVER}/simulation/speed`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ speed: multiplier }),
    });
  } catch (_) {
    /* server may not be running yet */
  }
}

function toggleSpeed() {
  const el = document.getElementById("speed");
  const current = parseFloat(el.textContent);
  const nextIndex = (speedSteps.indexOf(current) + 1) % speedSteps.length;
  const next = speedSteps[nextIndex];
  el.textContent = next + "x";
  setSpeed(next);
}

document.getElementById("speed").addEventListener("click", toggleSpeed);
// ===========================================================================
// SCENARIO
// ===========================================================================
const SCENARIO_SERVER = "http://10.154.139.105:3000";

// ─── Simulation state ────────────────────────────────────────────────────────

let simRunning = false;

async function stopScenario() {
  simRunning = false;
  setSimRunning(false);
  try {
    await fetch(`${SCENARIO_SERVER}/scenario/stop`, { method: "PUT" });
  } catch (_) { /* non-fatal if server not reachable */ }
}

// Linked percentage pairs: changing one updates the other to 100 - value.
function linkPct(aId, bId, onChange) {
  const a = document.getElementById(aId);
  const b = document.getElementById(bId);
  a.addEventListener("input", () => {
    b.value = Math.max(0, 100 - (parseInt(a.value) || 0));
    onChange?.();
  });
  b.addEventListener("input", () => {
    a.value = Math.max(0, 100 - (parseInt(b.value) || 0));
    onChange?.();
  });
}

const attackTypeField = document.getElementById("attackTypeField");
function updateAttackTypeState() {
  const disabled =
    (parseInt(document.getElementById("pctAttack").value) || 0) === 0;
  attackTypeField.classList.toggle("disabled", disabled);
  attackTypeField
    .querySelectorAll("input")
    .forEach((i) => (i.disabled = disabled));
}

linkPct("pctAttack", "pctRecon", updateAttackTypeState);
linkPct("pctShort", "pctLong");
updateAttackTypeState();

// Scenario type toggle.
let scenarioType = "allatonce";

const spawnWindowField = document.getElementById("spawnWindowField");
const waveIntervalField = document.getElementById("waveIntervalField");

function updateScenarioFieldStates() {
  const isAllAtOnce = scenarioType === "allatonce";
  const isWave = scenarioType === "wave";

  spawnWindowField.classList.toggle("disabled", isAllAtOnce);
  spawnWindowField.querySelector("input").disabled = isAllAtOnce;

  waveIntervalField.classList.toggle("disabled", !isWave);
  waveIntervalField.querySelector("input").disabled = !isWave;
}

document.getElementById("scenarioType").addEventListener("click", (e) => {
  const btn = e.target.closest(".toggle");
  if (!btn) return;
  document
    .querySelectorAll("#scenarioType .toggle")
    .forEach((b) => b.classList.remove("active"));
  btn.classList.add("active");
  scenarioType = btn.dataset.value;
  updateScenarioFieldStates();
});

updateScenarioFieldStates();

// Send.
const statusEl = document.getElementById("scenarioStatus");

const METRES_PER_UNIT = (5 * 2 * 1000) / WORLD_SIZE; // 10 km diameter map → 10 m per world unit

document.getElementById("sendScenario").addEventListener("click", async () => {
  if (!pin) {
    statusEl.textContent = "Error: no coordinate selected";
    return;
  }

  const radarPositions = getRadarPositions();
  if (radarPositions.length === 0) {
    statusEl.textContent = "Warning: no radar sites placed";
    return;
  }

  const payload = {
    drone_count: parseInt(document.getElementById("droneCount").value) || 0,
    pct_attack: parseInt(document.getElementById("pctAttack").value) || 0,
    pct_recon: parseInt(document.getElementById("pctRecon").value) || 0,
    pct_short: parseInt(document.getElementById("pctShort").value) || 0,
    pct_long: parseInt(document.getElementById("pctLong").value) || 0,
    scenario_type: scenarioType,
    spawn_window: parseInt(document.getElementById("spawnWindow").value) || 0,
    wave_interval: parseInt(document.getElementById("waveInterval").value) || 0,
    dir: document
      .getElementById("dir")
      .value.split(",")
      .map((s) => s.trim())
      .filter((s) => s.length > 0)
      .map(Number)
      .filter((n) => Number.isFinite(n)),
    dir_spread: parseFloat(document.getElementById("dir_spread").value) || 0,
    lat: parseFloat(document.getElementById("lat").value),
    lon: parseFloat(document.getElementById("lon").value),
    targets: radarPositions.map((p) => [
      p.x * METRES_PER_UNIT,
      -p.z * METRES_PER_UNIT,
    ]),
  };

  clearDrones();
  clearMissiles();
  simRunning = false;
  setSimRunning(false);

  console.log("[scenario] PUT /scenario/start", payload);

  // Tell the missile server where the SAM sites are (xyz in server metres).
  // World y → real elevation: elev = (world_y / HEIGHT_SCALE) * hRange + minH
  const { minH, hRange } = getTerrainElevRange();
  const samPositions = getSamPositions().map(p => [
    p.x * METRES_PER_UNIT,
    -p.z * METRES_PER_UNIT,
    (p.y / HEIGHT_SCALE) * hRange + minH,
  ]);
  const MISSILE_SERVER = `http://${window.location.hostname}:4000`;
  fetch(`${MISSILE_SERVER}/sam/positions`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      positions: samPositions,
      lat: parseFloat(latInput.value),
      lon: parseFloat(lonInput.value),
    }),
  }).catch(() => {});  // non-fatal if missile server isn't running

  try {
    const res = await fetch(`${SCENARIO_SERVER}/scenario/start`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    statusEl.textContent = `Status: ${res.status}`;
    if (res.ok) {
      simRunning = true;
      setSimRunning(true);
      const data = await res.json();
      document.dispatchEvent(
        new CustomEvent("scenarioStarted", {
          detail: {
            target_xy: data.target_xy,
            target_z: data.target_z,
          },
        }),
      );
    }
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
  }
});
