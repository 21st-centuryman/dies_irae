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
import { clearDrones, setTargets } from "./drones.js";
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
// History panel
const historyPanel = document.getElementById("historyTable");
const closeHistory = document.getElementById("closeHistory");
const openHistory  = document.getElementById("openHistory");
const historyCount = document.getElementById("historyCount");
const historyList  = document.getElementById("historyList");

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

historyPanel.style.display = "none";
closeHistory.addEventListener("click", () => { historyPanel.style.display = "none"; });
openHistory.addEventListener("click",  () => { historyPanel.style.display = "flex"; });

// ─── History table ───────────────────────────────────────────────────────────
// Each row = one completed scenario.

let _scenarioCount = 0;
// Tracks the running scenario so we can log it when it ends.
let _currentScenario = null; // { targets, srCount, lrCount, seed, hitTargets: Set }

function addHistoryRow({ targets, srCount, lrCount, droneCount, missilesFired, seed, hitTargets }) {
  _scenarioCount++;
  historyCount.textContent = `${_scenarioCount} scenario${_scenarioCount === 1 ? "" : "s"}`;

  const tr = document.createElement("tr");

  // Radar 1/2/3 — coordinates + hit status in one cell each
  for (let i = 0; i < 3; i++) {
    const td = document.createElement("td");
    if (i < targets.length) {
      const t = targets[i];
      const hit = hitTargets.has(i);
      td.className = hit ? "status-hit" : "status-safe";
      td.innerHTML = `(${Math.round(t.x)},${Math.round(t.y)})<br><span>${hit ? "Hit" : "Safe"}</span>`;
    } else {
      td.textContent = "—";
    }
    tr.appendChild(td);
  }

  // SR SAM, LR SAM counts
  [srCount, lrCount].forEach(n => {
    const td = document.createElement("td");
    td.textContent = n;
    tr.appendChild(td);
  });

  // Drone count
  const droneTd = document.createElement("td");
  droneTd.textContent = droneCount;
  tr.appendChild(droneTd);

  // Cost
  const costTd = document.createElement("td");
  costTd.textContent = `€${(droneCount * 5000).toLocaleString()}`;
  tr.appendChild(costTd);

  // Missiles fired
  const missilesTd = document.createElement("td");
  missilesTd.textContent = missilesFired ?? "—";
  tr.appendChild(missilesTd);

  // Seed
  const seedTd = document.createElement("td");
  seedTd.textContent = seed ?? "—";
  tr.appendChild(seedTd);

  historyList.prepend(tr);
}

async function _commitScenario() {
  if (!_currentScenario) return;
  const scenario = _currentScenario;
  _currentScenario = null;

  try {
    const MISSILE_SERVER = `http://${window.location.hostname}:4000`;
    const res = await fetch(`${MISSILE_SERVER}/missiles/stats`);
    if (res.ok) scenario.missilesFired = (await res.json()).fired;
  } catch (_) { /* non-fatal if missile server isn't running */ }

  addHistoryRow(scenario);
}

document.addEventListener("droneHit", (e) => {
  if (_currentScenario) _currentScenario.hitTargets.add(e.detail.targetIdx);
});

document.addEventListener("scenarioEnded", () => {
  simRunning = false;
  setSimRunning(false);
  _commitScenario();
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
  const body = JSON.stringify({ speed: multiplier });
  const opts = { method: "PUT", headers: { "Content-Type": "application/json" }, body };
  const MISSILE_SERVER = `http://${window.location.hostname}:4000`;
  await Promise.allSettled([
    fetch(`${SCENARIO_SERVER}/simulation/speed`, opts),
    fetch(`${MISSILE_SERVER}/simulation/speed`, opts),
  ]);
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
const SCENARIO_SERVER = `http://${window.location.hostname}:3000`;

// ─── Simulation state ────────────────────────────────────────────────────────

let simRunning = false;

async function stopScenario() {
  simRunning = false;
  setSimRunning(false);
  await _commitScenario();
  try {
    await fetch(`${SCENARIO_SERVER}/scenario/stop`, { method: "PUT" });
  } catch (_) { /* non-fatal if server not reachable */ }
}

// Scenario type toggle.
let scenarioType = "allatonce";

const spawnWindowField = document.getElementById("spawnWindowField");

function updateScenarioFieldStates() {
  // "All at once" forces spawn_window = 0 (all drones in at t=0), so the
  // input is meaningless then — grey it out. "Spread" honors whatever
  // the user typed.
  const isAllAtOnce = scenarioType === "allatonce";
  spawnWindowField.classList.toggle("disabled", isAllAtOnce);
  spawnWindowField.querySelector("input").disabled = isAllAtOnce;
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
const seedEl   = document.getElementById("scenarioSeed");

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
    spawn_window:
      scenarioType === "allatonce"
        ? 0
        : parseInt(document.getElementById("spawnWindow").value) || 0,
    wave_interval: 0,  // dummy: backend still accepts this field but it's unused
    dir: document
      .getElementById("dir")
      .value.split(",")
      .map((s) => s.trim())
      .filter((s) => s.length > 0)
      .map(Number)
      .filter((n) => Number.isFinite(n)),
    dir_spread: parseFloat(document.getElementById("dir_spread").value) || 0,
    seed: (() => {
      // Empty input → null → backend draws a fresh non-deterministic seed.
      // Any integer → deterministic spawn azimuths, jitter, and spawn times.
      const raw = document.getElementById("seed").value.trim();
      if (raw === "") return null;
      const n = parseInt(raw, 10);
      return Number.isFinite(n) ? n : null;
    })(),
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
  const samPositions = getSamPositions().map(({ pos: p, type }) => [
    p.x * METRES_PER_UNIT,
    -p.z * METRES_PER_UNIT,
    (p.y / HEIGHT_SCALE) * hRange + minH,
    type,
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
      seedEl.textContent = data.seed != null ? `Seed: ${data.seed}` : "";
      // Register current scenario for history logging when it ends.
      const targetObjs = payload.targets.map(([x, y]) => ({ x, y }));
      setTargets(targetObjs);
      const placed = getSamPositions();
      _currentScenario = {
        targets:       targetObjs,
        srCount:       placed.filter(s => s.type === "SR").length,
        lrCount:       placed.filter(s => s.type === "LR").length,
        droneCount:    payload.drone_count,
        missilesFired: null,
        seed:          data.seed ?? null,
        hitTargets:    new Set(),
      };
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
