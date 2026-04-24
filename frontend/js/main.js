// ===========================================================================
// PANELS
// ===========================================================================
// Scenario panel
const scenarioPanel = document.getElementById('scenarioTable');
const closeScenario = document.getElementById('closeScenario');
const openScenario = document.getElementById('openScenario');
// Map panel
const mapPanel = document.getElementById('mapTable');
const closeMap = document.getElementById('closeMap');
const openMap = document.getElementById('openMap');
const expandBtn = document.getElementById('expandMap');

// ===========================================================================
// CSS VARIABLES
// ===========================================================================
const cssVars = getComputedStyle(document.documentElement);
const color = {
  container: cssVars.getPropertyValue('--container').trim(),
  background: cssVars.getPropertyValue('--background').trim(),
};

// ===========================================================================
// MAP (Leaflet — Sweden)
// ===========================================================================
const swedenMap = L.map('sweden-map', {
  zoomControl: false,
  attributionControl: false,
}).setView([62.5, 16.5], 5);

swedenMap.setMaxBounds([[54.0, 9.5], [70.0, 25.0]]);

let swedenGeoJSON = null;

fetch('https://raw.githubusercontent.com/okfse/sweden-geojson/master/swedish_regions.geojson')
  .then(r => r.json())
  .then(data => {
    swedenGeoJSON = data;
    L.geoJSON(data, {
      style: {
        color: color.container,
        weight: 1,
        opacity: 0.6,
        fillColor: color.container,
        fillOpacity: 0.04,
      }
    }).addTo(swedenMap);
  });

let pin = null;
const coordInput = document.getElementById('coordInput');
const latInput = document.getElementById('lat');
const lonInput = document.getElementById('lon');

function setCoordinate(lat, lng) {
  if (pin) {
    pin.setLatLng([lat, lng]);
  } else {
    pin = L.marker([lat, lng]).addTo(swedenMap);
  }
  coordInput.value = `${lat.toFixed(5)}, ${lng.toFixed(5)}`;
  latInput.value = lat;
  lonInput.value = lng;
}

coordInput.addEventListener('keydown', (e) => {
  if (e.key !== 'Enter') return;
  const [latStr, lngStr] = coordInput.value.split(',').map(s => s.trim());
  const lat = parseFloat(latStr);
  const lng = parseFloat(lngStr);
  if (isNaN(lat) || isNaN(lng)) return;
  setCoordinate(lat, lng);
  swedenMap.setView([lat, lng]);
  coordInput.blur();
});

swedenMap.on('click', (e) => {
  const { lat, lng } = e.latlng;

  if (swedenGeoJSON) {
    const point = turf.point([lng, lat]);
    const inside = swedenGeoJSON.features.some(f => turf.booleanPointInPolygon(point, f));
    if (!inside) return;
  }

  setCoordinate(lat, lng);
});

document.getElementById('sendMap').addEventListener('click', () => {
  if (!latInput.value || !lonInput.value) return;
  document.dispatchEvent(new CustomEvent('fetchTerrain'));
});

// ===========================================================================
// PANELS
// ===========================================================================
// Init
scenarioPanel.style.display = 'none';
closeScenario.addEventListener('click', () => {
  scenarioPanel.style.display = 'none';
});
openScenario.addEventListener('click', () => {
  scenarioPanel.style.display = 'flex'; // matches your existing display:flex
});

mapPanel.style.display = 'none';
closeMap.addEventListener('click', () => {
  mapPanel.style.display = 'none';
  mapPanel.classList.remove('expanded');
  expandBtn.textContent = 'Expand Map';
});
openMap.addEventListener('click', () => {
  mapPanel.style.display = 'flex';
  setTimeout(() => swedenMap.invalidateSize(), 50);
});

expandBtn.addEventListener('click', () => {
  const isExpanded = mapPanel.classList.toggle('expanded');
  expandBtn.textContent = isExpanded ? 'Collapse Map' : 'Expand Map';
  setTimeout(() => swedenMap.invalidateSize(), 510);
});


// ===========================================================================
// Parse toml scenarios
// ===========================================================================
const tomlText = await fetch('./scenarios/scenarios.toml').then(r => r.text());
const scenarioNames = [];
const scenarioRegex = /^\[scenarios\.(\w+)\]\s*\nname\s*=\s*"([^"]+)"/gm;
let match;
while ((match = scenarioRegex.exec(tomlText)) !== null) {
  scenarioNames.push({ key: match[1], name: match[2] });
}

// Populate the dropdown
const dropdown = document.getElementById('scenarioDropdown');
dropdown.innerHTML = ''; // clear placeholder options

for (const scenario of scenarioNames) {
  const option = document.createElement('option');
  option.value = scenario.key;
  option.textContent = scenario.name;
  dropdown.appendChild(option);
}

// ===========================================================================
// SEND SCENARIO
// ===========================================================================
const statusEl = document.querySelector('#scenarioTable h6');

document.getElementById('sendScenario').addEventListener('click', async () => {
  const server = document.getElementById('serverInput').value.trim();
  const port = document.getElementById('serverPort').value.trim();
  if (!server || !port) return;

  try {
    const { encode } = await import('../proto/drone.js');
    const encoded = await encode(dropdown.value, tomlText);

    const res = await fetch(`http://${server}:${port}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-protobuf' },
      body: encoded,
    });
    statusEl.textContent = `Status: ${res.status}`;
  } catch (err) {
    statusEl.textContent = `Status: error — ${err.message}`;
  }
});

