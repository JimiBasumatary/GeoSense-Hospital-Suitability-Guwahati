
import {
  Viewer,
  Cartesian3,
  Color,
  HeightReference,
  OpenStreetMapImageryProvider,
  Ion,
  IonWorldImageryStyle,
  IonImageryProvider,
  createWorldTerrainAsync,
  ImageryLayer,
  Rectangle,
  SingleTileImageryProvider,
  GeoJsonDataSource,
} from "cesium";
import "cesium/Build/Cesium/Widgets/widgets.css";
import "./style.css";
const API_URL = "http://127.0.0.1:8002/agent/latest_result";
const POLL_INTERVAL_MS = 3000;

document.querySelector("#app").innerHTML = `
  <header class="topbar">
    <div>
      <h1>GeoSense Digital Twin</h1>
      <p>Guwahati City · Assam, India</p>
    </div>
    <span class="status">PHASE IV · PROTOTYPE</span>
  </header>

  <section class="info-panel">
    <strong>Study Area</strong>
    <span>Guwahati City, Assam, India</span>
    <span class="coordinates">26.1445° N, 91.7362° E (approx.)</span>
  </section>

  <section class="agent-panel" aria-live="polite">
    <div class="agent-panel-heading">
      <strong>GeoSense Agent Results</strong>
      <span id="agent-status">Connecting…</span>
    </div>
    <p><strong>Question:</strong> <span id="agent-question">Waiting for an agent result…</span></p>
    <p><strong>Answer:</strong> <span id="agent-answer">No result received yet.</span></p>
    <p><strong>Tools:</strong> <span id="agent-tools">—</span></p>
    <p class="agent-note" id="agent-note">Checking for updates every 3 seconds.</p>
  </section>

  <section class="building-controls" aria-label="Building layer controls">
    <label>
      <input type="checkbox" id="buildings-visible" checked>
      Show OpenStreetMap building footprints
    </label>
    <span id="buildings-status" role="status">Loading building footprints?</span>
  </section>

  <section class="ndvi-controls" aria-label="NDVI layer controls">
    <label for="ndvi-year">NDVI year:</label>
    <select id="ndvi-year">
  <option value="2015">2015</option>
  <option value="2021">2021</option>
  <option value="2023">2023</option>
</select>
    <label class="ndvi-toggle">
      <input type="checkbox" id="ndvi-visible">
      Show NDVI overlay
    </label>
    <span id="ndvi-status" role="status">Preparing NDVI layers?</span>
  </section>
  <section class="ndvi-controls" aria-label="Elevation layer controls">
    <label class="ndvi-toggle">
      <input type="checkbox" id="elevation-visible">
      Show elevation overlay
    </label>
    <span id="elevation-status" role="status">Preparing elevation layer…</span>
  </section>
<section class="ndvi-controls" aria-label="Slope layer controls">
  <label class="ndvi-toggle">
    <input type="checkbox" id="slope-visible">
    Show slope overlay
  </label>
  <span id="slope-status" role="status">Preparing slope layer…</span>
</section>

  <main id="cesiumContainer"></main>

  <footer class="footer">
    GeoSense Agent | CesiumJS 3D Digital Twin
  </footer>
`;

const statusElement = document.querySelector("#agent-status");
const questionElement = document.querySelector("#agent-question");
const answerElement = document.querySelector("#agent-answer");
const toolsElement = document.querySelector("#agent-tools");
const noteElement = document.querySelector("#agent-note");

let lastResultId = null;
Ion.defaultAccessToken = import.meta.env.VITE_CESIUM_ION_TOKEN;
let viewer = null;

try {
  viewer = new Viewer("cesiumContainer", {
    animation: false,
    timeline: false,
    baseLayerPicker: false,
    baseLayer: false,
    terrain: undefined,
    geocoder: false,
    homeButton: true,
    sceneModePicker: true,
    navigationHelpButton: true,
    fullscreenButton: true,
    infoBox: true,
    selectionIndicator: true,
  });

  viewer.scene.globe.baseColor =
    Color.fromCssColorString("#24435b");

  // Load Cesium ion satellite imagery.
  const imageryProvider = await IonImageryProvider.fromAssetId(2);
  viewer.imageryLayers.addImageryProvider(imageryProvider);

  // Load Cesium World Terrain.
  const terrainProvider = await createWorldTerrainAsync();
  viewer.terrainProvider = terrainProvider;

  viewer.camera.flyTo({
    destination: Cartesian3.fromDegrees(91.7362, 26.1445, 30000),
    duration: 2,
  });

  viewer.entities.add({
    name: "Guwahati City Study Area",
    position: Cartesian3.fromDegrees(91.7362, 26.1445, 0),
    point: {
      pixelSize: 12,
      color: Color.ORANGE,
      outlineColor: Color.WHITE,
      outlineWidth: 2,
      heightReference: HeightReference.CLAMP_TO_GROUND,
    },
    label: {
      text: "Guwahati City",
      font: "14px sans-serif",
      fillColor: Color.WHITE,
      outlineColor: Color.BLACK,
      outlineWidth: 3,
      style: 2,
      pixelOffset: { x: 0, y: -20 },
      heightReference: HeightReference.CLAMP_TO_GROUND,
    },
  });

  window.geoSenseViewer = viewer;
} catch (error) {
  console.error("GeoSense viewer initialization failed:", error);
  console.error("Original error message:", error?.message);
  console.error("Original error stack:", error?.stack);

  const container = document.querySelector("#cesiumContainer");

  if (container) {
    container.innerHTML = `
      <div class="error">
        <h2>Unable to initialize the 3D globe</h2>
        <p>Check the browser developer console for details.</p>
      </div>
    `;
  }
}
const ndviYearSelect = document.querySelector("#ndvi-year");
const ndviVisibleCheckbox = document.querySelector("#ndvi-visible");
const ndviStatusElement = document.querySelector("#ndvi-status");
const ndviLayers = new Map();

async function setupNdviLayers() {
  if (!viewer || viewer.isDestroyed()) {
    ndviStatusElement.textContent = "Globe is unavailable.";
    return;
  }

  try {
    const response = await fetch("/data/imagery/ndvi_bounds.json", {
      cache: "no-cache",
    });

    if (!response.ok) {
      throw new Error(`Metadata request failed: HTTP ${response.status}`);
    }

    const metadata = await response.json();

    for (const year of ["2015", "2021", "2023"]) {
      const item = metadata[year];

      if (
        !item ||
        !item.image ||
        ![item.west, item.south, item.east, item.north].every(Number.isFinite)
      ) {
        throw new Error(`Invalid NDVI metadata for ${year}.`);
      }

      const rectangle = Rectangle.fromDegrees(
        item.west,
        item.south,
        item.east,
        item.north
      );

      const provider = await SingleTileImageryProvider.fromUrl(
        `/data/imagery/${item.image}`,
        { rectangle }
      );

      const layer = viewer.imageryLayers.addImageryProvider(provider);
      layer.show = false;
      layer.alpha = 0.75;
      ndviLayers.set(year, layer);
    }

    updateNdviLayer();
    ndviStatusElement.textContent = "NDVI layers ready.";
  } catch (error) {
    console.error("NDVI layer loading failed:", error);
    ndviStatusElement.textContent = `NDVI loading failed: ${error.message}`;
  }
}

function updateNdviLayer() {
  const selectedYear = ndviYearSelect.value;
  const shouldDisplay = ndviVisibleCheckbox.checked;

  for (const [year, layer] of ndviLayers.entries()) {
    layer.show = shouldDisplay && year === selectedYear;
  }

  if (ndviLayers.size > 0) {
    ndviStatusElement.textContent = shouldDisplay
      ? `Displaying NDVI ${selectedYear}`
      : "NDVI overlay hidden";
  }

  if (viewer && !viewer.isDestroyed()) {
    viewer.scene.requestRender();
  }
}

ndviYearSelect.addEventListener("change", updateNdviLayer);
ndviVisibleCheckbox.addEventListener("change", updateNdviLayer);
setupNdviLayers();
// Exercise 4: elevation raster overlay
const elevationVisibleCheckbox =
  document.querySelector("#elevation-visible");
const elevationStatusElement =
  document.querySelector("#elevation-status");

let elevationLayer = null;

async function setupElevationLayer() {
  if (!viewer || viewer.isDestroyed()) {
    elevationStatusElement.textContent = "Globe is unavailable.";
    return;
  }

  try {
    const response = await fetch(
      "/data/terrain/elevation_bounds.json",
      { cache: "no-cache" }
    );

    if (!response.ok) {
      throw new Error(`Metadata request failed: HTTP ${response.status}`);
    }

    const metadata = await response.json();

    const bounds = [
      metadata.west,
      metadata.south,
      metadata.east,
      metadata.north,
    ];

    if (
      !metadata.image ||
      !bounds.every(Number.isFinite) ||
      metadata.west >= metadata.east ||
      metadata.south >= metadata.north
    ) {
      throw new Error("Invalid elevation bounds metadata.");
    }

    const rectangle = Rectangle.fromDegrees(...bounds);

    const provider = await SingleTileImageryProvider.fromUrl(
      `/data/terrain/${metadata.image}`,
      { rectangle }
    );

    elevationLayer = viewer.imageryLayers.addImageryProvider(provider);
    elevationLayer.show = elevationVisibleCheckbox.checked;
    elevationLayer.alpha = 0.8;

    elevationStatusElement.textContent = "Elevation overlay ready.";
  } catch (error) {
    console.error("Elevation layer loading failed:", error);
    elevationStatusElement.textContent =
      `Elevation loading failed: ${error.message}`;
  }
}

elevationVisibleCheckbox.addEventListener("change", () => {
  if (elevationLayer) {
    elevationLayer.show = elevationVisibleCheckbox.checked;
    viewer.scene.requestRender();
    elevationStatusElement.textContent =
      elevationVisibleCheckbox.checked
        ? "Elevation overlay visible."
        : "Elevation overlay hidden.";
  }
});

setupElevationLayer();
// Exercise 4: slope raster overlay
const slopeVisibleCheckbox =
  document.querySelector("#slope-visible");
const slopeStatusElement =
  document.querySelector("#slope-status");

let slopeLayer = null;

async function setupSlopeLayer() {
  if (!viewer || viewer.isDestroyed()) {
    slopeStatusElement.textContent = "Globe is unavailable.";
    return;
  }

  try {
    const response = await fetch(
      "/data/terrain/slope_bounds.json",
      { cache: "no-cache" }
    );

    if (!response.ok) {
      throw new Error(`Metadata request failed: HTTP ${response.status}`);
    }

    const metadata = await response.json();

    const bounds = [
      metadata.west,
      metadata.south,
      metadata.east,
      metadata.north,
    ];

    if (
      !metadata.image ||
      !bounds.every(Number.isFinite) ||
      metadata.west >= metadata.east ||
      metadata.south >= metadata.north
    ) {
      throw new Error("Invalid slope bounds metadata.");
    }

    const rectangle = Rectangle.fromDegrees(...bounds);

    const provider = await SingleTileImageryProvider.fromUrl(
      `/data/terrain/${metadata.image}`,
      { rectangle }
    );

    slopeLayer = viewer.imageryLayers.addImageryProvider(provider);
    slopeLayer.show = slopeVisibleCheckbox.checked;
    slopeLayer.alpha = 0.8;

    slopeStatusElement.textContent = "Slope overlay ready.";
  } catch (error) {
    console.error("Slope layer loading failed:", error);
    slopeStatusElement.textContent =
      `Slope loading failed: ${error.message}`;
  }
}

slopeVisibleCheckbox.addEventListener("change", () => {
  if (slopeLayer) {
    slopeLayer.show = slopeVisibleCheckbox.checked;
    viewer.scene.requestRender();
    slopeStatusElement.textContent =
      slopeVisibleCheckbox.checked
        ? "Slope overlay visible."
        : "Slope overlay hidden.";
  }
});

setupSlopeLayer();

const buildingsCheckbox = document.querySelector("#buildings-visible");
const buildingsStatus = document.querySelector("#buildings-status");
let buildingDataSource = null;

async function setupBuildingLayer() {
  if (!viewer || viewer.isDestroyed()) {
    buildingsStatus.textContent = "Globe is unavailable.";
    return;
  }

  try {
    
buildingDataSource = await GeoJsonDataSource.load(
  "/data/buildings/osm_buildings.geojson",
  {
    clampToGround: false,
    stroke: Color.fromCssColorString("#263238"),
    fill: Color.fromCssColorString("#e9a23b").withAlpha(0.8),
    strokeWidth: 0.5
  }
);

// Illustrative extrusion only: source data has no measured heights.
for (const entity of buildingDataSource.entities.values) {
  if (entity.polygon) {
    entity.polygon.height = 0;
    entity.polygon.extrudedHeight = 12;
    entity.polygon.heightReference = HeightReference.NONE;
    entity.polygon.extrudedHeightReference = HeightReference.NONE;
  }
}


    viewer.dataSources.add(buildingDataSource);
    buildingDataSource.show = buildingsCheckbox.checked;

    buildingsStatus.textContent =
      `${buildingDataSource.entities.values.length.toLocaleString()} OpenStreetMap footprints loaded`;

    buildingsCheckbox.addEventListener("change", () => {
      if (buildingDataSource) {
        buildingDataSource.show = buildingsCheckbox.checked;
      }
    });
  } catch (error) {
    console.error("Building GeoJSON loading failed:", error);
    buildingsStatus.textContent = `Building layer failed: ${error.message}`;
  }
}

setupBuildingLayer();

async function pollLatestResult() {
  try {
    const response = await fetch(API_URL, { cache: "no-store" });

    if (!response.ok) {
      throw new Error(`API returned HTTP ${response.status}`);
    }

    const payload = await response.json();
    statusElement.textContent = "Connected";
    statusElement.classList.add("connected");

    if (payload.status !== "ok" || !payload.result) {
      noteElement.textContent = "No agent result has been published yet.";
      return;
    }

    const result = payload.result;

    if (result.result_id === lastResultId) {
      return;
    }

    lastResultId = result.result_id;
    questionElement.textContent = result.question || "Not provided";
    answerElement.textContent = result.answer || "No answer provided";

    const selectedTools = Array.isArray(result.selected_tools)
      ? result.selected_tools
      : [];

    toolsElement.textContent = selectedTools.length
      ? selectedTools.join(", ")
      : "No tools recorded";

    const location = result.location;
    const latitude = Number(location?.latitude);
    const longitude = Number(location?.longitude);
    const hasCoordinates =
      location !== null &&
      location !== undefined &&
      Number.isFinite(latitude) &&
      Number.isFinite(longitude) &&
      latitude >= -90 &&
      latitude <= 90 &&
      longitude >= -180 &&
      longitude <= 180;

    if (hasCoordinates && viewer && !viewer.isDestroyed()) {
      viewer.entities.add({
        name: `Agent result ${result.result_id}`,
        position: Cartesian3.fromDegrees(longitude, latitude, 0),
        point: {
          pixelSize: 14,
          color: Color.CYAN,
          outlineColor: Color.WHITE,
          outlineWidth: 2,
          heightReference: HeightReference.CLAMP_TO_GROUND,
        },
        label: {
          text: "Latest Agent Result",
          font: "13px sans-serif",
          fillColor: Color.WHITE,
          outlineColor: Color.BLACK,
          outlineWidth: 3,
          style: 2,
          pixelOffset: { x: 0, y: -20 },
          heightReference: HeightReference.CLAMP_TO_GROUND,
        },
      });

      viewer.camera.flyTo({
        destination: Cartesian3.fromDegrees(longitude, latitude, 12000),
        duration: 2,
      });

      noteElement.textContent =
        `New result received at ${result.timestamp || "unknown time"}. Camera moved to its coordinates.`;
    } else {
      noteElement.textContent =
        "Result received, but it has no valid coordinates. Camera remains at Guwahati.";
    }
  } catch (error) {
    statusElement.textContent = "Disconnected";
    statusElement.classList.remove("connected");
    noteElement.textContent =
      `Unable to retrieve agent results: ${error.message}`;
  }
}

pollLatestResult();
window.geoSensePollingTimer = window.setInterval(
  pollLatestResult,
  POLL_INTERVAL_MS
);

