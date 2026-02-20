use axum::{
    extract::Query,
    http::header,
    response::{Html, IntoResponse},
    routing::get,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::net::SocketAddr;

#[path = "../physics.rs"]
mod physics;
#[path = "../atomic_data.rs"]
mod atomic_data;
#[path = "../atomic_lda.rs"]
mod atomic_lda;

use physics::{
    angular_wavefunction, generate_orbital_samples, radial_wavefunction, spherical_harmonic,
    QuantumNumbers,
};
use atomic_data::{load_element_data, symbol_for_z, ElementData, Orbital};
use atomic_lda::{load_lda_element, LdaElement, LdaOrbital};

#[derive(Deserialize)]
struct SampleQuery {
    n: Option<u32>,
    l: Option<u32>,
    m: Option<i32>,
    n2: Option<u32>,
    l2: Option<u32>,
    m2: Option<i32>,
    z: Option<u32>,
    count: Option<usize>,
    max: Option<f32>,
    mode: Option<String>,
    mix: Option<f32>,
    t: Option<f32>,
    valence_style: Option<String>,
    animated: Option<bool>,
    bubble: Option<bool>,
}

#[derive(Serialize)]
struct SampleResponse {
    n: u32,
    l: u32,
    m: i32,
    n2: Option<u32>,
    l2: Option<u32>,
    m2: Option<i32>,
    z: u32,
    count: usize,
    max_radius: f32,
    samples: Vec<[f32; 3]>,
    mode: String,
    source: String,
    note: Option<String>,
    available_orbitals: Vec<OrbitalInfo>,
    selected_orbital: Option<String>,
    selected_orbital_b: Option<String>,
    mix: Option<f32>,
    time: Option<f32>,
    psi1: Option<Vec<[f32; 2]>>,
    psi2: Option<Vec<[f32; 2]>>,
    delta_e: Option<f32>,
    signs: Option<Vec<i8>>,
}

#[derive(Serialize, Clone)]
struct OrbitalInfo {
    label: String,
    n: u32,
    l: u32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ViewMode {
    Total,
    Valence,
    Orbital,
    Superposition,
}

impl ViewMode {
    fn from_query(value: Option<&str>) -> Self {
        match value.unwrap_or("total").to_lowercase().as_str() {
            "valence" => ViewMode::Valence,
            "orbital" => ViewMode::Orbital,
            "superposition" => ViewMode::Superposition,
            _ => ViewMode::Total,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            ViewMode::Total => "total",
            ViewMode::Valence => "valence",
            ViewMode::Orbital => "orbital",
            ViewMode::Superposition => "superposition",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ValenceStyle {
    Spherical,
    Orbitals,
}

impl ValenceStyle {
    fn from_query(value: Option<&str>) -> Self {
        match value.unwrap_or("spherical").to_lowercase().as_str() {
            "orbitals" => ValenceStyle::Orbitals,
            _ => ValenceStyle::Spherical,
        }
    }
}

#[derive(Clone, Copy)]
enum RadialKind {
    R,
    Chi,
}

const INDEX_HTML: &str = r##"<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Quantum Orbitals 3D</title>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&display=swap" rel="stylesheet" />
    <style>
      html, body { margin: 0; padding: 0; height: 100%; background: #0b0c10; color: #e6e6e6; font-family: "Space Grotesk", "Segoe UI", sans-serif; }
      canvas { display: block; }
      #panel { position: absolute; top: 12px; left: 12px; width: 340px; background: rgba(10,12,16,0.9); padding: 12px; border: 1px solid #2a2f36; border-radius: 10px; box-shadow: 0 10px 28px rgba(0,0,0,0.35); }
      #infoButton { position: absolute; top: 12px; right: 12px; background: #11151b; border: 1px solid #2a2f36; color: #e6e6e6; border-radius: 8px; padding: 6px 10px; font-size: 12px; text-decoration: none; }
      #infoButton:hover { border-color: #3c6a9e; }
      .brand { font-size: 16px; font-weight: 600; letter-spacing: 0.02em; }
      .section { margin-top: 12px; padding-top: 10px; border-top: 1px solid #1f2630; }
      .section:first-of-type { margin-top: 8px; padding-top: 0; border-top: none; }
      .section-title { font-size: 11px; text-transform: uppercase; letter-spacing: 0.12em; color: #9aa3ad; margin-bottom: 6px; }
      .row { display: flex; align-items: center; gap: 6px; margin-top: 6px; flex-wrap: wrap; }
      .row label { font-size: 11px; color: #a7b0ba; min-width: 24px; }
      input, select { background: #0f141b; color: #e6e6e6; border: 1px solid #2a2f36; border-radius: 6px; padding: 4px 6px; font-size: 12px; }
      input[type="number"] { width: 58px; }
      select { flex: 1; min-width: 120px; }
      button { background: #11151b; color: #e6e6e6; border: 1px solid #2a2f36; border-radius: 6px; padding: 6px 10px; font-size: 12px; cursor: pointer; }
      button.primary { background: #1a2736; border-color: #3c6a9e; }
      button:disabled { opacity: 0.6; cursor: default; }
      #controls { margin-top: 6px; font-size: 12px; color: #9aa3ad; }
      #status { margin-top: 8px; font-size: 12px; color: #b2bac4; }
      .hint { font-size: 11px; color: #7f8895; margin-top: 6px; }
      #animControls { margin-top: 6px; display: flex; align-items: center; gap: 8px; font-size: 12px; color: #c9d1d9; }
      #animControls input[type="range"] { width: 140px; }
      #mixRow { margin-top: 6px; display: none; align-items: center; gap: 8px; font-size: 12px; color: #c9d1d9; flex-wrap: wrap; }
      #mixRow input[type="range"] { width: 140px; }
      #orbitalRow, #superRow { margin-top: 6px; display: none; align-items: center; gap: 6px; font-size: 12px; color: #c9d1d9; flex-wrap: wrap; }
      #superRow input[type="number"] { width: 58px; }
      .dropdown { position: relative; }
      .dropdown-btn { width: 100%; text-align: left; display: flex; justify-content: space-between; align-items: center; }
      .dropdown-panel { position: absolute; left: 0; right: 0; top: 100%; margin-top: 6px; background: #0f1218; border: 1px solid #2a2f36; border-radius: 8px; padding: 8px; max-height: 380px; overflow: auto; display: none; z-index: 5; }
      .dropdown-panel.open { display: block; }
      #elementSearch { width: 100%; margin-bottom: 6px; }
      #periodicGrid { display: grid; grid-template-columns: repeat(18, minmax(0, 1fr)); gap: 4px; }
      .series-label { font-size: 11px; color: #9aa3ad; margin-top: 8px; margin-bottom: 4px; }
      .series-grid { display: grid; grid-template-columns: repeat(15, minmax(0, 1fr)); gap: 4px; }
      .periodic-cell { height: 30px; display: flex; align-items: center; justify-content: center; font-size: 11px; border-radius: 6px; }
      .el-btn { background: #11151b; border: 1px solid #2a2f36; color: #e6e6e6; cursor: pointer; }
      .el-btn:hover { border-color: #3c6a9e; }
      .el-btn.active { background: #1e2a3a; border-color: #3c6a9e; color: #cbe3ff; }
      .el-empty { border: 1px dashed #1e252f; color: #3a4450; }
    </style>
  </head>
  <body>
    <a id="infoButton" href="/info">Info</a>
    <div id="panel">
      <div class="brand">Quantum Orbitals</div>
      <div class="row" style="margin-top: 8px;">
        <label>Render</label>
        <select id="renderMode">
          <option value="dots" selected>Dots</option>
          <option value="bubbles">Bubbles</option>
        </select>
      </div>

      <div class="section">
        <div class="section-title">Element</div>
        <div class="dropdown">
          <button id="elementButton" class="dropdown-btn">H Hydrogen (Z=1)</button>
          <div id="elementDropdown" class="dropdown-panel">
            <input id="elementSearch" type="text" placeholder="Filter by symbol or name" />
            <div id="periodicGrid"></div>
            <div class="series-label">Lanthanides</div>
            <div id="lanthRow" class="series-grid"></div>
            <div class="series-label">Actinides</div>
            <div id="actRow" class="series-grid"></div>
          </div>
        </div>
        <div class="row">
          <label>Z</label><input id="z" type="number" min="1" max="118" value="1" />
          <button id="go" class="primary">Generate</button>
        </div>
      </div>

      <div class="section">
        <div class="section-title">View</div>
        <div class="row">
          <label>Mode</label>
          <select id="mode">
            <option value="total" selected>Total density</option>
            <option value="valence">Valence density</option>
            <option value="orbital">Single orbital</option>
            <option value="superposition">Superposition</option>
          </select>
        </div>
        <div id="valenceRow" class="row" style="display: none;">
          <label>Valence</label>
          <select id="valenceStyle">
            <option value="spherical" selected>Spherical density</option>
            <option value="orbitals">Orbital lobes (m=0)</option>
          </select>
        </div>
        <div id="orbitalRow" class="row">
          <label>Orb</label>
          <select id="orbitalSelect"></select>
        </div>
        <div id="superRow" class="row">
          <label>Orb B</label>
          <select id="orbitalSelectB"></select>
          <label>n2</label><input id="n2" type="number" min="1" value="2" />
          <label>l2</label><input id="l2" type="number" min="0" value="1" />
          <label>m2</label><input id="m2" type="number" value="0" />
        </div>
        <div id="superPickRow" class="row" style="display: none;">
          <button id="pickPair">Pick animating pair</button>
        </div>
        <div class="row" id="quantumRow">
          <label>n</label><input id="n" type="number" min="1" value="2" />
          <label>l</label><input id="l" type="number" min="0" value="1" />
          <label>m</label><input id="m" type="number" value="0" />
        </div>
        <div id="mixRow" class="row">
          <label>mix</label>
          <input id="mix" type="range" min="0.05" max="0.95" step="0.01" value="0.50" />
          <span id="mixVal">0.50 / 0.50</span>
        </div>
        <div class="hint">Occupied orbitals shown for LDA. For H, type any n/l/m.</div>
      </div>

      <div class="section">
        <div class="section-title">Sampling</div>
        <div class="row">
          <label>cnt</label><input id="count" type="number" min="1000" step="1000" value="50000" />
          <label>max</label><input id="max" type="number" min="1" step="1" value="20" />
        </div>
      </div>

      <div class="section">
        <div class="section-title">Controls</div>
        <div id="controls">Drag to orbit - Scroll to zoom - WASD to move (bounded)</div>
        <div class="row">
          <button id="resetCamera">Reset camera</button>
        </div>
        <div id="animControls">
          <label><input id="animated" type="checkbox" /> Animated (time evolution)</label>
          <label>Speed</label>
          <input id="animSpeed" type="range" min="0" max="3" step="0.05" value="1" />
          <span id="animSpeedVal">1.00x</span>
        </div>
      </div>

      <div id="status">Ready.</div>
    </div>
    <script type="importmap">
      {
        "imports": {
          "three": "/static/three.module.js"
        }
      }
    </script>
    <script type="module">
      import * as THREE from "/static/three.module.js";
      import { MarchingCubes } from "/static/MarchingCubes.js";

      const statusEl = document.getElementById("status");
      const elementButton = document.getElementById("elementButton");
      const elementDropdown = document.getElementById("elementDropdown");
      const elementSearch = document.getElementById("elementSearch");
      const periodicGrid = document.getElementById("periodicGrid");
      const lanthRow = document.getElementById("lanthRow");
      const actRow = document.getElementById("actRow");
      const zInput = document.getElementById("z");
      const orbitalRow = document.getElementById("orbitalRow");
      const orbitalSelect = document.getElementById("orbitalSelect");
      const orbitalSelectB = document.getElementById("orbitalSelectB");
      const superRow = document.getElementById("superRow");
      const superPickRow = document.getElementById("superPickRow");
      const mixRow = document.getElementById("mixRow");
      const mixInput = document.getElementById("mix");
      const mixVal = document.getElementById("mixVal");
      const modeSelect = document.getElementById("mode");
      const renderModeSelect = document.getElementById("renderMode");
      const valenceRow = document.getElementById("valenceRow");
      const valenceStyleSelect = document.getElementById("valenceStyle");
      const countInput = document.getElementById("count");
      const maxInput = document.getElementById("max");
      const nInput = document.getElementById("n");
      const lInput = document.getElementById("l");
      const mInput = document.getElementById("m");
      const n2Input = document.getElementById("n2");
      const l2Input = document.getElementById("l2");
      const m2Input = document.getElementById("m2");
      const pickPairButton = document.getElementById("pickPair");
      const resetCameraButton = document.getElementById("resetCamera");
      const animControls = document.getElementById("animControls");
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x0b0c10);

      const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.01, 100);
      camera.position.set(0, 0, 8);

      const renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setSize(window.innerWidth, window.innerHeight);
      document.body.appendChild(renderer.domElement);

      const group = new THREE.Group();
      scene.add(group);

      const circleTexture = (() => {
        const size = 64;
        const canvas = document.createElement("canvas");
        canvas.width = size;
        canvas.height = size;
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, size, size);
        ctx.beginPath();
        ctx.arc(size / 2, size / 2, size / 2 - 1, 0, Math.PI * 2);
        ctx.fillStyle = "#ffffff";
        ctx.fill();
        const tex = new THREE.CanvasTexture(canvas);
        tex.generateMipmaps = false;
        tex.minFilter = THREE.LinearFilter;
        tex.magFilter = THREE.LinearFilter;
        return tex;
      })();

      const animToggle = document.getElementById("animated");
      const animSpeedInput = document.getElementById("animSpeed");
      const animSpeedVal = document.getElementById("animSpeedVal");
      let points = null;
      let posAttr = null;
      let animateEnabled = animToggle.checked;
      let animSpeed = 1.0;
      let superpositionTime = 0.0;
      let superFetchInFlight = false;
      let superPsi = null;
      let superProb = null;
      let baseColors = null;
      let colorAttr = null;
      let animFrom = null;
      let animTo = null;
      let animStart = 0;
      let animDurationMs = 600;
      let lastSampleTime = 0;
      let lastOrbitals = [];
      let renderMode = "dots";
      let bubbleGroup = null;
      let bubblePos = null;
      let bubbleNeg = null;
      let bubbleLightsAdded = false;
      let lastSigns = null;
      let lastExtent = 1.0;
      let lastBubbleUpdate = 0;
      let bubbleDirty = false;
      const bubbleSampleTarget = 5000;
      const bubbleResolution = 48;
      const bubbleKernelRadius = 2;
      const bubbleKernelSigma = 0.9;
      const bubbleIsoFraction = 0.3;
      const bubbleUpdateInterval = 60;

      const bubbleKernel = (() => {
        const entries = [];
        const sigma2 = bubbleKernelSigma * bubbleKernelSigma;
        for (let dz = -bubbleKernelRadius; dz <= bubbleKernelRadius; dz++) {
          for (let dy = -bubbleKernelRadius; dy <= bubbleKernelRadius; dy++) {
            for (let dx = -bubbleKernelRadius; dx <= bubbleKernelRadius; dx++) {
              const d2 = dx * dx + dy * dy + dz * dz;
              const w = Math.exp(-d2 / (2 * sigma2));
              if (w > 0.02) {
                entries.push([dx, dy, dz, w]);
              }
            }
          }
        }
        return entries;
      })();

      function updateAnimUI() {
        animSpeedVal.textContent = animSpeed.toFixed(2) + "x";
        animSpeedInput.disabled = !animateEnabled || animToggle.disabled;
      }

      function initBubbles() {
        if (bubbleGroup) return;
        bubbleGroup = new THREE.Group();
        const matPos = new THREE.MeshStandardMaterial({ color: 0xff3b4a, transparent: true, opacity: 0.75, roughness: 0.35, metalness: 0.0, side: THREE.DoubleSide });
        const matNeg = new THREE.MeshStandardMaterial({ color: 0x3b5bff, transparent: true, opacity: 0.75, roughness: 0.35, metalness: 0.0, side: THREE.DoubleSide });
        bubblePos = new MarchingCubes(bubbleResolution, matPos, true, true);
        bubbleNeg = new MarchingCubes(bubbleResolution, matNeg, true, true);
        bubblePos.isolation = 0.06;
        bubbleNeg.isolation = 0.06;
        bubbleGroup.add(bubblePos);
        bubbleGroup.add(bubbleNeg);
        bubbleGroup.visible = false;
        scene.add(bubbleGroup);

        if (!bubbleLightsAdded) {
          const ambient = new THREE.AmbientLight(0xffffff, 0.5);
          const dir = new THREE.DirectionalLight(0xffffff, 0.6);
          dir.position.set(1, 1, 1);
          scene.add(ambient);
          scene.add(dir);
          bubbleLightsAdded = true;
        }
      }

      function updateRenderMode() {
        renderMode = renderModeSelect.value;
        localStorage.setItem("renderMode", renderMode);
        const showBubbles = renderMode === "bubbles";
        if (points) {
          points.visible = !showBubbles;
        }
        if (showBubbles) {
          initBubbles();
          bubbleGroup.visible = true;
          if (posAttr) {
            updateBubblesFromPositions(posAttr.array, lastSigns);
          }
        } else if (bubbleGroup) {
          bubbleGroup.visible = false;
        }
      }

      function updateBubblesFromPositions(arr, signs) {
        if (!bubbleGroup || !bubblePos || !bubbleNeg) return;
        const extent = Math.max(lastExtent, 1e-4);
        bubblePos.reset();
        bubbleNeg.reset();
        bubblePos.scale.setScalar(extent * 2.0);
        bubbleNeg.scale.setScalar(extent * 2.0);
        bubblePos.position.set(0, 0, 0);
        bubbleNeg.position.set(0, 0, 0);

        const size = bubbleResolution;
        const size2 = size * size;
        const fieldPos = bubblePos.field;
        const fieldNeg = bubbleNeg.field;
        const count = Math.floor(arr.length / 3);
        if (count === 0) return;
        const step = Math.max(1, Math.floor(count / bubbleSampleTarget));
        const scale = (size - 1) / (2.0 * extent);
        const useSigns = signs && signs.length === count;
        let posCount = 0;
        let negCount = 0;
        let maxPos = 0.0;
        let maxNeg = 0.0;

        for (let i = 0; i < count; i += step) {
          const idx = i * 3;
          const gx = (arr[idx + 0] + extent) * scale;
          const gy = (arr[idx + 1] + extent) * scale;
          const gz = (arr[idx + 2] + extent) * scale;
          const ix = Math.round(gx);
          const iy = Math.round(gy);
          const iz = Math.round(gz);
          if (ix < 0 || ix >= size || iy < 0 || iy >= size || iz < 0 || iz >= size) {
            continue;
          }
          const sign = useSigns ? signs[i] : 1;
          const kernel = bubbleKernel;
          if (sign >= 0) {
            posCount++;
            for (let k = 0; k < kernel.length; k++) {
              const dx = kernel[k][0];
              const dy = kernel[k][1];
              const dz = kernel[k][2];
              const x = ix + dx;
              const y = iy + dy;
              const z = iz + dz;
              if (x < 0 || x >= size || y < 0 || y >= size || z < 0 || z >= size) continue;
              const offset = x + size * y + size2 * z;
              const v = fieldPos[offset] + kernel[k][3];
              fieldPos[offset] = v;
              if (v > maxPos) maxPos = v;
            }
          } else {
            negCount++;
            for (let k = 0; k < kernel.length; k++) {
              const dx = kernel[k][0];
              const dy = kernel[k][1];
              const dz = kernel[k][2];
              const x = ix + dx;
              const y = iy + dy;
              const z = iz + dz;
              if (x < 0 || x >= size || y < 0 || y >= size || z < 0 || z >= size) continue;
              const offset = x + size * y + size2 * z;
              const v = fieldNeg[offset] + kernel[k][3];
              fieldNeg[offset] = v;
              if (v > maxNeg) maxNeg = v;
            }
          }
        }
        bubblePos.isolation = maxPos > 0 ? maxPos * bubbleIsoFraction : 1.0;
        bubbleNeg.isolation = maxNeg > 0 ? maxNeg * bubbleIsoFraction : 1.0;
        bubblePos.visible = posCount > 0 && maxPos > 0;
        bubbleNeg.visible = negCount > 0 && maxNeg > 0;
        bubblePos.update();
        bubbleNeg.update();
        bubbleDirty = false;
      }

      function updateMixUI() {
        const mix = Number(mixInput.value);
        const a = mix.toFixed(2);
        const b = (1.0 - mix).toFixed(2);
        mixVal.textContent = `${a} / ${b}`;
      }

      function packPsi(arr) {
        if (!Array.isArray(arr)) return null;
        const out = new Float32Array(arr.length * 2);
        for (let i = 0; i < arr.length; i++) {
          const pair = arr[i];
          out[i * 2 + 0] = pair[0] ?? 0;
          out[i * 2 + 1] = pair[1] ?? 0;
        }
        return out;
      }

      animSpeedInput.addEventListener("input", () => {
        animSpeed = Number(animSpeedInput.value);
        updateAnimUI();
      });

      animToggle.addEventListener("change", () => {
        animateEnabled = animToggle.checked;
        superpositionTime = 0.0;
        if (animateEnabled && modeSelect.value === "superposition") {
          superPsi = null;
          superFetchInFlight = false;
          animFrom = null;
          animTo = null;
          lastSampleTime = 0;
          if (Number(n2Input.value) === Number(nInput.value)) {
            pickPairButton.click();
          } else {
            fetchSamples().catch((err) => { statusEl.textContent = err.toString(); });
          }
        }
        updateAnimUI();
      });

      renderMode = localStorage.getItem("renderMode") || "dots";
      renderModeSelect.value = renderMode;
      renderModeSelect.addEventListener("change", () => {
        updateRenderMode();
        fetchSamples().catch((err) => { statusEl.textContent = err.toString(); });
      });

      updateAnimUI();
      updateMixUI();
      updateRenderMode();

      function updateOrbitalList(list, selectedLabel, selectedLabelB) {
        lastOrbitals = Array.isArray(list) ? list : [];
        orbitalSelect.innerHTML = "";
        orbitalSelectB.innerHTML = "";
        if (!list || list.length === 0) {
          orbitalRow.style.display = "none";
          superRow.style.display = modeSelect.value === "superposition" ? "flex" : "none";
          return;
        }
        const mode = modeSelect.value;
        const showOrbital = mode === "orbital" || mode === "superposition";
        orbitalRow.style.display = showOrbital ? "flex" : "none";
        superRow.style.display = mode === "superposition" ? "flex" : "none";
        for (const orb of list) {
          const opt = document.createElement("option");
          opt.value = `${orb.n},${orb.l},${orb.label}`;
          opt.textContent = `${orb.label} (n=${orb.n}, l=${orb.l})`;
          if (selectedLabel && orb.label === selectedLabel) {
            opt.selected = true;
          }
          orbitalSelect.appendChild(opt);
          const optB = document.createElement("option");
          optB.value = opt.value;
          optB.textContent = opt.textContent;
          if (selectedLabelB && orb.label === selectedLabelB) {
            optB.selected = true;
          }
          orbitalSelectB.appendChild(optB);
        }

        if (!selectedLabel && orbitalSelect.options.length > 0) {
          orbitalSelect.selectedIndex = 0;
        }
        if (!selectedLabelB && orbitalSelectB.options.length > 1) {
          orbitalSelectB.selectedIndex = 1;
        } else if (!selectedLabelB && orbitalSelectB.options.length > 0) {
          orbitalSelectB.selectedIndex = 0;
        }

        const [nStr, lStr] = orbitalSelect.value.split(",", 2);
        if (nStr && lStr) {
          nInput.value = nStr;
          lInput.value = lStr;
        }
        const [n2Str, l2Str] = orbitalSelectB.value.split(",", 2);
        if (n2Str && l2Str) {
          n2Input.value = n2Str;
          l2Input.value = l2Str;
        }
      }

      function updateModeUI() {
        const mode = modeSelect.value;
        const orbitalMode = mode === "orbital";
        const superMode = mode === "superposition";
        valenceRow.style.display = mode === "valence" ? "flex" : "none";
        nInput.disabled = !(orbitalMode || superMode);
        lInput.disabled = !(orbitalMode || superMode);
        mInput.disabled = !(orbitalMode || superMode);
        n2Input.disabled = !superMode;
        l2Input.disabled = !superMode;
        m2Input.disabled = !superMode;
        mixInput.disabled = !superMode;
        mixRow.style.display = superMode ? "flex" : "none";
        superPickRow.style.display = superMode ? "flex" : "none";
        if (!orbitalMode && !superMode) {
          orbitalRow.style.display = "none";
          superRow.style.display = "none";
        }
        if (!superMode && animateEnabled) {
          animateEnabled = false;
          animToggle.checked = false;
        }
        animToggle.disabled = !superMode;
        animControls.style.display = superMode ? "flex" : "none";
        updateAnimUI();
        updateMixUI();
      }

      orbitalSelect.addEventListener("change", () => {
        const [nStr, lStr] = orbitalSelect.value.split(",", 2);
        if (nStr && lStr) {
          nInput.value = nStr;
          lInput.value = lStr;
          fetchSamples().catch((err) => { statusEl.textContent = err.toString(); });
        }
      });
      orbitalSelectB.addEventListener("change", () => {
        const [n2Str, l2Str] = orbitalSelectB.value.split(",", 2);
        if (n2Str && l2Str) {
          n2Input.value = n2Str;
          l2Input.value = l2Str;
        }
        fetchSamples().catch((err) => { statusEl.textContent = err.toString(); });
      });
      pickPairButton.addEventListener("click", () => {
        const nA = Number(nInput.value) || 1;
        const lA = Number(lInput.value) || 0;
        let chosen = null;
        if (lastOrbitals.length > 0) {
          chosen = lastOrbitals.find((o) => o.n !== nA) || null;
          if (!chosen) {
            chosen = lastOrbitals.find((o) => o.l !== lA) || null;
          }
          if (!chosen) {
            chosen = lastOrbitals[0];
          }
        }
        if (chosen) {
          const targetValue = `${chosen.n},${chosen.l},${chosen.label}`;
          const opt = Array.from(orbitalSelectB.options).find((o) => o.value === targetValue);
          if (opt) {
            orbitalSelectB.value = targetValue;
          }
          n2Input.value = chosen.n;
          l2Input.value = chosen.l;
        } else {
          const n2 = nA + 1;
          const l2 = Math.min(1, n2 - 1);
          n2Input.value = n2;
          l2Input.value = l2;
        }
        m2Input.value = 0;
        fetchSamples().catch((err) => { statusEl.textContent = err.toString(); });
      });
      n2Input.addEventListener("change", () => {
        fetchSamples().catch((err) => { statusEl.textContent = err.toString(); });
      });
      l2Input.addEventListener("change", () => {
        fetchSamples().catch((err) => { statusEl.textContent = err.toString(); });
      });
      m2Input.addEventListener("change", () => {
        fetchSamples().catch((err) => { statusEl.textContent = err.toString(); });
      });
      mixInput.addEventListener("input", () => {
        updateMixUI();
        fetchSamples().catch((err) => { statusEl.textContent = err.toString(); });
      });
      resetCameraButton.addEventListener("click", () => {
        resetCamera();
      });
      valenceStyleSelect.addEventListener("change", () => {
        fetchSamples().catch((err) => { statusEl.textContent = err.toString(); });
      });
      modeSelect.addEventListener("change", () => {
        updateModeUI();
        superpositionTime = 0.0;
        superFetchInFlight = false;
        superPsi = null;
        animFrom = null;
        animTo = null;
        lastSampleTime = 0;
        resetCamera();
        fetchSamples().catch((err) => { statusEl.textContent = err.toString(); });
      });
      const ELEMENTS = [
        { Z: 1, symbol: "H", name: "Hydrogen" },
        { Z: 2, symbol: "He", name: "Helium" },
        { Z: 3, symbol: "Li", name: "Lithium" },
        { Z: 4, symbol: "Be", name: "Beryllium" },
        { Z: 5, symbol: "B", name: "Boron" },
        { Z: 6, symbol: "C", name: "Carbon" },
        { Z: 7, symbol: "N", name: "Nitrogen" },
        { Z: 8, symbol: "O", name: "Oxygen" },
        { Z: 9, symbol: "F", name: "Fluorine" },
        { Z: 10, symbol: "Ne", name: "Neon" },
        { Z: 11, symbol: "Na", name: "Sodium" },
        { Z: 12, symbol: "Mg", name: "Magnesium" },
        { Z: 13, symbol: "Al", name: "Aluminum" },
        { Z: 14, symbol: "Si", name: "Silicon" },
        { Z: 15, symbol: "P", name: "Phosphorus" },
        { Z: 16, symbol: "S", name: "Sulfur" },
        { Z: 17, symbol: "Cl", name: "Chlorine" },
        { Z: 18, symbol: "Ar", name: "Argon" },
        { Z: 19, symbol: "K", name: "Potassium" },
        { Z: 20, symbol: "Ca", name: "Calcium" },
        { Z: 21, symbol: "Sc", name: "Scandium" },
        { Z: 22, symbol: "Ti", name: "Titanium" },
        { Z: 23, symbol: "V", name: "Vanadium" },
        { Z: 24, symbol: "Cr", name: "Chromium" },
        { Z: 25, symbol: "Mn", name: "Manganese" },
        { Z: 26, symbol: "Fe", name: "Iron" },
        { Z: 27, symbol: "Co", name: "Cobalt" },
        { Z: 28, symbol: "Ni", name: "Nickel" },
        { Z: 29, symbol: "Cu", name: "Copper" },
        { Z: 30, symbol: "Zn", name: "Zinc" },
        { Z: 31, symbol: "Ga", name: "Gallium" },
        { Z: 32, symbol: "Ge", name: "Germanium" },
        { Z: 33, symbol: "As", name: "Arsenic" },
        { Z: 34, symbol: "Se", name: "Selenium" },
        { Z: 35, symbol: "Br", name: "Bromine" },
        { Z: 36, symbol: "Kr", name: "Krypton" },
        { Z: 37, symbol: "Rb", name: "Rubidium" },
        { Z: 38, symbol: "Sr", name: "Strontium" },
        { Z: 39, symbol: "Y", name: "Yttrium" },
        { Z: 40, symbol: "Zr", name: "Zirconium" },
        { Z: 41, symbol: "Nb", name: "Niobium" },
        { Z: 42, symbol: "Mo", name: "Molybdenum" },
        { Z: 43, symbol: "Tc", name: "Technetium" },
        { Z: 44, symbol: "Ru", name: "Ruthenium" },
        { Z: 45, symbol: "Rh", name: "Rhodium" },
        { Z: 46, symbol: "Pd", name: "Palladium" },
        { Z: 47, symbol: "Ag", name: "Silver" },
        { Z: 48, symbol: "Cd", name: "Cadmium" },
        { Z: 49, symbol: "In", name: "Indium" },
        { Z: 50, symbol: "Sn", name: "Tin" },
        { Z: 51, symbol: "Sb", name: "Antimony" },
        { Z: 52, symbol: "Te", name: "Tellurium" },
        { Z: 53, symbol: "I", name: "Iodine" },
        { Z: 54, symbol: "Xe", name: "Xenon" },
        { Z: 55, symbol: "Cs", name: "Cesium" },
        { Z: 56, symbol: "Ba", name: "Barium" },
        { Z: 57, symbol: "La", name: "Lanthanum" },
        { Z: 58, symbol: "Ce", name: "Cerium" },
        { Z: 59, symbol: "Pr", name: "Praseodymium" },
        { Z: 60, symbol: "Nd", name: "Neodymium" },
        { Z: 61, symbol: "Pm", name: "Promethium" },
        { Z: 62, symbol: "Sm", name: "Samarium" },
        { Z: 63, symbol: "Eu", name: "Europium" },
        { Z: 64, symbol: "Gd", name: "Gadolinium" },
        { Z: 65, symbol: "Tb", name: "Terbium" },
        { Z: 66, symbol: "Dy", name: "Dysprosium" },
        { Z: 67, symbol: "Ho", name: "Holmium" },
        { Z: 68, symbol: "Er", name: "Erbium" },
        { Z: 69, symbol: "Tm", name: "Thulium" },
        { Z: 70, symbol: "Yb", name: "Ytterbium" },
        { Z: 71, symbol: "Lu", name: "Lutetium" },
        { Z: 72, symbol: "Hf", name: "Hafnium" },
        { Z: 73, symbol: "Ta", name: "Tantalum" },
        { Z: 74, symbol: "W", name: "Tungsten" },
        { Z: 75, symbol: "Re", name: "Rhenium" },
        { Z: 76, symbol: "Os", name: "Osmium" },
        { Z: 77, symbol: "Ir", name: "Iridium" },
        { Z: 78, symbol: "Pt", name: "Platinum" },
        { Z: 79, symbol: "Au", name: "Gold" },
        { Z: 80, symbol: "Hg", name: "Mercury" },
        { Z: 81, symbol: "Tl", name: "Thallium" },
        { Z: 82, symbol: "Pb", name: "Lead" },
        { Z: 83, symbol: "Bi", name: "Bismuth" },
        { Z: 84, symbol: "Po", name: "Polonium" },
        { Z: 85, symbol: "At", name: "Astatine" },
        { Z: 86, symbol: "Rn", name: "Radon" },
        { Z: 87, symbol: "Fr", name: "Francium" },
        { Z: 88, symbol: "Ra", name: "Radium" },
        { Z: 89, symbol: "Ac", name: "Actinium" },
        { Z: 90, symbol: "Th", name: "Thorium" },
        { Z: 91, symbol: "Pa", name: "Protactinium" },
        { Z: 92, symbol: "U", name: "Uranium" },
        { Z: 93, symbol: "Np", name: "Neptunium" },
        { Z: 94, symbol: "Pu", name: "Plutonium" },
        { Z: 95, symbol: "Am", name: "Americium" },
        { Z: 96, symbol: "Cm", name: "Curium" },
        { Z: 97, symbol: "Bk", name: "Berkelium" },
        { Z: 98, symbol: "Cf", name: "Californium" },
        { Z: 99, symbol: "Es", name: "Einsteinium" },
        { Z: 100, symbol: "Fm", name: "Fermium" },
        { Z: 101, symbol: "Md", name: "Mendelevium" },
        { Z: 102, symbol: "No", name: "Nobelium" },
        { Z: 103, symbol: "Lr", name: "Lawrencium" },
        { Z: 104, symbol: "Rf", name: "Rutherfordium" },
        { Z: 105, symbol: "Db", name: "Dubnium" },
        { Z: 106, symbol: "Sg", name: "Seaborgium" },
        { Z: 107, symbol: "Bh", name: "Bohrium" },
        { Z: 108, symbol: "Hs", name: "Hassium" },
        { Z: 109, symbol: "Mt", name: "Meitnerium" },
        { Z: 110, symbol: "Ds", name: "Darmstadtium" },
        { Z: 111, symbol: "Rg", name: "Roentgenium" },
        { Z: 112, symbol: "Cn", name: "Copernicium" },
        { Z: 113, symbol: "Nh", name: "Nihonium" },
        { Z: 114, symbol: "Fl", name: "Flerovium" },
        { Z: 115, symbol: "Mc", name: "Moscovium" },
        { Z: 116, symbol: "Lv", name: "Livermorium" },
        { Z: 117, symbol: "Ts", name: "Tennessine" },
        { Z: 118, symbol: "Og", name: "Oganesson" },
      ];
      const ELEMENT_BY_SYMBOL = new Map(ELEMENTS.map((el) => [el.symbol, el]));
      const PERIODIC_LAYOUT = [
        ["H", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "He"],
        ["Li", "Be", "", "", "", "", "", "", "", "", "", "", "B", "C", "N", "O", "F", "Ne"],
        ["Na", "Mg", "", "", "", "", "", "", "", "", "", "", "Al", "Si", "P", "S", "Cl", "Ar"],
        ["K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr"],
        ["Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe"],
        ["Cs", "Ba", "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn"],
        ["Fr", "Ra", "Ac", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"],
      ];
      const LANTHANIDES = ["Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"];
      const ACTINIDES = ["Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"];

      const elementButtons = new Map();

      function updateElementButton(el) {
        elementButton.textContent = `${el.symbol} ${el.name} (Z=${el.Z})`;
      }

      function setActiveElementByZ(z) {
        const el = ELEMENTS.find((item) => item.Z === z);
        if (!el) return;
        updateElementButton(el);
        zInput.value = el.Z;
        for (const [symbol, btn] of elementButtons.entries()) {
          btn.classList.toggle("active", symbol === el.symbol);
        }
      }

      function createElementButton(symbol) {
        const el = ELEMENT_BY_SYMBOL.get(symbol);
        if (!el) {
          const empty = document.createElement("div");
          empty.className = "periodic-cell el-empty";
          return empty;
        }
        const btn = document.createElement("button");
        btn.className = "periodic-cell el-btn";
        btn.textContent = el.symbol;
        btn.title = `${el.Z} ${el.name}`;
        btn.dataset.symbol = el.symbol;
        btn.dataset.name = el.name.toLowerCase();
        btn.addEventListener("click", () => {
          setActiveElementByZ(el.Z);
          elementDropdown.classList.remove("open");
          resetCamera();
          fetchSamples().catch((err) => { statusEl.textContent = err.toString(); });
        });
        elementButtons.set(el.symbol, btn);
        return btn;
      }

      function renderPeriodicTable() {
        periodicGrid.innerHTML = "";
        elementButtons.clear();
        for (const row of PERIODIC_LAYOUT) {
          for (const symbol of row) {
            if (!symbol) {
              const empty = document.createElement("div");
              empty.className = "periodic-cell el-empty";
              periodicGrid.appendChild(empty);
            } else {
              periodicGrid.appendChild(createElementButton(symbol));
            }
          }
        }
        lanthRow.innerHTML = "";
        for (const symbol of LANTHANIDES) {
          lanthRow.appendChild(createElementButton(symbol));
        }
        actRow.innerHTML = "";
        for (const symbol of ACTINIDES) {
          actRow.appendChild(createElementButton(symbol));
        }
        setActiveElementByZ(Number(zInput.value));
      }

      elementButton.addEventListener("click", (e) => {
        e.stopPropagation();
        elementDropdown.classList.toggle("open");
      });

      document.addEventListener("click", (e) => {
        if (!elementDropdown.contains(e.target) && e.target !== elementButton) {
          elementDropdown.classList.remove("open");
        }
      });

      elementSearch.addEventListener("input", () => {
        const q = elementSearch.value.trim().toLowerCase();
        for (const btn of elementButtons.values()) {
          const symbol = (btn.dataset.symbol || "").toLowerCase();
          const name = (btn.dataset.name || "").toLowerCase();
          const match = q === "" || symbol.includes(q) || name.includes(q);
          btn.style.display = match ? "" : "none";
        }
      });

      renderPeriodicTable();
      updateModeUI();

      const target = new THREE.Vector3(0, 0, 0);
      const defaultDistance = 8;
      const defaultTheta = 1.2;
      const defaultPhi = 0.8;
      let distance = defaultDistance;
      let theta = defaultTheta;
      let phi = defaultPhi;
      const maxDistance = 20.0;
      const minDistance = 0.001;
      const maxMove = 4.5;
      const keys = new Set();
      const tmpForward = new THREE.Vector3();
      const tmpRight = new THREE.Vector3();
      const tmpMove = new THREE.Vector3();
      const up = new THREE.Vector3(0, 1, 0);

      function updateCamera() {
        const sinTheta = Math.sin(theta);
        camera.position.set(
          target.x + distance * sinTheta * Math.cos(phi),
          target.y + distance * Math.cos(theta),
          target.z + distance * sinTheta * Math.sin(phi)
        );
        camera.lookAt(target);
      }

      function resetCamera() {
        target.set(0, 0, 0);
        distance = defaultDistance;
        theta = defaultTheta;
        phi = defaultPhi;
        updateCamera();
      }

      updateCamera();

      let dragging = false;
      let lastX = 0;
      let lastY = 0;

      renderer.domElement.addEventListener("pointerdown", (e) => {
        if (e.button !== 0) {
          return;
        }
        dragging = true;
        lastX = e.clientX;
        lastY = e.clientY;
        renderer.domElement.setPointerCapture(e.pointerId);
      });

      renderer.domElement.addEventListener("pointermove", (e) => {
        if (!dragging) {
          return;
        }
        const dx = e.clientX - lastX;
        const dy = e.clientY - lastY;
        lastX = e.clientX;
        lastY = e.clientY;
        phi -= dx * 0.005;
        theta -= dy * 0.005;
        const eps = 0.1;
        theta = Math.max(eps, Math.min(Math.PI - eps, theta));
        updateCamera();
      });

      renderer.domElement.addEventListener("pointerup", (e) => {
        if (dragging) {
          dragging = false;
          renderer.domElement.releasePointerCapture(e.pointerId);
        }
      });

      renderer.domElement.addEventListener("pointerleave", () => {
        dragging = false;
      });

      renderer.domElement.addEventListener("wheel", (e) => {
        e.preventDefault();
        const delta = Math.max(-200, Math.min(200, e.deltaY));
        const zoom = Math.exp(delta * 0.001);
        distance = distance * zoom;
        if (distance > maxDistance) distance = maxDistance;
        if (distance < minDistance) distance = minDistance;
        updateCamera();
      }, { passive: false });

      function isTyping() {
        const el = document.activeElement;
        return el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA");
      }

      window.addEventListener("keydown", (e) => {
        if (isTyping()) {
          return;
        }
        keys.add(e.code);
      });

      window.addEventListener("keyup", (e) => {
        keys.delete(e.code);
      });

      function colorForDistance(d, max) {
        const t = Math.min(d / max, 1.0);
        if (t < 0.25) {
          const k = t / 0.25;
          return new THREE.Color(0, k, 1);
        } else if (t < 0.5) {
          const k = (t - 0.25) / 0.25;
          return new THREE.Color(0, 1, 1 - k);
        } else if (t < 0.75) {
          const k = (t - 0.5) / 0.25;
          return new THREE.Color(k, 1, 0);
        } else {
          const k = (t - 0.75) / 0.25;
          return new THREE.Color(1, 1 - k, 0);
        }
      }

      function updateSuperpositionColors() {
        if (!superPsi || !colorAttr || !baseColors || !superProb) {
          return;
        }
        const psi1 = superPsi.psi1;
        const psi2 = superPsi.psi2;
        const deltaE = superPsi.deltaE || 0;
        const isDegenerate = Math.abs(deltaE) < 1e-6;
        const loopPeriod = 6.0;
        const phaseSpeed = isDegenerate ? (Math.PI * 2.0 / loopPeriod) : deltaE;
        const phase = phaseSpeed * superpositionTime;
        const phaseRe = Math.cos(phase);
        const phaseIm = -Math.sin(phase);
        let maxProb = 0.0;
        const count = superProb.length;
        for (let i = 0; i < count; i++) {
          const idx2 = i * 2;
          const psi1Re = psi1[idx2 + 0];
          const psi1Im = psi1[idx2 + 1];
          const psi2ReBase = psi2[idx2 + 0];
          const psi2ImBase = psi2[idx2 + 1];
          const psi2Re = psi2ReBase * phaseRe - psi2ImBase * phaseIm;
          const psi2Im = psi2ReBase * phaseIm + psi2ImBase * phaseRe;
          const re = psi1Re + psi2Re;
          const im = psi1Im + psi2Im;
          const prob = re * re + im * im;
          superProb[i] = prob;
          if (prob > maxProb) {
            maxProb = prob;
          }
        }
        if (isDegenerate) {
          if (!superPsi.baseMax || superPsi.baseMax <= 0) {
            superPsi.baseMax = maxProb;
          }
          maxProb = superPsi.baseMax || maxProb;
        }
        const invMax = maxProb > 0 ? 1.0 / maxProb : 1.0;
        const colors = colorAttr.array;
        for (let i = 0; i < count; i++) {
          const baseIdx = i * 3;
        const norm = Math.pow(superProb[i] * invMax, 0.5);
        const brightness = 0.05 + 0.95 * norm;
          colors[baseIdx + 0] = baseColors[baseIdx + 0] * brightness;
          colors[baseIdx + 1] = baseColors[baseIdx + 1] * brightness;
          colors[baseIdx + 2] = baseColors[baseIdx + 2] * brightness;
        }
        colorAttr.needsUpdate = true;
      }

      async function fetchSamples(forceTime = null, countOverride = null) {
        const n = Number(nInput.value);
        const l = Number(lInput.value);
        const m = Number(mInput.value);
        const m2 = Number(m2Input.value);
        const z = Number(zInput.value);
        const count = countOverride !== null ? countOverride : Number(countInput.value);
        const max = Number(maxInput.value);
        const mode = modeSelect.value;
        const valenceStyle = valenceStyleSelect.value;
        const wantMorph = animateEnabled && mode === "superposition";
        const wantPsi = false;
        const wantBubbles = renderMode === "bubbles";
        let effectiveCount = count;
        if (wantMorph) {
          effectiveCount = Math.min(count, 20000);
        }
        let n2 = Number(n2Input.value);
        let l2 = Number(l2Input.value);
        if ((!n2 || !l2) && orbitalSelectB.value) {
          const [n2Str, l2Str] = orbitalSelectB.value.split(",", 2);
          n2 = Number(n2Str);
          l2 = Number(l2Str);
        }
        const mix = Number(mixInput.value);
        const t = forceTime !== null ? forceTime : superpositionTime;

        if (wantMorph) {
          superFetchInFlight = true;
        }
        try {
          statusEl.textContent = forceTime !== null ? "Animating..." : "Sampling...";
          setActiveElementByZ(z);
          const params = new URLSearchParams({ n, l, m, n2, l2, m2, z, count: effectiveCount, max, mode, mix, t, valence_style: valenceStyle, animated: wantPsi, bubble: wantBubbles });
          const res = await fetch(`/samples?${params.toString()}`);
          if (!res.ok) {
            statusEl.textContent = "Error: " + res.status;
            return;
          }
          const data = await res.json();
        if (data.mode && data.mode !== modeSelect.value) {
          modeSelect.value = data.mode;
          updateModeUI();
        }
        if (data.mode === "orbital") {
          nInput.value = data.n;
          lInput.value = data.l;
          mInput.value = data.m;
        }
        const element = ELEMENTS.find((el) => el.Z === data.z);
        const elementLabel = element ? `${element.symbol} ${element.name}` : `Z=${data.z}`;
        const sourceLabel = data.source === "openmx_lda"
          ? "OpenMX LDA"
          : (data.source === "pslibrary" ? "PSlibrary" : "Hydrogenic");
        const note = data.note ? ` | ${data.note}` : "";
        const modeLabel = data.mode || mode;
        let detail = "total density";
        if (modeLabel === "valence") {
          detail = "valence density";
        } else if (modeLabel === "orbital") {
          detail = `${data.selected_orbital || "orbital"} (n=${data.n} l=${data.l} m=${data.m})`;
        } else if (modeLabel === "superposition") {
          const orbA = data.selected_orbital || `${data.n}l=${data.l}`;
          const orbB = data.selected_orbital_b || `${data.n2 ?? "?"}l=${data.l2 ?? "?"}`;
          const mixValText = data.mix ? data.mix.toFixed(2) : mix.toFixed(2);
          detail = `superposition ${orbA} + ${orbB} (mix ${mixValText})`;
        }
        statusEl.textContent = `${elementLabel} | ${detail} | count=${data.count} | ${sourceLabel}${note}`;
        updateOrbitalList(data.available_orbitals, data.selected_orbital, data.selected_orbital_b);
        if (data.mode === "superposition") {
          if (data.n2 !== null && data.n2 !== undefined) {
            n2Input.value = data.n2;
          }
          if (data.l2 !== null && data.l2 !== undefined) {
            l2Input.value = data.l2;
          }
          if (data.m2 !== null && data.m2 !== undefined) {
            m2Input.value = data.m2;
          }
          if (data.mix !== null && data.mix !== undefined) {
            mixInput.value = data.mix;
            updateMixUI();
          }
        }
        if (data.mode === "superposition" && data.psi1 && data.psi2) {
          const psi1 = packPsi(data.psi1);
          const psi2 = packPsi(data.psi2);
          if (psi1 && psi2 && psi1.length === psi2.length) {
            superPsi = {
              psi1,
              psi2,
              deltaE: Number(data.delta_e ?? 0),
              baseMax: null,
            };
            superProb = new Float32Array(data.samples.length);
          } else {
            superPsi = null;
            superProb = null;
          }
        } else {
          superPsi = null;
          superProb = null;
        }
        if (data.signs && Array.isArray(data.signs)) {
          lastSigns = new Int8Array(data.signs);
        } else {
          lastSigns = null;
        }
        lastExtent = Math.max((data.max_radius || 1) * 0.1, 1e-4);

        const positions = new Float32Array(data.samples.length * 3);
        const colors = new Float32Array(data.samples.length * 3);
        for (let i = 0; i < data.samples.length; i++) {
          const p = data.samples[i];
          positions[i * 3 + 0] = p[0] * 0.1;
          positions[i * 3 + 1] = p[1] * 0.1;
          positions[i * 3 + 2] = p[2] * 0.1;
          const dist = Math.sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]) * 0.1;
          const c = colorForDistance(dist, data.max_radius * 0.1);
          colors[i * 3 + 0] = c.r;
          colors[i * 3 + 1] = c.g;
          colors[i * 3 + 2] = c.b;
        }

        const reuse = points && posAttr && positions.length === posAttr.array.length;
        if (reuse) {
          if (wantMorph) {
            animFrom = posAttr.array.slice(0);
            animTo = positions;
            animStart = performance.now();
            animDurationMs = Math.max(250, 700 / Math.max(animSpeed, 0.1));
          } else {
            posAttr.array.set(positions);
            posAttr.needsUpdate = true;
            animFrom = null;
            animTo = null;
          }
          if (colorAttr && colorAttr.array.length === colors.length) {
            colorAttr.array.set(colors);
            colorAttr.needsUpdate = true;
            baseColors = new Float32Array(colors);
          }
        } else {
          if (points) {
            group.remove(points);
            points.geometry.dispose();
            points.material.dispose();
          }

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
        posAttr = geometry.getAttribute("position");
        colorAttr = geometry.getAttribute("color");
        baseColors = new Float32Array(colors);

          const material = new THREE.PointsMaterial({
            size: 0.002,
            vertexColors: true,
            transparent: true,
            opacity: 0.6,
            depthWrite: false,
            blending: THREE.AdditiveBlending,
            map: circleTexture,
            alphaTest: 0.4,
          });

        points = new THREE.Points(geometry, material);
        group.add(points);
        animFrom = null;
        animTo = null;
        }

        updateRenderMode();
        if (renderMode === "bubbles") {
          updateBubblesFromPositions(posAttr.array, lastSigns);
        }
        if (modeLabel === "superposition" && animateEnabled && superPsi) {
          updateSuperpositionColors();
        }
        } finally {
          if (wantMorph) {
            superFetchInFlight = false;
          }
        }
      }

      document.getElementById("go").addEventListener("click", () => {
        superpositionTime = 0.0;
        superFetchInFlight = false;
        superPsi = null;
        animFrom = null;
        animTo = null;
        lastSampleTime = 0;
        fetchSamples().catch((err) => { statusEl.textContent = err.toString(); });
      });

      zInput.addEventListener("change", () => {
        const z = Number(zInput.value);
        setActiveElementByZ(z);
        resetCamera();
        fetchSamples().catch((err) => { statusEl.textContent = err.toString(); });
      });

      window.addEventListener("resize", () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
      });

      let lastTime = performance.now();
      function animate() {
        requestAnimationFrame(animate);
        const now = performance.now();
        const dt = Math.min((now - lastTime) / 1000, 0.05);
        lastTime = now;

        if (modeSelect.value === "superposition" && animateEnabled) {
          superpositionTime += dt * animSpeed;
          if (animTo && animFrom && posAttr) {
            const t = Math.min((now - animStart) / Math.max(animDurationMs, 1), 1);
            const k = t * t * (3 - 2 * t);
            const arr = posAttr.array;
            for (let i = 0; i < arr.length; i++) {
              arr[i] = animFrom[i] + (animTo[i] - animFrom[i]) * k;
            }
            posAttr.needsUpdate = true;
            if (renderMode === "bubbles") {
              bubbleDirty = true;
            }
            if (t >= 1) {
              animFrom = null;
              animTo = null;
            }
          }
          if (superPsi) {
            updateSuperpositionColors();
          }
          if (!superFetchInFlight && (now - lastSampleTime) > animDurationMs * 0.9) {
            lastSampleTime = now;
            superFetchInFlight = true;
            fetchSamples()
              .catch((err) => { statusEl.textContent = err.toString(); })
              .finally(() => { superFetchInFlight = false; });
          }
        }
        if (renderMode === "bubbles" && posAttr && bubbleDirty && (now - lastBubbleUpdate) > bubbleUpdateInterval) {
          lastBubbleUpdate = now;
          updateBubblesFromPositions(posAttr.array, lastSigns);
        }
        group.scale.setScalar(1.0);

        if (keys.size > 0) {
          camera.getWorldDirection(tmpForward);
          tmpForward.y = 0;
          if (tmpForward.lengthSq() > 1e-6) {
            tmpForward.normalize();
            tmpRight.crossVectors(tmpForward, up).normalize();

            let moveX = 0;
            let moveZ = 0;
            if (keys.has("KeyW")) moveZ += 1;
            if (keys.has("KeyS")) moveZ -= 1;
            if (keys.has("KeyA")) moveX -= 1;
            if (keys.has("KeyD")) moveX += 1;

            if (moveX !== 0 || moveZ !== 0) {
              tmpMove.set(0, 0, 0);
              if (moveZ !== 0) {
                tmpMove.addScaledVector(tmpForward, moveZ);
              }
              if (moveX !== 0) {
                tmpMove.addScaledVector(tmpRight, moveX);
              }
              if (tmpMove.lengthSq() > 0) {
                const speed = 2.5;
                tmpMove.normalize().multiplyScalar(speed * dt);
                target.add(tmpMove);
                target.x = THREE.MathUtils.clamp(target.x, -maxMove, maxMove);
                target.z = THREE.MathUtils.clamp(target.z, -maxMove, maxMove);
                updateCamera();
              }
            }
          }
        }
        renderer.render(scene, camera);
      }

      fetchSamples().then(animate);
    </script>
  </body>
</html>
"##;

const THREE_JS: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/public/three.module.js"));
const MARCHING_CUBES_JS: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/public/MarchingCubes.js"));

const INFO_HTML: &str = r##"<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Quantum Orbitals 3D - Info</title>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&display=swap" rel="stylesheet" />
    <style>
      html, body { margin: 0; padding: 0; height: 100%; background: #0b0c10; color: #e6e6e6; font-family: "Space Grotesk", "Segoe UI", sans-serif; }
      .page { max-width: 980px; margin: 0 auto; padding: 24px; }
      .topbar { display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; }
      .title { font-size: 24px; font-weight: 600; }
      .back { background: #11151b; border: 1px solid #2a2f36; color: #e6e6e6; border-radius: 8px; padding: 6px 10px; font-size: 12px; text-decoration: none; }
      .back:hover { border-color: #3c6a9e; }
      h2 { margin-top: 24px; font-size: 16px; letter-spacing: 0.02em; }
      p, li { color: #c7cdd6; line-height: 1.6; }
      code, pre { background: #10151c; padding: 10px 12px; border-radius: 6px; display: block; overflow-x: auto; }
      .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
      .card { background: #0f1218; border: 1px solid #1f2630; border-radius: 10px; padding: 12px; }
      @media (max-width: 860px) { .grid { grid-template-columns: 1fr; } }
    </style>
  </head>
  <body>
    <div class="page">
      <div class="topbar">
        <div class="title">Quantum Orbitals 3D - Info</div>
        <a class="back" href="/">Back to app</a>
      </div>

      <div class="card">
        <p>This page is a full physics and usage reference for the visualizer. It explains the underlying models, how the UI maps to physical quantities, and how the animation is generated.</p>
      </div>

      <h2>How To Use</h2>
      <div class="grid">
        <div class="card">
          <p>Choose an element from the dropdown, or edit Z directly, then click Generate.</p>
          <p>Select a mode:</p>
          <ul>
            <li>Total density: all occupied orbitals, spherical average</li>
            <li>Valence density: outermost occupied orbitals</li>
            <li>Single orbital: one (n, l, m) state</li>
            <li>Superposition: two orbitals with time evolution</li>
          </ul>
        </div>
        <div class="card">
          <p>Controls:</p>
          <ul>
            <li>Drag to orbit</li>
            <li>Scroll to zoom</li>
            <li>WASD to move the camera target</li>
            <li>Reset camera returns to default view</li>
          </ul>
        </div>
      </div>

      <h2>UI Terms</h2>
      <div class="card">
        <ul>
          <li>n: principal quantum number</li>
          <li>l: azimuthal quantum number (s=0, p=1, d=2, f=3, ...)</li>
          <li>m: magnetic quantum number (orientation)</li>
          <li>cnt: number of sample points drawn</li>
          <li>max: maximum radial extent used for sampling</li>
          <li>mix: superposition weight between orbital A and B</li>
        </ul>
      </div>

      <h2>What The Dots Represent</h2>
      <div class="card">
        <p>Each dot is a Monte Carlo sample from the probability density |psi|^2. Dots are not particle trajectories. Dense regions indicate higher probability of finding an electron.</p>
      </div>

      <h2>What The Colors Mean</h2>
      <div class="card">
        <p>Color encodes radial distance from the nucleus. The gradient runs from blue near the core through cyan and green to yellow at larger radii.</p>
      </div>

      <h2>Dots vs Bubbles</h2>
      <div class="card">
        <p>Dots mode renders the raw Monte Carlo samples. Bubbles mode builds a smooth isosurface from those samples and renders it as a closed surface.</p>
        <p>When a phase sign is defined (single orbital and superposition), positive regions are shown in red and negative regions in blue. Density-only modes do not have a sign, so only one surface is shown.</p>
      </div>

      <h2>Hydrogenic Physics Model</h2>
      <div class="card">
        <p>The hydrogenic model solves the time-independent Schrodinger equation in atomic units:</p>
        <pre>[-1/2 * nabla^2 - Z/r] psi(r) = E psi(r)</pre>
        <p>Separation of variables in spherical coordinates gives:</p>
        <pre>psi(n,l,m)(r,theta,phi) = R_nl(r) * Y_lm(theta,phi)</pre>
        <p>Quantum numbers:</p>
        <ul>
          <li>n = 1, 2, 3, ... (shell index)</li>
          <li>l = 0 to n-1 (orbital shape)</li>
          <li>m = -l to +l (orientation)</li>
        </ul>
        <p>Energy depends only on n:</p>
        <pre>E_n = -Z^2 / (2 n^2)</pre>
      </div>

      <h2>Radial And Angular Structure</h2>
      <div class="card">
        <p>The angular part Y_lm is a spherical harmonic that sets the lobe pattern. Nodes are surfaces where psi = 0. The radial part R_nl controls shell structure and radial nodes.</p>
        <p>The radial probability density is:</p>
        <pre>P(r) = r^2 |R_nl(r)|^2</pre>
        <p>This is why most electron density does not sit at r = 0 even for s orbitals.</p>
      </div>

      <h2>Superposition And Time Evolution</h2>
      <div class="card">
        <p>A single stationary eigenstate has a global phase exp(-i E t). The probability density |psi|^2 is time independent, so a single orbital does not animate in a physically meaningful way.</p>
        <p>Time dependence appears when combining at least two eigenstates with different energies:</p>
        <pre>psi(r,t) = a * psi1(r) + b * psi2(r) * exp(-i * DeltaE * t)</pre>
        <p>The density includes an interference term:</p>
        <pre>|psi|^2 = |a psi1|^2 + |b psi2|^2 + 2 Re[a b* psi1 psi2* exp(-i DeltaE t)]</pre>
        <p>If DeltaE = 0 (degenerate states with the same n in the hydrogenic model), the density is static. The app loops the phase in that case for visual continuity.</p>
      </div>

      <h2>How The Animation Is Rendered</h2>
      <div class="card">
        <p>Superposition uses repeated sampling to build successive point clouds. The UI morphs point positions between clouds to create smooth motion. This shows evolving density, not electron paths.</p>
        <p>In bubbles mode, the isosurface is rebuilt from the current samples and updated at a fixed cadence for a smoother appearance.</p>
        <p>More points (cnt) reduce Monte Carlo noise but take longer to sample.</p>
      </div>

      <h2>Multi-Electron Densities (LDA)</h2>
      <div class="card">
        <p>When available, OpenMX LDA radial wavefunctions and occupancies are used. Two density views are built:</p>
        <ul>
          <li>Total density: sum of all occupied orbitals (spherical average)</li>
          <li>Valence density: sum of outermost occupied orbitals</li>
        </ul>
        <p>Single-orbital view uses the selected radial function and Y_lm. In valence lobe mode, m is set to 0 because LDA data is not m-resolved.</p>
      </div>

      <h2>Fallbacks And Approximations</h2>
      <div class="card">
        <ul>
          <li>PSLibrary data is used for single-orbital fallback when LDA is missing.</li>
          <li>Superposition uses hydrogenic orbitals for any element and scales coordinates by 1/Z.</li>
          <li>This is not a full time-dependent many-body solver.</li>
        </ul>
      </div>

      <h2>Sampling Method</h2>
      <div class="card">
        <p>Sampling uses rejection sampling in spherical coordinates. Radial samples are drawn from a CDF built from |R_nl|^2 and r^2. Angular samples are accepted according to |Y_lm|^2.</p>
      </div>
    </div>
  </body>
</html>
"##;

async fn index() -> impl IntoResponse {
    Html(INDEX_HTML)
}

async fn info() -> impl IntoResponse {
    Html(INFO_HTML)
}

async fn three_module() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "application/javascript")], THREE_JS)
}

async fn marching_cubes() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "application/javascript")], MARCHING_CUBES_JS)
}

async fn samples(Query(q): Query<SampleQuery>) -> impl IntoResponse {
    let n = q.n.unwrap_or(2).max(1);
    let l = q.l.unwrap_or(1);
    let m = q.m.unwrap_or(0);
    let z = q.z.unwrap_or(1).clamp(1, 118);
    let count = q.count.unwrap_or(50_000).clamp(1_000, 500_000);
    let max_radius = q.max.unwrap_or(20.0).max(1.0);
    let requested_mode = ViewMode::from_query(q.mode.as_deref());
    let valence_style = ValenceStyle::from_query(q.valence_style.as_deref());
    let want_super_psi =
        q.animated.unwrap_or(false) && requested_mode == ViewMode::Superposition;
    let bubble = q.bubble.unwrap_or(false);
    let n2 = q.n2.unwrap_or(n);
    let l2 = q.l2.unwrap_or(l);
    let m2 = q.m2.unwrap_or(0);
    let mix = q.mix.unwrap_or(0.5).clamp(0.05, 0.95);
    let time = q.t.unwrap_or(0.0);

    let mut note: Option<String> = None;
    if let Some(symbol) = symbol_for_z(z) {
        let use_lda =
            !(z == 1 && (requested_mode == ViewMode::Orbital || requested_mode == ViewMode::Superposition));
        if use_lda {
            if let Ok(data) = load_lda_element(symbol).await {
                let available = lda_available_orbitals(&data);
                let max_r = data.r_max.min(max_radius);

                match requested_mode {
                    ViewMode::Total => {
                        let occupied = occupied_orbitals(&data);
                        if occupied.is_empty() {
                            note = Some("no occupied orbitals in LDA dataset".to_string());
                        } else {
                            let owned: Vec<OwnedWeightedOrbital> = occupied
                                .iter()
                                .map(|(orb, occ)| OwnedWeightedOrbital {
                                    radial_r: orb.radial_r.clone(),
                                    radial_val: orb.radial_rfn.clone(),
                                    weight: *occ,
                                })
                                .collect();
                            let samples = tokio::task::spawn_blocking(move || {
                                let weighted: Vec<WeightedOrbital> = owned
                                    .iter()
                                    .map(|orb| WeightedOrbital {
                                        radial_r: &orb.radial_r,
                                        radial_val: &orb.radial_val,
                                        weight: orb.weight,
                                    })
                                    .collect();
                                generate_isotropic_density_samples(
                                    &weighted,
                                    count,
                                    max_r,
                                    RadialKind::R,
                                )
                            })
                            .await
                            .unwrap_or_default();
                            let sign_count = samples.len();
                            let mode_note = format!(
                                "OpenMX LDA spherical total density ({:.0}e)",
                                data.total_electrons
                            );
                            let out = SampleResponse {
                                n,
                                l,
                                m,
                                n2: None,
                                l2: None,
                                m2: None,
                                z,
                                count,
                                max_radius: max_r,
                                samples,
                                mode: ViewMode::Total.as_str().to_string(),
                                source: "openmx_lda".to_string(),
                                note: Some(mode_note),
                                available_orbitals: available,
                                selected_orbital: None,
                                selected_orbital_b: None,
                                mix: None,
                                time: None,
                                psi1: None,
                                psi2: None,
                                delta_e: None,
                                signs: if bubble { Some(vec![1; sign_count]) } else { None },
                            };
                            return Json(out).into_response();
                        }
                    }
                    ViewMode::Valence => {
                        let (valence_orbitals, valence_note) = valence_orbitals(&data);
                        let selection = if valence_orbitals.is_empty() {
                            note = Some(
                                valence_note.unwrap_or_else(|| {
                                    "valence set unavailable; using total density".to_string()
                                }),
                            );
                            occupied_orbitals(&data)
                        } else {
                            valence_orbitals
                        };

                        if selection.is_empty() {
                            note = Some("no occupied orbitals in LDA dataset".to_string());
                        } else {
                            let (samples, mode_note) = if valence_style == ValenceStyle::Orbitals {
                                let owned: Vec<OwnedAngularOrbital> = selection
                                    .iter()
                                    .map(|(orb, occ)| OwnedAngularOrbital {
                                        radial_r: orb.radial_r.clone(),
                                        radial_val: orb.radial_rfn.clone(),
                                        weight: *occ,
                                        l: orb.l,
                                        m: 0,
                                    })
                                    .collect();
                                let samples = tokio::task::spawn_blocking(move || {
                                    generate_weighted_orbital_samples(
                                        &owned,
                                        count,
                                        max_r,
                                        RadialKind::R,
                                    )
                                })
                                .await
                                .unwrap_or_default();
                                let mode_note = note.take().unwrap_or_else(|| {
                                    "OpenMX LDA valence orbitals (m=0 projection)".to_string()
                                });
                                (samples, mode_note)
                            } else {
                                let owned: Vec<OwnedWeightedOrbital> = selection
                                    .iter()
                                    .map(|(orb, occ)| OwnedWeightedOrbital {
                                        radial_r: orb.radial_r.clone(),
                                        radial_val: orb.radial_rfn.clone(),
                                        weight: *occ,
                                    })
                                    .collect();
                                let samples = tokio::task::spawn_blocking(move || {
                                    let weighted: Vec<WeightedOrbital> = owned
                                        .iter()
                                        .map(|orb| WeightedOrbital {
                                            radial_r: &orb.radial_r,
                                            radial_val: &orb.radial_val,
                                            weight: orb.weight,
                                        })
                                        .collect();
                                    generate_isotropic_density_samples(
                                        &weighted,
                                        count,
                                        max_r,
                                        RadialKind::R,
                                    )
                                })
                                .await
                                .unwrap_or_default();
                                let mode_note = note.take().unwrap_or_else(|| {
                                    format!(
                                        "OpenMX LDA spherical valence density ({:.0}e)",
                                        data.valence_electrons
                                    )
                                });
                                (samples, mode_note)
                            };
                            let sign_count = samples.len();
                            let out = SampleResponse {
                                n,
                                l,
                                m,
                                n2: None,
                                l2: None,
                                m2: None,
                                z,
                                count,
                                max_radius: max_r,
                                samples,
                                mode: ViewMode::Valence.as_str().to_string(),
                                source: "openmx_lda".to_string(),
                                note: Some(mode_note),
                                available_orbitals: available,
                                selected_orbital: None,
                                selected_orbital_b: None,
                                mix: None,
                                time: None,
                                psi1: None,
                                psi2: None,
                                delta_e: None,
                                signs: if bubble { Some(vec![1; sign_count]) } else { None },
                            };
                            return Json(out).into_response();
                        }
                    }
                    ViewMode::Orbital => {
                        if let Some((orbital, exact)) = select_lda_orbital(&data, n, l) {
                            let m_used = m.clamp(-(orbital.l as i32), orbital.l as i32);
                            let radial_r = orbital.radial_r.clone();
                            let radial_val = orbital.radial_rfn.clone();
                            let radial_r_sign = radial_r.clone();
                            let radial_val_sign = radial_val.clone();
                            let l_used = orbital.l;
                            let samples = tokio::task::spawn_blocking(move || {
                                generate_orbital_samples_from_radial(
                                    &radial_r,
                                    &radial_val,
                                    l_used,
                                    m_used,
                                    count,
                                    max_r,
                                    RadialKind::R,
                                )
                            })
                            .await
                            .unwrap_or_default();
                            let signs = if bubble {
                                Some(signs_from_radial_samples(
                                    &samples,
                                    &radial_r_sign,
                                    &radial_val_sign,
                                    l_used,
                                    m_used,
                                    RadialKind::R,
                                ))
                            } else {
                                None
                            };
                            let used_label = orbital.label.clone();
                            let mode_note = if exact {
                                format!("OpenMX LDA {}", used_label)
                            } else {
                                format!("requested n/l not in dataset; using {}", used_label)
                            };
                            let out = SampleResponse {
                                n: orbital.n,
                                l: orbital.l,
                                m: m_used,
                                n2: None,
                                l2: None,
                                m2: None,
                                z,
                                count,
                                max_radius: max_r,
                                samples,
                                mode: ViewMode::Orbital.as_str().to_string(),
                                source: "openmx_lda".to_string(),
                                note: Some(mode_note),
                                available_orbitals: available,
                                selected_orbital: Some(used_label),
                                selected_orbital_b: None,
                                mix: None,
                                time: None,
                                psi1: None,
                                psi2: None,
                                delta_e: None,
                                signs,
                            };
                            return Json(out).into_response();
                        }
                        note = Some("orbital not available in LDA dataset".to_string());
                    }
                    ViewMode::Superposition => {
                        if let Some((orb_a, exact_a, orb_b, exact_b)) =
                            select_lda_orbital_pair(&data, n, l, n2, l2)
                        {
                            let m_a = m.clamp(-(orb_a.l as i32), orb_a.l as i32);
                            let m_b = m2.clamp(-(orb_b.l as i32), orb_b.l as i32);
                            let e1 = data.eigenvalues.get(&(orb_a.n, orb_a.l)).copied();
                            let e2 = data.eigenvalues.get(&(orb_b.n, orb_b.l)).copied();
                            let delta_e = match (e1, e2) {
                                (Some(a), Some(b)) => b - a,
                                _ => 0.0,
                            };
                            let orb_a_cl = orb_a.clone();
                            let orb_b_cl = orb_b.clone();
                            let (samples, psi1, psi2) = tokio::task::spawn_blocking(move || {
                                generate_superposition_samples_lda(
                                    &orb_a_cl,
                                    &orb_b_cl,
                                    m_a,
                                    m_b,
                                    mix,
                                    time,
                                    count,
                                    max_r,
                                    delta_e,
                                    want_super_psi,
                                )
                            })
                            .await
                            .unwrap_or_default();
                            let signs = if bubble {
                                Some(signs_from_superposition_lda(
                                    &samples,
                                    &orb_a,
                                    &orb_b,
                                    m_a,
                                    m_b,
                                    mix,
                                    time,
                                    delta_e,
                                ))
                            } else {
                                None
                            };
                            let mut mode_note = String::from("OpenMX LDA superposition");
                            if !exact_a || !exact_b {
                                mode_note.push_str(" (closest orbitals used)");
                            }
                            if e1.is_none() || e2.is_none() {
                                mode_note.push_str(" | missing eigenvalues, static phase");
                            }
                            if delta_e.abs() < 1e-6 {
                                mode_note.push_str(" | degenerate energies, static density");
                            }
                            let out = SampleResponse {
                                n: orb_a.n,
                                l: orb_a.l,
                                m: m_a,
                                n2: Some(orb_b.n),
                                l2: Some(orb_b.l),
                                m2: Some(m_b),
                                z,
                                count,
                                max_radius: max_r,
                                samples,
                                mode: ViewMode::Superposition.as_str().to_string(),
                                source: "openmx_lda".to_string(),
                                note: Some(mode_note),
                                available_orbitals: available,
                                selected_orbital: Some(orb_a.label.clone()),
                                selected_orbital_b: Some(orb_b.label.clone()),
                                mix: Some(mix),
                                time: Some(time),
                                psi1: if want_super_psi { Some(psi1) } else { None },
                                psi2: if want_super_psi { Some(psi2) } else { None },
                                delta_e: Some(delta_e),
                                signs,
                            };
                            return Json(out).into_response();
                        }
                        note = Some("superposition orbitals not available".to_string());
                    }
                }
            } else {
                note = Some("OpenMX LDA unavailable; trying fallback".to_string());
            }
        }
    }

    if requested_mode == ViewMode::Orbital && z != 1 {
        if let Some(symbol) = symbol_for_z(z) {
            if let Ok(data) = load_element_data(symbol, z).await {
                let available = data
                    .orbitals
                    .iter()
                    .map(|o| OrbitalInfo {
                        label: o.label.clone(),
                        n: o.n,
                        l: o.l,
                    })
                    .collect::<Vec<_>>();

                if let Some((orbital, exact)) = select_pslib_orbital(&data, n, l) {
                    let max_r = data.r_max.min(max_radius);
                    let m_used = m.clamp(-(orbital.l as i32), orbital.l as i32);
                    let radial_r = orbital.radial_r.clone();
                    let radial_val = orbital.radial_chi.clone();
                    let radial_r_sign = radial_r.clone();
                    let radial_val_sign = radial_val.clone();
                    let l_used = orbital.l;
                    let samples = tokio::task::spawn_blocking(move || {
                        generate_orbital_samples_from_radial(
                            &radial_r,
                            &radial_val,
                            l_used,
                            m_used,
                            count,
                            max_r,
                            RadialKind::Chi,
                        )
                    })
                    .await
                    .unwrap_or_default();
                    let signs = if bubble {
                        Some(signs_from_radial_samples(
                            &samples,
                            &radial_r_sign,
                            &radial_val_sign,
                            l_used,
                            m_used,
                            RadialKind::Chi,
                        ))
                    } else {
                        None
                    };
                    let used_label = orbital.label.clone();
                    let mode_note = if exact {
                        format!("PSlibrary {}", used_label)
                    } else {
                        format!("requested n/l not in dataset; using {}", used_label)
                    };
                    let out = SampleResponse {
                        n: orbital.n,
                        l: orbital.l,
                        m: m_used,
                        n2: None,
                        l2: None,
                        m2: None,
                        z,
                        count,
                        max_radius: max_r,
                        samples,
                        mode: ViewMode::Orbital.as_str().to_string(),
                        source: "pslibrary".to_string(),
                        note: Some(mode_note),
                        available_orbitals: available,
                        selected_orbital: Some(used_label),
                        selected_orbital_b: None,
                        mix: None,
                        time: None,
                        psi1: None,
                        psi2: None,
                        delta_e: None,
                        signs,
                    };
                    return Json(out).into_response();
                }

                note = Some("orbital not available in dataset".to_string());
                let out = SampleResponse {
                    n,
                    l,
                    m,
                    n2: None,
                    l2: None,
                    m2: None,
                    z,
                    count: 0,
                    max_radius,
                    samples: Vec::new(),
                    mode: ViewMode::Orbital.as_str().to_string(),
                    source: "pslibrary".to_string(),
                    note,
                    available_orbitals: available,
                    selected_orbital: None,
                    selected_orbital_b: None,
                    mix: None,
                    time: None,
                    psi1: None,
                    psi2: None,
                    delta_e: None,
                    signs: None,
                };
                return Json(out).into_response();
            } else {
                note = Some("dataset unavailable; using hydrogenic".to_string());
            }
        }
    }

    if requested_mode == ViewMode::Superposition {
        let qn_a = QuantumNumbers::new(n, l, m);
        let qn_b = QuantumNumbers::new(n2, l2, m2);
        if let (Some(q1), Some(q2)) = (qn_a, qn_b) {
            let e1 = hydrogenic_energy(q1.n);
            let e2 = hydrogenic_energy(q2.n);
            let delta_e = e2 - e1;
            let (samples, psi1, psi2) = tokio::task::spawn_blocking(move || {
                generate_superposition_samples_hydrogenic(
                    q1,
                    q2,
                    mix,
                    time,
                    count,
                    max_radius,
                    delta_e,
                    want_super_psi,
                )
            })
            .await
            .unwrap_or_default();
            let signs = if bubble {
                Some(signs_from_superposition_hydrogenic(
                    &samples,
                    q1,
                    q2,
                    mix,
                    time,
                    delta_e,
                ))
            } else {
                None
            };
            let inv_z = 1.0 / z as f32;
            let scaled_max = if z > 1 { max_radius * inv_z } else { max_radius };
            let scaled_samples = if z > 1 {
                samples
                    .into_iter()
                    .map(|p| [p[0] * inv_z, p[1] * inv_z, p[2] * inv_z])
                    .collect::<Vec<_>>()
            } else {
                samples
            };
            let mut note_text = "Hydrogenic superposition (time-dependent)".to_string();
            if delta_e.abs() < 1e-6 {
                note_text.push_str(" | same n -> no time evolution");
            }
            if z > 1 {
                note_text.push_str(" | hydrogenic approximation scaled by Z");
            }
            let out = SampleResponse {
                n: q1.n,
                l: q1.l,
                m: q1.m_l,
                n2: Some(q2.n),
                l2: Some(q2.l),
                m2: Some(q2.m_l),
                z,
                count,
                max_radius: scaled_max,
                samples: scaled_samples,
                mode: ViewMode::Superposition.as_str().to_string(),
                source: "hydrogenic".to_string(),
                note: Some(note_text),
                available_orbitals: Vec::new(),
                selected_orbital: None,
                selected_orbital_b: None,
                mix: Some(mix),
                time: Some(time),
                psi1: if want_super_psi { Some(psi1) } else { None },
                psi2: if want_super_psi { Some(psi2) } else { None },
                delta_e: Some(delta_e),
                signs,
            };
            return Json(out).into_response();
        } else {
            note = Some("invalid quantum numbers for superposition".to_string());
        }
    }

    if requested_mode != ViewMode::Orbital {
        note = Some("density dataset unavailable; using single orbital".to_string());
    } else if z == 1 {
        note = Some("hydrogenic (exact)".to_string());
    }

    let qn = match QuantumNumbers::new(n, l, m) {
        Some(qn) => qn,
        None => {
            let empty = SampleResponse {
                n,
                l,
                m,
                n2: None,
                l2: None,
                m2: None,
                z,
                count: 0,
                max_radius,
                samples: Vec::new(),
                mode: ViewMode::Orbital.as_str().to_string(),
                source: "hydrogenic".to_string(),
                note,
                available_orbitals: Vec::new(),
                    selected_orbital: None,
                    selected_orbital_b: None,
                    mix: None,
                    time: None,
                    psi1: None,
                    psi2: None,
                    delta_e: None,
                    signs: None,
                };
            return Json(empty).into_response();
        }
    };

    let raw = tokio::task::spawn_blocking(move || generate_orbital_samples(qn, count, max_radius))
        .await
        .unwrap_or_default();
    let signs = if bubble {
        Some(signs_from_hydrogenic_samples(
            &raw.iter().map(|(x, y, z)| [*x, *y, *z]).collect::<Vec<_>>(),
            qn,
        ))
    } else {
        None
    };
    let inv_z = 1.0 / z as f32;
    let samples = raw
        .into_iter()
        .map(|(x, y, z_pos)| [x * inv_z, y * inv_z, z_pos * inv_z])
        .collect();

    let out = SampleResponse {
        n: qn.n,
        l: qn.l,
        m: qn.m_l,
        n2: None,
        l2: None,
        m2: None,
        z,
        count,
        max_radius,
        samples,
        mode: ViewMode::Orbital.as_str().to_string(),
        source: "hydrogenic".to_string(),
        note,
        available_orbitals: Vec::new(),
        selected_orbital: None,
        selected_orbital_b: None,
        mix: None,
        time: None,
        psi1: None,
        psi2: None,
        delta_e: None,
        signs,
    };
    Json(out).into_response()
}

fn lda_available_orbitals(data: &LdaElement) -> Vec<OrbitalInfo> {
    let mut list = Vec::new();
    for orb in &data.orbitals {
        let occ = data.occupancy.get(&(orb.n, orb.l)).copied().unwrap_or(0.0);
        if occ > 0.0 {
            list.push(OrbitalInfo {
                label: orb.label.clone(),
                n: orb.n,
                l: orb.l,
            });
        }
    }
    list
}

fn occupied_orbitals(data: &LdaElement) -> Vec<(&LdaOrbital, f32)> {
    let mut list = Vec::new();
    for orb in &data.orbitals {
        if let Some(&occ) = data.occupancy.get(&(orb.n, orb.l)) {
            if occ > 0.0 {
                list.push((orb, occ));
            }
        }
    }
    list
}

fn valence_orbitals(data: &LdaElement) -> (Vec<(&LdaOrbital, f32)>, Option<String>) {
    let mut occupied: Vec<(&LdaOrbital, f32, f32)> = Vec::new();
    for orb in &data.orbitals {
        if let Some(&occ) = data.occupancy.get(&(orb.n, orb.l)) {
            if occ > 0.0 {
                let energy = data
                    .eigenvalues
                    .get(&(orb.n, orb.l))
                    .copied()
                    .unwrap_or(f32::NEG_INFINITY);
                occupied.push((orb, occ, energy));
            }
        }
    }

    if occupied.is_empty() {
        return (Vec::new(), Some("no occupied orbitals in dataset".to_string()));
    }

    let use_energy = occupied.iter().any(|o| o.2.is_finite());
    if use_energy {
        occupied.sort_by(|a, b| {
            b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal)
        });
    } else {
        occupied.sort_by(|a, b| (b.0.n, b.0.l).cmp(&(a.0.n, a.0.l)));
    }

    let mut remaining = data.valence_electrons;
    if remaining <= 0.0 {
        return (Vec::new(), Some("valence electron count missing".to_string()));
    }

    let mut out = Vec::new();
    for (orb, occ, _) in occupied {
        if remaining <= 0.0 {
            break;
        }
        out.push((orb, occ));
        remaining -= occ;
    }

    (out, None)
}

fn select_lda_orbital(data: &LdaElement, n: u32, l: u32) -> Option<(LdaOrbital, bool)> {
    let mut same_l = None;
    for orb in &data.orbitals {
        if orb.l == l && orb.n == n {
            return Some((orb.clone(), true));
        }
        if orb.l == l && same_l.is_none() {
            same_l = Some(orb.clone());
        }
    }
    if let Some(orb) = same_l {
        return Some((orb, false));
    }
    data.orbitals.first().cloned().map(|orb| (orb, false))
}

fn select_pslib_orbital(data: &ElementData, n: u32, l: u32) -> Option<(Orbital, bool)> {
    let mut same_l = None;
    for orb in &data.orbitals {
        if orb.l == l && orb.n == n {
            return Some((orb.clone(), true));
        }
        if orb.l == l && same_l.is_none() {
            same_l = Some(orb.clone());
        }
    }
    if let Some(orb) = same_l {
        return Some((orb, false));
    }
    data.orbitals.first().cloned().map(|orb| (orb, false))
}

fn select_lda_orbital_pair(
    data: &LdaElement,
    n1: u32,
    l1: u32,
    n2: u32,
    l2: u32,
) -> Option<(LdaOrbital, bool, LdaOrbital, bool)> {
    let (orb_a, exact_a) = select_lda_orbital(data, n1, l1)?;
    if let Some((orb_b, exact_b)) = select_lda_orbital(data, n2, l2) {
        if orb_b.n != orb_a.n || orb_b.l != orb_a.l {
            return Some((orb_a, exact_a, orb_b, exact_b));
        }
    }

    for orb in &data.orbitals {
        if orb.n != orb_a.n || orb.l != orb_a.l {
            return Some((orb_a, exact_a, orb.clone(), false));
        }
    }
    None
}

struct WeightedOrbital<'a> {
    radial_r: &'a [f32],
    radial_val: &'a [f32],
    weight: f32,
}

struct OwnedWeightedOrbital {
    radial_r: Vec<f32>,
    radial_val: Vec<f32>,
    weight: f32,
}

struct OwnedAngularOrbital {
    radial_r: Vec<f32>,
    radial_val: Vec<f32>,
    weight: f32,
    l: u32,
    m: i32,
}

fn generate_orbital_samples_from_radial(
    radial_r: &[f32],
    radial_val: &[f32],
    l: u32,
    m_l: i32,
    num_samples: usize,
    max_radius: f32,
    radial_kind: RadialKind,
) -> Vec<[f32; 3]> {
    use rand::Rng;
    use std::f32::consts::PI;

    let mut samples = Vec::with_capacity(num_samples);
    let mut rng = rand::thread_rng();

    let cdf = build_radial_cdf(radial_r, radial_val, max_radius, radial_kind);
    let max_ang = max_angular_prob(l, m_l);

    while samples.len() < num_samples {
        let r = sample_r(&cdf, radial_r, &mut rng);
        let phi = rng.gen::<f32>() * 2.0 * PI;

        // Rejection sample theta from |Y_lm|^2
        loop {
            let cos_theta = rng.gen::<f32>() * 2.0 - 1.0;
            let theta = cos_theta.acos();
            let ang = angular_wavefunction(theta, phi, l, m_l);
            if rng.gen::<f32>() < (ang * ang) / max_ang {
                let x = r * theta.sin() * phi.cos();
                let y = r * theta.sin() * phi.sin();
                let z = r * theta.cos();
                samples.push([x, y, z]);
                break;
            }
        }
    }

    samples
}

fn generate_superposition_samples_lda(
    orb_a: &LdaOrbital,
    orb_b: &LdaOrbital,
    m_a: i32,
    m_b: i32,
    mix: f32,
    time: f32,
    num_samples: usize,
    max_radius: f32,
    delta_e: f32,
    with_psi: bool,
) -> (Vec<[f32; 3]>, Vec<[f32; 2]>, Vec<[f32; 2]>) {
    use rand::Rng;
    use std::f32::consts::PI;

    let mut rng = rand::thread_rng();
    let mut samples = Vec::with_capacity(num_samples);
    let mut psi1 = Vec::new();
    let mut psi2 = Vec::new();
    if with_psi {
        psi1.reserve(num_samples);
        psi2.reserve(num_samples);
    }

    let a = mix.sqrt();
    let b = (1.0 - mix).sqrt();
    let phase_re = (delta_e * time).cos();
    let phase_im = -(delta_e * time).sin();

    let cdf_a = build_radial_cdf(&orb_a.radial_r, &orb_a.radial_rfn, max_radius, RadialKind::R);
    let cdf_b = build_radial_cdf(&orb_b.radial_r, &orb_b.radial_rfn, max_radius, RadialKind::R);
    let max_ang_a = max_angular_prob(orb_a.l, m_a);
    let max_ang_b = max_angular_prob(orb_b.l, m_b);
    if cdf_a.is_empty() || cdf_b.is_empty() {
        return (samples, psi1, psi2);
    }

    let mut attempts = 0usize;
    let max_attempts = num_samples.saturating_mul(200);
    while samples.len() < num_samples && attempts < max_attempts {
        attempts += 1;
        let pick_a = rng.gen::<f32>() < mix;
        let (r, theta, phi) = if pick_a {
            let r = sample_r(&cdf_a, &orb_a.radial_r, &mut rng);
            let phi = rng.gen::<f32>() * 2.0 * PI;
            let theta = loop {
                let cos_theta = rng.gen::<f32>() * 2.0 - 1.0;
                let theta = cos_theta.acos();
                let ang = angular_wavefunction(theta, phi, orb_a.l, m_a);
                if rng.gen::<f32>() < (ang * ang) / max_ang_a {
                    break theta;
                }
            };
            (r, theta, phi)
        } else {
            let r = sample_r(&cdf_b, &orb_b.radial_r, &mut rng);
            let phi = rng.gen::<f32>() * 2.0 * PI;
            let theta = loop {
                let cos_theta = rng.gen::<f32>() * 2.0 - 1.0;
                let theta = cos_theta.acos();
                let ang = angular_wavefunction(theta, phi, orb_b.l, m_b);
                if rng.gen::<f32>() < (ang * ang) / max_ang_b {
                    break theta;
                }
            };
            (r, theta, phi)
        };

        let r1 = interp_radial(r, &orb_a.radial_r, &orb_a.radial_rfn);
        let r2 = interp_radial(r, &orb_b.radial_r, &orb_b.radial_rfn);

        let (y1_re, y1_im) = spherical_harmonic(theta, phi, orb_a.l, m_a);
        let (y2_re, y2_im) = spherical_harmonic(theta, phi, orb_b.l, m_b);

        let psi1_re = a * r1 * y1_re;
        let psi1_im = a * r1 * y1_im;
        let psi2_base_re = b * r2 * y2_re;
        let psi2_base_im = b * r2 * y2_im;
        let y2p_re = y2_re * phase_re - y2_im * phase_im;
        let y2p_im = y2_re * phase_im + y2_im * phase_re;
        let psi2_re = b * r2 * y2p_re;
        let psi2_im = b * r2 * y2p_im;

        let re = psi1_re + psi2_re;
        let im = psi1_im + psi2_im;
        let prob = re * re + im * im;

        let y1_sq = y1_re * y1_re + y1_im * y1_im;
        let y2_sq = y2_re * y2_re + y2_im * y2_im;
        let psi1_sq = r1 * r1 * y1_sq;
        let psi2_sq = r2 * r2 * y2_sq;
        let proposal = mix * psi1_sq + (1.0 - mix) * psi2_sq;
        if proposal <= 0.0 {
            continue;
        }
        let accept = if with_psi {
            1.0
        } else {
            (prob / (2.0 * proposal)).clamp(0.0, 1.0)
        };
        if with_psi || rng.gen::<f32>() < accept {
            let x = r * theta.sin() * phi.cos();
            let y = r * theta.sin() * phi.sin();
            let z = r * theta.cos();
            samples.push([x, y, z]);
            if with_psi {
                psi1.push([psi1_re, psi1_im]);
                psi2.push([psi2_base_re, psi2_base_im]);
            }
        }
    }

    (samples, psi1, psi2)
}

fn generate_superposition_samples_hydrogenic(
    qn_a: QuantumNumbers,
    qn_b: QuantumNumbers,
    mix: f32,
    time: f32,
    num_samples: usize,
    max_radius: f32,
    delta_e: f32,
    with_psi: bool,
) -> (Vec<[f32; 3]>, Vec<[f32; 2]>, Vec<[f32; 2]>) {
    use rand::Rng;
    use std::f32::consts::PI;

    let mut rng = rand::thread_rng();
    let mut samples = Vec::with_capacity(num_samples);
    let mut psi1 = Vec::new();
    let mut psi2 = Vec::new();
    if with_psi {
        psi1.reserve(num_samples);
        psi2.reserve(num_samples);
    }
    let a = mix.sqrt();
    let b = (1.0 - mix).sqrt();
    let phase_re = (delta_e * time).cos();
    let phase_im = -(delta_e * time).sin();

    let radial_steps = 800usize;
    let rs = build_radial_grid(max_radius, radial_steps);
    let rfn_a: Vec<f32> = rs
        .iter()
        .map(|r| radial_wavefunction(*r, qn_a.n, qn_a.l))
        .collect();
    let rfn_b: Vec<f32> = rs
        .iter()
        .map(|r| radial_wavefunction(*r, qn_b.n, qn_b.l))
        .collect();
    let cdf_a = build_radial_cdf(&rs, &rfn_a, max_radius, RadialKind::R);
    let cdf_b = build_radial_cdf(&rs, &rfn_b, max_radius, RadialKind::R);
    let max_ang_a = max_angular_prob(qn_a.l, qn_a.m_l);
    let max_ang_b = max_angular_prob(qn_b.l, qn_b.m_l);
    if cdf_a.is_empty() || cdf_b.is_empty() {
        return (samples, psi1, psi2);
    }

    let mut attempts = 0usize;
    let max_attempts = num_samples.saturating_mul(200);
    while samples.len() < num_samples && attempts < max_attempts {
        attempts += 1;
        let pick_a = rng.gen::<f32>() < mix;
        let (r, theta, phi) = if pick_a {
            let r = sample_r(&cdf_a, &rs, &mut rng);
            let phi = rng.gen::<f32>() * 2.0 * PI;
            let theta = loop {
                let cos_theta = rng.gen::<f32>() * 2.0 - 1.0;
                let theta = cos_theta.acos();
                let ang = angular_wavefunction(theta, phi, qn_a.l, qn_a.m_l);
                if rng.gen::<f32>() < (ang * ang) / max_ang_a {
                    break theta;
                }
            };
            (r, theta, phi)
        } else {
            let r = sample_r(&cdf_b, &rs, &mut rng);
            let phi = rng.gen::<f32>() * 2.0 * PI;
            let theta = loop {
                let cos_theta = rng.gen::<f32>() * 2.0 - 1.0;
                let theta = cos_theta.acos();
                let ang = angular_wavefunction(theta, phi, qn_b.l, qn_b.m_l);
                if rng.gen::<f32>() < (ang * ang) / max_ang_b {
                    break theta;
                }
            };
            (r, theta, phi)
        };

        let r1 = interp_radial(r, &rs, &rfn_a);
        let r2 = interp_radial(r, &rs, &rfn_b);
        let (y1_re, y1_im) = spherical_harmonic(theta, phi, qn_a.l, qn_a.m_l);
        let (y2_re, y2_im) = spherical_harmonic(theta, phi, qn_b.l, qn_b.m_l);

        let psi1_re = a * r1 * y1_re;
        let psi1_im = a * r1 * y1_im;
        let psi2_base_re = b * r2 * y2_re;
        let psi2_base_im = b * r2 * y2_im;
        let y2p_re = y2_re * phase_re - y2_im * phase_im;
        let y2p_im = y2_re * phase_im + y2_im * phase_re;
        let psi2_re = b * r2 * y2p_re;
        let psi2_im = b * r2 * y2p_im;

        let re = psi1_re + psi2_re;
        let im = psi1_im + psi2_im;
        let prob = re * re + im * im;

        let y1_sq = y1_re * y1_re + y1_im * y1_im;
        let y2_sq = y2_re * y2_re + y2_im * y2_im;
        let psi1_sq = r1 * r1 * y1_sq;
        let psi2_sq = r2 * r2 * y2_sq;
        let proposal = mix * psi1_sq + (1.0 - mix) * psi2_sq;
        if proposal <= 0.0 {
            continue;
        }
        let accept = if with_psi {
            1.0
        } else {
            (prob / (2.0 * proposal)).clamp(0.0, 1.0)
        };
        if with_psi || rng.gen::<f32>() < accept {
            let x = r * theta.sin() * phi.cos();
            let y = r * theta.sin() * phi.sin();
            let z = r * theta.cos();
            samples.push([x, y, z]);
            if with_psi {
                psi1.push([psi1_re, psi1_im]);
                psi2.push([psi2_base_re, psi2_base_im]);
            }
        }
    }

    (samples, psi1, psi2)
}

fn build_radial_grid(max_radius: f32, steps: usize) -> Vec<f32> {
    let count = steps.max(2);
    let mut rs = Vec::with_capacity(count);
    let denom = (count - 1) as f32;
    for i in 0..count {
        let t = (i as f32) / denom;
        rs.push(max_radius * t);
    }
    rs
}

fn interp_radial(r: f32, rs: &[f32], vs: &[f32]) -> f32 {
    if rs.is_empty() || vs.is_empty() {
        return 0.0;
    }
    if r <= rs[0] {
        return vs[0];
    }
    if r >= rs[rs.len() - 1] {
        return *vs.last().unwrap_or(&0.0);
    }
    let idx = match rs.binary_search_by(|v| v.partial_cmp(&r).unwrap()) {
        Ok(i) => i,
        Err(i) => i.min(rs.len() - 1),
    };
    if idx == 0 {
        return vs[0];
    }
    let r0 = rs[idx - 1];
    let r1 = rs[idx];
    let v0 = vs[idx - 1];
    let v1 = vs[idx];
    let t = if r1 > r0 { (r - r0) / (r1 - r0) } else { 0.0 };
    v0 + (v1 - v0) * t
}

fn hydrogenic_energy(n: u32) -> f32 {
    let n_f = n as f32;
    -0.5 / (n_f * n_f)
}

fn generate_isotropic_density_samples(
    orbitals: &[WeightedOrbital],
    num_samples: usize,
    max_radius: f32,
    radial_kind: RadialKind,
) -> Vec<[f32; 3]> {
    use rand::Rng;
    use std::f32::consts::PI;

    let mut rng = rand::thread_rng();
    let mut samplers = Vec::new();
    let mut weight_cdf = Vec::new();
    let mut total_weight = 0.0_f32;

    for orb in orbitals {
        if orb.weight <= 0.0 {
            continue;
        }
        let cdf = build_radial_cdf(orb.radial_r, orb.radial_val, max_radius, radial_kind);
        if cdf.is_empty() {
            continue;
        }
        total_weight += orb.weight;
        weight_cdf.push(total_weight);
        samplers.push((orb.radial_r, cdf));
    }

    if samplers.is_empty() || total_weight <= 0.0 {
        return Vec::new();
    }

    for v in &mut weight_cdf {
        *v /= total_weight;
    }

    let mut samples = Vec::with_capacity(num_samples);
    while samples.len() < num_samples {
        let u = rng.gen::<f32>();
        let idx = match weight_cdf.binary_search_by(|v| v.partial_cmp(&u).unwrap()) {
            Ok(i) => i,
            Err(i) => i.min(weight_cdf.len() - 1),
        };
        let (rs, cdf) = &samplers[idx];
        let r = sample_r(cdf, rs, &mut rng);

        let cos_theta = rng.gen::<f32>() * 2.0 - 1.0;
        let theta = cos_theta.acos();
        let phi = rng.gen::<f32>() * 2.0 * PI;

        let x = r * theta.sin() * phi.cos();
        let y = r * theta.sin() * phi.sin();
        let z = r * theta.cos();
        samples.push([x, y, z]);
    }

    samples
}

fn generate_weighted_orbital_samples(
    orbitals: &[OwnedAngularOrbital],
    num_samples: usize,
    max_radius: f32,
    radial_kind: RadialKind,
) -> Vec<[f32; 3]> {
    let total_weight: f32 = orbitals.iter().map(|orb| orb.weight).sum();
    if total_weight <= 0.0 || orbitals.is_empty() {
        return Vec::new();
    }

    let mut samples = Vec::with_capacity(num_samples);
    let mut remaining = num_samples;

    for (idx, orb) in orbitals.iter().enumerate() {
        if remaining == 0 {
            break;
        }
        let mut count =
            ((num_samples as f32) * (orb.weight / total_weight)).round() as usize;
        if idx == orbitals.len() - 1 {
            count = remaining;
        }
        if count > remaining {
            count = remaining;
        }
        remaining -= count;
        if count == 0 {
            continue;
        }
        let mut part = generate_orbital_samples_from_radial(
            &orb.radial_r,
            &orb.radial_val,
            orb.l,
            orb.m,
            count,
            max_radius,
            radial_kind,
        );
        samples.append(&mut part);
    }

    samples
}

fn sign_from_value(v: f32) -> i8 {
    if v >= 0.0 {
        1
    } else {
        -1
    }
}

fn signs_from_radial_samples(
    samples: &[[f32; 3]],
    radial_r: &[f32],
    radial_val: &[f32],
    l: u32,
    m_l: i32,
    radial_kind: RadialKind,
) -> Vec<i8> {
    let mut out = Vec::with_capacity(samples.len());
    for p in samples {
        let x = p[0];
        let y = p[1];
        let z = p[2];
        let r = (x * x + y * y + z * z).sqrt();
        if r <= 1e-8 {
            out.push(1);
            continue;
        }
        let cos_theta = (z / r).clamp(-1.0, 1.0);
        let theta = cos_theta.acos();
        let phi = y.atan2(x);
        let mut radial = interp_radial(r, radial_r, radial_val);
        if matches!(radial_kind, RadialKind::Chi) && r > 1e-8 {
            radial /= r;
        }
        let (y_re, _) = spherical_harmonic(theta, phi, l, m_l);
        let psi_re = radial * y_re;
        out.push(sign_from_value(psi_re));
    }
    out
}

fn signs_from_hydrogenic_samples(samples: &[[f32; 3]], qn: QuantumNumbers) -> Vec<i8> {
    let mut out = Vec::with_capacity(samples.len());
    for p in samples {
        let x = p[0];
        let y = p[1];
        let z = p[2];
        let r = (x * x + y * y + z * z).sqrt();
        if r <= 1e-8 {
            out.push(1);
            continue;
        }
        let cos_theta = (z / r).clamp(-1.0, 1.0);
        let theta = cos_theta.acos();
        let phi = y.atan2(x);
        let radial = radial_wavefunction(r, qn.n, qn.l);
        let (y_re, _) = spherical_harmonic(theta, phi, qn.l, qn.m_l);
        let psi_re = radial * y_re;
        out.push(sign_from_value(psi_re));
    }
    out
}

fn signs_from_superposition_hydrogenic(
    samples: &[[f32; 3]],
    q1: QuantumNumbers,
    q2: QuantumNumbers,
    mix: f32,
    time: f32,
    delta_e: f32,
) -> Vec<i8> {
    let mut out = Vec::with_capacity(samples.len());
    let a = mix.sqrt();
    let b = (1.0 - mix).sqrt();
    let phase_re = (delta_e * time).cos();
    let phase_im = -(delta_e * time).sin();
    for p in samples {
        let x = p[0];
        let y = p[1];
        let z = p[2];
        let r = (x * x + y * y + z * z).sqrt();
        if r <= 1e-8 {
            out.push(1);
            continue;
        }
        let cos_theta = (z / r).clamp(-1.0, 1.0);
        let theta = cos_theta.acos();
        let phi = y.atan2(x);
        let r1 = radial_wavefunction(r, q1.n, q1.l);
        let r2 = radial_wavefunction(r, q2.n, q2.l);
        let (y1_re, _) = spherical_harmonic(theta, phi, q1.l, q1.m_l);
        let (y2_re, y2_im) = spherical_harmonic(theta, phi, q2.l, q2.m_l);
        let psi1_re = a * r1 * y1_re;
        let psi2_re = b * r2 * (y2_re * phase_re - y2_im * phase_im);
        out.push(sign_from_value(psi1_re + psi2_re));
    }
    out
}

fn signs_from_superposition_lda(
    samples: &[[f32; 3]],
    orb_a: &LdaOrbital,
    orb_b: &LdaOrbital,
    m_a: i32,
    m_b: i32,
    mix: f32,
    time: f32,
    delta_e: f32,
) -> Vec<i8> {
    let mut out = Vec::with_capacity(samples.len());
    let a = mix.sqrt();
    let b = (1.0 - mix).sqrt();
    let phase_re = (delta_e * time).cos();
    let phase_im = -(delta_e * time).sin();
    for p in samples {
        let x = p[0];
        let y = p[1];
        let z = p[2];
        let r = (x * x + y * y + z * z).sqrt();
        if r <= 1e-8 {
            out.push(1);
            continue;
        }
        let cos_theta = (z / r).clamp(-1.0, 1.0);
        let theta = cos_theta.acos();
        let phi = y.atan2(x);
        let r1 = interp_radial(r, &orb_a.radial_r, &orb_a.radial_rfn);
        let r2 = interp_radial(r, &orb_b.radial_r, &orb_b.radial_rfn);
        let (y1_re, _) = spherical_harmonic(theta, phi, orb_a.l, m_a);
        let (y2_re, y2_im) = spherical_harmonic(theta, phi, orb_b.l, m_b);
        let psi1_re = a * r1 * y1_re;
        let psi2_re = b * r2 * (y2_re * phase_re - y2_im * phase_im);
        out.push(sign_from_value(psi1_re + psi2_re));
    }
    out
}

fn build_radial_cdf(
    rs: &[f32],
    vs: &[f32],
    max_radius: f32,
    radial_kind: RadialKind,
) -> Vec<f32> {
    let mut cdf = vec![0.0; rs.len()];
    let mut total = 0.0_f32;
    for i in 1..rs.len() {
        let dr = rs[i] - rs[i - 1];
        let v0 = vs[i - 1];
        let v1 = vs[i];
        let w0 = match radial_kind {
            RadialKind::R => rs[i - 1] * rs[i - 1],
            RadialKind::Chi => 1.0,
        };
        let w1 = match radial_kind {
            RadialKind::R => rs[i] * rs[i],
            RadialKind::Chi => 1.0,
        };
        let area = if rs[i] <= max_radius {
            0.5 * (v0 * v0 * w0 + v1 * v1 * w1) * dr
        } else {
            0.0
        };
        total += area;
        cdf[i] = total;
    }
    if total > 0.0 {
        for v in &mut cdf {
            *v /= total;
        }
    }
    cdf
}

fn sample_r<R: rand::Rng>(cdf: &[f32], rs: &[f32], rng: &mut R) -> f32 {
    let u = rng.gen::<f32>();
    let idx = match cdf.binary_search_by(|v| v.partial_cmp(&u).unwrap()) {
        Ok(i) => i,
        Err(i) => i.min(cdf.len() - 1),
    };
    if idx == 0 {
        return rs[0];
    }
    let c0 = cdf[idx - 1];
    let c1 = cdf[idx];
    let r0 = rs[idx - 1];
    let r1 = rs[idx];
    let t = if c1 > c0 { (u - c0) / (c1 - c0) } else { 0.0 };
    r0 + (r1 - r0) * t
}

fn max_angular_prob(l: u32, m_l: i32) -> f32 {
    use std::f32::consts::PI;
    let mut max_val = 0.0_f32;
    for i in 0..720 {
        let theta = (i as f32 + 0.5) / 720.0 * PI;
        let ang = angular_wavefunction(theta, 0.0, l, m_l);
        let p = ang * ang;
        if p > max_val {
            max_val = p;
        }
    }
    max_val.max(1e-8)
}

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/", get(index))
        .route("/info", get(info))
        .route("/samples", get(samples))
        .route("/static/three.module.js", get(three_module))
        .route("/static/MarchingCubes.js", get(marching_cubes));
    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    println!("Serving on http://127.0.0.1:3000");
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
