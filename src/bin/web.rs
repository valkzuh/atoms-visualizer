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
    angular_wavefunction_basis, generate_orbital_samples, generate_orbital_samples_basis,
    radial_wavefunction, real_spherical_harmonic, spherical_harmonic, AngularBasis, QuantumNumbers,
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
    basis: Option<String>,
    color_mode: Option<String>,
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
    phases: Option<Vec<f32>>,
    intensities: Option<Vec<f32>>,
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
      :root {
        --bg: #0a0c12;
        --bg-2: #0c111a;
        --panel: rgba(14, 18, 26, 0.92);
        --panel-border: #1f2732;
        --text: #e7edf5;
        --muted: #9aa6b6;
        --muted-2: #7b8796;
        --accent: #46d7c6;
        --accent-2: #f7b059;
        --accent-3: #4aa3ff;
      }
      html, body { margin: 0; padding: 0; height: 100%; background: #0b1016; color: var(--text); font-family: "Space Grotesk", "Segoe UI", sans-serif; }
      body::before {
        content: "";
        position: fixed;
        inset: 0;
        background-image:
          radial-gradient(circle at 20% 20%, rgba(70,215,198,0.12), transparent 35%),
          radial-gradient(circle at 80% 30%, rgba(74,163,255,0.14), transparent 35%),
          radial-gradient(circle at 30% 80%, rgba(247,176,89,0.08), transparent 40%),
          radial-gradient(rgba(255,255,255,0.08) 1px, transparent 1px),
          radial-gradient(rgba(255,255,255,0.04) 1px, transparent 1px);
        background-size: 100% 100%, 100% 100%, 100% 100%, 120px 120px, 240px 240px;
        opacity: 0.35;
        pointer-events: none;
      }
      canvas { display: block; }
      #panel { position: fixed; top: 24px; left: 24px; width: 420px; height: calc(100vh - 48px); background: var(--panel); padding: 18px; border: 1px solid var(--panel-border); border-radius: 18px; box-shadow: 0 22px 60px rgba(0,0,0,0.55); backdrop-filter: blur(8px); overflow: hidden; display: flex; flex-direction: column; transition: transform 0.2s ease, opacity 0.2s ease; }
      #panel.collapsed { transform: translateX(-120%); opacity: 0; pointer-events: none; }
      #panel.collapsed #panelInner { display: none; }
      #panel.collapsed .panel-meta { display: none; }
      #panel.collapsed .brand { font-size: 14px; }
      #panelInner { flex: 1; overflow: auto; padding-left: 12px; padding-right: 6px; direction: rtl; scrollbar-width: thin; scrollbar-color: rgba(70, 215, 198, 0.65) rgba(8, 12, 18, 0.6); }
      #panelInner > * { direction: ltr; }
      #panelInner::-webkit-scrollbar { width: 10px; }
      #panelInner::-webkit-scrollbar-track { background: rgba(8, 12, 18, 0.6); border-radius: 999px; }
      #panelInner::-webkit-scrollbar-thumb { background: linear-gradient(180deg, rgba(70, 215, 198, 0.85), rgba(74, 163, 255, 0.7)); border-radius: 999px; border: 2px solid rgba(8, 12, 18, 0.7); }
      #panelInner::-webkit-scrollbar-thumb:hover { background: linear-gradient(180deg, rgba(86, 235, 220, 0.95), rgba(120, 190, 255, 0.9)); }
      #infoButton { position: absolute; top: 16px; right: 16px; background: #111722; border: 1px solid #2b3545; color: var(--text); border-radius: 10px; padding: 8px 12px; font-size: 12px; text-decoration: none; box-shadow: 0 6px 18px rgba(0,0,0,0.3); }
      #infoButton:hover { border-color: var(--accent-3); color: #ffffff; }
      .panel-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; gap: 8px; }
      .panel-meta { font-size: 10px; text-transform: uppercase; letter-spacing: 0.28em; color: var(--muted-2); }
      .brand { font-size: 17px; font-weight: 600; letter-spacing: 0.04em; display: flex; align-items: center; gap: 8px; }
      #menuToggle { background: #0f1623; border: 1px solid #2a364a; color: var(--text); border-radius: 10px; padding: 6px 10px; font-size: 12px; cursor: pointer; }
      #menuToggle:hover { border-color: var(--accent-3); color: #ffffff; }
      #panelDock { position: fixed; top: 24px; left: 24px; display: none; align-items: center; gap: 10px; background: var(--panel); border: 1px solid var(--panel-border); border-radius: 14px; padding: 10px 12px; box-shadow: 0 16px 40px rgba(0,0,0,0.5); backdrop-filter: blur(8px); }
      #panelDock .brand { font-size: 14px; white-space: nowrap; }
      #panelDock.show { display: flex; }
      .brand::before { content: ""; width: 10px; height: 10px; border-radius: 50%; background: linear-gradient(120deg, var(--accent), var(--accent-2)); display: inline-block; box-shadow: 0 0 12px rgba(70,215,198,0.6); }
      .section { margin-top: 12px; padding: 12px; border: 1px solid #1b2431; border-radius: 14px; background: rgba(10, 14, 22, 0.7); }
      .section:first-of-type { margin-top: 8px; }
      .section-title { font-size: 11px; text-transform: uppercase; letter-spacing: 0.2em; color: var(--muted-2); margin-bottom: 6px; }
      .section-toggle { width: 100%; display: flex; align-items: center; justify-content: space-between; background: transparent; border: none; color: var(--text); padding: 6px 2px; font-size: 12px; text-transform: uppercase; letter-spacing: 0.22em; cursor: pointer; }
      .section-toggle::after { content: "+"; color: var(--muted-2); }
      .section-toggle.open::after { content: "-"; }
      .section-body { display: none; margin-top: 8px; }
      .section-body.open { display: block; }
      .row { display: flex; align-items: center; gap: 10px; margin-top: 10px; flex-wrap: wrap; }
      .row label { font-size: 11px; color: var(--muted); min-width: 42px; }
      #quantumRow { display: grid; grid-template-columns: auto 1fr auto 1fr auto 1fr; align-items: center; gap: 8px; }
      #quantumRow label { min-width: 0; }
      #quantumRow input { width: 64px; }
      @media (max-width: 520px) {
        #quantumRow { grid-template-columns: repeat(2, auto 1fr); }
      }
      input, select { background: #0e141f; color: var(--text); border: 1px solid #263042; border-radius: 9px; padding: 7px 9px; font-size: 12px; }
      input[type="number"] { width: 70px; }
      select { flex: 1; min-width: 160px; }
      input[type="range"] { accent-color: var(--accent); }
      button { background: #111a28; color: var(--text); border: 1px solid #2a364a; border-radius: 9px; padding: 8px 12px; font-size: 12px; cursor: pointer; }
      button.primary { background: linear-gradient(120deg, #1c2c3d, #1f3b52); border-color: #3f6f9d; }
      button.ghost { background: transparent; border-color: #2a364a; color: var(--muted); }
      button.ghost:hover { border-color: var(--accent-3); color: var(--text); }
      button:disabled { opacity: 0.6; cursor: default; }
      #controls { margin-top: 6px; font-size: 12px; color: var(--muted); }
      #status { margin-top: 12px; font-size: 12px; color: #b7c3d3; }
      .hint { font-size: 11px; color: var(--muted-2); margin-top: 6px; }
      #animControls { margin-top: 8px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; font-size: 12px; color: #c9d1d9; }
      #animatedRow { display: inline-flex; align-items: center; gap: 6px; }
      #animControls input[type="range"] { width: 140px; }
      #mixRow { margin-top: 8px; display: none; align-items: center; gap: 8px; font-size: 12px; color: #c9d1d9; flex-wrap: wrap; }
      #mixRow input[type="range"] { width: 140px; }
      #orbitalRow, #superRow { margin-top: 8px; display: none; align-items: center; gap: 8px; font-size: 12px; color: #c9d1d9; flex-wrap: wrap; }
      #superRow input[type="number"] { width: 58px; }
      .dropdown { position: relative; }
      .dropdown-btn { width: 100%; text-align: left; display: flex; justify-content: space-between; align-items: center; gap: 6px; }
      .dropdown-panel { position: absolute; left: 0; right: 0; top: 100%; margin-top: 8px; background: #0f141f; border: 1px solid #263042; border-radius: 12px; padding: 10px; max-height: 380px; overflow: auto; display: none; z-index: 5; box-shadow: 0 14px 30px rgba(0,0,0,0.4); }
      .dropdown-panel.open { display: block; }
      #elementSearch { width: 100%; margin-bottom: 12px; }
      #periodicGrid { display: grid; grid-template-columns: repeat(18, minmax(0, 1fr)); gap: 8px; width: 100%; min-width: 0; }
      .series-label { font-size: 11px; color: var(--muted-2); margin-top: 10px; margin-bottom: 6px; }
      .series-grid { display: grid; grid-template-columns: repeat(15, minmax(0, 1fr)); gap: 6px; width: 100%; min-width: 0; }
      .periodic-cell { height: 38px; display: flex; align-items: center; justify-content: center; font-size: 12px; border-radius: 9px; }
      .el-btn { background: #121a27; border: 1px solid #2a364a; color: var(--text); cursor: pointer; }
      .el-btn:hover { border-color: var(--accent-3); }
      .el-btn.active { background: #1c2b3d; border-color: var(--accent-3); color: #d8ebff; }
      .el-empty { border: 1px dashed #233043; color: #3a4450; }
      .modal { position: fixed; inset: 0; display: none; align-items: center; justify-content: center; z-index: 20; }
      .modal.open { display: flex; }
      .modal::before { content: ""; position: absolute; inset: 0; background: rgba(4,8,12,0.7); backdrop-filter: blur(6px); }
      .modal-card { position: relative; width: min(1100px, 94vw); max-height: 86vh; overflow-y: auto; overflow-x: hidden; background: #0f141f; border: 1px solid #263042; border-radius: 16px; padding: 18px; box-shadow: 0 24px 60px rgba(0,0,0,0.6); box-sizing: border-box; }
      .modal-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; }
      .modal-title { font-size: 16px; letter-spacing: 0.12em; text-transform: uppercase; }
      .modal-sub { font-size: 12px; color: var(--muted-2); margin-top: 2px; }
      .modal-grid { display: grid; gap: 10px; width: 100%; box-sizing: border-box; }
    </style>
  </head>
  <body>
    <a id="infoButton" href="/info">Info</a>
    <div id="panelDock">
      <div class="brand">Quantum Orbitals</div>
      <button id="menuShow" class="ghost">Show</button>
    </div>
    <div id="panel">
      <div class="panel-header">
        <div class="brand">Quantum Orbitals</div>
        <button id="menuToggle" title="Toggle menu">Hide</button>
        <div class="panel-meta">Menu</div>
      </div>
      <div id="panelInner">
        <div class="section" data-section="render">
          <button class="section-toggle open" data-target="renderBody">Render</button>
          <div id="renderBody" class="section-body open">
            <div class="row">
              <label>Render</label>
              <select id="renderMode">
                <option value="dots" selected>Dots</option>
                <option value="bubbles">Bubbles</option>
              </select>
            </div>
            <div id="dotColorRow" class="row">
              <label>Dot color</label>
              <select id="dotColorMode">
                <option value="radial" selected>Radial</option>
                <option value="phase">Phase</option>
                <option value="intensity">Intensity</option>
              </select>
            </div>
            <div id="dotSizeRow" class="row">
              <label>Dot size</label>
              <input id="dotSize" type="range" min="0.0005" max="0.1" step="0.0005" value="0.002" />
              <span id="dotSizeVal">0.002</span>
            </div>
            <div id="bubbleThresholdRow" class="row" style="display: none;">
              <label>Threshold</label>
              <input id="bubbleThreshold" type="range" min="0.10" max="0.90" step="0.02" value="0.45" />
              <span id="bubbleThresholdVal">0.45</span>
            </div>
            <div id="bubbleQualityRow" class="row" style="display: none;">
              <label>Quality</label>
              <input id="bubbleQuality" type="range" min="1" max="4" step="1" value="2" />
              <span id="bubbleQualityVal">Medium (48^3)</span>
            </div>
          </div>
        </div>

        <div class="section" data-section="element">
          <button class="section-toggle open" data-target="elementBody">Element</button>
          <div id="elementBody" class="section-body open">
            <div class="row">
              <button id="elementButton" class="dropdown-btn primary">H Hydrogen (Z=1)</button>
            </div>
            <div class="row">
              <label>Z</label><input id="z" type="number" min="1" max="118" value="1" />
              <button id="go" class="primary">Generate</button>
            </div>
            <div class="hint">Click the element name to open the periodic table.</div>
          </div>
        </div>

        <div class="section" data-section="view">
          <button class="section-toggle open" data-target="viewBody">View</button>
          <div id="viewBody" class="section-body open">
            <div class="row">
              <label>Mode</label>
              <select id="mode">
                <option value="total" selected>Total density</option>
                <option value="valence">Valence density</option>
                <option value="orbital">Single orbital</option>
                <option value="superposition">Superposition</option>
              </select>
            </div>
            <div id="basisRow" class="row" style="display: none;">
              <label>Basis</label>
              <select id="basis">
                <option value="real" selected>Real (chemistry)</option>
                <option value="complex">Complex (m)</option>
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
        </div>

        <div class="section" data-section="sampling">
          <button class="section-toggle" data-target="samplingBody">Sampling</button>
          <div id="samplingBody" class="section-body">
            <div class="row">
              <label>cnt</label><input id="count" type="number" min="1000" step="1000" value="50000" />
              <label>max</label><input id="max" type="number" min="1" step="1" value="20" />
            </div>
          </div>
        </div>

        <div class="section" data-section="controls">
          <button class="section-toggle open" data-target="controlsBody">Controls</button>
          <div id="controlsBody" class="section-body open">
            <div id="controls">Drag to orbit - Scroll to zoom - WASD to move (bounded)</div>
            <div class="row">
              <button id="resetCamera">Reset camera</button>
            </div>
            <div id="animControls">
              <span id="animatedRow"><label><input id="animated" type="checkbox" /> Animated (time evolution)</label></span>
              <label id="animSpeedLabel">Speed</label>
              <input id="animSpeed" type="range" min="0" max="3" step="0.05" value="1" />
              <span id="animSpeedVal">1.00x</span>
            </div>
          </div>
        </div>

        <div id="status">Ready.</div>
      </div>
    </div>

    <div id="elementModal" class="modal">
      <div class="modal-card">
        <div class="modal-header">
          <div>
            <div class="modal-title">Periodic Table</div>
            <div class="modal-sub">Choose an element preset</div>
          </div>
          <button id="closeTable" class="ghost">Close</button>
        </div>
        <input id="elementSearch" type="text" placeholder="Filter by symbol or name" />
        <div class="modal-grid">
          <div id="periodicGrid"></div>
          <div class="series-label">Lanthanides</div>
          <div id="lanthRow" class="series-grid"></div>
          <div class="series-label">Actinides</div>
          <div id="actRow" class="series-grid"></div>
        </div>
      </div>
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
      const panel = document.getElementById("panel");
      const panelInner = document.getElementById("panelInner");
      const menuToggle = document.getElementById("menuToggle");
      const panelDock = document.getElementById("panelDock");
      const menuShow = document.getElementById("menuShow");
      const elementButton = document.getElementById("elementButton");
      const elementModal = document.getElementById("elementModal");
      const closeTableButton = document.getElementById("closeTable");
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
      const dotColorSelect = document.getElementById("dotColorMode");
      const dotColorRow = document.getElementById("dotColorRow");
      const dotSizeRow = document.getElementById("dotSizeRow");
      const dotSizeInput = document.getElementById("dotSize");
      const dotSizeVal = document.getElementById("dotSizeVal");
      const valenceRow = document.getElementById("valenceRow");
      const valenceStyleSelect = document.getElementById("valenceStyle");
      const basisRow = document.getElementById("basisRow");
      const basisSelect = document.getElementById("basis");
      const bubbleThresholdRow = document.getElementById("bubbleThresholdRow");
      const bubbleThresholdInput = document.getElementById("bubbleThreshold");
      const bubbleThresholdVal = document.getElementById("bubbleThresholdVal");
      const bubbleQualityRow = document.getElementById("bubbleQualityRow");
      const bubbleQualityInput = document.getElementById("bubbleQuality");
      const bubbleQualityVal = document.getElementById("bubbleQualityVal");
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
      const animatedRow = document.getElementById("animatedRow");
      const animSpeedLabel = document.getElementById("animSpeedLabel");
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x0b1016);

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
      let bubbleSampleTarget = 3500;
      let bubbleResolution = 48;
      const bubbleKernelRadius = 1;
      let bubbleKernelSigma = 0.45;
      let bubbleIsoFraction = 0.45;
      let bubbleUpdateInterval = 60;
      let bubbleQuality = 2;
      let dotColorMode = "radial";
      let dotSize = 0.002;
      let spinTime = 0;
      let spinPhi0 = null;
      let spinRho = null;
      let spinOmega = null;
      let spinZ = null;

      function buildBubbleKernel() {
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
      }

      let bubbleKernel = buildBubbleKernel();

      function updateAnimUI() {
        animSpeedVal.textContent = animSpeed.toFixed(2) + "x";
        const isSuper = modeSelect.value === "superposition";
        const isOrbital = modeSelect.value === "orbital";
        animSpeedInput.disabled = isSuper ? !animateEnabled : !isOrbital;
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

      const bubbleQualityPresets = [
        { label: "Low", resolution: 36, samples: 3000, sigma: 0.5, interval: 80 },
        { label: "Medium", resolution: 48, samples: 5000, sigma: 0.45, interval: 60 },
        { label: "High", resolution: 64, samples: 12000, sigma: 0.45, interval: 70 },
        { label: "Ultra", resolution: 80, samples: 20000, sigma: 0.42, interval: 90 }
      ];

      function applyBubbleQuality(level, persist = true) {
        const index = Math.min(Math.max(level, 1), bubbleQualityPresets.length) - 1;
        const preset = bubbleQualityPresets[index];
        bubbleQuality = index + 1;
        bubbleResolution = preset.resolution;
        bubbleSampleTarget = preset.samples;
        bubbleKernelSigma = preset.sigma;
        bubbleUpdateInterval = preset.interval;
        bubbleKernel = buildBubbleKernel();
        bubbleQualityVal.textContent = `${preset.label} (${preset.resolution}^3)`;
        bubbleQualityInput.value = String(bubbleQuality);
        if (persist) {
          localStorage.setItem("bubbleQuality", String(bubbleQuality));
        }
        if (bubbleQuality >= 3 && bubbleIsoFraction > 0.8) {
          bubbleIsoFraction = 0.65;
          bubbleThresholdInput.value = bubbleIsoFraction.toFixed(2);
          updateBubbleThresholdUI();
          localStorage.setItem("bubbleIso", bubbleIsoFraction.toFixed(2));
          statusEl.textContent = "Threshold lowered for high quality to keep bubbles visible.";
        }
        if (bubbleGroup) {
          scene.remove(bubbleGroup);
          bubbleGroup = null;
          bubblePos = null;
          bubbleNeg = null;
        }
        if (renderMode === "bubbles") {
          initBubbles();
          if (posAttr) {
            updateBubblesFromPositions(posAttr.array, lastSigns);
          }
        }
      }

      function updateRenderMode() {
        renderMode = renderModeSelect.value;
        localStorage.setItem("renderMode", renderMode);
        const showBubbles = renderMode === "bubbles";
        bubbleThresholdRow.style.display = showBubbles ? "flex" : "none";
        bubbleQualityRow.style.display = showBubbles ? "flex" : "none";
        dotColorRow.style.display = showBubbles ? "none" : "flex";
        dotSizeRow.style.display = showBubbles ? "none" : "flex";
        dotColorSelect.disabled = showBubbles;
        updateModeUI();
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

      function updateBubbleThresholdUI() {
        bubbleThresholdVal.textContent = bubbleIsoFraction.toFixed(2);
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

      dotColorMode = localStorage.getItem("dotColorMode") || "radial";
      dotColorSelect.value = dotColorMode;
      dotColorSelect.addEventListener("change", () => {
        dotColorMode = dotColorSelect.value;
        localStorage.setItem("dotColorMode", dotColorMode);
        fetchSamples().catch((err) => { statusEl.textContent = err.toString(); });
      });

      const storedDotSize = localStorage.getItem("dotSize");
      if (storedDotSize) {
        const parsed = Number(storedDotSize);
        if (!Number.isNaN(parsed)) {
          dotSize = parsed;
        }
      }
      function updateDotSizeUI() {
        dotSizeVal.textContent = dotSize.toFixed(4);
        dotSizeInput.value = dotSize.toFixed(4);
      }
      updateDotSizeUI();
      dotSizeInput.addEventListener("input", () => {
        dotSize = Number(dotSizeInput.value);
        updateDotSizeUI();
        localStorage.setItem("dotSize", dotSize.toFixed(4));
        if (points && points.material) {
          points.material.size = dotSize;
          points.material.needsUpdate = true;
        }
      });

      const storedQuality = localStorage.getItem("bubbleQuality");
      if (storedQuality) {
        const parsedQuality = parseInt(storedQuality, 10);
        if (!Number.isNaN(parsedQuality)) {
          bubbleQuality = parsedQuality;
        }
      }
      applyBubbleQuality(bubbleQuality, false);
      bubbleQualityInput.addEventListener("input", () => {
        applyBubbleQuality(parseInt(bubbleQualityInput.value, 10));
      });

      const storedIso = localStorage.getItem("bubbleIso");
      if (storedIso) {
        const parsed = Number(storedIso);
        if (!Number.isNaN(parsed)) {
          bubbleIsoFraction = parsed;
          bubbleThresholdInput.value = parsed.toFixed(2);
        }
      }
      updateBubbleThresholdUI();
      bubbleThresholdInput.addEventListener("input", () => {
        bubbleIsoFraction = Number(bubbleThresholdInput.value);
        localStorage.setItem("bubbleIso", bubbleIsoFraction.toFixed(2));
        updateBubbleThresholdUI();
        if (renderMode === "bubbles" && posAttr) {
          bubbleDirty = true;
          updateBubblesFromPositions(posAttr.array, lastSigns);
        }
      });

      const storedBasis = localStorage.getItem("orbitalBasis");
      if (storedBasis) {
        basisSelect.value = storedBasis;
      } else {
        basisSelect.value = "real";
        localStorage.setItem("orbitalBasis", "real");
      }
      basisSelect.addEventListener("change", () => {
        localStorage.setItem("orbitalBasis", basisSelect.value);
        if (basisSelect.value === "real") {
          const lVal = Number(lInput.value);
          if (lVal > 0 && Number(mInput.value) === 0) {
            mInput.value = Math.min(lVal, 3);
          }
          const l2Val = Number(l2Input.value);
          if (l2Val > 0 && Number(m2Input.value) === 0) {
            m2Input.value = Math.min(l2Val, 3);
          }
          if (renderMode === "bubbles" && bubbleIsoFraction < 0.35) {
            bubbleIsoFraction = 0.45;
            bubbleThresholdInput.value = bubbleIsoFraction.toFixed(2);
            localStorage.setItem("bubbleIso", bubbleIsoFraction.toFixed(2));
            updateBubbleThresholdUI();
          }
        }
        fetchSamples().catch((err) => { statusEl.textContent = err.toString(); });
      });

      updateAnimUI();
      updateMixUI();
      updateRenderMode();

      function setPanelCollapsed(collapsed, persist = true) {
        panel.classList.toggle("collapsed", collapsed);
        menuToggle.textContent = collapsed ? "Show" : "Hide";
        panelDock.classList.toggle("show", collapsed);
        if (persist) {
          localStorage.setItem("panelCollapsed", collapsed ? "1" : "0");
        }
      }

      const storedCollapsed = localStorage.getItem("panelCollapsed");
      if (storedCollapsed === "1") {
        setPanelCollapsed(true, false);
      }
      menuToggle.addEventListener("click", () => {
        const next = !panel.classList.contains("collapsed");
        setPanelCollapsed(next);
      });
      menuShow.addEventListener("click", () => {
        setPanelCollapsed(false);
      });

      const sectionToggles = Array.from(document.querySelectorAll(".section-toggle"));
      for (const toggle of sectionToggles) {
        const targetId = toggle.dataset.target;
        const body = targetId ? document.getElementById(targetId) : null;
        if (body && body.classList.contains("open")) {
          toggle.classList.add("open");
        }
        toggle.addEventListener("click", () => {
          if (!body) return;
          const isOpen = body.classList.toggle("open");
          toggle.classList.toggle("open", isOpen);
        });
      }

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
        const showBubbles = renderMode === "bubbles";
        valenceRow.style.display = mode === "valence" ? "flex" : "none";
        basisRow.style.display = (orbitalMode || superMode) ? "flex" : "none";
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
        const showAnim = superMode || orbitalMode;
        animControls.style.display = showAnim ? "flex" : "none";
        if (animatedRow) {
          animatedRow.style.display = superMode ? "inline-flex" : "none";
        }
        if (animSpeedLabel) {
          animSpeedLabel.textContent = superMode ? "Speed" : "Spin speed";
        }
        updateAnimUI();
        updateMixUI();
      }

      orbitalSelect.addEventListener("change", () => {
        const [nStr, lStr] = orbitalSelect.value.split(",", 2);
        if (nStr && lStr) {
          nInput.value = nStr;
          lInput.value = lStr;
          if (basisSelect.value === "real") {
            const lVal = Number(lInput.value);
            if (lVal > 0 && Number(mInput.value) === 0) {
              mInput.value = Math.min(lVal, 3);
            }
          }
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
      nInput.addEventListener("change", () => {
        fetchSamples().catch((err) => { statusEl.textContent = err.toString(); });
      });
      lInput.addEventListener("change", () => {
        if (basisSelect.value === "real") {
          const lVal = Number(lInput.value);
          if (lVal > 0 && Number(mInput.value) === 0) {
            mInput.value = Math.min(lVal, 3);
          }
        }
        fetchSamples().catch((err) => { statusEl.textContent = err.toString(); });
      });
      mInput.addEventListener("change", () => {
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
          elementModal.classList.remove("open");
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

      function openElementModal() {
        elementModal.classList.add("open");
        elementSearch.focus();
      }

      function closeElementModal() {
        elementModal.classList.remove("open");
      }

      elementButton.addEventListener("click", () => {
        openElementModal();
      });

      closeTableButton.addEventListener("click", () => {
        closeElementModal();
      });

      elementModal.addEventListener("click", (e) => {
        if (e.target === elementModal) {
          closeElementModal();
        }
      });

      document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
          closeElementModal();
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
        const twoPi = Math.PI * 2;
        if (theta < 0) {
          theta = (theta % twoPi) + twoPi;
        } else if (theta > twoPi) {
          theta = theta % twoPi;
        }
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

      function hsvToRgb(h, s, v) {
        const i = Math.floor(h * 6.0);
        const f = h * 6.0 - i;
        const p = v * (1.0 - s);
        const q = v * (1.0 - f * s);
        const t = v * (1.0 - (1.0 - f) * s);
        switch (i % 6) {
          case 0: return [v, t, p];
          case 1: return [q, v, p];
          case 2: return [p, v, t];
          case 3: return [p, q, v];
          case 4: return [t, p, v];
          case 5: return [v, p, q];
          default: return [v, t, p];
        }
      }

      function colorForPhase(phase) {
        const t = (phase + Math.PI) / (2.0 * Math.PI);
        const h = ((t % 1) + 1) % 1;
        const [r, g, b] = hsvToRgb(h, 0.95, 0.95);
        return new THREE.Color(r, g, b);
      }

      function colorForIntensity(value, maxValue) {
        const tRaw = maxValue > 0 ? Math.min(value / maxValue, 1) : 0;
        const t = Math.pow(tRaw, 0.4);
        const stops = [
          { t: 0.0, c: [0.02, 0.02, 0.08] },
          { t: 0.25, c: [0.25, 0.05, 0.45] },
          { t: 0.55, c: [0.85, 0.2, 0.2] },
          { t: 0.8, c: [0.98, 0.72, 0.2] },
          { t: 1.0, c: [1.0, 1.0, 1.0] },
        ];
        let a = stops[0];
        let b = stops[stops.length - 1];
        for (let i = 0; i < stops.length - 1; i++) {
          if (t >= stops[i].t && t <= stops[i + 1].t) {
            a = stops[i];
            b = stops[i + 1];
            break;
          }
        }
        const k = (t - a.t) / Math.max(1e-6, (b.t - a.t));
        const r = a.c[0] + (b.c[0] - a.c[0]) * k;
        const g = a.c[1] + (b.c[1] - a.c[1]) * k;
        const bcol = a.c[2] + (b.c[2] - a.c[2]) * k;
        return new THREE.Color(r, g, bcol);
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

      function updateSuperpositionPhaseColors() {
        if (!superPsi || !colorAttr) {
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
        const colors = colorAttr.array;
        const count = colors.length / 3;
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
          const phi = Math.atan2(im, re);
          const c = colorForPhase(phi);
          const baseIdx = i * 3;
          colors[baseIdx + 0] = c.r;
          colors[baseIdx + 1] = c.g;
          colors[baseIdx + 2] = c.b;
        }
        colorAttr.needsUpdate = true;
      }

      function updateSuperpositionIntensityColors() {
        if (!superPsi || !colorAttr || !superProb) {
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
        const colors = colorAttr.array;
        for (let i = 0; i < count; i++) {
          const baseIdx = i * 3;
          const c = colorForIntensity(superProb[i], maxProb);
          colors[baseIdx + 0] = c.r;
          colors[baseIdx + 1] = c.g;
          colors[baseIdx + 2] = c.b;
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
        const wantPhaseMode = renderMode === "dots" && dotColorMode === "phase";
        const wantIntensityMode = renderMode === "dots" && dotColorMode === "intensity";
        const wantPsi = animateEnabled && mode === "superposition" && (wantPhaseMode || wantIntensityMode);
        const wantBubbles = renderMode === "bubbles";
        let effectiveCount = count;
        if (wantMorph) {
          effectiveCount = count;
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
          const basisMode = (mode === "orbital" || mode === "superposition") ? basisSelect.value : "complex";
          const colorModeParam = wantPhaseMode ? "phase" : (wantIntensityMode ? "intensity" : "radial");
          const params = new URLSearchParams({ n, l, m, n2, l2, m2, z, count: effectiveCount, max, mode, mix, t, valence_style: valenceStyle, animated: wantPsi, bubble: wantBubbles, basis: basisMode, color_mode: colorModeParam });
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
        const basisLabel = (basisSelect.value === "real" && (modeLabel === "orbital" || modeLabel === "superposition"))
          ? " | real basis"
          : "";
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
        statusEl.textContent = `${elementLabel} | ${detail} | count=${data.count} | ${sourceLabel}${note}${basisLabel}`;
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
        const usePhase = dotColorMode === "phase"
          && Array.isArray(data.phases)
          && data.phases.length === data.samples.length;
        const useIntensity = dotColorMode === "intensity"
          && Array.isArray(data.intensities)
          && data.intensities.length === data.samples.length;
        let maxIntensity = 0.0;
        if (useIntensity) {
          for (let i = 0; i < data.intensities.length; i++) {
            const v = data.intensities[i];
            if (v > maxIntensity) maxIntensity = v;
          }
        }
        for (let i = 0; i < data.samples.length; i++) {
          const p = data.samples[i];
          positions[i * 3 + 0] = p[0] * 0.1;
          positions[i * 3 + 1] = p[1] * 0.1;
          positions[i * 3 + 2] = p[2] * 0.1;
          let c;
          if (usePhase) {
            c = colorForPhase(data.phases[i]);
          } else if (useIntensity) {
            c = colorForIntensity(data.intensities[i], maxIntensity);
          } else {
            const dist = Math.sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]) * 0.1;
            c = colorForDistance(dist, data.max_radius * 0.1);
          }
          colors[i * 3 + 0] = c.r;
          colors[i * 3 + 1] = c.g;
          colors[i * 3 + 2] = c.b;
        }

        const mValue = Number.isFinite(Number(data.m)) ? Number(data.m) : 0;
        if (renderMode === "dots" && modeLabel === "orbital" && mValue !== 0) {
          spinPhi0 = new Float32Array(data.samples.length);
          spinRho = new Float32Array(data.samples.length);
          spinOmega = new Float32Array(data.samples.length);
          spinZ = new Float32Array(data.samples.length);
          const mSign = mValue >= 0 ? 1 : -1;
          const mScale = Math.max(1, Math.abs(mValue));
          let maxR = 0.0;
          for (let i = 0; i < data.samples.length; i++) {
            const x = positions[i * 3 + 0];
            const y = positions[i * 3 + 1];
            const z = positions[i * 3 + 2];
            const r = Math.sqrt(x * x + y * y + z * z);
            if (r > maxR) maxR = r;
          }
          const invMaxR = maxR > 0 ? 1.0 / maxR : 1.0;
          const maxI = useIntensity && maxIntensity > 0 ? maxIntensity : 1.0;
          for (let i = 0; i < data.samples.length; i++) {
            const baseIdx = i * 3;
            const x0 = positions[baseIdx + 0];
            const y0 = positions[baseIdx + 1];
            const z0 = positions[baseIdx + 2];
            const rho = Math.sqrt(x0 * x0 + y0 * y0);
            const phi0 = Math.atan2(y0, x0);
            let norm;
            if (useIntensity) {
              norm = Math.min(data.intensities[i] / maxI, 1.0);
            } else {
              const r = Math.sqrt(x0 * x0 + y0 * y0 + z0 * z0);
              norm = 1.0 - Math.min(r * invMaxR, 1.0);
            }
            const curve = Math.pow(norm, 1.6);
            const base = 0.05;
            const max = 2.8;
            const omega = (base + (max - base) * curve) * mScale * mSign;
            spinPhi0[i] = phi0;
            spinRho[i] = rho;
            spinZ[i] = z0;
            spinOmega[i] = omega;
          }
          spinTime = 0;
        } else {
          spinPhi0 = null;
          spinRho = null;
          spinOmega = null;
          spinZ = null;
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
            size: dotSize,
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
          if (dotColorMode === "phase") {
            updateSuperpositionPhaseColors();
          } else if (dotColorMode === "intensity") {
            updateSuperpositionIntensityColors();
          } else {
            updateSuperpositionColors();
          }
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
            if (dotColorMode === "phase") {
              updateSuperpositionPhaseColors();
            } else if (dotColorMode === "intensity") {
              updateSuperpositionIntensityColors();
            } else {
              updateSuperpositionColors();
            }
          }
          if (!superFetchInFlight && (now - lastSampleTime) > animDurationMs * 0.9) {
            lastSampleTime = now;
            superFetchInFlight = true;
            fetchSamples()
              .catch((err) => { statusEl.textContent = err.toString(); })
              .finally(() => { superFetchInFlight = false; });
          }
        }
        const orbitalSpinEnabled = modeSelect.value === "orbital"
          && renderMode === "dots"
          && spinPhi0
          && spinRho
          && spinOmega
          && spinZ
          && posAttr;
        if (orbitalSpinEnabled) {
          spinTime += dt * animSpeed;
          const arr = posAttr.array;
          const count = spinOmega.length;
          for (let i = 0; i < count; i++) {
            const baseIdx = i * 3;
            const theta = spinPhi0[i] + spinTime * spinOmega[i];
            const rho = spinRho[i];
            arr[baseIdx + 0] = rho * Math.cos(theta);
            arr[baseIdx + 1] = rho * Math.sin(theta);
            arr[baseIdx + 2] = spinZ[i];
          }
          posAttr.needsUpdate = true;
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
      :root {
        --bg: #070b10;
        --panel: rgba(12, 16, 24, 0.88);
        --panel-2: rgba(10, 14, 20, 0.75);
        --border: #1b2534;
        --text: #e7edf5;
        --muted: #98a4b4;
        --muted-2: #728195;
        --accent: #4aa3ff;
        --accent-2: #46d7c6;
        --accent-3: #f7b059;
      }
      html, body { margin: 0; padding: 0; height: 100%; background: var(--bg); color: var(--text); font-family: "Space Grotesk", "Segoe UI", sans-serif; }
      body::before {
        content: "";
        position: fixed;
        inset: 0;
        background-image:
          radial-gradient(rgba(255,255,255,0.04) 1px, transparent 1px),
          radial-gradient(rgba(255,255,255,0.02) 1px, transparent 1px);
        background-size: 120px 120px, 26px 26px;
        opacity: 0.35;
        pointer-events: none;
      }
      #infoApp {
        display: grid;
        grid-template-columns: 260px 1fr;
        grid-template-rows: auto 1fr;
        grid-template-areas: "header header" "nav content";
        gap: 22px;
        min-height: 100vh;
        padding: 24px;
        box-sizing: border-box;
      }
      .info-header { grid-area: header; display: flex; align-items: center; justify-content: space-between; }
      .info-title { font-size: 28px; font-weight: 600; letter-spacing: 0.03em; }
      .info-subtitle { margin-top: 6px; font-size: 12px; text-transform: uppercase; letter-spacing: 0.22em; color: var(--muted-2); }
      .back { background: #111722; border: 1px solid #2b3545; color: var(--text); border-radius: 10px; padding: 8px 12px; font-size: 12px; text-decoration: none; box-shadow: 0 6px 18px rgba(0,0,0,0.3); }
      .back:hover { border-color: var(--accent); color: #ffffff; }

      .info-nav {
        grid-area: nav;
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 16px;
        display: flex;
        flex-direction: column;
        gap: 12px;
        height: calc(100vh - 140px);
        position: sticky;
        top: 24px;
        overflow: hidden;
      }
      .nav-title { font-size: 12px; text-transform: uppercase; letter-spacing: 0.22em; color: var(--muted-2); }
      #navSearch { width: 100%; background: #0c121b; border: 1px solid #263042; border-radius: 10px; padding: 8px 10px; color: var(--text); font-size: 12px; }
      .nav-list { display: flex; flex-direction: column; gap: 8px; overflow-y: auto; padding-right: 4px; flex: 1; min-height: 0; }
      .nav-item { text-align: left; background: transparent; border: 1px solid #202a3a; color: var(--text); padding: 8px 10px; border-radius: 10px; font-size: 13px; cursor: pointer; }
      .nav-item:hover { border-color: var(--accent); }
      .nav-item.active { background: #152234; border-color: var(--accent); color: #e9f4ff; box-shadow: 0 8px 18px rgba(0,0,0,0.25); }
      .nav-hint { font-size: 12px; color: var(--muted-2); line-height: 1.5; margin-top: auto; }

      .info-content {
        grid-area: content;
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 18px;
        display: flex;
        flex-direction: column;
        height: calc(100vh - 140px);
      }
      .content-header { display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #1a2230; padding-bottom: 10px; margin-bottom: 12px; }
      .content-header-title { font-size: 20px; font-weight: 600; }
      .content-meta { font-size: 11px; text-transform: uppercase; letter-spacing: 0.18em; color: var(--muted-2); }
      .section-container { flex: 1; overflow-y: auto; padding-right: 8px; }
      .info-section { display: none; animation: fadeIn 0.2s ease; }
      .info-section.active { display: block; }
      @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }

      .card { background: var(--panel-2); border: 1px solid #1b2433; border-radius: 14px; padding: 14px; margin-bottom: 14px; }
      .card h3 { margin: 0 0 8px 0; font-size: 15px; letter-spacing: 0.08em; text-transform: uppercase; color: #c4d0e0; }
      p { color: #c7d1df; line-height: 1.7; margin: 0 0 10px 0; }
      ul { margin: 0; padding-left: 18px; color: #c7d1df; line-height: 1.7; }
      li { margin-bottom: 6px; }
      pre { background: #0b1017; border: 1px solid #1f2a3a; border-radius: 10px; padding: 10px 12px; color: #d7e2f2; overflow-x: auto; }
      .grid-2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
      .grid-3 { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }
      .tag { display: inline-flex; align-items: center; gap: 6px; background: #0d141f; border: 1px solid #263042; border-radius: 999px; padding: 4px 10px; font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.14em; }
      .diagram-tabs { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
      .diagram-tab { background: #0d141f; border: 1px solid #263042; color: var(--text); border-radius: 10px; padding: 6px 10px; font-size: 12px; cursor: pointer; }
      .diagram-tab.active { border-color: var(--accent-2); color: #e9f4ff; }
      .diagram-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
      .diagram { display: none; background: #0b111a; border: 1px solid #1f2a3a; border-radius: 12px; padding: 12px; text-align: center; }
      .diagram.active { display: block; }
      .diagram svg { width: 100%; height: auto; max-height: 220px; }
      .diagram figcaption { margin-top: 8px; font-size: 12px; color: var(--muted); }
      .equation-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
      .equation-card { background: #0b111a; border: 1px solid #1f2a3a; border-radius: 12px; padding: 12px; }
      table { width: 100%; border-collapse: collapse; font-size: 12px; }
      th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #1b2534; }
      th { text-transform: uppercase; letter-spacing: 0.12em; font-size: 11px; color: var(--muted-2); }

      @media (max-width: 980px) {
        #infoApp { grid-template-columns: 1fr; grid-template-areas: "header" "nav" "content"; }
        .info-nav, .info-content { height: auto; position: static; }
        .grid-2, .grid-3, .equation-grid, .diagram-grid { grid-template-columns: 1fr; }
      }
    </style>
  </head>
  <body>
    <div id="infoApp">
      <header class="info-header">
        <div>
          <div class="info-title">Quantum Orbitals 3D</div>
          <div class="info-subtitle">Physics reference and UI guide</div>
        </div>
        <a class="back" href="/">Back to app</a>
      </header>

      <aside class="info-nav">
        <div class="nav-title">Explore</div>
        <input id="navSearch" type="text" placeholder="Filter topics" />
        <div id="navList" class="nav-list">
          <button class="nav-item active" data-target="overview" data-title="Overview">Overview</button>
          <button class="nav-item" data-target="quantum" data-title="Quantum Numbers">Quantum numbers</button>
          <button class="nav-item" data-target="hydrogenic" data-title="Hydrogenic Model">Hydrogenic model</button>
          <button class="nav-item" data-target="angular" data-title="Angular Shapes">Angular shapes</button>
          <button class="nav-item" data-target="radial" data-title="Radial Structure">Radial structure</button>
          <button class="nav-item" data-target="many" data-title="Many Electron Atoms">Many electron atoms</button>
          <button class="nav-item" data-target="rendering" data-title="Sampling and Rendering">Sampling and rendering</button>
          <button class="nav-item" data-target="color" data-title="Color and Phase">Color and phase</button>
          <button class="nav-item" data-target="superposition" data-title="Superposition">Superposition</button>
          <button class="nav-item" data-target="glossary" data-title="UI Glossary">UI glossary</button>
          <button class="nav-item" data-target="limits" data-title="Limitations">Limitations</button>
        </div>
        <div class="nav-hint">Click a topic to load the full section. The content switches instantly without hunting through long pages.</div>
      </aside>

      <main class="info-content">
        <div class="content-header">
          <div id="sectionTitle" class="content-header-title">Overview</div>
          <div class="content-meta">Physics and UI</div>
        </div>
        <div class="section-container">
          <section id="overview" class="info-section active">
            <div class="card">
              <h3>What this visualizer shows</h3>
              <p>The dots and bubbles visualize the probability density of an electron in an atom. Each dot is a Monte Carlo sample from |psi|^2, and each bubble is an isosurface derived from the same density. These are not particle paths. They are spatial probability distributions.</p>
              <p>The app supports hydrogenic orbitals, LDA single particle densities where available, and time dependent superpositions of two orbitals. Use the menu to switch between total density, valence density, single orbital, or superposition modes.</p>
            </div>
            <div class="grid-2">
              <div class="card">
                <h3>Getting started</h3>
                <ul>
                  <li>Select an element from the periodic table or edit Z and click Generate.</li>
                  <li>Choose dots for raw samples or bubbles for smooth isosurfaces.</li>
                  <li>Pick a mode and adjust quantum numbers for single orbital and superposition views.</li>
                  <li>Enable Animated to see time dependent interference in superposition mode.</li>
                </ul>
              </div>
              <div class="card">
                <h3>Controls</h3>
                <ul>
                  <li>Drag to orbit the camera around the nucleus.</li>
                  <li>Scroll to zoom in and out.</li>
                  <li>WASD moves the camera target within bounds.</li>
                  <li>Reset camera returns to the default view.</li>
                </ul>
              </div>
            </div>
          </section>

          <section id="quantum" class="info-section">
            <div class="card">
              <h3>Quantum numbers</h3>
              <p>An orbital is labeled by three quantum numbers: n, l, and m. These arise from separation of the Schrodinger equation in spherical coordinates and fully specify a stationary state.</p>
              <ul>
                <li>n is the principal quantum number. It controls the energy in hydrogenic atoms and the overall radial scale.</li>
                <li>l is the orbital angular momentum quantum number. It sets the lobe pattern. l = 0, 1, 2, 3 correspond to s, p, d, f.</li>
                <li>m is the magnetic quantum number, which selects the orientation or azimuthal structure. m ranges from -l to +l.</li>
              </ul>
              <p>Hydrogenic energy depends only on n, so all states with the same n are degenerate. This matters for superposition animation.</p>
              <p>The number of angular nodes equals l, and the number of radial nodes equals n - l - 1. Together they determine the lobe count and the number of radial shells.</p>
            </div>
          </section>

          <section id="hydrogenic" class="info-section">
            <div class="card">
              <h3>Hydrogenic model</h3>
              <p>Hydrogenic orbitals solve the time independent Schrodinger equation for a Coulomb potential. In atomic units:</p>
              <pre>[-1/2 * nabla^2 - Z / r] psi(r) = E psi(r)</pre>
              <p>Separation of variables yields:</p>
              <pre>psi(n,l,m)(r,theta,phi) = R_nl(r) * Y_lm(theta,phi)</pre>
              <p>The energy depends on n only:</p>
              <pre>E_n = -Z^2 / (2 n^2)</pre>
              <p>Angular structure comes from Y_lm, while radial structure comes from R_nl. The visualizer uses analytical R_nl and Y_lm to generate samples.</p>
              <p>Closed form solutions use associated Laguerre polynomials for R_nl and spherical harmonics for Y_lm. Atomic units set hbar = 1, m_e = 1, and e = 1, simplifying the equations.</p>
            </div>
            <div class="equation-grid">
              <div class="equation-card">
                <h3>Radial probability</h3>
                <pre>P(r) = r^2 * |R_nl(r)|^2</pre>
                <p>The r^2 term is why s orbitals still peak away from r = 0 even though R_nl is finite at the origin.</p>
              </div>
              <div class="equation-card">
                <h3>Angular probability</h3>
                <pre>P(theta,phi) = |Y_lm(theta,phi)|^2</pre>
                <p>This function defines the nodal planes and the number of lobes in the orbital.</p>
              </div>
              <div class="equation-card">
                <h3>Energy ladder</h3>
                <svg viewBox="0 0 220 160" aria-label="energy ladder">
                  <line x1="40" y1="130" x2="180" y2="130" stroke="#344052" stroke-width="2"></line>
                  <line x1="60" y1="110" x2="160" y2="110" stroke="#4aa3ff" stroke-width="3"></line>
                  <line x1="70" y1="80" x2="150" y2="80" stroke="#4aa3ff" stroke-width="3"></line>
                  <line x1="80" y1="55" x2="140" y2="55" stroke="#4aa3ff" stroke-width="3"></line>
                  <text x="22" y="113" fill="#8a98ac" font-size="10">n=1</text>
                  <text x="22" y="83" fill="#8a98ac" font-size="10">n=2</text>
                  <text x="22" y="58" fill="#8a98ac" font-size="10">n=3</text>
                </svg>
                <p>Energy scales as -Z^2 / (2 n^2), so the levels get closer together as n increases.</p>
              </div>
            </div>
          </section>

          <section id="angular" class="info-section">
            <div class="card">
              <h3>Angular shapes and real orbitals</h3>
              <p>The complex spherical harmonics Y_lm have a phase factor exp(i m phi). For a single complex Y_lm, the density |Y_lm|^2 is often azimuthally symmetric, producing rings or shells. Chemistry diagrams usually use real linear combinations of Y_lm and Y_l,-m to create lobes with clear nodal planes.</p>
              <p>Use the Basis selector in orbital and superposition views (dots or bubbles) to switch between complex and real orbitals.</p>
              <p>Red and blue indicate the sign of psi in bubbles mode. Nodal surfaces separate regions of opposite sign.</p>
            </div>
            <div class="card">
              <div class="diagram-tabs">
                <button class="diagram-tab active" data-diagram-group="orbitals" data-diagram-tab="s">s orbital</button>
                <button class="diagram-tab" data-diagram-group="orbitals" data-diagram-tab="p">p orbital</button>
                <button class="diagram-tab" data-diagram-group="orbitals" data-diagram-tab="d">d orbital</button>
                <button class="diagram-tab" data-diagram-group="orbitals" data-diagram-tab="f">f orbital</button>
              </div>
              <div class="diagram-grid">
                <figure class="diagram active" data-diagram-group="orbitals" data-diagram="s">
                  <svg viewBox="0 0 200 200" aria-label="s orbital">
                    <circle cx="100" cy="100" r="60" fill="#ff5c5c" opacity="0.85"></circle>
                    <circle cx="100" cy="100" r="60" fill="none" stroke="#ffffff" stroke-opacity="0.2" stroke-width="2"></circle>
                  </svg>
                  <figcaption>S orbitals are spherically symmetric. There is no angular node.</figcaption>
                </figure>
                <figure class="diagram" data-diagram-group="orbitals" data-diagram="p">
                  <svg viewBox="0 0 200 200" aria-label="p orbital">
                    <ellipse cx="100" cy="60" rx="42" ry="32" fill="#ff5c5c" opacity="0.85"></ellipse>
                    <ellipse cx="100" cy="140" rx="42" ry="32" fill="#4b68ff" opacity="0.85"></ellipse>
                    <rect x="20" y="98" width="160" height="4" fill="#202a3a"></rect>
                  </svg>
                  <figcaption>P orbitals have one nodal plane and two lobes with opposite sign.</figcaption>
                </figure>
                <figure class="diagram" data-diagram-group="orbitals" data-diagram="d">
                  <svg viewBox="0 0 200 200" aria-label="d orbital">
                    <ellipse cx="60" cy="60" rx="26" ry="20" fill="#ff5c5c" opacity="0.85"></ellipse>
                    <ellipse cx="140" cy="60" rx="26" ry="20" fill="#4b68ff" opacity="0.85"></ellipse>
                    <ellipse cx="60" cy="140" rx="26" ry="20" fill="#4b68ff" opacity="0.85"></ellipse>
                    <ellipse cx="140" cy="140" rx="26" ry="20" fill="#ff5c5c" opacity="0.85"></ellipse>
                  </svg>
                  <figcaption>D orbitals have two angular nodes and four main lobes.</figcaption>
                </figure>
                <figure class="diagram" data-diagram-group="orbitals" data-diagram="f">
                  <svg viewBox="0 0 200 200" aria-label="f orbital">
                    <circle cx="60" cy="40" r="18" fill="#ff5c5c" opacity="0.85"></circle>
                    <circle cx="140" cy="40" r="18" fill="#4b68ff" opacity="0.85"></circle>
                    <circle cx="40" cy="100" r="18" fill="#4b68ff" opacity="0.85"></circle>
                    <circle cx="160" cy="100" r="18" fill="#ff5c5c" opacity="0.85"></circle>
                    <circle cx="60" cy="160" r="18" fill="#ff5c5c" opacity="0.85"></circle>
                    <circle cx="140" cy="160" r="18" fill="#4b68ff" opacity="0.85"></circle>
                  </svg>
                  <figcaption>F orbitals add more angular nodes and complex lobe structures.</figcaption>
                </figure>
              </div>
            </div>
          </section>

          <section id="radial" class="info-section">
            <div class="card">
              <h3>Radial structure and nodes</h3>
              <p>Radial nodes are spherical shells where the wavefunction changes sign. The number of radial nodes equals n - l - 1. Increasing n adds more shells and pushes probability outward, while increasing l changes the angular structure without adding radial nodes.</p>
            </div>
            <div class="card">
              <div class="diagram-grid">
                <figure class="diagram active" data-diagram="radial">
                  <svg viewBox="0 0 260 180" aria-label="radial probability">
                    <line x1="30" y1="150" x2="240" y2="150" stroke="#354055" stroke-width="2"></line>
                    <line x1="30" y1="150" x2="30" y2="20" stroke="#354055" stroke-width="2"></line>
                    <path d="M30 140 C60 40, 120 30, 180 110 C210 150, 230 140, 240 120" fill="none" stroke="#46d7c6" stroke-width="3"></path>
                    <text x="210" y="165" fill="#8897ab" font-size="10">r</text>
                    <text x="10" y="30" fill="#8897ab" font-size="10">P(r)</text>
                  </svg>
                  <figcaption>Radial probability includes an r^2 factor, producing a peak away from the origin even for s orbitals.</figcaption>
                </figure>
              </div>
            </div>
          </section>

          <section id="many" class="info-section">
            <div class="card">
              <h3>Many electron atoms</h3>
              <p>For heavier elements the visualizer uses LDA radial functions when available. LDA is a Kohn-Sham density functional approximation that replaces the many body problem with an effective single particle potential. The output is a set of radial functions and occupancies for each orbital channel.</p>
              <p>Electron screening reduces the effective nuclear charge for outer shells. That is why valence orbitals are larger and more diffuse than the hydrogenic Z scaling alone would suggest.</p>
              <ul>
                <li>Total density sums all occupied orbitals, producing a spherical average.</li>
                <li>Valence density isolates the outermost occupied shells.</li>
                <li>Single orbital shows one selected (n, l) channel combined with Y_lm.</li>
              </ul>
              <p>LDA orbitals are not m resolved, so valence lobe mode uses m = 0 for shape visualization.</p>
            </div>
          </section>

          <section id="rendering" class="info-section">
            <div class="card">
              <h3>Sampling and rendering</h3>
              <p>Dots mode uses Monte Carlo rejection sampling in spherical coordinates. Radial samples are drawn from the radial probability distribution, and angular samples are accepted according to |Y_lm|^2. The count parameter controls the number of samples and therefore the noise level.</p>
              <p>Bubbles mode converts the point cloud into a smooth density grid and extracts an isosurface. The threshold control sets which density fraction becomes the surface. Lower thresholds show more diffuse lobes. Higher thresholds highlight the dense core.</p>
              <p>The surface is built from a kernel smoothed density, so it is an approximation of |psi|^2 and depends on grid resolution as well as the threshold.</p>
              <p>The Quality slider adjusts bubble grid resolution and the number of samples used to build the density field. Higher quality looks smoother but costs more performance.</p>
            </div>
            <div class="grid-2">
              <div class="card">
                <h3>Dots</h3>
                <p>Fast and faithful to the raw samples. Ideal for exploring high n orbitals and superpositions without heavy meshing costs.</p>
              </div>
              <div class="card">
                <h3>Bubbles</h3>
                <p>Shows the orbital as a continuous surface. Positive and negative lobes appear in red and blue when the sign of psi is defined.</p>
              </div>
            </div>
          </section>

          <section id="color" class="info-section">
            <div class="card">
              <h3>Color and phase</h3>
              <p>Dots can be colored by radial distance or by the complex phase of the wavefunction. Phase is not observable directly in |psi|^2, but it is essential for interference and superposition. Bubbles use red and blue to indicate positive and negative regions of psi when sign is defined.</p>
              <p>Phase hue mapping uses h = (phase + pi) / (2 pi) * 360 degrees with saturation and value set to 0.95. This means phase = -pi or +pi maps to red, phase = -pi/2 maps near yellow-green, phase = 0 maps to cyan, and phase = +pi/2 maps near purple-blue.</p>
              <p>Intensity mode maps |psi|^2 to a heat-style gradient from deep violet through red and gold to white, emphasizing the highest probability density regions.</p>
            </div>
            <div class="card">
              <div class="diagram-grid">
                <figure class="diagram active" data-diagram="phase">
                  <svg viewBox="0 0 200 200" aria-label="phase wheel">
                    <defs>
                      <linearGradient id="phaseGrad" x1="0" y1="0" x2="1" y2="0">
                        <stop offset="0%" stop-color="#ff4d4d" />
                        <stop offset="25%" stop-color="#f7b059" />
                        <stop offset="50%" stop-color="#46d7c6" />
                        <stop offset="75%" stop-color="#4aa3ff" />
                        <stop offset="100%" stop-color="#ff4d4d" />
                      </linearGradient>
                    </defs>
                    <circle cx="100" cy="100" r="70" fill="none" stroke="url(#phaseGrad)" stroke-width="18"></circle>
                    <text x="72" y="105" fill="#95a4b8" font-size="12">phase</text>
                  </svg>
                  <figcaption>Phase coloring wraps continuously. Opposite colors represent phase shifts of approximately pi.</figcaption>
                </figure>
              </div>
            </div>
          </section>

          <section id="superposition" class="info-section">
            <div class="card">
              <h3>Superposition and time evolution</h3>
              <p>A single eigenstate evolves only by a global phase factor exp(-i E t). The probability density is static. True animation requires at least two orbitals with different energies.</p>
              <pre>psi(r,t) = a * psi1(r) + b * psi2(r) * exp(-i * DeltaE * t)</pre>
              <pre>|psi|^2 = |a psi1|^2 + |b psi2|^2 + 2 Re[a b* psi1 psi2* exp(-i DeltaE t)]</pre>
              <p>The interference term produces real spatial motion in the density. In the hydrogenic model, states with the same n are degenerate, so DeltaE = 0 and the density does not evolve. Choose orbitals with different n for visible dynamics.</p>
              <p>The oscillation period is T = 2 pi / DeltaE in atomic units. Larger energy gaps yield faster beat motion.</p>
            </div>
            <div class="card">
              <div class="diagram-grid">
                <figure class="diagram active" data-diagram="beat">
                  <svg viewBox="0 0 260 180" aria-label="beat pattern">
                    <line x1="20" y1="90" x2="240" y2="90" stroke="#2f3948" stroke-width="2"></line>
                    <path d="M20 90 C50 30, 90 150, 120 90 C150 30, 190 150, 220 90" fill="none" stroke="#4aa3ff" stroke-width="3"></path>
                    <path d="M20 90 C50 50, 90 130, 120 90 C150 50, 190 130, 220 90" fill="none" stroke="#f7b059" stroke-width="2" opacity="0.7"></path>
                  </svg>
                  <figcaption>Two close frequencies produce a beat pattern. The superposition density oscillates with DeltaE.</figcaption>
                </figure>
              </div>
            </div>
          </section>

          <section id="glossary" class="info-section">
            <div class="card">
              <h3>UI glossary</h3>
              <table>
                <thead>
                  <tr><th>Control</th><th>Meaning</th></tr>
                </thead>
                <tbody>
                  <tr><td>Render</td><td>Dots shows samples, Bubbles shows an isosurface.</td></tr>
                  <tr><td>Mode</td><td>Total density, valence density, single orbital, or superposition.</td></tr>
                  <tr><td>Basis</td><td>Complex (m) or real chemistry combinations for orbital lobes.</td></tr>
                  <tr><td>n, l, m</td><td>Quantum numbers that define a single orbital.</td></tr>
                  <tr><td>cnt</td><td>Number of Monte Carlo samples to draw.</td></tr>
                  <tr><td>max</td><td>Maximum radius for sampling. Larger values show more diffuse tails.</td></tr>
                  <tr><td>mix</td><td>Superposition weight between orbital A and B.</td></tr>
                  <tr><td>Animated</td><td>Enables time dependent evolution in superposition mode.</td></tr>
                  <tr><td>Speed</td><td>Scales the animation time variable.</td></tr>
                  <tr><td>Threshold</td><td>Bubble isosurface level as a fraction of peak density.</td></tr>
                  <tr><td>Quality</td><td>Bubble grid resolution and sample count preset.</td></tr>
                </tbody>
              </table>
            </div>
          </section>

          <section id="limits" class="info-section">
            <div class="card">
              <h3>Limitations and interpretation</h3>
              <ul>
                <li>This is not a full time dependent many body solver. Superpositions are built from analytical hydrogenic states.</li>
                <li>LDA orbitals are radial averages and do not include explicit electron correlation effects.</li>
                <li>Dots show Monte Carlo samples, so low counts will look noisy.</li>
                <li>Bubbles show an isosurface, which depends on the chosen threshold.</li>
                <li>Spin, spin orbit coupling, and relativistic corrections are not modeled.</li>
                <li>Excited state lifetimes and transitions are not simulated.</li>
              </ul>
              <p>Despite these limitations, the visualizer is physically grounded and useful for exploring orbital geometry, nodal structure, and interference effects.</p>
            </div>
          </section>
        </div>
      </main>
    </div>
    <script>
      const navItems = Array.from(document.querySelectorAll(".nav-item"));
      const sections = Array.from(document.querySelectorAll(".info-section"));
      const sectionTitle = document.getElementById("sectionTitle");
      const navSearch = document.getElementById("navSearch");

      function showSection(target) {
        navItems.forEach((item) => item.classList.toggle("active", item.dataset.target === target));
        sections.forEach((section) => section.classList.toggle("active", section.id === target));
        const activeItem = navItems.find((item) => item.dataset.target === target);
        sectionTitle.textContent = activeItem ? activeItem.dataset.title : "Overview";
      }

      navItems.forEach((item) => {
        item.addEventListener("click", () => showSection(item.dataset.target));
      });

      navSearch.addEventListener("input", (event) => {
        const query = event.target.value.trim().toLowerCase();
        navItems.forEach((item) => {
          const text = item.textContent.toLowerCase();
          item.style.display = text.includes(query) ? "block" : "none";
        });
      });

      const diagramTabs = Array.from(document.querySelectorAll('[data-diagram-group=\"orbitals\"][data-diagram-tab]'));
      const diagrams = Array.from(document.querySelectorAll('[data-diagram-group=\"orbitals\"][data-diagram]'));
      diagramTabs.forEach((tab) => {
        tab.addEventListener("click", () => {
          const key = tab.dataset.diagramTab;
          diagramTabs.forEach((btn) => btn.classList.toggle("active", btn === tab));
          diagrams.forEach((diagram) => diagram.classList.toggle("active", diagram.dataset.diagram === key));
        });
      });
    </script>
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
    let basis = AngularBasis::from_query(q.basis.as_deref());
    let want_super_psi =
        q.animated.unwrap_or(false) && requested_mode == ViewMode::Superposition;
    let want_phase = matches!(q.color_mode.as_deref(), Some("phase"));
    let want_intensity = matches!(q.color_mode.as_deref(), Some("intensity"));
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
                                phases: None,
                                intensities: None,
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
                                        basis,
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
                                phases: None,
                                intensities: None,
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
                                    basis,
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
                                    basis,
                                ))
                            } else {
                                None
                            };
                            let phases = if want_phase {
                                Some(phases_from_radial_samples(
                                    &samples,
                                    &radial_r_sign,
                                    &radial_val_sign,
                                    l_used,
                                    m_used,
                                    RadialKind::R,
                                    basis,
                                ))
                            } else {
                                None
                            };
                            let intensities = if want_intensity {
                                Some(intensities_from_radial_samples(
                                    &samples,
                                    &radial_r_sign,
                                    &radial_val_sign,
                                    l_used,
                                    m_used,
                                    RadialKind::R,
                                    basis,
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
                                phases,
                                intensities,
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
                                    basis,
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
                                    basis,
                                ))
                            } else {
                                None
                            };
                            let phases = if want_phase {
                                Some(phases_from_superposition_lda(
                                    &samples,
                                    &orb_a,
                                    &orb_b,
                                    m_a,
                                    m_b,
                                    mix,
                                    time,
                                    delta_e,
                                    basis,
                                ))
                            } else {
                                None
                            };
                            let intensities = if want_intensity {
                                Some(intensities_from_superposition_lda(
                                    &samples,
                                    &orb_a,
                                    &orb_b,
                                    m_a,
                                    m_b,
                                    mix,
                                    time,
                                    delta_e,
                                    basis,
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
                                phases,
                                intensities,
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
                            basis,
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
                            basis,
                        ))
                    } else {
                        None
                    };
                    let phases = if want_phase {
                        Some(phases_from_radial_samples(
                            &samples,
                            &radial_r_sign,
                            &radial_val_sign,
                            l_used,
                            m_used,
                            RadialKind::Chi,
                            basis,
                        ))
                    } else {
                        None
                    };
                    let intensities = if want_intensity {
                        Some(intensities_from_radial_samples(
                            &samples,
                            &radial_r_sign,
                            &radial_val_sign,
                            l_used,
                            m_used,
                            RadialKind::Chi,
                            basis,
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
                        phases,
                        intensities,
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
                    phases: None,
                    intensities: None,
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
                    basis,
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
                    basis,
                ))
            } else {
                None
            };
            let phases = if want_phase {
                Some(phases_from_superposition_hydrogenic(
                    &samples,
                    q1,
                    q2,
                    mix,
                    time,
                    delta_e,
                    basis,
                ))
            } else {
                None
            };
            let intensities = if want_intensity {
                Some(intensities_from_superposition_hydrogenic(
                    &samples,
                    q1,
                    q2,
                    mix,
                    time,
                    delta_e,
                    basis,
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
                phases,
                intensities,
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
                    phases: None,
                    intensities: None,
                };
            return Json(empty).into_response();
        }
    };

    let raw = tokio::task::spawn_blocking(move || match basis {
        AngularBasis::Complex => generate_orbital_samples(qn, count, max_radius),
        AngularBasis::Real => generate_orbital_samples_basis(qn, count, max_radius, basis),
    })
    .await
    .unwrap_or_default();
    let signs = if bubble {
        Some(signs_from_hydrogenic_samples(
            &raw.iter().map(|(x, y, z)| [*x, *y, *z]).collect::<Vec<_>>(),
            qn,
            basis,
        ))
    } else {
        None
    };
    let phases = if want_phase {
        Some(phases_from_hydrogenic_samples(
            &raw.iter().map(|(x, y, z)| [*x, *y, *z]).collect::<Vec<_>>(),
            qn,
            basis,
        ))
    } else {
        None
    };
    let intensities = if want_intensity {
        Some(intensities_from_hydrogenic_samples(
            &raw.iter().map(|(x, y, z)| [*x, *y, *z]).collect::<Vec<_>>(),
            qn,
            basis,
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
        phases,
        intensities,
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
    basis: AngularBasis,
) -> Vec<[f32; 3]> {
    use rand::Rng;
    use std::f32::consts::PI;

    let mut samples = Vec::with_capacity(num_samples);
    let mut rng = rand::thread_rng();

    let cdf = build_radial_cdf(radial_r, radial_val, max_radius, radial_kind);
    let max_ang = max_angular_prob(l, m_l, basis);
    let mut attempts = 0usize;
    let max_attempts = num_samples.saturating_mul(300).max(1000);

    while samples.len() < num_samples && attempts < max_attempts {
        attempts += 1;
        let r = sample_r(&cdf, radial_r, &mut rng);
        let phi = rng.gen::<f32>() * 2.0 * PI;

        // Rejection sample theta from |Y_lm|^2 with a bounded loop
        let mut accepted = false;
        for _ in 0..256 {
            let cos_theta = rng.gen::<f32>() * 2.0 - 1.0;
            let theta = cos_theta.acos();
            let ang = angular_wavefunction_basis(theta, phi, l, m_l, basis);
            if !ang.is_finite() {
                continue;
            }
            let p = (ang * ang) / max_ang;
            if rng.gen::<f32>() < p.min(1.0) {
                let x = r * theta.sin() * phi.cos();
                let y = r * theta.sin() * phi.sin();
                let z = r * theta.cos();
                samples.push([x, y, z]);
                accepted = true;
                break;
            }
        }
        if !accepted {
            continue;
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
    basis: AngularBasis,
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
    let max_ang_a = max_angular_prob(orb_a.l, m_a, basis);
    let max_ang_b = max_angular_prob(orb_b.l, m_b, basis);
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
                let ang = angular_wavefunction_basis(theta, phi, orb_a.l, m_a, basis);
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
                let ang = angular_wavefunction_basis(theta, phi, orb_b.l, m_b, basis);
                if rng.gen::<f32>() < (ang * ang) / max_ang_b {
                    break theta;
                }
            };
            (r, theta, phi)
        };

        let r1 = interp_radial(r, &orb_a.radial_r, &orb_a.radial_rfn);
        let r2 = interp_radial(r, &orb_b.radial_r, &orb_b.radial_rfn);

        let (y1_re, y1_im) = spherical_harmonic_basis(theta, phi, orb_a.l, m_a, basis);
        let (y2_re, y2_im) = spherical_harmonic_basis(theta, phi, orb_b.l, m_b, basis);

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
    basis: AngularBasis,
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
    let max_ang_a = max_angular_prob(qn_a.l, qn_a.m_l, basis);
    let max_ang_b = max_angular_prob(qn_b.l, qn_b.m_l, basis);
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
                let ang = angular_wavefunction_basis(theta, phi, qn_a.l, qn_a.m_l, basis);
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
                let ang = angular_wavefunction_basis(theta, phi, qn_b.l, qn_b.m_l, basis);
                if rng.gen::<f32>() < (ang * ang) / max_ang_b {
                    break theta;
                }
            };
            (r, theta, phi)
        };

        let r1 = interp_radial(r, &rs, &rfn_a);
        let r2 = interp_radial(r, &rs, &rfn_b);
        let (y1_re, y1_im) = spherical_harmonic_basis(theta, phi, qn_a.l, qn_a.m_l, basis);
        let (y2_re, y2_im) = spherical_harmonic_basis(theta, phi, qn_b.l, qn_b.m_l, basis);

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
    basis: AngularBasis,
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
            basis,
        );
        samples.append(&mut part);
    }

    samples
}

fn spherical_harmonic_basis(
    theta: f32,
    phi: f32,
    l: u32,
    m_l: i32,
    basis: AngularBasis,
) -> (f32, f32) {
    match basis {
        AngularBasis::Complex => spherical_harmonic(theta, phi, l, m_l),
        AngularBasis::Real => (real_spherical_harmonic(theta, phi, l, m_l), 0.0),
    }
}

fn sign_from_value(v: f32) -> i8 {
    if v >= 0.0 {
        1
    } else {
        -1
    }
}

fn phase_from_components(re: f32, im: f32) -> f32 {
    if re.abs() + im.abs() < 1e-12 {
        0.0
    } else {
        im.atan2(re)
    }
}

fn intensity_from_components(re: f32, im: f32) -> f32 {
    re * re + im * im
}

fn signs_from_radial_samples(
    samples: &[[f32; 3]],
    radial_r: &[f32],
    radial_val: &[f32],
    l: u32,
    m_l: i32,
    radial_kind: RadialKind,
    basis: AngularBasis,
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
        let (y_re, _) = spherical_harmonic_basis(theta, phi, l, m_l, basis);
        let psi_re = radial * y_re;
        out.push(sign_from_value(psi_re));
    }
    out
}

fn phases_from_radial_samples(
    samples: &[[f32; 3]],
    radial_r: &[f32],
    radial_val: &[f32],
    l: u32,
    m_l: i32,
    radial_kind: RadialKind,
    basis: AngularBasis,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(samples.len());
    for p in samples {
        let x = p[0];
        let y = p[1];
        let z = p[2];
        let r = (x * x + y * y + z * z).sqrt();
        if r <= 1e-8 {
            out.push(0.0);
            continue;
        }
        let cos_theta = (z / r).clamp(-1.0, 1.0);
        let theta = cos_theta.acos();
        let phi = y.atan2(x);
        let mut radial = interp_radial(r, radial_r, radial_val);
        if matches!(radial_kind, RadialKind::Chi) && r > 1e-8 {
            radial /= r;
        }
        let (y_re, y_im) = spherical_harmonic_basis(theta, phi, l, m_l, basis);
        let psi_re = radial * y_re;
        let psi_im = radial * y_im;
        out.push(phase_from_components(psi_re, psi_im));
    }
    out
}

fn intensities_from_radial_samples(
    samples: &[[f32; 3]],
    radial_r: &[f32],
    radial_val: &[f32],
    l: u32,
    m_l: i32,
    radial_kind: RadialKind,
    basis: AngularBasis,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(samples.len());
    for p in samples {
        let x = p[0];
        let y = p[1];
        let z = p[2];
        let r = (x * x + y * y + z * z).sqrt();
        if r <= 1e-8 {
            out.push(0.0);
            continue;
        }
        let cos_theta = (z / r).clamp(-1.0, 1.0);
        let theta = cos_theta.acos();
        let phi = y.atan2(x);
        let mut radial = interp_radial(r, radial_r, radial_val);
        if matches!(radial_kind, RadialKind::Chi) && r > 1e-8 {
            radial /= r;
        }
        let (y_re, y_im) = spherical_harmonic_basis(theta, phi, l, m_l, basis);
        let psi_re = radial * y_re;
        let psi_im = radial * y_im;
        out.push(intensity_from_components(psi_re, psi_im));
    }
    out
}

fn signs_from_hydrogenic_samples(
    samples: &[[f32; 3]],
    qn: QuantumNumbers,
    basis: AngularBasis,
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
        let radial = radial_wavefunction(r, qn.n, qn.l);
        let (y_re, _) = spherical_harmonic_basis(theta, phi, qn.l, qn.m_l, basis);
        let psi_re = radial * y_re;
        out.push(sign_from_value(psi_re));
    }
    out
}

fn phases_from_hydrogenic_samples(
    samples: &[[f32; 3]],
    qn: QuantumNumbers,
    basis: AngularBasis,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(samples.len());
    for p in samples {
        let x = p[0];
        let y = p[1];
        let z = p[2];
        let r = (x * x + y * y + z * z).sqrt();
        if r <= 1e-8 {
            out.push(0.0);
            continue;
        }
        let cos_theta = (z / r).clamp(-1.0, 1.0);
        let theta = cos_theta.acos();
        let phi = y.atan2(x);
        let radial = radial_wavefunction(r, qn.n, qn.l);
        let (y_re, y_im) = spherical_harmonic_basis(theta, phi, qn.l, qn.m_l, basis);
        let psi_re = radial * y_re;
        let psi_im = radial * y_im;
        out.push(phase_from_components(psi_re, psi_im));
    }
    out
}

fn intensities_from_hydrogenic_samples(
    samples: &[[f32; 3]],
    qn: QuantumNumbers,
    basis: AngularBasis,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(samples.len());
    for p in samples {
        let x = p[0];
        let y = p[1];
        let z = p[2];
        let r = (x * x + y * y + z * z).sqrt();
        if r <= 1e-8 {
            out.push(0.0);
            continue;
        }
        let cos_theta = (z / r).clamp(-1.0, 1.0);
        let theta = cos_theta.acos();
        let phi = y.atan2(x);
        let radial = radial_wavefunction(r, qn.n, qn.l);
        let (y_re, y_im) = spherical_harmonic_basis(theta, phi, qn.l, qn.m_l, basis);
        let psi_re = radial * y_re;
        let psi_im = radial * y_im;
        out.push(intensity_from_components(psi_re, psi_im));
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
    basis: AngularBasis,
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
        let (y1_re, _) = spherical_harmonic_basis(theta, phi, q1.l, q1.m_l, basis);
        let (y2_re, y2_im) = spherical_harmonic_basis(theta, phi, q2.l, q2.m_l, basis);
        let psi1_re = a * r1 * y1_re;
        let psi2_re = b * r2 * (y2_re * phase_re - y2_im * phase_im);
        out.push(sign_from_value(psi1_re + psi2_re));
    }
    out
}

fn phases_from_superposition_hydrogenic(
    samples: &[[f32; 3]],
    q1: QuantumNumbers,
    q2: QuantumNumbers,
    mix: f32,
    time: f32,
    delta_e: f32,
    basis: AngularBasis,
) -> Vec<f32> {
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
            out.push(0.0);
            continue;
        }
        let cos_theta = (z / r).clamp(-1.0, 1.0);
        let theta = cos_theta.acos();
        let phi = y.atan2(x);
        let r1 = radial_wavefunction(r, q1.n, q1.l);
        let r2 = radial_wavefunction(r, q2.n, q2.l);
        let (y1_re, y1_im) = spherical_harmonic_basis(theta, phi, q1.l, q1.m_l, basis);
        let (y2_re, y2_im) = spherical_harmonic_basis(theta, phi, q2.l, q2.m_l, basis);
        let psi1_re = a * r1 * y1_re;
        let psi1_im = a * r1 * y1_im;
        let psi2_re = b * r2 * (y2_re * phase_re - y2_im * phase_im);
        let psi2_im = b * r2 * (y2_re * phase_im + y2_im * phase_re);
        out.push(phase_from_components(psi1_re + psi2_re, psi1_im + psi2_im));
    }
    out
}

fn intensities_from_superposition_hydrogenic(
    samples: &[[f32; 3]],
    q1: QuantumNumbers,
    q2: QuantumNumbers,
    mix: f32,
    time: f32,
    delta_e: f32,
    basis: AngularBasis,
) -> Vec<f32> {
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
            out.push(0.0);
            continue;
        }
        let cos_theta = (z / r).clamp(-1.0, 1.0);
        let theta = cos_theta.acos();
        let phi = y.atan2(x);
        let r1 = radial_wavefunction(r, q1.n, q1.l);
        let r2 = radial_wavefunction(r, q2.n, q2.l);
        let (y1_re, y1_im) = spherical_harmonic_basis(theta, phi, q1.l, q1.m_l, basis);
        let (y2_re, y2_im) = spherical_harmonic_basis(theta, phi, q2.l, q2.m_l, basis);
        let psi1_re = a * r1 * y1_re;
        let psi1_im = a * r1 * y1_im;
        let psi2_re = b * r2 * (y2_re * phase_re - y2_im * phase_im);
        let psi2_im = b * r2 * (y2_re * phase_im + y2_im * phase_re);
        out.push(intensity_from_components(psi1_re + psi2_re, psi1_im + psi2_im));
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
    basis: AngularBasis,
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
        let (y1_re, _) = spherical_harmonic_basis(theta, phi, orb_a.l, m_a, basis);
        let (y2_re, y2_im) = spherical_harmonic_basis(theta, phi, orb_b.l, m_b, basis);
        let psi1_re = a * r1 * y1_re;
        let psi2_re = b * r2 * (y2_re * phase_re - y2_im * phase_im);
        out.push(sign_from_value(psi1_re + psi2_re));
    }
    out
}

fn phases_from_superposition_lda(
    samples: &[[f32; 3]],
    orb_a: &LdaOrbital,
    orb_b: &LdaOrbital,
    m_a: i32,
    m_b: i32,
    mix: f32,
    time: f32,
    delta_e: f32,
    basis: AngularBasis,
) -> Vec<f32> {
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
            out.push(0.0);
            continue;
        }
        let cos_theta = (z / r).clamp(-1.0, 1.0);
        let theta = cos_theta.acos();
        let phi = y.atan2(x);
        let r1 = interp_radial(r, &orb_a.radial_r, &orb_a.radial_rfn);
        let r2 = interp_radial(r, &orb_b.radial_r, &orb_b.radial_rfn);
        let (y1_re, y1_im) = spherical_harmonic_basis(theta, phi, orb_a.l, m_a, basis);
        let (y2_re, y2_im) = spherical_harmonic_basis(theta, phi, orb_b.l, m_b, basis);
        let psi1_re = a * r1 * y1_re;
        let psi1_im = a * r1 * y1_im;
        let psi2_re = b * r2 * (y2_re * phase_re - y2_im * phase_im);
        let psi2_im = b * r2 * (y2_re * phase_im + y2_im * phase_re);
        out.push(phase_from_components(psi1_re + psi2_re, psi1_im + psi2_im));
    }
    out
}

fn intensities_from_superposition_lda(
    samples: &[[f32; 3]],
    orb_a: &LdaOrbital,
    orb_b: &LdaOrbital,
    m_a: i32,
    m_b: i32,
    mix: f32,
    time: f32,
    delta_e: f32,
    basis: AngularBasis,
) -> Vec<f32> {
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
            out.push(0.0);
            continue;
        }
        let cos_theta = (z / r).clamp(-1.0, 1.0);
        let theta = cos_theta.acos();
        let phi = y.atan2(x);
        let r1 = interp_radial(r, &orb_a.radial_r, &orb_a.radial_rfn);
        let r2 = interp_radial(r, &orb_b.radial_r, &orb_b.radial_rfn);
        let (y1_re, y1_im) = spherical_harmonic_basis(theta, phi, orb_a.l, m_a, basis);
        let (y2_re, y2_im) = spherical_harmonic_basis(theta, phi, orb_b.l, m_b, basis);
        let psi1_re = a * r1 * y1_re;
        let psi1_im = a * r1 * y1_im;
        let psi2_re = b * r2 * (y2_re * phase_re - y2_im * phase_im);
        let psi2_im = b * r2 * (y2_re * phase_im + y2_im * phase_re);
        out.push(intensity_from_components(psi1_re + psi2_re, psi1_im + psi2_im));
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

fn max_angular_prob(l: u32, m_l: i32, basis: AngularBasis) -> f32 {
    use std::f32::consts::PI;
    let mut max_val = 0.0_f32;
    let theta_steps = 180;
    let phi_steps = if matches!(basis, AngularBasis::Complex) { 1 } else { 72 };
    for i in 0..theta_steps {
        let theta = (i as f32 + 0.5) / theta_steps as f32 * PI;
        for j in 0..phi_steps {
            let phi = (j as f32 + 0.5) / phi_steps as f32 * 2.0 * PI;
            let ang = angular_wavefunction_basis(theta, phi, l, m_l, basis);
            let p = ang * ang;
            if p.is_finite() && p > max_val {
                max_val = p;
            }
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


