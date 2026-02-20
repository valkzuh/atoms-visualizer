# Quantum Orbitals 3D

A 3D orbital visualizer served as a local web app. It renders electron probability distributions as point clouds and supports hydrogenic orbitals, LDA-based multi-electron densities, and superposition animation.

## Quick Start

```bash
cargo run --bin web
```

Open `http://127.0.0.1:3000` in your browser.

For full documentation, open `http://127.0.0.1:3000/info`.

## Features

- Hydrogenic orbitals for any valid (n, l, m)
- Multi-electron densities from OpenMX LDA radial data
- Valence density views (spherical or lobe projection)
- Single-orbital view using LDA or PSLibrary when available
- Superposition mode with time evolution
- Dots or bubbles rendering (smooth isosurfaces)
- Real orbital basis option for classic p/d/f lobe shapes (Bubbles mode)
- Dot color toggle for radial or phase visualization
- 3D orbit controls, WASD translation, and zoom

## Controls

- Drag: orbit the camera
- Scroll wheel: zoom in/out
- WASD: move the camera target within bounds
- Reset camera button: return to default view

## What The Dots Mean

Each dot is a Monte Carlo sample drawn from the probability density. Dots are not electron trajectories. Dense regions represent higher probability.

## Bubble Mode And Sign

Bubble mode reconstructs a smooth surface from the sampled density. When a phase sign is defined (single orbital and superposition), positive regions are shown in red and negative regions in blue. Density-only modes show a single surface.

## Color Meaning In Dots Mode

Color indicates radial distance from the nucleus. The gradient runs from blue near the core through cyan and green to yellow at larger radii.

## Data Sources And Approximations

- OpenMX LDA radial wavefunctions are used when available for elements.
- PSLibrary data is used for single-orbital view when LDA is not available.
- Hydrogenic formulas are used for H and as a fallback for superposition on any element (scaled by 1/Z).

The visualization is physically motivated but not a full many-body solver. For details, see the info page and `PHYSICS.md`.
