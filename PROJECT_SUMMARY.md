# Project Summary

Quantum Orbitals 3D is a web-based visualizer for electron probability densities. It renders point clouds sampled from hydrogenic and LDA-based orbital models and includes superposition animation.

## Current Capabilities

- Hydrogenic orbitals for any valid (n, l, m)
- Multi-electron LDA densities (total and valence)
- Single-orbital mode using LDA or PSLibrary when available
- Superposition mode with time evolution
- Dots or bubbles rendering (smooth surfaces)
- 3D navigation with orbit, zoom, and WASD movement
- Local web UI at `http://127.0.0.1:3000`

## Data Sources

- OpenMX LDA radial wavefunctions (downloaded on demand)
- PSLibrary radial data for single-orbital fallback
- Hydrogenic formulas for H and for superposition fallback

## Visualization Model

Each dot is a Monte Carlo sample from |psi|^2. Colors encode radial distance. Bubble mode reconstructs smooth isosurfaces and can show positive and negative phase in red and blue. Superposition animation shows evolving densities; dots are not particle trajectories.

## Status

Active development. The primary entry point is `cargo run --bin web`.
