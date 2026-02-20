# Developer Guide

This document describes the project structure and how the web app works.

## Project Layout

- `src/bin/web.rs` - Axum server, HTML, and sampling endpoints
- `src/physics.rs` - Hydrogenic math helpers and spherical harmonics
- `src/atomic_lda.rs` - OpenMX LDA downloader and parser
- `src/atomic_data.rs` - PSLibrary radial data parser

## Running Locally

```bash
cargo run --bin web
```

Open `http://127.0.0.1:3000`.

## HTTP Endpoints

- `GET /` serves the main UI
- `GET /info` serves the documentation page
- `GET /samples` returns sampled point clouds

### /samples Query Parameters

- `mode`: total, valence, orbital, superposition
- `n, l, m`: quantum numbers for orbital A
- `n2, l2, m2`: quantum numbers for orbital B (superposition)
- `z`: atomic number
- `count`: number of sample points
- `max`: maximum radial extent
- `mix`: mixing fraction for superposition
- `animated`: hint to return animation-friendly data
- `bubble`: request sign data for bubbles mode
- `valence_style`: spherical or orbitals
- `basis`: complex or real (chemistry-style)

## Rendering Pipeline

The UI uses Three.js point rendering. Each sample point is scaled and colored based on radial distance. Superposition animation morphs between successive sample clouds.

Bubbles mode reconstructs a smooth surface using a Marching Cubes field fed by the sample points. When a phase sign is defined, positive and negative surfaces are shown in red and blue.

For orbital and superposition modes, the basis selector chooses between complex spherical harmonics (phi-symmetric) and real combinations that produce textbook p/d/f lobes.

## Data Sources

- OpenMX LDA: radial wavefunctions and occupancy for many elements
- PSLibrary: fallback for single-orbital mode
- Hydrogenic formulas: used for H and for superposition fallback on any Z

## Adding Or Updating Data

- LDA data is fetched from OpenMX on demand and stored under `data/openmx_lda/`.
- PSLibrary data can be extended by adding new UPF files to `data/`.

## Performance Notes

Sampling is CPU-intensive and runs in `spawn_blocking` to keep the server responsive. If animation feels heavy, reduce `count` or increase the morph interval in the UI.
