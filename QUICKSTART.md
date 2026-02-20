# Quick Start

## Run The Web App

```bash
cargo run --bin web
```

Then open `http://127.0.0.1:3000`.

## Info Page

Open `http://127.0.0.1:3000/info` for full documentation of the physics, UI terms, and visualization behavior.

## Controls Summary

- Drag: orbit the camera
- Scroll: zoom
- WASD: move the camera target (bounded)
- Reset camera: returns to default view

## Render Mode

Use the Render switch at the top to select:

- Dots: Monte Carlo samples
- Bubbles: smooth isosurfaces, red/blue for phase sign when defined

## Notes

If superposition looks static, make sure the two states have different n values. States with the same n are degenerate and do not evolve in time in the hydrogenic model.
