mod physics;
mod graphics;

use graphics::{Graphics, Vertex};
use physics::{QuantumNumbers, generate_orbital_samples};
use winit::{
    event::{Event, WindowEvent, ElementState},
    event_loop::EventLoop,
    window::WindowBuilder,
};
use std::sync::Arc;

struct AppState {
    quantum_n: u32,
    quantum_l: u32,
    quantum_m: i32,
    num_particles: usize,
    max_radius: f32,
    rotation_x: f32,
    rotation_y: f32,
    samples: Vec<(f32, f32, f32)>, // cached raw (unrotated) samples
    samples_dirty: bool,           // true when re-sampling is needed
}

impl AppState {
    fn new() -> Self {
        AppState {
            quantum_n: 2,
            quantum_l: 1,
            quantum_m: 0,
            num_particles: 50000,
            max_radius: 20.0,
            rotation_x: 0.0,
            rotation_y: 0.0,
            samples: Vec::new(),
            samples_dirty: true, // trigger generation on first render
        }
    }

    fn generate_vertices(&mut self) -> Vec<Vertex> {
        // Re-sample only when orbital or particle count changed
        if self.samples_dirty || self.samples.is_empty() {
            let qn = match QuantumNumbers::new(self.quantum_n, self.quantum_l, self.quantum_m) {
                Some(qn) => qn,
                None => return vec![],
            };
            println!("Generating orbital ({}, {}, {}) with {} particles...",
                     self.quantum_n, self.quantum_l, self.quantum_m, self.num_particles);
            self.samples = generate_orbital_samples(qn, self.num_particles, self.max_radius);
            self.samples_dirty = false;
        }

        // Re-apply rotation to cached samples every frame (fast: no physics recomputation)
        let mut vertices = Vec::with_capacity(self.samples.len());
        for &(x, y, z) in &self.samples {
            // Scale down for visualization
            let scale = 0.1;
            let x = x * scale;
            let y = y * scale;
            let z = z * scale;

            // Calculate distance from origin for color mapping
            let dist = (x * x + y * y + z * z).sqrt();
            let max_dist = self.max_radius * scale;
            let normalized_dist = (dist / max_dist).min(1.0);

            // Color gradient: blue (near nucleus) → cyan → green → yellow → red
            let color = if normalized_dist < 0.25 {
                let t = normalized_dist / 0.25;
                [0.0, t, 1.0]
            } else if normalized_dist < 0.5 {
                let t = (normalized_dist - 0.25) / 0.25;
                [0.0, 1.0, 1.0 - t]
            } else if normalized_dist < 0.75 {
                let t = (normalized_dist - 0.5) / 0.25;
                [t, 1.0, 0.0]
            } else {
                let t = (normalized_dist - 0.75) / 0.25;
                [1.0, 1.0 - t, 0.0]
            };

            let (x_rot, y_rot, z_rot) = rotate_point(x, y, z, self.rotation_x, self.rotation_y);
            vertices.push(Vertex {
                position: [x_rot, y_rot, z_rot],
                color,
            });
        }
        vertices
    }
}

fn rotate_point(x: f32, y: f32, z: f32, rot_x: f32, rot_y: f32) -> (f32, f32, f32) {
    // Rotate around X axis
    let cos_x = rot_x.cos();
    let sin_x = rot_x.sin();
    let y1 = y * cos_x - z * sin_x;
    let z1 = y * sin_x + z * cos_x;

    // Rotate around Y axis
    let cos_y = rot_y.cos();
    let sin_y = rot_y.sin();
    let x2 = x * cos_y + z1 * sin_y;
    let z2 = -x * sin_y + z1 * cos_y;

    (x2, y1, z2)
}

#[tokio::main]
async fn main() {
    println!("Hydrogen Quantum Orbital Visualizer - Rust");
    println!("==========================================");

    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .with_title("Quantum Orbitals 3D")
        .with_inner_size(winit::dpi::LogicalSize::new(1200.0, 800.0))
        .build(&event_loop)
        .unwrap();

    let window = Arc::new(window);
    let mut graphics = Graphics::new(window.clone()).await;
    let mut app_state = AppState::new();

    // Generate initial orbital
    let vertices = app_state.generate_vertices();
    graphics.update_vertices(&vertices);

    let mut last_render = std::time::Instant::now();

    event_loop
        .run(move |event, target| {
            match event {
                Event::WindowEvent {
                    ref event,
                    window_id,
                } if window_id == window.id() => {
                    match event {
                        WindowEvent::CloseRequested => {
                            target.exit();
                        }
                        WindowEvent::Resized(physical_size) => {
                            graphics.resize(*physical_size);
                        }
                        WindowEvent::KeyboardInput {
                            event,
                            ..
                        } => {
                            if event.state == ElementState::Pressed {
                                match event.logical_key.as_ref() {
                                    winit::keyboard::Key::Character(c) => {
                                        let c_str = c.to_string();
                                        match c_str.as_str() {
                                            "1" => {
                                                app_state.quantum_n = 1;
                                                app_state.quantum_l = 0;
                                                app_state.quantum_m = 0;
                                                app_state.samples_dirty = true;
                                                println!("Set orbital to 1s");
                                            }
                                            "2" => {
                                                app_state.quantum_n = 2;
                                                app_state.quantum_l = 0;
                                                app_state.quantum_m = 0;
                                                app_state.samples_dirty = true;
                                                println!("Set orbital to 2s");
                                            }
                                            "3" => {
                                                app_state.quantum_n = 2;
                                                app_state.quantum_l = 1;
                                                app_state.quantum_m = 0;
                                                app_state.samples_dirty = true;
                                                println!("Set orbital to 2p (m=0)");
                                            }
                                            "4" => {
                                                app_state.quantum_n = 3;
                                                app_state.quantum_l = 2;
                                                app_state.quantum_m = 0;
                                                app_state.samples_dirty = true;
                                                println!("Set orbital to 3d (m=0)");
                                            }
                                            "5" => {
                                                app_state.quantum_n = 4;
                                                app_state.quantum_l = 3;
                                                app_state.quantum_m = 0;
                                                app_state.samples_dirty = true;
                                                println!("Set orbital to 4f (m=0)");
                                            }
                                            "+" | "=" => {
                                                app_state.num_particles = (app_state.num_particles as f32 * 1.5) as usize;
                                                app_state.samples_dirty = true;
                                            }
                                            "-" => {
                                                app_state.num_particles = (app_state.num_particles / 2).max(1000);
                                                app_state.samples_dirty = true;
                                            }
                                            "m" => {
                                                app_state.quantum_m = (app_state.quantum_m + 1).min(app_state.quantum_l as i32);
                                                app_state.samples_dirty = true;
                                                println!("m_l = {}", app_state.quantum_m);
                                            }
                                            "n" => {
                                                app_state.quantum_m = (app_state.quantum_m - 1).max(-(app_state.quantum_l as i32));
                                                app_state.samples_dirty = true;
                                                println!("m_l = {}", app_state.quantum_m);
                                            }
                                            _ => {}
                                        }
                                    }
                                    winit::keyboard::Key::Named(named_key) => {
                                        match named_key {
                                            winit::keyboard::NamedKey::ArrowLeft => {
                                                app_state.rotation_y -= 0.1;
                                            }
                                            winit::keyboard::NamedKey::ArrowRight => {
                                                app_state.rotation_y += 0.1;
                                            }
                                            winit::keyboard::NamedKey::ArrowUp => {
                                                app_state.rotation_x -= 0.1;
                                            }
                                            winit::keyboard::NamedKey::ArrowDown => {
                                                app_state.rotation_x += 0.1;
                                            }
                                            _ => {}
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                        _ => {}
                    }
                }
                Event::AboutToWait => {
                    let now = std::time::Instant::now();
                    if now.duration_since(last_render).as_millis() > 16 {
                        // 60 FPS
                        window.request_redraw();
                        last_render = now;
                    }
                }
                Event::WindowEvent {
                    event: WindowEvent::RedrawRequested,
                    window_id,
                } if window_id == window.id() => {
                    let vertices = app_state.generate_vertices();
                    graphics.update_vertices(&vertices);

                    if let Err(e) = graphics.render() {
                        eprintln!("Render error: {:?}", e);
                    }
                }
                _ => {}
            }
        })
        .unwrap();
}
