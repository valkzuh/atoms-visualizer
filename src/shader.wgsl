// Standard vertex shader for point rendering with camera transformation

struct CameraUniform {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) quad_pos: vec2<f32>,
    @location(1) position: vec3<f32>,
    @location(2) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    let point_size = 0.03;
    let world_pos = vec3<f32>(
        model.position.x + model.quad_pos.x * point_size,
        model.position.y + model.quad_pos.y * point_size,
        model.position.z
    );
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.color = model.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Semi-transparent quads so dense regions appear brighter.
    return vec4<f32>(in.color, 0.2);
}
