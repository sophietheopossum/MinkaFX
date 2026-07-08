// Rounded-rect SDF snap preview. One fullscreen triangle; the fragment
// stage draws a translucent fill with a solid border ring, anti-aliased,
// premultiplied alpha (surface composite mode is PreMultiplied).

struct Uniforms {
    rect: vec4<f32>,    // x, y, w, h in physical pixels
    border: vec4<f32>,  // rgba, straight alpha
    fill: vec4<f32>,    // rgba, straight alpha
    params: vec4<f32>,  // radius, global alpha, border width, unused
};

@group(0) @binding(0) var<uniform> u: Uniforms;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(3.0, 1.0),
        vec2<f32>(-1.0, 1.0),
    );
    return vec4<f32>(positions[index], 0.0, 1.0);
}

fn sd_round_rect(p: vec2<f32>, half_size: vec2<f32>, radius: f32) -> f32 {
    let q = abs(p) - half_size + vec2<f32>(radius);
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - radius;
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let center = u.rect.xy + u.rect.zw * 0.5;
    let half_size = max(u.rect.zw * 0.5, vec2<f32>(0.0));
    let d = sd_round_rect(frag.xy - center, half_size, u.params.x);

    let aa = 1.0;
    // 1 inside the rect, 0 outside (soft edge).
    let inside = 1.0 - smoothstep(-aa, aa, d);
    // 1 deeper than the border band.
    let inner = 1.0 - smoothstep(-u.params.z - aa, -u.params.z + aa, d);
    let ring = max(inside - inner, 0.0);

    let alpha = (u.fill.a * inner + u.border.a * ring) * u.params.y;
    let rgb = (u.fill.rgb * u.fill.a * inner + u.border.rgb * u.border.a * ring) * u.params.y;
    return vec4<f32>(rgb, alpha);
}
