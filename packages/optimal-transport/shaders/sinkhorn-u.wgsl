// Sinkhorn u-update compute shader.
// Computes u_i = a_i / Σ_j exp(-C_ij / ε) * v_j

struct Params {
  N: u32,          // Number of sources (venues)
  M: u32,          // Number of targets (events)
  inv_epsilon: f32  // 1.0 / epsilon
}

@group(0) @binding(0) var<storage, read> C: array<f32>;       // [N*M] cost matrix
@group(0) @binding(1) var<storage, read> v: array<f32>;       // [M] current v vector
@group(0) @binding(2) var<storage, read> a: array<f32>;       // [N] source marginal
@group(0) @binding(3) var<storage, read_write> u: array<f32>; // [N] output u vector
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.N) { return; }

  // u_i = a_i / Σ_j exp(-C_ij / ε) * v_j
  var sum: f32 = 0.0;
  for (var j: u32 = 0u; j < params.M; j++) {
    sum += exp(-C[i * params.M + j] * params.inv_epsilon) * v[j];
  }
  u[i] = a[i] / max(sum, 1e-30);
}
