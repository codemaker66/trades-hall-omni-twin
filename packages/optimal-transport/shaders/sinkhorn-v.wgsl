// Sinkhorn v-update compute shader.
// Computes v_j = b_j / Σ_i exp(-C_ij / ε) * u_i

struct Params {
  N: u32,          // Number of sources (venues)
  M: u32,          // Number of targets (events)
  inv_epsilon: f32  // 1.0 / epsilon
}

@group(0) @binding(0) var<storage, read> C: array<f32>;       // [N*M] cost matrix
@group(0) @binding(1) var<storage, read> u: array<f32>;       // [N] current u vector
@group(0) @binding(2) var<storage, read> b: array<f32>;       // [M] target marginal
@group(0) @binding(3) var<storage, read_write> v: array<f32>; // [M] output v vector
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let j = gid.x;
  if (j >= params.M) { return; }

  // v_j = b_j / Σ_i exp(-C_ij / ε) * u_i
  var sum: f32 = 0.0;
  for (var i: u32 = 0u; i < params.N; i++) {
    sum += exp(-C[i * params.M + j] * params.inv_epsilon) * u[i];
  }
  v[j] = b[j] / max(sum, 1e-30);
}
