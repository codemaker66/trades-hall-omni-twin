// Recover transport plan from dual scaling vectors.
// T_ij = u_i * exp(-C_ij / ε) * v_j

struct Params {
  N: u32,          // Number of sources
  M: u32,          // Number of targets
  inv_epsilon: f32  // 1.0 / epsilon
}

@group(0) @binding(0) var<storage, read> C: array<f32>;       // [N*M] cost matrix
@group(0) @binding(1) var<storage, read> u: array<f32>;       // [N] u vector
@group(0) @binding(2) var<storage, read> v: array<f32>;       // [M] v vector
@group(0) @binding(3) var<storage, read_write> T: array<f32>; // [N*M] output plan
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let total = params.N * params.M;
  if (idx >= total) { return; }

  let i = idx / params.M;
  let j = idx % params.M;

  T[idx] = u[i] * exp(-C[idx] * params.inv_epsilon) * v[j];
}
