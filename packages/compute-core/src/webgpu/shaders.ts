// ---------------------------------------------------------------------------
// @omni-twin/compute-core — HPC-1: WebGPU Compute Shader Infrastructure
// ---------------------------------------------------------------------------
// WGSL shader source strings for GPU compute pipelines.
// Pure constants — no external dependencies.
// ---------------------------------------------------------------------------

/**
 * Parallel reduction shader (workgroup size 256).
 *
 * Performs a tree-based reduction in shared workgroup memory. Each invocation
 * loads one element, then participates in log2(256) = 8 reduction steps
 * separated by workgroupBarrier() calls.
 *
 * Bindings:
 *   @group(0) @binding(0) — input:  array<f32> (read-only storage)
 *   @group(0) @binding(1) — output: array<f32> (storage, one element per workgroup)
 *
 * Uniforms:
 *   @group(0) @binding(2) — params: struct { count: u32 }
 */
export const REDUCE_WGSL: string = /* wgsl */ `
struct Params {
  count: u32,
}

@group(0) @binding(0) var<storage, read>       input:  array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform>             params: Params;

const WG_SIZE: u32 = 256u;

var<workgroup> shared: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id)  lid: vec3<u32>,
  @builtin(workgroup_id)         wid: vec3<u32>,
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  let local_idx = lid.x;
  let global_idx = gid.x;

  // Load from global memory, zero-pad out-of-bounds threads
  if (global_idx < params.count) {
    shared[local_idx] = input[global_idx];
  } else {
    shared[local_idx] = 0.0;
  }

  workgroupBarrier();

  // Tree reduction in shared memory
  var stride: u32 = WG_SIZE >> 1u;
  loop {
    if (stride == 0u) {
      break;
    }
    if (local_idx < stride) {
      shared[local_idx] = shared[local_idx] + shared[local_idx + stride];
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }

  // First thread writes workgroup result
  if (local_idx == 0u) {
    output[wid.x] = shared[0];
  }
}
`;

/**
 * Tiled 16x16 matrix multiply shader.
 *
 * Computes C = A * B using shared-memory tiling for improved cache locality.
 * Each workgroup computes a 16x16 tile of the output matrix. Tiles of A and B
 * are loaded into workgroup-shared memory in a sliding window along the K
 * dimension.
 *
 * Bindings:
 *   @group(0) @binding(0) — A:      array<f32> (read-only, row-major M x K)
 *   @group(0) @binding(1) — B:      array<f32> (read-only, row-major K x N)
 *   @group(0) @binding(2) — C:      array<f32> (storage, row-major M x N)
 *   @group(0) @binding(3) — params: struct { M, N, K: u32 }
 */
export const MATMUL_WGSL: string = /* wgsl */ `
struct Params {
  M: u32,
  N: u32,
  K: u32,
}

@group(0) @binding(0) var<storage, read>       A:      array<f32>;
@group(0) @binding(1) var<storage, read>       B:      array<f32>;
@group(0) @binding(2) var<storage, read_write> C:      array<f32>;
@group(0) @binding(3) var<uniform>             params: Params;

const TILE: u32 = 16u;

var<workgroup> tileA: array<f32, 256>;  // 16 * 16
var<workgroup> tileB: array<f32, 256>;  // 16 * 16

@compute @workgroup_size(16, 16)
fn main(
  @builtin(local_invocation_id)  lid: vec3<u32>,
  @builtin(workgroup_id)         wid: vec3<u32>,
) {
  let row = wid.y * TILE + lid.y;
  let col = wid.x * TILE + lid.x;

  var acc: f32 = 0.0;

  let numTiles = (params.K + TILE - 1u) / TILE;

  for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
    // Load tile of A into shared memory
    let aCol = t * TILE + lid.x;
    if (row < params.M && aCol < params.K) {
      tileA[lid.y * TILE + lid.x] = A[row * params.K + aCol];
    } else {
      tileA[lid.y * TILE + lid.x] = 0.0;
    }

    // Load tile of B into shared memory
    let bRow = t * TILE + lid.y;
    if (bRow < params.K && col < params.N) {
      tileB[lid.y * TILE + lid.x] = B[bRow * params.N + col];
    } else {
      tileB[lid.y * TILE + lid.x] = 0.0;
    }

    workgroupBarrier();

    // Accumulate dot product for this tile
    for (var k: u32 = 0u; k < TILE; k = k + 1u) {
      acc = acc + tileA[lid.y * TILE + k] * tileB[k * TILE + lid.x];
    }

    workgroupBarrier();
  }

  // Write result
  if (row < params.M && col < params.N) {
    C[row * params.N + col] = acc;
  }
}
`;

/**
 * Blelloch-style exclusive prefix sum (scan) shader.
 *
 * Two-phase algorithm within a single workgroup:
 *   Phase 1 (up-sweep):   build partial sums in a binary tree
 *   Phase 2 (down-sweep): propagate prefix sums back down
 *
 * For arrays larger than one workgroup, a multi-level approach is needed
 * (dispatch this shader hierarchically on the per-workgroup sums).
 *
 * Bindings:
 *   @group(0) @binding(0) — data:   array<u32> (storage, in-place)
 *   @group(0) @binding(1) — params: struct { count: u32 }
 */
export const PREFIX_SUM_WGSL: string = /* wgsl */ `
struct Params {
  count: u32,
}

@group(0) @binding(0) var<storage, read_write> data:   array<u32>;
@group(0) @binding(1) var<uniform>             params: Params;

const WG_SIZE: u32 = 256u;
const N: u32 = 512u;  // Each workgroup processes 2 * WG_SIZE elements

var<workgroup> shared: array<u32, 512>;

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id)        wid: vec3<u32>,
) {
  let local_idx = lid.x;
  let base = wid.x * N;

  // Load two elements per thread into shared memory
  let idx0 = base + local_idx * 2u;
  let idx1 = idx0 + 1u;

  if (idx0 < params.count) {
    shared[local_idx * 2u] = data[idx0];
  } else {
    shared[local_idx * 2u] = 0u;
  }
  if (idx1 < params.count) {
    shared[local_idx * 2u + 1u] = data[idx1];
  } else {
    shared[local_idx * 2u + 1u] = 0u;
  }

  // --- Up-sweep (reduce) phase ---
  var offset: u32 = 1u;
  var d: u32 = N >> 1u;
  loop {
    if (d == 0u) {
      break;
    }
    workgroupBarrier();
    if (local_idx < d) {
      let ai = offset * (2u * local_idx + 1u) - 1u;
      let bi = offset * (2u * local_idx + 2u) - 1u;
      shared[bi] = shared[bi] + shared[ai];
    }
    offset = offset << 1u;
    d = d >> 1u;
  }

  // Clear the last element (exclusive scan identity)
  if (local_idx == 0u) {
    shared[N - 1u] = 0u;
  }

  // --- Down-sweep phase ---
  d = 1u;
  loop {
    if (offset == 0u) {
      break;
    }
    offset = offset >> 1u;
    workgroupBarrier();
    if (local_idx < d) {
      let ai = offset * (2u * local_idx + 1u) - 1u;
      let bi = offset * (2u * local_idx + 2u) - 1u;
      let temp = shared[ai];
      shared[ai] = shared[bi];
      shared[bi] = shared[bi] + temp;
    }
    d = d << 1u;
  }

  workgroupBarrier();

  // Write results back to global memory
  if (idx0 < params.count) {
    data[idx0] = shared[local_idx * 2u];
  }
  if (idx1 < params.count) {
    data[idx1] = shared[local_idx * 2u + 1u];
  }
}
`;

/**
 * Log-domain Sinkhorn row update shader for optimal transport.
 *
 * Operates in log-domain for f32 numerical stability:
 *   f_i = -epsilon * logsumexp_j( (f_i + g_j - C_ij) / epsilon )
 *
 * Uses the log-sum-exp trick: logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
 * to prevent overflow/underflow with f32 precision.
 *
 * Bindings:
 *   @group(0) @binding(0) — cost:   array<f32> (read-only, M x N cost matrix)
 *   @group(0) @binding(1) — f:      array<f32> (storage, dual variable, length M)
 *   @group(0) @binding(2) — g:      array<f32> (read-only, dual variable, length N)
 *   @group(0) @binding(3) — params: struct { M, N: u32, epsilon: f32 }
 */
export const SINKHORN_ROW_WGSL: string = /* wgsl */ `
struct Params {
  M:       u32,
  N:       u32,
  epsilon: f32,
}

@group(0) @binding(0) var<storage, read>       cost:   array<f32>;
@group(0) @binding(1) var<storage, read_write> f:      array<f32>;
@group(0) @binding(2) var<storage, read>       g:      array<f32>;
@group(0) @binding(3) var<uniform>             params: Params;

const WG_SIZE: u32 = 256u;

var<workgroup> smax: array<f32, 256>;
var<workgroup> ssum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id)  lid: vec3<u32>,
  @builtin(workgroup_id)         wid: vec3<u32>,
) {
  let row = wid.x;
  let local_idx = lid.x;
  let eps = params.epsilon;
  let inv_eps = 1.0 / eps;

  if (row >= params.M) {
    return;
  }

  let f_i = f[row];

  // Step 1: Find max over columns for numerical stability (log-sum-exp trick)
  var local_max: f32 = -1e30;
  var j = local_idx;
  loop {
    if (j >= params.N) {
      break;
    }
    let val = (f_i + g[j] - cost[row * params.N + j]) * inv_eps;
    local_max = max(local_max, val);
    j = j + WG_SIZE;
  }
  smax[local_idx] = local_max;

  workgroupBarrier();

  // Parallel max reduction
  var stride: u32 = WG_SIZE >> 1u;
  loop {
    if (stride == 0u) {
      break;
    }
    if (local_idx < stride) {
      smax[local_idx] = max(smax[local_idx], smax[local_idx + stride]);
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }

  let row_max = smax[0];

  // Step 2: Compute sum of exp(val - max) for log-sum-exp
  var local_sum: f32 = 0.0;
  j = local_idx;
  loop {
    if (j >= params.N) {
      break;
    }
    let val = (f_i + g[j] - cost[row * params.N + j]) * inv_eps;
    local_sum = local_sum + exp(val - row_max);
    j = j + WG_SIZE;
  }
  ssum[local_idx] = local_sum;

  workgroupBarrier();

  // Parallel sum reduction
  stride = WG_SIZE >> 1u;
  loop {
    if (stride == 0u) {
      break;
    }
    if (local_idx < stride) {
      ssum[local_idx] = ssum[local_idx] + ssum[local_idx + stride];
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }

  // Step 3: Update dual variable
  if (local_idx == 0u) {
    f[row] = -eps * (row_max + log(ssum[0]));
  }
}
`;

/**
 * Checkerboard Ising model Metropolis update shader.
 *
 * Uses a checkerboard decomposition to update half the spins in parallel
 * without data races. Each dispatch updates either "black" or "white" sites.
 * The parity uniform selects which sublattice to update.
 *
 * The Metropolis acceptance criterion uses a random number compared against
 * exp(-beta * deltaE), where deltaE = 2 * J * s_i * sum(neighbors).
 * Random numbers are generated via a simple PCG hash of the global index
 * combined with a per-iteration seed.
 *
 * Bindings:
 *   @group(0) @binding(0) — spins:  array<i32> (storage, W x H lattice, +1/-1)
 *   @group(0) @binding(1) — params: struct { W, H: u32, beta: f32, J: f32, parity: u32, seed: u32 }
 */
export const ISING_METROPOLIS_WGSL: string = /* wgsl */ `
struct Params {
  W:      u32,
  H:      u32,
  beta:   f32,
  J:      f32,
  parity: u32,
  seed:   u32,
}

@group(0) @binding(0) var<storage, read_write> spins:  array<i32>;
@group(0) @binding(1) var<uniform>             params: Params;

// PCG-style hash for per-site random number generation
fn pcg_hash(input: u32) -> u32 {
  var state = input * 747796405u + 2891336453u;
  var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn rand_f32(idx: u32, seed: u32) -> f32 {
  return f32(pcg_hash(idx ^ seed)) / 4294967296.0;
}

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  let linear = gid.x;
  let W = params.W;
  let H = params.H;

  // Map linear index to 2D grid position
  let row = linear / (W / 2u);
  let half_col = linear % (W / 2u);

  if (row >= H) {
    return;
  }

  // Checkerboard: offset column based on row parity and dispatch parity
  let col = half_col * 2u + ((row + params.parity) % 2u);

  if (col >= W) {
    return;
  }

  let idx = row * W + col;
  let s_i = spins[idx];

  // Sum nearest neighbors with periodic boundary conditions
  let up    = spins[((row + H - 1u) % H) * W + col];
  let down  = spins[((row + 1u)     % H) * W + col];
  let left  = spins[row * W + ((col + W - 1u) % W)];
  let right = spins[row * W + ((col + 1u)     % W)];

  let neighbor_sum = up + down + left + right;

  // Energy change for flipping spin s_i:
  //   deltaE = 2 * J * s_i * sum(neighbors)
  let deltaE = 2.0 * params.J * f32(s_i) * f32(neighbor_sum);

  // Metropolis acceptance
  let r = rand_f32(idx, params.seed);

  if (deltaE <= 0.0 || r < exp(-params.beta * deltaE)) {
    spins[idx] = -s_i;
  }
}
`;
