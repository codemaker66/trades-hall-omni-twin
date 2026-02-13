# TECHNIQUE_06_HPC_PARALLEL_ARCHITECTURE.md — High-Performance Computing & Parallel Architecture

> ## CRITICAL: NEVER OMIT, SIMPLIFY, OR WEAKEN
>
> Claude Code must NEVER:
> - Omit features, techniques, or subsystems to fit within a single session
> - Simplify implementations to make them "easier" or "more practical"
> - Replace a specified technique with a simpler alternative without explicit approval
> - Skip sections because they seem redundant or low-priority
> - Truncate implementations to keep diffs small — instead, split across commits
>
> If a task is too large for one session: split it into sub-tasks across multiple
> sessions. Document what remains in PROGRESS.md and pick it up next session.
> The project must contain the FULL depth of every technique specified.
> "Good enough" is not good enough. Every feature exists at maximum fidelity.

> **Purpose**: Feed this to Claude Code after the main phases and earlier techniques
> are complete. Implements the three-tier compute architecture: WebGPU compute shaders
> in the browser, Rust→WASM with Web Workers for CPU-parallel work, and GPU server
> backends for problems exceeding browser limits. Also covers collaborative real-time
> editing, spatial indexing, streaming algorithms, profiling, and deployment.
>
> **How to use**: Tell Claude Code: "Read TECHNIQUE_06_HPC_PARALLEL_ARCHITECTURE.md and
> implement incrementally, starting from HPC-1."
>
> **This is 12 sub-domains. Implement ALL of them. Do not skip any.**

---

## Architecture Overview — The Three-Tier Compute Model

```
┌──────────────────── BROWSER ─────────────────────┐
│ Tier 1: WebGPU Compute Shaders                    │
│   Matrix ops, Sinkhorn, parallel sort/scan         │
│   ~50-80% of native CUDA for compute-bound kernels │
│   ~10-23× faster than WASM for parallel work       │
│                                                    │
│ Tier 2: Rust→WASM + Web Workers                    │
│   Monte Carlo, spatial indexing, layout energy      │
│   ~60-90% of native Rust                           │
│   4-16 workers via navigator.hardwareConcurrency   │
│                                                    │
│ Tier 3: Collaborative CRDTs (Yjs)                  │
│   LWW-Register per furniture property              │
│   OR-Set for furniture collections                 │
└──────────────────────┬───────────────────────────┘
                       │ WebSocket / SSE
┌──────────────────── SERVER ──────────────────────┐
│ GPU Compute: JAX (JIT + grad + pmap)              │
│ Spatial: RAPIDS cuSpatial                         │
│ Jobs: Temporal (durable workflows)                │
│ Queues: BullMQ (Node) / Celery (Python)           │
│ Decision: <500MB & <5s → browser, else → server   │
└──────────────────────────────────────────────────┘
```

### Performance Targets (Concrete Benchmarks from Research)

| Workload | Target | Platform |
|---|---|---|
| 1000×1000 matmul | **1-5ms** | WebGPU |
| 10K path Monte Carlo (BS) | **<10ms** | WASM |
| Layout energy (50 items) | **<1ms** per eval → **1000+ evals/sec** | WASM |
| Sinkhorn 500×500 (50-200 iter) | **10-50ms** | WebGPU |
| Order book matching | **500K-2M ops/sec** | WASM |
| 3D rendering + compute | **60fps** (8-12ms render, 2-6ms compute, 1-2ms JS) | WebGPU |

---

## Directory Structure

```
packages/
  compute-core/                    — TypeScript orchestration layer
    src/
      webgpu/
        device.ts                  — WebGPU device init + feature detection + fallback
        shaders/
          reduce.wgsl              — Parallel reduction
          prefix-sum.wgsl          — Three-pass prefix sum
          matmul.wgsl              — Tiled matrix multiply
          sinkhorn.wgsl            — Log-domain Sinkhorn (f32-safe)
          ising-metropolis.wgsl    — Checkerboard Ising updates
        pipeline-cache.ts          — Pre-created pipelines (avoid per-frame recompilation)
        compute-manager.ts         — Dispatch orchestration, timestamp queries
      workers/
        worker-pool.ts             — SharedArrayBuffer-based worker pool
        atomics-mutex.ts           — Lock via Atomics.compareExchange
        ringbuf.ts                 — Lock-free SPSC ring buffer
        parallel-tempering.ts      — PT worker coordinator
      spatial/
        rtree.ts                   — rbush wrapper for 2D layout queries
      streaming/
        bloom-filter.ts            — Bloom filter for booking dedup
        count-min.ts               — CMS for demand frequency
        hyperloglog.ts             — HLL for cardinality estimation
        tdigest.ts                 — Streaming quantiles (p95/p99)
      crdt/
        layout-doc.ts              — Yjs document model for collaborative layout editing
        awareness.ts               — Cursor positions, selections, presence
      offload.ts                   — Browser ↔ server decision logic
      index.ts

  compute-wasm/                    — Rust WASM (CPU-parallel numerical work)
    src/
      lib.rs
      spatial/
        kdtree.rs                  — kiddo wrapper (Van Emde Boas layout)
        bvh.rs                     — BVH for collision detection
        rtree.rs                   — rstar for bounding box queries
      linalg.rs                    — faer wrapper for WASM linear algebra
      simd.rs                      — WASM SIMD-accelerated kernels
    Cargo.toml

apps/
  ml-api/
    src/
      compute/
        gpu_compute.py             — JAX/CuPy/PyTorch GPU compute
        spatial_gpu.py             — RAPIDS cuSpatial operations
      jobs/
        temporal_workflows.py      — Temporal durable workflow definitions
        celery_tasks.py            — Celery GPU task routing
      routes/
        compute.py                 — FastAPI endpoints for server-offloaded compute

  web/
    src/
      components/
        compute/
          WebGPUStatus.tsx         — Feature detection + fallback indicator
          ComputeProfiler.tsx      — Real-time timing display (timestamp queries)
          OffloadIndicator.tsx     — Shows browser vs server execution
```

---

## HPC-1: WebGPU Compute Shader Infrastructure

### What to Build

WebGPU device initialization with feature detection, fallback to WASM CPU,
and a compute pipeline cache that pre-creates all pipelines at init (not per-frame).

### Device Initialization

```typescript
// packages/compute-core/src/webgpu/device.ts

export interface GPUCapabilities {
  available: boolean;
  maxWorkgroupSize: number;
  maxWorkgroupsPerDimension: number;
  maxStorageBufferSize: number;
  maxWorkgroupStorageSize: number;
  f16Supported: boolean;
  timestampQuerySupported: boolean;
}

export async function initWebGPU(): Promise<{
  device: GPUDevice;
  capabilities: GPUCapabilities;
} | null> {
  if (!navigator.gpu) return null;  // Fallback to WASM

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: 'high-performance',
  });
  if (!adapter) return null;

  // Request maximum limits
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupsPerDimension: adapter.limits.maxComputeWorkgroupsPerDimension,
    },
    requiredFeatures: [
      ...(adapter.features.has('shader-f16') ? ['shader-f16'] : []),
      ...(adapter.features.has('timestamp-query') ? ['timestamp-query'] : []),
    ],
  });

  return {
    device,
    capabilities: {
      available: true,
      maxWorkgroupSize: device.limits.maxComputeInvocationsPerWorkgroup,
      maxWorkgroupsPerDimension: device.limits.maxComputeWorkgroupsPerDimension,
      maxStorageBufferSize: device.limits.maxStorageBufferBindingSize,
      maxWorkgroupStorageSize: device.limits.maxComputeWorkgroupStorageSize,
      f16Supported: device.features.has('shader-f16'),
      timestampQuerySupported: device.features.has('timestamp-query'),
    },
  };
}
```

### WGSL Compute Shaders

**Parallel Reduction:**
```wgsl
// packages/compute-core/src/webgpu/shaders/reduce.wgsl

const WG_SIZE: u32 = 256u;
var<workgroup> shared_data: array<f32, 256>;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>; // x = array length

@compute @workgroup_size(256)
fn reduce(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let gid = wid.x * WG_SIZE + lid.x;
  // Load with bounds check
  shared_data[lid.x] = select(0.0, input[gid], gid < params.x);
  workgroupBarrier();

  // Tree reduction — NO cross-workgroup sync (Metal constraint)
  for (var stride = WG_SIZE / 2u; stride > 0u; stride /= 2u) {
    if (lid.x < stride) {
      shared_data[lid.x] += shared_data[lid.x + stride];
    }
    workgroupBarrier();
  }

  if (lid.x == 0u) {
    output[wid.x] = shared_data[0];
  }
}
```

**Tiled Matrix Multiply:**
```wgsl
// packages/compute-core/src/webgpu/shaders/matmul.wgsl

const TILE: u32 = 16u;
var<workgroup> tileA: array<f32, 256>; // 16×16
var<workgroup> tileB: array<f32, 256>;

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec4<u32>; // M, N, K, 0

@compute @workgroup_size(16, 16)
fn matmul(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let M = dims.x; let N = dims.y; let K = dims.z;
  let row = gid.y; let col = gid.x;
  var sum: f32 = 0.0;

  for (var t = 0u; t < (K + TILE - 1u) / TILE; t++) {
    // Cooperative loading — coalesced access pattern
    let a_col = t * TILE + lid.x;
    let b_row = t * TILE + lid.y;
    tileA[lid.y * TILE + lid.x] = select(0.0, A[row * K + a_col], row < M && a_col < K);
    tileB[lid.y * TILE + lid.x] = select(0.0, B[b_row * N + col], b_row < K && col < N);
    workgroupBarrier();

    for (var k = 0u; k < TILE; k++) {
      sum += tileA[lid.y * TILE + k] * tileB[k * TILE + lid.x];
    }
    workgroupBarrier();
  }

  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}
```

**Log-Domain Sinkhorn (f32-safe):**
```wgsl
// packages/compute-core/src/webgpu/shaders/sinkhorn.wgsl

// Log-domain Sinkhorn is ESSENTIAL for f32 precision.
// Standard Sinkhorn underflows at small ε; log-domain avoids this.
// Viable for ε ≥ ~0.01 in f32.

@group(0) @binding(0) var<storage, read> log_K: array<f32>;  // -C/ε (log-kernel)
@group(0) @binding(1) var<storage, read_write> u: array<f32>; // dual variable
@group(0) @binding(2) var<storage, read_write> v: array<f32>; // dual variable
@group(0) @binding(3) var<uniform> params: vec4<u32>; // n_rows, n_cols, 0, 0

// Log-sum-exp reduction for numerical stability
fn log_sum_exp(values: ptr<workgroup, array<f32, 256>>, n: u32, lid: u32) -> f32 {
  // Find max, then compute log(Σ exp(x_i - max))
  // ... (workgroup reduction pattern)
}

@compute @workgroup_size(256)
fn sinkhorn_row_update(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let row = gid.x;
  let n_cols = params.y;
  if (row >= params.x) { return; }

  // u_i = log(a_i) - logsumexp_j(log_K[i,j] + v_j)
  // ... log-sum-exp over columns for this row
}
```

### Pipeline Cache

```typescript
// packages/compute-core/src/webgpu/pipeline-cache.ts

/**
 * CRITICAL ANTIPATTERN AVOIDED: Never create pipelines per frame.
 * Pre-create ALL compute pipelines at initialization.
 * Pipeline creation involves shader compilation → 10-100ms each.
 */
export class PipelineCache {
  private pipelines: Map<string, GPUComputePipeline> = new Map();

  async init(device: GPUDevice): Promise<void> {
    // Pre-create all pipelines
    this.pipelines.set('reduce', await this.createPipeline(device, reduceWGSL));
    this.pipelines.set('matmul', await this.createPipeline(device, matmulWGSL));
    this.pipelines.set('sinkhorn_row', await this.createPipeline(device, sinkhornWGSL));
    this.pipelines.set('ising', await this.createPipeline(device, isingWGSL));
    this.pipelines.set('prefix_sum', await this.createPipeline(device, prefixSumWGSL));
  }

  get(name: string): GPUComputePipeline { ... }
}
```

### Timestamp Queries for Profiling

```typescript
/**
 * Measure shader execution in nanoseconds.
 * Default precision: 100µs (security). Full resolution via chrome://flags.
 * Apple Silicon does NOT support timestamp queries (Metal limitation).
 */
export async function profileCompute(
  device: GPUDevice,
  commandEncoder: GPUCommandEncoder,
  computeFn: (pass: GPUComputePassEncoder) => void,
): Promise<number> {
  const querySet = device.createQuerySet({ type: 'timestamp', count: 2 });
  const resolveBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const readBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

  const pass = commandEncoder.beginComputePass({
    timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 }
  });
  computeFn(pass);
  pass.end();

  commandEncoder.resolveQuerySet(querySet, 0, 2, resolveBuffer, 0);
  commandEncoder.copyBufferToBuffer(resolveBuffer, 0, readBuffer, 0, 16);

  device.queue.submit([commandEncoder.finish()]);
  await readBuffer.mapAsync(GPUMapMode.READ);
  const times = new BigInt64Array(readBuffer.getMappedRange());
  const ns = Number(times[1] - times[0]);
  readBuffer.unmap();
  return ns;
}
```

---

## HPC-2: Rust→WASM Compilation Pipeline

### What to Build

Maximum-performance WASM build configuration, SIMD acceleration, and zero-copy
data transfer patterns.

### Cargo.toml Configuration

```toml
# packages/compute-wasm/Cargo.toml

[lib]
crate-type = ["cdylib"]

[dependencies]
wasm-bindgen = "0.2.106"
serde = { version = "1", features = ["derive"] }
serde-wasm-bindgen = "0.6"
rand = { version = "0.8", features = ["js"] }
rand_distr = "0.4"
faer = "0.20"               # WASM linear algebra (SIMD, no Fortran deps)
nalgebra = "0.34"            # Small-matrix geometric types
kiddo = "5.2"                # k-d tree with Van Emde Boas layout
rstar = "0.12"               # R-tree for bounding box queries
bvh = "0.9"                  # BVH for collision detection
sprs = "0.11"                # Sparse matrices (CSR/CSC/COO)
ordered-float = "4"

[profile.release]
opt-level = 3
lto = true                   # Link-time optimization
codegen-units = 1            # Single codegen unit for best optimization
panic = "abort"              # Remove unwinding infrastructure (~10% smaller)
strip = true                 # Remove debug symbols
```

### Build Script

```bash
#!/bin/bash
# packages/compute-wasm/build.sh

# Enable SIMD (Chrome 91+, Firefox 89+, Safari 16.4+ — universal)
export RUSTFLAGS="-C target-feature=+simd128"

# Build WASM
wasm-pack build --target web --release

# Post-process with wasm-opt for 15-20% additional size reduction
wasm-opt -O3 pkg/compute_wasm_bg.wasm -o pkg/compute_wasm_bg.wasm

# Analyze binary size
twiggy top -n 20 pkg/compute_wasm_bg.wasm
```

### Zero-Copy Data Transfer

```typescript
// CRITICAL: TypedArray views are invalidated when memory.grow() is called.
// Always re-derive views from memory.buffer AFTER any WASM call that might allocate.

import init, { memory, compute_layout_energy } from './pkg/compute_wasm.js';

await init();

// Allocate buffer in WASM, get pointer
const ptr = allocate_f64_buffer(1000);

// Create zero-copy view into WASM memory
let view = new Float64Array(memory.buffer, ptr, 1000);

// Fill data (writes directly to WASM memory — no copy)
view.set(myData);

// Call WASM function
const result = compute_layout_energy(ptr, 1000);

// CRITICAL: re-derive view after WASM call (buffer may have grown)
view = new Float64Array(memory.buffer, ptr, 1000);
```

### WASM SIMD Kernels

```rust
// packages/compute-wasm/src/simd.rs

#[cfg(target_arch = "wasm32")]
use std::arch::wasm32::*;

/// SIMD-accelerated dot product — 2-4× speedup over scalar
#[cfg(target_arch = "wasm32")]
pub fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 4;
    let mut sum = f32x4_splat(0.0);

    for i in 0..chunks {
        let va = v128_load(a[i * 4..].as_ptr() as *const v128);
        let vb = v128_load(b[i * 4..].as_ptr() as *const v128);
        sum = f32x4_add(sum, f32x4_mul(va, vb));
    }

    // Horizontal sum
    let mut result = f32x4_extract_lane::<0>(sum)
        + f32x4_extract_lane::<1>(sum)
        + f32x4_extract_lane::<2>(sum)
        + f32x4_extract_lane::<3>(sum);

    // Handle remainder
    for i in (chunks * 4)..n {
        result += a[i] * b[i];
    }
    result
}
```

### faer for WASM Linear Algebra

```rust
// packages/compute-wasm/src/linalg.rs

use faer::prelude::*;
use wasm_bindgen::prelude::*;

/// faer is the DEFINITIVE choice for WASM linear algebra:
/// - Pure Rust, no Fortran dependencies
/// - Explicit WASM simd128 support
/// - Performance competitive with OpenBLAS:
///   n=1024 matmul: faer 28.5ms vs OpenBLAS 11.1ms vs nalgebra 117.7ms
///   n=1024 Cholesky: faer 7.8ms vs OpenBLAS 14.5ms (faer WINS)
///
/// nalgebra (0.34) is 5-20× slower for large matrices.
/// ndarray-linalg CANNOT compile to WASM (Fortran LAPACK dependency).

#[wasm_bindgen]
pub fn solve_linear_system(
    a_data: Vec<f64>,  // Flattened n×n matrix
    b_data: Vec<f64>,  // n-vector
    n: usize,
) -> Vec<f64> {
    let a = Mat::from_fn(n, n, |i, j| a_data[i * n + j]);
    let b = Col::from_fn(n, |i| b_data[i]);
    let lu = a.partial_piv_lu();
    let x = lu.solve(&b);
    (0..n).map(|i| x[i]).collect()
}
```

---

## HPC-3: Web Worker Pool with SharedArrayBuffer

### What to Build

A worker pool with SharedArrayBuffer-based shared memory, Atomics mutex,
lock-free ring buffer for SPSC communication, and Comlink for ergonomic RPC.

### CORS Headers (Required)

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

For Vite: use `vite-plugin-cross-origin-isolation` or custom middleware.
Verify at runtime: `if (!crossOriginIsolated) { /* fallback to postMessage */ }`

### Atomics Mutex

```typescript
// packages/compute-core/src/workers/atomics-mutex.ts

const UNLOCKED = 0;
const LOCKED = 1;

export class AtomicsMutex {
  private view: Int32Array;

  constructor(sab: SharedArrayBuffer, offset: number) {
    this.view = new Int32Array(sab, offset, 1);
  }

  lock(): void {
    for (;;) {
      if (Atomics.compareExchange(this.view, 0, UNLOCKED, LOCKED) === UNLOCKED) return;
      Atomics.wait(this.view, 0, LOCKED); // Blocks worker thread (NOT main thread)
    }
  }

  unlock(): void {
    Atomics.store(this.view, 0, UNLOCKED);
    Atomics.notify(this.view, 0, 1);
  }
}
```

### Worker Pool

```typescript
// packages/compute-core/src/workers/worker-pool.ts

/**
 * Worker pool with SharedArrayBuffer coordination.
 *
 * navigator.hardwareConcurrency returns logical CPU count:
 *   Mobile: ~4, Laptop: ~8, Desktop: ~16+
 * Use hardwareConcurrency - 1 to leave a core for main thread.
 *
 * Each worker costs ~2MB memory + 40-100ms to create → pool them.
 * For tasks <16ms, postMessage overhead dominates → batch work.
 *
 * Transfer ArrayBuffers via postMessage(data, [data]) for O(1) moves.
 * 32MB round-trip: ~6.6ms transfer vs ~302ms clone (45× faster).
 */
export class WorkerPool {
  private workers: Worker[];
  private idle: Worker[];
  private sharedMemory?: SharedArrayBuffer;

  constructor(size?: number) {
    const n = (size ?? navigator.hardwareConcurrency) - 1;
    this.workers = Array.from({ length: n }, () =>
      new Worker(new URL('./compute-worker.ts', import.meta.url), { type: 'module' })
    );
    this.idle = [...this.workers];
  }

  // SharedArrayBuffer mode: all workers share memory
  initShared(sizeBytes: number): SharedArrayBuffer {
    this.sharedMemory = new SharedArrayBuffer(sizeBytes);
    for (const w of this.workers) {
      w.postMessage({ type: 'init-shared', buffer: this.sharedMemory });
    }
    return this.sharedMemory;
  }

  // Transfer mode: zero-copy O(1) buffer transfer
  async dispatch(buffer: ArrayBuffer): Promise<ArrayBuffer> {
    const worker = this.idle.pop()!;
    return new Promise(resolve => {
      worker.onmessage = (e) => {
        this.idle.push(worker);
        resolve(e.data);
      };
      worker.postMessage(buffer, [buffer]); // Transfer, not clone
    });
  }

  // Data-parallel: split array across all workers
  async parallelMap<T>(
    data: Float64Array,
    operation: string,
  ): Promise<Float64Array> {
    const chunkSize = Math.ceil(data.length / this.workers.length);
    // Create 10× more chunks than workers for load balancing
    // (faster workers naturally consume more chunks)
    ...
  }

  terminate(): void {
    this.workers.forEach(w => w.terminate());
  }
}
```

### OffscreenCanvas for Worker-Side Rendering

```typescript
/**
 * Transfer canvas rendering to a worker → eliminates main-thread jank.
 * Worker creates WebGPU context on the OffscreenCanvas.
 * Supported: Chrome 69+, Firefox 105+, Safari 16.4+.
 */
const offscreen = canvas.transferControlToOffscreen();
worker.postMessage({ type: 'init-render', canvas: offscreen }, [offscreen]);
```

---

## HPC-4: GPU Server Compute (JAX + RAPIDS + PyTorch)

### What to Build

Server-side GPU compute for problems exceeding browser limits (>500MB data
or >5s compute time).

```python
# apps/ml-api/src/compute/gpu_compute.py

import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, pmap

# JAX: the most elegant GPU compute model
# jit = compile to XLA kernel, vmap = auto-vectorize, grad = auto-diff
# pmap/shard_map = multi-GPU SPMD parallelism

@jit
def sinkhorn_gpu(cost_matrix: jnp.ndarray, eps: float, n_iter: int) -> jnp.ndarray:
    """GPU-accelerated Sinkhorn via JAX JIT compilation.
    Automatically compiled to optimized XLA kernels on first call."""
    log_K = -cost_matrix / eps
    u = jnp.zeros(cost_matrix.shape[0])
    v = jnp.zeros(cost_matrix.shape[1])
    for _ in range(n_iter):
        u = -jax.nn.logsumexp(log_K + v[None, :], axis=1)
        v = -jax.nn.logsumexp(log_K + u[:, None], axis=0)
    return jnp.exp(log_K + u[:, None] + v[None, :])

@jit
def batch_layout_energy(layouts: jnp.ndarray, room: dict) -> jnp.ndarray:
    """Evaluate energy for a batch of layouts simultaneously on GPU.
    vmap auto-vectorizes the single-layout function across the batch."""
    return vmap(single_layout_energy, in_axes=(0, None))(layouts, room)

# PyTorch for GPU linear algebra
# 1000×1000 matmul: ~39µs on CUDA vs ~16.8ms on NumPy (430× speedup)
import torch
A = torch.randn(1000, 1000, device='cuda')
B = torch.randn(1000, 1000, device='cuda')
C = torch.linalg.solve(A, B)  # cuSOLVER backend
```

### RAPIDS cuSpatial for Venue Layout

```python
# apps/ml-api/src/compute/spatial_gpu.py

"""
RAPIDS cuSpatial: GPU-accelerated spatial operations.
Point-in-polygon, spatial joins, proximity queries.
Directly relevant to venue layout optimization.
"""
import cudf
import cuspatial
```

---

## HPC-5: Spatial Indexing (R-tree, k-d Tree, BVH)

### What to Build

Spatial data structures for furniture collision detection, nearest-neighbor
queries, and bounding-box queries — all compiled to WASM.

### Rust Implementation

```rust
// packages/compute-wasm/src/spatial/kdtree.rs

use kiddo::KdTree;
use wasm_bindgen::prelude::*;

/// kiddo k-d tree with Van Emde Boas layout for cache-optimal traversal.
/// Critical for WASM where cache hierarchies are unknown.
/// Use for: nearest-neighbor venue similarity, furniture proximity.
///
/// Browser scale: <100K venues in kiddo WASM.
/// Server scale: HNSW (hnswlib) for 100K-10M, FAISS GPU for 10M+.
#[wasm_bindgen]
pub struct VenueSpatialIndex {
    tree: KdTree<f64, 3>,  // 3D: lat, lon, capacity (normalized)
}

#[wasm_bindgen]
impl VenueSpatialIndex {
    pub fn new() -> Self { ... }
    pub fn insert(&mut self, lat: f64, lon: f64, capacity: f64, id: u32) { ... }
    pub fn nearest(&self, lat: f64, lon: f64, capacity: f64, k: u32) -> Vec<u32> { ... }
    pub fn within_radius(&self, lat: f64, lon: f64, capacity: f64, r: f64) -> Vec<u32> { ... }
}
```

```rust
// packages/compute-wasm/src/spatial/bvh.rs

use bvh::bvh::Bvh;
use wasm_bindgen::prelude::*;

/// BVH for O(log n) collision detection in layout optimization.
/// Each furniture item is an axis-aligned bounding box.
/// Parallel build via Rayon (via wasm-bindgen-rayon).
#[wasm_bindgen]
pub struct LayoutCollisionDetector {
    bvh: Bvh<f64, 2>,  // 2D bounding boxes for floor plan
}

#[wasm_bindgen]
impl LayoutCollisionDetector {
    pub fn build(items: &[FurnitureItem]) -> Self { ... }
    pub fn check_overlaps(&self) -> Vec<(u32, u32)> { ... }
    pub fn query_point(&self, x: f64, y: f64) -> Vec<u32> { ... }
}
```

### TypeScript R-tree (rbush)

```typescript
// packages/compute-core/src/spatial/rtree.ts

import RBush from 'rbush';

/**
 * rbush: 2D R-tree for layout bounding box queries.
 * Bulk loading: 2-3× faster, 20-30% better query than incremental insert.
 */
export class LayoutRTree {
  private tree = new RBush<FurnitureBBox>();

  buildFromLayout(items: FurnitureItem[]): void {
    const bboxes = items.map(item => ({
      minX: item.x - item.width / 2,
      minY: item.y - item.depth / 2,
      maxX: item.x + item.width / 2,
      maxY: item.y + item.depth / 2,
      id: item.id,
    }));
    this.tree.load(bboxes); // Bulk load — O(n log n)
  }

  queryRegion(minX: number, minY: number, maxX: number, maxY: number): string[] {
    return this.tree.search({ minX, minY, maxX, maxY }).map(b => b.id);
  }
}
```

---

## HPC-6: Streaming Algorithms for Booking Analytics

### What to Build

Memory-efficient probabilistic data structures for real-time booking analytics.

```typescript
// packages/compute-core/src/streaming/

/**
 * Bloom filter: set membership with tunable false positive rate.
 * Size formula: m = -n·ln(p)/(ln 2)²
 * 1M items at 1% FP = ~1.2MB
 * Use for: "has this user already been shown this venue?"
 */

/**
 * Count-Min Sketch: frequency estimation in O(1) space.
 * Use for: "how many bookings has this venue type received today?"
 */

/**
 * HyperLogLog: cardinality estimation using ~1.5KB for >10⁹ items (~2% error).
 * Use for: "how many unique planners searched for venues this week?"
 */

/**
 * T-Digest: streaming quantile estimation (p95, p99).
 * Use for: "what is the 95th percentile booking price?"
 */

// npm: bloom-filters (Bloom, CMS, HLL, Top-K, MinHash in one package)
// npm: tdigest
// Rust: streaming_algorithms = "0.3" (SIMD-accelerated)
```

---

## HPC-7: Collaborative Real-Time Editing via Yjs CRDTs

### What to Build

Yjs-based collaborative layout editing where multiple planners can simultaneously
edit the same venue layout with automatic conflict resolution.

```typescript
// packages/compute-core/src/crdt/layout-doc.ts

import * as Y from 'yjs';
import { WebsocketProvider } from 'y-websocket';
import { IndexeddbPersistence } from 'y-indexeddb';

/**
 * Furniture modeled as Y.Map entries within Y.Array:
 * - Each property (x, y, width, height, rotation) is an LWW-Register
 *   (last-writer-wins: concurrent moves resolve to later timestamp)
 * - The collection of items is an OR-Set (add-wins: concurrent adds both survive)
 *
 * Yjs handles 10M+ characters and hundreds of concurrent users.
 */
export class CollaborativeLayoutDoc {
  private doc: Y.Doc;
  private furniture: Y.Array<Y.Map<any>>;
  private provider: WebsocketProvider;
  private persistence: IndexeddbPersistence;

  constructor(roomId: string, wsUrl: string) {
    this.doc = new Y.Doc();
    this.furniture = this.doc.getArray('furniture');

    // WebSocket sync for real-time collaboration
    this.provider = new WebsocketProvider(wsUrl, `room-${roomId}`, this.doc);

    // IndexedDB for offline persistence
    this.persistence = new IndexeddbPersistence(`room-${roomId}`, this.doc);
  }

  addItem(item: FurnitureSpec): void {
    const yItem = new Y.Map();
    this.doc.transact(() => {
      yItem.set('id', crypto.randomUUID());
      yItem.set('type', item.type);
      yItem.set('x', item.x);
      yItem.set('y', item.y);
      yItem.set('width', item.width);
      yItem.set('depth', item.depth);
      yItem.set('rotation', item.rotation);
      this.furniture.push([yItem]);
    });
  }

  moveItem(index: number, x: number, y: number): void {
    const item = this.furniture.get(index);
    this.doc.transact(() => {
      item.set('x', x);
      item.set('y', y);
    });
  }

  onUpdate(callback: (items: FurnitureItem[]) => void): void {
    this.furniture.observeDeep(() => {
      callback(this.getAll());
    });
  }

  // Awareness protocol: cursor positions, selections, presence
  setAwareness(user: { name: string; color: string; cursor: { x: number; y: number } }): void {
    this.provider.awareness.setLocalStateField('user', user);
  }

  getAll(): FurnitureItem[] { ... }
}
```

### Event Sourcing for Booking State

```typescript
/**
 * Every booking state change stored as an immutable event:
 *   BookingCreated, BookingConfirmed, SeatAssigned, LayoutChanged, PriceAdjusted
 *
 * Current state derived by replaying events.
 * Snapshot every ~100 events for performance.
 *
 * Options: EventStoreDB (purpose-built) or PostgreSQL with JSONB events table
 * + optimistic concurrency via unique (stream_id, version).
 *
 * CQRS: separate write (command → event) and read (event → materialized view).
 */
```

---

## HPC-8: Task Scheduling & Job Queues

### Server-Side Job Orchestration

```python
# apps/ml-api/src/jobs/temporal_workflows.py

"""
Temporal: durable workflow orchestration for long-running optimization.
Every state transition persisted via event sourcing.
If workers crash, Temporal replays history to restore state.
First-class cancellation + heartbeat monitoring for GPU activities.
"""
from temporalio import workflow, activity
from temporalio.common import RetryPolicy

@activity.defn
async def optimize_layout_gpu(layout: dict, config: dict) -> dict:
    """GPU layout optimization activity with heartbeat."""
    for iteration in range(config['max_iterations']):
        activity.heartbeat(iteration)
        # ... GPU compute ...
    return result

@workflow.defn
class VenueOptimizationWorkflow:
    @workflow.run
    async def run(self, request: dict) -> dict:
        # Step 1: Generate initial layout (LLM)
        initial = await workflow.execute_activity(
            generate_layout_llm, request,
            start_to_close_timeout=timedelta(seconds=30))

        # Step 2: Optimize via SA/PT on GPU
        optimized = await workflow.execute_activity(
            optimize_layout_gpu, initial,
            start_to_close_timeout=timedelta(minutes=10),
            heartbeat_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(maximum_attempts=3))

        # Step 3: Generate diverse alternatives
        alternatives = await workflow.execute_activity(
            sample_diverse_layouts, optimized,
            start_to_close_timeout=timedelta(minutes=5))

        return {'optimized': optimized, 'alternatives': alternatives}
```

### BullMQ for Node.js GPU Queue

```typescript
// apps/api/src/jobs/gpu-queue.ts
import { Queue, Worker } from 'bullmq';

const gpuQueue = new Queue('gpu-optimization', { connection: redisConfig });

// Forward progress to WebSocket
const worker = new Worker('gpu-optimization', async (job) => {
  for (let i = 0; i < job.data.maxIterations; i++) {
    await job.updateProgress(i / job.data.maxIterations * 100);
    // ... compute ...
  }
  return result;
}, { connection: redisConfig, concurrency: 1 }); // 1 job per GPU worker
```

---

## HPC-9: Browser ↔ Server Offload Decision Logic

```typescript
// packages/compute-core/src/offload.ts

/**
 * Decision boundary for browser vs server execution:
 *
 * BROWSER (Tier 1 + 2):
 * - Data < 500MB
 * - Compute < 5 seconds
 * - Layout energy evaluation (50 items → <1ms)
 * - Sinkhorn ≤ 500×500 (10-50ms on WebGPU)
 * - Monte Carlo ≤ 10K paths (<10ms)
 * - Spatial queries via rbush/kiddo
 *
 * SERVER (GPU):
 * - Data > 2GB or requiring GPU acceleration
 * - Multi-GPU workloads
 * - Diffusion model layout generation
 * - >50K path Monte Carlo with complex Greeks
 * - Million-spin simulated bifurcation
 * - FAISS nearest neighbor for 10M+ venues
 *
 * GPU kernel launch overhead: ~5-10µs
 * GPU transfer overhead: ~10-20µs
 * Break-even: matrix ops above ~256×256, vector ops above ~32K elements
 */
export function shouldOffload(task: ComputeTask): 'browser' | 'server' {
  if (task.dataSize > 500_000_000) return 'server';
  if (task.estimatedTimeMs > 5000) return 'server';
  if (task.requiresGPU && !webgpuAvailable) return 'server';
  if (task.type === 'diffusion_layout') return 'server';
  return 'browser';
}
```

---

## HPC-10: Profiling Toolkit

```typescript
/**
 * Browser: Chrome DevTools Performance + WebGPU timestamp queries
 * Rust: criterion.rs v0.5.1 (statistically rigorous microbenchmarks)
 * Python: py-spy (sampling), Scalene (CPU+GPU+memory), Nsight Systems (CUDA timeline)
 *
 * Common antipatterns to detect:
 * - WebGPU: excessive GPU→CPU readbacks, dispatches too small, per-frame pipeline creation
 * - WASM: per-element JS↔WASM boundary crossing, hot-loop allocation
 * - Workers: many small postMessage calls, structured clone instead of transfer
 * - GPU: synchronous CPU-GPU calls causing pipeline stalls
 */
```

---

## HPC-11: Deployment Architecture

### Edge (Light Workloads)

```
Cloudflare Workers: WASM at 300+ edge locations, <5ms cold start, 10MB limit
Deno Deploy: 35+ edges, native WASM
Vercel Edge: sub-50ms cold start, 10s execution limit
Use for: spatial queries, booking validation, CDN-like compute
```

### GPU Cloud (Heavy Workloads)

```
Modal:       ~$2.50/hr A100, per-second billing, sub-second cold starts
RunPod Flex: $2.72/hr A100, 48% cold starts <200ms
Lambda Labs: $1.79/hr A100, no egress fees
Spot saves:  60-91% off on-demand
Rule: if utilization >40-50%, dedicated beats serverless
```

### Progressive WASM Loading

```
Typical WASM binary: 1-10MB uncompressed
Brotli compression: 25-30% smaller than gzip
WebAssembly.compileStreaming(fetch('module.wasm')) — parallel download+compile
Cache compiled modules in IndexedDB for instant subsequent loads
Load UI first → WASM in parallel → compute modules lazy on demand
```

---

## HPC-12: Numerical Linear Algebra Decision Matrix

| Target | Library | Performance |
|---|---|---|
| WASM (general) | **faer** 0.20 | Competitive with OpenBLAS, SIMD support |
| WASM (small/geometric) | **nalgebra** 0.34 | Good for rotations, quaternions, <64×64 |
| WASM (sparse) | **sprs** 0.11 | CSR/CSC/COO + sparse Cholesky |
| WebGPU | Custom WGSL shaders | Tiled matmul, f32 only |
| GPU server | JAX `jnp.linalg` | JIT + auto-diff + multi-GPU |
| GPU server | PyTorch `torch.linalg` | cuBLAS/cuSOLVER, 39µs for 1K×1K |
| GPU sparse | PETSc / SuiteSparse | Industrial-strength, AMG preconditioners |

### f32 vs f64 Guidelines

- WGSL: f32 only. Use **log-domain Sinkhorn** for OT. Double-single emulation
  available but 4-8× slower.
- WASM: f64 native. Use faer for full double precision.
- Iterative solvers: f32 converges for well-conditioned (κ < 10⁴). Use **mixed-precision
  iterative refinement** for ill-conditioned: factor in f32, refine in f64.
- QP solvers in WASM: **Clarabel** (pure Rust, WASM-compatible conic solver) or
  **OSQP** (via eigen-js WASM).

---

## Integration with Other Techniques

- **Category Theory** (CT): Compute tier selection is a functor from problem category to
  execution target (browser/server).
- **Optimal Transport** (OT): Sinkhorn runs on WebGPU (shader in HPC-1), large problems
  offload to JAX server.
- **Physics Solvers** (PS): SA/PT use Web Workers (HPC-3), parallel tempering coordinates
  via SharedArrayBuffer + Atomics. Layout energy (PS-5) evaluated in WASM at <1ms.
- **Stochastic Pricing** (SP): Monte Carlo runs in WASM for ≤10K paths, offloads to GPU
  for complex multi-asset Greeks.
- **TDA**: Ripser WASM in browser for ≤500 points, server for larger datasets.

---

## Session Management

1. **HPC-1** (WebGPU compute: device init, shaders, pipeline cache, profiling) — 1-2 sessions
2. **HPC-2** (Rust→WASM: build pipeline, SIMD, zero-copy, faer) — 1 session
3. **HPC-3** (Web Workers: pool, SharedArrayBuffer, Atomics, Comlink) — 1 session
4. **HPC-4** (GPU server: JAX + RAPIDS + PyTorch compute) — 1 session
5. **HPC-5** (Spatial indexing: kiddo, rstar, bvh, rbush) — 1 session
6. **HPC-6** (Streaming algorithms: Bloom, CMS, HLL, T-Digest) — 1 session
7. **HPC-7** (Yjs CRDT collaborative editing + event sourcing) — 1-2 sessions
8. **HPC-8** (Temporal workflows + BullMQ + Celery GPU routing) — 1 session
9. **HPC-9** (Offload decision logic + hybrid patterns) — 1 session
10. **HPC-10 + 11** (Profiling toolkit + deployment architecture) — 1 session
11. **HPC-12** (Linear algebra integration + mixed-precision) — 1 session

Total: ~10-13 Claude Code sessions for the full HPC infrastructure.
