// ---------------------------------------------------------------------------
// @omni-twin/compute-core — High-Performance Computing & Parallel Architecture
// ---------------------------------------------------------------------------
// Barrel re-export for all 12 HPC sub-domains (HPC-1 through HPC-12).
// ---------------------------------------------------------------------------

// Infrastructure: types, PRNG, hash utilities
export * from './types.js';

// HPC-1: WebGPU Compute Shader Infrastructure
export * from './webgpu/index.js';

// HPC-2: Rust→WASM Pipeline
export * from './wasm/index.js';

// HPC-3: Worker Pool with SharedArrayBuffer
export * from './workers/index.js';

// HPC-4: GPU Server Compute
export * from './gpu-server/index.js';

// HPC-5: Spatial Indexing (R-tree, k-d Tree, BVH)
export * from './spatial/index.js';

// HPC-6: Streaming Algorithms
export * from './streaming/index.js';

// HPC-7: Collaborative CRDTs
export * from './crdt/index.js';

// HPC-8: Task Scheduling & Job Queues
export * from './scheduling/index.js';

// HPC-9: Browser ↔ Server Offload Decision
export * from './offload/index.js';

// HPC-10: Profiling Toolkit
export * from './profiling/index.js';

// HPC-11: Deployment Architecture
export * from './deployment/index.js';

// HPC-12: Numerical Linear Algebra
export * from './linalg/index.js';
