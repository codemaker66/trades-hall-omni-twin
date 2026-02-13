# HPC Command Track

Canonical command track: `docs/commands/hpc/`

Execution authority for agent commands is `docs/commands/**`.

## Commands

- `HPC-1` - WebGPU Compute Shader Infrastructure (depends_on: None)
- `HPC-2` - Rust→WASM Compilation Pipeline (depends_on: `HPC-1`)
- `HPC-3` - Web Worker Pool with SharedArrayBuffer (depends_on: `HPC-2`)
- `HPC-4` - GPU Server Compute (JAX + RAPIDS + PyTorch) (depends_on: `HPC-3`)
- `HPC-5` - Spatial Indexing (R-tree, k-d Tree, BVH) (depends_on: `HPC-INT-1`)
- `HPC-6` - Streaming Algorithms for Booking Analytics (depends_on: `HPC-5`)
- `HPC-7` - Collaborative Real-Time Editing via Yjs CRDTs (depends_on: `HPC-6`)
- `HPC-8` - Task Scheduling & Job Queues (depends_on: `HPC-7`)
- `HPC-9` - Browser ↔ Server Offload Decision Logic (depends_on: `HPC-INT-2`)
- `HPC-10` - Profiling Toolkit (depends_on: `HPC-9`)
- `HPC-11` - Deployment Architecture (depends_on: `HPC-10`)
- `HPC-12` - Numerical Linear Algebra Decision Matrix (depends_on: `HPC-11`)
- `HPC-INT-1` - HPC integration checkpoint 1 (depends_on: `HPC-1`, `HPC-2`, `HPC-3`, `HPC-4`)
- `HPC-INT-2` - HPC integration checkpoint 2 (depends_on: `HPC-5`, `HPC-6`, `HPC-7`, `HPC-8`)
- `HPC-INT-3` - HPC integration checkpoint 3 (depends_on: `HPC-9`, `HPC-10`, `HPC-11`, `HPC-12`)
