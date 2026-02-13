// ---------------------------------------------------------------------------
// @omni-twin/compute-core — High-Performance Computing & Parallel Architecture
// ---------------------------------------------------------------------------
// Shared types and utilities for all 12 HPC sub-domains.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Random number generation
// ---------------------------------------------------------------------------

/** Seedable PRNG function (same pattern as physics-solvers / learning-core). */
export type PRNG = () => number;

/**
 * Creates a seedable mulberry32 PRNG returning values in [0, 1).
 * Deterministic: identical seeds produce identical sequences.
 */
export function createPRNG(seed: number): PRNG {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------------------
// Hash utilities
// ---------------------------------------------------------------------------

/**
 * MurmurHash3 (32-bit) — pure TypeScript implementation.
 * Used by Bloom filter, Count-Min Sketch, HyperLogLog, and spatial hashing.
 *
 * Reference: Austin Appleby, MurmurHash3 (public domain).
 */
export function murmurHash3_32(key: string, seed: number): number {
  let h = seed | 0;
  const len = key.length;

  // Process 4-byte chunks using char codes
  const nBlocks = len >> 2;
  for (let i = 0; i < nBlocks; i++) {
    const i4 = i << 2;
    let k =
      (key.charCodeAt(i4) & 0xff) |
      ((key.charCodeAt(i4 + 1) & 0xff) << 8) |
      ((key.charCodeAt(i4 + 2) & 0xff) << 16) |
      ((key.charCodeAt(i4 + 3) & 0xff) << 24);

    k = Math.imul(k, 0xcc9e2d51);
    k = (k << 15) | (k >>> 17);
    k = Math.imul(k, 0x1b873593);

    h ^= k;
    h = (h << 13) | (h >>> 19);
    h = (Math.imul(h, 5) + 0xe6546b64) | 0;
  }

  // Process remaining bytes (avoid fallthrough switch for noFallthroughCasesInSwitch)
  const tailStart = nBlocks << 2;
  const remainder = len & 3;
  if (remainder > 0) {
    let k1 = 0;
    if (remainder >= 3) k1 ^= (key.charCodeAt(tailStart + 2) & 0xff) << 16;
    if (remainder >= 2) k1 ^= (key.charCodeAt(tailStart + 1) & 0xff) << 8;
    k1 ^= key.charCodeAt(tailStart) & 0xff;
    k1 = Math.imul(k1, 0xcc9e2d51);
    k1 = (k1 << 15) | (k1 >>> 17);
    k1 = Math.imul(k1, 0x1b873593);
    h ^= k1;
  }

  // Finalization mix
  h ^= len;
  h ^= h >>> 16;
  h = Math.imul(h, 0x85ebca6b);
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35);
  h ^= h >>> 16;

  return h >>> 0;
}

/**
 * FNV-1a hash (32-bit) — simple, fast hash for raw byte data.
 * Useful as a lightweight alternative to MurmurHash for small inputs.
 */
export function fnv1a(data: Uint8Array): number {
  let hash = 0x811c9dc5; // FNV offset basis
  for (let i = 0; i < data.length; i++) {
    hash ^= data[i]!;
    hash = Math.imul(hash, 0x01000193); // FNV prime
  }
  return hash >>> 0;
}

// ---------------------------------------------------------------------------
// HPC-1: WebGPU Compute Infrastructure
// ---------------------------------------------------------------------------

/** Detected capabilities of the GPU device, queried at startup. */
export interface GPUCapabilities {
  readonly available: boolean;
  readonly maxWorkgroupSize: number;
  readonly maxWorkgroupsPerDimension: number;
  readonly maxStorageBufferSize: number;
  readonly maxWorkgroupStorageSize: number;
  readonly f16Supported: boolean;
  readonly timestampQuerySupported: boolean;
}

/** Descriptor for creating a compute pipeline from WGSL source. */
export interface ComputePipelineDescriptor {
  readonly name: string;
  readonly shaderSource: string;
  readonly entryPoint: string;
  readonly workgroupSize: readonly [number, number, number];
}

/** A single compute dispatch specifying pipeline, workgroup counts, and buffer bindings. */
export interface ComputeDispatch {
  readonly pipeline: string;
  readonly workgroups: readonly [number, number, number];
  readonly buffers: readonly BufferBinding[];
}

/** Binding metadata for a GPU buffer used in a compute dispatch. */
export interface BufferBinding {
  readonly binding: number;
  readonly size: number;
  readonly usage: 'storage' | 'uniform' | 'read-only';
}

/** GPU timestamp query result for profiling pipeline execution. */
export interface TimestampResult {
  readonly pipelineName: string;
  readonly durationNs: number;
  readonly workgroupCount: number;
}

// ---------------------------------------------------------------------------
// HPC-2: WASM Pipeline
// ---------------------------------------------------------------------------

/** Configuration for loading a WASM module with optional features. */
export interface WASMModuleConfig {
  readonly url: string;
  readonly simdRequired: boolean;
  readonly sharedMemory: boolean;
  readonly initialMemoryPages: number;
  readonly maximumMemoryPages: number;
}

/** Feature detection results for the WASM runtime environment. */
export interface WASMCapabilities {
  readonly simd: boolean;
  readonly threads: boolean;
  readonly bulkMemory: boolean;
  readonly exceptionHandling: boolean;
  readonly gc: boolean;
}

/**
 * A typed region of WASM linear memory that can be shared with JS
 * without copying — the "zero-copy" bridge between WASM and TypeScript.
 */
export interface ZeroCopyRegion {
  readonly ptr: number;
  readonly byteLength: number;
  readonly dtype: 'f32' | 'f64' | 'i32' | 'u32' | 'u8';
}

// ---------------------------------------------------------------------------
// HPC-3: Worker Pool
// ---------------------------------------------------------------------------

/** Configuration for the browser Web Worker pool. */
export interface WorkerPoolConfig {
  readonly size: number;
  readonly sharedMemoryBytes: number;
  readonly taskTimeoutMs: number;
}

/** A task to be dispatched to a worker thread. */
export interface WorkerTask {
  readonly id: string;
  readonly type: string;
  readonly priority: number;
  readonly payload: Float64Array;
  readonly transferable: boolean;
}

/** Result returned from a worker after task execution. */
export interface WorkerResult {
  readonly taskId: string;
  readonly data: Float64Array;
  readonly durationMs: number;
  readonly workerId: number;
}

/**
 * State snapshot of a lock-free ring buffer backed by SharedArrayBuffer.
 * Used for high-throughput inter-worker communication.
 */
export type RingBufferState = {
  readonly head: number;
  readonly tail: number;
  readonly capacity: number;
  readonly full: boolean;
};

// ---------------------------------------------------------------------------
// HPC-4: GPU Server Compute
// ---------------------------------------------------------------------------

/** A compute task to be offloaded to a remote GPU server. */
export interface ServerComputeTask {
  readonly id: string;
  readonly type: 'sinkhorn' | 'layout_optimization' | 'monte_carlo' | 'diffusion' | 'faiss_search';
  readonly dataSize: number;
  readonly estimatedTimeMs: number;
  readonly priority: 'low' | 'medium' | 'high' | 'critical';
}

/** Result returned from the GPU server after task completion. */
export interface ServerComputeResult {
  readonly taskId: string;
  readonly status: 'completed' | 'failed' | 'timeout';
  readonly data: Float64Array;
  readonly gpuTimeMs: number;
  readonly transferTimeMs: number;
}

/** Configuration for a cloud GPU provider (Modal, RunPod, Lambda, etc.). */
export interface GPUCloudConfig {
  readonly provider: 'modal' | 'runpod' | 'lambda' | 'custom';
  readonly gpuType: 'a100' | 'h100' | 'a10g' | 't4';
  readonly costPerHour: number;
  readonly maxConcurrency: number;
}

// ---------------------------------------------------------------------------
// HPC-5: Spatial Indexing
// ---------------------------------------------------------------------------

/** Axis-aligned bounding box in 2D space. */
export interface BoundingBox2D {
  readonly minX: number;
  readonly minY: number;
  readonly maxX: number;
  readonly maxY: number;
}

/** Axis-aligned bounding box in 3D space. */
export interface BoundingBox3D {
  readonly minX: number;
  readonly minY: number;
  readonly minZ: number;
  readonly maxX: number;
  readonly maxY: number;
  readonly maxZ: number;
}

/** An item stored in a spatial index, carrying a bounding box and optional payload. */
export interface SpatialItem<T = unknown> {
  readonly id: string;
  readonly bbox: BoundingBox2D;
  readonly data: T;
}

/** Node in a k-d tree for nearest-neighbor and range queries. */
export interface KDTreeNode {
  readonly point: Float64Array;
  readonly id: string;
  readonly splitDimension: number;
  readonly left: KDTreeNode | null;
  readonly right: KDTreeNode | null;
}

/**
 * Node in a bounding volume hierarchy.
 * Leaf nodes have `itemIndex >= 0`; internal nodes have `itemIndex === -1`.
 */
export interface BVHNode {
  readonly bbox: BoundingBox2D;
  readonly left: BVHNode | null;
  readonly right: BVHNode | null;
  readonly itemIndex: number;
}

/** Result of a nearest-neighbor query from a spatial index. */
export interface NearestResult {
  readonly id: string;
  readonly distance: number;
  readonly point: Float64Array;
}

// ---------------------------------------------------------------------------
// HPC-6: Streaming Algorithms
// ---------------------------------------------------------------------------

/**
 * Configuration for a Bloom filter — a probabilistic set membership test.
 * Optimal bit array size and hash count are derived from these parameters.
 */
export interface BloomFilterConfig {
  readonly expectedItems: number;
  readonly falsePositiveRate: number;
}

/** Internal state of a Bloom filter. */
export interface BloomFilterState {
  readonly bits: Uint8Array;
  readonly numHashes: number;
  readonly size: number;
  readonly count: number;
}

/**
 * Configuration for a Count-Min Sketch — a probabilistic frequency estimator.
 * `width` controls accuracy; `depth` controls confidence.
 */
export interface CountMinSketchConfig {
  readonly width: number;
  readonly depth: number;
}

/** Internal state of a Count-Min Sketch. */
export interface CountMinSketchState {
  readonly table: Int32Array;
  readonly width: number;
  readonly depth: number;
  readonly totalCount: number;
}

/**
 * Internal state of a HyperLogLog cardinality estimator.
 * Uses `p` bits for register addressing, giving `m = 2^p` registers.
 * Valid precision range: 4-18.
 */
export interface HyperLogLogState {
  readonly registers: Uint8Array;
  readonly precision: number;
  readonly numRegisters: number;
}

/** A centroid in a t-digest, representing a cluster of data points. */
export interface TDigestCentroid {
  readonly mean: number;
  readonly count: number;
}

/** Configuration for a t-digest quantile estimator. */
export interface TDigestConfig {
  /** Compression parameter (delta). Higher = more centroids = better accuracy. Default 100. */
  readonly compression: number;
}

/** Internal state of a t-digest. */
export interface TDigestState {
  readonly centroids: readonly TDigestCentroid[];
  readonly totalCount: number;
  readonly compression: number;
  readonly min: number;
  readonly max: number;
}

// ---------------------------------------------------------------------------
// HPC-7: CRDT (Conflict-Free Replicated Data Types)
// ---------------------------------------------------------------------------

/**
 * Vector clock for tracking causal ordering across distributed peers.
 * Each entry maps a peer ID to that peer's logical timestamp.
 */
export interface VectorClock {
  readonly entries: ReadonlyMap<string, number>;
}

/**
 * Last-Writer-Wins Register — resolves conflicts by highest timestamp.
 * Ties broken by lexicographic peer ID comparison.
 */
export interface LWWRegister<T> {
  readonly value: T;
  readonly timestamp: number;
  readonly peerId: string;
}

/**
 * Element in an Observed-Remove Set (OR-Set).
 * Uses unique add/remove tags to resolve concurrent add/remove conflicts.
 */
export interface ORSetElement<T> {
  readonly value: T;
  readonly addTags: ReadonlySet<string>;
  readonly removeTags: ReadonlySet<string>;
}

/** A single CRDT operation to be broadcast or applied. */
export interface CRDTOperation {
  readonly type: 'set' | 'add' | 'remove' | 'move';
  readonly peerId: string;
  readonly timestamp: number;
  readonly key: string;
  readonly value?: unknown;
}

/**
 * Awareness state for real-time collaboration — cursor position,
 * selection, and user identity for a single peer.
 */
export interface AwarenessState {
  readonly peerId: string;
  readonly name: string;
  readonly color: string;
  readonly cursor: { readonly x: number; readonly y: number };
  readonly selection: readonly string[];
  readonly lastActive: number;
}

/** Mutable vector clock used internally during CRDT merge operations. */
export interface MutableVectorClock {
  entries: Map<string, number>;
}

// ---------------------------------------------------------------------------
// HPC-8: Task Scheduling
// ---------------------------------------------------------------------------

/** A job tracked by the task scheduler with priority and retry metadata. */
export interface ScheduledJob {
  readonly id: string;
  readonly type: string;
  readonly priority: number;
  readonly createdAt: number;
  readonly scheduledAt: number;
  readonly timeoutMs: number;
  readonly retries: number;
  readonly maxRetries: number;
  readonly payload: unknown;
}

/** Lifecycle status of a scheduled job. */
export type JobStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | 'retrying';

/** A single step in a multi-step workflow DAG. */
export interface WorkflowStep {
  readonly name: string;
  readonly action: string;
  readonly dependsOn: readonly string[];
  readonly timeoutMs: number;
  readonly retryPolicy: RetryPolicy;
}

/** Exponential backoff retry configuration. */
export interface RetryPolicy {
  readonly maxAttempts: number;
  readonly backoffMs: number;
  readonly backoffMultiplier: number;
  readonly maxBackoffMs: number;
}

/** A named workflow comprising an ordered DAG of steps. */
export interface WorkflowDefinition {
  readonly name: string;
  readonly steps: readonly WorkflowStep[];
}

/** Runtime state of a workflow execution. */
export interface WorkflowExecution {
  readonly workflowId: string;
  readonly status: 'running' | 'completed' | 'failed';
  readonly completedSteps: readonly string[];
  readonly currentStep: string | null;
  readonly startedAt: number;
}

// ---------------------------------------------------------------------------
// HPC-9: Offload Decision
// ---------------------------------------------------------------------------

/** Description of a compute task for the offload decision engine. */
export interface ComputeTask {
  readonly type: string;
  readonly dataSize: number;
  readonly estimatedTimeMs: number;
  readonly requiresGPU: boolean;
  readonly memoryMB: number;
}

/**
 * The offload engine's decision on where to run a given compute task.
 * Considers latency, cost, GPU availability, and data transfer overhead.
 */
export interface OffloadDecision {
  readonly target: 'browser-gpu' | 'browser-wasm' | 'browser-js' | 'server-gpu' | 'edge';
  readonly reason: string;
  readonly estimatedLatencyMs: number;
  readonly estimatedCost: number;
}

/** Cost model parameters for the offload decision engine. */
export interface CostModel {
  readonly serverGPUCostPerMs: number;
  readonly edgeCostPerMs: number;
  readonly networkLatencyMs: number;
  readonly transferBytesPerMs: number;
}

// ---------------------------------------------------------------------------
// HPC-10: Profiling
// ---------------------------------------------------------------------------

/** A single profiling sample with optional key-value metadata. */
export interface ProfileSample {
  readonly name: string;
  readonly startMs: number;
  readonly durationMs: number;
  readonly metadata?: Record<string, number>;
}

/** Aggregated statistics for a named profiling category. */
export interface ProfileStatistics {
  readonly name: string;
  readonly count: number;
  readonly meanMs: number;
  readonly medianMs: number;
  readonly p95Ms: number;
  readonly p99Ms: number;
  readonly minMs: number;
  readonly maxMs: number;
  readonly stdDevMs: number;
}

/**
 * A detected performance antipattern with its measured value and threshold.
 * Severity levels guide whether automatic remediation should be attempted.
 */
export interface AntipatternReport {
  readonly name: string;
  readonly severity: 'info' | 'warning' | 'critical';
  readonly description: string;
  readonly metric: number;
  readonly threshold: number;
}

/** Well-known GPU/worker antipattern categories detected by the profiler. */
export type AntipatternType =
  | 'excessive_gpu_readback'
  | 'small_dispatch'
  | 'per_frame_pipeline'
  | 'hot_loop_allocation'
  | 'frequent_postmessage'
  | 'structured_clone_large'
  | 'sync_gpu_call';

// ---------------------------------------------------------------------------
// HPC-11: Deployment
// ---------------------------------------------------------------------------

/** Configuration for edge compute deployment targets. */
export interface EdgeConfig {
  readonly provider: 'cloudflare' | 'deno' | 'vercel' | 'custom';
  readonly wasmSizeLimit: number;
  readonly executionTimeLimit: number;
  readonly memoryLimit: number;
}

/** Strategy for loading WASM modules in the browser. */
export interface WASMLoadingStrategy {
  readonly streaming: boolean;
  readonly cacheInIDB: boolean;
  readonly preloadCritical: boolean;
  readonly lazyModules: readonly string[];
}

/** A stage in the progressive module loading waterfall. */
export interface ProgressiveLoadStage {
  readonly name: string;
  readonly modules: readonly string[];
  readonly priority: number;
  readonly sizeBytes: number;
}

// ---------------------------------------------------------------------------
// HPC-12: Numerical Linear Algebra
// ---------------------------------------------------------------------------

/** Floating-point precision selector. */
export type Precision = 'f32' | 'f64';

/** Configuration for an iterative linear solver. */
export interface SolverConfig {
  readonly precision: Precision;
  readonly maxIterations: number;
  readonly tolerance: number;
  readonly preconditioner?: 'none' | 'jacobi' | 'ilu' | 'amg';
}

/** Result of a linear solve, including convergence diagnostics. */
export interface SolverResult {
  readonly solution: Float64Array;
  readonly iterations: number;
  readonly residual: number;
  readonly converged: boolean;
}

/**
 * Mixed-precision iterative refinement configuration.
 * Factor in lower precision, refine in higher precision for speed + accuracy.
 */
export interface MixedPrecisionConfig {
  readonly factorPrecision: Precision;
  readonly refinePrecision: Precision;
  readonly maxRefinements: number;
  readonly targetResidual: number;
}

/**
 * Recommendation from the solver selection heuristic based on
 * detected matrix properties (symmetry, sparsity, condition number).
 */
export interface SolverRecommendation {
  readonly solver: 'direct_lu' | 'direct_cholesky' | 'cg' | 'gmres' | 'bicgstab' | 'minres';
  readonly precision: Precision;
  readonly reason: string;
  readonly estimatedTimeMs: number;
}

/** Structural and numerical properties of a matrix, used for solver selection. */
export interface MatrixProperties {
  readonly rows: number;
  readonly cols: number;
  readonly symmetric: boolean;
  readonly positiveDefinite: boolean;
  readonly sparse: boolean;
  readonly conditionNumber: number;
  /** Number of non-zero entries. */
  readonly nnz: number;
}
