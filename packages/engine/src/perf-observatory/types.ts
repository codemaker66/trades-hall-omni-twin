/**
 * Types for the Performance Observatory.
 *
 * Tracks frame budget, memory, network, solver, and incremental graph metrics
 * in real-time with ring buffers for rolling window analysis.
 */

// ─── Frame Budget ────────────────────────────────────────────────────────────

export interface FrameSample {
  /** Frame timestamp (ms, from performance.now or similar) */
  readonly timestamp: number
  /** Total frame time in ms */
  readonly totalMs: number
  /** Render time in ms */
  readonly renderMs: number
  /** Physics/collision time in ms */
  readonly physicsMs: number
  /** CRDT sync time in ms */
  readonly crdtSyncMs: number
  /** UI update time in ms */
  readonly uiMs: number
  /** Idle time (budget remaining) in ms */
  readonly idleMs: number
}

// ─── Memory ──────────────────────────────────────────────────────────────────

export interface MemorySample {
  readonly timestamp: number
  /** JS heap used (bytes), from performance.memory if available */
  readonly jsHeapUsed: number
  /** JS heap total (bytes) */
  readonly jsHeapTotal: number
  /** GPU memory estimate (bytes): geometries + textures + buffers */
  readonly gpuMemory: number
  /** Count of geometries in scene */
  readonly geometryCount: number
  /** Count of textures in scene */
  readonly textureCount: number
}

// ─── Network ─────────────────────────────────────────────────────────────────

export interface NetworkSample {
  readonly timestamp: number
  /** Operations sent per second */
  readonly opsSentPerSec: number
  /** Operations received per second */
  readonly opsRecvPerSec: number
  /** Bytes sent per second */
  readonly bytesSentPerSec: number
  /** Bytes received per second */
  readonly bytesRecvPerSec: number
  /** Compression ratio (original / compressed) */
  readonly compressionRatio: number
  /** Round-trip latency in ms */
  readonly latencyMs: number
}

// ─── Solver ──────────────────────────────────────────────────────────────────

export interface SolverSample {
  readonly timestamp: number
  /** Total solver execution time in ms */
  readonly solveTimeMs: number
  /** Number of placement attempts */
  readonly placementAttempts: number
  /** Number of annealing iterations */
  readonly annealingIterations: number
  /** Final solution quality score (0–1) */
  readonly qualityScore: number
  /** Number of hard constraint violations remaining */
  readonly violations: number
  /** Items successfully placed */
  readonly itemsPlaced: number
  /** Total items requested */
  readonly itemsRequested: number
}

// ─── Incremental Graph ───────────────────────────────────────────────────────

export interface IncrementalSample {
  readonly timestamp: number
  /** Number of nodes stabilized in this cycle */
  readonly nodesStabilized: number
  /** Total nodes in graph */
  readonly totalNodes: number
  /** Cutoff ratio: fraction of nodes skipped (0–1) */
  readonly cutoffRatio: number
  /** Max propagation depth reached */
  readonly maxDepth: number
  /** Time to stabilize in ms */
  readonly stabilizeMs: number
}

// ─── Aggregate Metrics ───────────────────────────────────────────────────────

export interface FrameStats {
  readonly avgFps: number
  readonly minFps: number
  readonly maxFps: number
  readonly p99FrameMs: number
  readonly avgRenderMs: number
  readonly avgPhysicsMs: number
  readonly avgCrdtMs: number
  readonly avgUiMs: number
  readonly avgIdleMs: number
  readonly droppedFrames: number
}

export interface MemoryStats {
  readonly currentJsHeapMB: number
  readonly peakJsHeapMB: number
  readonly currentGpuMemoryMB: number
  readonly geometryCount: number
  readonly textureCount: number
  readonly leakSuspected: boolean
}

export interface NetworkStats {
  readonly avgOpsSentPerSec: number
  readonly avgOpsRecvPerSec: number
  readonly avgBytesSentPerSec: number
  readonly avgBytesRecvPerSec: number
  readonly avgCompressionRatio: number
  readonly avgLatencyMs: number
  readonly p99LatencyMs: number
}

export interface PerformanceSnapshot {
  readonly capturedAt: string
  readonly windowSize: number
  readonly frame: FrameStats
  readonly memory: MemoryStats
  readonly network: NetworkStats
  readonly solver: SolverSample | null
  readonly incremental: IncrementalSample | null
}
