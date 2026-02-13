// ---------------------------------------------------------------------------
// @omni-twin/compute-core — HPC-11: Progressive WASM Loading Strategy
// ---------------------------------------------------------------------------
// Derives loading strategies from module configurations, builds prioritized
// loading plans, and estimates load times under varying bandwidth conditions.
// ---------------------------------------------------------------------------

import type { WASMModuleConfig, WASMLoadingStrategy, ProgressiveLoadStage } from '../types.js';

// ---------------------------------------------------------------------------
// Streaming compilation threshold (bytes)
// ---------------------------------------------------------------------------

/** Modules larger than this benefit from streaming compilation. */
const STREAMING_THRESHOLD = 4096;

// ---------------------------------------------------------------------------
// Strategy derivation
// ---------------------------------------------------------------------------

/**
 * Derives a WASM loading strategy from a set of module configurations.
 *
 * - Streaming is enabled if any module exceeds the streaming threshold.
 * - IndexedDB caching is enabled if there are 2+ modules or any is large.
 * - The first module is always preloaded as critical; the rest are lazy.
 */
export function createLoadingStrategy(
  modules: WASMModuleConfig[],
): WASMLoadingStrategy {
  if (modules.length === 0) {
    return {
      streaming: false,
      cacheInIDB: false,
      preloadCritical: false,
      lazyModules: [],
    };
  }

  const totalPages = modules.reduce(
    (sum, m) => sum + m.initialMemoryPages,
    0,
  );
  const largestModulePages = modules.reduce(
    (max, m) => Math.max(max, m.initialMemoryPages),
    0,
  );

  // Streaming compilation is worthwhile for non-trivial modules
  const anyLargeEnough = largestModulePages * 65536 > STREAMING_THRESHOLD;

  // Cache in IDB when we have multiple modules or significant total memory
  const shouldCache = modules.length >= 2 || totalPages > 16;

  // First module is critical; rest are lazy
  const lazyModules = modules.slice(1).map((m) => m.url);

  return {
    streaming: anyLargeEnough,
    cacheInIDB: shouldCache,
    preloadCritical: true,
    lazyModules,
  };
}

// ---------------------------------------------------------------------------
// Loading plan
// ---------------------------------------------------------------------------

/**
 * Builds a prioritized loading plan by sorting stages in descending priority
 * order and computing cumulative byte sizes.
 *
 * Returns a new array of stages (does not mutate the input).
 */
export function buildLoadingPlan(
  stages: ProgressiveLoadStage[],
): ProgressiveLoadStage[] {
  // Sort by priority descending (highest priority first)
  const sorted = stages.slice().sort((a, b) => b.priority - a.priority);
  return sorted;
}

// ---------------------------------------------------------------------------
// Load time estimation
// ---------------------------------------------------------------------------

/**
 * Estimates the time in milliseconds to load the first (highest-priority)
 * stage, given a network bandwidth in bytes per second.
 *
 * Returns 0 if there are no stages or bandwidth is non-positive.
 */
export function estimateInitialLoadMs(
  stages: ProgressiveLoadStage[],
  bandwidthBps: number,
): number {
  if (stages.length === 0 || bandwidthBps <= 0) {
    return 0;
  }

  // Build plan to determine which stage loads first
  const plan = buildLoadingPlan(stages);
  const firstStage = plan[0]!;
  return (firstStage.sizeBytes / bandwidthBps) * 1000;
}

/**
 * Estimates the total time in milliseconds to sequentially load all
 * stages, given a network bandwidth in bytes per second.
 *
 * Returns 0 if there are no stages or bandwidth is non-positive.
 */
export function estimateTotalLoadMs(
  stages: ProgressiveLoadStage[],
  bandwidthBps: number,
): number {
  if (stages.length === 0 || bandwidthBps <= 0) {
    return 0;
  }

  const totalBytes = stages.reduce((sum, s) => sum + s.sizeBytes, 0);
  return (totalBytes / bandwidthBps) * 1000;
}

// ---------------------------------------------------------------------------
// Streaming compilation decision
// ---------------------------------------------------------------------------

/**
 * Returns `true` if the module is large enough to benefit from
 * streaming compilation (> 4 KB).
 *
 * For very small modules the overhead of setting up a streaming decode
 * pipeline outweighs the benefit of incremental compilation.
 */
export function shouldStreamCompile(moduleSize: number): boolean {
  return moduleSize > STREAMING_THRESHOLD;
}

// ---------------------------------------------------------------------------
// Compression estimation
// ---------------------------------------------------------------------------

/**
 * Estimates the compressed size of a WASM module in bytes given the
 * original size and a compression format.
 *
 * Typical savings:
 * - **gzip**: ~65% reduction (ratio 0.35)
 * - **brotli**: ~72% reduction (ratio 0.28)
 * - **none**: no compression
 */
export function compressionSavings(
  sizeBytes: number,
  format: 'none' | 'gzip' | 'brotli',
): number {
  if (sizeBytes <= 0) {
    return 0;
  }

  switch (format) {
    case 'gzip':
      return Math.round(sizeBytes * 0.35);
    case 'brotli':
      return Math.round(sizeBytes * 0.28);
    case 'none':
      return sizeBytes;
  }
}
