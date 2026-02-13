// ---------------------------------------------------------------------------
// @omni-twin/compute-core — HPC-2: Progressive WASM Module Loader
// ---------------------------------------------------------------------------
// Utilities for estimating load times, prioritizing modules for progressive
// loading, compression ratio modeling, and preload decision-making.
// Pure logic — does not perform actual network requests or WASM instantiation.
// ---------------------------------------------------------------------------

import type { WASMModuleConfig } from '../types.js';

// ---------------------------------------------------------------------------
// Load time estimation
// ---------------------------------------------------------------------------

/**
 * Estimates the time in milliseconds to load a WASM module of the given
 * byte size over a connection with the specified bandwidth.
 *
 * The model adds a fixed 5ms overhead for connection setup / cache lookup
 * on top of the raw transfer time.
 *
 * @param sizeBytes    - Module size in bytes.
 * @param bandwidthBps - Available bandwidth in bytes per second.
 * @returns Estimated load time in milliseconds.
 */
export function estimateLoadTimeMs(
  sizeBytes: number,
  bandwidthBps: number,
): number {
  if (sizeBytes <= 0) {
    return 0;
  }
  if (bandwidthBps <= 0) {
    return Infinity;
  }

  const connectionOverheadMs = 5;
  const transferTimeMs = (sizeBytes / bandwidthBps) * 1000;

  return connectionOverheadMs + transferTimeMs;
}

// ---------------------------------------------------------------------------
// Module prioritisation
// ---------------------------------------------------------------------------

/**
 * Sorts WASM modules for optimal progressive loading order.
 *
 * Priority heuristic (highest priority first):
 * 1. Modules requiring SIMD are loaded first (they enable critical compute
 *    paths and their features cannot be polyfilled).
 * 2. Modules with shared memory are next (enables worker parallelism).
 * 3. Smaller modules are preferred within the same priority tier (faster
 *    time-to-interactive).
 *
 * The input array is not mutated; a new sorted array is returned.
 */
export function prioritizeModules(
  modules: WASMModuleConfig[],
): WASMModuleConfig[] {
  return modules.slice().sort((a, b) => {
    // Priority tier: SIMD > shared memory > plain
    const tierA = moduleTier(a);
    const tierB = moduleTier(b);

    if (tierA !== tierB) {
      return tierA - tierB; // lower tier number = higher priority
    }

    // Within the same tier, prefer smaller modules (faster load)
    const sizeA = a.initialMemoryPages;
    const sizeB = b.initialMemoryPages;
    return sizeA - sizeB;
  });
}

/**
 * Returns a numeric priority tier for sorting.
 * Lower number = higher priority.
 */
function moduleTier(config: WASMModuleConfig): number {
  if (config.simdRequired) return 0;
  if (config.sharedMemory) return 1;
  return 2;
}

// ---------------------------------------------------------------------------
// Compression ratios
// ---------------------------------------------------------------------------

/**
 * Returns the typical compression ratio for a given format when applied
 * to WASM binary data.
 *
 * Values represent the compressed-to-original size ratio (lower = better
 * compression). Based on empirical measurements across typical WASM
 * binaries:
 * - `'none'`:   1.0 (no compression)
 * - `'gzip'`:   ~0.45 (45% of original — zlib level 6 typical)
 * - `'brotli'`: ~0.35 (35% of original — brotli quality 6 typical)
 */
export function computeCompressionRatio(
  format: 'none' | 'gzip' | 'brotli',
): number {
  switch (format) {
    case 'none':
      return 1.0;
    case 'gzip':
      return 0.45;
    case 'brotli':
      return 0.35;
  }
}

// ---------------------------------------------------------------------------
// Preload decisions
// ---------------------------------------------------------------------------

/**
 * Determines whether a WASM module should be preloaded during initial
 * page load versus loaded lazily on demand.
 *
 * Decision criteria:
 * - Always preload if the module is on the critical path.
 * - Preload SIMD modules (they enable core compute features).
 * - Preload small modules (initial pages <= 16, i.e., <= 1 MiB) even
 *   if not critical, since the cost is low.
 * - Do not preload large non-critical modules (saves bandwidth).
 *
 * @param config       - Module configuration.
 * @param criticalPath - Whether this module is required before first render.
 */
export function shouldPreload(
  config: WASMModuleConfig,
  criticalPath: boolean,
): boolean {
  // Critical-path modules are always preloaded
  if (criticalPath) {
    return true;
  }

  // SIMD modules enable core compute — preload them
  if (config.simdRequired) {
    return true;
  }

  // Small modules (<=1 MiB initial memory) are cheap to preload
  if (config.initialMemoryPages <= 16) {
    return true;
  }

  // Large non-critical modules: lazy load
  return false;
}
