// ---------------------------------------------------------------------------
// @omni-twin/compute-core — HPC-11: Compiled Module Cache Strategy
// ---------------------------------------------------------------------------
// Deterministic cache key generation, size estimation, invalidation logic,
// priority-based eviction, and hit-rate modeling for WASM compiled modules.
// ---------------------------------------------------------------------------

import type { WASMModuleConfig } from '../types.js';
import { murmurHash3_32 } from '../types.js';

// ---------------------------------------------------------------------------
// Cache key generation
// ---------------------------------------------------------------------------

/**
 * Computes a deterministic cache key from a module URL and version string.
 * Uses MurmurHash3 to produce a fixed-length hex key suitable for use as
 * an IndexedDB or Cache API key.
 */
export function computeCacheKey(url: string, version: string): string {
  const input = `${url}::${version}`;
  const hash = murmurHash3_32(input, 0x2a2a2a2a);
  return `wasm-${hash.toString(16).padStart(8, '0')}`;
}

// ---------------------------------------------------------------------------
// Size estimation
// ---------------------------------------------------------------------------

/**
 * Estimates the total cache size in bytes required to store compiled
 * representations of all given WASM modules.
 *
 * Compiled modules are typically 1.5-3x the size of their source WASM
 * binary. We use a conservative 2x multiplier applied to the linear memory
 * page footprint (each page = 64 KB).
 */
export function estimateCacheSize(modules: WASMModuleConfig[]): number {
  const PAGE_SIZE = 65536; // 64 KB per WASM memory page
  const COMPILED_MULTIPLIER = 2;

  let total = 0;
  for (let i = 0; i < modules.length; i++) {
    const m = modules[i]!;
    total += m.initialMemoryPages * PAGE_SIZE * COMPILED_MULTIPLIER;
  }
  return total;
}

// ---------------------------------------------------------------------------
// Cache invalidation
// ---------------------------------------------------------------------------

/**
 * Returns `true` if the cache should be invalidated because the current
 * version differs from the cached version.
 *
 * A simple string comparison: any difference triggers invalidation.
 */
export function shouldInvalidateCache(
  currentVersion: string,
  cachedVersion: string,
): boolean {
  return currentVersion !== cachedVersion;
}

// ---------------------------------------------------------------------------
// Eviction priority
// ---------------------------------------------------------------------------

/**
 * Computes a priority score for a cached module, used for eviction
 * decisions. Higher scores indicate higher value (should be kept longer).
 *
 * The score is based on:
 * - Hit count (more frequently used = higher priority)
 * - Memory page count (larger modules are more expensive to recompile)
 *
 * Formula: `hitCount * (1 + log2(initialMemoryPages + 1))`
 */
export function cachePriority(
  config: WASMModuleConfig,
  hitCount: number,
): number {
  if (hitCount <= 0) {
    return 0;
  }
  const pageFactor = 1 + Math.log2(config.initialMemoryPages + 1);
  return hitCount * pageFactor;
}

// ---------------------------------------------------------------------------
// Eviction selection
// ---------------------------------------------------------------------------

/**
 * Selects cache entries to evict in order to free at least `targetFreeBytes`
 * of space. Entries are evicted in ascending priority order (lowest priority
 * first) until the cumulative freed space meets the target.
 *
 * Returns an array of cache keys to remove.
 */
export function evictionCandidates(
  entries: Array<{ key: string; size: number; priority: number }>,
  targetFreeBytes: number,
): string[] {
  if (targetFreeBytes <= 0) {
    return [];
  }

  // Sort by priority ascending (lowest priority = first to evict)
  const sorted = entries.slice().sort((a, b) => a.priority - b.priority);

  const toEvict: string[] = [];
  let freed = 0;

  for (let i = 0; i < sorted.length; i++) {
    if (freed >= targetFreeBytes) {
      break;
    }
    const entry = sorted[i]!;
    toEvict.push(entry.key);
    freed += entry.size;
  }

  return toEvict;
}

// ---------------------------------------------------------------------------
// Hit-rate estimation
// ---------------------------------------------------------------------------

/**
 * Estimates the cache hit rate given total requests, unique module count,
 * and cache size (number of entries that can be stored).
 *
 * Uses a simplified model based on the independent reference model:
 * if the cache can hold all unique modules, hit rate approaches
 * `(totalRequests - uniqueModules) / totalRequests` (first access is always
 * a miss). Otherwise, it scales by the fraction of unique modules that fit.
 *
 * Returns a value in [0, 1].
 */
export function estimateCacheHitRate(
  totalRequests: number,
  uniqueModules: number,
  cacheSize: number,
): number {
  if (totalRequests <= 0 || uniqueModules <= 0 || cacheSize <= 0) {
    return 0;
  }

  // Fraction of unique modules that fit in the cache
  const coverage = Math.min(cacheSize / uniqueModules, 1);

  // Idealized miss rate: at least one miss per unique module (cold start)
  const coldMissRate = Math.min(uniqueModules / totalRequests, 1);

  // Hit rate = coverage * (1 - cold miss rate)
  const hitRate = coverage * (1 - coldMissRate);

  return Math.max(0, Math.min(hitRate, 1));
}
