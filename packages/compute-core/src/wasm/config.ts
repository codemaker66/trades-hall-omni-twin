// ---------------------------------------------------------------------------
// @omni-twin/compute-core — HPC-2: WASM Build Configuration
// ---------------------------------------------------------------------------
// Capability detection, build flag generation, and memory page calculation
// for the Rust -> WASM compilation pipeline.
// Pure logic — no actual WASM instantiation or file system access.
// ---------------------------------------------------------------------------

import type { WASMCapabilities, WASMModuleConfig } from '../types.js';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Size of a single WebAssembly memory page in bytes (64 KiB). */
const PAGE_SIZE = 65536;

// ---------------------------------------------------------------------------
// Capability detection
// ---------------------------------------------------------------------------

/**
 * Returns a conservative set of WASM capabilities based on known feature
 * availability across major browser engines.
 *
 * This is a pure-logic function that does not perform actual feature
 * detection (no `WebAssembly.validate` calls). It returns the baseline
 * capabilities that are safe to assume in modern browsers (2024+):
 * - SIMD: widely supported (Chrome 91+, Firefox 89+, Safari 16.4+)
 * - Bulk memory: universally supported
 * - Threads (SharedArrayBuffer): requires cross-origin isolation headers
 * - Exception handling: partial support
 * - GC: emerging, not yet reliable
 */
export function detectWASMCapabilities(): WASMCapabilities {
  return {
    simd: true,
    threads: false,       // Requires COOP/COEP headers; assume unavailable
    bulkMemory: true,
    exceptionHandling: false,
    gc: false,
  };
}

// ---------------------------------------------------------------------------
// Module size estimation
// ---------------------------------------------------------------------------

/**
 * Estimates the compiled WASM binary size in bytes based on configuration.
 *
 * Heuristic model:
 * - Base: 8 KiB (module overhead, type section, function table)
 * - SIMD: +12 KiB (wider instruction encoding, v128 types)
 * - Shared memory: +4 KiB (atomics section, shared memory flag)
 * - Memory: each initial page adds ~64 bytes of data segment overhead
 *
 * These are rough estimates for planning purposes (CDN budgets,
 * progressive loading thresholds). Actual sizes depend on Rust code
 * complexity and wasm-opt passes.
 */
export function estimateModuleSize(config: WASMModuleConfig): number {
  let size = 8 * 1024; // 8 KiB base

  if (config.simdRequired) {
    size += 12 * 1024;
  }
  if (config.sharedMemory) {
    size += 4 * 1024;
  }

  // Data segment overhead per initial page
  size += config.initialMemoryPages * 64;

  return size;
}

// ---------------------------------------------------------------------------
// Rust compiler flags
// ---------------------------------------------------------------------------

/**
 * Generates Rust/`wasm-pack` compiler flags corresponding to the module
 * configuration.
 *
 * Returned flags are suitable for passing to `RUSTFLAGS` or as
 * `wasm-pack build` arguments.
 */
export function buildFlags(config: WASMModuleConfig): string[] {
  const flags: string[] = [];

  // Target triple
  if (config.sharedMemory) {
    flags.push('--target=wasm32-unknown-unknown');
    flags.push('-C', 'target-feature=+atomics,+bulk-memory,+mutable-globals');
  } else {
    flags.push('--target=wasm32-unknown-unknown');
  }

  // SIMD
  if (config.simdRequired) {
    flags.push('-C', 'target-feature=+simd128');
  }

  // Optimisation (always release)
  flags.push('-C', 'opt-level=3');
  flags.push('-C', 'lto=thin');

  // Strip debug info for smaller binaries
  flags.push('-C', 'strip=debuginfo');

  return flags;
}

// ---------------------------------------------------------------------------
// Memory page calculation
// ---------------------------------------------------------------------------

/**
 * Calculates the optimal initial and maximum WebAssembly memory pages for
 * a given data size.
 *
 * Each WASM page is 64 KiB (65536 bytes). The initial allocation covers
 * the data with a 25% growth headroom. The maximum allows up to 4x the
 * initial allocation (capped at the WASM 4 GiB limit = 65536 pages).
 *
 * @param dataSize - Minimum required data size in bytes.
 */
export function optimalMemoryPages(
  dataSize: number,
): { initial: number; maximum: number } {
  if (dataSize <= 0) {
    return { initial: 1, maximum: 4 };
  }

  // Initial: cover data + 25% headroom, minimum 1 page
  const minPages = Math.ceil(dataSize / PAGE_SIZE);
  const initial = Math.max(1, Math.ceil(minPages * 1.25));

  // Maximum: 4x initial, capped at 65536 (4 GiB)
  const maximum = Math.min(initial * 4, 65536);

  return { initial, maximum };
}

/**
 * Returns the WebAssembly memory page size in bytes (65536).
 */
export function wasmPageSize(): number {
  return PAGE_SIZE;
}
