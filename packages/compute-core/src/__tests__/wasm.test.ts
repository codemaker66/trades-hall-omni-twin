import { describe, it, expect } from 'vitest';
import {
  detectWASMCapabilities,
  estimateModuleSize,
  buildFlags,
  optimalMemoryPages,
  wasmPageSize,
  computeAlignment,
  bytesPerElement,
  regionOverlaps,
  allocateRegion,
  validateRegion,
  splitRegion,
  estimateLoadTimeMs,
  prioritizeModules,
  computeCompressionRatio,
  shouldPreload,
} from '../wasm/index.js';

import type { WASMModuleConfig, ZeroCopyRegion } from '../types.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeModuleConfig(
  overrides?: Partial<WASMModuleConfig>,
): WASMModuleConfig {
  return {
    url: 'test.wasm',
    simdRequired: false,
    sharedMemory: false,
    initialMemoryPages: 4,
    maximumMemoryPages: 16,
    ...overrides,
  };
}

function makeRegion(
  ptr: number,
  byteLength: number,
  dtype: ZeroCopyRegion['dtype'],
): ZeroCopyRegion {
  return { ptr, byteLength, dtype };
}

// ---------------------------------------------------------------------------
// wasmPageSize
// ---------------------------------------------------------------------------

describe('wasmPageSize', () => {
  it('returns 65536 (64 KiB)', () => {
    expect(wasmPageSize()).toBe(65536);
  });
});

// ---------------------------------------------------------------------------
// detectWASMCapabilities
// ---------------------------------------------------------------------------

describe('detectWASMCapabilities', () => {
  it('returns an object with all capability fields', () => {
    const caps = detectWASMCapabilities();
    expect(caps).toHaveProperty('simd');
    expect(caps).toHaveProperty('threads');
    expect(caps).toHaveProperty('bulkMemory');
    expect(caps).toHaveProperty('exceptionHandling');
    expect(caps).toHaveProperty('gc');
  });

  it('reports simd as true (widely supported)', () => {
    expect(detectWASMCapabilities().simd).toBe(true);
  });

  it('reports threads as false (requires COOP/COEP)', () => {
    expect(detectWASMCapabilities().threads).toBe(false);
  });

  it('reports bulkMemory as true', () => {
    expect(detectWASMCapabilities().bulkMemory).toBe(true);
  });

  it('reports gc as false (emerging, not reliable)', () => {
    expect(detectWASMCapabilities().gc).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// computeAlignment
// ---------------------------------------------------------------------------

describe('computeAlignment', () => {
  it('returns 1 for u8', () => {
    expect(computeAlignment('u8')).toBe(1);
  });

  it('returns 4 for i32', () => {
    expect(computeAlignment('i32')).toBe(4);
  });

  it('returns 4 for u32', () => {
    expect(computeAlignment('u32')).toBe(4);
  });

  it('returns 4 for f32', () => {
    expect(computeAlignment('f32')).toBe(4);
  });

  it('returns 8 for f64', () => {
    expect(computeAlignment('f64')).toBe(8);
  });
});

// ---------------------------------------------------------------------------
// bytesPerElement
// ---------------------------------------------------------------------------

describe('bytesPerElement', () => {
  it('returns 1 for u8', () => {
    expect(bytesPerElement('u8')).toBe(1);
  });

  it('returns 4 for f32', () => {
    expect(bytesPerElement('f32')).toBe(4);
  });

  it('returns 8 for f64', () => {
    expect(bytesPerElement('f64')).toBe(8);
  });

  it('returns 4 for i32 and u32', () => {
    expect(bytesPerElement('i32')).toBe(4);
    expect(bytesPerElement('u32')).toBe(4);
  });

  it('alignment matches bytes per element for all types', () => {
    const dtypes: ZeroCopyRegion['dtype'][] = ['u8', 'i32', 'u32', 'f32', 'f64'];
    for (const d of dtypes) {
      expect(computeAlignment(d)).toBe(bytesPerElement(d));
    }
  });
});

// ---------------------------------------------------------------------------
// regionOverlaps
// ---------------------------------------------------------------------------

describe('regionOverlaps', () => {
  it('detects overlapping regions', () => {
    const a = makeRegion(0, 100, 'u8');
    const b = makeRegion(50, 100, 'u8');
    expect(regionOverlaps(a, b)).toBe(true);
  });

  it('returns false for non-overlapping regions', () => {
    const a = makeRegion(0, 50, 'u8');
    const b = makeRegion(50, 50, 'u8');
    expect(regionOverlaps(a, b)).toBe(false);
  });

  it('returns false when region a has zero length', () => {
    const a = makeRegion(25, 0, 'u8');
    const b = makeRegion(0, 100, 'u8');
    expect(regionOverlaps(a, b)).toBe(false);
  });

  it('returns false when region b has zero length', () => {
    const a = makeRegion(0, 100, 'u8');
    const b = makeRegion(50, 0, 'u8');
    expect(regionOverlaps(a, b)).toBe(false);
  });

  it('detects overlap when one region contains the other', () => {
    const outer = makeRegion(0, 200, 'f32');
    const inner = makeRegion(40, 40, 'f32');
    expect(regionOverlaps(outer, inner)).toBe(true);
    expect(regionOverlaps(inner, outer)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// allocateRegion
// ---------------------------------------------------------------------------

describe('allocateRegion', () => {
  it('computes correct byteLength for f32', () => {
    const region = allocateRegion(0, 10, 'f32');
    expect(region.byteLength).toBe(40);
  });

  it('computes correct byteLength for f64', () => {
    const region = allocateRegion(0, 5, 'f64');
    expect(region.byteLength).toBe(40);
  });

  it('computes correct byteLength for u8', () => {
    const region = allocateRegion(0, 100, 'u8');
    expect(region.byteLength).toBe(100);
  });

  it('preserves the pointer value', () => {
    const region = allocateRegion(1024, 10, 'f32');
    expect(region.ptr).toBe(1024);
  });

  it('preserves the dtype value', () => {
    const region = allocateRegion(0, 10, 'i32');
    expect(region.dtype).toBe('i32');
  });
});

// ---------------------------------------------------------------------------
// validateRegion
// ---------------------------------------------------------------------------

describe('validateRegion', () => {
  it('returns no errors for a valid aligned region', () => {
    const region = makeRegion(0, 40, 'f32');
    const errors = validateRegion(region, 1024);
    expect(errors).toEqual([]);
  });

  it('reports error when region exceeds total memory', () => {
    const region = makeRegion(0, 200, 'u8');
    const errors = validateRegion(region, 100);
    expect(errors.some((e) => e.includes('exceeds total memory'))).toBe(true);
  });

  it('reports error for misaligned pointer', () => {
    const region = makeRegion(3, 4, 'f32');
    const errors = validateRegion(region, 1024);
    expect(errors.some((e) => e.includes('not aligned'))).toBe(true);
  });

  it('reports error for negative pointer', () => {
    const region = makeRegion(-1, 4, 'u8');
    const errors = validateRegion(region, 1024);
    expect(errors.some((e) => e.includes('non-negative'))).toBe(true);
  });

  it('reports error when byte length is not a multiple of element size', () => {
    const region = makeRegion(0, 5, 'f32');
    const errors = validateRegion(region, 1024);
    expect(errors.some((e) => e.includes('not a multiple'))).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// splitRegion
// ---------------------------------------------------------------------------

describe('splitRegion', () => {
  it('splits a region into correct number of chunks', () => {
    // 100 f32 elements = 400 bytes, split into chunks of 30 elements
    const region = makeRegion(0, 400, 'f32');
    const chunks = splitRegion(region, 30);
    // 100 / 30 = 4 chunks (30, 30, 30, 10)
    expect(chunks.length).toBe(4);
  });

  it('chunk byte lengths sum to original region byte length', () => {
    const region = makeRegion(0, 400, 'f32');
    const chunks = splitRegion(region, 30);
    const totalBytes = chunks.reduce((sum, c) => sum + c.byteLength, 0);
    expect(totalBytes).toBe(400);
  });

  it('returns original region when chunk size exceeds total elements', () => {
    const region = makeRegion(0, 40, 'f32');
    const chunks = splitRegion(region, 100);
    expect(chunks.length).toBe(1);
    expect(chunks[0]!.byteLength).toBe(40);
  });

  it('returns original region for chunkElements <= 0', () => {
    const region = makeRegion(0, 40, 'f32');
    expect(splitRegion(region, 0).length).toBe(1);
    expect(splitRegion(region, -5).length).toBe(1);
  });

  it('preserves dtype across all chunks', () => {
    const region = makeRegion(0, 80, 'f64');
    const chunks = splitRegion(region, 3);
    for (const chunk of chunks) {
      expect(chunk.dtype).toBe('f64');
    }
  });

  it('chunks have contiguous non-overlapping pointers', () => {
    const region = makeRegion(100, 400, 'f32');
    const chunks = splitRegion(region, 25);
    for (let i = 1; i < chunks.length; i++) {
      const prev = chunks[i - 1]!;
      const curr = chunks[i]!;
      expect(curr.ptr).toBe(prev.ptr + prev.byteLength);
    }
  });
});

// ---------------------------------------------------------------------------
// estimateModuleSize
// ---------------------------------------------------------------------------

describe('estimateModuleSize', () => {
  it('returns a positive number for any config', () => {
    const config = makeModuleConfig();
    expect(estimateModuleSize(config)).toBeGreaterThan(0);
  });

  it('base size is 8 KiB for plain module', () => {
    const config = makeModuleConfig({ initialMemoryPages: 0 });
    expect(estimateModuleSize(config)).toBe(8 * 1024);
  });

  it('adds 12 KiB for SIMD', () => {
    const plain = estimateModuleSize(makeModuleConfig({ initialMemoryPages: 0 }));
    const simd = estimateModuleSize(
      makeModuleConfig({ simdRequired: true, initialMemoryPages: 0 }),
    );
    expect(simd - plain).toBe(12 * 1024);
  });

  it('adds 4 KiB for shared memory', () => {
    const plain = estimateModuleSize(makeModuleConfig({ initialMemoryPages: 0 }));
    const shared = estimateModuleSize(
      makeModuleConfig({ sharedMemory: true, initialMemoryPages: 0 }),
    );
    expect(shared - plain).toBe(4 * 1024);
  });

  it('adds 64 bytes per initial memory page', () => {
    const base = estimateModuleSize(makeModuleConfig({ initialMemoryPages: 0 }));
    const with10 = estimateModuleSize(makeModuleConfig({ initialMemoryPages: 10 }));
    expect(with10 - base).toBe(10 * 64);
  });
});

// ---------------------------------------------------------------------------
// buildFlags
// ---------------------------------------------------------------------------

describe('buildFlags', () => {
  it('always includes --target=wasm32-unknown-unknown', () => {
    const flags = buildFlags(makeModuleConfig());
    expect(flags.some((f) => f.includes('wasm32-unknown-unknown'))).toBe(true);
  });

  it('includes atomics flags when sharedMemory is true', () => {
    const flags = buildFlags(makeModuleConfig({ sharedMemory: true }));
    expect(flags.some((f) => f.includes('atomics'))).toBe(true);
  });

  it('includes simd128 when simdRequired is true', () => {
    const flags = buildFlags(makeModuleConfig({ simdRequired: true }));
    expect(flags.some((f) => f.includes('simd128'))).toBe(true);
  });

  it('always includes opt-level=3', () => {
    const flags = buildFlags(makeModuleConfig());
    expect(flags.some((f) => f.includes('opt-level=3'))).toBe(true);
  });

  it('always includes strip=debuginfo', () => {
    const flags = buildFlags(makeModuleConfig());
    expect(flags.some((f) => f.includes('strip=debuginfo'))).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// optimalMemoryPages
// ---------------------------------------------------------------------------

describe('optimalMemoryPages', () => {
  it('returns { initial: 1, maximum: 4 } for zero or negative data size', () => {
    expect(optimalMemoryPages(0)).toEqual({ initial: 1, maximum: 4 });
    expect(optimalMemoryPages(-100)).toEqual({ initial: 1, maximum: 4 });
  });

  it('includes 25% headroom in initial pages', () => {
    // 65536 bytes = 1 page, with 25% headroom -> ceil(1 * 1.25) = 2
    const result = optimalMemoryPages(65536);
    expect(result.initial).toBe(2);
  });

  it('maximum is 4x initial', () => {
    const result = optimalMemoryPages(65536);
    expect(result.maximum).toBe(result.initial * 4);
  });

  it('caps maximum at 65536 pages (4 GiB)', () => {
    // Very large data that would exceed the 4 GiB limit
    const result = optimalMemoryPages(4 * 1024 * 1024 * 1024);
    expect(result.maximum).toBeLessThanOrEqual(65536);
  });

  it('initial is at least 1 for small data', () => {
    const result = optimalMemoryPages(1);
    expect(result.initial).toBeGreaterThanOrEqual(1);
  });
});

// ---------------------------------------------------------------------------
// estimateLoadTimeMs
// ---------------------------------------------------------------------------

describe('estimateLoadTimeMs', () => {
  it('returns 0 for zero or negative size', () => {
    expect(estimateLoadTimeMs(0, 1_000_000)).toBe(0);
    expect(estimateLoadTimeMs(-10, 1_000_000)).toBe(0);
  });

  it('returns Infinity for zero or negative bandwidth', () => {
    expect(estimateLoadTimeMs(1024, 0)).toBe(Infinity);
    expect(estimateLoadTimeMs(1024, -1)).toBe(Infinity);
  });

  it('includes a 5ms connection overhead', () => {
    // 0 transfer time would still have 5ms overhead, but size must be > 0
    // At very high bandwidth the transfer portion approaches 0
    const time = estimateLoadTimeMs(1, 1e15);
    expect(time).toBeGreaterThanOrEqual(5);
    expect(time).toBeCloseTo(5, 3);
  });

  it('transfer time scales linearly with size', () => {
    const bw = 1_000_000; // 1 MB/s
    const t1 = estimateLoadTimeMs(1_000_000, bw);
    const t2 = estimateLoadTimeMs(2_000_000, bw);
    // t2 - overhead should be ~2x (t1 - overhead)
    expect(t2 - 5).toBeCloseTo(2 * (t1 - 5), 5);
  });

  it('returns correct time for known values', () => {
    // 1 MB at 1 MB/s = 1000ms transfer + 5ms overhead = 1005ms
    expect(estimateLoadTimeMs(1_000_000, 1_000_000)).toBeCloseTo(1005, 5);
  });
});

// ---------------------------------------------------------------------------
// computeCompressionRatio
// ---------------------------------------------------------------------------

describe('computeCompressionRatio', () => {
  it('returns 1.0 for none', () => {
    expect(computeCompressionRatio('none')).toBe(1.0);
  });

  it('returns 0.45 for gzip', () => {
    expect(computeCompressionRatio('gzip')).toBe(0.45);
  });

  it('returns 0.35 for brotli', () => {
    expect(computeCompressionRatio('brotli')).toBe(0.35);
  });

  it('brotli compresses better than gzip', () => {
    expect(computeCompressionRatio('brotli')).toBeLessThan(
      computeCompressionRatio('gzip'),
    );
  });

  it('all ratios are in (0, 1]', () => {
    for (const fmt of ['none', 'gzip', 'brotli'] as const) {
      const ratio = computeCompressionRatio(fmt);
      expect(ratio).toBeGreaterThan(0);
      expect(ratio).toBeLessThanOrEqual(1);
    }
  });
});

// ---------------------------------------------------------------------------
// prioritizeModules
// ---------------------------------------------------------------------------

describe('prioritizeModules', () => {
  it('SIMD modules come before non-SIMD', () => {
    const modules: WASMModuleConfig[] = [
      makeModuleConfig({ url: 'plain.wasm', simdRequired: false }),
      makeModuleConfig({ url: 'simd.wasm', simdRequired: true }),
    ];
    const sorted = prioritizeModules(modules);
    expect(sorted[0]!.url).toBe('simd.wasm');
  });

  it('shared memory modules come before plain modules', () => {
    const modules: WASMModuleConfig[] = [
      makeModuleConfig({ url: 'plain.wasm', sharedMemory: false }),
      makeModuleConfig({ url: 'shared.wasm', sharedMemory: true }),
    ];
    const sorted = prioritizeModules(modules);
    expect(sorted[0]!.url).toBe('shared.wasm');
  });

  it('within same tier, smaller modules come first', () => {
    const modules: WASMModuleConfig[] = [
      makeModuleConfig({ url: 'big.wasm', initialMemoryPages: 100 }),
      makeModuleConfig({ url: 'small.wasm', initialMemoryPages: 4 }),
    ];
    const sorted = prioritizeModules(modules);
    expect(sorted[0]!.url).toBe('small.wasm');
  });

  it('does not mutate the input array', () => {
    const modules: WASMModuleConfig[] = [
      makeModuleConfig({ url: 'b.wasm', initialMemoryPages: 100 }),
      makeModuleConfig({ url: 'a.wasm', initialMemoryPages: 4 }),
    ];
    const originalFirst = modules[0]!.url;
    prioritizeModules(modules);
    expect(modules[0]!.url).toBe(originalFirst);
  });

  it('full priority order: SIMD > shared > plain, then by size', () => {
    const modules: WASMModuleConfig[] = [
      makeModuleConfig({ url: 'plain-big.wasm', initialMemoryPages: 100 }),
      makeModuleConfig({ url: 'simd-small.wasm', simdRequired: true, initialMemoryPages: 2 }),
      makeModuleConfig({ url: 'shared.wasm', sharedMemory: true, initialMemoryPages: 10 }),
      makeModuleConfig({ url: 'plain-small.wasm', initialMemoryPages: 4 }),
      makeModuleConfig({ url: 'simd-big.wasm', simdRequired: true, initialMemoryPages: 50 }),
    ];
    const sorted = prioritizeModules(modules);
    expect(sorted[0]!.url).toBe('simd-small.wasm');
    expect(sorted[1]!.url).toBe('simd-big.wasm');
    expect(sorted[2]!.url).toBe('shared.wasm');
    expect(sorted[3]!.url).toBe('plain-small.wasm');
    expect(sorted[4]!.url).toBe('plain-big.wasm');
  });
});

// ---------------------------------------------------------------------------
// shouldPreload
// ---------------------------------------------------------------------------

describe('shouldPreload', () => {
  it('always preloads critical-path modules', () => {
    const config = makeModuleConfig({ initialMemoryPages: 1000 });
    expect(shouldPreload(config, true)).toBe(true);
  });

  it('preloads SIMD modules even when not critical', () => {
    const config = makeModuleConfig({ simdRequired: true, initialMemoryPages: 1000 });
    expect(shouldPreload(config, false)).toBe(true);
  });

  it('preloads small modules (<= 16 pages) even when not critical', () => {
    const config = makeModuleConfig({ initialMemoryPages: 16 });
    expect(shouldPreload(config, false)).toBe(true);
  });

  it('does not preload large non-critical modules', () => {
    const config = makeModuleConfig({ initialMemoryPages: 100 });
    expect(shouldPreload(config, false)).toBe(false);
  });

  it('preloads a 1-page non-critical module', () => {
    const config = makeModuleConfig({ initialMemoryPages: 1 });
    expect(shouldPreload(config, false)).toBe(true);
  });
});
