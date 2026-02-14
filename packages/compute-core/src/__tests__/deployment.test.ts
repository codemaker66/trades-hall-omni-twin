import { describe, it, expect } from 'vitest';
import {
  createEdgeConfig,
  validateEdgeDeployment,
  fitsEdge,
  selectEdgeProvider,
  createLoadingStrategy,
  buildLoadingPlan,
  estimateInitialLoadMs,
  estimateTotalLoadMs,
  shouldStreamCompile,
  compressionSavings,
  computeCacheKey,
  estimateCacheSize,
  shouldInvalidateCache,
  cachePriority,
  evictionCandidates,
  estimateCacheHitRate,
} from '../deployment/index.js';

import type {
  EdgeConfig,
  WASMModuleConfig,
  ProgressiveLoadStage,
} from '../types.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const MB = 1024 * 1024;

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

function makeStage(
  overrides?: Partial<ProgressiveLoadStage>,
): ProgressiveLoadStage {
  return {
    name: 'stage-a',
    modules: ['a.wasm'],
    priority: 1,
    sizeBytes: 10000,
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// Edge Config
// ---------------------------------------------------------------------------

describe('Edge Config', () => {
  it('createEdgeConfig returns correct limits for cloudflare', () => {
    const config = createEdgeConfig('cloudflare');
    expect(config.provider).toBe('cloudflare');
    expect(config.wasmSizeLimit).toBe(10 * MB);
    expect(config.executionTimeLimit).toBe(30_000);
    expect(config.memoryLimit).toBe(128 * MB);
  });

  it('createEdgeConfig returns correct limits for deno', () => {
    const config = createEdgeConfig('deno');
    expect(config.provider).toBe('deno');
    expect(config.wasmSizeLimit).toBe(20 * MB);
    expect(config.executionTimeLimit).toBe(50_000);
    expect(config.memoryLimit).toBe(512 * MB);
  });

  it('createEdgeConfig returns correct limits for vercel', () => {
    const config = createEdgeConfig('vercel');
    expect(config.provider).toBe('vercel');
    expect(config.wasmSizeLimit).toBe(10 * MB);
    expect(config.executionTimeLimit).toBe(10_000);
  });

  it('validateEdgeDeployment catches oversized WASM module', () => {
    const config = createEdgeConfig('cloudflare');
    const violations = validateEdgeDeployment(config, 15 * MB, 1000, 64);
    expect(violations.length).toBeGreaterThanOrEqual(1);
    expect(violations[0]!).toContain('WASM module size');
  });

  it('validateEdgeDeployment catches excess execution time', () => {
    const config = createEdgeConfig('vercel');
    const violations = validateEdgeDeployment(config, 1 * MB, 20_000, 64);
    expect(violations.length).toBeGreaterThanOrEqual(1);
    expect(violations.some((v) => v.includes('execution time'))).toBe(true);
  });

  it('validateEdgeDeployment returns empty array when all constraints met', () => {
    const config = createEdgeConfig('deno');
    const violations = validateEdgeDeployment(config, 5 * MB, 10_000, 128);
    expect(violations).toEqual([]);
  });

  it('fitsEdge returns true when within limits', () => {
    const config = createEdgeConfig('cloudflare');
    expect(fitsEdge(config, 5 * MB, 10_000, 64)).toBe(true);
  });

  it('fitsEdge returns false when over size limit', () => {
    const config = createEdgeConfig('cloudflare');
    expect(fitsEdge(config, 15 * MB, 1000, 64)).toBe(false);
  });

  it('selectEdgeProvider picks the first provider that fits', () => {
    const configs: EdgeConfig[] = [
      createEdgeConfig('vercel'),
      createEdgeConfig('cloudflare'),
      createEdgeConfig('deno'),
    ];
    // 15 MB module: too big for vercel/cloudflare (10 MB), fits deno (20 MB)
    const selected = selectEdgeProvider(configs, 15 * MB, 5_000, 128);
    expect(selected).not.toBeNull();
    expect(selected!.provider).toBe('deno');
  });

  it('selectEdgeProvider returns null when nothing fits', () => {
    const configs: EdgeConfig[] = [createEdgeConfig('vercel')];
    const selected = selectEdgeProvider(configs, 100 * MB, 100_000, 2048);
    expect(selected).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// WASM Loader
// ---------------------------------------------------------------------------

describe('WASM Loader', () => {
  it('createLoadingStrategy enables streaming for large modules', () => {
    const modules = [makeModuleConfig({ initialMemoryPages: 2 })];
    const strategy = createLoadingStrategy(modules);
    // 2 pages * 65536 = 131072 > 4096 threshold
    expect(strategy.streaming).toBe(true);
    expect(strategy.preloadCritical).toBe(true);
  });

  it('createLoadingStrategy marks second+ modules as lazy', () => {
    const modules = [
      makeModuleConfig({ url: 'core.wasm' }),
      makeModuleConfig({ url: 'extra.wasm' }),
    ];
    const strategy = createLoadingStrategy(modules);
    expect(strategy.lazyModules).toEqual(['extra.wasm']);
  });

  it('createLoadingStrategy for empty modules returns all-false strategy', () => {
    const strategy = createLoadingStrategy([]);
    expect(strategy.streaming).toBe(false);
    expect(strategy.cacheInIDB).toBe(false);
    expect(strategy.preloadCritical).toBe(false);
    expect(strategy.lazyModules).toEqual([]);
  });

  it('buildLoadingPlan orders stages by descending priority', () => {
    const stages = [
      makeStage({ name: 'low', priority: 1 }),
      makeStage({ name: 'high', priority: 10 }),
      makeStage({ name: 'mid', priority: 5 }),
    ];
    const plan = buildLoadingPlan(stages);
    expect(plan[0]!.name).toBe('high');
    expect(plan[1]!.name).toBe('mid');
    expect(plan[2]!.name).toBe('low');
  });

  it('estimateInitialLoadMs returns positive value for non-empty stages', () => {
    const stages = [makeStage({ sizeBytes: 100_000, priority: 1 })];
    const ms = estimateInitialLoadMs(stages, 1_000_000); // 1 MB/s
    expect(ms).toBeGreaterThan(0);
    // 100_000 / 1_000_000 * 1000 = 100 ms
    expect(ms).toBeCloseTo(100, 5);
  });

  it('estimateTotalLoadMs sums all stages', () => {
    const stages = [
      makeStage({ sizeBytes: 50_000, priority: 2 }),
      makeStage({ sizeBytes: 50_000, priority: 1 }),
    ];
    const ms = estimateTotalLoadMs(stages, 1_000_000);
    // (50000 + 50000) / 1000000 * 1000 = 100 ms
    expect(ms).toBeCloseTo(100, 5);
  });

  it('shouldStreamCompile returns true for large modules, false for small', () => {
    expect(shouldStreamCompile(10_000)).toBe(true);
    expect(shouldStreamCompile(4096)).toBe(false);
    expect(shouldStreamCompile(100)).toBe(false);
  });

  it('compressionSavings: brotli < gzip < none', () => {
    const size = 1_000_000;
    const none = compressionSavings(size, 'none');
    const gzip = compressionSavings(size, 'gzip');
    const brotli = compressionSavings(size, 'brotli');
    expect(none).toBe(size);
    expect(gzip).toBeLessThan(none);
    expect(brotli).toBeLessThan(gzip);
    // gzip ratio 0.35
    expect(gzip).toBe(Math.round(size * 0.35));
    // brotli ratio 0.28
    expect(brotli).toBe(Math.round(size * 0.28));
  });
});

// ---------------------------------------------------------------------------
// Cache Strategy
// ---------------------------------------------------------------------------

describe('Cache Strategy', () => {
  it('computeCacheKey is deterministic for same inputs', () => {
    const key1 = computeCacheKey('https://cdn.example.com/mod.wasm', '1.0.0');
    const key2 = computeCacheKey('https://cdn.example.com/mod.wasm', '1.0.0');
    expect(key1).toBe(key2);
    expect(key1.startsWith('wasm-')).toBe(true);
  });

  it('computeCacheKey differs for different versions', () => {
    const key1 = computeCacheKey('mod.wasm', '1.0.0');
    const key2 = computeCacheKey('mod.wasm', '2.0.0');
    expect(key1).not.toBe(key2);
  });

  it('estimateCacheSize computes 2x the page footprint', () => {
    const modules = [makeModuleConfig({ initialMemoryPages: 4 })];
    const size = estimateCacheSize(modules);
    // 4 pages * 65536 bytes * 2x multiplier
    expect(size).toBe(4 * 65536 * 2);
  });

  it('shouldInvalidateCache returns true on version change', () => {
    expect(shouldInvalidateCache('2.0.0', '1.0.0')).toBe(true);
  });

  it('shouldInvalidateCache returns false when versions match', () => {
    expect(shouldInvalidateCache('1.0.0', '1.0.0')).toBe(false);
  });

  it('cachePriority combines hit count with page factor', () => {
    const config = makeModuleConfig({ initialMemoryPages: 8 });
    const prio = cachePriority(config, 10);
    expect(prio).toBeGreaterThan(0);
    // hitCount * (1 + log2(8 + 1)) = 10 * (1 + ~3.17) = ~41.7
    expect(prio).toBeCloseTo(10 * (1 + Math.log2(9)), 5);
  });

  it('cachePriority returns 0 for zero hits', () => {
    expect(cachePriority(makeModuleConfig(), 0)).toBe(0);
  });

  it('evictionCandidates removes lowest priority first', () => {
    const entries = [
      { key: 'high', size: 100, priority: 10 },
      { key: 'low', size: 100, priority: 1 },
      { key: 'mid', size: 100, priority: 5 },
    ];
    const toEvict = evictionCandidates(entries, 150);
    // Should evict 'low' (prio 1) then 'mid' (prio 5) = freed 200 >= 150
    expect(toEvict).toEqual(['low', 'mid']);
  });

  it('evictionCandidates returns empty for targetFreeBytes <= 0', () => {
    const entries = [{ key: 'a', size: 100, priority: 1 }];
    expect(evictionCandidates(entries, 0)).toEqual([]);
  });

  it('estimateCacheHitRate returns value in [0, 1]', () => {
    const rate = estimateCacheHitRate(1000, 10, 10);
    expect(rate).toBeGreaterThanOrEqual(0);
    expect(rate).toBeLessThanOrEqual(1);
  });

  it('estimateCacheHitRate is 0 for no requests', () => {
    expect(estimateCacheHitRate(0, 10, 10)).toBe(0);
  });

  it('estimateCacheHitRate increases with more cache capacity', () => {
    const smallCache = estimateCacheHitRate(1000, 100, 10);
    const largeCache = estimateCacheHitRate(1000, 100, 100);
    expect(largeCache).toBeGreaterThanOrEqual(smallCache);
  });
});
