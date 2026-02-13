import { describe, it, expect } from 'vitest';
import {
  defaultCapabilities,
  estimateWorkgroups,
  computeOptimalWorkgroupSize,
  validateDispatch,
  createPipelineRegistry,
  registerPipeline,
  getPipelineDescriptor,
  listPipelines,
  estimateCompilationTimeMs,
  planDispatches,
  estimateComputeTimeMs,
  aggregateTimestamps,
  computeGPUUtilization,
  REDUCE_WGSL,
  MATMUL_WGSL,
  PREFIX_SUM_WGSL,
  SINKHORN_ROW_WGSL,
  ISING_METROPOLIS_WGSL,
} from '../webgpu/index.js';

import type {
  GPUCapabilities,
  ComputeDispatch,
  ComputePipelineDescriptor,
  TimestampResult,
} from '../types.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeDispatch(
  pipeline: string,
  workgroups: readonly [number, number, number],
  buffers: ComputeDispatch['buffers'],
): ComputeDispatch {
  return { pipeline, workgroups, buffers };
}

function makeCaps(overrides?: Partial<GPUCapabilities>): GPUCapabilities {
  return { ...defaultCapabilities(), ...overrides };
}

// ---------------------------------------------------------------------------
// defaultCapabilities
// ---------------------------------------------------------------------------

describe('defaultCapabilities', () => {
  it('returns an object with all required GPUCapabilities fields', () => {
    const caps = defaultCapabilities();
    expect(caps).toHaveProperty('available');
    expect(caps).toHaveProperty('maxWorkgroupSize');
    expect(caps).toHaveProperty('maxWorkgroupsPerDimension');
    expect(caps).toHaveProperty('maxStorageBufferSize');
    expect(caps).toHaveProperty('maxWorkgroupStorageSize');
    expect(caps).toHaveProperty('f16Supported');
    expect(caps).toHaveProperty('timestampQuerySupported');
  });

  it('reports available as false for the conservative fallback', () => {
    expect(defaultCapabilities().available).toBe(false);
  });

  it('has maxWorkgroupSize of 256 (WebGPU spec minimum)', () => {
    expect(defaultCapabilities().maxWorkgroupSize).toBe(256);
  });

  it('has maxStorageBufferSize of 128 MiB', () => {
    expect(defaultCapabilities().maxStorageBufferSize).toBe(128 * 1024 * 1024);
  });

  it('has maxWorkgroupsPerDimension of 65535', () => {
    expect(defaultCapabilities().maxWorkgroupsPerDimension).toBe(65535);
  });
});

// ---------------------------------------------------------------------------
// estimateWorkgroups
// ---------------------------------------------------------------------------

describe('estimateWorkgroups', () => {
  it('returns ceil division for even split', () => {
    expect(estimateWorkgroups(256, 256)).toBe(1);
  });

  it('returns ceil division for uneven split', () => {
    expect(estimateWorkgroups(257, 256)).toBe(2);
    expect(estimateWorkgroups(1000, 256)).toBe(4);
  });

  it('returns 1 for degenerate totalItems <= 0', () => {
    expect(estimateWorkgroups(0, 256)).toBe(1);
    expect(estimateWorkgroups(-5, 256)).toBe(1);
  });

  it('returns 1 for degenerate workgroupSize <= 0', () => {
    expect(estimateWorkgroups(100, 0)).toBe(1);
    expect(estimateWorkgroups(100, -1)).toBe(1);
  });

  it('handles large totalItems correctly', () => {
    expect(estimateWorkgroups(1_000_000, 256)).toBe(Math.ceil(1_000_000 / 256));
  });
});

// ---------------------------------------------------------------------------
// computeOptimalWorkgroupSize
// ---------------------------------------------------------------------------

describe('computeOptimalWorkgroupSize', () => {
  it('returns a power of 2', () => {
    const size = computeOptimalWorkgroupSize(1000, 256);
    expect(size).toBe(256);
    expect(Math.log2(size) % 1).toBe(0);
  });

  it('returns largest power of 2 <= max for large dispatches', () => {
    expect(computeOptimalWorkgroupSize(10000, 1024)).toBe(1024);
    expect(computeOptimalWorkgroupSize(10000, 300)).toBe(256);
  });

  it('returns a smaller power of 2 for small dispatches', () => {
    expect(computeOptimalWorkgroupSize(10, 256)).toBe(16);
    expect(computeOptimalWorkgroupSize(1, 256)).toBe(1);
    expect(computeOptimalWorkgroupSize(5, 256)).toBe(8);
  });

  it('returns 1 for degenerate inputs', () => {
    expect(computeOptimalWorkgroupSize(0, 256)).toBe(1);
    expect(computeOptimalWorkgroupSize(-1, 256)).toBe(1);
    expect(computeOptimalWorkgroupSize(100, 0)).toBe(1);
  });

  it('never exceeds maxWorkgroupSize', () => {
    expect(computeOptimalWorkgroupSize(512, 64)).toBe(64);
  });
});

// ---------------------------------------------------------------------------
// validateDispatch
// ---------------------------------------------------------------------------

describe('validateDispatch', () => {
  it('returns no errors for a valid dispatch with available GPU', () => {
    const caps = makeCaps({ available: true });
    const dispatch = makeDispatch('test', [4, 1, 1], [
      { binding: 0, size: 1024, usage: 'storage' as const },
    ]);
    expect(validateDispatch(dispatch, caps)).toEqual([]);
  });

  it('reports error when GPU is not available', () => {
    const caps = makeCaps({ available: false });
    const dispatch = makeDispatch('test', [1, 1, 1], [
      { binding: 0, size: 1024, usage: 'storage' as const },
    ]);
    const errors = validateDispatch(dispatch, caps);
    expect(errors.length).toBeGreaterThanOrEqual(1);
    expect(errors.some((e) => e.includes('not available'))).toBe(true);
  });

  it('reports error for oversized workgroup dimensions', () => {
    const caps = makeCaps({ available: true, maxWorkgroupsPerDimension: 100 });
    const dispatch = makeDispatch('test', [200, 1, 1], [
      { binding: 0, size: 1024, usage: 'storage' as const },
    ]);
    const errors = validateDispatch(dispatch, caps);
    expect(errors.some((e) => e.includes('exceeds maximum'))).toBe(true);
  });

  it('reports error for buffer exceeding max storage buffer size', () => {
    const caps = makeCaps({ available: true, maxStorageBufferSize: 512 });
    const dispatch = makeDispatch('test', [1, 1, 1], [
      { binding: 0, size: 1024, usage: 'storage' as const },
    ]);
    const errors = validateDispatch(dispatch, caps);
    expect(errors.some((e) => e.includes('exceeds'))).toBe(true);
  });

  it('detects duplicate buffer binding indices', () => {
    const caps = makeCaps({ available: true });
    const dispatch = makeDispatch('test', [1, 1, 1], [
      { binding: 0, size: 128, usage: 'storage' as const },
      { binding: 0, size: 256, usage: 'uniform' as const },
    ]);
    const errors = validateDispatch(dispatch, caps);
    expect(errors.some((e) => e.includes('Duplicate'))).toBe(true);
  });

  it('reports error for zero or negative buffer size', () => {
    const caps = makeCaps({ available: true });
    const dispatch = makeDispatch('test', [1, 1, 1], [
      { binding: 0, size: 0, usage: 'storage' as const },
    ]);
    const errors = validateDispatch(dispatch, caps);
    expect(errors.some((e) => e.includes('positive size'))).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Pipeline registry CRUD
// ---------------------------------------------------------------------------

describe('pipeline registry', () => {
  const descriptorA: ComputePipelineDescriptor = {
    name: 'reduce',
    shaderSource: REDUCE_WGSL,
    entryPoint: 'main',
    workgroupSize: [256, 1, 1],
  };

  const descriptorB: ComputePipelineDescriptor = {
    name: 'matmul',
    shaderSource: MATMUL_WGSL,
    entryPoint: 'main',
    workgroupSize: [16, 16, 1],
  };

  it('creates an empty registry', () => {
    const reg = createPipelineRegistry();
    expect(listPipelines(reg)).toEqual([]);
  });

  it('registers and retrieves a pipeline descriptor', () => {
    const reg = createPipelineRegistry();
    registerPipeline(reg, descriptorA);
    const retrieved = getPipelineDescriptor(reg, 'reduce');
    expect(retrieved).not.toBeNull();
    expect(retrieved!.entryPoint).toBe('main');
  });

  it('returns null for an unknown pipeline name', () => {
    const reg = createPipelineRegistry();
    expect(getPipelineDescriptor(reg, 'nonexistent')).toBeNull();
  });

  it('lists pipeline names in sorted order', () => {
    const reg = createPipelineRegistry();
    registerPipeline(reg, descriptorB);
    registerPipeline(reg, descriptorA);
    expect(listPipelines(reg)).toEqual(['matmul', 'reduce']);
  });

  it('overwrites an existing descriptor with the same name', () => {
    const reg = createPipelineRegistry();
    registerPipeline(reg, descriptorA);
    const updated: ComputePipelineDescriptor = {
      ...descriptorA,
      entryPoint: 'mainV2',
    };
    registerPipeline(reg, updated);
    const retrieved = getPipelineDescriptor(reg, 'reduce');
    expect(retrieved!.entryPoint).toBe('mainV2');
    expect(listPipelines(reg)).toEqual(['reduce']);
  });
});

// ---------------------------------------------------------------------------
// estimateCompilationTimeMs
// ---------------------------------------------------------------------------

describe('estimateCompilationTimeMs', () => {
  it('returns at least 2ms for any shader (base cost)', () => {
    expect(estimateCompilationTimeMs('')).toBeGreaterThanOrEqual(2);
  });

  it('increases with shader source length', () => {
    const short = estimateCompilationTimeMs('fn main() {}');
    const long = estimateCompilationTimeMs('fn main() {}'.repeat(100));
    expect(long).toBeGreaterThan(short);
  });

  it('accounts for @compute and fn declarations', () => {
    const simple = estimateCompilationTimeMs('x');
    const withFns = estimateCompilationTimeMs('@compute fn main() {} fn helper() {}');
    expect(withFns).toBeGreaterThan(simple);
  });

  it('accounts for workgroupBarrier calls', () => {
    const base = estimateCompilationTimeMs('fn main() {}');
    const withBarriers = estimateCompilationTimeMs(
      'fn main() { workgroupBarrier(); workgroupBarrier(); }',
    );
    expect(withBarriers).toBeGreaterThan(base);
  });

  it('accounts for loop constructs', () => {
    const base = estimateCompilationTimeMs('fn main() {}');
    const withLoops = estimateCompilationTimeMs('fn main() { loop { } for (var i = 0; i < 10; ) {} }');
    expect(withLoops).toBeGreaterThan(base);
  });
});

// ---------------------------------------------------------------------------
// WGSL shader constants
// ---------------------------------------------------------------------------

describe('WGSL shader constants', () => {
  it('REDUCE_WGSL contains @compute and workgroupBarrier', () => {
    expect(REDUCE_WGSL).toContain('@compute');
    expect(REDUCE_WGSL).toContain('workgroupBarrier');
  });

  it('MATMUL_WGSL contains var<workgroup> for tiling', () => {
    expect(MATMUL_WGSL).toContain('var<workgroup>');
    expect(MATMUL_WGSL).toContain('tileA');
    expect(MATMUL_WGSL).toContain('tileB');
  });

  it('PREFIX_SUM_WGSL contains shared memory and @compute', () => {
    expect(PREFIX_SUM_WGSL).toContain('var<workgroup>');
    expect(PREFIX_SUM_WGSL).toContain('@compute');
  });

  it('SINKHORN_ROW_WGSL contains logsumexp-related code', () => {
    expect(SINKHORN_ROW_WGSL).toContain('exp');
    expect(SINKHORN_ROW_WGSL).toContain('log');
  });

  it('ISING_METROPOLIS_WGSL contains Metropolis acceptance and PCG hash', () => {
    expect(ISING_METROPOLIS_WGSL).toContain('pcg_hash');
    expect(ISING_METROPOLIS_WGSL).toContain('deltaE');
    expect(ISING_METROPOLIS_WGSL).toContain('@compute');
  });
});

// ---------------------------------------------------------------------------
// planDispatches
// ---------------------------------------------------------------------------

describe('planDispatches', () => {
  const caps = makeCaps({ available: true });

  it('filters out dispatches exceeding device workgroup limits', () => {
    const smallCaps = makeCaps({
      available: true,
      maxWorkgroupsPerDimension: 10,
    });
    const tasks: ComputeDispatch[] = [
      makeDispatch('a', [5, 1, 1], [{ binding: 0, size: 64, usage: 'storage' }]),
      makeDispatch('b', [20, 1, 1], [{ binding: 0, size: 64, usage: 'storage' }]),
    ];
    const result = planDispatches(tasks, smallCaps);
    expect(result.length).toBe(1);
    expect(result[0]!.pipeline).toBe('a');
  });

  it('groups dispatches by pipeline name', () => {
    const tasks: ComputeDispatch[] = [
      makeDispatch('b', [1, 1, 1], [{ binding: 0, size: 64, usage: 'storage' }]),
      makeDispatch('a', [1, 1, 1], [{ binding: 0, size: 64, usage: 'storage' }]),
      makeDispatch('b', [2, 1, 1], [{ binding: 0, size: 128, usage: 'storage' }]),
    ];
    const result = planDispatches(tasks, caps);
    expect(result[0]!.pipeline).toBe('a');
    expect(result[1]!.pipeline).toBe('b');
    expect(result[2]!.pipeline).toBe('b');
  });

  it('within same pipeline, sorts by total buffer size descending', () => {
    const tasks: ComputeDispatch[] = [
      makeDispatch('x', [1, 1, 1], [{ binding: 0, size: 100, usage: 'storage' }]),
      makeDispatch('x', [1, 1, 1], [{ binding: 0, size: 500, usage: 'storage' }]),
      makeDispatch('x', [1, 1, 1], [{ binding: 0, size: 300, usage: 'storage' }]),
    ];
    const result = planDispatches(tasks, caps);
    expect(result[0]!.buffers[0]!.size).toBe(500);
    expect(result[1]!.buffers[0]!.size).toBe(300);
    expect(result[2]!.buffers[0]!.size).toBe(100);
  });

  it('does not mutate the input array', () => {
    const tasks: ComputeDispatch[] = [
      makeDispatch('b', [1, 1, 1], [{ binding: 0, size: 64, usage: 'storage' }]),
      makeDispatch('a', [1, 1, 1], [{ binding: 0, size: 64, usage: 'storage' }]),
    ];
    const original = [...tasks];
    planDispatches(tasks, caps);
    expect(tasks[0]!.pipeline).toBe(original[0]!.pipeline);
    expect(tasks[1]!.pipeline).toBe(original[1]!.pipeline);
  });

  it('returns empty array when all dispatches exceed limits', () => {
    const tinyCaps = makeCaps({
      available: true,
      maxWorkgroupsPerDimension: 1,
      maxStorageBufferSize: 10,
    });
    const tasks: ComputeDispatch[] = [
      makeDispatch('a', [5, 1, 1], [{ binding: 0, size: 64, usage: 'storage' }]),
    ];
    expect(planDispatches(tasks, tinyCaps)).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// estimateComputeTimeMs
// ---------------------------------------------------------------------------

describe('estimateComputeTimeMs', () => {
  it('returns a positive number for any valid dispatch', () => {
    const caps = makeCaps({ available: true });
    const dispatch = makeDispatch('test', [4, 1, 1], [
      { binding: 0, size: 1024, usage: 'storage' },
    ]);
    expect(estimateComputeTimeMs(dispatch, caps)).toBeGreaterThan(0);
  });

  it('increases with more workgroups', () => {
    const caps = makeCaps({ available: true });
    const small = makeDispatch('test', [1, 1, 1], [
      { binding: 0, size: 1024, usage: 'storage' },
    ]);
    const large = makeDispatch('test', [100, 100, 1], [
      { binding: 0, size: 1024, usage: 'storage' },
    ]);
    expect(estimateComputeTimeMs(large, caps)).toBeGreaterThan(
      estimateComputeTimeMs(small, caps),
    );
  });

  it('scales by 2x for smaller GPUs', () => {
    const bigCaps = makeCaps({
      available: true,
      maxStorageBufferSize: 256 * 1024 * 1024,
    });
    const smallCaps = makeCaps({
      available: true,
      maxStorageBufferSize: 64 * 1024 * 1024,
    });
    const dispatch = makeDispatch('test', [10, 1, 1], [
      { binding: 0, size: 1024, usage: 'storage' },
    ]);
    const bigTime = estimateComputeTimeMs(dispatch, bigCaps);
    const smallTime = estimateComputeTimeMs(dispatch, smallCaps);
    expect(smallTime).toBeCloseTo(bigTime * 2, 10);
  });
});

// ---------------------------------------------------------------------------
// aggregateTimestamps
// ---------------------------------------------------------------------------

describe('aggregateTimestamps', () => {
  it('returns zero total for empty results', () => {
    const { totalNs, breakdown } = aggregateTimestamps([]);
    expect(totalNs).toBe(0);
    expect(breakdown.size).toBe(0);
  });

  it('sums total nanoseconds across all results', () => {
    const results: TimestampResult[] = [
      { pipelineName: 'a', durationNs: 100, workgroupCount: 1 },
      { pipelineName: 'b', durationNs: 200, workgroupCount: 2 },
      { pipelineName: 'a', durationNs: 50, workgroupCount: 1 },
    ];
    const { totalNs } = aggregateTimestamps(results);
    expect(totalNs).toBe(350);
  });

  it('produces correct per-pipeline breakdown', () => {
    const results: TimestampResult[] = [
      { pipelineName: 'reduce', durationNs: 100, workgroupCount: 1 },
      { pipelineName: 'matmul', durationNs: 200, workgroupCount: 2 },
      { pipelineName: 'reduce', durationNs: 50, workgroupCount: 1 },
    ];
    const { breakdown } = aggregateTimestamps(results);
    expect(breakdown.get('reduce')).toBe(150);
    expect(breakdown.get('matmul')).toBe(200);
  });

  it('breakdown values sum to totalNs', () => {
    const results: TimestampResult[] = [
      { pipelineName: 'a', durationNs: 111, workgroupCount: 1 },
      { pipelineName: 'b', durationNs: 222, workgroupCount: 2 },
      { pipelineName: 'c', durationNs: 333, workgroupCount: 3 },
    ];
    const { totalNs, breakdown } = aggregateTimestamps(results);
    let sum = 0;
    for (const v of breakdown.values()) sum += v;
    expect(sum).toBe(totalNs);
  });

  it('handles a single result', () => {
    const results: TimestampResult[] = [
      { pipelineName: 'solo', durationNs: 42, workgroupCount: 1 },
    ];
    const { totalNs, breakdown } = aggregateTimestamps(results);
    expect(totalNs).toBe(42);
    expect(breakdown.get('solo')).toBe(42);
  });
});

// ---------------------------------------------------------------------------
// computeGPUUtilization
// ---------------------------------------------------------------------------

describe('computeGPUUtilization', () => {
  it('returns 0 when frameBudgetMs is 0', () => {
    expect(computeGPUUtilization(1_000_000, 0)).toBe(0);
  });

  it('returns 0 when frameBudgetMs is negative', () => {
    expect(computeGPUUtilization(1_000_000, -1)).toBe(0);
  });

  it('computes correct ratio for a 60 FPS budget', () => {
    // 8.335ms of compute = 8_335_000 ns, budget = 16.67ms
    const util = computeGPUUtilization(8_335_000, 16.67);
    expect(util).toBeCloseTo(0.5, 1);
  });

  it('returns > 1 when compute exceeds frame budget', () => {
    // 20ms of compute in a 16.67ms budget
    const util = computeGPUUtilization(20_000_000, 16.67);
    expect(util).toBeGreaterThan(1);
  });

  it('returns 0 when no compute time', () => {
    expect(computeGPUUtilization(0, 16.67)).toBe(0);
  });
});
