// ---------------------------------------------------------------------------
// @omni-twin/compute-core — HPC-1: Pipeline Cache Management
// ---------------------------------------------------------------------------
// Registry for compute pipeline descriptors. Provides lookup, listing, and
// a heuristic compilation time estimator based on shader source complexity.
// ---------------------------------------------------------------------------

import type { ComputePipelineDescriptor } from '../types.js';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** A registry of named compute pipeline descriptors. */
export interface PipelineRegistry {
  readonly descriptors: Map<string, ComputePipelineDescriptor>;
}

// ---------------------------------------------------------------------------
// Registry lifecycle
// ---------------------------------------------------------------------------

/**
 * Creates an empty pipeline registry.
 */
export function createPipelineRegistry(): PipelineRegistry {
  return { descriptors: new Map() };
}

/**
 * Registers a compute pipeline descriptor under its `name`.
 * Overwrites any existing descriptor with the same name.
 */
export function registerPipeline(
  registry: PipelineRegistry,
  descriptor: ComputePipelineDescriptor,
): void {
  registry.descriptors.set(descriptor.name, descriptor);
}

/**
 * Retrieves the pipeline descriptor registered under `name`, or `null` if
 * no such pipeline exists.
 */
export function getPipelineDescriptor(
  registry: PipelineRegistry,
  name: string,
): ComputePipelineDescriptor | null {
  return registry.descriptors.get(name) ?? null;
}

/**
 * Returns a sorted list of all registered pipeline names.
 */
export function listPipelines(registry: PipelineRegistry): string[] {
  return Array.from(registry.descriptors.keys()).sort();
}

// ---------------------------------------------------------------------------
// Compilation time heuristic
// ---------------------------------------------------------------------------

/**
 * Estimates the shader compilation time in milliseconds based on source
 * complexity heuristics.
 *
 * Factors considered:
 * - Total source length (proxy for instruction count)
 * - Number of function definitions (control flow complexity)
 * - Presence of shared memory (`var<workgroup>`)
 * - Number of barrier calls (synchronization overhead)
 * - Number of loop constructs
 *
 * These are rough estimates based on observed WebGPU driver behavior.
 * Actual times vary dramatically across GPU vendors.
 */
export function estimateCompilationTimeMs(shaderSource: string): number {
  // Base cost: ~2ms minimum for any shader
  let estimate = 2.0;

  // Source length: ~0.005ms per character (heuristic)
  estimate += shaderSource.length * 0.005;

  // Function definitions add control flow graph complexity
  const fnCount = countOccurrences(shaderSource, '@compute') +
    countOccurrences(shaderSource, 'fn ');
  estimate += fnCount * 1.5;

  // Workgroup shared memory requires additional register allocation
  const sharedCount = countOccurrences(shaderSource, 'var<workgroup>');
  estimate += sharedCount * 2.0;

  // Barriers add synchronization analysis overhead
  const barrierCount = countOccurrences(shaderSource, 'workgroupBarrier');
  estimate += barrierCount * 0.8;

  // Loops increase analysis time (unrolling, bounds checking)
  const loopCount = countOccurrences(shaderSource, 'loop {') +
    countOccurrences(shaderSource, 'for (') +
    countOccurrences(shaderSource, 'for(');
  estimate += loopCount * 1.2;

  return Math.round(estimate * 100) / 100;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Counts non-overlapping occurrences of `needle` in `haystack`.
 */
function countOccurrences(haystack: string, needle: string): number {
  let count = 0;
  let pos = 0;
  while (true) {
    const idx = haystack.indexOf(needle, pos);
    if (idx === -1) break;
    count++;
    pos = idx + needle.length;
  }
  return count;
}
