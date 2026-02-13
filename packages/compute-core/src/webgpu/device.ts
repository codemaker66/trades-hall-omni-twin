// ---------------------------------------------------------------------------
// @omni-twin/compute-core — HPC-1: WebGPU Device Capability Detection
// ---------------------------------------------------------------------------
// Pure-logic GPU capability detection, workgroup sizing, and dispatch
// validation. Does not call the WebGPU API — operates entirely on the
// GPUCapabilities and ComputeDispatch type descriptors.
// ---------------------------------------------------------------------------

import type { GPUCapabilities, ComputeDispatch } from '../types.js';

// ---------------------------------------------------------------------------
// Default capabilities
// ---------------------------------------------------------------------------

/**
 * Returns conservative default GPU capabilities suitable as a fallback
 * when WebGPU is unavailable. Values reflect the minimum guaranteed
 * limits from the WebGPU specification.
 */
export function defaultCapabilities(): GPUCapabilities {
  return {
    available: false,
    maxWorkgroupSize: 256,
    maxWorkgroupsPerDimension: 65535,
    maxStorageBufferSize: 128 * 1024 * 1024, // 128 MiB spec minimum
    maxWorkgroupStorageSize: 16384,           // 16 KiB spec minimum
    f16Supported: false,
    timestampQuerySupported: false,
  };
}

// ---------------------------------------------------------------------------
// Workgroup sizing
// ---------------------------------------------------------------------------

/**
 * Computes the number of workgroups required to cover `totalItems` when
 * each workgroup processes `workgroupSize` items.
 *
 * Returns ceil(totalItems / workgroupSize), or 1 for degenerate inputs.
 */
export function estimateWorkgroups(
  totalItems: number,
  workgroupSize: number,
): number {
  if (totalItems <= 0 || workgroupSize <= 0) {
    return 1;
  }
  return Math.ceil(totalItems / workgroupSize);
}

/**
 * Selects an optimal workgroup size for the given total item count.
 *
 * Strategy:
 * - Prefers powers of two for efficient warp/wave utilization.
 * - Never exceeds `maxWorkgroupSize`.
 * - For very small dispatches, rounds down to the nearest power of two
 *   that is >= totalItems (capped at max).
 */
export function computeOptimalWorkgroupSize(
  totalItems: number,
  maxWorkgroupSize: number,
): number {
  if (totalItems <= 0 || maxWorkgroupSize <= 0) {
    return 1;
  }

  // Find largest power of 2 <= maxWorkgroupSize
  let best = 1;
  while (best * 2 <= maxWorkgroupSize) {
    best *= 2;
  }

  // For small dispatches, use a smaller power of 2 that still covers all items
  if (totalItems < best) {
    let size = 1;
    while (size < totalItems) {
      size *= 2;
    }
    return Math.min(size, best);
  }

  return best;
}

// ---------------------------------------------------------------------------
// Dispatch validation
// ---------------------------------------------------------------------------

/**
 * Validates a ComputeDispatch against GPU capabilities.
 * Returns an array of human-readable error strings (empty = valid).
 */
export function validateDispatch(
  dispatch: ComputeDispatch,
  capabilities: GPUCapabilities,
): string[] {
  const errors: string[] = [];

  if (!capabilities.available) {
    errors.push('WebGPU is not available on this device');
  }

  // Validate workgroup counts per dimension
  const [wx, wy, wz] = dispatch.workgroups;
  const maxDim = capabilities.maxWorkgroupsPerDimension;

  if (wx !== undefined && wx > maxDim) {
    errors.push(
      `Workgroup count X (${wx}) exceeds maximum per dimension (${maxDim})`,
    );
  }
  if (wy !== undefined && wy > maxDim) {
    errors.push(
      `Workgroup count Y (${wy}) exceeds maximum per dimension (${maxDim})`,
    );
  }
  if (wz !== undefined && wz > maxDim) {
    errors.push(
      `Workgroup count Z (${wz}) exceeds maximum per dimension (${maxDim})`,
    );
  }

  // Validate zero workgroup counts
  if (wx !== undefined && wx <= 0) {
    errors.push('Workgroup count X must be greater than 0');
  }
  if (wy !== undefined && wy <= 0) {
    errors.push('Workgroup count Y must be greater than 0');
  }
  if (wz !== undefined && wz <= 0) {
    errors.push('Workgroup count Z must be greater than 0');
  }

  // Validate buffer sizes
  for (const buffer of dispatch.buffers) {
    if (buffer.size > capabilities.maxStorageBufferSize) {
      errors.push(
        `Buffer at binding ${buffer.binding} size (${buffer.size}) exceeds ` +
        `maximum storage buffer size (${capabilities.maxStorageBufferSize})`,
      );
    }
    if (buffer.size <= 0) {
      errors.push(
        `Buffer at binding ${buffer.binding} must have a positive size`,
      );
    }
  }

  // Check for duplicate bindings
  const bindingSet = new Set<number>();
  for (const buffer of dispatch.buffers) {
    if (bindingSet.has(buffer.binding)) {
      errors.push(`Duplicate buffer binding index: ${buffer.binding}`);
    }
    bindingSet.add(buffer.binding);
  }

  return errors;
}
