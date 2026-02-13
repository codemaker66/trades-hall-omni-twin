// ---------------------------------------------------------------------------
// @omni-twin/compute-core — HPC-2: Zero-Copy Data Transfer Utilities
// ---------------------------------------------------------------------------
// Utilities for managing typed regions of WASM linear memory that can be
// shared with JavaScript without copying. Handles alignment, overlap
// detection, validation, and region splitting.
// ---------------------------------------------------------------------------

import type { ZeroCopyRegion } from '../types.js';

// ---------------------------------------------------------------------------
// Type metadata
// ---------------------------------------------------------------------------

/** Byte alignment requirements for each supported dtype. */
const ALIGNMENT: Record<ZeroCopyRegion['dtype'], number> = {
  u8: 1,
  i32: 4,
  u32: 4,
  f32: 4,
  f64: 8,
};

/** Byte size of a single element for each supported dtype. */
const BYTES_PER_ELEMENT: Record<ZeroCopyRegion['dtype'], number> = {
  u8: 1,
  i32: 4,
  u32: 4,
  f32: 4,
  f64: 8,
};

/**
 * Returns the required byte alignment for the given dtype.
 *
 * Alignment must be respected when calculating region pointers into
 * WASM linear memory to avoid undefined behavior on strict architectures.
 */
export function computeAlignment(dtype: ZeroCopyRegion['dtype']): number {
  return ALIGNMENT[dtype];
}

/**
 * Returns the number of bytes occupied by a single element of the given dtype.
 */
export function bytesPerElement(dtype: ZeroCopyRegion['dtype']): number {
  return BYTES_PER_ELEMENT[dtype];
}

// ---------------------------------------------------------------------------
// Overlap detection
// ---------------------------------------------------------------------------

/**
 * Checks whether two zero-copy regions overlap in linear memory.
 *
 * Two regions overlap if their byte ranges [ptr, ptr + byteLength)
 * intersect. Zero-length regions never overlap with anything.
 */
export function regionOverlaps(a: ZeroCopyRegion, b: ZeroCopyRegion): boolean {
  if (a.byteLength === 0 || b.byteLength === 0) {
    return false;
  }

  const aEnd = a.ptr + a.byteLength;
  const bEnd = b.ptr + b.byteLength;

  return a.ptr < bEnd && b.ptr < aEnd;
}

// ---------------------------------------------------------------------------
// Region allocation
// ---------------------------------------------------------------------------

/**
 * Creates a ZeroCopyRegion descriptor from a pointer, element count, and dtype.
 *
 * The byte length is computed as `count * bytesPerElement(dtype)`.
 * This function does not actually allocate memory — it constructs the
 * descriptor that refers to an already-allocated region in WASM memory.
 */
export function allocateRegion(
  ptr: number,
  count: number,
  dtype: ZeroCopyRegion['dtype'],
): ZeroCopyRegion {
  const bpe = BYTES_PER_ELEMENT[dtype];
  return {
    ptr,
    byteLength: count * bpe,
    dtype,
  };
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/**
 * Validates a zero-copy region against total WASM memory bounds and
 * alignment requirements.
 *
 * Returns an array of human-readable error strings. An empty array
 * indicates a valid region.
 */
export function validateRegion(
  region: ZeroCopyRegion,
  totalMemoryBytes: number,
): string[] {
  const errors: string[] = [];

  // Pointer must be non-negative
  if (region.ptr < 0) {
    errors.push(`Pointer must be non-negative, got ${region.ptr}`);
  }

  // Byte length must be non-negative
  if (region.byteLength < 0) {
    errors.push(`Byte length must be non-negative, got ${region.byteLength}`);
  }

  // Region must fit within WASM memory
  if (region.ptr + region.byteLength > totalMemoryBytes) {
    errors.push(
      `Region [${region.ptr}, ${region.ptr + region.byteLength}) ` +
      `exceeds total memory (${totalMemoryBytes} bytes)`,
    );
  }

  // Alignment check
  const align = ALIGNMENT[region.dtype];
  if (region.ptr % align !== 0) {
    errors.push(
      `Pointer ${region.ptr} is not aligned to ${align} bytes ` +
      `(required for ${region.dtype})`,
    );
  }

  // Byte length must be a multiple of element size
  const bpe = BYTES_PER_ELEMENT[region.dtype];
  if (region.byteLength % bpe !== 0) {
    errors.push(
      `Byte length ${region.byteLength} is not a multiple of element ` +
      `size ${bpe} (${region.dtype})`,
    );
  }

  return errors;
}

// ---------------------------------------------------------------------------
// Region splitting
// ---------------------------------------------------------------------------

/**
 * Splits a region into contiguous chunks of at most `chunkElements` elements.
 *
 * The last chunk may contain fewer elements if the region is not evenly
 * divisible by the chunk size.
 *
 * Useful for breaking large data transfers into smaller pieces that can
 * be processed incrementally without blocking the main thread.
 */
export function splitRegion(
  region: ZeroCopyRegion,
  chunkElements: number,
): ZeroCopyRegion[] {
  if (chunkElements <= 0) {
    return [region];
  }

  const bpe = BYTES_PER_ELEMENT[region.dtype];
  const totalElements = Math.floor(region.byteLength / bpe);

  if (totalElements <= chunkElements) {
    return [region];
  }

  const chunks: ZeroCopyRegion[] = [];
  let offset = 0;
  let remaining = totalElements;

  while (remaining > 0) {
    const count = Math.min(chunkElements, remaining);
    chunks.push({
      ptr: region.ptr + offset * bpe,
      byteLength: count * bpe,
      dtype: region.dtype,
    });
    offset += count;
    remaining -= count;
  }

  return chunks;
}
