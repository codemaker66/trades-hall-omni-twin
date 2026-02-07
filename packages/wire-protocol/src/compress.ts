/**
 * Delta compression for position updates.
 *
 * When sending a stream of move operations for the same object (e.g., during drag),
 * consecutive positions often have small deltas. This module provides:
 *
 * 1. Delta encoding: store position diffs as int16 (fixed-point) instead of float32
 *    when the delta is small enough. Saves 6 bytes per move (18 → 12 bytes payload,
 *    plus 1 byte flag).
 *
 * 2. Run-length suppression: skip sending updates when the delta is below a threshold
 *    (deadzone filtering for high-frequency drag events).
 *
 * Wire format for delta-compressed move:
 *   [1B flags] [variable payload]
 *
 *   flags bit 0: 0 = full float32, 1 = delta int16
 *   flags bit 1: 0 = absolute position, 1 = relative displacement
 *
 *   Full:  [4B float32 x] [4B float32 y] [4B float32 z]  = 12 bytes
 *   Delta: [2B int16 dx]  [2B int16 dy]  [2B int16 dz]   = 6 bytes
 *
 *   Delta values are fixed-point: int16 / SCALE_FACTOR = float meters.
 *   With SCALE_FACTOR = 1000, range is ±32.767 meters at 1mm precision.
 */

import type { HlcTimestamp } from './types'

// ─── Constants ──────────────────────────────────────────────────────────────

/** Fixed-point scale factor: 1 unit = 1mm precision, ±32.767m range. */
export const SCALE_FACTOR = 1000

/** Maximum absolute delta expressible as int16 (32.767 meters). */
export const MAX_DELTA = 32767 / SCALE_FACTOR

/** Minimum position change to emit an update (deadzone, 0.5mm). */
export const DEADZONE = 0.0005

// ─── Flags ──────────────────────────────────────────────────────────────────

export const FLAG_DELTA    = 0x01
export const FLAG_RELATIVE = 0x02

// ─── Delta State Tracker ────────────────────────────────────────────────────

interface TrackedObject {
  lastX: number
  lastY: number
  lastZ: number
}

/**
 * Tracks per-object last-known positions for delta encoding.
 * Use one DeltaCompressor per connection/peer.
 */
export class DeltaCompressor {
  private tracked = new Map<number, TrackedObject>()

  /** Reset all tracked state. */
  clear(): void {
    this.tracked.clear()
  }

  /** Remove tracking for a specific object. */
  forget(objectId: number): void {
    this.tracked.delete(objectId)
  }

  /**
   * Compress a position update. Returns null if the change is within the deadzone.
   * Returns a CompressedMove with either delta or full encoding.
   */
  compress(
    objectId: number,
    x: number,
    y: number,
    z: number,
    hlc: HlcTimestamp,
  ): CompressedMove | null {
    const prev = this.tracked.get(objectId)

    if (!prev) {
      // First time seeing this object — send full position
      this.tracked.set(objectId, { lastX: x, lastY: y, lastZ: z })
      return { objectId, hlc, isDelta: false, x, y, z }
    }

    const dx = x - prev.lastX
    const dy = y - prev.lastY
    const dz = z - prev.lastZ

    // Deadzone: skip if change is negligible
    if (Math.abs(dx) < DEADZONE && Math.abs(dy) < DEADZONE && Math.abs(dz) < DEADZONE) {
      return null
    }

    // Update tracked position
    prev.lastX = x
    prev.lastY = y
    prev.lastZ = z

    // Try delta encoding
    if (Math.abs(dx) <= MAX_DELTA && Math.abs(dy) <= MAX_DELTA && Math.abs(dz) <= MAX_DELTA) {
      return { objectId, hlc, isDelta: true, dx, dy, dz }
    }

    // Fallback: full position
    return { objectId, hlc, isDelta: false, x, y, z }
  }

  /**
   * Decompress: apply a compressed move to update tracked state and return absolute position.
   */
  decompress(move: CompressedMove): { x: number; y: number; z: number } {
    if (move.isDelta) {
      const prev = this.tracked.get(move.objectId)
      const baseX = prev?.lastX ?? 0
      const baseY = prev?.lastY ?? 0
      const baseZ = prev?.lastZ ?? 0
      const x = baseX + move.dx
      const y = baseY + move.dy
      const z = baseZ + move.dz
      this.tracked.set(move.objectId, { lastX: x, lastY: y, lastZ: z })
      return { x, y, z }
    }

    this.tracked.set(move.objectId, { lastX: move.x, lastY: move.y, lastZ: move.z })
    return { x: move.x, y: move.y, z: move.z }
  }
}

// ─── Compressed Move Types ──────────────────────────────────────────────────

export interface FullMove {
  objectId: number
  hlc: HlcTimestamp
  isDelta: false
  x: number
  y: number
  z: number
}

export interface DeltaMove {
  objectId: number
  hlc: HlcTimestamp
  isDelta: true
  dx: number
  dy: number
  dz: number
}

export type CompressedMove = FullMove | DeltaMove

// ─── Binary encode/decode for CompressedMove ────────────────────────────────

/**
 * Encode a CompressedMove into a DataView at the given offset.
 * Format: [1B flags] [4B objectId] [8B hlc] [payload: 12B full or 6B delta]
 * Returns new offset.
 */
export function encodeCompressedMove(view: DataView, offset: number, move: CompressedMove): number {
  const flags = move.isDelta ? (FLAG_DELTA | FLAG_RELATIVE) : 0
  view.setUint8(offset, flags); offset += 1

  view.setUint32(offset, move.objectId, true); offset += 4

  // HLC inline (8 bytes)
  const packed = (BigInt(move.hlc.wallMs) << 16n) | BigInt(move.hlc.counter & 0xFFFF)
  view.setUint32(offset, Number(packed & 0xFFFFFFFFn), true)
  view.setUint32(offset + 4, Number(packed >> 32n), true)
  offset += 8

  if (move.isDelta) {
    view.setInt16(offset, Math.round(move.dx * SCALE_FACTOR), true); offset += 2
    view.setInt16(offset, Math.round(move.dy * SCALE_FACTOR), true); offset += 2
    view.setInt16(offset, Math.round(move.dz * SCALE_FACTOR), true); offset += 2
  } else {
    view.setFloat32(offset, move.x, true); offset += 4
    view.setFloat32(offset, move.y, true); offset += 4
    view.setFloat32(offset, move.z, true); offset += 4
  }

  return offset
}

/** Byte size for a compressed move. */
export function compressedMoveSize(move: CompressedMove): number {
  // 1B flags + 4B objectId + 8B hlc + payload
  return 13 + (move.isDelta ? 6 : 12)
}

/**
 * Decode a CompressedMove from a DataView at the given offset.
 * Returns the decoded move and the new offset.
 */
export function decodeCompressedMove(view: DataView, offset: number): { move: CompressedMove; newOffset: number } {
  const flags = view.getUint8(offset); offset += 1
  const isDelta = (flags & FLAG_DELTA) !== 0

  const objectId = view.getUint32(offset, true); offset += 4

  const lo = view.getUint32(offset, true)
  const hi = view.getUint32(offset + 4, true)
  const packed = (BigInt(hi) << 32n) | BigInt(lo)
  const hlc = {
    wallMs: Number(packed >> 16n),
    counter: Number(packed & 0xFFFFn),
  }
  offset += 8

  if (isDelta) {
    const dx = view.getInt16(offset, true) / SCALE_FACTOR; offset += 2
    const dy = view.getInt16(offset, true) / SCALE_FACTOR; offset += 2
    const dz = view.getInt16(offset, true) / SCALE_FACTOR; offset += 2
    return { move: { objectId, hlc, isDelta: true, dx, dy, dz }, newOffset: offset }
  }

  const x = view.getFloat32(offset, true); offset += 4
  const y = view.getFloat32(offset, true); offset += 4
  const z = view.getFloat32(offset, true); offset += 4
  return { move: { objectId, hlc, isDelta: false, x, y, z }, newOffset: offset }
}
