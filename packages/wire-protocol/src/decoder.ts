/**
 * Binary decoder: ArrayBuffer → WireOp via DataView.
 *
 * Reads the wire format produced by encoder.ts.
 * All multi-byte values are little-endian.
 */

import { uint64ToHlc } from './clock'
import type {
  WireOp, MoveOp, RotateOp, PlaceOp, RemoveOp, ScaleOp,
  BatchMoveOp, BatchRotateOp, HlcTimestamp, FurnitureTypeIndex,
} from './types'
import {
  OP_MOVE, OP_ROTATE, OP_PLACE, OP_REMOVE, OP_SCALE,
  OP_BATCH_MOVE, OP_BATCH_ROTATE,
} from './types'

// ─── Decode result with cursor ──────────────────────────────────────────────

export interface DecodeResult {
  op: WireOp
  bytesRead: number
}

// ─── Header decoder ─────────────────────────────────────────────────────────

function readHeader(view: DataView, offset: number): {
  opType: number
  hlc: HlcTimestamp
  objectId: number
  newOffset: number
} {
  const opType = view.getUint8(offset)
  offset += 1

  // 8 bytes: HLC as uint64 (little-endian, two 32-bit words)
  const lo = view.getUint32(offset, true)
  const hi = view.getUint32(offset + 4, true)
  const packed = (BigInt(hi) << 32n) | BigInt(lo)
  const hlc = uint64ToHlc(packed)
  offset += 8

  const objectId = view.getUint32(offset, true)
  offset += 4

  return { opType, hlc, objectId, newOffset: offset }
}

// ─── Decode single operation ────────────────────────────────────────────────

/** Decode a single wire operation from a DataView at the given offset. */
export function decodeAt(view: DataView, offset: number): DecodeResult {
  const startOffset = offset
  const { opType, hlc, objectId, newOffset } = readHeader(view, offset)
  offset = newOffset

  let op: WireOp

  switch (opType) {
    case OP_MOVE: {
      const dx = view.getFloat32(offset, true); offset += 4
      const dy = view.getFloat32(offset, true); offset += 4
      const dz = view.getFloat32(offset, true); offset += 4
      op = { op: OP_MOVE, hlc, objectId, dx, dy, dz } satisfies MoveOp
      break
    }

    case OP_ROTATE: {
      const rx = view.getFloat32(offset, true); offset += 4
      const ry = view.getFloat32(offset, true); offset += 4
      const rz = view.getFloat32(offset, true); offset += 4
      op = { op: OP_ROTATE, hlc, objectId, rx, ry, rz } satisfies RotateOp
      break
    }

    case OP_PLACE: {
      const furnitureType = view.getUint8(offset) as FurnitureTypeIndex; offset += 1
      const x = view.getFloat32(offset, true); offset += 4
      const y = view.getFloat32(offset, true); offset += 4
      const z = view.getFloat32(offset, true); offset += 4
      const rx = view.getFloat32(offset, true); offset += 4
      const ry = view.getFloat32(offset, true); offset += 4
      const rz = view.getFloat32(offset, true); offset += 4
      op = { op: OP_PLACE, hlc, objectId, furnitureType, x, y, z, rx, ry, rz } satisfies PlaceOp
      break
    }

    case OP_REMOVE: {
      op = { op: OP_REMOVE, hlc, objectId } satisfies RemoveOp
      break
    }

    case OP_SCALE: {
      const sx = view.getFloat32(offset, true); offset += 4
      const sy = view.getFloat32(offset, true); offset += 4
      const sz = view.getFloat32(offset, true); offset += 4
      op = { op: OP_SCALE, hlc, objectId, sx, sy, sz } satisfies ScaleOp
      break
    }

    case OP_BATCH_MOVE: {
      const count = view.getUint16(offset, true); offset += 2
      const moves: BatchMoveOp['moves'] = []
      for (let i = 0; i < count; i++) {
        const oid = view.getUint32(offset, true); offset += 4
        const dx = view.getFloat32(offset, true); offset += 4
        const dy = view.getFloat32(offset, true); offset += 4
        const dz = view.getFloat32(offset, true); offset += 4
        moves.push({ objectId: oid, dx, dy, dz })
      }
      op = { op: OP_BATCH_MOVE, hlc, objectId: 0, moves } satisfies BatchMoveOp
      break
    }

    case OP_BATCH_ROTATE: {
      const count = view.getUint16(offset, true); offset += 2
      const rotations: BatchRotateOp['rotations'] = []
      for (let i = 0; i < count; i++) {
        const oid = view.getUint32(offset, true); offset += 4
        const rx = view.getFloat32(offset, true); offset += 4
        const ry = view.getFloat32(offset, true); offset += 4
        const rz = view.getFloat32(offset, true); offset += 4
        rotations.push({ objectId: oid, rx, ry, rz })
      }
      op = { op: OP_BATCH_ROTATE, hlc, objectId: 0, rotations } satisfies BatchRotateOp
      break
    }

    default:
      throw new Error(`Unknown op type: 0x${opType.toString(16).padStart(2, '0')}`)
  }

  return { op, bytesRead: offset - startOffset }
}

/** Decode a single wire operation from an ArrayBuffer. */
export function decode(buffer: ArrayBuffer): WireOp {
  return decodeAt(new DataView(buffer), 0).op
}
