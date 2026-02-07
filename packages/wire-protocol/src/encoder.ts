/**
 * Binary encoder: WireOp → ArrayBuffer via DataView.
 *
 * Wire format per operation:
 *   [1B opType] [8B hlcTimestamp] [4B objectId] [variable payload]
 *
 * All multi-byte values are little-endian for WebAssembly compatibility.
 */

import { hlcToUint64 } from './clock'
import type { WireOp, BatchMoveOp, BatchRotateOp } from './types'
import {
  OP_MOVE, OP_ROTATE, OP_PLACE, OP_REMOVE, OP_SCALE,
  OP_BATCH_MOVE, OP_BATCH_ROTATE,
  HEADER_SIZE, MOVE_PAYLOAD, ROTATE_PAYLOAD, PLACE_PAYLOAD,
  REMOVE_PAYLOAD, SCALE_PAYLOAD, BATCH_ITEM_SIZE,
} from './types'

// ─── Size calculation ───────────────────────────────────────────────────────

/** Calculate the exact byte size for encoding an operation. */
export function encodedSize(op: WireOp): number {
  switch (op.op) {
    case OP_MOVE:         return HEADER_SIZE + MOVE_PAYLOAD
    case OP_ROTATE:       return HEADER_SIZE + ROTATE_PAYLOAD
    case OP_PLACE:        return HEADER_SIZE + PLACE_PAYLOAD
    case OP_REMOVE:       return HEADER_SIZE + REMOVE_PAYLOAD
    case OP_SCALE:        return HEADER_SIZE + SCALE_PAYLOAD
    case OP_BATCH_MOVE:   return HEADER_SIZE + 2 + op.moves.length * BATCH_ITEM_SIZE
    case OP_BATCH_ROTATE: return HEADER_SIZE + 2 + op.rotations.length * BATCH_ITEM_SIZE
  }
}

// ─── Header encoder ─────────────────────────────────────────────────────────

function writeHeader(view: DataView, offset: number, op: WireOp): number {
  // 1 byte: op type
  view.setUint8(offset, op.op)
  offset += 1

  // 8 bytes: HLC timestamp as uint64 (little-endian)
  const packed = hlcToUint64(op.hlc)
  // Write as two 32-bit words (little-endian)
  view.setUint32(offset, Number(packed & 0xFFFFFFFFn), true)
  view.setUint32(offset + 4, Number(packed >> 32n), true)
  offset += 8

  // 4 bytes: object ID
  view.setUint32(offset, op.objectId, true)
  offset += 4

  return offset
}

// ─── Encode single operation ────────────────────────────────────────────────

/** Encode a single wire operation into a new ArrayBuffer. */
export function encode(op: WireOp): ArrayBuffer {
  const size = encodedSize(op)
  const buffer = new ArrayBuffer(size)
  const view = new DataView(buffer)
  encodeInto(view, 0, op)
  return buffer
}

/** Encode a wire operation into an existing DataView at the given offset. Returns new offset. */
export function encodeInto(view: DataView, offset: number, op: WireOp): number {
  offset = writeHeader(view, offset, op)

  switch (op.op) {
    case OP_MOVE:
      view.setFloat32(offset, op.dx, true); offset += 4
      view.setFloat32(offset, op.dy, true); offset += 4
      view.setFloat32(offset, op.dz, true); offset += 4
      break

    case OP_ROTATE:
      view.setFloat32(offset, op.rx, true); offset += 4
      view.setFloat32(offset, op.ry, true); offset += 4
      view.setFloat32(offset, op.rz, true); offset += 4
      break

    case OP_PLACE:
      view.setUint8(offset, op.furnitureType); offset += 1
      view.setFloat32(offset, op.x, true); offset += 4
      view.setFloat32(offset, op.y, true); offset += 4
      view.setFloat32(offset, op.z, true); offset += 4
      view.setFloat32(offset, op.rx, true); offset += 4
      view.setFloat32(offset, op.ry, true); offset += 4
      view.setFloat32(offset, op.rz, true); offset += 4
      break

    case OP_REMOVE:
      // No payload
      break

    case OP_SCALE:
      view.setFloat32(offset, op.sx, true); offset += 4
      view.setFloat32(offset, op.sy, true); offset += 4
      view.setFloat32(offset, op.sz, true); offset += 4
      break

    case OP_BATCH_MOVE:
      offset = encodeBatchMove(view, offset, op)
      break

    case OP_BATCH_ROTATE:
      offset = encodeBatchRotate(view, offset, op)
      break
  }

  return offset
}

function encodeBatchMove(view: DataView, offset: number, op: BatchMoveOp): number {
  view.setUint16(offset, op.moves.length, true); offset += 2
  for (const m of op.moves) {
    view.setUint32(offset, m.objectId, true); offset += 4
    view.setFloat32(offset, m.dx, true); offset += 4
    view.setFloat32(offset, m.dy, true); offset += 4
    view.setFloat32(offset, m.dz, true); offset += 4
  }
  return offset
}

function encodeBatchRotate(view: DataView, offset: number, op: BatchRotateOp): number {
  view.setUint16(offset, op.rotations.length, true); offset += 2
  for (const r of op.rotations) {
    view.setUint32(offset, r.objectId, true); offset += 4
    view.setFloat32(offset, r.rx, true); offset += 4
    view.setFloat32(offset, r.ry, true); offset += 4
    view.setFloat32(offset, r.rz, true); offset += 4
  }
  return offset
}
