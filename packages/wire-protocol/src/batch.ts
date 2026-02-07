/**
 * Batch framing: pack multiple operations into a single network frame.
 *
 * Frame format:
 *   [4B total_length] [2B op_count] [ops...]
 *
 * Each operation is self-describing (starts with 1B opType) and variable-length.
 * The total_length includes the 6-byte frame header.
 */

import type { WireOp } from './types'
import { BATCH_FRAME_HEADER } from './types'
import { encodeInto, encodedSize } from './encoder'
import { decodeAt } from './decoder'

/** Encode multiple operations into a single framed ArrayBuffer. */
export function encodeBatchFrame(ops: WireOp[]): ArrayBuffer {
  // Calculate total size
  let payloadSize = 0
  for (const op of ops) {
    payloadSize += encodedSize(op)
  }

  const totalSize = BATCH_FRAME_HEADER + payloadSize
  const buffer = new ArrayBuffer(totalSize)
  const view = new DataView(buffer)

  // Frame header
  view.setUint32(0, totalSize, true)
  view.setUint16(4, ops.length, true)

  // Pack operations
  let offset = BATCH_FRAME_HEADER
  for (const op of ops) {
    offset = encodeInto(view, offset, op)
  }

  return buffer
}

/** Decode a batch frame into an array of operations. */
export function decodeBatchFrame(buffer: ArrayBuffer): WireOp[] {
  const view = new DataView(buffer)

  const totalLength = view.getUint32(0, true)
  if (totalLength !== buffer.byteLength) {
    throw new Error(`Frame length mismatch: header says ${totalLength}, buffer is ${buffer.byteLength}`)
  }

  const opCount = view.getUint16(4, true)
  const ops: WireOp[] = []

  let offset = BATCH_FRAME_HEADER
  for (let i = 0; i < opCount; i++) {
    const { op, bytesRead } = decodeAt(view, offset)
    ops.push(op)
    offset += bytesRead
  }

  return ops
}

/** Read the op count from a batch frame without fully decoding. */
export function peekBatchCount(buffer: ArrayBuffer): number {
  if (buffer.byteLength < BATCH_FRAME_HEADER) {
    throw new Error(`Buffer too small for batch frame header: ${buffer.byteLength} bytes`)
  }
  return new DataView(buffer).getUint16(4, true)
}

/** Read the total frame length from the first 4 bytes (useful for streaming). */
export function peekFrameLength(buffer: ArrayBuffer): number {
  if (buffer.byteLength < 4) {
    throw new Error(`Buffer too small for frame length: ${buffer.byteLength} bytes`)
  }
  return new DataView(buffer).getUint32(0, true)
}
