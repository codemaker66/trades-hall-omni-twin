/**
 * Benchmark: binary wire protocol vs JSON for 10,000 operations.
 *
 * Measures:
 * - Encoding speed (ops/second)
 * - Decoding speed (ops/second)
 * - Size comparison (bytes)
 * - Batch frame overhead
 * - Delta compression savings
 */

import { describe, it, expect } from 'vitest'

import type { MoveOp, WireOp, FurnitureTypeIndex, HlcTimestamp } from '../types'
import { OP_MOVE, OP_PLACE, OP_REMOVE, OP_ROTATE } from '../types'
import { encode, encodedSize } from '../encoder'
import { decode } from '../decoder'
import { encodeBatchFrame, decodeBatchFrame } from '../batch'
import {
  DeltaCompressor,
  encodeCompressedMove, decodeCompressedMove, compressedMoveSize,
} from '../compress'

const OPS_COUNT = 10_000

function makeHlc(i: number): HlcTimestamp {
  return { wallMs: 1700000000000 + i, counter: i & 0xFFFF }
}

/** Generate a realistic mix of operations. */
function generateOps(count: number): WireOp[] {
  const ops: WireOp[] = []
  for (let i = 0; i < count; i++) {
    const r = i % 10
    if (r < 6) {
      // 60% moves (most common during drag)
      ops.push({
        op: OP_MOVE, hlc: makeHlc(i), objectId: i % 200,
        dx: (Math.random() - 0.5) * 2,
        dy: 0,
        dz: (Math.random() - 0.5) * 2,
      })
    } else if (r < 8) {
      // 20% rotates
      ops.push({
        op: OP_ROTATE, hlc: makeHlc(i), objectId: i % 200,
        rx: 0, ry: Math.random() * Math.PI * 2, rz: 0,
      })
    } else if (r < 9) {
      // 10% places
      ops.push({
        op: OP_PLACE, hlc: makeHlc(i), objectId: i,
        furnitureType: (i % 7) as FurnitureTypeIndex,
        x: Math.random() * 20, y: 0, z: Math.random() * 15,
        rx: 0, ry: Math.random() * Math.PI, rz: 0,
      })
    } else {
      // 10% removes
      ops.push({
        op: OP_REMOVE, hlc: makeHlc(i), objectId: i % 200,
      })
    }
  }
  return ops
}

/** JSON representation matching the wire op shape. */
function opsToJson(ops: WireOp[]): string[] {
  return ops.map(op => JSON.stringify(op))
}

describe('Benchmark: Binary vs JSON (10K operations)', () => {
  const ops = generateOps(OPS_COUNT)
  const jsonStrings = opsToJson(ops)

  it('measures encoding speed', () => {
    // Binary encode
    const binStart = performance.now()
    const buffers: ArrayBuffer[] = []
    for (const op of ops) {
      buffers.push(encode(op))
    }
    const binTime = performance.now() - binStart

    // JSON encode
    const jsonStart = performance.now()
    const jsons: string[] = []
    for (const op of ops) {
      jsons.push(JSON.stringify(op))
    }
    const jsonTime = performance.now() - jsonStart

    const speedup = jsonTime / binTime

    console.log(`\n  Encoding ${OPS_COUNT} operations:`)
    console.log(`    Binary: ${binTime.toFixed(1)}ms (${(OPS_COUNT / binTime * 1000).toFixed(0)} ops/sec)`)
    console.log(`    JSON:   ${jsonTime.toFixed(1)}ms (${(OPS_COUNT / jsonTime * 1000).toFixed(0)} ops/sec)`)
    console.log(`    Speedup: ${speedup.toFixed(1)}x`)

    // Binary should be at least somewhat competitive
    expect(buffers.length).toBe(OPS_COUNT)
    expect(jsons.length).toBe(OPS_COUNT)
  })

  it('measures decoding speed', () => {
    // Prepare binary buffers
    const buffers = ops.map(op => encode(op))

    // Binary decode
    const binStart = performance.now()
    const decodedBin: WireOp[] = []
    for (const buf of buffers) {
      decodedBin.push(decode(buf))
    }
    const binTime = performance.now() - binStart

    // JSON decode
    const jsonStart = performance.now()
    const decodedJson: unknown[] = []
    for (const str of jsonStrings) {
      decodedJson.push(JSON.parse(str))
    }
    const jsonTime = performance.now() - jsonStart

    const speedup = jsonTime / binTime

    console.log(`\n  Decoding ${OPS_COUNT} operations:`)
    console.log(`    Binary: ${binTime.toFixed(1)}ms (${(OPS_COUNT / binTime * 1000).toFixed(0)} ops/sec)`)
    console.log(`    JSON:   ${jsonTime.toFixed(1)}ms (${(OPS_COUNT / jsonTime * 1000).toFixed(0)} ops/sec)`)
    console.log(`    Speedup: ${speedup.toFixed(1)}x`)

    expect(decodedBin.length).toBe(OPS_COUNT)
    expect(decodedJson.length).toBe(OPS_COUNT)
  })

  it('measures total size', () => {
    // Binary total size
    let binaryTotal = 0
    for (const op of ops) {
      binaryTotal += encodedSize(op)
    }

    // JSON total size
    let jsonTotal = 0
    for (const str of jsonStrings) {
      jsonTotal += str.length // UTF-8 ASCII chars = bytes
    }

    const ratio = jsonTotal / binaryTotal

    console.log(`\n  Total size for ${OPS_COUNT} operations:`)
    console.log(`    Binary: ${(binaryTotal / 1024).toFixed(1)} KB`)
    console.log(`    JSON:   ${(jsonTotal / 1024).toFixed(1)} KB`)
    console.log(`    Reduction: ${ratio.toFixed(1)}x smaller`)

    // Binary should be significantly smaller
    expect(ratio).toBeGreaterThan(3)
  })

  it('measures batch frame size vs individual', () => {
    // Batch all ops into a single frame
    const frame = encodeBatchFrame(ops)
    const batchSize = frame.byteLength

    // Sum of individual buffers
    let individualTotal = 0
    for (const op of ops) {
      individualTotal += encodedSize(op)
    }

    // Batch frame adds 6 bytes header — minimal overhead
    const overhead = batchSize - individualTotal
    console.log(`\n  Batch framing:`)
    console.log(`    Individual total: ${(individualTotal / 1024).toFixed(1)} KB`)
    console.log(`    Batch frame:      ${(batchSize / 1024).toFixed(1)} KB`)
    console.log(`    Overhead:         ${overhead} bytes (${(overhead / batchSize * 100).toFixed(2)}%)`)

    expect(overhead).toBe(6) // Just the frame header
  })

  it('measures batch frame round-trip speed', () => {
    // Encode batch
    const encStart = performance.now()
    const frame = encodeBatchFrame(ops)
    const encTime = performance.now() - encStart

    // Decode batch
    const decStart = performance.now()
    const decoded = decodeBatchFrame(frame)
    const decTime = performance.now() - decStart

    console.log(`\n  Batch frame (${OPS_COUNT} ops):`)
    console.log(`    Encode: ${encTime.toFixed(1)}ms`)
    console.log(`    Decode: ${decTime.toFixed(1)}ms`)
    console.log(`    Frame size: ${(frame.byteLength / 1024).toFixed(1)} KB`)

    expect(decoded.length).toBe(OPS_COUNT)
  })

  it('measures delta compression savings for drag sequence', () => {
    // Simulate a drag: 500 small position updates for one object
    const dragCount = 500
    const comp = new DeltaCompressor()
    let x = 5.0, y = 0, z = 10.0

    let fullBytes = 0
    let compressedBytes = 0
    let suppressedCount = 0

    for (let i = 0; i < dragCount; i++) {
      x += (Math.random() - 0.5) * 0.1
      z += (Math.random() - 0.5) * 0.1

      const hlc = makeHlc(i)
      const compressed = comp.compress(1, x, y, z, hlc)

      // Full wire op size for comparison
      fullBytes += encodedSize({
        op: OP_MOVE, hlc, objectId: 1,
        dx: x, dy: y, dz: z,
      })

      if (compressed) {
        compressedBytes += compressedMoveSize(compressed)
      } else {
        suppressedCount++
      }
    }

    const savings = 1 - compressedBytes / fullBytes
    console.log(`\n  Delta compression (${dragCount} drag updates):`)
    console.log(`    Full:       ${(fullBytes / 1024).toFixed(1)} KB`)
    console.log(`    Compressed: ${(compressedBytes / 1024).toFixed(1)} KB`)
    console.log(`    Savings:    ${(savings * 100).toFixed(1)}%`)
    console.log(`    Suppressed: ${suppressedCount} updates (deadzone)`)

    // Delta should save at least 15%
    expect(savings).toBeGreaterThan(0.15)
  })
})
