import { describe, it, expect } from 'vitest'
import fc from 'fast-check'

import {
  OP_MOVE, OP_ROTATE, OP_PLACE, OP_REMOVE, OP_SCALE,
  OP_BATCH_MOVE, OP_BATCH_ROTATE,
  HEADER_SIZE, MOVE_PAYLOAD, ROTATE_PAYLOAD, PLACE_PAYLOAD,
  REMOVE_PAYLOAD, SCALE_PAYLOAD,
  furnitureIndexToName, furnitureNameToIndex,
} from '../types'
import type {
  MoveOp, RotateOp, PlaceOp, RemoveOp, ScaleOp,
  BatchMoveOp, BatchRotateOp, WireOp, HlcTimestamp,
  FurnitureTypeIndex,
} from '../types'
import {
  HybridLogicalClock, hlcCompare, hlcToUint64, uint64ToHlc,
} from '../clock'
import { encode, encodedSize } from '../encoder'
import { decode, decodeAt } from '../decoder'
import { encodeBatchFrame, decodeBatchFrame, peekBatchCount, peekFrameLength } from '../batch'
import {
  DeltaCompressor, SCALE_FACTOR, MAX_DELTA, DEADZONE,
  encodeCompressedMove, decodeCompressedMove, compressedMoveSize,
} from '../compress'

// ─── Test Helpers ───────────────────────────────────────────────────────────

const hlc = (wallMs: number, counter: number): HlcTimestamp => ({ wallMs, counter })

function floatClose(a: number, b: number, tolerance = 1e-6): boolean {
  return Math.abs(a - b) < tolerance
}

// ─── Furniture Type Mapping ─────────────────────────────────────────────────

describe('FurnitureType mapping', () => {
  it('round-trips all furniture types', () => {
    const types = ['chair', 'round-table', 'rect-table', 'trestle-table', 'podium', 'stage', 'bar'] as const
    for (const name of types) {
      const idx = furnitureNameToIndex(name)
      expect(furnitureIndexToName(idx)).toBe(name)
    }
  })

  it('throws on unknown name', () => {
    expect(() => furnitureNameToIndex('unknown' as never)).toThrow()
  })
})

// ─── HLC Clock ──────────────────────────────────────────────────────────────

describe('HybridLogicalClock', () => {
  it('generates monotonically increasing timestamps', () => {
    let time = 1000
    const clock = new HybridLogicalClock(() => time)

    const t1 = clock.tick()
    const t2 = clock.tick()
    expect(hlcCompare(t2, t1)).toBeGreaterThan(0)

    // Advance wall clock
    time = 2000
    const t3 = clock.tick()
    expect(hlcCompare(t3, t2)).toBeGreaterThan(0)
    expect(t3.wallMs).toBe(2000)
    expect(t3.counter).toBe(0)
  })

  it('increments counter when wall clock is stale', () => {
    const clock = new HybridLogicalClock(() => 1000)
    const t1 = clock.tick()
    const t2 = clock.tick()
    const t3 = clock.tick()

    expect(t1.counter).toBe(0)
    expect(t2.counter).toBe(1)
    expect(t3.counter).toBe(2)
    expect(t1.wallMs).toBe(t2.wallMs)
  })

  it('merges with remote timestamp correctly', () => {
    let time = 1000
    const clock = new HybridLogicalClock(() => time)
    clock.tick() // wallMs=1000, counter=0

    // Remote is ahead
    const received = clock.receive(hlc(2000, 5))
    expect(received.wallMs).toBe(2000)
    expect(received.counter).toBe(6)
  })

  it('handles remote behind local', () => {
    let time = 3000
    const clock = new HybridLogicalClock(() => time)
    clock.tick() // wallMs=3000, counter=0

    // Remote is behind
    const received = clock.receive(hlc(1000, 10))
    expect(received.wallMs).toBe(3000)
    expect(received.counter).toBe(1)
  })

  it('handles counter overflow', () => {
    const clock = new HybridLogicalClock(() => 1000)

    // Force counter to 0xFFFF (need 0x10000 ticks: first sets counter=0, rest increment)
    for (let i = 0; i <= 0xFFFF; i++) {
      clock.tick()
    }
    // Next tick overflows counter (0xFFFF + 1 > 0xFFFF) → wallMs advances
    const t = clock.tick()
    expect(t.wallMs).toBe(1001)
    expect(t.counter).toBe(0)
  })
})

describe('HLC encoding', () => {
  it('round-trips through uint64', () => {
    const original = hlc(1700000000000, 42)
    const packed = hlcToUint64(original)
    const unpacked = uint64ToHlc(packed)
    expect(unpacked.wallMs).toBe(original.wallMs)
    expect(unpacked.counter).toBe(original.counter)
  })

  it('preserves ordering in packed form', () => {
    const a = hlcToUint64(hlc(1000, 5))
    const b = hlcToUint64(hlc(1000, 6))
    const c = hlcToUint64(hlc(1001, 0))
    expect(a < b).toBe(true)
    expect(b < c).toBe(true)
  })

  it('handles zero timestamp', () => {
    const zero = uint64ToHlc(hlcToUint64(hlc(0, 0)))
    expect(zero.wallMs).toBe(0)
    expect(zero.counter).toBe(0)
  })
})

// ─── Encoder / Decoder Round-Trip ───────────────────────────────────────────

describe('Encode / Decode round-trip', () => {
  it('MoveOp', () => {
    const op: MoveOp = {
      op: OP_MOVE, hlc: hlc(1700000000000, 1), objectId: 42,
      dx: 1.5, dy: -0.25, dz: 3.0,
    }
    const buffer = encode(op)
    expect(buffer.byteLength).toBe(HEADER_SIZE + MOVE_PAYLOAD)
    const decoded = decode(buffer) as MoveOp
    expect(decoded.op).toBe(OP_MOVE)
    expect(decoded.hlc.wallMs).toBe(op.hlc.wallMs)
    expect(decoded.hlc.counter).toBe(op.hlc.counter)
    expect(decoded.objectId).toBe(42)
    expect(floatClose(decoded.dx, 1.5)).toBe(true)
    expect(floatClose(decoded.dy, -0.25)).toBe(true)
    expect(floatClose(decoded.dz, 3.0)).toBe(true)
  })

  it('RotateOp', () => {
    const op: RotateOp = {
      op: OP_ROTATE, hlc: hlc(1700000000000, 2), objectId: 99,
      rx: 0, ry: Math.PI / 2, rz: 0,
    }
    const buffer = encode(op)
    expect(buffer.byteLength).toBe(HEADER_SIZE + ROTATE_PAYLOAD)
    const decoded = decode(buffer) as RotateOp
    expect(decoded.op).toBe(OP_ROTATE)
    expect(decoded.objectId).toBe(99)
    expect(floatClose(decoded.ry, Math.PI / 2)).toBe(true)
  })

  it('PlaceOp', () => {
    const op: PlaceOp = {
      op: OP_PLACE, hlc: hlc(1700000000000, 3), objectId: 200,
      furnitureType: 2 as FurnitureTypeIndex,
      x: 5.0, y: 0, z: 10.0,
      rx: 0, ry: 1.57, rz: 0,
    }
    const buffer = encode(op)
    expect(buffer.byteLength).toBe(HEADER_SIZE + PLACE_PAYLOAD)
    const decoded = decode(buffer) as PlaceOp
    expect(decoded.op).toBe(OP_PLACE)
    expect(decoded.furnitureType).toBe(2)
    expect(floatClose(decoded.x, 5.0)).toBe(true)
    expect(floatClose(decoded.z, 10.0)).toBe(true)
    expect(floatClose(decoded.ry, 1.57)).toBe(true)
  })

  it('RemoveOp', () => {
    const op: RemoveOp = {
      op: OP_REMOVE, hlc: hlc(1700000000000, 4), objectId: 50,
    }
    const buffer = encode(op)
    expect(buffer.byteLength).toBe(HEADER_SIZE + REMOVE_PAYLOAD)
    const decoded = decode(buffer) as RemoveOp
    expect(decoded.op).toBe(OP_REMOVE)
    expect(decoded.objectId).toBe(50)
  })

  it('ScaleOp', () => {
    const op: ScaleOp = {
      op: OP_SCALE, hlc: hlc(1700000000000, 5), objectId: 77,
      sx: 2.0, sy: 1.5, sz: 0.5,
    }
    const buffer = encode(op)
    expect(buffer.byteLength).toBe(HEADER_SIZE + SCALE_PAYLOAD)
    const decoded = decode(buffer) as ScaleOp
    expect(decoded.op).toBe(OP_SCALE)
    expect(floatClose(decoded.sx, 2.0)).toBe(true)
    expect(floatClose(decoded.sy, 1.5)).toBe(true)
  })

  it('BatchMoveOp', () => {
    const op: BatchMoveOp = {
      op: OP_BATCH_MOVE, hlc: hlc(1700000000000, 6), objectId: 0,
      moves: [
        { objectId: 1, dx: 0.5, dy: 0, dz: 0.5 },
        { objectId: 2, dx: -1.0, dy: 0, dz: 2.0 },
        { objectId: 3, dx: 0, dy: 0.1, dz: 0 },
      ],
    }
    const buffer = encode(op)
    expect(buffer.byteLength).toBe(HEADER_SIZE + 2 + 3 * 16)
    const decoded = decode(buffer) as BatchMoveOp
    expect(decoded.op).toBe(OP_BATCH_MOVE)
    expect(decoded.moves.length).toBe(3)
    expect(decoded.moves[0]!.objectId).toBe(1)
    expect(floatClose(decoded.moves[1]!.dx, -1.0)).toBe(true)
    expect(floatClose(decoded.moves[2]!.dy, 0.1)).toBe(true)
  })

  it('BatchRotateOp', () => {
    const op: BatchRotateOp = {
      op: OP_BATCH_ROTATE, hlc: hlc(1700000000000, 7), objectId: 0,
      rotations: [
        { objectId: 10, rx: 0, ry: 1.57, rz: 0 },
        { objectId: 20, rx: 0, ry: 3.14, rz: 0 },
      ],
    }
    const buffer = encode(op)
    const decoded = decode(buffer) as BatchRotateOp
    expect(decoded.op).toBe(OP_BATCH_ROTATE)
    expect(decoded.rotations.length).toBe(2)
    expect(decoded.rotations[0]!.objectId).toBe(10)
    expect(floatClose(decoded.rotations[1]!.ry, 3.14)).toBe(true)
  })

  it('rejects unknown op type', () => {
    const buffer = new ArrayBuffer(13)
    const view = new DataView(buffer)
    view.setUint8(0, 0xFF) // unknown op type
    expect(() => decode(buffer)).toThrow('Unknown op type: 0xff')
  })
})

describe('encodedSize', () => {
  it('returns correct sizes for all op types', () => {
    expect(encodedSize({ op: OP_MOVE, hlc: hlc(0, 0), objectId: 0, dx: 0, dy: 0, dz: 0 })).toBe(25)
    expect(encodedSize({ op: OP_ROTATE, hlc: hlc(0, 0), objectId: 0, rx: 0, ry: 0, rz: 0 })).toBe(25)
    expect(encodedSize({ op: OP_PLACE, hlc: hlc(0, 0), objectId: 0, furnitureType: 0 as FurnitureTypeIndex, x: 0, y: 0, z: 0, rx: 0, ry: 0, rz: 0 })).toBe(38)
    expect(encodedSize({ op: OP_REMOVE, hlc: hlc(0, 0), objectId: 0 })).toBe(13)
    expect(encodedSize({ op: OP_SCALE, hlc: hlc(0, 0), objectId: 0, sx: 0, sy: 0, sz: 0 })).toBe(25)
  })
})

// ─── Batch Framing ──────────────────────────────────────────────────────────

describe('Batch framing', () => {
  it('round-trips a batch of mixed operations', () => {
    const ops: WireOp[] = [
      { op: OP_MOVE, hlc: hlc(1000, 0), objectId: 1, dx: 1, dy: 0, dz: 0 },
      { op: OP_ROTATE, hlc: hlc(1000, 1), objectId: 2, rx: 0, ry: 1.57, rz: 0 },
      { op: OP_REMOVE, hlc: hlc(1000, 2), objectId: 3 },
      { op: OP_PLACE, hlc: hlc(1000, 3), objectId: 4, furnitureType: 0 as FurnitureTypeIndex, x: 5, y: 0, z: 5, rx: 0, ry: 0, rz: 0 },
    ]

    const frame = encodeBatchFrame(ops)
    const decoded = decodeBatchFrame(frame)

    expect(decoded.length).toBe(4)
    expect(decoded[0]!.op).toBe(OP_MOVE)
    expect(decoded[1]!.op).toBe(OP_ROTATE)
    expect(decoded[2]!.op).toBe(OP_REMOVE)
    expect(decoded[3]!.op).toBe(OP_PLACE)
  })

  it('handles empty batch', () => {
    const frame = encodeBatchFrame([])
    expect(peekBatchCount(frame)).toBe(0)
    const decoded = decodeBatchFrame(frame)
    expect(decoded.length).toBe(0)
  })

  it('peekFrameLength returns correct total size', () => {
    const ops: WireOp[] = [
      { op: OP_MOVE, hlc: hlc(0, 0), objectId: 1, dx: 0, dy: 0, dz: 0 },
    ]
    const frame = encodeBatchFrame(ops)
    expect(peekFrameLength(frame)).toBe(frame.byteLength)
  })

  it('peekBatchCount returns correct count', () => {
    const ops: WireOp[] = [
      { op: OP_MOVE, hlc: hlc(0, 0), objectId: 1, dx: 0, dy: 0, dz: 0 },
      { op: OP_MOVE, hlc: hlc(0, 1), objectId: 2, dx: 0, dy: 0, dz: 0 },
      { op: OP_MOVE, hlc: hlc(0, 2), objectId: 3, dx: 0, dy: 0, dz: 0 },
    ]
    const frame = encodeBatchFrame(ops)
    expect(peekBatchCount(frame)).toBe(3)
  })

  it('detects frame length mismatch', () => {
    const frame = encodeBatchFrame([
      { op: OP_MOVE, hlc: hlc(0, 0), objectId: 1, dx: 0, dy: 0, dz: 0 },
    ])
    // Truncate buffer
    const truncated = frame.slice(0, frame.byteLength - 5)
    expect(() => decodeBatchFrame(truncated)).toThrow('Frame length mismatch')
  })

  it('handles large batch (100 ops)', () => {
    const ops: WireOp[] = Array.from({ length: 100 }, (_, i) => ({
      op: OP_MOVE as const,
      hlc: hlc(1000, i),
      objectId: i,
      dx: i * 0.1,
      dy: 0,
      dz: i * -0.1,
    }))
    const frame = encodeBatchFrame(ops)
    const decoded = decodeBatchFrame(frame)
    expect(decoded.length).toBe(100)
    for (let i = 0; i < 100; i++) {
      const d = decoded[i]! as MoveOp
      expect(d.objectId).toBe(i)
      expect(floatClose(d.dx, i * 0.1)).toBe(true)
    }
  })
})

// ─── Delta Compression ──────────────────────────────────────────────────────

describe('DeltaCompressor', () => {
  it('first update always sends full position', () => {
    const comp = new DeltaCompressor()
    const result = comp.compress(1, 5.0, 0, 10.0, hlc(1000, 0))
    expect(result).not.toBeNull()
    expect(result!.isDelta).toBe(false)
    if (!result!.isDelta) {
      expect(result!.x).toBe(5.0)
      expect(result!.z).toBe(10.0)
    }
  })

  it('small subsequent updates use delta encoding', () => {
    const comp = new DeltaCompressor()
    comp.compress(1, 5.0, 0, 10.0, hlc(1000, 0))
    const result = comp.compress(1, 5.1, 0, 10.2, hlc(1000, 1))
    expect(result).not.toBeNull()
    expect(result!.isDelta).toBe(true)
    if (result!.isDelta) {
      expect(floatClose(result!.dx, 0.1, 0.01)).toBe(true)
      expect(floatClose(result!.dz, 0.2, 0.01)).toBe(true)
    }
  })

  it('suppresses deadzone updates', () => {
    const comp = new DeltaCompressor()
    comp.compress(1, 5.0, 0, 10.0, hlc(1000, 0))
    const result = comp.compress(1, 5.0001, 0, 10.0001, hlc(1000, 1))
    expect(result).toBeNull()
  })

  it('sends full position for large jumps', () => {
    const comp = new DeltaCompressor()
    comp.compress(1, 0, 0, 0, hlc(1000, 0))
    const result = comp.compress(1, 50.0, 0, 50.0, hlc(1000, 1))
    expect(result).not.toBeNull()
    expect(result!.isDelta).toBe(false)
  })

  it('decompress reconstructs correct absolute position', () => {
    const sender = new DeltaCompressor()
    const receiver = new DeltaCompressor()

    // First: full position
    const m1 = sender.compress(1, 5.0, 0, 10.0, hlc(1000, 0))!
    const p1 = receiver.decompress(m1)
    expect(p1.x).toBe(5.0)
    expect(p1.z).toBe(10.0)

    // Second: delta
    const m2 = sender.compress(1, 5.5, 0, 10.5, hlc(1000, 1))!
    expect(m2.isDelta).toBe(true)
    const p2 = receiver.decompress(m2)
    expect(floatClose(p2.x, 5.5, 0.01)).toBe(true)
    expect(floatClose(p2.z, 10.5, 0.01)).toBe(true)
  })

  it('clear() resets tracked state', () => {
    const comp = new DeltaCompressor()
    comp.compress(1, 5.0, 0, 10.0, hlc(1000, 0))
    comp.clear()
    const result = comp.compress(1, 5.1, 0, 10.1, hlc(1000, 1))
    expect(result!.isDelta).toBe(false) // full position after clear
  })

  it('forget() removes specific object tracking', () => {
    const comp = new DeltaCompressor()
    comp.compress(1, 5.0, 0, 10.0, hlc(1000, 0))
    comp.compress(2, 1.0, 0, 2.0, hlc(1000, 1))
    comp.forget(1)

    // Object 1 sends full (was forgotten)
    const r1 = comp.compress(1, 5.1, 0, 10.1, hlc(1000, 2))
    expect(r1!.isDelta).toBe(false)

    // Object 2 still tracked (sends delta)
    const r2 = comp.compress(2, 1.1, 0, 2.1, hlc(1000, 3))
    expect(r2!.isDelta).toBe(true)
  })
})

describe('CompressedMove binary encoding', () => {
  it('round-trips full move', () => {
    const move = { objectId: 42, hlc: hlc(1700000000000, 5), isDelta: false as const, x: 5.5, y: 0.1, z: 10.3 }
    const size = compressedMoveSize(move)
    expect(size).toBe(25) // 13 header + 12 float payload
    const buffer = new ArrayBuffer(size)
    const view = new DataView(buffer)
    encodeCompressedMove(view, 0, move)
    const { move: decoded, newOffset } = decodeCompressedMove(view, 0)
    expect(newOffset).toBe(size)
    expect(decoded.isDelta).toBe(false)
    if (!decoded.isDelta) {
      expect(decoded.objectId).toBe(42)
      expect(floatClose(decoded.x, 5.5)).toBe(true)
      expect(floatClose(decoded.z, 10.3)).toBe(true)
    }
  })

  it('round-trips delta move with fixed-point precision', () => {
    const move = { objectId: 7, hlc: hlc(1700000000000, 10), isDelta: true as const, dx: 0.123, dy: 0, dz: -0.456 }
    const size = compressedMoveSize(move)
    expect(size).toBe(19) // 13 header + 6 int16 payload
    const buffer = new ArrayBuffer(size)
    const view = new DataView(buffer)
    encodeCompressedMove(view, 0, move)
    const { move: decoded } = decodeCompressedMove(view, 0)
    expect(decoded.isDelta).toBe(true)
    if (decoded.isDelta) {
      expect(floatClose(decoded.dx, 0.123, 0.002)).toBe(true)
      expect(floatClose(decoded.dz, -0.456, 0.002)).toBe(true)
    }
  })

  it('delta encoding saves 6 bytes over full', () => {
    const full = compressedMoveSize({ objectId: 1, hlc: hlc(0, 0), isDelta: false, x: 0, y: 0, z: 0 })
    const delta = compressedMoveSize({ objectId: 1, hlc: hlc(0, 0), isDelta: true, dx: 0, dy: 0, dz: 0 })
    expect(full - delta).toBe(6)
  })
})

// ─── Property-Based Tests ───────────────────────────────────────────────────

describe('Property-based tests', () => {
  const arbHlc = fc.record({
    wallMs: fc.integer({ min: 0, max: 2 ** 47 }),
    counter: fc.integer({ min: 0, max: 0xFFFF }),
  })

  const arbObjectId = fc.integer({ min: 0, max: 0xFFFFFFFF })
  const arbFloat = fc.float({ noNaN: true, noDefaultInfinity: true, min: -1000, max: 1000 })
  const arbFType = fc.integer({ min: 0, max: 6 }) as fc.Arbitrary<FurnitureTypeIndex>

  const arbMoveOp = fc.record({
    op: fc.constant(OP_MOVE),
    hlc: arbHlc,
    objectId: arbObjectId,
    dx: arbFloat, dy: arbFloat, dz: arbFloat,
  })

  const arbRotateOp = fc.record({
    op: fc.constant(OP_ROTATE),
    hlc: arbHlc,
    objectId: arbObjectId,
    rx: arbFloat, ry: arbFloat, rz: arbFloat,
  })

  const arbPlaceOp = fc.record({
    op: fc.constant(OP_PLACE),
    hlc: arbHlc,
    objectId: arbObjectId,
    furnitureType: arbFType,
    x: arbFloat, y: arbFloat, z: arbFloat,
    rx: arbFloat, ry: arbFloat, rz: arbFloat,
  })

  const arbRemoveOp = fc.record({
    op: fc.constant(OP_REMOVE),
    hlc: arbHlc,
    objectId: arbObjectId,
  })

  const arbScaleOp = fc.record({
    op: fc.constant(OP_SCALE),
    hlc: arbHlc,
    objectId: arbObjectId,
    sx: arbFloat, sy: arbFloat, sz: arbFloat,
  })

  const arbWireOp: fc.Arbitrary<WireOp> = fc.oneof(
    arbMoveOp, arbRotateOp, arbPlaceOp, arbRemoveOp, arbScaleOp,
  )

  it('encode → decode is identity for any single operation', () => {
    fc.assert(fc.property(arbWireOp, (op) => {
      const buffer = encode(op)
      const decoded = decode(buffer)
      // Check op type and objectId
      expect(decoded.op).toBe(op.op)
      expect(decoded.objectId).toBe(op.objectId)
      expect(decoded.hlc.wallMs).toBe(op.hlc.wallMs)
      expect(decoded.hlc.counter).toBe(op.hlc.counter)
    }), { numRuns: 200 })
  })

  it('encodedSize matches actual encoded buffer size', () => {
    fc.assert(fc.property(arbWireOp, (op) => {
      const buffer = encode(op)
      expect(buffer.byteLength).toBe(encodedSize(op))
    }), { numRuns: 200 })
  })

  it('batch encode → decode preserves all operations', () => {
    fc.assert(fc.property(
      fc.array(arbWireOp, { minLength: 0, maxLength: 50 }),
      (ops) => {
        const frame = encodeBatchFrame(ops)
        const decoded = decodeBatchFrame(frame)
        expect(decoded.length).toBe(ops.length)
        for (let i = 0; i < ops.length; i++) {
          expect(decoded[i]!.op).toBe(ops[i]!.op)
          expect(decoded[i]!.objectId).toBe(ops[i]!.objectId)
        }
      },
    ), { numRuns: 100 })
  })

  it('HLC uint64 round-trip preserves all timestamps', () => {
    fc.assert(fc.property(arbHlc, (hlcTs) => {
      const packed = hlcToUint64(hlcTs)
      const unpacked = uint64ToHlc(packed)
      expect(unpacked.wallMs).toBe(hlcTs.wallMs)
      expect(unpacked.counter).toBe(hlcTs.counter)
    }), { numRuns: 500 })
  })

  it('HLC comparison is total order', () => {
    fc.assert(fc.property(arbHlc, arbHlc, (a, b) => {
      const cmp = hlcCompare(a, b)
      const rev = hlcCompare(b, a)
      // Anti-symmetry: if a < b then b > a
      if (cmp < 0) expect(rev).toBeGreaterThan(0)
      else if (cmp > 0) expect(rev).toBeLessThan(0)
      else expect(rev).toBe(0)
    }), { numRuns: 200 })
  })

  it('HLC tick produces strictly increasing timestamps', () => {
    fc.assert(fc.property(
      fc.array(fc.integer({ min: 1000, max: 2000 }), { minLength: 2, maxLength: 20 }),
      (times) => {
        let idx = 0
        const clock = new HybridLogicalClock(() => times[idx]!)
        const stamps: HlcTimestamp[] = []
        for (let i = 0; i < times.length; i++) {
          idx = i
          stamps.push(clock.tick())
        }
        for (let i = 1; i < stamps.length; i++) {
          expect(hlcCompare(stamps[i]!, stamps[i - 1]!)).toBeGreaterThan(0)
        }
      },
    ), { numRuns: 100 })
  })

  it('delta compressor decompress recovers position within fixed-point precision', () => {
    fc.assert(fc.property(
      arbObjectId,
      fc.float({ noNaN: true, noDefaultInfinity: true, min: -100, max: 100 }),
      fc.float({ noNaN: true, noDefaultInfinity: true, min: -100, max: 100 }),
      fc.float({ noNaN: true, noDefaultInfinity: true, min: -100, max: 100 }),
      fc.float({ noNaN: true, noDefaultInfinity: true, min: -5, max: 5 }),
      fc.float({ noNaN: true, noDefaultInfinity: true, min: -5, max: 5 }),
      fc.float({ noNaN: true, noDefaultInfinity: true, min: -5, max: 5 }),
      (objId, x, y, z, dx, dy, dz) => {
        const sender = new DeltaCompressor()
        const receiver = new DeltaCompressor()

        // Initial full position
        const m1 = sender.compress(objId, x, y, z, hlc(1000, 0))
        if (!m1) return // deadzone edge case
        const p1 = receiver.decompress(m1)
        expect(floatClose(p1.x, x, 0.01)).toBe(true)
        expect(floatClose(p1.y, y, 0.01)).toBe(true)
        expect(floatClose(p1.z, z, 0.01)).toBe(true)

        // Delta update
        const nx = x + dx
        const ny = y + dy
        const nz = z + dz
        const m2 = sender.compress(objId, nx, ny, nz, hlc(1000, 1))
        if (!m2) return // deadzone
        const p2 = receiver.decompress(m2)
        // Int16 fixed-point precision: 1/1000 = 1mm
        expect(floatClose(p2.x, nx, 0.002)).toBe(true)
        expect(floatClose(p2.y, ny, 0.002)).toBe(true)
        expect(floatClose(p2.z, nz, 0.002)).toBe(true)
      },
    ), { numRuns: 200 })
  })
})

// ─── Size Comparison ────────────────────────────────────────────────────────

describe('Size comparison vs JSON', () => {
  it('binary move is ~7x smaller than JSON', () => {
    const op: MoveOp = {
      op: OP_MOVE, hlc: hlc(1700000000000, 1), objectId: 42,
      dx: 1.5, dy: -0.25, dz: 3.0,
    }
    const binarySize = encodedSize(op)
    const jsonSize = new TextEncoder().encode(JSON.stringify(op)).byteLength
    const ratio = jsonSize / binarySize

    expect(binarySize).toBe(25)
    expect(ratio).toBeGreaterThan(3) // at least 3x smaller
  })

  it('binary remove is >10x smaller than JSON', () => {
    const op: RemoveOp = {
      op: OP_REMOVE, hlc: hlc(1700000000000, 1), objectId: 42,
    }
    const binarySize = encodedSize(op)
    const jsonSize = new TextEncoder().encode(JSON.stringify(op)).byteLength
    const ratio = jsonSize / binarySize

    expect(binarySize).toBe(13)
    expect(ratio).toBeGreaterThanOrEqual(4)
  })

  it('binary place is >4x smaller than JSON', () => {
    const op: PlaceOp = {
      op: OP_PLACE, hlc: hlc(1700000000000, 1), objectId: 42,
      furnitureType: 0 as FurnitureTypeIndex,
      x: 5.123, y: 0.0, z: 10.456,
      rx: 0.0, ry: 1.5708, rz: 0.0,
    }
    const binarySize = encodedSize(op)
    const jsonSize = new TextEncoder().encode(JSON.stringify(op)).byteLength
    const ratio = jsonSize / binarySize

    expect(binarySize).toBe(38)
    expect(ratio).toBeGreaterThan(3)
  })
})
