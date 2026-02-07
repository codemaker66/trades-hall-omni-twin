import { describe, it, expect, beforeEach } from 'vitest'
import fc from 'fast-check'
import type { HlcTimestamp } from '@omni-twin/wire-protocol'

import type { Vec3, SpatialOp, ObjectSnapshot } from '../types'
import { vec3Add, vec3Eq, VEC3_ZERO, VEC3_ONE } from '../types'
import {
  createAddOp, createRemoveOp, createMoveOp, createRotateOp, createScaleOp,
  resetOpIdCounter, opCompare,
} from '../operation'
import { reconstructObject } from '../state'
import { mergeOpSets, opSetDifference } from '../merge'
import { SpatialDocument } from '../document'
import { fullSync, computeSyncMessage } from '../sync'

// ─── Helpers ────────────────────────────────────────────────────────────────

function hlc(wallMs: number, counter: number): HlcTimestamp {
  return { wallMs, counter }
}

function snapshotEq(a: ObjectSnapshot, b: ObjectSnapshot, eps = 1e-6): boolean {
  return a.id === b.id
    && a.furnitureType === b.furnitureType
    && a.alive === b.alive
    && vec3Eq(a.position, b.position, eps)
    && vec3Eq(a.rotation, b.rotation, eps)
    && vec3Eq(a.scale, b.scale, eps)
}

function documentsEqual(a: SpatialDocument, b: SpatialDocument, eps = 1e-6): boolean {
  const aObjs = a.objects()
  const bObjs = b.objects()
  if (aObjs.size !== bObjs.size) return false
  for (const [id, aSnap] of aObjs) {
    const bSnap = bObjs.get(id)
    if (!bSnap || !snapshotEq(aSnap, bSnap, eps)) return false
  }
  return true
}

beforeEach(() => {
  resetOpIdCounter()
})

// ─── Vec3 ───────────────────────────────────────────────────────────────────

describe('Vec3', () => {
  it('adds two vectors', () => {
    expect(vec3Add({ x: 1, y: 2, z: 3 }, { x: 4, y: 5, z: 6 })).toEqual({ x: 5, y: 7, z: 9 })
  })

  it('equality check with epsilon', () => {
    expect(vec3Eq({ x: 1, y: 2, z: 3 }, { x: 1.0000000001, y: 2, z: 3 })).toBe(true)
    expect(vec3Eq({ x: 1, y: 2, z: 3 }, { x: 1.1, y: 2, z: 3 })).toBe(false)
  })
})

// ─── Operation Creation ─────────────────────────────────────────────────────

describe('Operations', () => {
  it('creates ops with unique IDs', () => {
    const op1 = createAddOp('r1', hlc(1000, 0), 'obj1', 0, { x: 0, y: 0, z: 0 })
    const op2 = createMoveOp('r1', hlc(1000, 1), 'obj1', { x: 1, y: 0, z: 0 })
    expect(op1.opId).not.toBe(op2.opId)
    expect(op1.opId.startsWith('r1:')).toBe(true)
  })

  it('opCompare provides total order', () => {
    const a = createAddOp('r1', hlc(1000, 0), 'obj1', 0, VEC3_ZERO)
    const b = createAddOp('r2', hlc(1000, 1), 'obj1', 0, VEC3_ZERO)
    const c = createAddOp('r1', hlc(2000, 0), 'obj1', 0, VEC3_ZERO)

    expect(opCompare(a, b)).toBeLessThan(0)
    expect(opCompare(b, c)).toBeLessThan(0)
    expect(opCompare(a, a)).toBe(0)
  })
})

// ─── State Reconstruction ───────────────────────────────────────────────────

describe('State reconstruction', () => {
  it('returns null for no ops', () => {
    expect(reconstructObject('obj1', [])).toBeNull()
  })

  it('reconstructs from add op alone', () => {
    const add = createAddOp('r1', hlc(1000, 0), 'obj1', 2, { x: 5, y: 0, z: 10 })
    const snap = reconstructObject('obj1', [add])!
    expect(snap.alive).toBe(true)
    expect(snap.furnitureType).toBe(2)
    expect(vec3Eq(snap.position, { x: 5, y: 0, z: 10 })).toBe(true)
  })

  it('accumulates move displacements', () => {
    const add = createAddOp('r1', hlc(1000, 0), 'obj1', 0, { x: 0, y: 0, z: 0 })
    const m1 = createMoveOp('r1', hlc(1000, 1), 'obj1', { x: 1, y: 0, z: 0 })
    const m2 = createMoveOp('r2', hlc(1000, 2), 'obj1', { x: 0, y: 0, z: 2 })
    const snap = reconstructObject('obj1', [add, m1, m2])!
    expect(vec3Eq(snap.position, { x: 1, y: 0, z: 2 })).toBe(true)
  })

  it('accumulates rotation deltas', () => {
    const add = createAddOp('r1', hlc(1000, 0), 'obj1', 0, VEC3_ZERO)
    const r1 = createRotateOp('r1', hlc(1000, 1), 'obj1', { x: 0, y: Math.PI / 2, z: 0 })
    const r2 = createRotateOp('r2', hlc(1000, 2), 'obj1', { x: 0, y: Math.PI / 4, z: 0 })
    const snap = reconstructObject('obj1', [add, r1, r2])!
    expect(snap.rotation.y).toBeCloseTo(Math.PI * 3 / 4)
  })

  it('removes object when remove is later than add', () => {
    const add = createAddOp('r1', hlc(1000, 0), 'obj1', 0, VEC3_ZERO)
    const rem = createRemoveOp('r2', hlc(2000, 0), 'obj1')
    const snap = reconstructObject('obj1', [add, rem])!
    expect(snap.alive).toBe(false)
  })

  it('add wins when concurrent with remove (same HLC)', () => {
    const rem = createRemoveOp('r2', hlc(1000, 0), 'obj1')
    const add = createAddOp('r1', hlc(1000, 0), 'obj1', 0, VEC3_ZERO)
    const snap = reconstructObject('obj1', [rem, add])!
    // add wins when HLC is equal (add-wins semantics)
    expect(snap.alive).toBe(true)
  })

  it('order of ops does not affect reconstruction', () => {
    const add = createAddOp('r1', hlc(1000, 0), 'obj1', 0, { x: 5, y: 0, z: 5 })
    const m1 = createMoveOp('r1', hlc(1000, 1), 'obj1', { x: 1, y: 0, z: 0 })
    const m2 = createMoveOp('r2', hlc(1000, 2), 'obj1', { x: 0, y: 0, z: 1 })
    const rot = createRotateOp('r1', hlc(1000, 3), 'obj1', { x: 0, y: 1.57, z: 0 })

    const snap1 = reconstructObject('obj1', [add, m1, m2, rot])!
    const snap2 = reconstructObject('obj1', [rot, m2, add, m1])!
    const snap3 = reconstructObject('obj1', [m1, rot, add, m2])!

    expect(snapshotEq(snap1, snap2)).toBe(true)
    expect(snapshotEq(snap2, snap3)).toBe(true)
  })
})

// ─── Merge ──────────────────────────────────────────────────────────────────

describe('Merge', () => {
  it('union of disjoint op sets', () => {
    const a = new Map<string, SpatialOp>()
    const b = new Map<string, SpatialOp>()
    const op1 = createAddOp('r1', hlc(1000, 0), 'obj1', 0, VEC3_ZERO)
    const op2 = createAddOp('r2', hlc(1000, 1), 'obj2', 0, VEC3_ZERO)
    a.set(op1.opId, op1)
    b.set(op2.opId, op2)

    const merged = mergeOpSets(a, b)
    expect(merged.size).toBe(2)
    expect(merged.has(op1.opId)).toBe(true)
    expect(merged.has(op2.opId)).toBe(true)
  })

  it('union is idempotent', () => {
    const a = new Map<string, SpatialOp>()
    const op1 = createAddOp('r1', hlc(1000, 0), 'obj1', 0, VEC3_ZERO)
    a.set(op1.opId, op1)

    const merged = mergeOpSets(a, a)
    expect(merged.size).toBe(1)
  })

  it('opSetDifference finds missing ops', () => {
    const a = new Map<string, SpatialOp>()
    const b = new Map<string, SpatialOp>()
    const op1 = createAddOp('r1', hlc(1000, 0), 'obj1', 0, VEC3_ZERO)
    const op2 = createMoveOp('r2', hlc(1000, 1), 'obj1', { x: 1, y: 0, z: 0 })
    a.set(op1.opId, op1)
    b.set(op1.opId, op1)
    b.set(op2.opId, op2)

    const diff = opSetDifference(a, b)
    expect(diff.length).toBe(1)
    expect(diff[0]!.opId).toBe(op2.opId)
  })
})

// ─── SpatialDocument ────────────────────────────────────────────────────────

describe('SpatialDocument', () => {
  it('add and query objects', () => {
    const doc = new SpatialDocument('r1')
    doc.addObject(hlc(1000, 0), 'obj1', 0, { x: 5, y: 0, z: 10 })
    doc.addObject(hlc(1000, 1), 'obj2', 1, { x: 1, y: 0, z: 2 })

    expect(doc.size).toBe(2)
    const snap = doc.getObject('obj1')!
    expect(snap.position.x).toBe(5)
  })

  it('move displaces object', () => {
    const doc = new SpatialDocument('r1')
    doc.addObject(hlc(1000, 0), 'table', 0, { x: 0, y: 0, z: 0 })
    doc.moveObject(hlc(1000, 1), 'table', { x: 3, y: 0, z: 0 })
    doc.moveObject(hlc(1000, 2), 'table', { x: 0, y: 0, z: 2 })

    const snap = doc.getObject('table')!
    expect(snap.position.x).toBeCloseTo(3)
    expect(snap.position.z).toBeCloseTo(2)
  })

  it('remove makes object disappear', () => {
    const doc = new SpatialDocument('r1')
    doc.addObject(hlc(1000, 0), 'obj1', 0, VEC3_ZERO)
    expect(doc.size).toBe(1)

    doc.removeObject(hlc(2000, 0), 'obj1')
    expect(doc.size).toBe(0)
  })

  it('apply is idempotent', () => {
    const doc = new SpatialDocument('r1')
    const op = createAddOp('r1', hlc(1000, 0), 'obj1', 0, VEC3_ZERO)
    expect(doc.apply(op)).toBe(true)
    expect(doc.apply(op)).toBe(false) // already applied
    expect(doc.size).toBe(1)
  })

  it('incremental cache matches full rebuild', () => {
    const doc = new SpatialDocument('r1')
    doc.addObject(hlc(1000, 0), 'obj1', 0, { x: 5, y: 0, z: 5 })
    doc.moveObject(hlc(1000, 1), 'obj1', { x: 1, y: 0, z: 0 })
    doc.addObject(hlc(1000, 2), 'obj2', 1, { x: 0, y: 0, z: 0 })
    doc.rotateObject(hlc(1000, 3), 'obj2', { x: 0, y: 1.57, z: 0 })
    doc.removeObject(hlc(2000, 0), 'obj2')

    const cache = doc.objects()
    const rebuilt = doc.rebuild()

    expect(cache.size).toBe(rebuilt.size)
    for (const [id, snap] of cache) {
      const rSnap = rebuilt.get(id)!
      expect(snapshotEq(snap, rSnap)).toBe(true)
    }
  })
})

// ─── Document Merge ─────────────────────────────────────────────────────────

describe('Document merge', () => {
  it('concurrent moves on same object merge additively', () => {
    const docA = new SpatialDocument('rA')
    const docB = new SpatialDocument('rB')

    // Both start with the same object
    const addOp = createAddOp('rA', hlc(1000, 0), 'table', 0, { x: 0, y: 0, z: 0 })
    docA.apply(addOp)
    docB.apply(addOp)

    // Concurrent moves
    const moveA = createMoveOp('rA', hlc(1001, 0), 'table', { x: -5, y: 0, z: 0 }) // left
    const moveB = createMoveOp('rB', hlc(1001, 1), 'table', { x: 0, y: 0, z: 3 })  // forward

    docA.apply(moveA)
    docB.apply(moveB)

    // Before merge: each sees only their own move
    expect(docA.getObject('table')!.position.x).toBeCloseTo(-5)
    expect(docA.getObject('table')!.position.z).toBeCloseTo(0)
    expect(docB.getObject('table')!.position.x).toBeCloseTo(0)
    expect(docB.getObject('table')!.position.z).toBeCloseTo(3)

    // Merge
    docA.merge(docB)
    docB.merge(docA)

    // After merge: both intents preserved
    const snapA = docA.getObject('table')!
    const snapB = docB.getObject('table')!
    expect(snapA.position.x).toBeCloseTo(-5)
    expect(snapA.position.z).toBeCloseTo(3)
    expect(snapshotEq(snapA, snapB)).toBe(true)
  })

  it('concurrent move + rotate merge cleanly', () => {
    const docA = new SpatialDocument('rA')
    const docB = new SpatialDocument('rB')

    const addOp = createAddOp('rA', hlc(1000, 0), 'table', 0, { x: 5, y: 0, z: 5 })
    docA.apply(addOp)
    docB.apply(addOp)

    docA.apply(createMoveOp('rA', hlc(1001, 0), 'table', { x: 2, y: 0, z: 0 }))
    docB.apply(createRotateOp('rB', hlc(1001, 1), 'table', { x: 0, y: Math.PI / 2, z: 0 }))

    docA.merge(docB)
    docB.merge(docA)

    const snapA = docA.getObject('table')!
    expect(snapA.position.x).toBeCloseTo(7)
    expect(snapA.rotation.y).toBeCloseTo(Math.PI / 2)
    expect(documentsEqual(docA, docB)).toBe(true)
  })

  it('merge is idempotent (merging same doc twice)', () => {
    const docA = new SpatialDocument('rA')
    const docB = new SpatialDocument('rB')

    const addOp = createAddOp('rA', hlc(1000, 0), 'table', 0, VEC3_ZERO)
    docA.apply(addOp)
    docB.apply(addOp)

    docA.merge(docB)
    const snap1 = docA.getObject('table')!

    docA.merge(docB) // merge again
    const snap2 = docA.getObject('table')!

    expect(snapshotEq(snap1, snap2)).toBe(true)
  })
})

// ─── Delta Sync ─────────────────────────────────────────────────────────────

describe('Delta sync', () => {
  it('full sync converges two divergent documents', () => {
    const docA = new SpatialDocument('rA')
    const docB = new SpatialDocument('rB')

    const addOp = createAddOp('rA', hlc(1000, 0), 'obj1', 0, VEC3_ZERO)
    docA.apply(addOp)
    docB.apply(addOp)

    // Diverge
    docA.moveObject(hlc(1001, 0), 'obj1', { x: 1, y: 0, z: 0 })
    docB.moveObject(hlc(1001, 1), 'obj1', { x: 0, y: 0, z: 1 })
    docA.addObject(hlc(1002, 0), 'obj2', 1, { x: 5, y: 0, z: 5 })

    expect(docA.size).toBe(2)
    expect(docB.size).toBe(1)

    // Sync
    const { aToB, bToA } = fullSync(docA, docB)
    expect(aToB).toBeGreaterThan(0)
    expect(bToA).toBeGreaterThan(0)

    // Converged
    expect(documentsEqual(docA, docB)).toBe(true)
    expect(docA.size).toBe(2)
  })

  it('state vector accurately represents seen ops', () => {
    const doc = new SpatialDocument('r1')
    doc.addObject(hlc(1000, 0), 'obj1', 0, VEC3_ZERO)
    doc.moveObject(hlc(1000, 1), 'obj1', { x: 1, y: 0, z: 0 })

    const sv = doc.stateVector()
    expect(sv.get('r1')).toBeDefined()
    expect(sv.get('r1')!).toBeGreaterThanOrEqual(2)
  })

  it('computeSyncMessage returns only missing ops', () => {
    const docA = new SpatialDocument('rA')
    const docB = new SpatialDocument('rB')

    const addOp = createAddOp('rA', hlc(1000, 0), 'obj1', 0, VEC3_ZERO)
    docA.apply(addOp)
    docB.apply(addOp)

    // A has an extra move
    docA.moveObject(hlc(1001, 0), 'obj1', { x: 1, y: 0, z: 0 })

    const missing = computeSyncMessage(docA, docB.stateVector())
    expect(missing.length).toBe(1)
    expect(missing[0]!.type).toBe('move')
  })
})

// ─── Property-Based Tests ───────────────────────────────────────────────────

describe('CRDT Property-Based Tests', () => {
  // Arbitraries
  const arbVec3 = fc.record({
    x: fc.float({ noNaN: true, noDefaultInfinity: true, min: -100, max: 100 }),
    y: fc.float({ noNaN: true, noDefaultInfinity: true, min: -100, max: 100 }),
    z: fc.float({ noNaN: true, noDefaultInfinity: true, min: -100, max: 100 }),
  })

  const arbFType = fc.integer({ min: 0, max: 6 })

  // Generate a sequence of operations for a set of objects
  function arbOpSequence(replicaId: string, minOps: number, maxOps: number) {
    return fc.array(
      fc.oneof(
        fc.record({
          type: fc.constant('move' as const),
          objectId: fc.constantFrom('obj1', 'obj2', 'obj3'),
          delta: arbVec3,
        }),
        fc.record({
          type: fc.constant('rotate' as const),
          objectId: fc.constantFrom('obj1', 'obj2', 'obj3'),
          delta: arbVec3,
        }),
        fc.record({
          type: fc.constant('scale' as const),
          objectId: fc.constantFrom('obj1', 'obj2', 'obj3'),
          delta: arbVec3,
        }),
      ),
      { minLength: minOps, maxLength: maxOps },
    ).map(actions => {
      resetOpIdCounter()
      const ops: SpatialOp[] = []
      // Always start with add ops for all objects
      ops.push(createAddOp(replicaId, hlc(1000, 0), 'obj1', 0, VEC3_ZERO))
      ops.push(createAddOp(replicaId, hlc(1000, 1), 'obj2', 1, { x: 5, y: 0, z: 0 }))
      ops.push(createAddOp(replicaId, hlc(1000, 2), 'obj3', 2, { x: 0, y: 0, z: 5 }))

      for (let i = 0; i < actions.length; i++) {
        const a = actions[i]!
        const h = hlc(1001 + i, 0)
        switch (a.type) {
          case 'move': ops.push(createMoveOp(replicaId, h, a.objectId, a.delta)); break
          case 'rotate': ops.push(createRotateOp(replicaId, h, a.objectId, a.delta)); break
          case 'scale': ops.push(createScaleOp(replicaId, h, a.objectId, a.delta)); break
        }
      }
      return ops
    })
  }

  it('1. Strong Eventual Consistency: same ops in any order → identical state', () => {
    fc.assert(fc.property(
      arbOpSequence('r1', 3, 15),
      (ops) => {
        // Apply in original order
        const doc1 = new SpatialDocument('r1')
        for (const op of ops) doc1.apply(op)

        // Apply in reversed order
        const doc2 = new SpatialDocument('r2')
        const reversed = [...ops].reverse()
        for (const op of reversed) doc2.apply(op)

        // Apply in shuffled order
        const doc3 = new SpatialDocument('r3')
        const shuffled = [...ops].sort(() => Math.random() - 0.5)
        for (const op of shuffled) doc3.apply(op)

        // All three must have identical state
        expect(documentsEqual(doc1, doc2)).toBe(true)
        expect(documentsEqual(doc2, doc3)).toBe(true)
      },
    ), { numRuns: 100 })
  })

  it('2. Commutativity: merge(A, B) = merge(B, A)', () => {
    fc.assert(fc.property(
      arbOpSequence('rA', 2, 10),
      arbOpSequence('rB', 2, 10),
      (opsA, opsB) => {
        // Create doc with A's ops merged with B's
        const docAB = new SpatialDocument('x1')
        for (const op of opsA) docAB.apply(op)
        for (const op of opsB) docAB.apply(op)

        // Create doc with B's ops merged with A's
        const docBA = new SpatialDocument('x2')
        for (const op of opsB) docBA.apply(op)
        for (const op of opsA) docBA.apply(op)

        expect(documentsEqual(docAB, docBA)).toBe(true)
      },
    ), { numRuns: 100 })
  })

  it('3. Associativity: merge(merge(A, B), C) = merge(A, merge(B, C))', () => {
    fc.assert(fc.property(
      arbOpSequence('rA', 1, 8),
      arbOpSequence('rB', 1, 8),
      arbOpSequence('rC', 1, 8),
      (opsA, opsB, opsC) => {
        // (A ∪ B) ∪ C
        const docABC = new SpatialDocument('x1')
        for (const op of opsA) docABC.apply(op)
        for (const op of opsB) docABC.apply(op)
        for (const op of opsC) docABC.apply(op)

        // A ∪ (B ∪ C)
        const docABC2 = new SpatialDocument('x2')
        for (const op of opsB) docABC2.apply(op)
        for (const op of opsC) docABC2.apply(op)
        for (const op of opsA) docABC2.apply(op)

        expect(documentsEqual(docABC, docABC2)).toBe(true)
      },
    ), { numRuns: 50 })
  })

  it('4. Idempotency: merge(A, A) = A', () => {
    fc.assert(fc.property(
      arbOpSequence('r1', 2, 15),
      (ops) => {
        const doc1 = new SpatialDocument('r1')
        for (const op of ops) doc1.apply(op)

        const before = new Map<string, ObjectSnapshot>()
        for (const [id, snap] of doc1.objects()) {
          before.set(id, { ...snap, position: { ...snap.position }, rotation: { ...snap.rotation }, scale: { ...snap.scale } })
        }

        // Merge with self
        const doc2 = new SpatialDocument('r2')
        for (const op of ops) doc2.apply(op)
        doc1.merge(doc2)

        // State unchanged
        for (const [id, snapBefore] of before) {
          const snapAfter = doc1.getObject(id)!
          expect(snapshotEq(snapBefore, snapAfter)).toBe(true)
        }
      },
    ), { numRuns: 100 })
  })

  it('5. Convergence: replicas converge after bidirectional sync', () => {
    fc.assert(fc.property(
      fc.array(
        fc.record({
          replica: fc.constantFrom('rA', 'rB', 'rC'),
          action: fc.oneof(
            fc.record({
              type: fc.constant('move' as const),
              objectId: fc.constantFrom('obj1', 'obj2'),
              delta: arbVec3,
            }),
            fc.record({
              type: fc.constant('rotate' as const),
              objectId: fc.constantFrom('obj1', 'obj2'),
              delta: arbVec3,
            }),
          ),
        }),
        { minLength: 5, maxLength: 20 },
      ),
      (actions) => {
        resetOpIdCounter()
        const docA = new SpatialDocument('rA')
        const docB = new SpatialDocument('rB')
        const docC = new SpatialDocument('rC')

        // All start with same base objects
        const add1 = createAddOp('rA', hlc(1000, 0), 'obj1', 0, VEC3_ZERO)
        const add2 = createAddOp('rA', hlc(1000, 1), 'obj2', 1, { x: 5, y: 0, z: 5 })
        for (const doc of [docA, docB, docC]) {
          doc.apply(add1)
          doc.apply(add2)
        }

        // Apply random actions to random replicas
        for (let i = 0; i < actions.length; i++) {
          const { replica, action } = actions[i]!
          const doc = replica === 'rA' ? docA : replica === 'rB' ? docB : docC
          const h = hlc(1001 + i, 0)
          switch (action.type) {
            case 'move':
              doc.apply(createMoveOp(replica, h, action.objectId, action.delta))
              break
            case 'rotate':
              doc.apply(createRotateOp(replica, h, action.objectId, action.delta))
              break
          }
        }

        // Full sync all pairs
        fullSync(docA, docB)
        fullSync(docB, docC)
        fullSync(docA, docC)
        // Second pass to ensure full convergence
        fullSync(docA, docB)

        // All three must converge
        expect(documentsEqual(docA, docB)).toBe(true)
        expect(documentsEqual(docB, docC)).toBe(true)
      },
    ), { numRuns: 50 })
  })

  it('6. Displacement additivity: concurrent moves sum correctly', () => {
    fc.assert(fc.property(
      arbVec3, // initial position
      arbVec3, // move from replica A
      arbVec3, // move from replica B
      (initial, deltaA, deltaB) => {
        resetOpIdCounter()
        const docA = new SpatialDocument('rA')
        const docB = new SpatialDocument('rB')

        const addOp = createAddOp('rA', hlc(1000, 0), 'obj', 0, initial)
        docA.apply(addOp)
        docB.apply(addOp)

        docA.apply(createMoveOp('rA', hlc(1001, 0), 'obj', deltaA))
        docB.apply(createMoveOp('rB', hlc(1001, 1), 'obj', deltaB))

        fullSync(docA, docB)

        const expected = vec3Add(vec3Add(initial, deltaA), deltaB)
        const snapA = docA.getObject('obj')!
        expect(vec3Eq(snapA.position, expected, 1e-5)).toBe(true)
        expect(documentsEqual(docA, docB)).toBe(true)
      },
    ), { numRuns: 200 })
  })

  it('7. Incremental cache matches full rebuild after random ops', () => {
    fc.assert(fc.property(
      arbOpSequence('r1', 5, 20),
      (ops) => {
        const doc = new SpatialDocument('r1')
        for (const op of ops) doc.apply(op)

        const cache = doc.objects()
        const rebuilt = doc.rebuild()

        expect(cache.size).toBe(rebuilt.size)
        for (const [id, snap] of cache) {
          const rSnap = rebuilt.get(id)!
          expect(snapshotEq(snap, rSnap)).toBe(true)
        }
      },
    ), { numRuns: 100 })
  })

  it('8. Vec3 addition is commutative and associative', () => {
    fc.assert(fc.property(arbVec3, arbVec3, arbVec3, (a, b, c) => {
      // Commutativity
      expect(vec3Eq(vec3Add(a, b), vec3Add(b, a), 1e-10)).toBe(true)
      // Associativity
      expect(vec3Eq(vec3Add(vec3Add(a, b), c), vec3Add(a, vec3Add(b, c)), 1e-5)).toBe(true)
    }), { numRuns: 200 })
  })
})
