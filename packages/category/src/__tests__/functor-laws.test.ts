/**
 * Functor law tests for all functors in the category package.
 */

import { describe, test, expect } from 'vitest'
import fc from 'fast-check'
import { id, compose } from '../core'
import type { Morphism } from '../core'
import {
  createFunctor, composeFunctors,
} from '../functor'
import {
  createEventSourceFunctor, verifyEventFunctorComposition, foldEvents,
} from '../event-source-functor'
import type { AggregateState, ProjectedView } from '../event-source-functor'
import {
  SpatialFunctor, verifySpatialFunctorLaw,
} from '../spatial-functor'
import type { Object2D, Operation2D } from '../spatial-functor'
import {
  createCodec, createSerializationFunctor,
  verifyRoundTrip, verifyCommutingDiagram,
  withEnvelope,
} from '../serialization-functor'

// ─── Generic Functor Laws ───────────────────────────────────────────────────

describe('CT-2: Generic Functor', () => {
  const F = createFunctor('Identity')

  test('identity law: F(id) ≡ id', () => {
    fc.assert(fc.property(
      fc.integer(),
      (x) => {
        const fa = F.of(x)
        const mapped = F.map(fa, id<number>())
        return mapped.value === fa.value
      },
    ))
  })

  test('composition law: F(g ∘ f) ≡ F(g) ∘ F(f)', () => {
    const f = (x: number) => x * 2
    const g = (x: number) => x + 1
    fc.assert(fc.property(
      fc.integer({ min: -100, max: 100 }),
      (x) => {
        const fa = F.of(x)
        const composed = F.map(fa, compose(f, g))
        const sequential = F.map(F.map(fa, f), g)
        return composed.value === sequential.value
      },
    ))
  })

  test('functor composition', () => {
    const G = createFunctor('Box')
    const FG = composeFunctors(F, G)
    const value = FG.of(42)
    expect(value.value.value).toBe(42)

    const mapped = FG.map(value, (x: number) => x * 2)
    expect(mapped.value.value).toBe(84)
  })
})

// ─── Event Source Functor ───────────────────────────────────────────────────

describe('CT-2: EventSourceFunctor', () => {
  type CounterState = AggregateState & { readonly data: { readonly count: number } }
  type CounterView = ProjectedView & { readonly data: { readonly displayCount: string } }

  const project = (state: CounterState): CounterView => ({
    viewType: 'counter',
    data: { displayCount: `Count: ${state.data.count}` },
    lastEventVersion: state.version,
  })

  const mapEventToUpdate = (applyEvent: Morphism<CounterState, CounterState>) =>
    (view: CounterView): CounterView => {
      // Re-derive from state change
      const mockState: CounterState = {
        version: view.lastEventVersion,
        data: { count: parseInt(view.data.displayCount.replace('Count: ', ''), 10) },
      }
      const newState = applyEvent(mockState)
      return project(newState)
    }

  const functor = createEventSourceFunctor<CounterState, CounterView>(project, mapEventToUpdate)

  const increment: Morphism<CounterState, CounterState> = (s) => ({
    ...s,
    version: s.version + 1,
    data: { count: s.data.count + 1 },
  })

  const decrement: Morphism<CounterState, CounterState> = (s) => ({
    ...s,
    version: s.version + 1,
    data: { count: s.data.count - 1 },
  })

  const initialState: CounterState = { version: 0, data: { count: 0 } }

  test('projects aggregate state to view', () => {
    const view = functor.project(initialState)
    expect(view.data.displayCount).toBe('Count: 0')
  })

  test('maps events to view updates', () => {
    const view = functor.project(initialState)
    const update = functor.mapEvent(increment)
    const newView = update(view)
    expect(newView.data.displayCount).toBe('Count: 1')
  })

  test('functor composition law for events', () => {
    const viewEquals = (a: CounterView, b: CounterView) =>
      a.data.displayCount === b.data.displayCount

    expect(verifyEventFunctorComposition(
      functor, increment, decrement, initialState, viewEquals,
    )).toBe(true)
  })

  test('foldEvents produces correct final view', () => {
    const events = [increment, increment, increment, decrement]
    const view = foldEvents(events, initialState, project)
    expect(view.data.displayCount).toBe('Count: 2')
  })
})

// ─── Spatial Functor ────────────────────────────────────────────────────────

describe('CT-2: SpatialFunctor', () => {
  const chair2D: Object2D = {
    id: 'chair-1',
    type: 'chair',
    x: 5,
    z: 3,
    rotation: Math.PI / 4,
    width: 0.5,
    depth: 0.5,
  }

  test('maps 2D object to 3D', () => {
    const obj3D = SpatialFunctor.mapObject(chair2D)
    expect(obj3D.x).toBe(5)
    expect(obj3D.y).toBe(0)
    expect(obj3D.z).toBe(3)
    expect(obj3D.rotationY).toBe(Math.PI / 4)
    expect(obj3D.height).toBe(0.85)  // chair height
  })

  test('maps 2D move to 3D move', () => {
    const op2D: Operation2D = { kind: 'move', dx: 2, dz: -1 }
    const op3D = SpatialFunctor.mapOperation(op2D)
    expect(op3D.kind).toBe('move')
    if (op3D.kind === 'move') {
      expect(op3D.dx).toBe(2)
      expect(op3D.dy).toBe(0)
      expect(op3D.dz).toBe(-1)
    }
  })

  test('maps 2D rotate to 3D rotate', () => {
    const op2D: Operation2D = { kind: 'rotate', dTheta: Math.PI / 2 }
    const op3D = SpatialFunctor.mapOperation(op2D)
    expect(op3D.kind).toBe('rotate')
    if (op3D.kind === 'rotate') {
      expect(op3D.dx).toBe(0)
      expect(op3D.dy).toBe(Math.PI / 2)
      expect(op3D.dz).toBe(0)
    }
  })

  test('functor law: operations commute with mapping', () => {
    const move: Operation2D = { kind: 'move', dx: 1, dz: 2 }
    const rotate: Operation2D = { kind: 'rotate', dTheta: 0.5 }
    expect(verifySpatialFunctorLaw(chair2D, move, rotate)).toBe(true)
  })

  test('functor law holds for arbitrary operations (property-based)', () => {
    const arbObj2D = fc.record({
      id: fc.string(),
      type: fc.constantFrom('chair', 'round-table', 'rect-table'),
      x: fc.float({ min: -50, max: 50, noNaN: true, noDefaultInfinity: true }),
      z: fc.float({ min: -50, max: 50, noNaN: true, noDefaultInfinity: true }),
      rotation: fc.float({ min: 0, max: Math.fround(2 * Math.PI), noNaN: true, noDefaultInfinity: true }),
      width: fc.float({ min: Math.fround(0.1), max: 5, noNaN: true, noDefaultInfinity: true }),
      depth: fc.float({ min: Math.fround(0.1), max: 5, noNaN: true, noDefaultInfinity: true }),
    })

    const arbOp2D = fc.oneof(
      fc.record({ kind: fc.constant('move' as const), dx: fc.float({ min: -10, max: 10, noNaN: true, noDefaultInfinity: true }), dz: fc.float({ min: -10, max: 10, noNaN: true, noDefaultInfinity: true }) }),
      fc.record({ kind: fc.constant('rotate' as const), dTheta: fc.float({ min: Math.fround(-Math.PI), max: Math.fround(Math.PI), noNaN: true, noDefaultInfinity: true }) }),
    )

    fc.assert(fc.property(arbObj2D, arbOp2D, arbOp2D, (obj, op1, op2) =>
      verifySpatialFunctorLaw(obj, op1, op2),
    ))
  })

  test('scene mapping preserves bounds', () => {
    const scene = SpatialFunctor.mapScene(
      { objects: [chair2D], bounds: { width: 20, depth: 15 } },
      3.5,
    )
    expect(scene.bounds.width).toBe(20)
    expect(scene.bounds.depth).toBe(15)
    expect(scene.bounds.height).toBe(3.5)
    expect(scene.objects).toHaveLength(1)
  })
})

// ─── Serialization Functor ──────────────────────────────────────────────────

describe('CT-2: SerializationFunctor', () => {
  interface Point { x: number; y: number }
  type WirePoint = { readonly x: number; readonly y: number }

  const pointCodec = createCodec<Point, WirePoint>(
    'Point',
    1,
    (p) => ({ x: p.x, y: p.y }),
    (w) => ({ x: w.x, y: w.y }),
  )

  const pointFunctor = createSerializationFunctor(pointCodec)

  test('round-trip law: deserialize(serialize(x)) ≡ x', () => {
    fc.assert(fc.property(
      fc.record({ x: fc.integer(), y: fc.integer() }),
      (point) => verifyRoundTrip(pointCodec, point, (a, b) => a.x === b.x && a.y === b.y),
    ))
  })

  test('commuting diagram: serialize(f(x)) ≡ mapMorphism(f)(serialize(x))', () => {
    const translate: Morphism<Point, Point> = (p) => ({ x: p.x + 1, y: p.y + 2 })

    fc.assert(fc.property(
      fc.record({ x: fc.integer(), y: fc.integer() }),
      (point) => verifyCommutingDiagram(
        pointFunctor,
        translate,
        point,
        (a, b) => a.x === b.x && a.y === b.y,
      ),
    ))
  })

  test('envelope wrapping', () => {
    const enveloped = withEnvelope(pointCodec)
    const wire = enveloped.serialize({ x: 1, y: 2 })
    expect(wire.type).toBe('Point')
    expect(wire.version).toBe(1)
    expect(wire.data).toEqual({ x: 1, y: 2 })

    const decoded = enveloped.deserialize(wire)
    expect(decoded).toEqual({ x: 1, y: 2 })
  })
})
