import { describe, it, expect, vi } from 'vitest'
import fc from 'fast-check'
import { IncrementalGraph } from '../incremental'

// ─── Unit Tests ─────────────────────────────────────────────────────────────

describe('IncrementalGraph', () => {
  describe('input nodes', () => {
    it('creates an input with an initial value', () => {
      const graph = new IncrementalGraph()
      const n = graph.input(42)
      expect(n.value).toBe(42)
      expect(n.kind).toBe('input')
    })

    it('setting same value is a no-op', () => {
      const graph = new IncrementalGraph()
      const n = graph.input(10)
      const derived = graph.derive([n], (x) => x * 2)
      graph.set(n, 10) // same value
      graph.stabilize()
      expect(derived.recomputeCount).toBe(0)
    })
  })

  describe('derived nodes', () => {
    it('computes initial value from dependencies', () => {
      const graph = new IncrementalGraph()
      const a = graph.input(3)
      const b = graph.input(4)
      const sum = graph.derive([a, b], (x, y) => x + y)
      expect(sum.value).toBe(7)
    })

    it('recomputes when input changes', () => {
      const graph = new IncrementalGraph()
      const a = graph.input(3)
      const doubled = graph.derive([a], (x) => x * 2)
      expect(doubled.value).toBe(6)

      graph.set(a, 5)
      graph.stabilize()
      expect(doubled.value).toBe(10)
    })

    it('chains recompute through multiple layers', () => {
      const graph = new IncrementalGraph()
      const a = graph.input(2)
      const b = graph.derive([a], (x) => x + 1) // 3
      const c = graph.derive([b], (x) => x * 10) // 30
      expect(c.value).toBe(30)

      graph.set(a, 5)
      graph.stabilize()
      expect(b.value).toBe(6)
      expect(c.value).toBe(60)
    })

    it('handles diamond dependencies', () => {
      const graph = new IncrementalGraph()
      const a = graph.input(1)
      const b = graph.derive([a], (x) => x + 1) // 2
      const c = graph.derive([a], (x) => x * 2) // 2
      const d = graph.derive([b, c], (x, y) => x + y) // 4
      expect(d.value).toBe(4)

      graph.set(a, 10)
      graph.stabilize()
      expect(b.value).toBe(11)
      expect(c.value).toBe(20)
      expect(d.value).toBe(31)
    })
  })

  describe('cutoff', () => {
    it('stops propagation when value unchanged', () => {
      const graph = new IncrementalGraph()
      const a = graph.input(5)
      // This derived node clamps to a range — changing a within range produces same output
      const clamped = graph.derive([a], (x) => Math.min(Math.max(x, 0), 10))
      const downstream = graph.derive([clamped], (x) => x.toString())

      graph.set(a, 7) // still in range, clamped = 7 (changed)
      graph.stabilize()
      expect(clamped.value).toBe(7)
      expect(downstream.value).toBe('7')

      graph.resetCounters()
      graph.set(a, 15) // clamped to 10
      graph.stabilize()
      expect(clamped.value).toBe(10)

      graph.resetCounters()
      graph.set(a, 20) // still clamped to 10 — cutoff!
      graph.stabilize()
      expect(clamped.value).toBe(10)
      expect(downstream.recomputeCount).toBe(0) // cutoff prevented recompute
    })

    it('custom equality function for cutoff', () => {
      const graph = new IncrementalGraph()
      const a = graph.input({ x: 1, y: 2 })
      // Use structural equality
      const derived = graph.derive(
        [a],
        (pos) => ({ x: pos.x, y: pos.y }),
        (a, b) => a.x === b.x && a.y === b.y,
      )
      const downstream = graph.derive([derived], (pos) => `${pos.x},${pos.y}`)

      graph.resetCounters()
      graph.set(a, { x: 1, y: 2 }) // same structural value
      graph.stabilize()
      // derived recomputed but produced same value, so cutoff
      expect(downstream.recomputeCount).toBe(0)
    })
  })

  describe('observer nodes', () => {
    it('triggers effect when value changes', () => {
      const graph = new IncrementalGraph()
      const a = graph.input(1)
      const effects: number[] = []
      graph.observe([a], (x) => x * 10, (v) => effects.push(v))

      graph.set(a, 2)
      graph.stabilize()
      expect(effects).toEqual([20])

      graph.set(a, 3)
      graph.stabilize()
      expect(effects).toEqual([20, 30])
    })

    it('does not trigger effect on cutoff', () => {
      const graph = new IncrementalGraph()
      const a = graph.input(5)
      const effects: number[] = []
      graph.observe([a], (x) => Math.min(x, 10), (v) => effects.push(v))

      graph.set(a, 15) // clamped to 10
      graph.stabilize()
      expect(effects).toEqual([10])

      graph.set(a, 20) // still clamped to 10 — cutoff
      graph.stabilize()
      expect(effects).toEqual([10]) // no new effect
    })
  })

  describe('batch stabilization', () => {
    it('multiple input changes with single stabilize', () => {
      const graph = new IncrementalGraph()
      const a = graph.input(1)
      const b = graph.input(2)
      const sum = graph.derive([a, b], (x, y) => x + y)

      graph.set(a, 10)
      graph.set(b, 20)
      graph.stabilize()
      expect(sum.value).toBe(30)
    })

    it('stabilize count tracks calls', () => {
      const graph = new IncrementalGraph()
      const a = graph.input(1)
      graph.derive([a], (x) => x)

      expect(graph.stabilizeCount).toBe(0)
      graph.stabilize()
      expect(graph.stabilizeCount).toBe(1)
      graph.stabilize()
      expect(graph.stabilizeCount).toBe(2)
    })
  })

  describe('graph metrics', () => {
    it('tracks total recomputation count', () => {
      const graph = new IncrementalGraph()
      const a = graph.input(1)
      const b = graph.derive([a], (x) => x + 1)
      const c = graph.derive([b], (x) => x * 2)

      graph.set(a, 5)
      graph.stabilize()
      expect(graph.totalRecomputations()).toBe(2) // b and c both recomputed

      graph.set(a, 10)
      graph.stabilize()
      expect(graph.totalRecomputations()).toBe(4) // cumulative
    })

    it('resetCounters clears all counters', () => {
      const graph = new IncrementalGraph()
      const a = graph.input(1)
      graph.derive([a], (x) => x + 1)

      graph.set(a, 2)
      graph.stabilize()
      expect(graph.totalRecomputations()).toBe(1)

      graph.resetCounters()
      expect(graph.totalRecomputations()).toBe(0)
    })

    it('size tracks all nodes', () => {
      const graph = new IncrementalGraph()
      expect(graph.size).toBe(0)
      const a = graph.input(1)
      expect(graph.size).toBe(1)
      graph.derive([a], (x) => x)
      expect(graph.size).toBe(2)
    })
  })
})

// ─── Property-Based Tests (T5) for Incremental Framework ────────────────────

describe('IncrementalGraph — property-based tests', () => {
  it('consistency: stabilize produces same result as recompute-from-scratch', () => {
    fc.assert(fc.property(
      fc.integer({ min: -1000, max: 1000 }),
      fc.integer({ min: -1000, max: 1000 }),
      fc.integer({ min: -1000, max: 1000 }),
      (a, b, c) => {
        const graph = new IncrementalGraph()
        const inA = graph.input(0)
        const inB = graph.input(0)
        const sum = graph.derive([inA, inB], (x, y) => x + y)
        const product = graph.derive([inA, inB], (x, y) => x * y)
        const combined = graph.derive([sum, product], (s, p) => s + p)

        // Apply changes and stabilize
        graph.set(inA, a)
        graph.set(inB, b)
        graph.stabilize()

        graph.set(inA, c)
        graph.stabilize()

        // Verify against from-scratch computation.
        // Use === (not Object.is) since the framework's cutoff uses ===,
        // e.g. 0 === -0 is true, so cutoff may preserve 0 instead of -0.
        expect(sum.value === (c + b)).toBe(true)
        expect(product.value === (c * b)).toBe(true)
        expect(combined.value === ((c + b) + (c * b))).toBe(true)
      },
    ), { numRuns: 200 })
  })

  it('idempotency: stabilize twice without changes = zero recomputations', () => {
    fc.assert(fc.property(
      fc.integer({ min: -100, max: 100 }),
      (initial) => {
        const graph = new IncrementalGraph()
        const a = graph.input(initial)
        const b = graph.derive([a], (x) => x * 2)
        const c = graph.derive([b], (x) => x + 1)

        graph.set(a, initial + 1)
        graph.stabilize()

        graph.resetCounters()
        graph.stabilize() // second stabilize, no changes
        expect(graph.totalRecomputations()).toBe(0)
      },
    ), { numRuns: 100 })
  })

  it('minimality: only affected subgraph recomputes', () => {
    fc.assert(fc.property(
      fc.integer({ min: 1, max: 100 }),
      fc.integer({ min: 1, max: 100 }),
      (valA, valB) => {
        const graph = new IncrementalGraph()
        const a = graph.input(0)
        const b = graph.input(0)
        const derivedA = graph.derive([a], (x) => x * 2)
        const derivedB = graph.derive([b], (x) => x + 1)
        const combined = graph.derive([derivedA, derivedB], (x, y) => x + y)

        // Change only a
        graph.set(a, valA)
        graph.resetCounters()
        graph.stabilize()

        // derivedA and combined should recompute, but NOT derivedB
        expect(derivedA.recomputeCount).toBe(1)
        expect(derivedB.recomputeCount).toBe(0) // not in affected subgraph
        expect(combined.recomputeCount).toBe(1)
      },
    ), { numRuns: 100 })
  })

  it('commutativity: setting A then B = same result as B then A', () => {
    fc.assert(fc.property(
      fc.integer({ min: -100, max: 100 }),
      fc.integer({ min: -100, max: 100 }),
      (valA, valB) => {
        // Graph 1: set A then B
        const g1 = new IncrementalGraph()
        const a1 = g1.input(0)
        const b1 = g1.input(0)
        const sum1 = g1.derive([a1, b1], (x, y) => x + y)
        g1.set(a1, valA)
        g1.stabilize()
        g1.set(b1, valB)
        g1.stabilize()

        // Graph 2: set B then A
        const g2 = new IncrementalGraph()
        const a2 = g2.input(0)
        const b2 = g2.input(0)
        const sum2 = g2.derive([a2, b2], (x, y) => x + y)
        g2.set(b2, valB)
        g2.stabilize()
        g2.set(a2, valA)
        g2.stabilize()

        expect(sum1.value).toBe(sum2.value)
      },
    ), { numRuns: 100 })
  })
})
