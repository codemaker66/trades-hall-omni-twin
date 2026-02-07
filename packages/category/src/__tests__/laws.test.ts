/**
 * Property-based tests verifying category laws.
 *
 * These tests use fast-check to verify that category laws hold
 * for ALL possible inputs, not just hand-picked examples.
 */

import { describe, test, expect } from 'vitest'
import fc from 'fast-check'
import {
  compose, compose3, id, ok, err, composeK, idK, liftK,
  pair, fst, snd, inl, inr, match,
  pipeline, kleisliPipeline,
} from '../core'
import type { Morphism, KleisliMorphism, Result } from '../core'
import {
  checkAssociativity, checkLeftIdentity, checkRightIdentity,
  checkKleisliAssociativity, checkKleisliLeftIdentity, checkKleisliRightIdentity,
  structuralEquals,
} from '../laws'

// ─── Helpers ────────────────────────────────────────────────────────────────

const double: Morphism<number, number> = (x) => x * 2
const addOne: Morphism<number, number> = (x) => x + 1
const square: Morphism<number, number> = (x) => x * x
const toString: Morphism<number, string> = (x) => String(x)

const safeDiv10: KleisliMorphism<number, number, string> = (x) =>
  x === 0 ? err('division by zero') : ok(10 / x)
const safeSqrt: KleisliMorphism<number, number, string> = (x) =>
  x < 0 ? err('negative sqrt') : ok(Math.sqrt(x))
const safeRecip: KleisliMorphism<number, number, string> = (x) =>
  x === 0 ? err('reciprocal of zero') : ok(1 / x)

// ─── Category Laws ──────────────────────────────────────────────────────────

describe('CT-1: Category Laws', () => {
  test('composition is associative (concrete)', () => {
    // compose(f, compose(g, h)) ≡ compose(compose(f, g), h)
    const lhs = compose(double, compose(addOne, square))
    const rhs = compose(compose(double, addOne), square)
    expect(lhs(3)).toBe(rhs(3))
    expect(lhs(0)).toBe(rhs(0))
    expect(lhs(-5)).toBe(rhs(-5))
  })

  test('composition is associative (property-based)', () => {
    fc.assert(fc.property(
      fc.integer({ min: -1000, max: 1000 }),
      (x) => checkAssociativity(double, addOne, square, x),
    ))
  })

  test('left identity: compose(id, f) ≡ f', () => {
    fc.assert(fc.property(
      fc.integer({ min: -1000, max: 1000 }),
      (x) => checkLeftIdentity(double, x),
    ))
  })

  test('right identity: compose(f, id) ≡ f', () => {
    fc.assert(fc.property(
      fc.integer({ min: -1000, max: 1000 }),
      (x) => checkRightIdentity(double, x),
    ))
  })

  test('compose3 is equivalent to nested compose', () => {
    fc.assert(fc.property(
      fc.integer({ min: -100, max: 100 }),
      (x) => {
        const c3 = compose3(double, addOne, square)
        const nested = compose(compose(double, addOne), square)
        return c3(x) === nested(x)
      },
    ))
  })

  test('identity morphism returns input unchanged', () => {
    fc.assert(fc.property(
      fc.anything(),
      (x) => id()(x) === x,
    ))
  })
})

// ─── Kleisli Category Laws ──────────────────────────────────────────────────

describe('CT-1: Kleisli Category Laws', () => {
  test('Kleisli composition is associative', () => {
    fc.assert(fc.property(
      fc.integer({ min: 1, max: 100 }),
      (x) => checkKleisliAssociativity(safeDiv10, safeSqrt, safeRecip, x),
    ))
  })

  test('Kleisli left identity: composeK(idK, f) ≡ f', () => {
    fc.assert(fc.property(
      fc.integer({ min: 1, max: 100 }),
      (x) => checkKleisliLeftIdentity(safeSqrt, x),
    ))
  })

  test('Kleisli right identity: composeK(f, idK) ≡ f', () => {
    fc.assert(fc.property(
      fc.integer({ min: 1, max: 100 }),
      (x) => checkKleisliRightIdentity(safeSqrt, x),
    ))
  })

  test('Kleisli composition short-circuits on error', () => {
    const fail: KleisliMorphism<number, number, string> = () => err('fail')
    const composed = composeK(fail, safeSqrt)
    const result = composed(42)
    expect(result.ok).toBe(false)
    if (!result.ok) expect(result.error).toBe('fail')
  })

  test('liftK lifts pure morphisms into Kleisli', () => {
    const lifted = liftK(double)
    const result = lifted(5)
    expect(result).toEqual(ok(10))
  })

  test('liftK preserves composition', () => {
    fc.assert(fc.property(
      fc.integer({ min: -100, max: 100 }),
      (x) => {
        const liftedCompose = composeK(liftK(double), liftK(addOne))
        const composedLift = liftK(compose(double, addOne))
        const r1 = liftedCompose(x)
        const r2 = composedLift(x)
        return r1.ok && r2.ok && r1.value === r2.value
      },
    ))
  })
})

// ─── Product and Coproduct ──────────────────────────────────────────────────

describe('CT-1: Product and Coproduct', () => {
  test('product projections', () => {
    const p = pair(42, 'hello')
    expect(fst(p)).toBe(42)
    expect(snd(p)).toBe('hello')
  })

  test('product universal property', () => {
    fc.assert(fc.property(
      fc.integer(),
      fc.string(),
      (n, s) => {
        const p = pair(n, s)
        return fst(p) === n && snd(p) === s
      },
    ))
  })

  test('coproduct injection and matching', () => {
    const left = inl(42)
    const right = inr('hello')

    const handler = match(
      (n: number) => `number: ${n}`,
      (s: string) => `string: ${s}`,
    )

    expect(handler(left)).toBe('number: 42')
    expect(handler(right)).toBe('string: hello')
  })

  test('coproduct universal property', () => {
    fc.assert(fc.property(
      fc.integer(),
      (n) => {
        const cp = inl(n)
        const f = match(
          (x: number) => x * 2,
          (_s: string) => 0,
        )
        return f(cp) === n * 2
      },
    ))
  })
})

// ─── Pipeline Builder ───────────────────────────────────────────────────────

describe('CT-1: Pipeline Builder', () => {
  test('pipeline composes morphisms', () => {
    const result = pipeline(double).then(addOne).then(square).run(3)
    // double(3)=6, addOne(6)=7, square(7)=49
    expect(result).toBe(49)
  })

  test('pipeline is equivalent to compose', () => {
    fc.assert(fc.property(
      fc.integer({ min: -100, max: 100 }),
      (x) => {
        const pipeResult = pipeline(double).then(addOne).run(x)
        const compResult = compose(double, addOne)(x)
        return pipeResult === compResult
      },
    ))
  })

  test('kleisli pipeline short-circuits', () => {
    const fail: KleisliMorphism<number, number, string> = () => err('fail')
    const result = pipeline(double)
      .thenK(fail)
      .thenK(safeSqrt)
      .run(5)
    expect(result.ok).toBe(false)
  })

  test('kleisli pipeline passes values through', () => {
    const result = pipeline(double)
      .thenK(safeSqrt)
      .run(8)
    expect(result.ok).toBe(true)
    if (result.ok) expect(result.value).toBe(4)
  })
})

// ─── Result ─────────────────────────────────────────────────────────────────

describe('CT-1: Result type', () => {
  test('ok wraps a value', () => {
    const r = ok(42)
    expect(r.ok).toBe(true)
    if (r.ok) expect(r.value).toBe(42)
  })

  test('err wraps an error', () => {
    const r = err('bad')
    expect(r.ok).toBe(false)
    if (!r.ok) expect(r.error).toBe('bad')
  })

  test('structuralEquals works for objects', () => {
    expect(structuralEquals({ a: 1, b: 2 }, { a: 1, b: 2 })).toBe(true)
    expect(structuralEquals({ a: 1 }, { a: 2 })).toBe(false)
  })
})
