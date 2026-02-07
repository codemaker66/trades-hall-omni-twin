/**
 * CT-1: Category Law Checkers
 *
 * These utilities verify the fundamental laws of our category:
 *   1. Associativity: compose(f, compose(g, h)) ≡ compose(compose(f, g), h)
 *   2. Left identity:  compose(id, f) ≡ f
 *   3. Right identity: compose(f, id) ≡ f
 *
 * Also includes Kleisli category laws and functor law checkers.
 * Designed for use with fast-check property-based tests.
 */

import type { Morphism, KleisliMorphism, Result } from './core'
import { compose, id, composeK, idK } from './core'

// ─── Category Laws ──────────────────────────────────────────────────────────

/**
 * Check associativity: compose(f, compose(g, h))(x) ≡ compose(compose(f, g), h)(x)
 */
export function checkAssociativity<A, B, C, D>(
  f: Morphism<A, B>,
  g: Morphism<B, C>,
  h: Morphism<C, D>,
  x: A,
  equals: (a: D, b: D) => boolean = Object.is,
): boolean {
  const lhs = compose(f, compose(g, h))(x)
  const rhs = compose(compose(f, g), h)(x)
  return equals(lhs, rhs)
}

/**
 * Check left identity: compose(id, f)(x) ≡ f(x)
 */
export function checkLeftIdentity<A, B>(
  f: Morphism<A, B>,
  x: A,
  equals: (a: B, b: B) => boolean = Object.is,
): boolean {
  const lhs = compose(id<A>(), f)(x)
  const rhs = f(x)
  return equals(lhs, rhs)
}

/**
 * Check right identity: compose(f, id)(x) ≡ f(x)
 */
export function checkRightIdentity<A, B>(
  f: Morphism<A, B>,
  x: A,
  equals: (a: B, b: B) => boolean = Object.is,
): boolean {
  const lhs = compose(f, id<B>())(x)
  const rhs = f(x)
  return equals(lhs, rhs)
}

// ─── Kleisli Category Laws ──────────────────────────────────────────────────

/**
 * Check Kleisli associativity:
 *   composeK(f, composeK(g, h))(x) ≡ composeK(composeK(f, g), h)(x)
 */
export function checkKleisliAssociativity<A, B, C, D, E>(
  f: KleisliMorphism<A, B, E>,
  g: KleisliMorphism<B, C, E>,
  h: KleisliMorphism<C, D, E>,
  x: A,
  equals: (a: Result<D, E>, b: Result<D, E>) => boolean = deepResultEquals,
): boolean {
  const lhs = composeK(f, composeK(g, h))(x)
  const rhs = composeK(composeK(f, g), h)(x)
  return equals(lhs, rhs)
}

/**
 * Check Kleisli left identity: composeK(idK, f)(x) ≡ f(x)
 */
export function checkKleisliLeftIdentity<A, B, E>(
  f: KleisliMorphism<A, B, E>,
  x: A,
  equals: (a: Result<B, E>, b: Result<B, E>) => boolean = deepResultEquals,
): boolean {
  const lhs = composeK(idK<A>(), f)(x)
  const rhs = f(x)
  return equals(lhs, rhs)
}

/**
 * Check Kleisli right identity: composeK(f, idK)(x) ≡ f(x)
 */
export function checkKleisliRightIdentity<A, B, E>(
  f: KleisliMorphism<A, B, E>,
  x: A,
  equals: (a: Result<B, E>, b: Result<B, E>) => boolean = deepResultEquals,
): boolean {
  const lhs = composeK(f, idK<B>())(x)
  const rhs = f(x)
  return equals(lhs, rhs)
}

// ─── Functor Laws ───────────────────────────────────────────────────────────

/**
 * Check functor identity law: F(id) ≡ id
 * In practice: map(fa, id) ≡ fa
 */
export function checkFunctorIdentity<FA, A>(
  map: (fa: FA, f: Morphism<A, A>) => FA,
  fa: FA,
  equals: (a: FA, b: FA) => boolean,
): boolean {
  const mapped = map(fa, id<A>())
  return equals(mapped, fa)
}

/**
 * Check functor composition law: F(g ∘ f) ≡ F(g) ∘ F(f)
 * In practice: map(fa, compose(f, g)) ≡ map(map(fa, f), g)
 */
export function checkFunctorComposition<FA, FB, FC, A, B, C>(
  mapAB: (fa: FA, f: Morphism<A, B>) => FB,
  mapBC: (fb: FB, f: Morphism<B, C>) => FC,
  mapAC: (fa: FA, f: Morphism<A, C>) => FC,
  fa: FA,
  f: Morphism<A, B>,
  g: Morphism<B, C>,
  equals: (a: FC, b: FC) => boolean,
): boolean {
  const composed = mapAC(fa, compose(f, g))
  const sequential = mapBC(mapAB(fa, f), g)
  return equals(composed, sequential)
}

// ─── Natural Transformation Laws ────────────────────────────────────────────

/**
 * Check naturality condition: η_B ∘ F(f) ≡ G(f) ∘ η_A
 *
 * For every morphism f: A → B:
 *   η_B(F_map(fa, f)) ≡ G_map(η_A(fa), f)
 */
export function checkNaturality<FA, FB, GA, GB, A, B>(
  eta_A: Morphism<FA, GA>,
  eta_B: Morphism<FB, GB>,
  F_map: (fa: FA, f: Morphism<A, B>) => FB,
  G_map: (ga: GA, f: Morphism<A, B>) => GB,
  fa: FA,
  f: Morphism<A, B>,
  equals: (a: GB, b: GB) => boolean,
): boolean {
  const lhs = eta_B(F_map(fa, f))
  const rhs = G_map(eta_A(fa), f)
  return equals(lhs, rhs)
}

// ─── Monoidal Laws ──────────────────────────────────────────────────────────

/**
 * Check monoid associativity: tensor(a, tensor(b, c)) ≡ tensor(tensor(a, b), c)
 */
export function checkMonoidAssociativity<M>(
  tensor: (a: M, b: M) => M,
  a: M,
  b: M,
  c: M,
  equals: (x: M, y: M) => boolean,
): boolean {
  const lhs = tensor(a, tensor(b, c))
  const rhs = tensor(tensor(a, b), c)
  return equals(lhs, rhs)
}

/**
 * Check left identity: tensor(unit, a) ≡ a
 */
export function checkMonoidLeftIdentity<M>(
  tensor: (a: M, b: M) => M,
  unit: M,
  a: M,
  equals: (x: M, y: M) => boolean,
): boolean {
  return equals(tensor(unit, a), a)
}

/**
 * Check right identity: tensor(a, unit) ≡ a
 */
export function checkMonoidRightIdentity<M>(
  tensor: (a: M, b: M) => M,
  unit: M,
  a: M,
  equals: (x: M, y: M) => boolean,
): boolean {
  return equals(tensor(a, unit), a)
}

// ─── Adjunction Laws ────────────────────────────────────────────────────────

/**
 * Check adjunction triangle identity: ε_F(A) ∘ F(η_A) ≡ id_{F(A)}
 *
 * Where F ⊣ G, η is the unit, ε is the counit.
 */
export function checkTriangleF<A, FA>(
  F: Morphism<A, FA>,
  eta_A: Morphism<A, A>,
  epsilon_FA: Morphism<FA, FA>,
  a: A,
  equals: (x: FA, y: FA) => boolean,
): boolean {
  const lhs = epsilon_FA(F(eta_A(a)))
  const rhs = F(a)
  return equals(lhs, rhs)
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/** Deep equality check for Result values. */
function deepResultEquals<T, E>(a: Result<T, E>, b: Result<T, E>): boolean {
  if (a.ok !== b.ok) return false
  if (a.ok && b.ok) return JSON.stringify(a.value) === JSON.stringify(b.value)
  if (!a.ok && !b.ok) return JSON.stringify(a.error) === JSON.stringify(b.error)
  return false
}

/** Structural equality using JSON serialization. */
export function structuralEquals<T>(a: T, b: T): boolean {
  return JSON.stringify(a) === JSON.stringify(b)
}
