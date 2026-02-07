/**
 * CT-2: Functor Interface
 *
 * A Functor F: C → D maps between categories while preserving structure:
 *   - Object mapping:   F(A) for objects A ∈ Ob(C)
 *   - Morphism mapping:  F(f: A→B) = F(f): F(A)→F(B)
 *
 * Functor Laws:
 *   1. Identity:    F(id_A) = id_{F(A)}
 *   2. Composition: F(g ∘ f) = F(g) ∘ F(f)
 */

import type { Morphism } from './core'

// ─── Functor Interface ──────────────────────────────────────────────────────

/**
 * A functor from category C to category D.
 * Generic over the container/context type F.
 *
 * @typeParam FA - The source type (F applied to A)
 * @typeParam FB - The target type (F applied to B)
 */
export interface Functor<F extends string> {
  readonly tag: F

  /**
   * Object mapping: lift a value into the functor.
   * F(A) — maps an object A in C to F(A) in D.
   */
  of<A>(a: A): FunctorValue<F, A>

  /**
   * Morphism mapping: apply a function inside the functor.
   * F(f: A→B) maps F(A) → F(B)
   */
  map<A, B>(fa: FunctorValue<F, A>, f: Morphism<A, B>): FunctorValue<F, B>
}

/**
 * A value wrapped in a functor context.
 */
export interface FunctorValue<F extends string, A> {
  readonly _tag: F
  readonly value: A
}

// ─── Contravariant Functor ──────────────────────────────────────────────────

/**
 * A contravariant functor reverses the direction of morphisms:
 *   F(f: A→B): F(B) → F(A)  (note: reversed!)
 *
 * Common examples: predicates, comparators, consumers.
 */
export interface ContravariantFunctor<F extends string> {
  readonly tag: F
  contramap<A, B>(fa: FunctorValue<F, A>, f: Morphism<B, A>): FunctorValue<F, B>
}

// ─── Bifunctor ──────────────────────────────────────────────────────────────

/**
 * A bifunctor maps from C × D to E.
 * Covariant in both arguments.
 */
export interface Bifunctor<F extends string> {
  bimap<A, B, C, D>(
    fab: BifunctorValue<F, A, B>,
    f: Morphism<A, C>,
    g: Morphism<B, D>,
  ): BifunctorValue<F, C, D>
}

export interface BifunctorValue<F extends string, A, B> {
  readonly _tag: F
  readonly first: A
  readonly second: B
}

// ─── Endofunctor ────────────────────────────────────────────────────────────

/**
 * An endofunctor maps a category to itself: F: C → C.
 * Most functors in programming are endofunctors on the category of types.
 */
export type Endofunctor<F extends string> = Functor<F>

// ─── Helper: Create a simple Functor ────────────────────────────────────────

/**
 * Create a functor from a tag and mapping function.
 */
export function createFunctor<F extends string>(tag: F): Functor<F> {
  return {
    tag,
    of<A>(a: A): FunctorValue<F, A> {
      return { _tag: tag, value: a }
    },
    map<A, B>(fa: FunctorValue<F, A>, f: Morphism<A, B>): FunctorValue<F, B> {
      return { _tag: tag, value: f(fa.value) }
    },
  }
}

/**
 * Compose two functors: F ∘ G.
 * If F: B→C and G: A→B, then F∘G: A→C.
 */
export function composeFunctors<F extends string, G extends string>(
  outer: Functor<F>,
  inner: Functor<G>,
): {
  of<A>(a: A): FunctorValue<F, FunctorValue<G, A>>
  map<A, B>(
    fga: FunctorValue<F, FunctorValue<G, A>>,
    f: Morphism<A, B>,
  ): FunctorValue<F, FunctorValue<G, B>>
} {
  return {
    of<A>(a: A) {
      return outer.of(inner.of(a))
    },
    map<A, B>(fga: FunctorValue<F, FunctorValue<G, A>>, f: Morphism<A, B>) {
      return outer.map(fga, (ga) => inner.map(ga, f))
    },
  }
}
