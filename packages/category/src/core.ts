/**
 * CT-1: Core Category Theory Primitives
 *
 * A Category C consists of:
 *   - A collection of objects Ob(C)
 *   - For each pair of objects A, B: a set of morphisms Hom(A, B)
 *   - Composition: ∘ : Hom(B, C) × Hom(A, B) → Hom(A, C)
 *   - Identity: for each object A, an identity morphism id_A ∈ Hom(A, A)
 *
 * Category Laws:
 *   1. Associativity: (h ∘ g) ∘ f = h ∘ (g ∘ f)
 *   2. Left identity:  id ∘ f = f
 *   3. Right identity: f ∘ id = f
 */

// ─── Morphism ──────────────────────────────────────────────────────────────

/** A morphism (arrow) from A to B: a structure-preserving map. */
export type Morphism<A, B> = (a: A) => B

/**
 * Compose two morphisms: g ∘ f (apply f first, then g).
 * This is the fundamental operation of category theory.
 *
 * Diagram:  A --f--> B --g--> C
 *           A ------g∘f-----> C
 */
export function compose<A, B, C>(f: Morphism<A, B>, g: Morphism<B, C>): Morphism<A, C> {
  return (a: A) => g(f(a))
}

/** Compose three morphisms: h ∘ g ∘ f. */
export function compose3<A, B, C, D>(
  f: Morphism<A, B>,
  g: Morphism<B, C>,
  h: Morphism<C, D>,
): Morphism<A, D> {
  return (a: A) => h(g(f(a)))
}

/** Identity morphism: id_A. Every object has one. */
export function id<A>(): Morphism<A, A> {
  return (a: A) => a
}

// ─── Kleisli Composition (for Result-returning morphisms) ──────────────────

/** A result type: morphisms that can fail. */
export type Result<T, E = string> =
  | { readonly ok: true; readonly value: T }
  | { readonly ok: false; readonly error: E }

export function ok<T>(value: T): Result<T, never> {
  return { ok: true, value }
}

export function err<E>(error: E): Result<never, E> {
  return { ok: false, error }
}

/**
 * A Kleisli morphism: a morphism that returns a Result.
 * These form a category under Kleisli composition.
 */
export type KleisliMorphism<A, B, E = string> = (a: A) => Result<B, E>

/**
 * Kleisli composition: compose fallible morphisms.
 * If f fails, g is never called. Short-circuit semantics.
 *
 * This models the Kleisli category for the Result monad.
 */
export function composeK<A, B, C, E = string>(
  f: KleisliMorphism<A, B, E>,
  g: KleisliMorphism<B, C, E>,
): KleisliMorphism<A, C, E> {
  return (a: A) => {
    const result = f(a)
    if (!result.ok) return result
    return g(result.value)
  }
}

/** Kleisli identity: wraps a value in Ok. */
export function idK<A>(): KleisliMorphism<A, A, never> {
  return (a: A) => ok(a)
}

/** Lift a pure morphism into the Kleisli category. */
export function liftK<A, B>(f: Morphism<A, B>): KleisliMorphism<A, B, never> {
  return (a: A) => ok(f(a))
}

// ─── Product and Coproduct ─────────────────────────────────────────────────

/** Product (pair) of two objects — the categorical product. */
export type Product<A, B> = readonly [A, B]

export function pair<A, B>(a: A, b: B): Product<A, B> {
  return [a, b] as const
}

export function fst<A, B>(p: Product<A, B>): A {
  return p[0]
}

export function snd<A, B>(p: Product<A, B>): B {
  return p[1]
}

/** Coproduct (tagged union) — the categorical coproduct. */
export type Coproduct<A, B> =
  | { readonly tag: 'left'; readonly value: A }
  | { readonly tag: 'right'; readonly value: B }

export function inl<A>(value: A): Coproduct<A, never> {
  return { tag: 'left', value }
}

export function inr<B>(value: B): Coproduct<never, B> {
  return { tag: 'right', value }
}

/** Universal property of coproduct: case analysis. */
export function match<A, B, C>(
  f: Morphism<A, C>,
  g: Morphism<B, C>,
): Morphism<Coproduct<A, B>, C> {
  return (cp) => cp.tag === 'left' ? f(cp.value) : g(cp.value)
}

// ─── Pipeline Builder ──────────────────────────────────────────────────────

/**
 * Fluent pipeline builder using morphism composition.
 * Provides a type-safe, readable alternative to nested compose() calls.
 *
 * Usage:
 *   pipeline(f).then(g).then(h).run(input)
 *   // equivalent to: compose(compose(f, g), h)(input)
 */
export interface Pipeline<A, B> {
  /** Append a morphism to the pipeline. */
  then<C>(g: Morphism<B, C>): Pipeline<A, C>
  /** Append a Kleisli morphism (fallible step). */
  thenK<C, E>(g: KleisliMorphism<B, C, E>): KleisliPipeline<A, C, E>
  /** Execute the pipeline on an input. */
  run: Morphism<A, B>
}

export interface KleisliPipeline<A, B, E> {
  /** Append a Kleisli morphism to the pipeline. */
  thenK<C>(g: KleisliMorphism<B, C, E>): KleisliPipeline<A, C, E>
  /** Execute the pipeline on an input. */
  run: KleisliMorphism<A, B, E>
}

export function pipeline<A, B>(f: Morphism<A, B>): Pipeline<A, B> {
  return {
    then<C>(g: Morphism<B, C>): Pipeline<A, C> {
      return pipeline(compose(f, g))
    },
    thenK<C, E>(g: KleisliMorphism<B, C, E>): KleisliPipeline<A, C, E> {
      return kleisliPipeline(composeK(liftK(f), g))
    },
    run: f,
  }
}

export function kleisliPipeline<A, B, E>(f: KleisliMorphism<A, B, E>): KleisliPipeline<A, B, E> {
  return {
    thenK<C>(g: KleisliMorphism<B, C, E>): KleisliPipeline<A, C, E> {
      return kleisliPipeline(composeK(f, g))
    },
    run: f,
  }
}
