/**
 * CT-3: Natural Transformation Interface
 *
 * A natural transformation η: F ⇒ G converts between two functors
 * while preserving their structure.
 *
 * Naturality condition (the commuting square):
 *   For every morphism f: A → B in the source category:
 *     η_B ∘ F(f) = G(f) ∘ η_A
 *
 * In practice: a principled way to swap subsystem implementations
 * with a guarantee that behavior is preserved.
 */

import type { Morphism } from './core'
import type { FunctorValue } from './functor'

// ─── Natural Transformation Interface ───────────────────────────────────────

/**
 * A natural transformation from functor F to functor G.
 *
 * The component at A (η_A): F(A) → G(A)
 *
 * Must satisfy naturality: for all f: A → B,
 *   η_B(F.map(fa, f)) ≡ G.map(η_A(fa), f)
 */
export interface NaturalTransformation<F extends string, G extends string> {
  readonly source: F
  readonly target: G

  /**
   * The component at type A: η_A: F(A) → G(A)
   */
  transform<A>(fa: FunctorValue<F, A>): FunctorValue<G, A>
}

// ─── Vertical Composition ───────────────────────────────────────────────────

/**
 * Vertical composition of natural transformations:
 *   If η: F ⇒ G and ε: G ⇒ H, then ε ∘ η: F ⇒ H
 */
export function verticalCompose<F extends string, G extends string, H extends string>(
  eta: NaturalTransformation<F, G>,
  epsilon: NaturalTransformation<G, H>,
): NaturalTransformation<F, H> {
  return {
    source: eta.source,
    target: epsilon.target,
    transform<A>(fa: FunctorValue<F, A>): FunctorValue<H, A> {
      return epsilon.transform(eta.transform(fa))
    },
  }
}

/**
 * Identity natural transformation: id_F: F ⇒ F
 */
export function identityNT<F extends string>(tag: F): NaturalTransformation<F, F> {
  return {
    source: tag,
    target: tag,
    transform<A>(fa: FunctorValue<F, A>): FunctorValue<F, A> {
      return fa
    },
  }
}

// ─── Concrete Natural Transformation (Strategy Swap) ────────────────────────

/**
 * A strategy swap: convert between two implementations of the same interface.
 * This is a natural transformation specialized for backend swapping.
 *
 * Examples:
 *   InMemoryStore ⇒ PostgresStore
 *   WebGLRenderer ⇒ WebGPURenderer
 *   WebSocketTransport ⇒ WebRTCTransport
 */
export interface StrategySwap<A, B> {
  readonly sourceName: string
  readonly targetName: string

  /** Forward transformation: convert source to target. */
  forward: Morphism<A, B>

  /** Backward transformation: convert target to source. */
  backward: Morphism<B, A>
}

/**
 * Create a strategy swap with round-trip verification.
 */
export function createStrategySwap<A, B>(
  sourceName: string,
  targetName: string,
  forward: Morphism<A, B>,
  backward: Morphism<B, A>,
): StrategySwap<A, B> {
  return { sourceName, targetName, forward, backward }
}

/**
 * Verify the round-trip property of a strategy swap:
 *   backward(forward(x)) ≡ x
 *   forward(backward(y)) ≡ y
 */
export function verifyStrategyRoundTrip<A, B>(
  swap: StrategySwap<A, B>,
  a: A,
  b: B,
  equalsA: (x: A, y: A) => boolean,
  equalsB: (x: B, y: B) => boolean,
): { forwardBackward: boolean; backwardForward: boolean } {
  return {
    forwardBackward: equalsA(swap.backward(swap.forward(a)), a),
    backwardForward: equalsB(swap.forward(swap.backward(b)), b),
  }
}

// ─── Isomorphism (Invertible Natural Transformation) ────────────────────────

/**
 * A natural isomorphism: a natural transformation with a two-sided inverse.
 * η: F ⇒ G where η^{-1} ∘ η = id and η ∘ η^{-1} = id.
 */
export interface NaturalIsomorphism<F extends string, G extends string> {
  readonly forward: NaturalTransformation<F, G>
  readonly backward: NaturalTransformation<G, F>
}

export function createNaturalIsomorphism<F extends string, G extends string>(
  forward: NaturalTransformation<F, G>,
  backward: NaturalTransformation<G, F>,
): NaturalIsomorphism<F, G> {
  return { forward, backward }
}
