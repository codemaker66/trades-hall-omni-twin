/**
 * CT-7: Adjunction Interface
 *
 * An adjunction F ⊣ G (F left adjoint to G) is a pair of functors
 * with unit η and counit ε satisfying the triangle identities:
 *
 *   ε_{F(A)} ∘ F(η_A) = id_{F(A)}   (left triangle)
 *   G(ε_B) ∘ η_{G(B)} = id_{G(B)}   (right triangle)
 *
 * Equivalently: Hom(F(A), B) ≅ Hom(A, G(B)) (natural in A and B)
 *
 * Application to optimization:
 *   F: Constraints → Layouts  (free functor: generate optimal layout)
 *   G: Layouts → Constraints  (forgetful functor: extract constraints)
 *   F ⊣ G means F(C) is the BEST layout satisfying C.
 */

import type { Morphism } from './core'

// ─── Adjunction Interface ───────────────────────────────────────────────────

/**
 * An adjunction between two categories.
 *
 * @typeParam A - Object type in the source category
 * @typeParam B - Object type in the target category
 */
export interface Adjunction<A, B> {
  readonly name: string

  /**
   * Left adjoint F: A → B
   * The "free" construction — generates optimal B from A.
   */
  leftAdjoint: Morphism<A, B>

  /**
   * Right adjoint G: B → A
   * The "forgetful" functor — extracts A from B.
   */
  rightAdjoint: Morphism<B, A>

  /**
   * Unit η: A → G(F(A))
   * "Embed A into the round-trip through both functors."
   */
  unit: Morphism<A, A>

  /**
   * Counit ε: F(G(B)) → B
   * "Extract B from the round-trip through both functors."
   */
  counit: Morphism<B, B>
}

// ─── Adjunction Construction ────────────────────────────────────────────────

/**
 * Create an adjunction from left/right adjoints and derive unit/counit.
 *
 * Unit:   η_A = G(F(A))  — round-trip through F then G
 * Counit: ε_B = F(G(B))  — round-trip through G then F
 */
export function createAdjunction<A, B>(
  name: string,
  leftAdjoint: Morphism<A, B>,
  rightAdjoint: Morphism<B, A>,
): Adjunction<A, B> {
  return {
    name,
    leftAdjoint,
    rightAdjoint,
    unit: (a: A) => rightAdjoint(leftAdjoint(a)),
    counit: (b: B) => leftAdjoint(rightAdjoint(b)),
  }
}

// ─── Adjunction Laws ────────────────────────────────────────────────────────

/**
 * Verify the left triangle identity: ε_{F(A)} ∘ F(η_A) = id_{F(A)}
 *
 * In terms of our adjunction:
 *   counit(leftAdjoint(unit(a))) ≡ leftAdjoint(a)
 */
export function verifyLeftTriangle<A, B>(
  adj: Adjunction<A, B>,
  a: A,
  equals: (x: B, y: B) => boolean,
): boolean {
  const lhs = adj.counit(adj.leftAdjoint(adj.unit(a)))
  const rhs = adj.leftAdjoint(a)
  return equals(lhs, rhs)
}

/**
 * Verify the right triangle identity: G(ε_B) ∘ η_{G(B)} = id_{G(B)}
 *
 * In terms of our adjunction:
 *   unit(rightAdjoint(counit(b))) ≡ rightAdjoint(b)
 */
export function verifyRightTriangle<A, B>(
  adj: Adjunction<A, B>,
  b: B,
  equals: (x: A, y: A) => boolean,
): boolean {
  const lhs = adj.unit(adj.rightAdjoint(adj.counit(b)))
  const rhs = adj.rightAdjoint(b)
  return equals(lhs, rhs)
}

// ─── Hom-Set Isomorphism ────────────────────────────────────────────────────

/**
 * The Hom-set isomorphism: Hom(F(A), B) ≅ Hom(A, G(B))
 *
 * Given f: F(A) → B, produce g: A → G(B)
 */
export function transposeLeft<A, B>(
  adj: Adjunction<A, B>,
  f: Morphism<B, B>,
): Morphism<A, A> {
  return (a: A) => adj.rightAdjoint(f(adj.leftAdjoint(a)))
}

/**
 * Given g: A → G(B), produce f: F(A) → B
 */
export function transposeRight<A, B>(
  adj: Adjunction<A, B>,
  g: Morphism<A, A>,
): Morphism<B, B> {
  return (b: B) => adj.leftAdjoint(g(adj.rightAdjoint(b)))
}

// ─── Monad from Adjunction ──────────────────────────────────────────────────

/**
 * Every adjunction F ⊣ G gives rise to a monad T = G ∘ F.
 *
 * T(A) = G(F(A))
 * return = unit: A → T(A)
 * join = G(counit): T(T(A)) → T(A)
 */
export interface MonadFromAdjunction<A> {
  readonly adjunctionName: string

  /** T(a) = G(F(a)): apply the monad. */
  apply: Morphism<A, A>

  /** return = unit: A → T(A). */
  pure: Morphism<A, A>

  /** join = G(ε): T(T(A)) → T(A). Flatten nested applications. */
  join: Morphism<A, A>

  /** bind (>>=): T(A) → (A → T(B)) → T(B). Monadic composition. */
  bind: (ta: A, f: Morphism<A, A>) => A
}

export function monadFromAdjunction<A, B>(adj: Adjunction<A, B>): MonadFromAdjunction<A> {
  const apply: Morphism<A, A> = (a) => adj.rightAdjoint(adj.leftAdjoint(a))

  return {
    adjunctionName: adj.name,
    apply,
    pure: adj.unit,
    join: (a: A) => adj.rightAdjoint(adj.counit(adj.leftAdjoint(a))),
    bind: (ta: A, f: Morphism<A, A>) => {
      // bind ta f = join (fmap f ta)
      // In our simplified setting: apply f to ta, then join
      return adj.rightAdjoint(adj.leftAdjoint(f(ta)))
    },
  }
}
