/**
 * CT-4: Monoidal Category
 *
 * A monoidal category (C, ⊗, I) has:
 *   - A tensor product ⊗: C × C → C
 *   - A unit object I
 *
 * Monoidal Laws:
 *   1. Associativity: (A ⊗ B) ⊗ C ≅ A ⊗ (B ⊗ C)
 *   2. Left unit:     I ⊗ A ≅ A
 *   3. Right unit:    A ⊗ I ≅ A
 *
 * Application: Service composition, event assembly.
 */

// ─── Monoid ─────────────────────────────────────────────────────────────────

/**
 * A monoid in a category: an object M with multiplication and unit.
 *
 * Laws:
 *   combine(combine(a, b), c) ≡ combine(a, combine(b, c))  (associativity)
 *   combine(empty, a) ≡ a                                    (left identity)
 *   combine(a, empty) ≡ a                                    (right identity)
 */
export interface Monoid<M> {
  readonly empty: M
  combine(a: M, b: M): M
}

/**
 * Fold a list using a monoid.
 */
export function mconcat<M>(monoid: Monoid<M>, values: readonly M[]): M {
  return values.reduce((acc, v) => monoid.combine(acc, v), monoid.empty)
}

// ─── Common Monoids ─────────────────────────────────────────────────────────

/** Additive monoid for numbers. */
export const additiveMonoid: Monoid<number> = {
  empty: 0,
  combine: (a, b) => a + b,
}

/** Multiplicative monoid for numbers. */
export const multiplicativeMonoid: Monoid<number> = {
  empty: 1,
  combine: (a, b) => a * b,
}

/** String concatenation monoid. */
export const stringMonoid: Monoid<string> = {
  empty: '',
  combine: (a, b) => a + b,
}

/** Array concatenation monoid. */
export function arrayMonoid<T>(): Monoid<readonly T[]> {
  return {
    empty: [],
    combine: (a, b) => [...a, ...b],
  }
}

/** Max monoid (for scoring). */
export const maxMonoid: Monoid<number> = {
  empty: -Infinity,
  combine: (a, b) => Math.max(a, b),
}

/** Min monoid (for constraint satisfaction). */
export const minMonoid: Monoid<number> = {
  empty: Infinity,
  combine: (a, b) => Math.min(a, b),
}

// ─── Monoidal Category (Tensor Product) ─────────────────────────────────────

/**
 * A monoidal category: a category with a tensor product and unit.
 */
export interface MonoidalCategory<Obj> {
  /** Tensor product: combine two objects. */
  tensor(a: Obj, b: Obj): Obj

  /** Unit object: the identity for tensor. */
  unit: Obj

  /** Associator: (A ⊗ B) ⊗ C → A ⊗ (B ⊗ C). */
  associator(abc: Obj): Obj

  /** Left unitor: I ⊗ A → A. */
  leftUnitor(ia: Obj): Obj

  /** Right unitor: A ⊗ I → A. */
  rightUnitor(ai: Obj): Obj
}

/**
 * Create a monoidal category from a monoid.
 * Every monoid gives rise to a one-object monoidal category.
 */
export function monoidToMonoidalCategory<M>(monoid: Monoid<M>): MonoidalCategory<M> {
  return {
    tensor: (a, b) => monoid.combine(a, b),
    unit: monoid.empty,
    // For a monoid (one-object category), these are all identity
    associator: (x) => x,
    leftUnitor: (x) => x,
    rightUnitor: (x) => x,
  }
}

// ─── Commutative Monoid ─────────────────────────────────────────────────────

/**
 * A commutative monoid: combine(a, b) ≡ combine(b, a).
 * Extra symmetry enables parallel evaluation.
 */
export interface CommutativeMonoid<M> extends Monoid<M> {
  readonly _commutative: true
}

export function createCommutativeMonoid<M>(
  empty: M,
  combine: (a: M, b: M) => M,
): CommutativeMonoid<M> {
  return { empty, combine, _commutative: true }
}

// ─── Free Monoid ────────────────────────────────────────────────────────────

/**
 * The free monoid on a type A is just A[] (lists).
 * This is the "initial" monoid — it makes no identifications.
 */
export function freeMonoid<A>(): Monoid<readonly A[]> {
  return arrayMonoid<A>()
}

/**
 * The universal property of the free monoid:
 * Given any function f: A → M (where M is a monoid),
 * there exists a unique monoid homomorphism from A[] to M.
 */
export function foldFree<A, M>(monoid: Monoid<M>, f: (a: A) => M): (as: readonly A[]) => M {
  return (as) => mconcat(monoid, as.map(f))
}
