/**
 * CT-4: Operad Interface
 *
 * An operad generalizes the monoidal structure to operations with
 * multiple inputs. Perfect for modeling operations like:
 *   "This event needs 1 venue + 2 caterers + 1 AV vendor + 3 decorators"
 *
 * An operad O consists of:
 *   - For each n ≥ 0, a set O(n) of n-ary operations
 *   - Composition: O(k) × O(n₁) × ... × O(nₖ) → O(n₁ + ... + nₖ)
 *   - A unit element in O(1) (the identity operation)
 *
 * Operad Laws:
 *   1. Associativity of composition
 *   2. Unit laws (identity operation is neutral)
 *   3. Equivariance (compatible with input permutations)
 */

import type { Result } from './core'
import { ok, err } from './core'

// ─── Operad Types ───────────────────────────────────────────────────────────

/**
 * An operad operation: takes n inputs and produces one output.
 *
 * @typeParam Inputs - Tuple type of input types
 * @typeParam Output - The output type
 */
export interface OperadOp<Inputs extends readonly unknown[], Output> {
  readonly name: string
  readonly arity: number

  /** Apply the operation to its inputs. */
  apply(...inputs: [...Inputs]): Output

  /**
   * Validate that inputs are compatible before applying.
   * Returns Ok(void) if valid, Err(reasons) if not.
   */
  validate(...inputs: [...Inputs]): Result<void, string[]>
}

/**
 * An operad: a collection of operations with composition.
 */
export interface Operad<BaseType> {
  readonly name: string

  /** The identity operation: O(1), passes input through unchanged. */
  identity: OperadOp<[BaseType], BaseType>

  /**
   * Compose operations:
   *   Given outer: (B₁, ..., Bₖ) → C
   *   and inner₁: (...) → B₁, ..., innerₖ: (...) → Bₖ
   *   produce: (...all inner inputs...) → C
   */
  compose<C>(
    outer: OperadOp<BaseType[], C>,
    ...inners: OperadOp<BaseType[], BaseType>[]
  ): OperadOp<BaseType[], C>
}

// ─── Operad Construction ────────────────────────────────────────────────────

/**
 * Create an operad operation from a function and optional validator.
 */
export function createOp<Inputs extends readonly unknown[], Output>(
  name: string,
  apply: (...inputs: [...Inputs]) => Output,
  validate?: (...inputs: [...Inputs]) => Result<void, string[]>,
): OperadOp<Inputs, Output> {
  return {
    name,
    arity: apply.length,
    apply,
    validate: validate ?? ((..._inputs: [...Inputs]) => ok(undefined)),
  }
}

/**
 * Create an operad from a base type and composition strategy.
 */
export function createOperad<BaseType>(name: string): Operad<BaseType> {
  const identity: OperadOp<[BaseType], BaseType> = createOp(
    'identity',
    (x: BaseType) => x,
  )

  return {
    name,
    identity,
    compose<C>(
      outer: OperadOp<BaseType[], C>,
      ...inners: OperadOp<BaseType[], BaseType>[]
    ): OperadOp<BaseType[], C> {
      // The composed operation collects inputs for each inner op,
      // runs them to produce intermediate results,
      // then passes those to the outer op.
      const totalArity = inners.reduce((sum, op) => sum + op.arity, 0)

      return createOp(
        `${outer.name}(${inners.map(i => i.name).join(', ')})`,
        (...allInputs: BaseType[]) => {
          let offset = 0
          const intermediates: BaseType[] = []

          for (const inner of inners) {
            const inputs = allInputs.slice(offset, offset + inner.arity)
            intermediates.push(inner.apply(...inputs))
            offset += inner.arity
          }

          return outer.apply(...intermediates)
        },
        (...allInputs: BaseType[]) => {
          // Validate each inner operation's inputs
          let offset = 0
          const allErrors: string[] = []

          for (const inner of inners) {
            const inputs = allInputs.slice(offset, offset + inner.arity)
            const result = inner.validate(...inputs)
            if (!result.ok) allErrors.push(...result.error)
            offset += inner.arity
          }

          if (allErrors.length > 0) return err(allErrors)

          // Validate outer operation's inputs
          const intermediates: BaseType[] = []
          offset = 0
          for (const inner of inners) {
            const inputs = allInputs.slice(offset, offset + inner.arity)
            intermediates.push(inner.apply(...inputs))
            offset += inner.arity
          }

          return outer.validate(...intermediates)
        },
      )
    },
  }
}

// ─── Parallel Composition ───────────────────────────────────────────────────

/**
 * Run multiple operations in parallel on separate inputs.
 * This is the tensor product in the operad.
 */
export function parallel<Inputs extends readonly unknown[], Output>(
  ...ops: OperadOp<Inputs, Output>[]
): OperadOp<Inputs[], Output[]> {
  return createOp(
    ops.map(o => o.name).join(' ⊗ '),
    (...inputGroups: Inputs[]) =>
      ops.map((op, i) => op.apply(...(inputGroups[i] as unknown as [...Inputs]))),
    (...inputGroups: Inputs[]) => {
      const errors: string[] = []
      for (let i = 0; i < ops.length; i++) {
        const result = ops[i]!.validate(...(inputGroups[i] as unknown as [...Inputs]))
        if (!result.ok) errors.push(...result.error)
      }
      return errors.length > 0 ? err(errors) : ok(undefined)
    },
  )
}
