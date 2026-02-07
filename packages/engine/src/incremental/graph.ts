/**
 * Incremental Computation Framework
 *
 * A reactive dependency graph inspired by Jane Street's Incremental (OCaml).
 * Tracks which derived computations depend on which inputs, and only recomputes
 * the minimum necessary subgraph when an input changes.
 *
 * Key properties:
 * - Height-based scheduling: nodes processed in topological order
 * - Change propagation with cutoff: if a node produces the same value, stop
 * - Batch stabilization: collect multiple input changes, stabilize once
 */

// ─── Node Types ─────────────────────────────────────────────────────────────

type EqualityFn<T> = (a: T, b: T) => boolean

const defaultEquals = <T>(a: T, b: T): boolean => a === b

/** A computed node is either derived or observer */
type ComputedNode = DerivedNode<unknown> | ObserverNode<unknown>

interface BaseNode {
  id: number
  height: number
  dirty: boolean
  dependents: Set<ComputedNode>
}

export interface InputNode<T> extends BaseNode {
  kind: 'input'
  value: T
}

export interface DerivedNode<T> extends BaseNode {
  kind: 'derived'
  value: T
  dependencies: BaseNode[]
  compute: (...args: unknown[]) => T
  equals: EqualityFn<T>
  recomputeCount: number
}

export interface ObserverNode<T> extends BaseNode {
  kind: 'observer'
  value: T
  dependencies: BaseNode[]
  compute: (...args: unknown[]) => T
  equals: EqualityFn<T>
  effect: (value: T) => void
  recomputeCount: number
}

type AnyNode = InputNode<unknown> | DerivedNode<unknown> | ObserverNode<unknown>

// ─── Graph ──────────────────────────────────────────────────────────────────

export class IncrementalGraph {
  private nextId = 0
  private nodes: AnyNode[] = []
  private dirtySet: Set<AnyNode> = new Set()
  private _stabilizeCount = 0

  /** Number of times stabilize() has been called */
  get stabilizeCount(): number {
    return this._stabilizeCount
  }

  /**
   * Create an input node (adjustable by the user).
   */
  input<T>(initialValue: T): InputNode<T> {
    const node: InputNode<T> = {
      id: this.nextId++,
      kind: 'input',
      height: 0,
      dirty: false,
      value: initialValue,
      dependents: new Set(),
    }
    this.nodes.push(node as AnyNode)
    return node
  }

  /**
   * Create a derived node that recomputes when any dependency changes.
   */
  derive<T, Deps extends BaseNode[]>(
    dependencies: [...Deps],
    compute: (...args: { [K in keyof Deps]: Deps[K] extends BaseNode & { value: infer V } ? V : never }) => T,
    equals: EqualityFn<T> = defaultEquals,
  ): DerivedNode<T> {
    const height = Math.max(...dependencies.map((d) => d.height)) + 1

    // Compute initial value
    const args = dependencies.map((d) => (d as BaseNode & { value: unknown }).value) as unknown[]
    const initialValue = (compute as (...a: unknown[]) => T)(...args)

    const node: DerivedNode<T> = {
      id: this.nextId++,
      kind: 'derived',
      height,
      dirty: false,
      value: initialValue,
      dependencies,
      dependents: new Set(),
      compute: compute as (...a: unknown[]) => T,
      equals,
      recomputeCount: 0,
    }

    // Register as dependent of each dependency
    for (const dep of dependencies) {
      dep.dependents.add(node as ComputedNode)
    }

    this.nodes.push(node as AnyNode)
    return node
  }

  /**
   * Create an observer node — a leaf that triggers a side effect when value changes.
   */
  observe<T, Deps extends BaseNode[]>(
    dependencies: [...Deps],
    compute: (...args: { [K in keyof Deps]: Deps[K] extends BaseNode & { value: infer V } ? V : never }) => T,
    effect: (value: T) => void,
    equals: EqualityFn<T> = defaultEquals,
  ): ObserverNode<T> {
    const height = Math.max(...dependencies.map((d) => d.height)) + 1

    const args = dependencies.map((d) => (d as BaseNode & { value: unknown }).value) as unknown[]
    const initialValue = (compute as (...a: unknown[]) => T)(...args)

    const node: ObserverNode<T> = {
      id: this.nextId++,
      kind: 'observer',
      height,
      dirty: false,
      value: initialValue,
      dependencies,
      dependents: new Set(),
      compute: compute as (...a: unknown[]) => T,
      equals,
      effect,
      recomputeCount: 0,
    }

    for (const dep of dependencies) {
      dep.dependents.add(node as ComputedNode)
    }

    this.nodes.push(node as AnyNode)
    return node
  }

  /**
   * Set an input node's value. Marks dependents as dirty.
   * Call stabilize() after setting inputs to propagate changes.
   */
  set<T>(node: InputNode<T>, value: T): void {
    if (node.value === value) return // no-op
    node.value = value
    this.markDependentsDirty(node)
  }

  /**
   * Propagate all pending changes through the graph.
   * Processes dirty nodes in height order (lowest first = topological).
   * Uses cutoff: if a derived node produces the same value, its dependents are NOT dirtied.
   */
  stabilize(): void {
    this._stabilizeCount++

    // Sort dirty nodes by height (ascending) for topological processing
    const sortedDirty = Array.from(this.dirtySet).sort((a, b) => a.height - b.height)

    for (const node of sortedDirty) {
      if (!node.dirty) continue // may have been un-dirtied by cutoff

      if (node.kind === 'derived' || node.kind === 'observer') {
        const derived = node as DerivedNode<unknown> | ObserverNode<unknown>
        const args = derived.dependencies.map((d) => (d as BaseNode & { value: unknown }).value)
        const newValue = derived.compute(...args)
        derived.recomputeCount++

        if (derived.equals(derived.value, newValue)) {
          // Cutoff: same value, don't propagate further
          derived.dirty = false
          this.dirtySet.delete(node)
          continue
        }

        derived.value = newValue
        derived.dirty = false
        this.dirtySet.delete(node)

        // Trigger observer effect
        if (node.kind === 'observer') {
          ;(node as ObserverNode<unknown>).effect(newValue)
        }

        // Mark dependents dirty (they need to recompute with the new value)
        this.markDependentsDirty(derived)
      }
    }

    this.dirtySet.clear()
  }

  /**
   * Total recomputation count across all derived/observer nodes.
   */
  totalRecomputations(): number {
    let total = 0
    for (const node of this.nodes) {
      if (node.kind === 'derived' || node.kind === 'observer') {
        total += (node as DerivedNode<unknown>).recomputeCount
      }
    }
    return total
  }

  /**
   * Reset all recomputation counters (useful for measuring a specific stabilize).
   */
  resetCounters(): void {
    for (const node of this.nodes) {
      if (node.kind === 'derived' || node.kind === 'observer') {
        ;(node as DerivedNode<unknown>).recomputeCount = 0
      }
    }
  }

  /** Number of nodes in the graph */
  get size(): number {
    return this.nodes.length
  }

  private markDependentsDirty(node: BaseNode): void {
    for (const dep of node.dependents) {
      if (!dep.dirty) {
        dep.dirty = true
        this.dirtySet.add(dep as AnyNode)
        // Recursively mark transitive dependents
        this.markDependentsDirty(dep)
      }
    }
  }
}
