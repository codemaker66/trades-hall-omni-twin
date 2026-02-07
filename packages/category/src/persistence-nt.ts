/**
 * CT-3: Persistence Natural Transformation
 *
 * η: InMemoryReadModel ⇒ PostgresReadModel
 *
 * Naturality condition:
 *   η(InMemoryProjection(event)) ≡ PostgresProjection(η(event))
 *
 * This guarantees: projecting-then-persisting gives the same result
 * as persisting-then-projecting. You can swap storage backends without
 * changing behavior.
 */

import type { Morphism } from './core'
import type { StrategySwap } from './natural-transformation'
import { createStrategySwap } from './natural-transformation'

// ─── Read Model Representations ─────────────────────────────────────────────

/**
 * In-memory read model: simple objects in memory.
 * Fast reads, lost on process restart.
 */
export interface InMemoryReadModel<T> {
  readonly kind: 'in-memory'
  readonly data: ReadonlyMap<string, T>
  readonly version: number
}

/**
 * Postgres read model: rows in a database.
 * Durable, queryable, slower writes.
 */
export interface PostgresReadModel<T> {
  readonly kind: 'postgres'
  readonly rows: readonly PostgresRow<T>[]
  readonly tableName: string
  readonly version: number
}

export interface PostgresRow<T> {
  readonly id: string
  readonly data: T
  readonly updatedAt: number
}

// ─── Persistence Natural Transformation ─────────────────────────────────────

/**
 * Create the persistence strategy swap: InMemory ↔ Postgres.
 *
 * The naturality condition ensures that the order of
 * "apply event" and "persist" doesn't matter.
 */
export function createPersistenceSwap<T>(): StrategySwap<InMemoryReadModel<T>, PostgresReadModel<T>> {
  return createStrategySwap(
    'InMemory',
    'Postgres',
    inMemoryToPostgres,
    postgresToInMemory,
  )
}

function inMemoryToPostgres<T>(model: InMemoryReadModel<T>): PostgresReadModel<T> {
  const now = Date.now()
  const rows: PostgresRow<T>[] = []
  for (const [id, data] of model.data) {
    rows.push({ id, data, updatedAt: now })
  }
  return {
    kind: 'postgres',
    rows,
    tableName: 'read_model',
    version: model.version,
  }
}

function postgresToInMemory<T>(model: PostgresReadModel<T>): InMemoryReadModel<T> {
  const data = new Map<string, T>()
  for (const row of model.rows) {
    data.set(row.id, row.data)
  }
  return {
    kind: 'in-memory',
    data,
    version: model.version,
  }
}

// ─── Read Model Update (Morphism) ──────────────────────────────────────────

/**
 * Apply an update to an in-memory read model.
 */
export function updateInMemory<T>(
  id: string,
  updater: Morphism<T | undefined, T>,
): Morphism<InMemoryReadModel<T>, InMemoryReadModel<T>> {
  return (model) => {
    const newData = new Map(model.data)
    const existing = newData.get(id)
    newData.set(id, updater(existing))
    return { ...model, data: newData, version: model.version + 1 }
  }
}

/**
 * Apply an update to a Postgres read model.
 */
export function updatePostgres<T>(
  id: string,
  updater: Morphism<T | undefined, T>,
): Morphism<PostgresReadModel<T>, PostgresReadModel<T>> {
  return (model) => {
    const now = Date.now()
    const existingRow = model.rows.find(r => r.id === id)
    const newData = updater(existingRow?.data)
    const newRows = existingRow
      ? model.rows.map(r => r.id === id ? { id, data: newData, updatedAt: now } : r)
      : [...model.rows, { id, data: newData, updatedAt: now }]
    return { ...model, rows: newRows, version: model.version + 1 }
  }
}
