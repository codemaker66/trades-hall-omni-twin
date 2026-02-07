/**
 * CT-6: Migration Builder DSL
 *
 * A type-safe DSL for building schema migrations categorically.
 * Each migration is a functor between schema categories,
 * guaranteeing referential integrity by construction.
 */

import type { SchemaCategory } from './schema-category'
import { schema, stringCol, numberCol, boolCol, dateCol, uuidCol, jsonCol } from './schema-category'
import type { SchemaFunctor } from './schema-functor'
import { schemaFunctor, composeSchemaFunctors } from './schema-functor'
import type { ColumnMapping } from './schema-functor'

// ─── Migration Step ─────────────────────────────────────────────────────────

export interface MigrationStep {
  readonly name: string
  readonly description: string
  readonly functor: SchemaFunctor
}

// ─── Migration Chain ────────────────────────────────────────────────────────

/**
 * A chain of migrations: F₁, F₂, ..., Fₙ
 * Can be composed into a single functor: Fₙ ∘ ... ∘ F₂ ∘ F₁
 */
export class MigrationChain {
  private steps: MigrationStep[] = []

  constructor(readonly initialSchema: SchemaCategory) {}

  /**
   * Add a migration step.
   */
  addStep(step: MigrationStep): this {
    this.steps.push(step)
    return this
  }

  /**
   * Get the composed functor for all steps.
   */
  compose(): SchemaFunctor | null {
    if (this.steps.length === 0) return null
    if (this.steps.length === 1) return this.steps[0]!.functor

    let composed = this.steps[0]!.functor
    for (let i = 1; i < this.steps.length; i++) {
      composed = composeSchemaFunctors(composed, this.steps[i]!.functor)
    }
    return composed
  }

  /**
   * Migrate data through all steps.
   */
  migrateRow(tableName: string, row: Record<string, unknown>): Record<string, unknown> | null {
    let currentRow: Record<string, unknown> | null = row
    let currentTable = tableName

    for (const step of this.steps) {
      if (!currentRow) return null
      currentRow = step.functor.pushforward(currentTable, currentRow)
      // Update table name for next step
      const mapping = step.functor.tableMappings.get(currentTable)
      if (mapping) currentTable = mapping.targetTable
    }

    return currentRow
  }

  /**
   * Get all steps in the chain.
   */
  getSteps(): readonly MigrationStep[] {
    return [...this.steps]
  }

  /**
   * Get the final schema after all migrations.
   */
  finalSchema(): SchemaCategory {
    if (this.steps.length === 0) return this.initialSchema
    return this.steps[this.steps.length - 1]!.functor.target
  }

  /**
   * Get the schema at a specific version.
   */
  schemaAtStep(stepIndex: number): SchemaCategory {
    if (stepIndex < 0 || stepIndex >= this.steps.length) {
      return this.initialSchema
    }
    return this.steps[stepIndex]!.functor.target
  }
}

// ─── Convenience Builders ───────────────────────────────────────────────────

/**
 * Create a migration that adds a column to a table.
 */
export function addColumnMigration(
  source: SchemaCategory,
  tableName: string,
  columnName: string,
  columnType: 'string' | 'number' | 'boolean' | 'date' | 'uuid' | 'json',
  defaultValue: unknown,
): MigrationStep {
  const sourceTable = source.tables.get(tableName)
  if (!sourceTable) throw new Error(`Table '${tableName}' not found`)

  // Build target schema with new column
  const colFn = { string: stringCol, number: numberCol, boolean: boolCol, date: dateCol, uuid: uuidCol, json: jsonCol }[columnType]
  const newCol = colFn()

  const targetBuilder = schema(source.name, source.version + 1)
  for (const [name, table] of source.tables) {
    const columns: Record<string, typeof newCol> = {}
    for (const [cName, cSchema] of table.columns) {
      columns[cName] = cSchema
    }
    if (name === tableName) {
      columns[columnName] = { ...newCol, name: columnName }
    }
    targetBuilder.table(name, table.primaryKey, columns)
  }
  for (const fk of source.relations) {
    targetBuilder.foreignKey(fk.name, fk.sourceTable, fk.sourceColumn, fk.targetTable, fk.targetColumn, fk.onDelete)
  }
  const target = targetBuilder.build()

  // Build functor: identity for all columns, default for new one
  const builder = schemaFunctor(`add-${columnName}-to-${tableName}`, source, target)
  for (const [name, table] of source.tables) {
    const columnMappings: Record<string, ColumnMapping> = {}
    for (const [cName] of table.columns) {
      columnMappings[cName] = { kind: 'rename', targetColumn: cName }
    }
    if (name === tableName) {
      // The new column gets a default value
      columnMappings[columnName] = { kind: 'default', targetColumn: columnName, defaultValue }
    }
    builder.mapTable(name, name, columnMappings)
  }

  return {
    name: `add-${columnName}-to-${tableName}`,
    description: `Add column '${columnName}' (${columnType}) to '${tableName}'`,
    functor: builder.build(),
  }
}

/**
 * Create a migration that renames a column.
 */
export function renameColumnMigration(
  source: SchemaCategory,
  tableName: string,
  oldColumnName: string,
  newColumnName: string,
): MigrationStep {
  const sourceTable = source.tables.get(tableName)
  if (!sourceTable) throw new Error(`Table '${tableName}' not found`)
  if (!sourceTable.columns.has(oldColumnName)) {
    throw new Error(`Column '${oldColumnName}' not found in '${tableName}'`)
  }

  // Build target schema with renamed column
  const targetBuilder = schema(source.name, source.version + 1)
  for (const [name, table] of source.tables) {
    const columns: Record<string, any> = {}
    for (const [cName, cSchema] of table.columns) {
      if (name === tableName && cName === oldColumnName) {
        columns[newColumnName] = { ...cSchema, name: newColumnName }
      } else {
        columns[cName] = cSchema
      }
    }
    targetBuilder.table(name, table.primaryKey, columns)
  }
  for (const fk of source.relations) {
    targetBuilder.foreignKey(fk.name, fk.sourceTable, fk.sourceColumn, fk.targetTable, fk.targetColumn, fk.onDelete)
  }
  const target = targetBuilder.build()

  // Build functor
  const builder = schemaFunctor(`rename-${oldColumnName}-to-${newColumnName}`, source, target)
  for (const [name, table] of source.tables) {
    const columnMappings: Record<string, ColumnMapping> = {}
    for (const [cName] of table.columns) {
      if (name === tableName && cName === oldColumnName) {
        columnMappings[cName] = { kind: 'rename', targetColumn: newColumnName }
      } else {
        columnMappings[cName] = { kind: 'rename', targetColumn: cName }
      }
    }
    builder.mapTable(name, name, columnMappings)
  }

  return {
    name: `rename-${oldColumnName}-to-${newColumnName}`,
    description: `Rename column '${oldColumnName}' to '${newColumnName}' in '${tableName}'`,
    functor: builder.build(),
  }
}
