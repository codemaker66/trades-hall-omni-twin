/**
 * CT-6: Schema Functor (Functorial Data Migration)
 *
 * A schema migration IS a functor between schema categories:
 *   F: OldSchema → NewSchema
 *
 * Maps:
 *   - Old tables to new tables
 *   - Old relationships to new relationships
 *   - PRESERVES composition (referential integrity by construction)
 *
 * Data migration operators (from Spivak):
 *   Δ_F (pullback):         migrate data from new schema back to old
 *   Σ_F (left pushforward):  migrate data forward (merge/union)
 *   Π_F (right pushforward): migrate data forward (product/join)
 */

import type { SchemaCategory, TableSchema, ForeignKey } from './schema-category'

// ─── Schema Functor ─────────────────────────────────────────────────────────

/** A table mapping: how one table maps to another. */
export interface TableMapping {
  readonly sourceTable: string
  readonly targetTable: string
  readonly columnMappings: ReadonlyMap<string, ColumnMapping>
}

/** How a column maps during migration. */
export type ColumnMapping =
  | { readonly kind: 'rename'; readonly targetColumn: string }
  | { readonly kind: 'transform'; readonly targetColumn: string; readonly transform: (value: unknown) => unknown }
  | { readonly kind: 'drop' }  // column is removed
  | { readonly kind: 'default'; readonly targetColumn: string; readonly defaultValue: unknown }

/**
 * A schema functor: maps from one schema category to another.
 * This represents a schema migration.
 */
export interface SchemaFunctor {
  readonly name: string
  readonly source: SchemaCategory
  readonly target: SchemaCategory
  readonly tableMappings: ReadonlyMap<string, TableMapping>

  /** Δ_F (pullback): migrate a row from new schema back to old. */
  pullback(targetTable: string, row: Record<string, unknown>): Record<string, unknown> | null

  /** Σ_F (left pushforward): migrate a row forward. */
  pushforward(sourceTable: string, row: Record<string, unknown>): Record<string, unknown> | null
}

// ─── Schema Functor Builder ─────────────────────────────────────────────────

export class SchemaFunctorBuilder {
  private tableMappings = new Map<string, TableMapping>()

  constructor(
    private readonly name: string,
    private readonly source: SchemaCategory,
    private readonly target: SchemaCategory,
  ) {}

  /**
   * Map a source table to a target table with column mappings.
   */
  mapTable(
    sourceTable: string,
    targetTable: string,
    columnMappings: Record<string, ColumnMapping>,
  ): this {
    this.tableMappings.set(sourceTable, {
      sourceTable,
      targetTable,
      columnMappings: new Map(Object.entries(columnMappings)),
    })
    return this
  }

  /**
   * Simple 1:1 table rename (all columns preserved).
   */
  renameTable(sourceTable: string, targetTable: string): this {
    const srcTable = this.source.tables.get(sourceTable)
    if (!srcTable) throw new Error(`Source table '${sourceTable}' not found`)

    const columnMappings: Record<string, ColumnMapping> = {}
    for (const [colName] of srcTable.columns) {
      columnMappings[colName] = { kind: 'rename', targetColumn: colName }
    }

    return this.mapTable(sourceTable, targetTable, columnMappings)
  }

  build(): SchemaFunctor {
    const mappings = new Map(this.tableMappings)

    return {
      name: this.name,
      source: this.source,
      target: this.target,
      tableMappings: mappings,

      pullback(targetTable: string, row: Record<string, unknown>): Record<string, unknown> | null {
        // Find the source table that maps to this target
        for (const [_srcTable, mapping] of mappings) {
          if (mapping.targetTable === targetTable) {
            return pullbackRow(mapping, row)
          }
        }
        return null
      },

      pushforward(sourceTable: string, row: Record<string, unknown>): Record<string, unknown> | null {
        const mapping = mappings.get(sourceTable)
        if (!mapping) return null
        return pushforwardRow(mapping, row)
      },
    }
  }
}

export function schemaFunctor(
  name: string,
  source: SchemaCategory,
  target: SchemaCategory,
): SchemaFunctorBuilder {
  return new SchemaFunctorBuilder(name, source, target)
}

// ─── Row Migration ──────────────────────────────────────────────────────────

/**
 * Σ_F: Push a row forward through the functor (old → new).
 */
function pushforwardRow(
  mapping: TableMapping,
  row: Record<string, unknown>,
): Record<string, unknown> {
  const result: Record<string, unknown> = {}

  for (const [sourceCol, colMapping] of mapping.columnMappings) {
    switch (colMapping.kind) {
      case 'rename':
        result[colMapping.targetColumn] = row[sourceCol]
        break
      case 'transform':
        result[colMapping.targetColumn] = colMapping.transform(row[sourceCol])
        break
      case 'drop':
        // Column is dropped — don't include in result
        break
      case 'default':
        result[colMapping.targetColumn] = row[sourceCol] ?? colMapping.defaultValue
        break
    }
  }

  return result
}

/**
 * Δ_F: Pull a row back through the functor (new → old).
 */
function pullbackRow(
  mapping: TableMapping,
  row: Record<string, unknown>,
): Record<string, unknown> {
  const result: Record<string, unknown> = {}

  for (const [sourceCol, colMapping] of mapping.columnMappings) {
    switch (colMapping.kind) {
      case 'rename':
        result[sourceCol] = row[colMapping.targetColumn]
        break
      case 'transform':
        // Can't reverse an arbitrary transform — use target value as-is
        result[sourceCol] = row[colMapping.targetColumn]
        break
      case 'drop':
        // Column was dropped — set to null
        result[sourceCol] = null
        break
      case 'default':
        result[sourceCol] = row[colMapping.targetColumn]
        break
    }
  }

  return result
}

// ─── Composition ────────────────────────────────────────────────────────────

/**
 * Compose two schema functors: G ∘ F.
 * If F: A → B and G: B → C, then G ∘ F: A → C.
 */
export function composeSchemaFunctors(
  f: SchemaFunctor,
  g: SchemaFunctor,
): SchemaFunctor {
  const composedMappings = new Map<string, TableMapping>()

  for (const [srcTable, fMapping] of f.tableMappings) {
    const gMapping = g.tableMappings.get(fMapping.targetTable)
    if (!gMapping) continue

    const composedColumns = new Map<string, ColumnMapping>()

    for (const [srcCol, fColMap] of fMapping.columnMappings) {
      if (fColMap.kind === 'drop') {
        composedColumns.set(srcCol, { kind: 'drop' })
        continue
      }

      const intermediateCol = fColMap.kind === 'rename' || fColMap.kind === 'transform' || fColMap.kind === 'default'
        ? fColMap.targetColumn
        : srcCol

      const gColMap = gMapping.columnMappings.get(intermediateCol)
      if (!gColMap) {
        composedColumns.set(srcCol, { kind: 'drop' })
        continue
      }

      if (gColMap.kind === 'drop') {
        composedColumns.set(srcCol, { kind: 'drop' })
      } else if (fColMap.kind === 'transform' || gColMap.kind === 'transform') {
        const fTransform = fColMap.kind === 'transform' ? fColMap.transform : (x: unknown) => x
        const gTransform = gColMap.kind === 'transform' ? gColMap.transform : (x: unknown) => x
        const targetCol = gColMap.kind === 'rename' || gColMap.kind === 'transform' || gColMap.kind === 'default'
          ? gColMap.targetColumn
          : intermediateCol
        composedColumns.set(srcCol, {
          kind: 'transform',
          targetColumn: targetCol,
          transform: (x: unknown) => gTransform(fTransform(x)),
        })
      } else {
        const targetCol = gColMap.kind === 'rename' || gColMap.kind === 'default'
          ? gColMap.targetColumn
          : intermediateCol
        composedColumns.set(srcCol, { kind: 'rename', targetColumn: targetCol })
      }
    }

    composedMappings.set(srcTable, {
      sourceTable: srcTable,
      targetTable: gMapping.targetTable,
      columnMappings: composedColumns,
    })
  }

  const builder = new SchemaFunctorBuilder(`${g.name} ∘ ${f.name}`, f.source, g.target)
  // Directly build from composed mappings
  return {
    name: `${g.name} ∘ ${f.name}`,
    source: f.source,
    target: g.target,
    tableMappings: composedMappings,
    pullback(targetTable, row) {
      // G pullback then F pullback
      const intermediate = g.pullback(targetTable, row)
      if (!intermediate) return null
      // Find intermediate table name
      for (const [_src, gMap] of g.tableMappings) {
        if (gMap.targetTable === targetTable) {
          return f.pullback(gMap.sourceTable, intermediate)
        }
      }
      return null
    },
    pushforward(sourceTable, row) {
      const intermediate = f.pushforward(sourceTable, row)
      if (!intermediate) return null
      const fMapping = f.tableMappings.get(sourceTable)
      if (!fMapping) return null
      return g.pushforward(fMapping.targetTable, intermediate)
    },
  }
}
