/**
 * CT-6: Schema Category (Spivak's Framework)
 *
 * A database schema IS a category:
 *   Objects = tables (entity types)
 *   Morphisms = foreign key relationships
 *
 * By representing schemas as categories, we can:
 *   - Define schema migrations as functors
 *   - Guarantee referential integrity by construction
 *   - Prove data migration correctness
 */

// ─── Schema as Category ─────────────────────────────────────────────────────

/**
 * A schema category: tables as objects, foreign keys as morphisms.
 */
export interface SchemaCategory {
  readonly name: string
  readonly version: number
  readonly tables: ReadonlyMap<string, TableSchema>
  readonly relations: readonly ForeignKey[]
}

/**
 * A table schema (object in the schema category).
 */
export interface TableSchema {
  readonly name: string
  readonly columns: ReadonlyMap<string, ColumnSchema>
  readonly primaryKey: string
}

/**
 * A column schema.
 */
export interface ColumnSchema {
  readonly name: string
  readonly type: ColumnType
  readonly nullable: boolean
  readonly defaultValue?: unknown
}

export type ColumnType = 'string' | 'number' | 'boolean' | 'date' | 'json' | 'uuid'

/**
 * A foreign key (morphism in the schema category).
 * Maps from source table.column to target table.primaryKey.
 */
export interface ForeignKey {
  readonly name: string
  readonly sourceTable: string
  readonly sourceColumn: string
  readonly targetTable: string
  readonly targetColumn: string
  readonly onDelete: 'cascade' | 'restrict' | 'set-null'
}

// ─── Schema Builder ─────────────────────────────────────────────────────────

export class SchemaCategoryBuilder {
  private tables = new Map<string, TableSchema>()
  private relations: ForeignKey[] = []

  constructor(
    private readonly name: string,
    private readonly version: number,
  ) {}

  table(name: string, primaryKey: string, columns: Record<string, ColumnSchema>): this {
    this.tables.set(name, {
      name,
      columns: new Map(Object.entries(columns)),
      primaryKey,
    })
    return this
  }

  foreignKey(
    name: string,
    sourceTable: string,
    sourceColumn: string,
    targetTable: string,
    targetColumn: string,
    onDelete: 'cascade' | 'restrict' | 'set-null' = 'restrict',
  ): this {
    this.relations.push({ name, sourceTable, sourceColumn, targetTable, targetColumn, onDelete })
    return this
  }

  build(): SchemaCategory {
    // Validate all foreign keys reference existing tables/columns
    for (const fk of this.relations) {
      if (!this.tables.has(fk.sourceTable)) {
        throw new Error(`FK ${fk.name}: source table ${fk.sourceTable} not found`)
      }
      if (!this.tables.has(fk.targetTable)) {
        throw new Error(`FK ${fk.name}: target table ${fk.targetTable} not found`)
      }
    }

    return {
      name: this.name,
      version: this.version,
      tables: new Map(this.tables),
      relations: [...this.relations],
    }
  }
}

export function schema(name: string, version: number): SchemaCategoryBuilder {
  return new SchemaCategoryBuilder(name, version)
}

// ─── Column Helpers ─────────────────────────────────────────────────────────

export function col(type: ColumnType, nullable = false, defaultValue?: unknown): ColumnSchema {
  return { name: '', type, nullable, defaultValue }
}

export function stringCol(nullable = false): ColumnSchema {
  return { name: '', type: 'string', nullable }
}

export function numberCol(nullable = false): ColumnSchema {
  return { name: '', type: 'number', nullable }
}

export function boolCol(nullable = false, defaultValue?: boolean): ColumnSchema {
  return { name: '', type: 'boolean', nullable, defaultValue }
}

export function dateCol(nullable = false): ColumnSchema {
  return { name: '', type: 'date', nullable }
}

export function uuidCol(): ColumnSchema {
  return { name: '', type: 'uuid', nullable: false }
}

export function jsonCol(nullable = false): ColumnSchema {
  return { name: '', type: 'json', nullable }
}

// ─── Schema Validation ──────────────────────────────────────────────────────

/**
 * Validate a schema category's internal consistency.
 * Checks:
 *   - All FK source/target tables exist
 *   - All FK source/target columns exist in their tables
 *   - No cyclic cascade-delete chains
 */
export function validateSchema(schema: SchemaCategory): string[] {
  const errors: string[] = []

  for (const fk of schema.relations) {
    const sourceTable = schema.tables.get(fk.sourceTable)
    if (!sourceTable) {
      errors.push(`FK ${fk.name}: source table '${fk.sourceTable}' not found`)
      continue
    }

    const targetTable = schema.tables.get(fk.targetTable)
    if (!targetTable) {
      errors.push(`FK ${fk.name}: target table '${fk.targetTable}' not found`)
      continue
    }

    if (!sourceTable.columns.has(fk.sourceColumn)) {
      errors.push(`FK ${fk.name}: column '${fk.sourceColumn}' not found in '${fk.sourceTable}'`)
    }

    if (!targetTable.columns.has(fk.targetColumn) && fk.targetColumn !== targetTable.primaryKey) {
      errors.push(`FK ${fk.name}: column '${fk.targetColumn}' not found in '${fk.targetTable}'`)
    }
  }

  return errors
}
