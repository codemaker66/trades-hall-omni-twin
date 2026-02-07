import { drizzle } from 'drizzle-orm/postgres-js'
import postgres from 'postgres'
import * as schema from './schema/index'

const DATABASE_URL = process.env['DATABASE_URL'] ?? 'postgresql://omnitwin:omnitwin@localhost:5432/omnitwin'

/** Connection pool for queries. */
const queryClient = postgres(DATABASE_URL, {
  max: 10,
  idle_timeout: 20,
  connect_timeout: 10,
})

/** Drizzle ORM instance with full schema. */
export const db = drizzle(queryClient, { schema })

/** Raw postgres client for health checks and advanced usage. */
export { queryClient }
