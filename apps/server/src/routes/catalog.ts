import { Hono } from 'hono'
import { eq } from 'drizzle-orm'
import { db, furnitureCatalog } from '@omni-twin/db'
import { createCatalogItemSchema, updateCatalogItemSchema } from '@omni-twin/shared'
import { requireAuth, requireRole, getUser } from '../auth/middleware'
import { parseBody, isResponse } from '../lib/validate'

const catalogRoutes = new Hono()

catalogRoutes.use('*', requireAuth)

/** GET /catalog — list all furniture catalog items */
catalogRoutes.get('/', async (c) => {
  const items = await db.select().from(furnitureCatalog)
  return c.json({ items })
})

/** POST /catalog — create a catalog item (owner/manager only) */
catalogRoutes.post('/', requireRole('owner', 'manager'), async (c) => {
  const user = getUser(c)
  const data = await parseBody(c, createCatalogItemSchema)
  if (isResponse(data)) return data

  const [item] = await db
    .insert(furnitureCatalog)
    .values({ ...data, createdBy: user.id })
    .returning()

  if (!item) return c.json({ error: 'Failed to create catalog item.' }, 500)
  return c.json({ item }, 201)
})

/** GET /catalog/:itemId — get a catalog item */
catalogRoutes.get('/:itemId', async (c) => {
  const itemId = c.req.param('itemId')!
  const [item] = await db.select().from(furnitureCatalog).where(eq(furnitureCatalog.id, itemId))
  if (!item) return c.json({ error: 'Catalog item not found.' }, 404)
  return c.json({ item })
})

/** PATCH /catalog/:itemId — update a catalog item (owner/manager only) */
catalogRoutes.patch('/:itemId', requireRole('owner', 'manager'), async (c) => {
  const itemId = c.req.param('itemId')!
  const data = await parseBody(c, updateCatalogItemSchema)
  if (isResponse(data)) return data

  const [item] = await db
    .update(furnitureCatalog)
    .set({ ...data, updatedAt: new Date() })
    .where(eq(furnitureCatalog.id, itemId))
    .returning()

  if (!item) return c.json({ error: 'Catalog item not found.' }, 404)
  return c.json({ item })
})

/** DELETE /catalog/:itemId — delete a catalog item (owner only) */
catalogRoutes.delete('/:itemId', requireRole('owner'), async (c) => {
  const itemId = c.req.param('itemId')!
  const [item] = await db
    .delete(furnitureCatalog)
    .where(eq(furnitureCatalog.id, itemId))
    .returning()

  if (!item) return c.json({ error: 'Catalog item not found.' }, 404)
  return c.json({ ok: true })
})

export { catalogRoutes }
