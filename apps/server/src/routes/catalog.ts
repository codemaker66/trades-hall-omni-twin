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
  try {
    const items = await db.select().from(furnitureCatalog)
    return c.json({ items })
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error'
    process.stderr.write(JSON.stringify({ ts: new Date().toISOString(), level: 'error', op: 'list_catalog', error: msg }) + '\n')
    return c.json({ error: 'Failed to list catalog items.' }, 500)
  }
})

/** POST /catalog — create a catalog item (owner/manager only) */
catalogRoutes.post('/', requireRole('owner', 'manager'), async (c) => {
  try {
    const user = getUser(c)
    const data = await parseBody(c, createCatalogItemSchema)
    if (isResponse(data)) return data

    const [item] = await db
      .insert(furnitureCatalog)
      .values({ ...data, createdBy: user.id })
      .returning()

    if (!item) return c.json({ error: 'Failed to create catalog item.' }, 500)
    return c.json({ item }, 201)
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error'
    process.stderr.write(JSON.stringify({ ts: new Date().toISOString(), level: 'error', op: 'create_catalog_item', error: msg }) + '\n')
    return c.json({ error: 'Failed to create catalog item.' }, 500)
  }
})

/** GET /catalog/:itemId — get a catalog item */
catalogRoutes.get('/:itemId', async (c) => {
  try {
    const itemId = c.req.param('itemId')!
    const [item] = await db.select().from(furnitureCatalog).where(eq(furnitureCatalog.id, itemId))
    if (!item) return c.json({ error: 'Catalog item not found.' }, 404)
    return c.json({ item })
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error'
    process.stderr.write(JSON.stringify({ ts: new Date().toISOString(), level: 'error', op: 'get_catalog_item', error: msg }) + '\n')
    return c.json({ error: 'Failed to retrieve catalog item.' }, 500)
  }
})

/** PATCH /catalog/:itemId — update a catalog item (owner/manager only) */
catalogRoutes.patch('/:itemId', requireRole('owner', 'manager'), async (c) => {
  try {
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
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error'
    process.stderr.write(JSON.stringify({ ts: new Date().toISOString(), level: 'error', op: 'update_catalog_item', error: msg }) + '\n')
    return c.json({ error: 'Failed to update catalog item.' }, 500)
  }
})

/** DELETE /catalog/:itemId — delete a catalog item (owner only) */
catalogRoutes.delete('/:itemId', requireRole('owner'), async (c) => {
  try {
    const itemId = c.req.param('itemId')!
    const [item] = await db
      .delete(furnitureCatalog)
      .where(eq(furnitureCatalog.id, itemId))
      .returning()

    if (!item) return c.json({ error: 'Catalog item not found.' }, 404)
    return c.json({ ok: true })
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error'
    process.stderr.write(JSON.stringify({ ts: new Date().toISOString(), level: 'error', op: 'delete_catalog_item', error: msg }) + '\n')
    return c.json({ error: 'Failed to delete catalog item.' }, 500)
  }
})

export { catalogRoutes }
