import { Hono } from 'hono'
import { eq, and } from 'drizzle-orm'
import { db, occasions } from '@omni-twin/db'
import { createOccasionSchema, updateOccasionSchema, updateOccasionStatusSchema } from '@omni-twin/shared'
import { requireAuth, requireVenueAccess, getUser } from '../auth/middleware'
import { parseBody, isResponse } from '../lib/validate'

const occasionRoutes = new Hono()

occasionRoutes.use('*', requireAuth)

/** GET /venues/:venueId/occasions — list occasions for a venue */
occasionRoutes.get('/', requireVenueAccess('venueId', 'owner', 'editor', 'viewer', 'commenter'), async (c) => {
  try {
    const venueId = c.req.param('venueId')!
    const results = await db.select().from(occasions).where(eq(occasions.venueId, venueId))
    return c.json({ occasions: results })
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error'
    process.stderr.write(JSON.stringify({ ts: new Date().toISOString(), level: 'error', op: 'list_occasions', error: msg }) + '\n')
    return c.json({ error: 'Failed to list occasions.' }, 500)
  }
})

/** POST /venues/:venueId/occasions — create an occasion */
occasionRoutes.post('/', requireVenueAccess('venueId', 'owner', 'editor'), async (c) => {
  try {
    const venueId = c.req.param('venueId')!
    const user = getUser(c)
    const data = await parseBody(c, createOccasionSchema)
    if (isResponse(data)) return data

    const [occasion] = await db
      .insert(occasions)
      .values({
        ...data,
        venueId,
        organizerId: user.id,
        dateStart: new Date(data.dateStart),
        dateEnd: new Date(data.dateEnd),
        setupTime: data.setupTime ? new Date(data.setupTime) : undefined,
        teardownTime: data.teardownTime ? new Date(data.teardownTime) : undefined,
      })
      .returning()

    if (!occasion) return c.json({ error: 'Failed to create occasion.' }, 500)
    return c.json({ occasion }, 201)
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error'
    process.stderr.write(JSON.stringify({ ts: new Date().toISOString(), level: 'error', op: 'create_occasion', error: msg }) + '\n')
    return c.json({ error: 'Failed to create occasion.' }, 500)
  }
})

/** GET /venues/:venueId/occasions/:occasionId — get occasion */
occasionRoutes.get('/:occasionId', requireVenueAccess('venueId', 'owner', 'editor', 'viewer', 'commenter'), async (c) => {
  try {
    const venueId = c.req.param('venueId')!
    const occasionId = c.req.param('occasionId')!
    const [occasion] = await db
      .select()
      .from(occasions)
      .where(and(eq(occasions.id, occasionId), eq(occasions.venueId, venueId)))

    if (!occasion) return c.json({ error: 'Occasion not found.' }, 404)
    return c.json({ occasion })
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error'
    process.stderr.write(JSON.stringify({ ts: new Date().toISOString(), level: 'error', op: 'get_occasion', error: msg }) + '\n')
    return c.json({ error: 'Failed to retrieve occasion.' }, 500)
  }
})

/** PATCH /venues/:venueId/occasions/:occasionId — update occasion */
occasionRoutes.patch('/:occasionId', requireVenueAccess('venueId', 'owner', 'editor'), async (c) => {
  try {
    const venueId = c.req.param('venueId')!
    const occasionId = c.req.param('occasionId')!
    const data = await parseBody(c, updateOccasionSchema)
    if (isResponse(data)) return data

    const values: Record<string, unknown> = { ...data, updatedAt: new Date() }
    if (data.dateStart) values['dateStart'] = new Date(data.dateStart)
    if (data.dateEnd) values['dateEnd'] = new Date(data.dateEnd)
    if (data.setupTime) values['setupTime'] = new Date(data.setupTime)
    if (data.teardownTime) values['teardownTime'] = new Date(data.teardownTime)

    const [occasion] = await db
      .update(occasions)
      .set(values)
      .where(and(eq(occasions.id, occasionId), eq(occasions.venueId, venueId)))
      .returning()

    if (!occasion) return c.json({ error: 'Occasion not found.' }, 404)
    return c.json({ occasion })
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error'
    process.stderr.write(JSON.stringify({ ts: new Date().toISOString(), level: 'error', op: 'update_occasion', error: msg }) + '\n')
    return c.json({ error: 'Failed to update occasion.' }, 500)
  }
})

/** PATCH /venues/:venueId/occasions/:occasionId/status — update status */
occasionRoutes.patch('/:occasionId/status', requireVenueAccess('venueId', 'owner', 'editor'), async (c) => {
  try {
    const venueId = c.req.param('venueId')!
    const occasionId = c.req.param('occasionId')!
    const data = await parseBody(c, updateOccasionStatusSchema)
    if (isResponse(data)) return data

    const [occasion] = await db
      .update(occasions)
      .set({ status: data.status, updatedAt: new Date() })
      .where(and(eq(occasions.id, occasionId), eq(occasions.venueId, venueId)))
      .returning()

    if (!occasion) return c.json({ error: 'Occasion not found.' }, 404)
    return c.json({ occasion })
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error'
    process.stderr.write(JSON.stringify({ ts: new Date().toISOString(), level: 'error', op: 'update_occasion_status', error: msg }) + '\n')
    return c.json({ error: 'Failed to update occasion status.' }, 500)
  }
})

/** DELETE /venues/:venueId/occasions/:occasionId — delete occasion */
occasionRoutes.delete('/:occasionId', requireVenueAccess('venueId', 'owner'), async (c) => {
  try {
    const venueId = c.req.param('venueId')!
    const occasionId = c.req.param('occasionId')!
    const [occasion] = await db
      .delete(occasions)
      .where(and(eq(occasions.id, occasionId), eq(occasions.venueId, venueId)))
      .returning()

    if (!occasion) return c.json({ error: 'Occasion not found.' }, 404)
    return c.json({ ok: true })
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error'
    process.stderr.write(JSON.stringify({ ts: new Date().toISOString(), level: 'error', op: 'delete_occasion', error: msg }) + '\n')
    return c.json({ error: 'Failed to delete occasion.' }, 500)
  }
})

export { occasionRoutes }
