import { Hono } from 'hono'
import { eq, and, isNull } from 'drizzle-orm'
import { db, venues, venuePermissions } from '@omni-twin/db'
import { createVenueSchema, updateVenueSchema } from '@omni-twin/shared'
import { requireAuth, requireVenueAccess, getUser } from '../auth/middleware'
import { parseBody, isResponse } from '../lib/validate'

const venueRoutes = new Hono()

// All venue routes require authentication
venueRoutes.use('*', requireAuth)

/** GET /venues — list venues the current user has access to */
venueRoutes.get('/', async (c) => {
  try {
    const user = getUser(c)

    const perms = await db
      .select({
        venueId: venuePermissions.venueId,
        role: venuePermissions.role,
      })
      .from(venuePermissions)
      .where(eq(venuePermissions.userId, user.id))

    if (perms.length === 0) return c.json({ venues: [] })

    const venueIds = perms.map((p) => p.venueId)
    const results = await db
      .select()
      .from(venues)
      .where(and(isNull(venues.archivedAt)))

    const accessible = results.filter((v) => venueIds.includes(v.id))
    return c.json({ venues: accessible })
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error'
    process.stderr.write(JSON.stringify({ ts: new Date().toISOString(), level: 'error', op: 'list_venues', error: msg }) + '\n')
    return c.json({ error: 'Failed to list venues.' }, 500)
  }
})

/** POST /venues — create a new venue */
venueRoutes.post('/', async (c) => {
  try {
    const user = getUser(c)
    const data = await parseBody(c, createVenueSchema)
    if (isResponse(data)) return data

    const [venue] = await db
      .insert(venues)
      .values({ ...data, ownerId: user.id })
      .returning()

    if (!venue) return c.json({ error: 'Failed to create venue.' }, 500)

    // Auto-assign owner permission
    await db.insert(venuePermissions).values({
      venueId: venue.id,
      userId: user.id,
      role: 'owner',
    })

    return c.json({ venue }, 201)
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error'
    process.stderr.write(JSON.stringify({ ts: new Date().toISOString(), level: 'error', op: 'create_venue', error: msg }) + '\n')
    return c.json({ error: 'Failed to create venue.' }, 500)
  }
})

/** GET /venues/:venueId — get venue details */
venueRoutes.get('/:venueId', requireVenueAccess('venueId', 'owner', 'editor', 'viewer', 'commenter'), async (c) => {
  try {
    const venueId = c.req.param('venueId')
    const [venue] = await db.select().from(venues).where(eq(venues.id, venueId!))
    if (!venue) return c.json({ error: 'Venue not found.' }, 404)
    return c.json({ venue })
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error'
    process.stderr.write(JSON.stringify({ ts: new Date().toISOString(), level: 'error', op: 'get_venue', error: msg }) + '\n')
    return c.json({ error: 'Failed to retrieve venue.' }, 500)
  }
})

/** PATCH /venues/:venueId — update venue */
venueRoutes.patch('/:venueId', requireVenueAccess('venueId', 'owner', 'editor'), async (c) => {
  try {
    const venueId = c.req.param('venueId')
    const data = await parseBody(c, updateVenueSchema)
    if (isResponse(data)) return data

    const [venue] = await db
      .update(venues)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(venues.id, venueId!))
      .returning()

    if (!venue) return c.json({ error: 'Venue not found.' }, 404)
    return c.json({ venue })
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error'
    process.stderr.write(JSON.stringify({ ts: new Date().toISOString(), level: 'error', op: 'update_venue', error: msg }) + '\n')
    return c.json({ error: 'Failed to update venue.' }, 500)
  }
})

/** DELETE /venues/:venueId — archive venue (soft delete) */
venueRoutes.delete('/:venueId', requireVenueAccess('venueId', 'owner'), async (c) => {
  try {
    const venueId = c.req.param('venueId')

    const [venue] = await db
      .update(venues)
      .set({ archivedAt: new Date(), status: 'archived', updatedAt: new Date() })
      .where(eq(venues.id, venueId!))
      .returning()

    if (!venue) return c.json({ error: 'Venue not found.' }, 404)
    return c.json({ venue })
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error'
    process.stderr.write(JSON.stringify({ ts: new Date().toISOString(), level: 'error', op: 'archive_venue', error: msg }) + '\n')
    return c.json({ error: 'Failed to archive venue.' }, 500)
  }
})

export { venueRoutes }
