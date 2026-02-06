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
})

/** POST /venues — create a new venue */
venueRoutes.post('/', async (c) => {
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
})

/** GET /venues/:venueId — get venue details */
venueRoutes.get('/:venueId', requireVenueAccess('venueId', 'owner', 'editor', 'viewer', 'commenter'), async (c) => {
  const venueId = c.req.param('venueId')
  const [venue] = await db.select().from(venues).where(eq(venues.id, venueId!))
  if (!venue) return c.json({ error: 'Venue not found.' }, 404)
  return c.json({ venue })
})

/** PATCH /venues/:venueId — update venue */
venueRoutes.patch('/:venueId', requireVenueAccess('venueId', 'owner', 'editor'), async (c) => {
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
})

/** DELETE /venues/:venueId — archive venue (soft delete) */
venueRoutes.delete('/:venueId', requireVenueAccess('venueId', 'owner'), async (c) => {
  const venueId = c.req.param('venueId')

  const [venue] = await db
    .update(venues)
    .set({ archivedAt: new Date(), status: 'archived', updatedAt: new Date() })
    .where(eq(venues.id, venueId!))
    .returning()

  if (!venue) return c.json({ error: 'Venue not found.' }, 404)
  return c.json({ venue })
})

export { venueRoutes }
