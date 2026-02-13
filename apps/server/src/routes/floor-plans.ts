import { Hono } from 'hono'
import { eq, and } from 'drizzle-orm'
import { db, floorPlans } from '@omni-twin/db'
import { createFloorPlanSchema, updateFloorPlanSchema } from '@omni-twin/shared'
import { requireAuth, requireVenueAccess, getUser } from '../auth/middleware'
import { parseBody, isResponse } from '../lib/validate'

const floorPlanRoutes = new Hono()

floorPlanRoutes.use('*', requireAuth)

/** GET /venues/:venueId/floor-plans — list floor plans for a venue */
floorPlanRoutes.get('/', requireVenueAccess('venueId', 'owner', 'editor', 'viewer', 'commenter'), async (c) => {
  try {
    const venueId = c.req.param('venueId')!
    const plans = await db.select().from(floorPlans).where(eq(floorPlans.venueId, venueId))
    return c.json({ floorPlans: plans })
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error'
    process.stderr.write(JSON.stringify({ ts: new Date().toISOString(), level: 'error', op: 'list_floor_plans', error: msg }) + '\n')
    return c.json({ error: 'Failed to list floor plans.' }, 500)
  }
})

/** POST /venues/:venueId/floor-plans — create a floor plan */
floorPlanRoutes.post('/', requireVenueAccess('venueId', 'owner', 'editor'), async (c) => {
  try {
    const venueId = c.req.param('venueId')!
    const user = getUser(c)
    const data = await parseBody(c, createFloorPlanSchema)
    if (isResponse(data)) return data

    const [plan] = await db
      .insert(floorPlans)
      .values({ ...data, venueId, createdBy: user.id })
      .returning()

    if (!plan) return c.json({ error: 'Failed to create floor plan.' }, 500)
    return c.json({ floorPlan: plan }, 201)
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error'
    process.stderr.write(JSON.stringify({ ts: new Date().toISOString(), level: 'error', op: 'create_floor_plan', error: msg }) + '\n')
    return c.json({ error: 'Failed to create floor plan.' }, 500)
  }
})

/** GET /venues/:venueId/floor-plans/:planId — get floor plan */
floorPlanRoutes.get('/:planId', requireVenueAccess('venueId', 'owner', 'editor', 'viewer', 'commenter'), async (c) => {
  try {
    const venueId = c.req.param('venueId')!
    const planId = c.req.param('planId')!
    const [plan] = await db
      .select()
      .from(floorPlans)
      .where(and(eq(floorPlans.id, planId), eq(floorPlans.venueId, venueId)))

    if (!plan) return c.json({ error: 'Floor plan not found.' }, 404)
    return c.json({ floorPlan: plan })
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error'
    process.stderr.write(JSON.stringify({ ts: new Date().toISOString(), level: 'error', op: 'get_floor_plan', error: msg }) + '\n')
    return c.json({ error: 'Failed to retrieve floor plan.' }, 500)
  }
})

/** PATCH /venues/:venueId/floor-plans/:planId — update floor plan */
floorPlanRoutes.patch('/:planId', requireVenueAccess('venueId', 'owner', 'editor'), async (c) => {
  try {
    const venueId = c.req.param('venueId')!
    const planId = c.req.param('planId')!
    const data = await parseBody(c, updateFloorPlanSchema)
    if (isResponse(data)) return data

    const [plan] = await db
      .update(floorPlans)
      .set({ ...data, updatedAt: new Date() })
      .where(and(eq(floorPlans.id, planId), eq(floorPlans.venueId, venueId)))
      .returning()

    if (!plan) return c.json({ error: 'Floor plan not found.' }, 404)
    return c.json({ floorPlan: plan })
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error'
    process.stderr.write(JSON.stringify({ ts: new Date().toISOString(), level: 'error', op: 'update_floor_plan', error: msg }) + '\n')
    return c.json({ error: 'Failed to update floor plan.' }, 500)
  }
})

/** DELETE /venues/:venueId/floor-plans/:planId — delete floor plan */
floorPlanRoutes.delete('/:planId', requireVenueAccess('venueId', 'owner', 'editor'), async (c) => {
  try {
    const venueId = c.req.param('venueId')!
    const planId = c.req.param('planId')!
    const [plan] = await db
      .delete(floorPlans)
      .where(and(eq(floorPlans.id, planId), eq(floorPlans.venueId, venueId)))
      .returning()

    if (!plan) return c.json({ error: 'Floor plan not found.' }, 404)
    return c.json({ ok: true })
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error'
    process.stderr.write(JSON.stringify({ ts: new Date().toISOString(), level: 'error', op: 'delete_floor_plan', error: msg }) + '\n')
    return c.json({ error: 'Failed to delete floor plan.' }, 500)
  }
})

export { floorPlanRoutes }
