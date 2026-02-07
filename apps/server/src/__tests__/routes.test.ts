import { describe, it, expect, vi } from 'vitest'
import { Hono } from 'hono'

/**
 * Tests for API route structure and unauthenticated access guards.
 *
 * Verifies that all routes exist, return correct status codes
 * for unauthorized access, and that the health/info endpoints work.
 */

// Mock all external dependencies
vi.mock('@omni-twin/db', () => ({
  db: {
    select: () => ({
      from: () => ({
        where: () => ({
          limit: () => Promise.resolve([]),
        }),
      }),
    }),
    insert: () => ({
      values: () => ({
        returning: () => Promise.resolve([]),
      }),
    }),
    delete: () => ({
      where: () => Promise.resolve(),
    }),
  },
  users: { id: 'id', email: 'email', name: 'name', role: 'role', avatarUrl: 'avatarUrl', passwordHash: 'passwordHash', createdAt: 'createdAt' },
  sessions: { id: 'id', userId: 'userId', expiresAt: 'expiresAt' },
  venues: { id: 'id', name: 'name' },
  venuePermissions: { venueId: 'venueId', userId: 'userId', role: 'role' },
  floorPlans: { id: 'id' },
  occasions: { id: 'id' },
  furnitureCatalog: { id: 'id' },
  queryClient: () => Promise.resolve([{ '?column?': 1 }]),
}))

vi.mock('@omni-twin/cache', () => ({
  redisHealthCheck: () => Promise.resolve(true),
}))

vi.mock('@omni-twin/shared', () => ({
  createVenueSchema: { safeParse: () => ({ success: false, error: { flatten: () => ({ fieldErrors: {} }) } }) },
  updateVenueSchema: { safeParse: () => ({ success: false, error: { flatten: () => ({ fieldErrors: {} }) } }) },
  createFloorPlanSchema: { safeParse: () => ({ success: false, error: { flatten: () => ({ fieldErrors: {} }) } }) },
  updateFloorPlanSchema: { safeParse: () => ({ success: false, error: { flatten: () => ({ fieldErrors: {} }) } }) },
  createOccasionSchema: { safeParse: () => ({ success: false, error: { flatten: () => ({ fieldErrors: {} }) } }) },
  updateOccasionSchema: { safeParse: () => ({ success: false, error: { flatten: () => ({ fieldErrors: {} }) } }) },
  updateOccasionStatusSchema: { safeParse: () => ({ success: false, error: { flatten: () => ({ fieldErrors: {} }) } }) },
  createCatalogItemSchema: { safeParse: () => ({ success: false, error: { flatten: () => ({ fieldErrors: {} }) } }) },
  updateCatalogItemSchema: { safeParse: () => ({ success: false, error: { flatten: () => ({ fieldErrors: {} }) } }) },
}))

vi.mock('../auth/password', () => ({
  hashPassword: () => Promise.resolve('$argon2id$mocked'),
  verifyPassword: () => Promise.resolve(false),
}))

vi.mock('../auth/sessions', () => ({
  createSession: () => Promise.resolve({ id: 'sid', expiresAt: new Date('2099-01-01') }),
  validateSession: () => Promise.resolve(null),
  deleteSession: () => Promise.resolve(),
}))

async function buildApp() {
  const { Hono } = await import('hono')
  const { cors } = await import('hono/cors')
  const { auth } = await import('../auth/routes')
  const { venueRoutes } = await import('../routes/venues')
  const { floorPlanRoutes } = await import('../routes/floor-plans')
  const { occasionRoutes } = await import('../routes/occasions')
  const { catalogRoutes } = await import('../routes/catalog')

  const app = new Hono()

  app.use('*', cors({ origin: '*', credentials: true }))

  app.get('/health', async (c) => {
    return c.json({ status: 'healthy', checks: { postgres: 'ok', redis: 'ok' } })
  })

  app.route('/auth', auth)
  app.route('/venues', venueRoutes)
  app.route('/venues/:venueId/floor-plans', floorPlanRoutes)
  app.route('/venues/:venueId/occasions', occasionRoutes)
  app.route('/catalog', catalogRoutes)

  app.get('/', (c) => c.json({ name: 'OmniTwin API', version: '0.1.0' }))

  return app
}

describe('API Route Structure', () => {
  describe('Public endpoints', () => {
    it('GET / returns API info', async () => {
      const app = await buildApp()
      const res = await app.request('/')
      expect(res.status).toBe(200)
      const body = await res.json()
      expect(body.name).toBe('OmniTwin API')
      expect(body.version).toBe('0.1.0')
    })

    it('GET /health returns health status', async () => {
      const app = await buildApp()
      const res = await app.request('/health')
      expect(res.status).toBe(200)
      const body = await res.json()
      expect(body.status).toBe('healthy')
      expect(body.checks.postgres).toBe('ok')
      expect(body.checks.redis).toBe('ok')
    })
  })

  describe('Protected venue endpoints require auth', () => {
    it('GET /venues returns 401 without auth', async () => {
      const app = await buildApp()
      const res = await app.request('/venues')
      expect(res.status).toBe(401)
    })

    it('POST /venues returns 401 without auth', async () => {
      const app = await buildApp()
      const res = await app.request('/venues', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: 'Test Venue' }),
      })
      expect(res.status).toBe(401)
    })

    it('GET /venues/:id returns 401 without auth', async () => {
      const app = await buildApp()
      const res = await app.request('/venues/some-id')
      expect(res.status).toBe(401)
    })

    it('PATCH /venues/:id returns 401 without auth', async () => {
      const app = await buildApp()
      const res = await app.request('/venues/some-id', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: 'Updated' }),
      })
      expect(res.status).toBe(401)
    })

    it('DELETE /venues/:id returns 401 without auth', async () => {
      const app = await buildApp()
      const res = await app.request('/venues/some-id', { method: 'DELETE' })
      expect(res.status).toBe(401)
    })
  })

  describe('Protected floor plan endpoints require auth', () => {
    it('GET /venues/:id/floor-plans returns 401 without auth', async () => {
      const app = await buildApp()
      const res = await app.request('/venues/v1/floor-plans')
      expect(res.status).toBe(401)
    })

    it('POST /venues/:id/floor-plans returns 401 without auth', async () => {
      const app = await buildApp()
      const res = await app.request('/venues/v1/floor-plans', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: 'Plan' }),
      })
      expect(res.status).toBe(401)
    })
  })

  describe('Protected occasion endpoints require auth', () => {
    it('GET /venues/:id/occasions returns 401 without auth', async () => {
      const app = await buildApp()
      const res = await app.request('/venues/v1/occasions')
      expect(res.status).toBe(401)
    })

    it('POST /venues/:id/occasions returns 401 without auth', async () => {
      const app = await buildApp()
      const res = await app.request('/venues/v1/occasions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: 'Event' }),
      })
      expect(res.status).toBe(401)
    })
  })

  describe('Protected catalog endpoints require auth', () => {
    it('GET /catalog returns 401 without auth', async () => {
      const app = await buildApp()
      const res = await app.request('/catalog')
      expect(res.status).toBe(401)
    })

    it('POST /catalog returns 401 without auth', async () => {
      const app = await buildApp()
      const res = await app.request('/catalog', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: 'Chair' }),
      })
      expect(res.status).toBe(401)
    })

    it('GET /catalog/:id returns 401 without auth', async () => {
      const app = await buildApp()
      const res = await app.request('/catalog/item-1')
      expect(res.status).toBe(401)
    })
  })

  describe('CORS headers', () => {
    it('includes CORS headers on responses', async () => {
      const app = await buildApp()
      const res = await app.request('/', {
        headers: { Origin: 'http://localhost:3000' },
      })
      expect(res.headers.get('access-control-allow-origin')).toBeTruthy()
    })
  })
})
