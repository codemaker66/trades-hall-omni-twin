import { describe, it, expect, vi, beforeEach } from 'vitest'
import { Hono } from 'hono'

/**
 * Tests for auth input validation and route-level guards.
 *
 * These test the validation logic in auth routes WITHOUT hitting the database.
 * DB-dependent operations (user lookup, session creation) are mocked.
 */

// Mock the DB module before any imports that use it
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
  venuePermissions: { venueId: 'venueId', userId: 'userId', role: 'role' },
  queryClient: () => Promise.resolve([{ '?column?': 1 }]),
}))

vi.mock('@omni-twin/cache', () => ({
  redisHealthCheck: () => Promise.resolve(true),
}))

vi.mock('../auth/password', () => ({
  hashPassword: () => Promise.resolve('$argon2id$mocked-hash'),
  verifyPassword: () => Promise.resolve(false),
}))

vi.mock('../auth/sessions', () => ({
  createSession: () => Promise.resolve({ id: 'mock-session-id', expiresAt: new Date('2099-01-01') }),
  validateSession: () => Promise.resolve(null),
  deleteSession: () => Promise.resolve(),
}))

describe('Auth Input Validation', () => {
  let app: Hono

  beforeEach(async () => {
    // Dynamic import to pick up mocks
    const { auth } = await import('../auth/routes')
    app = new Hono()
    app.route('/auth', auth)
  })

  describe('POST /auth/register', () => {
    it('rejects missing email', async () => {
      const res = await app.request('/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: 'Test', password: 'password123' }),
      })
      expect(res.status).toBe(400)
      const body = await res.json()
      expect(body.error).toMatch(/email/i)
    })

    it('rejects invalid email format', async () => {
      const res = await app.request('/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: 'not-an-email', name: 'Test', password: 'password123' }),
      })
      expect(res.status).toBe(400)
      const body = await res.json()
      expect(body.field).toBe('email')
    })

    it('rejects missing name', async () => {
      const res = await app.request('/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: 'test@example.com', password: 'password123' }),
      })
      expect(res.status).toBe(400)
      const body = await res.json()
      expect(body.error).toMatch(/name/i)
    })

    it('rejects empty name', async () => {
      const res = await app.request('/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: 'test@example.com', name: '   ', password: 'password123' }),
      })
      expect(res.status).toBe(400)
    })

    it('rejects password under 8 characters', async () => {
      const res = await app.request('/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: 'test@example.com', name: 'Test', password: 'short' }),
      })
      expect(res.status).toBe(400)
      const body = await res.json()
      expect(body.error).toMatch(/password/i)
    })

    it('rejects missing password', async () => {
      const res = await app.request('/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: 'test@example.com', name: 'Test' }),
      })
      expect(res.status).toBe(400)
    })
  })

  describe('POST /auth/login', () => {
    it('rejects missing email', async () => {
      const res = await app.request('/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password: 'password123' }),
      })
      expect(res.status).toBe(400)
    })

    it('rejects missing password', async () => {
      const res = await app.request('/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: 'test@example.com' }),
      })
      expect(res.status).toBe(400)
    })

    it('returns 401 for non-existent user', async () => {
      const res = await app.request('/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: 'noone@example.com', password: 'password123' }),
      })
      expect(res.status).toBe(401)
      const body = await res.json()
      expect(body.error).toMatch(/invalid/i)
    })
  })

  describe('POST /auth/logout', () => {
    it('returns 401 without session', async () => {
      const res = await app.request('/auth/logout', { method: 'POST' })
      expect(res.status).toBe(401)
    })
  })

  describe('GET /auth/me', () => {
    it('returns 401 without session', async () => {
      const res = await app.request('/auth/me')
      expect(res.status).toBe(401)
    })

    it('returns 401 with invalid bearer token', async () => {
      const res = await app.request('/auth/me', {
        headers: { Authorization: 'Bearer invalid-token' },
      })
      expect(res.status).toBe(401)
    })
  })
})
