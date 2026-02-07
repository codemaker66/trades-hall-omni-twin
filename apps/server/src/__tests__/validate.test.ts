import { describe, it, expect } from 'vitest'
import { Hono } from 'hono'
import { z } from 'zod'
import { parseBody, isResponse } from '../lib/validate'

describe('Validation Helper', () => {
  const schema = z.object({
    name: z.string().min(1),
    email: z.string().email(),
    age: z.number().int().positive().optional(),
  })

  describe('parseBody', () => {
    it('parses valid JSON body', async () => {
      const app = new Hono()
      app.post('/test', async (c) => {
        const result = await parseBody(c, schema)
        if (isResponse(result)) return result
        return c.json({ parsed: result })
      })

      const res = await app.request('/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: 'Test', email: 'test@example.com' }),
      })

      expect(res.status).toBe(200)
      const body = await res.json()
      expect(body.parsed.name).toBe('Test')
      expect(body.parsed.email).toBe('test@example.com')
    })

    it('returns 400 for invalid JSON', async () => {
      const app = new Hono()
      app.post('/test', async (c) => {
        const result = await parseBody(c, schema)
        if (isResponse(result)) return result
        return c.json({ parsed: result })
      })

      const res = await app.request('/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: 'not json',
      })

      expect(res.status).toBe(400)
      const body = await res.json()
      expect(body.error).toBe('Invalid JSON body.')
    })

    it('returns 400 with field errors for schema violations', async () => {
      const app = new Hono()
      app.post('/test', async (c) => {
        const result = await parseBody(c, schema)
        if (isResponse(result)) return result
        return c.json({ parsed: result })
      })

      const res = await app.request('/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: '', email: 'not-an-email' }),
      })

      expect(res.status).toBe(400)
      const body = await res.json()
      expect(body.error).toBe('Validation failed.')
      expect(body.fields).toBeDefined()
      expect(body.fields.name).toBeDefined()
      expect(body.fields.email).toBeDefined()
    })

    it('accepts optional fields when missing', async () => {
      const app = new Hono()
      app.post('/test', async (c) => {
        const result = await parseBody(c, schema)
        if (isResponse(result)) return result
        return c.json({ parsed: result })
      })

      const res = await app.request('/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: 'Test', email: 'a@b.com' }),
      })

      expect(res.status).toBe(200)
      const body = await res.json()
      expect(body.parsed.age).toBeUndefined()
    })

    it('rejects wrong types', async () => {
      const app = new Hono()
      app.post('/test', async (c) => {
        const result = await parseBody(c, schema)
        if (isResponse(result)) return result
        return c.json({ parsed: result })
      })

      const res = await app.request('/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: 123, email: 'a@b.com' }),
      })

      expect(res.status).toBe(400)
    })
  })

  describe('isResponse', () => {
    it('returns true for Response instances', () => {
      expect(isResponse(new Response())).toBe(true)
    })

    it('returns false for plain objects', () => {
      expect(isResponse({ name: 'test' })).toBe(false)
      expect(isResponse(null)).toBe(false)
      expect(isResponse(42)).toBe(false)
    })
  })
})
