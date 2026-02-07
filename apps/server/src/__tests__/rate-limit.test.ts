import { describe, it, expect, beforeEach } from 'vitest'
import { Hono } from 'hono'
import { rateLimit } from '../lib/rate-limit'

describe('Rate Limiter', () => {
  let app: Hono

  beforeEach(() => {
    app = new Hono()
  })

  it('allows requests under the limit', async () => {
    app.get('/test', rateLimit({ windowMs: 60_000, max: 3 }), (c) => c.text('ok'))

    for (let i = 0; i < 3; i++) {
      const res = await app.request('/test', {
        headers: { 'x-forwarded-for': 'test-under-limit' },
      })
      expect(res.status).toBe(200)
    }
  })

  it('returns 429 when limit is exceeded', async () => {
    app.get('/test', rateLimit({ windowMs: 60_000, max: 2 }), (c) => c.text('ok'))

    const ip = `test-exceed-${Date.now()}`
    // First 2 pass
    await app.request('/test', { headers: { 'x-forwarded-for': ip } })
    await app.request('/test', { headers: { 'x-forwarded-for': ip } })

    // Third should be rate limited
    const res = await app.request('/test', { headers: { 'x-forwarded-for': ip } })
    expect(res.status).toBe(429)
    const body = await res.json()
    expect(body.error).toMatch(/too many requests/i)
  })

  it('includes Retry-After header on 429', async () => {
    app.get('/test', rateLimit({ windowMs: 60_000, max: 1 }), (c) => c.text('ok'))

    const ip = `test-retry-${Date.now()}`
    await app.request('/test', { headers: { 'x-forwarded-for': ip } })

    const res = await app.request('/test', { headers: { 'x-forwarded-for': ip } })
    expect(res.status).toBe(429)
    expect(res.headers.get('Retry-After')).toBeTruthy()
    expect(Number(res.headers.get('Retry-After'))).toBeGreaterThan(0)
  })

  it('tracks different IPs independently', async () => {
    app.get('/test', rateLimit({ windowMs: 60_000, max: 1 }), (c) => c.text('ok'))

    const res1 = await app.request('/test', {
      headers: { 'x-forwarded-for': `ip-a-${Date.now()}` },
    })
    const res2 = await app.request('/test', {
      headers: { 'x-forwarded-for': `ip-b-${Date.now()}` },
    })

    expect(res1.status).toBe(200)
    expect(res2.status).toBe(200)
  })

  it('supports custom key function', async () => {
    app.get(
      '/test',
      rateLimit({
        windowMs: 60_000,
        max: 1,
        keyFn: (c) => c.req.header('x-api-key') ?? 'anon',
      }),
      (c) => c.text('ok'),
    )

    const key = `key-${Date.now()}`
    const res1 = await app.request('/test', { headers: { 'x-api-key': key } })
    const res2 = await app.request('/test', { headers: { 'x-api-key': key } })

    expect(res1.status).toBe(200)
    expect(res2.status).toBe(429)
  })

  it('resets after window expires', async () => {
    app.get('/test', rateLimit({ windowMs: 50, max: 1 }), (c) => c.text('ok'))

    const ip = `test-reset-${Date.now()}`
    const res1 = await app.request('/test', { headers: { 'x-forwarded-for': ip } })
    expect(res1.status).toBe(200)

    // Wait for window to expire
    await new Promise((r) => setTimeout(r, 80))

    const res2 = await app.request('/test', { headers: { 'x-forwarded-for': ip } })
    expect(res2.status).toBe(200)
  })
})
