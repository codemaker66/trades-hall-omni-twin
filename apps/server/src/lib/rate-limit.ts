/**
 * Simple in-memory sliding-window rate limiter middleware for Hono.
 *
 * For production at scale, swap to a Redis-backed implementation
 * using @omni-twin/cache.
 */

import type { Context, Next } from 'hono'

interface RateLimitConfig {
  /** Time window in milliseconds. */
  windowMs: number
  /** Maximum requests per window per key. */
  max: number
  /** Custom key extractor. Defaults to client IP. */
  keyFn?: (c: Context) => string
}

interface Entry {
  count: number
  resetAt: number
}

const store = new Map<string, Entry>()

// Periodic cleanup every 5 minutes to prevent memory leaks
setInterval(() => {
  const now = Date.now()
  for (const [key, entry] of store) {
    if (now > entry.resetAt) store.delete(key)
  }
}, 5 * 60 * 1000).unref()

export function rateLimit(config: RateLimitConfig) {
  return async (c: Context, next: Next): Promise<Response | void> => {
    const key =
      config.keyFn?.(c) ??
      c.req.header('x-forwarded-for')?.split(',')[0]?.trim() ??
      'unknown'

    const now = Date.now()
    const entry = store.get(key)

    if (!entry || now > entry.resetAt) {
      store.set(key, { count: 1, resetAt: now + config.windowMs })
      return next()
    }

    if (entry.count >= config.max) {
      c.header('Retry-After', String(Math.ceil((entry.resetAt - now) / 1000)))
      return c.json(
        { error: 'Too many requests. Please try again later.' },
        429,
      )
    }

    entry.count++
    return next()
  }
}
