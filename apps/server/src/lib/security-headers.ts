/**
 * Security headers middleware.
 *
 * Sets standard security headers on every response:
 * - X-Content-Type-Options: prevents MIME-type sniffing
 * - X-Frame-Options: prevents clickjacking
 * - Referrer-Policy: limits referrer leakage
 * - Permissions-Policy: disables browser features not needed
 * - HSTS: enforces HTTPS in production
 */

import type { Context, Next } from 'hono'
import { env } from './env'

export function securityHeaders() {
  return async (c: Context, next: Next): Promise<void> => {
    await next()
    c.header('X-Content-Type-Options', 'nosniff')
    c.header('X-Frame-Options', 'DENY')
    c.header('X-XSS-Protection', '0')
    c.header('Referrer-Policy', 'strict-origin-when-cross-origin')
    c.header('Permissions-Policy', 'camera=(), microphone=(), geolocation=()')
    if (env.NODE_ENV === 'production') {
      c.header('Strict-Transport-Security', 'max-age=31536000; includeSubDomains')
    }
  }
}
