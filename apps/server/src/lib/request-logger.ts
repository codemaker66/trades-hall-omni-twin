/**
 * Structured request logging middleware.
 *
 * Emits one JSON line per request to stdout with:
 * - timestamp, HTTP method, path, response status, duration in ms
 *
 * Zero external dependencies. Compatible with any log aggregator
 * that reads JSON from stdout (CloudWatch, Datadog, ELK, etc.).
 */

import type { Context, Next } from 'hono'

export function requestLogger() {
  return async (c: Context, next: Next): Promise<void> => {
    const start = performance.now()
    await next()
    const ms = (performance.now() - start).toFixed(1)

    const entry = {
      ts: new Date().toISOString(),
      method: c.req.method,
      path: c.req.path,
      status: c.res.status,
      ms: Number(ms),
    }

    // Use stdout for structured logs (not console.log which goes to stderr in some runtimes)
    process.stdout.write(JSON.stringify(entry) + '\n')
  }
}
