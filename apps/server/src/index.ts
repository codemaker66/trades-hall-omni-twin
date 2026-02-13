import { serve } from '@hono/node-server'
import { Hono } from 'hono'
import type { ContentfulStatusCode } from 'hono/utils/http-status'
import { cors } from 'hono/cors'
import { queryClient } from '@omni-twin/db'
import { redisHealthCheck } from '@omni-twin/cache'
import { env } from './lib/env'
import { securityHeaders } from './lib/security-headers'
import { requestLogger } from './lib/request-logger'
import { rateLimit } from './lib/rate-limit'
import { auth } from './auth/routes'
import { venueRoutes } from './routes/venues'
import { floorPlanRoutes } from './routes/floor-plans'
import { occasionRoutes } from './routes/occasions'
import { catalogRoutes } from './routes/catalog'

const app = new Hono()

// ---------------------------------------------------------------------------
// Global error handling
// ---------------------------------------------------------------------------

app.onError((err, c) => {
  const status = 'status' in err ? (err as { status: number }).status : 500
  if (status >= 500) {
    process.stderr.write(JSON.stringify({
      ts: new Date().toISOString(),
      level: 'error',
      method: c.req.method,
      path: c.req.path,
      error: err.message,
      stack: env.NODE_ENV !== 'production' ? err.stack : undefined,
    }) + '\n')
  }
  return c.json(
    { error: status >= 500 ? 'Internal server error.' : err.message },
    { status: status as ContentfulStatusCode },
  )
})

app.notFound((c) => c.json({ error: 'Not found.' }, 404))

// ---------------------------------------------------------------------------
// Middleware stack (order matters)
// ---------------------------------------------------------------------------

// 1. Request logging (first so it captures total duration)
app.use('*', requestLogger())

// 2. CORS
app.use(
  '*',
  cors({
    origin: env.CORS_ORIGINS,
    credentials: true,
    allowMethods: ['GET', 'POST', 'PATCH', 'DELETE', 'OPTIONS'],
    allowHeaders: ['Content-Type', 'Authorization'],
    maxAge: 86400,
  }),
)

// 3. Security headers
app.use('*', securityHeaders())

// 4. Rate limiting per route group
app.use('/venues/*', rateLimit({ windowMs: 60_000, max: 100 }))
app.use('/catalog/*', rateLimit({ windowMs: 60_000, max: 100 }))

// ---------------------------------------------------------------------------
// Health check (unauthenticated, unthrottled)
// ---------------------------------------------------------------------------

app.get('/health', async (c) => {
  const checks: Record<string, string> = {}

  try {
    await queryClient`SELECT 1`
    checks['postgres'] = 'ok'
  } catch {
    checks['postgres'] = 'error'
  }

  try {
    const ok = await redisHealthCheck()
    checks['redis'] = ok ? 'ok' : 'error'
  } catch {
    checks['redis'] = 'error'
  }

  const healthy = Object.values(checks).every((v) => v === 'ok')
  return c.json({ status: healthy ? 'healthy' : 'degraded', checks }, healthy ? 200 : 503)
})

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

app.route('/auth', auth)
app.route('/venues', venueRoutes)
app.route('/venues/:venueId/floor-plans', floorPlanRoutes)
app.route('/venues/:venueId/occasions', occasionRoutes)
app.route('/catalog', catalogRoutes)

app.get('/', (c) => c.json({ name: 'OmniTwin API', version: '0.1.0' }))

// ---------------------------------------------------------------------------
// Server start + graceful shutdown
// ---------------------------------------------------------------------------

const server = serve({ fetch: app.fetch, port: env.PORT }, (info) => {
  process.stdout.write(JSON.stringify({
    ts: new Date().toISOString(),
    event: 'server_started',
    port: info.port,
    env: env.NODE_ENV,
  }) + '\n')
})

function shutdown(signal: string) {
  process.stdout.write(JSON.stringify({
    ts: new Date().toISOString(),
    event: 'shutdown',
    signal,
  }) + '\n')

  server.close(() => process.exit(0))
  setTimeout(() => process.exit(1), 10_000).unref()
}

process.on('SIGTERM', () => shutdown('SIGTERM'))
process.on('SIGINT', () => shutdown('SIGINT'))
