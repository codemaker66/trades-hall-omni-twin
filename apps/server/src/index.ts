import { serve } from '@hono/node-server'
import { Hono } from 'hono'
import { cors } from 'hono/cors'
import { queryClient } from '@omni-twin/db'
import { redisHealthCheck } from '@omni-twin/cache'
import { auth } from './auth/routes'
import { venueRoutes } from './routes/venues'
import { floorPlanRoutes } from './routes/floor-plans'
import { occasionRoutes } from './routes/occasions'
import { catalogRoutes } from './routes/catalog'

const app = new Hono()

const ALLOWED_ORIGINS = (process.env['CORS_ORIGINS'] ?? 'http://localhost:3000').split(',')

app.use(
  '*',
  cors({
    origin: ALLOWED_ORIGINS,
    credentials: true,
    allowMethods: ['GET', 'POST', 'PATCH', 'DELETE', 'OPTIONS'],
    allowHeaders: ['Content-Type', 'Authorization'],
    maxAge: 86400,
  }),
)

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

app.route('/auth', auth)
app.route('/venues', venueRoutes)
app.route('/venues/:venueId/floor-plans', floorPlanRoutes)
app.route('/venues/:venueId/occasions', occasionRoutes)
app.route('/catalog', catalogRoutes)

app.get('/', (c) => c.json({ name: 'OmniTwin API', version: '0.1.0' }))

const port = Number(process.env['PORT'] ?? 4000)

serve({ fetch: app.fetch, port }, (info) => {
  console.log(`OmniTwin API server running on http://localhost:${info.port}`)
})
