import { Hono } from 'hono'
import { setCookie, deleteCookie } from 'hono/cookie'
import { db, users } from '@omni-twin/db'
import { eq } from 'drizzle-orm'
import { hashPassword, verifyPassword } from './password'
import { createSession, deleteSession } from './sessions'
import { requireAuth, getUser, getSessionIdFromContext, SESSION_COOKIE } from './middleware'
import { rateLimit } from '../lib/rate-limit'
import type { RegisterInput, LoginInput, AuthResponse, AuthError } from '@omni-twin/types'

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/

const authLimiter = rateLimit({ windowMs: 15 * 60 * 1000, max: 10 })

const auth = new Hono()

/** POST /auth/register */
auth.post('/register', authLimiter, async (c) => {
  const body = await c.req.json<RegisterInput>()

  if (!body.email || !EMAIL_RE.test(body.email)) {
    return c.json<AuthError>({ error: 'Invalid email address.', field: 'email' }, 400)
  }
  if (!body.name || body.name.trim().length === 0) {
    return c.json<AuthError>({ error: 'Name is required.', field: 'name' }, 400)
  }
  if (!body.password || body.password.length < 8) {
    return c.json<AuthError>({ error: 'Password must be at least 8 characters.', field: 'password' }, 400)
  }

  const email = body.email.toLowerCase().trim()

  // Check for existing user
  const existing = await db.select({ id: users.id }).from(users).where(eq(users.email, email)).limit(1)
  if (existing.length > 0) {
    return c.json<AuthError>({ error: 'An account with this email already exists.', field: 'email' }, 409)
  }

  const passwordHash = await hashPassword(body.password)

  const [user] = await db
    .insert(users)
    .values({
      email,
      name: body.name.trim(),
      passwordHash,
    })
    .returning({
      id: users.id,
      email: users.email,
      name: users.name,
      role: users.role,
      avatarUrl: users.avatarUrl,
      createdAt: users.createdAt,
    })

  if (!user) {
    return c.json<AuthError>({ error: 'Failed to create account.' }, 500)
  }

  const session = await createSession(user.id)

  setCookie(c, SESSION_COOKIE, session.id, {
    httpOnly: true,
    secure: process.env['NODE_ENV'] === 'production',
    sameSite: 'Lax',
    path: '/',
    expires: session.expiresAt,
  })

  return c.json<AuthResponse>({
    user: {
      id: user.id,
      email: user.email,
      name: user.name,
      role: user.role,
      avatarUrl: user.avatarUrl,
      createdAt: user.createdAt.toISOString(),
    },
    sessionId: session.id,
    expiresAt: session.expiresAt.toISOString(),
  }, 201)
})

/** POST /auth/login */
auth.post('/login', authLimiter, async (c) => {
  const body = await c.req.json<LoginInput>()

  if (!body.email || !body.password) {
    return c.json<AuthError>({ error: 'Email and password are required.' }, 400)
  }

  const email = body.email.toLowerCase().trim()

  const result = await db
    .select({
      id: users.id,
      email: users.email,
      name: users.name,
      role: users.role,
      avatarUrl: users.avatarUrl,
      passwordHash: users.passwordHash,
      createdAt: users.createdAt,
    })
    .from(users)
    .where(eq(users.email, email))
    .limit(1)

  const user = result[0]
  if (!user) {
    return c.json<AuthError>({ error: 'Invalid email or password.' }, 401)
  }

  const valid = await verifyPassword(user.passwordHash, body.password)
  if (!valid) {
    return c.json<AuthError>({ error: 'Invalid email or password.' }, 401)
  }

  const session = await createSession(user.id)

  setCookie(c, SESSION_COOKIE, session.id, {
    httpOnly: true,
    secure: process.env['NODE_ENV'] === 'production',
    sameSite: 'Lax',
    path: '/',
    expires: session.expiresAt,
  })

  return c.json<AuthResponse>({
    user: {
      id: user.id,
      email: user.email,
      name: user.name,
      role: user.role,
      avatarUrl: user.avatarUrl,
      createdAt: user.createdAt.toISOString(),
    },
    sessionId: session.id,
    expiresAt: session.expiresAt.toISOString(),
  })
})

/** POST /auth/logout */
auth.post('/logout', requireAuth, async (c) => {
  const sessionId = getSessionIdFromContext(c)
  await deleteSession(sessionId)
  deleteCookie(c, SESSION_COOKIE, { path: '/' })
  return c.json({ ok: true })
})

/** GET /auth/me */
auth.get('/me', requireAuth, async (c) => {
  const user = getUser(c)
  return c.json({ user })
})

export { auth }
