import type { Context, Next } from 'hono'
import { getCookie } from 'hono/cookie'
import { eq, and } from 'drizzle-orm'
import { db, venuePermissions } from '@omni-twin/db'
import { validateSession } from './sessions'
import type { AuthUser, UserRole, VenueRole } from '@omni-twin/types'

const SESSION_COOKIE = 'omnitwin_session'

/** Extract session ID from cookie or Authorization header. */
function getSessionId(c: Context): string | null {
  const cookieValue = getCookie(c, SESSION_COOKIE)
  if (cookieValue) return cookieValue

  const authHeader = c.req.header('Authorization')
  if (authHeader?.startsWith('Bearer ')) {
    return authHeader.slice(7)
  }

  return null
}

/** Middleware: require valid session. Sets c.set('user', ...) on success. */
export async function requireAuth(c: Context, next: Next): Promise<Response | void> {
  const sessionId = getSessionId(c)
  if (!sessionId) {
    return c.json({ error: 'Authentication required.' }, 401)
  }

  const user = await validateSession(sessionId)
  if (!user) {
    return c.json({ error: 'Invalid or expired session.' }, 401)
  }

  c.set('user' as never, user as never)
  c.set('sessionId' as never, sessionId as never)
  await next()
}

/**
 * Middleware factory: require the authenticated user to have one of the given global roles.
 * Must be used after requireAuth.
 */
export function requireRole(...roles: UserRole[]) {
  return async (c: Context, next: Next): Promise<Response | void> => {
    const user = getUser(c)
    if (!roles.includes(user.role)) {
      return c.json({ error: 'Insufficient permissions.' }, 403)
    }
    await next()
  }
}

/**
 * Middleware factory: require the authenticated user to have venue-level access.
 * Reads the venueId from `c.req.param(paramName)`.
 * Must be used after requireAuth.
 */
export function requireVenueAccess(paramName: string, ...roles: VenueRole[]) {
  return async (c: Context, next: Next): Promise<Response | void> => {
    const user = getUser(c)
    const venueId = c.req.param(paramName)
    if (!venueId) {
      return c.json({ error: 'Venue ID is required.' }, 400)
    }

    const result = await db
      .select({ role: venuePermissions.role })
      .from(venuePermissions)
      .where(
        and(
          eq(venuePermissions.venueId, venueId),
          eq(venuePermissions.userId, user.id),
        ),
      )
      .limit(1)

    const perm = result[0]
    if (!perm || !roles.includes(perm.role as VenueRole)) {
      return c.json({ error: 'You do not have access to this venue.' }, 403)
    }

    c.set('venueRole' as never, perm.role as never)
    await next()
  }
}

/** Get the authenticated user from context (after requireAuth). */
export function getUser(c: Context): AuthUser {
  return c.get('user' as never) as AuthUser
}

/** Get the session ID from context (after requireAuth). */
export function getSessionIdFromContext(c: Context): string {
  return c.get('sessionId' as never) as string
}

/** Get the venue role from context (after requireVenueAccess). */
export function getVenueRole(c: Context): VenueRole {
  return c.get('venueRole' as never) as VenueRole
}

export { SESSION_COOKIE }
