import { randomBytes } from 'node:crypto'
import { db, sessions, users } from '@omni-twin/db'
import { eq, and, gt } from 'drizzle-orm'
import type { AuthUser } from '@omni-twin/types'
import { env } from '../lib/env'

const SESSION_TTL_MS = env.SESSION_TTL_DAYS * 86_400_000 // default 7 days

function generateSessionId(): string {
  return randomBytes(32).toString('hex')
}

export async function createSession(userId: string): Promise<{ id: string; expiresAt: Date }> {
  const id = generateSessionId()
  const expiresAt = new Date(Date.now() + SESSION_TTL_MS)

  await db.insert(sessions).values({ id, userId, expiresAt })
  return { id, expiresAt }
}

export async function validateSession(sessionId: string): Promise<AuthUser | null> {
  const now = new Date()

  const result = await db
    .select({
      userId: sessions.userId,
      sessionExpiresAt: sessions.expiresAt,
      userEmail: users.email,
      userName: users.name,
      userRole: users.role,
      userAvatarUrl: users.avatarUrl,
      userCreatedAt: users.createdAt,
    })
    .from(sessions)
    .innerJoin(users, eq(sessions.userId, users.id))
    .where(and(eq(sessions.id, sessionId), gt(sessions.expiresAt, now)))
    .limit(1)

  const row = result[0]
  if (!row) return null

  return {
    id: row.userId,
    email: row.userEmail,
    name: row.userName,
    role: row.userRole as AuthUser['role'],
    avatarUrl: row.userAvatarUrl,
    createdAt: row.userCreatedAt.toISOString(),
  }
}

export async function deleteSession(sessionId: string): Promise<void> {
  await db.delete(sessions).where(eq(sessions.id, sessionId))
}

export async function deleteUserSessions(userId: string): Promise<void> {
  await db.delete(sessions).where(eq(sessions.userId, userId))
}
