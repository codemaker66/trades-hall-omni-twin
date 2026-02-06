/** Global user role (determines feature access). */
export type UserRole = 'owner' | 'manager' | 'planner' | 'viewer'

/** Venue-level permission role (determines what a user can do with a venue). */
export type VenueRole = 'owner' | 'editor' | 'viewer' | 'commenter'

/** Authenticated user (safe to send to client — no password hash). */
export interface AuthUser {
  id: string
  email: string
  name: string
  role: UserRole
  avatarUrl: string | null
  createdAt: string
}

/** Session data stored server-side. */
export interface Session {
  id: string
  userId: string
  expiresAt: string
}

/** Register request body. */
export interface RegisterInput {
  email: string
  name: string
  password: string
}

/** Login request body. */
export interface LoginInput {
  email: string
  password: string
}

/** Auth response with user + session token. */
export interface AuthResponse {
  user: AuthUser
  sessionId: string
  expiresAt: string
}

/** Auth error response. */
export interface AuthError {
  error: string
  field?: string
}
