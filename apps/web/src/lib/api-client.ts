/**
 * Shared API client for all frontend → backend communication.
 *
 * Centralises base URL, credentials, Content-Type, and global error handling
 * so individual hooks / pages don't duplicate fetch boilerplate.
 */

const API_BASE = process.env['NEXT_PUBLIC_API_URL'] ?? 'http://localhost:4000'

export interface ApiResult<T> {
  ok: boolean
  data?: T
  error?: string
  fieldErrors?: Record<string, string[]>
  status?: number
}

/**
 * Typed fetch wrapper that sends credentialed JSON requests to the API server.
 *
 * - Returns `{ ok: true, data }` on 2xx.
 * - Returns `{ ok: false, error, fieldErrors?, status }` on 4xx/5xx.
 * - Redirects to `/login` on 401 (expired session).
 */
export async function apiFetch<T>(
  path: string,
  init?: RequestInit,
): Promise<ApiResult<T>> {
  try {
    const res = await fetch(`${API_BASE}${path}`, {
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      ...init,
    })

    // Session expired → redirect to login
    if (res.status === 401 && typeof window !== 'undefined') {
      // Don't redirect if we're already checking auth (e.g. /auth/me)
      if (!path.startsWith('/auth/')) {
        window.location.href = `/login?redirect=${encodeURIComponent(window.location.pathname)}`
        return { ok: false, error: 'Session expired.', status: 401 }
      }
    }

    if (res.status === 204) return { ok: true, status: 204 }

    const data = (await res.json()) as T & {
      error?: string
      fields?: Record<string, string[]>
    }

    if (!res.ok) {
      return {
        ok: false,
        error: data.error ?? `HTTP ${res.status}`,
        fieldErrors: data.fields,
        status: res.status,
      }
    }

    return { ok: true, data, status: res.status }
  } catch {
    return { ok: false, error: 'Network error. Please try again.' }
  }
}
