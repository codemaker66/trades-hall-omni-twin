'use client'

import { createContext, useContext, useState, useCallback, useEffect } from 'react'
import type { AuthUser } from '@omni-twin/types'

export interface AuthContextValue {
  user: AuthUser | null
  loading: boolean
  login: (email: string, password: string) => Promise<{ ok: boolean; error?: string }>
  register: (email: string, name: string, password: string) => Promise<{ ok: boolean; error?: string }>
  logout: () => Promise<void>
}

const API_BASE = process.env['NEXT_PUBLIC_API_URL'] ?? 'http://localhost:4000'

async function apiFetch<T>(path: string, init?: RequestInit): Promise<{ ok: boolean; data?: T; error?: string }> {
  try {
    const res = await fetch(`${API_BASE}${path}`, {
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      ...init,
    })
    const data = await res.json() as T & { error?: string }
    if (!res.ok) return { ok: false, error: data.error ?? `HTTP ${res.status}` }
    return { ok: true, data }
  } catch {
    return { ok: false, error: 'Network error. Please try again.' }
  }
}

export const AuthContext = createContext<AuthContextValue | null>(null)

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within AuthProvider')
  return ctx
}

export function useAuthState(): AuthContextValue {
  const [user, setUser] = useState<AuthUser | null>(null)
  const [loading, setLoading] = useState(true)

  // Check existing session on mount
  useEffect(() => {
    apiFetch<{ user: AuthUser }>('/auth/me')
      .then((res) => {
        if (res.ok && res.data) setUser(res.data.user)
      })
      .finally(() => setLoading(false))
  }, [])

  const login = useCallback(async (email: string, password: string) => {
    const res = await apiFetch<{ user: AuthUser }>('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    })
    if (res.ok && res.data) {
      setUser(res.data.user)
      return { ok: true }
    }
    return { ok: false, error: res.error }
  }, [])

  const register = useCallback(async (email: string, name: string, password: string) => {
    const res = await apiFetch<{ user: AuthUser }>('/auth/register', {
      method: 'POST',
      body: JSON.stringify({ email, name, password }),
    })
    if (res.ok && res.data) {
      setUser(res.data.user)
      return { ok: true }
    }
    return { ok: false, error: res.error }
  }, [])

  const logout = useCallback(async () => {
    await apiFetch('/auth/logout', { method: 'POST' })
    setUser(null)
  }, [])

  return { user, loading, login, register, logout }
}
