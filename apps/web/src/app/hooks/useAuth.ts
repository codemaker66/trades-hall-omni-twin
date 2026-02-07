'use client'

import { createContext, useContext, useState, useCallback, useEffect } from 'react'
import type { AuthUser } from '@omni-twin/types'
import { apiFetch } from '../../lib/api-client'

export interface AuthContextValue {
  user: AuthUser | null
  loading: boolean
  login: (email: string, password: string) => Promise<{ ok: boolean; error?: string }>
  register: (email: string, name: string, password: string) => Promise<{ ok: boolean; error?: string }>
  logout: () => Promise<void>
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
