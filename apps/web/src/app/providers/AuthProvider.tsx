'use client'

import type { ReactNode } from 'react'
import { AuthContext, useAuthState } from '../hooks/useAuth'

export function AuthProvider({ children }: { children: ReactNode }) {
  const auth = useAuthState()
  return <AuthContext.Provider value={auth}>{children}</AuthContext.Provider>
}
