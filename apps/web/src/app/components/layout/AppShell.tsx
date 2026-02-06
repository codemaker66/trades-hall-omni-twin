'use client'

import type { ReactNode } from 'react'
import { Sidebar } from './Sidebar'
import { AppHeader } from './AppHeader'

interface AppShellProps {
  children: ReactNode
}

export function AppShell({ children }: AppShellProps) {
  return (
    <div className="flex h-[100dvh] bg-surface-0 text-surface-90">
      <Sidebar />
      <div className="flex flex-col flex-1 min-w-0">
        <AppHeader />
        <main className="flex-1 overflow-y-auto p-6">
          {children}
        </main>
      </div>
    </div>
  )
}
