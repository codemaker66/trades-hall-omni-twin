'use client'

import Link from 'next/link'

export function AppHeader() {
  return (
    <header className="h-14 flex items-center justify-between px-6 border-b border-surface-25 bg-surface-5">
      <div />
      <div className="flex items-center gap-3">
        <Link
          href="/editor"
          className="inline-flex items-center gap-2 px-3 py-1.5 text-sm font-medium rounded-lg
            bg-gold-30 text-surface-5 hover:bg-gold-40 transition-colors"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M2.036 12.322a1.012 1.012 0 010-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178z" />
            <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
          Open 3D Editor
        </Link>
      </div>
    </header>
  )
}
