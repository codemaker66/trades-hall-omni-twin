'use client'

import { useEffect } from 'react'

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  useEffect(() => {
    console.error('Unhandled error:', error)
  }, [error])

  return (
    <div className="flex min-h-screen items-center justify-center bg-slate-950 px-4">
      <div className="w-full max-w-md space-y-6 text-center">
        <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-red-500/10">
          <svg className="h-8 w-8 text-red-400" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z" />
          </svg>
        </div>
        <h1 className="text-2xl font-semibold text-slate-100">Something went wrong</h1>
        <p className="text-sm text-slate-400">
          An unexpected error occurred. Please try again or return to the dashboard.
        </p>
        {error.digest && (
          <p className="text-xs text-slate-600 font-mono">Error ID: {error.digest}</p>
        )}
        <div className="flex items-center justify-center gap-3">
          <button
            onClick={reset}
            className="rounded-lg bg-slate-800 px-4 py-2 text-sm font-medium text-slate-200 transition-colors hover:bg-slate-700"
          >
            Try again
          </button>
          <a
            href="/"
            className="rounded-lg bg-slate-800 px-4 py-2 text-sm font-medium text-slate-400 transition-colors hover:bg-slate-700 hover:text-slate-200"
          >
            Go to dashboard
          </a>
        </div>
      </div>
    </div>
  )
}
