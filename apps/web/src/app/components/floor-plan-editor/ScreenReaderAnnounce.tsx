'use client'

import { useState, useEffect, useCallback, createContext, useContext, type ReactNode } from 'react'

/**
 * Visually hidden live region for screen reader announcements.
 * Use the `useAnnounce` hook to push messages.
 */

type AnnounceFunction = (message: string) => void

const AnnounceContext = createContext<AnnounceFunction>(() => {})

export function useAnnounce(): AnnounceFunction {
  return useContext(AnnounceContext)
}

export function ScreenReaderProvider({ children }: { children: ReactNode }) {
  const [message, setMessage] = useState('')

  const announce = useCallback((msg: string) => {
    // Clear and re-set to ensure repeated identical messages are announced
    setMessage('')
    requestAnimationFrame(() => setMessage(msg))
  }, [])

  return (
    <AnnounceContext.Provider value={announce}>
      {children}
      <div
        role="status"
        aria-live="polite"
        aria-atomic="true"
        className="sr-only"
        style={{
          position: 'absolute',
          width: '1px',
          height: '1px',
          padding: 0,
          margin: '-1px',
          overflow: 'hidden',
          clip: 'rect(0, 0, 0, 0)',
          whiteSpace: 'nowrap',
          border: 0,
        }}
      >
        {message}
      </div>
    </AnnounceContext.Provider>
  )
}
