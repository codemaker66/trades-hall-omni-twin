'use client'

import type { SaveStatus } from './useAutoSave'

const config: Record<SaveStatus, { label: string; dot: string }> = {
  idle: { label: '', dot: '' },
  saving: { label: 'Saving...', dot: 'bg-warning-50 animate-pulse' },
  saved: { label: 'Saved', dot: 'bg-success-50' },
  error: { label: 'Save failed', dot: 'bg-danger-50' },
}

export function SaveIndicator({ status }: { status: SaveStatus }) {
  if (status === 'idle') return null

  const { label, dot } = config[status]

  return (
    <div className="flex items-center gap-1.5 text-xs text-surface-60" aria-live="polite">
      <div className={`w-1.5 h-1.5 rounded-full ${dot}`} />
      <span>{label}</span>
    </div>
  )
}
