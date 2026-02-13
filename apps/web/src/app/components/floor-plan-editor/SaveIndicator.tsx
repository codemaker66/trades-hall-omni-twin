'use client'

import type { SaveStatus } from './useAutoSave'

const config: Record<SaveStatus, { label: string; dot: string }> = {
  idle: { label: '', dot: '' },
  saving: { label: 'Saving...', dot: 'bg-warning-50 animate-pulse' },
  saved: { label: 'Saved', dot: 'bg-success-50' },
  error: { label: 'Save failed', dot: 'bg-danger-50' },
}

const serverConfig: Record<SaveStatus, { label: string; dot: string }> = {
  idle: { label: '', dot: '' },
  saving: { label: 'Syncing...', dot: 'bg-indigo-50 animate-pulse' },
  saved: { label: 'Synced', dot: 'bg-success-50' },
  error: { label: 'Sync failed', dot: 'bg-danger-50' },
}

interface SaveIndicatorProps {
  status: SaveStatus
  serverStatus?: SaveStatus
}

export function SaveIndicator({ status, serverStatus }: SaveIndicatorProps) {
  const showLocal = status !== 'idle'
  const showServer = serverStatus && serverStatus !== 'idle'

  if (!showLocal && !showServer) return null

  return (
    <div className="flex items-center gap-3 text-xs text-surface-60" aria-live="polite">
      {showLocal && (
        <div className="flex items-center gap-1.5">
          <div className={`w-1.5 h-1.5 rounded-full ${config[status].dot}`} />
          <span>{config[status].label}</span>
        </div>
      )}
      {showServer && (
        <div className="flex items-center gap-1.5">
          <div className={`w-1.5 h-1.5 rounded-full ${serverConfig[serverStatus].dot}`} />
          <span>{serverConfig[serverStatus].label}</span>
        </div>
      )}
    </div>
  )
}
