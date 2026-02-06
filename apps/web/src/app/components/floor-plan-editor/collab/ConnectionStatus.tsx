'use client'

import type { ConnectionStatus as Status } from './useCollaboration'
import type { UserPresence } from './awareness'

interface ConnectionStatusProps {
  status: Status
  userCount: number
  remoteUsers: Map<number, UserPresence>
}

const statusConfig: Record<Status, { label: string; dot: string }> = {
  connected: { label: 'Live', dot: 'bg-success-50' },
  connecting: { label: 'Connecting...', dot: 'bg-warning-50 animate-pulse' },
  disconnected: { label: 'Offline', dot: 'bg-surface-50' },
}

export function ConnectionStatusIndicator({ status, userCount, remoteUsers }: ConnectionStatusProps) {
  const { label, dot } = statusConfig[status]

  return (
    <div className="flex items-center gap-2 px-2 text-xs">
      {/* Status dot + label */}
      <div className="flex items-center gap-1.5">
        <div className={`w-2 h-2 rounded-full ${dot}`} />
        <span className="text-surface-60">{label}</span>
      </div>

      {/* User avatars */}
      {userCount > 1 && (
        <div className="flex items-center gap-1">
          <div className="flex -space-x-1.5">
            {Array.from(remoteUsers.entries()).slice(0, 5).map(([clientId, user]) => (
              <div
                key={clientId}
                className="w-5 h-5 rounded-full flex items-center justify-center text-[9px] font-bold text-white border border-surface-10"
                style={{ backgroundColor: user.color }}
                title={user.name}
              >
                {user.name.charAt(0).toUpperCase()}
              </div>
            ))}
          </div>
          <span className="text-surface-60">{userCount}</span>
        </div>
      )}
    </div>
  )
}
