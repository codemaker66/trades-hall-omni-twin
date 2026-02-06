'use client'

import type { ReactNode } from 'react'

interface EmptyStateProps {
  icon?: ReactNode
  title: string
  description?: string
  action?: ReactNode
}

export function EmptyState({ icon, title, description, action }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-16 px-4 text-center">
      {icon && (
        <div className="text-surface-40 mb-4" aria-hidden="true">
          {icon}
        </div>
      )}
      <h3 className="text-lg font-semibold text-surface-80 mb-1">{title}</h3>
      {description && (
        <p className="text-sm text-surface-60 max-w-sm mb-6">{description}</p>
      )}
      {action}
    </div>
  )
}
