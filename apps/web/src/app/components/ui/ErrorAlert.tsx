'use client'

import { Button } from './Button'

interface ErrorAlertProps {
  message: string
  onRetry?: () => void
}

export function ErrorAlert({ message, onRetry }: ErrorAlertProps) {
  return (
    <div className="bg-danger-10 border border-danger-50 rounded-lg p-4 flex items-center justify-between gap-3">
      <p className="text-sm text-danger-70">{message}</p>
      {onRetry && (
        <Button variant="ghost" size="sm" onClick={onRetry}>
          Retry
        </Button>
      )}
    </div>
  )
}
