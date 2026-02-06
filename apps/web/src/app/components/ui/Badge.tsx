'use client'

import type { HTMLAttributes } from 'react'

type BadgeVariant = 'default' | 'success' | 'warning' | 'danger' | 'info' | 'gold'

interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  variant?: BadgeVariant
}

const variantClasses: Record<BadgeVariant, string> = {
  default: 'bg-surface-25 text-surface-80 border-surface-40',
  success: 'bg-success-10 text-success-50 border-success-50/30',
  warning: 'bg-surface-20 text-warning-50 border-warning-50/30',
  danger: 'bg-danger-25 text-danger-50 border-danger-50/30',
  info: 'bg-surface-15 text-info-50 border-info-50/30',
  gold: 'bg-surface-20 text-gold-50 border-gold-20',
}

export function Badge({ variant = 'default', className = '', children, ...props }: BadgeProps) {
  return (
    <span
      className={`
        inline-flex items-center px-2 py-0.5 rounded-full
        text-xs font-medium border
        ${variantClasses[variant]}
        ${className}
      `.trim()}
      {...props}
    >
      {children}
    </span>
  )
}
