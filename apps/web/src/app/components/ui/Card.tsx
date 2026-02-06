'use client'

import { forwardRef, type HTMLAttributes, type ReactNode } from 'react'

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  /** Optional header content */
  header?: ReactNode
  /** Optional footer content */
  footer?: ReactNode
  /** Remove default padding */
  noPadding?: boolean
}

export const Card = forwardRef<HTMLDivElement, CardProps>(
  ({ header, footer, noPadding, className = '', children, ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={`
          bg-surface-10 border border-surface-25 rounded-xl shadow-lg
          ${className}
        `.trim()}
        {...props}
      >
        {header && (
          <div className="px-5 py-3 border-b border-surface-25 text-sm font-semibold text-surface-90">
            {header}
          </div>
        )}
        <div className={noPadding ? '' : 'p-5'}>
          {children}
        </div>
        {footer && (
          <div className="px-5 py-3 border-t border-surface-25">
            {footer}
          </div>
        )}
      </div>
    )
  },
)

Card.displayName = 'Card'
