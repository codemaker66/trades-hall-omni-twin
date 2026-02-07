'use client'

import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from 'react'

type IconButtonVariant = 'primary' | 'secondary' | 'danger' | 'ghost'

interface IconButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
    'aria-label': string
    variant?: IconButtonVariant
    icon: ReactNode
    /** Whether this is a toggle button in pressed state */
    pressed?: boolean
}

const variantClasses: Record<IconButtonVariant, { base: string; hover: string; active: string; disabled: string }> = {
    primary: {
        base: 'bg-surface-25 border-gold-20 text-gold-50',
        hover: 'hover:bg-surface-30 hover:border-gold-50',
        active: 'bg-surface-30 border-gold-50 text-gold-70',
        disabled: 'bg-surface-10/70 border-surface-20 text-surface-60 cursor-not-allowed',
    },
    secondary: {
        base: 'bg-surface-25 border-surface-70 text-surface-90',
        hover: 'hover:bg-surface-30',
        active: 'bg-surface-30 border-surface-90 text-white',
        disabled: 'bg-surface-10/70 border-surface-20 text-surface-60 cursor-not-allowed',
    },
    danger: {
        base: 'bg-surface-25 border-danger-50 text-danger-70',
        hover: 'hover:bg-surface-30',
        active: 'bg-danger-10 border-danger-50 text-danger-70',
        disabled: 'bg-surface-10/70 border-surface-20 text-surface-60 cursor-not-allowed',
    },
    ghost: {
        base: 'bg-surface-10/90 border-surface-25 text-surface-60',
        hover: 'hover:bg-surface-20 hover:text-surface-80',
        active: 'bg-surface-20 border-surface-40 text-surface-90',
        disabled: 'bg-surface-10/50 border-surface-15 text-surface-40 cursor-not-allowed',
    },
}

export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(
    ({ variant = 'secondary', icon, pressed, disabled, className = '', ...props }, ref) => {
        const v = variantClasses[variant]
        const isActive = pressed && !disabled

        return (
            <button
                ref={ref}
                disabled={disabled}
                aria-disabled={disabled || undefined}
                aria-pressed={pressed}
                className={`
                    inline-flex items-center justify-center
                    w-9 h-9 border-2 rounded-sm shadow-md
                    transition-all active:scale-95
                    focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-50
                    ${disabled ? v.disabled : isActive ? v.active : `${v.base} ${v.hover}`}
                    ${className}
                `.trim()}
                {...props}
            >
                {icon}
            </button>
        )
    }
)

IconButton.displayName = 'IconButton'
