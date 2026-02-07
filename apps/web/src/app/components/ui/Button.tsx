'use client'

import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from 'react'

type ButtonVariant = 'primary' | 'secondary' | 'danger' | 'ghost'
type ButtonSize = 'sm' | 'md' | 'lg'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
    variant?: ButtonVariant
    size?: ButtonSize
    leftIcon?: ReactNode
    rightIcon?: ReactNode
    /** Keyboard shortcut hint shown as <kbd> tag */
    shortcut?: string
    loading?: boolean
}

const variantClasses: Record<ButtonVariant, { base: string; hover: string; disabled: string }> = {
    primary: {
        base: 'bg-surface-25 border-gold-20 text-gold-50',
        hover: 'hover:bg-surface-30 hover:border-gold-50',
        disabled: 'bg-surface-10/70 border-surface-20 text-surface-60 cursor-not-allowed',
    },
    secondary: {
        base: 'bg-surface-25 border-surface-70 text-surface-90',
        hover: 'hover:bg-surface-30',
        disabled: 'bg-surface-10/70 border-surface-20 text-surface-60 cursor-not-allowed',
    },
    danger: {
        base: 'bg-surface-25 border-danger-50 text-danger-70',
        hover: 'hover:bg-surface-30',
        disabled: 'bg-surface-10/70 border-surface-20 text-surface-60 cursor-not-allowed',
    },
    ghost: {
        base: 'bg-surface-10/90 border-surface-25 text-surface-60',
        hover: 'hover:bg-surface-20 hover:text-surface-80',
        disabled: 'bg-surface-10/50 border-surface-15 text-surface-40 cursor-not-allowed',
    },
}

const sizeClasses: Record<ButtonSize, string> = {
    sm: 'px-2 py-1 text-xs gap-1',
    md: 'px-4 py-2 text-sm gap-1.5',
    lg: 'px-6 py-3 text-base gap-2',
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
    ({ variant = 'secondary', size = 'md', leftIcon, rightIcon, shortcut, loading, disabled, className = '', children, ...props }, ref) => {
        const v = variantClasses[variant]
        const isDisabled = disabled || loading

        return (
            <button
                ref={ref}
                disabled={isDisabled}
                aria-disabled={isDisabled || undefined}
                className={`
                    inline-flex items-center justify-center font-semibold
                    border-2 rounded-sm shadow-md
                    transition-all active:scale-95
                    focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-50
                    ${sizeClasses[size]}
                    ${isDisabled ? v.disabled : `${v.base} ${v.hover}`}
                    ${className}
                `.trim()}
                {...props}
            >
                {loading && <Spinner />}
                {!loading && leftIcon}
                {children}
                {!loading && rightIcon}
                {shortcut && (
                    <kbd className="ml-1.5 text-[0.65rem] opacity-50 font-mono bg-black/20 px-1 rounded">
                        {shortcut}
                    </kbd>
                )}
            </button>
        )
    }
)

Button.displayName = 'Button'

const Spinner = () => (
    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
)
