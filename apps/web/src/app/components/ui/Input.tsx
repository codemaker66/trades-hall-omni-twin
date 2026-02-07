'use client'

import { forwardRef, useRef, type InputHTMLAttributes } from 'react'

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
    /** Label shown above the input */
    label?: string
    /** Helper text shown below the input */
    description?: string
    /** Error message — shows red border + message */
    error?: string
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
    ({ label, description, error, id, className = '', ...props }, ref) => {
        const fallbackId = useRef(`input-${Math.random().toString(36).slice(2, 8)}`).current
        const inputId = id || fallbackId
        const descId = `${inputId}-desc`
        const errorId = `${inputId}-error`

        return (
            <div className="flex flex-col gap-1.5">
                {label && (
                    <label htmlFor={inputId} className="text-sm text-surface-80">
                        {label}
                    </label>
                )}
                <input
                    ref={ref}
                    id={inputId}
                    aria-describedby={error ? errorId : description ? descId : undefined}
                    aria-invalid={error ? true : undefined}
                    className={`
                        w-full bg-surface-5 border rounded-lg px-4 py-2
                        text-surface-90 text-sm
                        placeholder:text-surface-60
                        transition-colors
                        focus:outline-none
                        ${error
                            ? 'border-danger-50 focus:border-danger-50'
                            : 'border-white/10 focus:border-indigo-50'
                        }
                        ${className}
                    `.trim()}
                    {...props}
                />
                {error && (
                    <p id={errorId} className="text-xs text-danger-50" role="alert">
                        {error}
                    </p>
                )}
                {!error && description && (
                    <p id={descId} className="text-xs text-surface-60">
                        {description}
                    </p>
                )}
            </div>
        )
    }
)

Input.displayName = 'Input'
