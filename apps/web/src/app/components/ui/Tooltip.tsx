'use client'

import { useState, useRef, useCallback, type ReactNode, type ReactElement } from 'react'

interface TooltipProps {
    /** Content shown in the tooltip */
    content: string
    /** Optional keyboard shortcut displayed after content */
    shortcut?: string
    /** Position relative to trigger */
    position?: 'top' | 'bottom' | 'left' | 'right'
    /** The trigger element (must accept onMouseEnter/Leave) */
    children: ReactElement
}

export const Tooltip = ({ content, shortcut, position = 'top', children }: TooltipProps) => {
    const [visible, setVisible] = useState(false)
    const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

    const show = useCallback(() => {
        timerRef.current = setTimeout(() => setVisible(true), 200)
    }, [])

    const hide = useCallback(() => {
        if (timerRef.current) clearTimeout(timerRef.current)
        setVisible(false)
    }, [])

    const positionClasses: Record<string, string> = {
        top: 'bottom-full left-1/2 -translate-x-1/2 mb-2',
        bottom: 'top-full left-1/2 -translate-x-1/2 mt-2',
        left: 'right-full top-1/2 -translate-y-1/2 mr-2',
        right: 'left-full top-1/2 -translate-y-1/2 ml-2',
    }

    return (
        <div
            className="relative inline-flex"
            onMouseEnter={show}
            onMouseLeave={hide}
            onFocus={show}
            onBlur={hide}
        >
            {children}
            {visible && (
                <div
                    role="tooltip"
                    className={`
                        absolute z-50 pointer-events-none
                        px-2.5 py-1.5 rounded-md
                        bg-surface-20 border border-white/10 shadow-lg
                        text-xs text-surface-90 whitespace-nowrap
                        ${positionClasses[position]}
                    `}
                >
                    {content}
                    {shortcut && (
                        <kbd className="ml-1.5 text-[0.6rem] opacity-50 font-mono bg-black/30 px-1 rounded">
                            {shortcut}
                        </kbd>
                    )}
                </div>
            )}
        </div>
    )
}
