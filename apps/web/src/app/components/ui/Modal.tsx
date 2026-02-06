'use client'

import { useEffect, useRef, useCallback, useId, type ReactNode } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

interface ModalProps {
    open: boolean
    onClose: () => void
    /** Title displayed in the modal header */
    title: string
    children: ReactNode
    /** Width class override (default: w-80) */
    width?: string
}

export const Modal = ({ open, onClose, title, children, width = 'w-80' }: ModalProps) => {
    const titleId = useId()
    const contentRef = useRef<HTMLDivElement>(null)

    // Focus trap: cycle focus within modal
    const handleKeyDown = useCallback((e: KeyboardEvent) => {
        if (e.key === 'Escape') {
            e.preventDefault()
            onClose()
            return
        }

        if (e.key === 'Tab' && contentRef.current) {
            const focusable = contentRef.current.querySelectorAll<HTMLElement>(
                'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
            )
            if (focusable.length === 0) return

            const first = focusable[0]
            const last = focusable[focusable.length - 1]
            if (!first || !last) return

            if (e.shiftKey && document.activeElement === first) {
                e.preventDefault()
                last.focus()
            } else if (!e.shiftKey && document.activeElement === last) {
                e.preventDefault()
                first.focus()
            }
        }
    }, [onClose])

    // Auto-focus first focusable element on open
    useEffect(() => {
        if (!open) return

        const handler = handleKeyDown
        document.addEventListener('keydown', handler)

        // Focus first focusable element after animation
        const timer = setTimeout(() => {
            if (contentRef.current) {
                const first = contentRef.current.querySelector<HTMLElement>(
                    'input, button:not([disabled]), [tabindex]:not([tabindex="-1"])'
                )
                first?.focus()
            }
        }, 100)

        return () => {
            document.removeEventListener('keydown', handler)
            clearTimeout(timer)
        }
    }, [open, handleKeyDown])

    return (
        <AnimatePresence>
            {open && (
                <div
                    className="absolute inset-0 z-50 flex items-center justify-center pointer-events-auto"
                    role="dialog"
                    aria-modal="true"
                    aria-labelledby={titleId}
                >
                    {/* Backdrop */}
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        transition={{ duration: 0.15 }}
                        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
                        onClick={onClose}
                        aria-hidden="true"
                    />

                    {/* Content */}
                    <motion.div
                        ref={contentRef}
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.9 }}
                        transition={{ duration: 0.15 }}
                        className={`relative bg-surface-15 border border-white/10 p-6 rounded-2xl shadow-2xl ${width}`}
                    >
                        <h3 id={titleId} className="text-lg font-semibold text-surface-90 mb-4">
                            {title}
                        </h3>
                        {children}
                    </motion.div>
                </div>
            )}
        </AnimatePresence>
    )
}
