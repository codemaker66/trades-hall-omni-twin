'use client'

import { useEffect, useState, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

type ToastType = 'success' | 'error' | 'info' | 'warning'

interface ToastItem {
    id: string
    type: ToastType
    message: string
    duration?: number
}

const typeStyles: Record<ToastType, string> = {
    success: 'bg-success-10 border-success-50 text-success-80',
    error: 'bg-danger-10 border-danger-50 text-danger-70',
    info: 'bg-surface-20 border-info-50 text-info-80',
    warning: 'bg-surface-20 border-warning-50 text-gold-70',
}

// Global toast state — components call toast.success(), toast.error(), etc.
let listeners: ((toasts: ToastItem[]) => void)[] = []
let toastQueue: ToastItem[] = []

function notify() {
    listeners.forEach((fn) => fn([...toastQueue]))
}

export const toast = {
    success: (message: string, duration = 3000) => addToast('success', message, duration),
    error: (message: string, duration = 5000) => addToast('error', message, duration),
    info: (message: string, duration = 3000) => addToast('info', message, duration),
    warning: (message: string, duration = 4000) => addToast('warning', message, duration),
    dismiss: (id: string) => {
        toastQueue = toastQueue.filter((t) => t.id !== id)
        notify()
    },
}

function addToast(type: ToastType, message: string, duration: number) {
    const id = Math.random().toString(36).slice(2, 8)
    toastQueue.push({ id, type, message, duration })
    notify()
    if (duration > 0) {
        setTimeout(() => {
            toast.dismiss(id)
        }, duration)
    }
}

/** Place once at the root of your app to render toast notifications */
export const ToastContainer = () => {
    const [toasts, setToasts] = useState<ToastItem[]>([])

    useEffect(() => {
        listeners.push(setToasts)
        return () => {
            listeners = listeners.filter((fn) => fn !== setToasts)
        }
    }, [])

    return (
        <div
            className="fixed bottom-4 right-4 z-[60] flex flex-col gap-2 pointer-events-none"
            aria-live="polite"
            aria-label="Notifications"
        >
            <AnimatePresence>
                {toasts.map((t) => (
                    <motion.div
                        key={t.id}
                        initial={{ opacity: 0, x: 50, scale: 0.95 }}
                        animate={{ opacity: 1, x: 0, scale: 1 }}
                        exit={{ opacity: 0, x: 50, scale: 0.95 }}
                        transition={{ duration: 0.2 }}
                        className={`
                            pointer-events-auto
                            px-4 py-3 rounded-lg border shadow-xl
                            text-sm font-medium max-w-sm
                            ${typeStyles[t.type]}
                        `}
                    >
                        <div className="flex items-center justify-between gap-3">
                            <span>{t.message}</span>
                            <button
                                onClick={() => toast.dismiss(t.id)}
                                className="opacity-50 hover:opacity-100 transition-opacity text-xs"
                                aria-label="Dismiss notification"
                            >
                                ✕
                            </button>
                        </div>
                        {t.duration && t.duration > 0 && (
                            <ToastProgress duration={t.duration} />
                        )}
                    </motion.div>
                ))}
            </AnimatePresence>
        </div>
    )
}

const ToastProgress = ({ duration }: { duration: number }) => {
    const ref = useRef<HTMLDivElement>(null)

    useEffect(() => {
        if (!ref.current) return
        ref.current.style.transition = `width ${duration}ms linear`
        // Trigger reflow before setting width to 0
        ref.current.getBoundingClientRect()
        ref.current.style.width = '0%'
    }, [duration])

    return (
        <div className="mt-2 h-0.5 rounded-full bg-white/10 overflow-hidden">
            <div ref={ref} className="h-full bg-white/30 w-full rounded-full" />
        </div>
    )
}
