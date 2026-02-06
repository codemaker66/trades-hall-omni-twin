'use client'

import type { HTMLAttributes } from 'react'

interface SkeletonProps extends HTMLAttributes<HTMLDivElement> {
  /** Width (Tailwind class or CSS value). Default: 'w-full' */
  width?: string
  /** Height (Tailwind class). Default: 'h-4' */
  height?: string
  /** Render as a circle */
  circle?: boolean
}

export function Skeleton({ width = 'w-full', height = 'h-4', circle, className = '', ...props }: SkeletonProps) {
  return (
    <div
      className={`
        animate-pulse bg-surface-20
        ${circle ? 'rounded-full aspect-square' : 'rounded-md'}
        ${width} ${height}
        ${className}
      `.trim()}
      aria-hidden="true"
      {...props}
    />
  )
}

/** Pre-composed skeleton for a table row */
export function SkeletonRow({ columns = 4 }: { columns?: number }) {
  return (
    <div className="flex gap-4 px-4 py-3">
      {Array.from({ length: columns }, (_, i) => (
        <Skeleton key={i} width={i === 0 ? 'w-1/4' : 'w-1/6'} height="h-4" />
      ))}
    </div>
  )
}

/** Pre-composed skeleton for a card */
export function SkeletonCard() {
  return (
    <div className="bg-surface-10 border border-surface-25 rounded-xl p-5 space-y-3">
      <Skeleton width="w-2/3" height="h-5" />
      <Skeleton width="w-full" height="h-3" />
      <Skeleton width="w-4/5" height="h-3" />
      <div className="flex gap-2 pt-2">
        <Skeleton width="w-16" height="h-6" />
        <Skeleton width="w-16" height="h-6" />
      </div>
    </div>
  )
}
