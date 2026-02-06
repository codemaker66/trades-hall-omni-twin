'use client'

import type { ImgHTMLAttributes } from 'react'

type AvatarSize = 'sm' | 'md' | 'lg'

interface AvatarProps extends Omit<ImgHTMLAttributes<HTMLImageElement>, 'src'> {
  src?: string | null
  name: string
  size?: AvatarSize
}

const sizeClasses: Record<AvatarSize, string> = {
  sm: 'w-6 h-6 text-[10px]',
  md: 'w-8 h-8 text-xs',
  lg: 'w-10 h-10 text-sm',
}

function getInitials(name: string): string {
  return name
    .split(' ')
    .slice(0, 2)
    .map((p) => p[0])
    .join('')
    .toUpperCase()
}

function hashColor(name: string): string {
  let hash = 0
  for (let i = 0; i < name.length; i++) {
    hash = name.charCodeAt(i) + ((hash << 5) - hash)
  }
  const hue = Math.abs(hash) % 360
  return `hsl(${hue}, 45%, 40%)`
}

export function Avatar({ src, name, size = 'md', className = '', ...props }: AvatarProps) {
  const sizeClass = sizeClasses[size]

  if (src) {
    return (
      <img
        src={src}
        alt={name}
        className={`${sizeClass} rounded-full object-cover ring-1 ring-surface-25 ${className}`}
        {...props}
      />
    )
  }

  return (
    <span
      className={`${sizeClass} rounded-full inline-flex items-center justify-center font-semibold text-white ring-1 ring-surface-25 ${className}`}
      style={{ backgroundColor: hashColor(name) }}
      role="img"
      aria-label={name}
    >
      {getInitials(name)}
    </span>
  )
}
