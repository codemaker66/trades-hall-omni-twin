'use client'

import { Rect } from 'react-konva'

interface SelectionRectProps {
  x: number
  y: number
  width: number
  height: number
  visible: boolean
}

export function SelectionRect({ x, y, width, height, visible }: SelectionRectProps) {
  if (!visible) return null

  return (
    <Rect
      x={Math.min(x, x + width)}
      y={Math.min(y, y + height)}
      width={Math.abs(width)}
      height={Math.abs(height)}
      fill="rgba(99, 102, 241, 0.1)"
      stroke="#6366f1"
      strokeWidth={1}
      dash={[4, 4]}
      listening={false}
    />
  )
}
