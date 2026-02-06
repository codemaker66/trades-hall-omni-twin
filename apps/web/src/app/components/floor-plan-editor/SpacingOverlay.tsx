'use client'

import { Line, Circle } from 'react-konva'
import { useMemo } from 'react'
import type { FloorPlanItem } from './store'

interface SpacingOverlayProps {
  items: FloorPlanItem[]
  minSpacingFt: number
  scale: number
}

interface Violation {
  x1: number
  y1: number
  x2: number
  y2: number
  dist: number
}

/**
 * Highlights items that are placed too close together.
 * Draws red lines between items whose edges are closer than minSpacingFt.
 */
export function SpacingOverlay({ items, minSpacingFt, scale }: SpacingOverlayProps) {
  const violations = useMemo(() => {
    const result: Violation[] = []

    for (let i = 0; i < items.length; i++) {
      for (let j = i + 1; j < items.length; j++) {
        const a = items[i]!
        const b = items[j]!

        // Simple edge-to-edge distance (AABB approximation)
        const dx = Math.abs(a.x - b.x)
        const dy = Math.abs(a.y - b.y)
        const halfWidthA = a.widthFt / 2
        const halfWidthB = b.widthFt / 2
        const halfDepthA = a.depthFt / 2
        const halfDepthB = b.depthFt / 2

        const edgeDistX = dx - halfWidthA - halfWidthB
        const edgeDistY = dy - halfDepthA - halfDepthB

        // Items must be overlapping or close in both axes
        const edgeDist = Math.max(edgeDistX, edgeDistY)

        if (edgeDist >= 0 && edgeDist < minSpacingFt) {
          result.push({
            x1: a.x * scale,
            y1: a.y * scale,
            x2: b.x * scale,
            y2: b.y * scale,
            dist: edgeDist,
          })
        }
      }
    }

    return result
  }, [items, minSpacingFt, scale])

  if (violations.length === 0) return null

  return (
    <>
      {violations.map((v, i) => (
        <Line
          key={i}
          points={[v.x1, v.y1, v.x2, v.y2]}
          stroke="#ef5350"
          strokeWidth={1.5}
          dash={[4, 3]}
          opacity={0.7}
          listening={false}
        />
      ))}
      {violations.map((v, i) => (
        <Circle
          key={`dot-${i}`}
          x={(v.x1 + v.x2) / 2}
          y={(v.y1 + v.y2) / 2}
          radius={4}
          fill="#ef5350"
          listening={false}
        />
      ))}
    </>
  )
}
