'use client'

import { Line } from 'react-konva'
import { useMemo } from 'react'

interface Grid2DProps {
  widthFt: number
  heightFt: number
  gridSizeFt: number
  scale: number // pixels per foot
}

export function Grid2D({ widthFt, heightFt, gridSizeFt, scale }: Grid2DProps) {
  const lines = useMemo(() => {
    const result: { points: number[]; major: boolean }[] = []
    const w = widthFt * scale
    const h = heightFt * scale
    const step = gridSizeFt * scale
    const majorEvery = gridSizeFt < 1 ? 1 / gridSizeFt : 5

    // Vertical lines
    for (let i = 0; i <= widthFt / gridSizeFt; i++) {
      const x = i * step
      result.push({ points: [x, 0, x, h], major: i % majorEvery === 0 })
    }
    // Horizontal lines
    for (let i = 0; i <= heightFt / gridSizeFt; i++) {
      const y = i * step
      result.push({ points: [0, y, w, y], major: i % majorEvery === 0 })
    }
    return result
  }, [widthFt, heightFt, gridSizeFt, scale])

  return (
    <>
      {lines.map((line, i) => (
        <Line
          key={i}
          points={line.points}
          stroke={line.major ? '#3e2723' : '#2d1b15'}
          strokeWidth={line.major ? 1 : 0.5}
          listening={false}
        />
      ))}
    </>
  )
}
