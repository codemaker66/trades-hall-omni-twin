'use client'

/**
 * OT-6: Radar/spider chart showing the "ideal venue" barycenter profile.
 *
 * - Axes: capacity, price range, amenity coverage, location centrality, size
 * - Filled area: the barycenter distribution
 * - Overlay: individual past bookings as faded outlines
 * - Highlight: how the currently viewed venue compares to the ideal
 *
 * Makes the OT math interpretable: "Based on your booking history, your ideal
 * venue has 200-400 capacity, mid-range pricing, strong AV, and is within 5km
 * of downtown."
 */

import React, { useMemo } from 'react'

// ─── Types ─────────────────────────────────────────────────────────────────

interface RadarAxis {
  label: string
  /** Value in [0, 1] */
  value: number
}

interface RadarProfile {
  label: string
  axes: number[]   // values for each axis, in [0, 1]
  color: string
  opacity: number
  dashed?: boolean
}

interface BarycenterRadarProps {
  /** Axis labels (e.g., ["Capacity", "Price", "Amenities", "Location", "Size"]) */
  axisLabels: string[]
  /** The ideal barycenter profile */
  barycenter: RadarProfile
  /** Past booking profiles (shown as faded outlines) */
  pastBookings?: RadarProfile[]
  /** Current venue being viewed (highlighted) */
  currentVenue?: RadarProfile
  /** SVG size (square) */
  size?: number
}

// ─── Helpers ───────────────────────────────────────────────────────────────

function polarToCartesian(
  cx: number,
  cy: number,
  radius: number,
  angleRad: number,
): { x: number; y: number } {
  return {
    x: cx + radius * Math.cos(angleRad),
    y: cy + radius * Math.sin(angleRad),
  }
}

function profileToPath(
  values: number[],
  cx: number,
  cy: number,
  maxRadius: number,
  numAxes: number,
): string {
  const points: string[] = []
  for (let i = 0; i < numAxes; i++) {
    const angle = (2 * Math.PI * i) / numAxes - Math.PI / 2
    const r = (values[i] ?? 0) * maxRadius
    const { x, y } = polarToCartesian(cx, cy, r, angle)
    points.push(`${x},${y}`)
  }
  return `M ${points.join(' L ')} Z`
}

// ─── Component ─────────────────────────────────────────────────────────────

export function BarycenterRadar({
  axisLabels,
  barycenter,
  pastBookings = [],
  currentVenue,
  size = 400,
}: BarycenterRadarProps) {
  const numAxes = axisLabels.length
  const cx = size / 2
  const cy = size / 2
  const maxRadius = size * 0.35
  const levels = 5 // concentric rings

  const grid = useMemo(() => {
    // Concentric rings
    const rings: string[] = []
    for (let l = 1; l <= levels; l++) {
      const r = (l / levels) * maxRadius
      const points: string[] = []
      for (let i = 0; i < numAxes; i++) {
        const angle = (2 * Math.PI * i) / numAxes - Math.PI / 2
        const { x, y } = polarToCartesian(cx, cy, r, angle)
        points.push(`${x},${y}`)
      }
      rings.push(`M ${points.join(' L ')} Z`)
    }

    // Axis lines
    const axisLines: Array<{ x1: number; y1: number; x2: number; y2: number }> = []
    const labelPositions: Array<{ x: number; y: number; anchor: string }> = []
    for (let i = 0; i < numAxes; i++) {
      const angle = (2 * Math.PI * i) / numAxes - Math.PI / 2
      const end = polarToCartesian(cx, cy, maxRadius, angle)
      axisLines.push({ x1: cx, y1: cy, x2: end.x, y2: end.y })

      const labelR = maxRadius + 20
      const labelPos = polarToCartesian(cx, cy, labelR, angle)
      const anchor = Math.abs(labelPos.x - cx) < 5
        ? 'middle'
        : labelPos.x > cx
          ? 'start'
          : 'end'
      labelPositions.push({ x: labelPos.x, y: labelPos.y, anchor })
    }

    return { rings, axisLines, labelPositions }
  }, [numAxes, cx, cy, maxRadius, levels])

  return (
    <svg
      width={size}
      height={size}
      style={{ fontFamily: 'system-ui, sans-serif' }}
    >
      {/* Grid rings */}
      {grid.rings.map((d, i) => (
        <path
          key={`ring-${i}`}
          d={d}
          fill="none"
          stroke="#334155"
          strokeWidth={0.5}
          opacity={0.5}
        />
      ))}

      {/* Axis lines */}
      {grid.axisLines.map((line, i) => (
        <line
          key={`axis-${i}`}
          x1={line.x1}
          y1={line.y1}
          x2={line.x2}
          y2={line.y2}
          stroke="#475569"
          strokeWidth={0.5}
        />
      ))}

      {/* Axis labels */}
      {grid.labelPositions.map((pos, i) => (
        <text
          key={`label-${i}`}
          x={pos.x}
          y={pos.y}
          textAnchor={pos.anchor}
          dominantBaseline="middle"
          fontSize={11}
          fill="#94a3b8"
        >
          {axisLabels[i]}
        </text>
      ))}

      {/* Past bookings (faded outlines) */}
      {pastBookings.map((profile, idx) => (
        <path
          key={`past-${idx}`}
          d={profileToPath(profile.axes, cx, cy, maxRadius, numAxes)}
          fill="none"
          stroke={profile.color}
          strokeWidth={1}
          opacity={profile.opacity}
          strokeDasharray={profile.dashed ? '4,3' : undefined}
        />
      ))}

      {/* Barycenter (filled) */}
      <path
        d={profileToPath(barycenter.axes, cx, cy, maxRadius, numAxes)}
        fill={barycenter.color}
        fillOpacity={0.2}
        stroke={barycenter.color}
        strokeWidth={2}
        opacity={barycenter.opacity}
      />

      {/* Barycenter dots */}
      {barycenter.axes.map((val, i) => {
        const angle = (2 * Math.PI * i) / numAxes - Math.PI / 2
        const { x, y } = polarToCartesian(cx, cy, val * maxRadius, angle)
        return (
          <circle
            key={`bary-dot-${i}`}
            cx={x}
            cy={y}
            r={3}
            fill={barycenter.color}
          />
        )
      })}

      {/* Current venue highlight */}
      {currentVenue && (
        <>
          <path
            d={profileToPath(currentVenue.axes, cx, cy, maxRadius, numAxes)}
            fill={currentVenue.color}
            fillOpacity={0.1}
            stroke={currentVenue.color}
            strokeWidth={2}
            strokeDasharray="6,3"
            opacity={currentVenue.opacity}
          />
          {currentVenue.axes.map((val, i) => {
            const angle = (2 * Math.PI * i) / numAxes - Math.PI / 2
            const { x, y } = polarToCartesian(cx, cy, val * maxRadius, angle)
            return (
              <circle
                key={`venue-dot-${i}`}
                cx={x}
                cy={y}
                r={3}
                fill={currentVenue.color}
                stroke="white"
                strokeWidth={1}
              />
            )
          })}
        </>
      )}

      {/* Legend */}
      <g transform={`translate(10, ${size - 60})`}>
        <rect x={0} y={0} width={12} height={12} fill={barycenter.color} fillOpacity={0.3} stroke={barycenter.color} strokeWidth={1.5} />
        <text x={18} y={10} fontSize={11} fill="#e2e8f0">{barycenter.label}</text>

        {currentVenue && (
          <>
            <rect x={0} y={18} width={12} height={12} fill="none" stroke={currentVenue.color} strokeWidth={1.5} strokeDasharray="3,2" />
            <text x={18} y={28} fontSize={11} fill="#e2e8f0">{currentVenue.label}</text>
          </>
        )}

        {pastBookings.length > 0 && (
          <>
            <line x1={0} y1={42} x2={12} y2={42} stroke={pastBookings[0]!.color} strokeWidth={1} opacity={0.5} />
            <text x={18} y={46} fontSize={11} fill="#94a3b8">Past bookings ({pastBookings.length})</text>
          </>
        )}
      </g>
    </svg>
  )
}
