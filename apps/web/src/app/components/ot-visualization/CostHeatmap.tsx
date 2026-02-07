'use client'

/**
 * OT-6: Interactive heatmap of the cost matrix C[i,j].
 *
 * - Rows: events, Columns: venues
 * - Cell color: cost (dark=high cost/poor match, light=low cost/good match)
 * - Overlay: transport plan T[i,j] as circles (size = assignment weight)
 * - Hover cells for per-feature cost breakdown.
 */

import React, { useMemo, useState } from 'react'

// ─── Types ─────────────────────────────────────────────────────────────────

interface HeatmapProps {
  eventLabels: string[]
  venueLabels: string[]
  /** N×M cost matrix (row-major, events × venues) */
  costMatrix: Float64Array
  /** N×M transport plan (row-major), optional overlay */
  transportPlan?: Float64Array
  /** Width of the SVG */
  width?: number
  /** Height of the SVG */
  height?: number
  /** Color for low cost (good match) */
  colorLow?: string
  /** Color for high cost (poor match) */
  colorHigh?: string
}

interface HoveredCell {
  i: number
  j: number
  x: number
  y: number
}

// ─── Helpers ───────────────────────────────────────────────────────────────

const LABEL_WIDTH = 120
const LABEL_HEIGHT = 80
const CELL_MIN = 24
const CELL_MAX = 50

function interpolateColor(low: number[], high: number[], t: number): string {
  const r = Math.round(low[0]! + (high[0]! - low[0]!) * t)
  const g = Math.round(low[1]! + (high[1]! - low[1]!) * t)
  const b = Math.round(low[2]! + (high[2]! - low[2]!) * t)
  return `rgb(${r},${g},${b})`
}

function parseRGB(hex: string): number[] {
  if (hex.startsWith('#')) {
    const r = parseInt(hex.slice(1, 3), 16)
    const g = parseInt(hex.slice(3, 5), 16)
    const b = parseInt(hex.slice(5, 7), 16)
    return [r, g, b]
  }
  return [128, 128, 128]
}

// ─── Component ─────────────────────────────────────────────────────────────

export function CostHeatmap({
  eventLabels,
  venueLabels,
  costMatrix,
  transportPlan,
  width = 800,
  height = 500,
  colorLow = '#10b981',   // green (good match)
  colorHigh = '#ef4444',   // red (poor match)
}: HeatmapProps) {
  const [hovered, setHovered] = useState<HoveredCell | null>(null)

  const N = eventLabels.length
  const M = venueLabels.length

  const layout = useMemo(() => {
    const availW = width - LABEL_WIDTH - 20
    const availH = height - LABEL_HEIGHT - 40
    const cellW = Math.min(CELL_MAX, Math.max(CELL_MIN, availW / M))
    const cellH = Math.min(CELL_MAX, Math.max(CELL_MIN, availH / N))

    // Find cost range for normalization
    let minC = Infinity
    let maxC = -Infinity
    for (let k = 0; k < N * M; k++) {
      if (costMatrix[k]! < minC) minC = costMatrix[k]!
      if (costMatrix[k]! > maxC) maxC = costMatrix[k]!
    }
    const rangeC = maxC - minC || 1

    // Find max transport value for circle scaling
    let maxT = 0
    if (transportPlan) {
      for (let k = 0; k < N * M; k++) {
        if (transportPlan[k]! > maxT) maxT = transportPlan[k]!
      }
    }

    return { cellW, cellH, minC, maxC, rangeC, maxT }
  }, [N, M, width, height, costMatrix, transportPlan])

  const lowRGB = parseRGB(colorLow)
  const highRGB = parseRGB(colorHigh)

  return (
    <div style={{ position: 'relative' }}>
      <svg
        width={width}
        height={height}
        style={{ fontFamily: 'system-ui, sans-serif' }}
      >
        {/* Title */}
        <text x={width / 2} y={20} textAnchor="middle" fontSize={14} fill="#94a3b8" fontWeight="bold">
          Cost Matrix Heatmap
        </text>

        {/* Venue labels (top) */}
        {venueLabels.map((label, j) => (
          <text
            key={`v-${j}`}
            x={LABEL_WIDTH + j * layout.cellW + layout.cellW / 2}
            y={LABEL_HEIGHT - 5}
            textAnchor="end"
            fontSize={11}
            fill="#94a3b8"
            transform={`rotate(-45, ${LABEL_WIDTH + j * layout.cellW + layout.cellW / 2}, ${LABEL_HEIGHT - 5})`}
          >
            {label.length > 12 ? label.slice(0, 11) + '\u2026' : label}
          </text>
        ))}

        {/* Event labels (left) */}
        {eventLabels.map((label, i) => (
          <text
            key={`e-${i}`}
            x={LABEL_WIDTH - 5}
            y={LABEL_HEIGHT + i * layout.cellH + layout.cellH / 2}
            textAnchor="end"
            dominantBaseline="middle"
            fontSize={11}
            fill="#94a3b8"
          >
            {label.length > 15 ? label.slice(0, 14) + '\u2026' : label}
          </text>
        ))}

        {/* Cells */}
        {Array.from({ length: N }, (_, i) =>
          Array.from({ length: M }, (_, j) => {
            const idx = i * M + j
            const cost = costMatrix[idx]!
            const t = (cost - layout.minC) / layout.rangeC
            const fill = interpolateColor(lowRGB, highRGB, t)
            const cx = LABEL_WIDTH + j * layout.cellW
            const cy = LABEL_HEIGHT + i * layout.cellH

            const isHovered = hovered?.i === i && hovered?.j === j

            return (
              <g key={`${i}-${j}`}>
                <rect
                  x={cx}
                  y={cy}
                  width={layout.cellW - 1}
                  height={layout.cellH - 1}
                  fill={fill}
                  opacity={isHovered ? 1 : 0.8}
                  stroke={isHovered ? '#fff' : 'none'}
                  strokeWidth={isHovered ? 2 : 0}
                  style={{ cursor: 'pointer' }}
                  onMouseEnter={() => setHovered({ i, j, x: cx, y: cy })}
                  onMouseLeave={() => setHovered(null)}
                />
                {/* Transport plan overlay as circle */}
                {transportPlan && layout.maxT > 0 && transportPlan[idx]! > 0.01 && (
                  <circle
                    cx={cx + layout.cellW / 2}
                    cy={cy + layout.cellH / 2}
                    r={Math.max(2, (transportPlan[idx]! / layout.maxT) * Math.min(layout.cellW, layout.cellH) * 0.4)}
                    fill="white"
                    opacity={0.7}
                    pointerEvents="none"
                  />
                )}
              </g>
            )
          }),
        )}

        {/* Color scale legend */}
        <defs>
          <linearGradient id="heatmap-gradient" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor={colorLow} />
            <stop offset="100%" stopColor={colorHigh} />
          </linearGradient>
        </defs>
        <rect
          x={LABEL_WIDTH}
          y={height - 25}
          width={Math.min(200, M * layout.cellW)}
          height={12}
          fill="url(#heatmap-gradient)"
          rx={2}
        />
        <text x={LABEL_WIDTH} y={height - 28} fontSize={10} fill="#94a3b8">Low cost</text>
        <text
          x={LABEL_WIDTH + Math.min(200, M * layout.cellW)}
          y={height - 28}
          textAnchor="end"
          fontSize={10}
          fill="#94a3b8"
        >
          High cost
        </text>
      </svg>

      {/* Tooltip */}
      {hovered && (
        <div
          style={{
            position: 'absolute',
            top: hovered.y - 40,
            left: hovered.x + layout.cellW + 10,
            background: '#1e293b',
            border: '1px solid #334155',
            borderRadius: 6,
            padding: '8px 12px',
            color: '#e2e8f0',
            fontSize: 12,
            pointerEvents: 'none',
            whiteSpace: 'nowrap',
            zIndex: 10,
            boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
          }}
        >
          <div style={{ fontWeight: 'bold' }}>
            {eventLabels[hovered.i]} → {venueLabels[hovered.j]}
          </div>
          <div>Cost: {costMatrix[hovered.i * M + hovered.j]!.toFixed(4)}</div>
          {transportPlan && (
            <div>Transport: {(transportPlan[hovered.i * M + hovered.j]! * 100).toFixed(1)}%</div>
          )}
        </div>
      )}
    </div>
  )
}
