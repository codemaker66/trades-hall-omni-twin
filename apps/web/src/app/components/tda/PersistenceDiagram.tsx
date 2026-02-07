/**
 * PersistenceDiagram — interactive SVG persistence diagram (TDA-1).
 *
 * Renders birth-death pairs as points. Features above the diagonal
 * have positive lifespan. Further from diagonal = more significant.
 *
 * Color-coded by homology dimension:
 *   H₀ (blue) = clusters
 *   H₁ (orange) = loops
 *   H₂ (green) = voids (market gaps)
 */

'use client'

import React, { useState, useMemo } from 'react'

// ─── Types ────────────────────────────────────────────────────────────────

interface PersistenceDiagramProps {
  diagrams: {
    H0?: [number, number][]
    H1?: [number, number][]
    H2?: [number, number][]
  }
  width?: number
  height?: number
  title?: string
  showBarcode?: boolean
}

const DIM_COLORS: Record<string, string> = {
  H0: '#4a90d9',
  H1: '#e8922e',
  H2: '#45b854',
}

const DIM_LABELS: Record<string, string> = {
  H0: 'H\u2080 (clusters)',
  H1: 'H\u2081 (loops)',
  H2: 'H\u2082 (voids)',
}

// ─── Component ────────────────────────────────────────────────────────────

export default function PersistenceDiagram({
  diagrams,
  width = 500,
  height = 500,
  title = 'Persistence Diagram',
  showBarcode = false,
}: PersistenceDiagramProps) {
  const [hoveredPoint, setHoveredPoint] = useState<{
    dim: string
    birth: number
    death: number
    x: number
    y: number
  } | null>(null)

  const margin = { top: 40, right: 20, bottom: 40, left: 50 }
  const plotW = width - margin.left - margin.right
  const plotH = height - margin.top - margin.bottom

  // Compute data bounds
  const { minVal, maxVal, allPoints } = useMemo(() => {
    const pts: Array<{ dim: string; birth: number; death: number }> = []
    let lo = Infinity
    let hi = -Infinity

    for (const dim of ['H0', 'H1', 'H2'] as const) {
      const dgm = diagrams[dim]
      if (!dgm) continue
      for (const [b, d] of dgm) {
        if (!Number.isFinite(d)) continue
        pts.push({ dim, birth: b, death: d })
        lo = Math.min(lo, b, d)
        hi = Math.max(hi, b, d)
      }
    }

    if (lo === Infinity) {
      lo = 0
      hi = 1
    }

    const range = hi - lo || 1
    return {
      minVal: lo - range * 0.05,
      maxVal: hi + range * 0.05,
      allPoints: pts,
    }
  }, [diagrams])

  const scaleX = (v: number) =>
    margin.left + ((v - minVal) / (maxVal - minVal)) * plotW
  const scaleY = (v: number) =>
    margin.top + plotH - ((v - minVal) / (maxVal - minVal)) * plotH

  if (showBarcode) {
    return (
      <BarcodeView
        diagrams={diagrams}
        width={width}
        height={height}
        title={title}
      />
    )
  }

  return (
    <div style={{ position: 'relative' }}>
      <svg width={width} height={height} style={{ background: '#fafafa' }}>
        {/* Title */}
        <text x={width / 2} y={20} textAnchor="middle" fontSize={14} fontWeight={600}>
          {title}
        </text>

        {/* Diagonal line (birth = death) */}
        <line
          x1={scaleX(minVal)}
          y1={scaleY(minVal)}
          x2={scaleX(maxVal)}
          y2={scaleY(maxVal)}
          stroke="#ccc"
          strokeDasharray="4,4"
          strokeWidth={1}
        />

        {/* Axes */}
        <line
          x1={margin.left}
          y1={margin.top + plotH}
          x2={margin.left + plotW}
          y2={margin.top + plotH}
          stroke="#333"
          strokeWidth={1}
        />
        <line
          x1={margin.left}
          y1={margin.top}
          x2={margin.left}
          y2={margin.top + plotH}
          stroke="#333"
          strokeWidth={1}
        />

        {/* Axis labels */}
        <text
          x={margin.left + plotW / 2}
          y={height - 5}
          textAnchor="middle"
          fontSize={12}
        >
          Birth
        </text>
        <text
          x={15}
          y={margin.top + plotH / 2}
          textAnchor="middle"
          fontSize={12}
          transform={`rotate(-90, 15, ${margin.top + plotH / 2})`}
        >
          Death
        </text>

        {/* Points */}
        {allPoints.map((pt, i) => (
          <circle
            key={i}
            cx={scaleX(pt.birth)}
            cy={scaleY(pt.death)}
            r={5}
            fill={DIM_COLORS[pt.dim] ?? '#999'}
            stroke="#333"
            strokeWidth={0.5}
            opacity={0.8}
            style={{ cursor: 'pointer' }}
            onMouseEnter={() =>
              setHoveredPoint({
                ...pt,
                x: scaleX(pt.birth),
                y: scaleY(pt.death),
              })
            }
            onMouseLeave={() => setHoveredPoint(null)}
          />
        ))}

        {/* Legend */}
        {(['H0', 'H1', 'H2'] as const).map((dim, i) => {
          if (!diagrams[dim]?.length) return null
          return (
            <g key={dim} transform={`translate(${width - 130}, ${margin.top + i * 20})`}>
              <circle cx={0} cy={0} r={5} fill={DIM_COLORS[dim]} />
              <text x={10} y={4} fontSize={11}>
                {DIM_LABELS[dim]}
              </text>
            </g>
          )
        })}
      </svg>

      {/* Tooltip */}
      {hoveredPoint && (
        <div
          style={{
            position: 'absolute',
            left: hoveredPoint.x + 10,
            top: hoveredPoint.y - 30,
            background: 'rgba(0,0,0,0.85)',
            color: '#fff',
            padding: '6px 10px',
            borderRadius: 4,
            fontSize: 11,
            pointerEvents: 'none',
            zIndex: 10,
          }}
        >
          <div style={{ color: DIM_COLORS[hoveredPoint.dim] }}>
            {DIM_LABELS[hoveredPoint.dim]}
          </div>
          <div>Birth: {hoveredPoint.birth.toFixed(4)}</div>
          <div>Death: {hoveredPoint.death.toFixed(4)}</div>
          <div>Lifespan: {(hoveredPoint.death - hoveredPoint.birth).toFixed(4)}</div>
        </div>
      )}
    </div>
  )
}

// ─── Barcode View ─────────────────────────────────────────────────────────

function BarcodeView({
  diagrams,
  width,
  height,
  title,
}: {
  diagrams: PersistenceDiagramProps['diagrams']
  width: number
  height: number
  title: string
}) {
  const margin = { top: 30, right: 20, bottom: 30, left: 50 }
  const plotW = width - margin.left - margin.right

  // Collect all intervals
  const bars: Array<{ dim: string; birth: number; death: number }> = []
  let maxDeath = 0

  for (const dim of ['H0', 'H1', 'H2'] as const) {
    const dgm = diagrams[dim]
    if (!dgm) continue
    for (const [b, d] of dgm) {
      const finiteD = Number.isFinite(d) ? d : maxDeath * 1.2
      bars.push({ dim, birth: b, death: finiteD })
      if (Number.isFinite(d) && d > maxDeath) maxDeath = d
    }
  }

  // Update infinite deaths
  for (const bar of bars) {
    if (bar.death === 0 && maxDeath > 0) bar.death = maxDeath * 1.2
  }

  const plotH = height - margin.top - margin.bottom
  const barHeight = Math.min(4, plotH / Math.max(bars.length, 1))

  const scaleX = (v: number) =>
    margin.left + (v / (maxDeath * 1.1 || 1)) * plotW

  return (
    <svg width={width} height={height} style={{ background: '#fafafa' }}>
      <text x={width / 2} y={18} textAnchor="middle" fontSize={14} fontWeight={600}>
        {title} (Barcode)
      </text>

      {bars.map((bar, i) => (
        <rect
          key={i}
          x={scaleX(bar.birth)}
          y={margin.top + i * (barHeight + 1)}
          width={Math.max(1, scaleX(bar.death) - scaleX(bar.birth))}
          height={barHeight}
          fill={DIM_COLORS[bar.dim] ?? '#999'}
          opacity={0.8}
        />
      ))}
    </svg>
  )
}
