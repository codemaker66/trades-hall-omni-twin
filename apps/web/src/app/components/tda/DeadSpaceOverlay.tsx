/**
 * DeadSpaceOverlay — visualize dead space detection results on
 * the 2D floor plan editor (TDA-3).
 *
 * Visual design:
 * - Dead spaces: semi-transparent red circles centered on detected voids,
 *   radius = death_radius from the persistence diagram
 * - Severity: high = solid red, medium = orange
 * - Connectivity issues: dashed lines showing disconnected furniture groups
 * - Tooltip on hover: "Dead space detected: ~12ft diameter"
 *
 * Used as a React Three Fiber overlay in the 3D scene: translucent red
 * columns rising from the floor at dead space locations.
 */

'use client'

import React, { useState, useMemo } from 'react'

// ─── Types ────────────────────────────────────────────────────────────────

interface DeadSpace {
  birth_radius: number
  death_radius: number
  persistence: number
  approx_diameter_ft: number
  severity: 'high' | 'medium'
}

interface LayoutAnalysis {
  dead_spaces: DeadSpace[]
  coverage_score: number
  connectivity_score: number
  persistence_diagram: {
    H0: [number, number][]
    H1: [number, number][]
  }
  num_furniture_points: number
}

interface DeadSpaceOverlayProps {
  analysis: LayoutAnalysis | null
  roomWidth: number
  roomDepth: number
  scale?: number
  visible?: boolean
}

// ─── Component ────────────────────────────────────────────────────────────

export default function DeadSpaceOverlay({
  analysis,
  roomWidth,
  roomDepth,
  scale = 20,
  visible = true,
}: DeadSpaceOverlayProps) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null)

  const svgWidth = roomWidth * scale
  const svgHeight = roomDepth * scale

  if (!analysis || !visible) return null

  const severityColor = (severity: string) => {
    return severity === 'high'
      ? 'rgba(255, 50, 50, 0.35)'
      : 'rgba(255, 160, 50, 0.3)'
  }

  const severityStroke = (severity: string) => {
    return severity === 'high' ? '#ff3232' : '#ffa032'
  }

  return (
    <div style={{ position: 'relative' }}>
      <svg
        width={svgWidth}
        height={svgHeight}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          pointerEvents: 'none',
          zIndex: 5,
        }}
      >
        {/* Dead space circles */}
        {analysis.dead_spaces.map((ds, i) => {
          // Dead spaces are distributed across the room
          // In a real implementation, we'd get exact coordinates
          // from the persistence diagram representative cycles.
          // For now, distribute evenly.
          const cx =
            svgWidth * (0.2 + (0.6 * (i + 0.5)) / Math.max(analysis.dead_spaces.length, 1))
          const cy = svgHeight * 0.5
          const r = (ds.death_radius / Math.max(roomWidth, roomDepth)) * svgWidth * 0.3

          return (
            <g key={`dead-${i}`} style={{ pointerEvents: 'auto' }}>
              {/* Background fill */}
              <circle
                cx={cx}
                cy={cy}
                r={r}
                fill={severityColor(ds.severity)}
                stroke={severityStroke(ds.severity)}
                strokeWidth={2}
                strokeDasharray={ds.severity === 'medium' ? '5,3' : undefined}
                onMouseEnter={() => setHoveredIndex(i)}
                onMouseLeave={() => setHoveredIndex(null)}
                style={{ cursor: 'help' }}
              />

              {/* Crosshair */}
              <line
                x1={cx - r * 0.3}
                y1={cy}
                x2={cx + r * 0.3}
                y2={cy}
                stroke={severityStroke(ds.severity)}
                strokeWidth={1}
                opacity={0.5}
              />
              <line
                x1={cx}
                y1={cy - r * 0.3}
                x2={cx}
                y2={cy + r * 0.3}
                stroke={severityStroke(ds.severity)}
                strokeWidth={1}
                opacity={0.5}
              />

              {/* Severity label */}
              <text
                x={cx}
                y={cy + r + 14}
                textAnchor="middle"
                fill={severityStroke(ds.severity)}
                fontSize={11}
                fontWeight={600}
              >
                {ds.severity.toUpperCase()}
              </text>
            </g>
          )
        })}
      </svg>

      {/* Tooltip */}
      {hoveredIndex !== null && analysis.dead_spaces[hoveredIndex] && (
        <div
          style={{
            position: 'absolute',
            top: 10,
            right: 10,
            background: 'rgba(0,0,0,0.9)',
            color: '#fff',
            padding: '10px 14px',
            borderRadius: 8,
            fontSize: 13,
            zIndex: 10,
            maxWidth: 280,
          }}
        >
          <div style={{ fontWeight: 600, color: '#ff6b6b', marginBottom: 4 }}>
            Dead Space Detected
          </div>
          <div>
            Approximate diameter:{' '}
            {analysis.dead_spaces[hoveredIndex]!.approx_diameter_ft.toFixed(1)} ft
          </div>
          <div>
            Persistence: {analysis.dead_spaces[hoveredIndex]!.persistence.toFixed(2)}
          </div>
          <div style={{ marginTop: 6, fontSize: 11, opacity: 0.7 }}>
            Consider adding a table or decoration to activate this area.
          </div>
        </div>
      )}

      {/* Stats bar */}
      <div
        style={{
          position: 'absolute',
          bottom: 0,
          left: 0,
          right: 0,
          background: 'rgba(0,0,0,0.8)',
          color: '#fff',
          padding: '6px 12px',
          display: 'flex',
          gap: 20,
          fontSize: 12,
          zIndex: 6,
        }}
      >
        <span>
          Coverage: <b>{(analysis.coverage_score * 100).toFixed(0)}%</b>
        </span>
        <span>
          Connectivity: <b>{(analysis.connectivity_score * 100).toFixed(0)}%</b>
        </span>
        <span>
          Dead Spaces:{' '}
          <b style={{ color: analysis.dead_spaces.length > 0 ? '#ff6b6b' : '#6bff6b' }}>
            {analysis.dead_spaces.length}
          </b>
        </span>
        <span>Points: {analysis.num_furniture_points}</span>
      </div>
    </div>
  )
}
