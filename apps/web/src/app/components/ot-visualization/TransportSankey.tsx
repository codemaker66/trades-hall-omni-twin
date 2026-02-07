'use client'

/**
 * OT-6: Bipartite Sankey diagram for venue-event matching flow.
 *
 * - Left nodes: venues (sized by capacity)
 * - Right nodes: events (sized by guest count)
 * - Links: transport plan values (width = match strength)
 * - Color: match quality (green=strong, yellow=moderate, red=poor)
 *
 * Interactive: hover venue → highlight matched events, click link → cost breakdown.
 */

import React, { useMemo, useState, useCallback } from 'react'

// ─── Types ─────────────────────────────────────────────────────────────────

interface SankeyVenue {
  id: string
  name: string
  capacity: number
}

interface SankeyEvent {
  id: string
  name: string
  guestCount: number
}

interface CostBreakdown {
  capacity: number
  price: number
  amenity: number
  location: number
  total: number
}

interface TransportSankeyProps {
  venues: SankeyVenue[]
  events: SankeyEvent[]
  /** N×M transport plan (row-major, events × venues) */
  transportPlan: Float64Array
  /** Per-cell cost breakdowns (optional) */
  costBreakdowns?: CostBreakdown[][]
  /** Width of the SVG */
  width?: number
  /** Height of the SVG */
  height?: number
  /** Minimum link opacity to display */
  linkThreshold?: number
}

// ─── Helpers ───────────────────────────────────────────────────────────────

const PADDING = { top: 40, right: 160, bottom: 40, left: 160 }
const NODE_WIDTH = 20
const NODE_GAP = 8

function matchColor(value: number): string {
  // value ∈ [0,1]: 0=worst, 1=best
  if (value > 0.6) return '#22c55e'   // green
  if (value > 0.3) return '#eab308'   // yellow
  return '#ef4444'                     // red
}

function formatPercent(v: number): string {
  return `${(v * 100).toFixed(1)}%`
}

// ─── Component ─────────────────────────────────────────────────────────────

export function TransportSankey({
  venues,
  events,
  transportPlan,
  costBreakdowns,
  width = 800,
  height = 500,
  linkThreshold = 0.01,
}: TransportSankeyProps) {
  const [hoveredVenue, setHoveredVenue] = useState<string | null>(null)
  const [hoveredEvent, setHoveredEvent] = useState<string | null>(null)
  const [selectedLink, setSelectedLink] = useState<{ i: number; j: number } | null>(null)

  const N = events.length
  const M = venues.length

  // Layout computation
  const layout = useMemo(() => {
    const innerW = width - PADDING.left - PADDING.right
    const innerH = height - PADDING.top - PADDING.bottom

    // Event nodes (left)
    const totalGuests = events.reduce((s, e) => s + e.guestCount, 0) || 1
    let yOffset = 0
    const eventNodes = events.map((ev, i) => {
      const h = Math.max(20, (ev.guestCount / totalGuests) * innerH - NODE_GAP)
      const node = {
        id: ev.id,
        name: ev.name,
        x: PADDING.left,
        y: PADDING.top + yOffset,
        width: NODE_WIDTH,
        height: h,
        index: i,
      }
      yOffset += h + NODE_GAP
      return node
    })

    // Venue nodes (right)
    const totalCapacity = venues.reduce((s, v) => s + v.capacity, 0) || 1
    yOffset = 0
    const venueNodes = venues.map((ve, j) => {
      const h = Math.max(20, (ve.capacity / totalCapacity) * innerH - NODE_GAP)
      const node = {
        id: ve.id,
        name: ve.name,
        x: PADDING.left + innerW,
        y: PADDING.top + yOffset,
        width: NODE_WIDTH,
        height: h,
        index: j,
      }
      yOffset += h + NODE_GAP
      return node
    })

    // Links
    const links: Array<{
      i: number
      j: number
      value: number
      sourceY: number
      targetY: number
      sourceH: number
      targetH: number
    }> = []

    // Track cumulative offsets for stacking links within nodes
    const eventOffsets = new Float64Array(N)
    const venueOffsets = new Float64Array(M)

    for (let i = 0; i < N; i++) {
      for (let j = 0; j < M; j++) {
        const val = transportPlan[i * M + j]!
        if (val < linkThreshold) continue

        const eNode = eventNodes[i]!
        const vNode = venueNodes[j]!
        const linkH = val * Math.min(eNode.height, vNode.height)

        links.push({
          i,
          j,
          value: val,
          sourceY: eNode.y + eventOffsets[i]!,
          targetY: vNode.y + venueOffsets[j]!,
          sourceH: linkH,
          targetH: linkH,
        })

        eventOffsets[i] += linkH
        venueOffsets[j] += linkH
      }
    }

    return { eventNodes, venueNodes, links }
  }, [events, venues, transportPlan, width, height, linkThreshold, N, M])

  const handleLinkClick = useCallback((i: number, j: number) => {
    setSelectedLink(prev => (prev?.i === i && prev?.j === j ? null : { i, j }))
  }, [])

  return (
    <div style={{ position: 'relative' }}>
      <svg width={width} height={height} style={{ fontFamily: 'system-ui, sans-serif' }}>
        {/* Links */}
        {layout.links.map(link => {
          const isHighlighted =
            hoveredEvent === events[link.i]?.id ||
            hoveredVenue === venues[link.j]?.id
          const isSelected = selectedLink?.i === link.i && selectedLink?.j === link.j
          const opacity = isHighlighted || isSelected ? 0.8 : 0.3

          const x0 = PADDING.left + NODE_WIDTH
          const x1 = layout.venueNodes[0]?.x ?? width - PADDING.right
          const midX = (x0 + x1) / 2

          const path = `
            M ${x0} ${link.sourceY}
            C ${midX} ${link.sourceY}, ${midX} ${link.targetY}, ${x1} ${link.targetY}
            L ${x1} ${link.targetY + link.targetH}
            C ${midX} ${link.targetY + link.targetH}, ${midX} ${link.sourceY + link.sourceH}, ${x0} ${link.sourceY + link.sourceH}
            Z
          `

          return (
            <path
              key={`${link.i}-${link.j}`}
              d={path}
              fill={matchColor(link.value)}
              opacity={opacity}
              stroke={isSelected ? '#1d4ed8' : 'none'}
              strokeWidth={isSelected ? 2 : 0}
              style={{ cursor: 'pointer', transition: 'opacity 0.15s' }}
              onClick={() => handleLinkClick(link.i, link.j)}
            >
              <title>
                {events[link.i]?.name} → {venues[link.j]?.name}: {formatPercent(link.value)}
              </title>
            </path>
          )
        })}

        {/* Event nodes (left) */}
        {layout.eventNodes.map(node => (
          <g
            key={node.id}
            onMouseEnter={() => setHoveredEvent(node.id)}
            onMouseLeave={() => setHoveredEvent(null)}
            style={{ cursor: 'pointer' }}
          >
            <rect
              x={node.x}
              y={node.y}
              width={node.width}
              height={node.height}
              fill={hoveredEvent === node.id ? '#3b82f6' : '#6366f1'}
              rx={3}
            />
            <text
              x={node.x - 8}
              y={node.y + node.height / 2}
              textAnchor="end"
              dominantBaseline="middle"
              fontSize={12}
              fill="#e2e8f0"
            >
              {node.name}
            </text>
          </g>
        ))}

        {/* Venue nodes (right) */}
        {layout.venueNodes.map(node => (
          <g
            key={node.id}
            onMouseEnter={() => setHoveredVenue(node.id)}
            onMouseLeave={() => setHoveredVenue(null)}
            style={{ cursor: 'pointer' }}
          >
            <rect
              x={node.x}
              y={node.y}
              width={node.width}
              height={node.height}
              fill={hoveredVenue === node.id ? '#22c55e' : '#10b981'}
              rx={3}
            />
            <text
              x={node.x + node.width + 8}
              y={node.y + node.height / 2}
              textAnchor="start"
              dominantBaseline="middle"
              fontSize={12}
              fill="#e2e8f0"
            >
              {node.name}
            </text>
          </g>
        ))}

        {/* Title */}
        <text x={width / 2} y={20} textAnchor="middle" fontSize={14} fill="#94a3b8" fontWeight="bold">
          Optimal Transport: Venue-Event Matching Flow
        </text>
      </svg>

      {/* Cost breakdown popover */}
      {selectedLink && costBreakdowns?.[selectedLink.i]?.[selectedLink.j] && (
        <div
          style={{
            position: 'absolute',
            top: 60,
            left: width / 2 - 120,
            width: 240,
            background: '#1e293b',
            border: '1px solid #334155',
            borderRadius: 8,
            padding: 16,
            color: '#e2e8f0',
            fontSize: 13,
            boxShadow: '0 4px 12px rgba(0,0,0,0.4)',
            zIndex: 10,
          }}
        >
          <div style={{ fontWeight: 'bold', marginBottom: 8 }}>Cost Breakdown</div>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span>Capacity:</span>
            <span>{costBreakdowns[selectedLink.i]![selectedLink.j]!.capacity.toFixed(3)}</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span>Price:</span>
            <span>{costBreakdowns[selectedLink.i]![selectedLink.j]!.price.toFixed(3)}</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span>Amenity:</span>
            <span>{costBreakdowns[selectedLink.i]![selectedLink.j]!.amenity.toFixed(3)}</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span>Location:</span>
            <span>{costBreakdowns[selectedLink.i]![selectedLink.j]!.location.toFixed(3)}</span>
          </div>
          <hr style={{ border: 'none', borderTop: '1px solid #475569', margin: '8px 0' }} />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontWeight: 'bold' }}>
            <span>Total:</span>
            <span>{costBreakdowns[selectedLink.i]![selectedLink.j]!.total.toFixed(3)}</span>
          </div>
        </div>
      )}
    </div>
  )
}
