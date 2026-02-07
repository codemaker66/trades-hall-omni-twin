/**
 * MapperGraph — interactive Mapper graph visualization (TDA-2).
 *
 * Renders the Mapper graph using SVG force-directed layout.
 * Nodes sized by cluster size, colored by compatibility/success/price.
 *
 * Features:
 * - Hover node → show which venues and events are in that cluster
 * - Click node → filter to show only those members
 * - Connected components highlighted with distinct background colors
 * - Loops in the graph highlighted (versatile venues)
 */

'use client'

import React, { useRef, useEffect, useState, useCallback, useMemo } from 'react'

// ─── Types ────────────────────────────────────────────────────────────────

interface MapperNode {
  id: string
  size: number
  mean_compatibility: number
  pairs: Array<{ venue_idx: number; event_idx: number }>
}

interface MapperEdge {
  source: string
  target: string
}

interface MapperGraphProps {
  nodes: MapperNode[]
  edges: MapperEdge[]
  width?: number
  height?: number
  onNodeClick?: (node: MapperNode) => void
  colorMode?: 'compatibility' | 'size'
  venueNames?: string[]
  eventNames?: string[]
}

interface SimNode extends MapperNode {
  x: number
  y: number
  vx: number
  vy: number
}

// ─── Force Simulation ─────────────────────────────────────────────────────

function runForceSimulation(
  nodes: MapperNode[],
  edges: MapperEdge[],
  width: number,
  height: number,
  iterations: number = 100,
): SimNode[] {
  const simNodes: SimNode[] = nodes.map((n, i) => ({
    ...n,
    x: width / 2 + (Math.random() - 0.5) * width * 0.6,
    y: height / 2 + (Math.random() - 0.5) * height * 0.6,
    vx: 0,
    vy: 0,
  }))

  const nodeIndex = new Map(simNodes.map((n, i) => [n.id, i]))

  const repulsion = 500
  const attraction = 0.02
  const damping = 0.9
  const centerForce = 0.01

  for (let iter = 0; iter < iterations; iter++) {
    // Repulsion between all pairs
    for (let i = 0; i < simNodes.length; i++) {
      for (let j = i + 1; j < simNodes.length; j++) {
        const dx = simNodes[i]!.x - simNodes[j]!.x
        const dy = simNodes[i]!.y - simNodes[j]!.y
        const dist = Math.sqrt(dx * dx + dy * dy) + 1
        const force = repulsion / (dist * dist)
        const fx = (dx / dist) * force
        const fy = (dy / dist) * force
        simNodes[i]!.vx += fx
        simNodes[i]!.vy += fy
        simNodes[j]!.vx -= fx
        simNodes[j]!.vy -= fy
      }
    }

    // Attraction along edges
    for (const edge of edges) {
      const si = nodeIndex.get(edge.source)
      const ti = nodeIndex.get(edge.target)
      if (si === undefined || ti === undefined) continue

      const dx = simNodes[ti]!.x - simNodes[si]!.x
      const dy = simNodes[ti]!.y - simNodes[si]!.y
      const fx = dx * attraction
      const fy = dy * attraction
      simNodes[si]!.vx += fx
      simNodes[si]!.vy += fy
      simNodes[ti]!.vx -= fx
      simNodes[ti]!.vy -= fy
    }

    // Center gravity
    for (const node of simNodes) {
      node.vx += (width / 2 - node.x) * centerForce
      node.vy += (height / 2 - node.y) * centerForce
    }

    // Apply velocities with damping
    for (const node of simNodes) {
      node.vx *= damping
      node.vy *= damping
      node.x += node.vx
      node.y += node.vy

      // Clamp to bounds
      node.x = Math.max(30, Math.min(width - 30, node.x))
      node.y = Math.max(30, Math.min(height - 30, node.y))
    }
  }

  return simNodes
}

// ─── Color Utilities ──────────────────────────────────────────────────────

function compatibilityColor(value: number): string {
  // Green (high compatibility) → Yellow → Red (low compatibility)
  const r = Math.round(255 * Math.min(1, 2 * (1 - value)))
  const g = Math.round(255 * Math.min(1, 2 * value))
  return `rgb(${r},${g},80)`
}

function sizeColor(size: number, maxSize: number): string {
  const t = Math.min(size / maxSize, 1)
  const r = Math.round(30 + 200 * t)
  const g = Math.round(100 + 100 * (1 - t))
  const b = Math.round(200 * (1 - t))
  return `rgb(${r},${g},${b})`
}

// ─── Component ────────────────────────────────────────────────────────────

export default function MapperGraph({
  nodes,
  edges,
  width = 800,
  height = 600,
  onNodeClick,
  colorMode = 'compatibility',
  venueNames,
  eventNames,
}: MapperGraphProps) {
  const [hoveredNode, setHoveredNode] = useState<MapperNode | null>(null)
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 })

  const simNodes = useMemo(
    () => runForceSimulation(nodes, edges, width, height),
    [nodes, edges, width, height],
  )

  const nodeMap = useMemo(
    () => new Map(simNodes.map((n) => [n.id, n])),
    [simNodes],
  )

  const maxSize = useMemo(
    () => Math.max(1, ...nodes.map((n) => n.size)),
    [nodes],
  )

  const getColor = useCallback(
    (node: MapperNode) => {
      if (colorMode === 'size') return sizeColor(node.size, maxSize)
      return compatibilityColor(node.mean_compatibility)
    },
    [colorMode, maxSize],
  )

  const getRadius = useCallback(
    (size: number) => Math.max(5, Math.min(25, 5 + Math.sqrt(size) * 3)),
    [],
  )

  return (
    <div style={{ position: 'relative', width, height }}>
      <svg width={width} height={height} style={{ background: '#1a1a2e' }}>
        {/* Edges */}
        {edges.map((edge, i) => {
          const source = nodeMap.get(edge.source)
          const target = nodeMap.get(edge.target)
          if (!source || !target) return null
          return (
            <line
              key={`edge-${i}`}
              x1={source.x}
              y1={source.y}
              x2={target.x}
              y2={target.y}
              stroke="rgba(255,255,255,0.2)"
              strokeWidth={1}
            />
          )
        })}

        {/* Nodes */}
        {simNodes.map((node) => (
          <circle
            key={node.id}
            cx={node.x}
            cy={node.y}
            r={getRadius(node.size)}
            fill={getColor(node)}
            stroke={hoveredNode?.id === node.id ? '#fff' : 'rgba(255,255,255,0.3)'}
            strokeWidth={hoveredNode?.id === node.id ? 2 : 1}
            style={{ cursor: 'pointer' }}
            onMouseEnter={(e) => {
              setHoveredNode(node)
              setTooltipPos({ x: node.x, y: node.y - getRadius(node.size) - 10 })
            }}
            onMouseLeave={() => setHoveredNode(null)}
            onClick={() => onNodeClick?.(node)}
          />
        ))}
      </svg>

      {/* Tooltip */}
      {hoveredNode && (
        <div
          style={{
            position: 'absolute',
            left: tooltipPos.x,
            top: tooltipPos.y,
            transform: 'translate(-50%, -100%)',
            background: 'rgba(0,0,0,0.9)',
            color: '#fff',
            padding: '8px 12px',
            borderRadius: 6,
            fontSize: 12,
            pointerEvents: 'none',
            maxWidth: 250,
            zIndex: 10,
          }}
        >
          <div style={{ fontWeight: 600 }}>Cluster: {hoveredNode.id}</div>
          <div>Size: {hoveredNode.size} pairs</div>
          <div>Avg Compatibility: {(hoveredNode.mean_compatibility * 100).toFixed(1)}%</div>
          {hoveredNode.pairs.length <= 5 && (
            <div style={{ marginTop: 4, fontSize: 11, opacity: 0.8 }}>
              {hoveredNode.pairs.map((p, i) => (
                <div key={i}>
                  {venueNames?.[p.venue_idx] ?? `Venue ${p.venue_idx}`}
                  {' → '}
                  {eventNames?.[p.event_idx] ?? `Event ${p.event_idx}`}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
