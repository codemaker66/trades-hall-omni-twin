'use client'

import { useMemo, useState } from 'react'

/**
 * Layout Gallery — MCMC-Sampled Diverse Layout Browser
 *
 * Grid of thumbnail previews of sampled layouts.
 * Each thumbnail: 2D top-down view of furniture arrangement.
 * Click to expand. Side panel shows metrics.
 */

export interface LayoutItem {
  x: number; y: number; width: number; depth: number; rotation: number; type: string
}

export interface LayoutSample {
  id: string
  items: LayoutItem[]
  energy: number
  metrics: { capacity: number; flow: number; compliance: number }
}

export interface LayoutGalleryProps {
  layouts: LayoutSample[]
  roomWidth: number
  roomHeight: number
  selectedId?: string
  onSelect?: (id: string) => void
  sortBy?: 'energy' | 'capacity' | 'flow' | 'compliance'
  minCompliance?: number
  columns?: number
  title?: string
}

const TYPE_COLORS: Record<string, string> = {
  chair: '#3b82f6', 'round-table': '#92400e', 'rect-table': '#a16207',
  stage: '#6b7280', bar: '#16a34a', podium: '#7c3aed',
  'dance-floor': '#ec4899', 'av-booth': '#0891b2', 'service-station': '#ea580c',
}

export function LayoutGallery({
  layouts,
  roomWidth,
  roomHeight,
  selectedId,
  onSelect,
  sortBy = 'energy',
  minCompliance = 0,
  columns = 4,
  title = 'Layout Gallery',
}: LayoutGalleryProps) {
  const [sort, setSort] = useState(sortBy)
  const [compFilter, setCompFilter] = useState(minCompliance)

  const filtered = useMemo(() => {
    let result = layouts.filter(l => l.metrics.compliance >= compFilter)
    result.sort((a, b) => {
      switch (sort) {
        case 'energy': return a.energy - b.energy
        case 'capacity': return b.metrics.capacity - a.metrics.capacity
        case 'flow': return b.metrics.flow - a.metrics.flow
        case 'compliance': return b.metrics.compliance - a.metrics.compliance
      }
    })
    return result
  }, [layouts, sort, compFilter])

  const thumbW = 160
  const thumbH = Math.round(thumbW * (roomHeight / roomWidth))

  return (
    <div className="rounded-lg bg-slate-900 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-slate-200">{title}</h3>
        <div className="flex items-center gap-3 text-xs">
          <label className="text-slate-400">
            Sort:
            <select
              value={sort}
              onChange={e => setSort(e.target.value as typeof sort)}
              className="ml-1 bg-slate-800 text-slate-300 rounded px-1 py-0.5"
            >
              <option value="energy">Energy</option>
              <option value="capacity">Capacity</option>
              <option value="flow">Flow</option>
              <option value="compliance">Compliance</option>
            </select>
          </label>
          <label className="text-slate-400">
            Min compliance:
            <input
              type="range" min={0} max={1} step={0.1}
              value={compFilter}
              onChange={e => setCompFilter(parseFloat(e.target.value))}
              className="ml-1 w-16"
            />
            <span className="ml-1 text-slate-300">{compFilter.toFixed(1)}</span>
          </label>
          <span className="text-slate-500">{filtered.length} results</span>
        </div>
      </div>

      <div className="grid gap-3" style={{ gridTemplateColumns: `repeat(${columns}, 1fr)` }}>
        {filtered.map(layout => {
          const isSelected = layout.id === selectedId
          return (
            <div
              key={layout.id}
              className={`rounded-lg overflow-hidden cursor-pointer transition-all ${
                isSelected ? 'ring-2 ring-blue-500 bg-slate-800' : 'bg-slate-800/50 hover:bg-slate-800'
              }`}
              onClick={() => onSelect?.(layout.id)}
            >
              <svg width={thumbW} height={thumbH} viewBox={`0 0 ${roomWidth} ${roomHeight}`} className="w-full">
                <rect width={roomWidth} height={roomHeight} fill="#1e293b" />
                {layout.items.map((item, i) => (
                  <rect
                    key={i}
                    x={item.x - item.width / 2}
                    y={item.y - item.depth / 2}
                    width={item.width}
                    height={item.depth}
                    fill={TYPE_COLORS[item.type] ?? '#6b7280'}
                    opacity={0.8}
                    rx={0.3}
                    transform={item.rotation ? `rotate(${item.rotation * 180 / Math.PI},${item.x},${item.y})` : undefined}
                  />
                ))}
              </svg>
              <div className="px-2 py-1.5 text-[10px] text-slate-400 space-y-0.5">
                <div className="flex justify-between">
                  <span>E: {layout.energy.toFixed(0)}</span>
                  <span>Cap: {layout.metrics.capacity}</span>
                </div>
                <div className="flex justify-between">
                  <span>Flow: {layout.metrics.flow.toFixed(1)}</span>
                  <span className={layout.metrics.compliance > 0.8 ? 'text-green-400' : 'text-red-400'}>
                    {(layout.metrics.compliance * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
