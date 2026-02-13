'use client'

import { useState } from 'react'

/**
 * Traffic Light Uncertainty Badge
 *
 * Compact indicator showing uncertainty level with expandable detail panel.
 * Green/yellow/red dot with interval width and epistemic/aleatoric breakdown.
 */

export interface UncertaintyBadgeProps {
  level: 'low' | 'medium' | 'high'
  intervalWidth: number
  dataSupport: number
  epistemicStd?: number
  aleatoricStd?: number
}

const LEVEL_CONFIG = {
  low: { color: '#22c55e', bg: 'rgba(34,197,94,0.12)', border: 'rgba(34,197,94,0.3)', label: 'Low' },
  medium: { color: '#eab308', bg: 'rgba(234,179,8,0.12)', border: 'rgba(234,179,8,0.3)', label: 'Medium' },
  high: { color: '#ef4444', bg: 'rgba(239,68,68,0.12)', border: 'rgba(239,68,68,0.3)', label: 'High' },
} as const

export function UncertaintyBadge({
  level,
  intervalWidth,
  dataSupport,
  epistemicStd,
  aleatoricStd,
}: UncertaintyBadgeProps) {
  const [expanded, setExpanded] = useState(false)
  const cfg = LEVEL_CONFIG[level]
  const hasBreakdown = epistemicStd !== undefined || aleatoricStd !== undefined
  const totalStd = epistemicStd !== undefined && aleatoricStd !== undefined
    ? Math.sqrt(epistemicStd ** 2 + aleatoricStd ** 2)
    : undefined

  return (
    <div className="inline-block">
      {/* Compact badge */}
      <button
        onClick={() => hasBreakdown && setExpanded(e => !e)}
        className="flex items-center gap-2 rounded-md px-3 py-1.5 text-sm transition-colors"
        style={{ background: cfg.bg, border: `1px solid ${cfg.border}` }}
      >
        {/* Traffic light dot */}
        <span
          className="inline-block h-2.5 w-2.5 rounded-full"
          style={{ backgroundColor: cfg.color, boxShadow: `0 0 6px ${cfg.color}` }}
        />
        <span className="text-slate-200 font-medium">{cfg.label}</span>
        <span className="text-slate-400 text-xs">&plusmn;{intervalWidth.toFixed(2)}</span>
        {hasBreakdown && (
          <svg
            width={12}
            height={12}
            viewBox="0 0 12 12"
            className="ml-1 transition-transform"
            style={{ transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)' }}
          >
            <path d="M3 4.5 L6 7.5 L9 4.5" fill="none" stroke="#94a3b8" strokeWidth={1.5} strokeLinecap="round" />
          </svg>
        )}
      </button>

      {/* Expandable detail panel */}
      {expanded && hasBreakdown && (
        <div className="mt-1 rounded-md bg-slate-900 border border-slate-700 p-3 text-xs w-64">
          <div className="text-slate-300 font-medium mb-2">Uncertainty Breakdown</div>

          {/* Stacked bar for epistemic vs aleatoric */}
          {epistemicStd !== undefined && aleatoricStd !== undefined && totalStd !== undefined && totalStd > 0 && (
            <div className="mb-3">
              <div className="flex h-3 rounded overflow-hidden bg-slate-800">
                <div
                  className="h-full"
                  style={{
                    width: `${((epistemicStd / totalStd) * 100).toFixed(1)}%`,
                    backgroundColor: '#8b5cf6',
                  }}
                />
                <div
                  className="h-full"
                  style={{
                    width: `${((aleatoricStd / totalStd) * 100).toFixed(1)}%`,
                    backgroundColor: '#06b6d4',
                  }}
                />
              </div>
              <div className="flex justify-between mt-1">
                <span className="text-violet-400">Epistemic</span>
                <span className="text-cyan-400">Aleatoric</span>
              </div>
            </div>
          )}

          <div className="space-y-1 text-slate-400">
            {epistemicStd !== undefined && (
              <div className="flex justify-between">
                <span>Epistemic std</span>
                <span className="text-violet-400">{epistemicStd.toFixed(3)}</span>
              </div>
            )}
            {aleatoricStd !== undefined && (
              <div className="flex justify-between">
                <span>Aleatoric std</span>
                <span className="text-cyan-400">{aleatoricStd.toFixed(3)}</span>
              </div>
            )}
            <div className="flex justify-between">
              <span>Interval width</span>
              <span className="text-slate-200">{intervalWidth.toFixed(3)}</span>
            </div>
            <div className="flex justify-between">
              <span>Data support</span>
              <span className="text-slate-200">{dataSupport}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
