'use client'

import { useMemo, useState, useCallback, type MouseEvent } from 'react'

/**
 * Temperature Ladder Visualization
 *
 * Real-time visualization of parallel tempering:
 * - Left panel: per-replica energy traces over iterations
 * - Right panel: temperature ladder bar chart
 * - Swap events shown as connecting lines
 * - Best-so-far energy as bold overlay
 */

export interface TemperatureVizProps {
  energyTraces: number[][]
  temperatures: number[]
  swapEvents?: Array<{ iteration: number; replica1: number; replica2: number }>
  swapAcceptanceRates?: number[]
  bestEnergy?: number[]
  width?: number
  height?: number
  title?: string
}

const PADDING = { top: 32, right: 16, bottom: 40, left: 64 }
const COLORS = ['#3b82f6', '#ef4444', '#22c55e', '#f97316', '#a855f7', '#06b6d4', '#ec4899', '#eab308',
  '#6366f1', '#14b8a6', '#f43f5e', '#84cc16', '#8b5cf6', '#0ea5e9', '#d946ef', '#facc15']

export function TemperatureViz({
  energyTraces,
  temperatures,
  swapEvents,
  swapAcceptanceRates,
  bestEnergy,
  width = 800,
  height = 400,
  title = 'Parallel Tempering',
}: TemperatureVizProps) {
  const [hoverIter, setHoverIter] = useState<number | null>(null)

  const traceW = Math.floor((width - PADDING.left - PADDING.right) * 0.72)
  const ladderX = PADDING.left + traceW + 24
  const ladderW = width - ladderX - PADDING.right
  const plotH = height - PADDING.top - PADDING.bottom

  // Compute scales
  const { maxIter, minE, maxE, yScale, xScale } = useMemo(() => {
    let mi = 0, minEv = Infinity, maxEv = -Infinity
    for (const trace of energyTraces) {
      mi = Math.max(mi, trace.length)
      for (const e of trace) {
        if (e < minEv) minEv = e
        if (e > maxEv) maxEv = e
      }
    }
    if (bestEnergy) {
      for (const e of bestEnergy) {
        if (e < minEv) minEv = e
        if (e > maxEv) maxEv = e
      }
    }
    const pad = (maxEv - minEv) * 0.05 || 1
    return {
      maxIter: mi,
      minE: minEv - pad,
      maxE: maxEv + pad,
      yScale: (v: number) => PADDING.top + plotH - ((v - (minEv - pad)) / (maxEv - minEv + 2 * pad)) * plotH,
      xScale: (i: number) => PADDING.left + (i / Math.max(1, mi - 1)) * traceW,
    }
  }, [energyTraces, bestEnergy, plotH, traceW])

  const onMouseMove = useCallback((e: MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const x = e.clientX - rect.left - PADDING.left
    const iter = Math.round((x / traceW) * (maxIter - 1))
    setHoverIter(iter >= 0 && iter < maxIter ? iter : null)
  }, [maxIter, traceW])

  // Temperature ladder scale (log)
  const { tempLogScale, tempMin, tempMax } = useMemo(() => {
    const mn = Math.min(...temperatures)
    const mx = Math.max(...temperatures)
    const lMn = Math.log10(mn || 0.01)
    const lMx = Math.log10(mx || 100)
    return {
      tempLogScale: (t: number) => PADDING.top + plotH - ((Math.log10(t) - lMn) / (lMx - lMn || 1)) * plotH,
      tempMin: mn, tempMax: mx,
    }
  }, [temperatures, plotH])

  return (
    <svg width={width} height={height} className="select-none" onMouseMove={onMouseMove} onMouseLeave={() => setHoverIter(null)}>
      <text x={width / 2} y={16} textAnchor="middle" className="fill-slate-200 text-xs font-medium">{title}</text>

      {/* Y-axis grid */}
      {Array.from({ length: 5 }, (_, i) => {
        const v = minE + ((maxE - minE) * i) / 4
        return (
          <g key={i}>
            <line x1={PADDING.left} y1={yScale(v)} x2={PADDING.left + traceW} y2={yScale(v)} stroke="rgb(51,65,85)" strokeDasharray="2,3" />
            <text x={PADDING.left - 6} y={yScale(v) + 4} textAnchor="end" className="fill-slate-400 text-[10px]">{v.toFixed(0)}</text>
          </g>
        )
      })}

      {/* Energy traces */}
      {energyTraces.map((trace, r) => (
        <polyline
          key={r}
          fill="none"
          stroke={COLORS[r % COLORS.length]}
          strokeWidth={1.2}
          strokeOpacity={0.7}
          points={trace.map((e, i) => `${xScale(i).toFixed(1)},${yScale(e).toFixed(1)}`).join(' ')}
        />
      ))}

      {/* Best energy overlay */}
      {bestEnergy && bestEnergy.length > 1 && (
        <polyline
          fill="none"
          stroke="#fbbf24"
          strokeWidth={2}
          strokeDasharray="4,2"
          points={bestEnergy.map((e, i) => `${xScale(i).toFixed(1)},${yScale(e).toFixed(1)}`).join(' ')}
        />
      )}

      {/* Hover crosshair */}
      {hoverIter !== null && (
        <g>
          <line x1={xScale(hoverIter)} y1={PADDING.top} x2={xScale(hoverIter)} y2={PADDING.top + plotH} stroke="rgba(148,163,184,0.4)" />
          <rect x={xScale(hoverIter) + 8} y={PADDING.top + 4} width={100} height={14 + energyTraces.length * 12} rx={4} fill="rgb(30,41,59)" stroke="rgb(51,65,85)" />
          <text x={xScale(hoverIter) + 14} y={PADDING.top + 16} className="fill-slate-300 text-[10px]">iter: {hoverIter}</text>
          {energyTraces.map((trace, r) => (
            <text key={r} x={xScale(hoverIter) + 14} y={PADDING.top + 28 + r * 12} className="text-[10px]" fill={COLORS[r % COLORS.length]}>
              R{r}: {(trace[hoverIter] ?? 0).toFixed(1)}
            </text>
          ))}
        </g>
      )}

      {/* X-axis label */}
      <text x={PADDING.left + traceW / 2} y={height - 8} textAnchor="middle" className="fill-slate-400 text-[10px]">Iteration</text>

      {/* Temperature ladder */}
      <text x={ladderX + ladderW / 2} y={PADDING.top - 4} textAnchor="middle" className="fill-slate-400 text-[10px]">T (log)</text>
      {temperatures.map((t, r) => {
        const y = tempLogScale(t)
        const barW = ladderW * 0.6
        return (
          <g key={r}>
            <rect x={ladderX} y={y - 4} width={barW} height={8} rx={2} fill={COLORS[r % COLORS.length]} opacity={0.8} />
            <text x={ladderX + barW + 4} y={y + 3} className="fill-slate-400 text-[9px]">{t.toFixed(1)}</text>
            {swapAcceptanceRates && r < swapAcceptanceRates.length && (
              <text x={ladderX + barW + 4} y={y + 13} className="fill-slate-500 text-[8px]">
                {(swapAcceptanceRates[r]! * 100).toFixed(0)}%
              </text>
            )}
          </g>
        )
      })}
    </svg>
  )
}
