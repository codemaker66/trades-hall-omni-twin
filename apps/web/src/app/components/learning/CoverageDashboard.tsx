'use client'

import { useMemo, useState, useCallback, type MouseEvent } from 'react'

/**
 * Coverage Dashboard
 *
 * Rolling coverage monitoring with nominal line and alert state.
 * Red shading when coverage drops below threshold.
 */

export interface CoverageDashboardProps {
  nominal: number
  rolling: number[]
  windowSize: number
  alert: boolean
  title?: string
  width?: number
  height?: number
}

const PAD = { top: 36, right: 20, bottom: 40, left: 56 }

export function CoverageDashboard({
  nominal,
  rolling,
  windowSize,
  alert,
  title = 'Coverage Monitoring',
  width = 600,
  height = 320,
}: CoverageDashboardProps) {
  const [hoverIdx, setHoverIdx] = useState<number | null>(null)

  const plotW = width - PAD.left - PAD.right
  const plotH = height - PAD.top - PAD.bottom

  const { yMin, yMax, xScale, yScale } = useMemo(() => {
    if (rolling.length === 0) {
      return { yMin: 0, yMax: 1, xScale: () => 0, yScale: () => 0 }
    }
    const mn = Math.min(...rolling, nominal)
    const mx = Math.max(...rolling, nominal)
    const pad = (mx - mn) * 0.1 || 0.05
    const yMn = Math.max(0, mn - pad)
    const yMx = Math.min(1, mx + pad)
    return {
      yMin: yMn,
      yMax: yMx,
      xScale: (i: number) => PAD.left + (i / Math.max(1, rolling.length - 1)) * plotW,
      yScale: (v: number) => PAD.top + plotH - ((v - yMn) / (yMx - yMn)) * plotH,
    }
  }, [rolling, nominal, plotW, plotH])

  // Build the undercoverage fill path (areas below nominal)
  const underPath = useMemo(() => {
    if (rolling.length < 2) return ''
    const segments: string[] = []
    let inUnder = false
    let segStart = 0

    for (let i = 0; i < rolling.length; i++) {
      const below = rolling[i]! < nominal
      if (below && !inUnder) { inUnder = true; segStart = i }
      if (!below && inUnder) {
        // Close segment
        const pts = rolling.slice(segStart, i).map((v, j) =>
          `${xScale(segStart + j).toFixed(1)},${yScale(v).toFixed(1)}`
        )
        const nomY = yScale(nominal).toFixed(1)
        segments.push(
          `M${xScale(segStart).toFixed(1)},${nomY} L${pts.join(' L')} L${xScale(i - 1).toFixed(1)},${nomY} Z`
        )
        inUnder = false
      }
    }
    // Close trailing segment
    if (inUnder) {
      const pts = rolling.slice(segStart).map((v, j) =>
        `${xScale(segStart + j).toFixed(1)},${yScale(v).toFixed(1)}`
      )
      const nomY = yScale(nominal).toFixed(1)
      segments.push(
        `M${xScale(segStart).toFixed(1)},${nomY} L${pts.join(' L')} L${xScale(rolling.length - 1).toFixed(1)},${nomY} Z`
      )
    }
    return segments.join(' ')
  }, [rolling, nominal, xScale, yScale])

  const linePath = useMemo(() => {
    if (rolling.length === 0) return ''
    return rolling.map((v, i) => `${i === 0 ? 'M' : 'L'}${xScale(i).toFixed(1)},${yScale(v).toFixed(1)}`).join(' ')
  }, [rolling, xScale, yScale])

  const onMove = useCallback(
    (e: MouseEvent<SVGSVGElement>) => {
      if (rolling.length === 0) return
      const rect = e.currentTarget.getBoundingClientRect()
      const mx = e.clientX - rect.left - PAD.left
      const idx = Math.round((mx / plotW) * (rolling.length - 1))
      setHoverIdx(idx >= 0 && idx < rolling.length ? idx : null)
    },
    [rolling, plotW],
  )

  return (
    <div className="rounded-lg bg-slate-900 p-3 inline-block">
      {/* Alert badge */}
      {alert && (
        <div className="mb-2 flex items-center gap-2 rounded-md bg-red-950 border border-red-800 px-3 py-1.5 text-xs text-red-300">
          <span className="inline-block h-2 w-2 rounded-full bg-red-500 animate-pulse" />
          Coverage below nominal ({(nominal * 100).toFixed(0)}%) -- window size: {windowSize}
        </div>
      )}

      <svg
        width={width}
        height={height}
        className="select-none"
        onMouseMove={onMove}
        onMouseLeave={() => setHoverIdx(null)}
      >
        <text x={width / 2} y={18} textAnchor="middle" className="fill-slate-200 text-xs font-medium">
          {title}
        </text>

        {/* Y-axis grid */}
        {Array.from({ length: 5 }, (_, i) => {
          const yv = yMin + ((yMax - yMin) * i) / 4
          return (
            <g key={i}>
              <line x1={PAD.left} y1={yScale(yv)} x2={PAD.left + plotW} y2={yScale(yv)} stroke="rgb(51,65,85)" strokeDasharray="2,3" />
              <text x={PAD.left - 6} y={yScale(yv) + 4} textAnchor="end" className="fill-slate-400 text-[10px]">
                {(yv * 100).toFixed(0)}%
              </text>
            </g>
          )
        })}

        {/* Nominal line */}
        <line
          x1={PAD.left} y1={yScale(nominal)}
          x2={PAD.left + plotW} y2={yScale(nominal)}
          stroke="#eab308" strokeWidth={1.5} strokeDasharray="6,3"
        />
        <text x={PAD.left + plotW + 4} y={yScale(nominal) + 4} className="fill-yellow-500 text-[10px]">
          {(nominal * 100).toFixed(0)}%
        </text>

        {/* Red undercoverage shading */}
        {underPath && <path d={underPath} fill="rgba(239,68,68,0.15)" />}

        {/* Rolling coverage line */}
        <path d={linePath} fill="none" stroke="#3b82f6" strokeWidth={2} />

        {/* X-axis labels */}
        {Array.from({ length: 5 }, (_, i) => {
          const idx = Math.round(((rolling.length - 1) * i) / 4)
          return (
            <text key={i} x={xScale(idx)} y={height - 10} textAnchor="middle" className="fill-slate-400 text-[10px]">
              {idx}
            </text>
          )
        })}
        <text x={PAD.left + plotW / 2} y={height - 2} textAnchor="middle" className="fill-slate-400 text-[10px]">
          Window Index
        </text>

        {/* Hover */}
        {hoverIdx !== null && rolling[hoverIdx] !== undefined && (
          <g>
            <line x1={xScale(hoverIdx)} y1={PAD.top} x2={xScale(hoverIdx)} y2={PAD.top + plotH} stroke="rgba(148,163,184,0.4)" />
            <circle cx={xScale(hoverIdx)} cy={yScale(rolling[hoverIdx]!)} r={4} fill="#3b82f6" stroke="#e2e8f0" strokeWidth={1.5} />
            <rect x={xScale(hoverIdx) + 10} y={yScale(rolling[hoverIdx]!) - 20} width={80} height={26} rx={4} fill="rgb(30,41,59)" stroke="rgb(51,65,85)" />
            <text x={xScale(hoverIdx) + 16} y={yScale(rolling[hoverIdx]!) - 4} className="fill-slate-300 text-[10px]">
              {(rolling[hoverIdx]! * 100).toFixed(1)}%
            </text>
          </g>
        )}
      </svg>
    </div>
  )
}
