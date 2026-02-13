'use client'

import { useMemo, useState, useCallback } from 'react'

/**
 * Pareto Front Dashboard
 *
 * Interactive scatter plot of Pareto-optimal solutions:
 * X = cost, Y = attendee flow, color = compliance score.
 * Click a point to preview that layout. Shows dominated vs non-dominated.
 */

export interface ParetoSolution {
  id: string
  objectives: number[]
  frontRank: number
  crowdingDistance: number
  label?: string
}

export interface ParetoDashboardProps {
  solutions: ParetoSolution[]
  objectiveNames?: [string, string, string]
  selectedId?: string
  onSelect?: (id: string) => void
  width?: number
  height?: number
  title?: string
}

const PADDING = { top: 32, right: 16, bottom: 44, left: 64 }

export function ParetoDashboard({
  solutions,
  objectiveNames = ['Cost', 'Flow', 'Compliance'],
  selectedId,
  onSelect,
  width = 700,
  height = 420,
  title = 'Pareto Front',
}: ParetoDashboardProps) {
  const [hoveredId, setHoveredId] = useState<string | null>(null)

  const scatterW = Math.floor((width - PADDING.left - PADDING.right) * 0.68)
  const panelX = PADDING.left + scatterW + 20
  const panelW = width - panelX - PADDING.right
  const plotH = height - PADDING.top - PADDING.bottom

  const { xScale, yScale, xMin, xMax, yMin, yMax, compMin, compMax } = useMemo(() => {
    if (solutions.length === 0) {
      return { xScale: () => 0, yScale: () => 0, xMin: 0, xMax: 1, yMin: 0, yMax: 1, compMin: 0, compMax: 1 }
    }
    const xs = solutions.map(s => s.objectives[0] ?? 0)
    const ys = solutions.map(s => s.objectives[1] ?? 0)
    const cs = solutions.map(s => s.objectives[2] ?? 0)
    const xMn = Math.min(...xs), xMx = Math.max(...xs)
    const yMn = Math.min(...ys), yMx = Math.max(...ys)
    const cMn = Math.min(...cs), cMx = Math.max(...cs)
    const xPad = (xMx - xMn) * 0.05 || 1
    const yPad = (yMx - yMn) * 0.05 || 1
    return {
      xScale: (v: number) => PADDING.left + ((v - (xMn - xPad)) / (xMx - xMn + 2 * xPad)) * scatterW,
      yScale: (v: number) => PADDING.top + plotH - ((v - (yMn - yPad)) / (yMx - yMn + 2 * yPad)) * plotH,
      xMin: xMn - xPad, xMax: xMx + xPad, yMin: yMn - yPad, yMax: yMx + yPad,
      compMin: cMn, compMax: cMx,
    }
  }, [solutions, scatterW, plotH])

  const compColor = useCallback((v: number) => {
    const t = compMax > compMin ? (v - compMin) / (compMax - compMin) : 0.5
    const r = Math.round(220 * t + 34 * (1 - t))
    const g = Math.round(34 * t + 197 * (1 - t))
    return `rgb(${r},${g},50)`
  }, [compMin, compMax])

  const paretoLine = useMemo(() => {
    const front = solutions.filter(s => s.frontRank === 0)
      .sort((a, b) => (a.objectives[0] ?? 0) - (b.objectives[0] ?? 0))
    if (front.length < 2) return ''
    return front.map((s, i) =>
      `${i === 0 ? 'M' : 'L'}${xScale(s.objectives[0] ?? 0).toFixed(1)},${yScale(s.objectives[1] ?? 0).toFixed(1)}`
    ).join(' ')
  }, [solutions, xScale, yScale])

  const active = solutions.find(s => s.id === (hoveredId ?? selectedId))

  return (
    <svg width={width} height={height} className="select-none">
      <text x={width / 2} y={16} textAnchor="middle" className="fill-slate-200 text-xs font-medium">{title}</text>

      {Array.from({ length: 5 }, (_, i) => {
        const xv = xMin + ((xMax - xMin) * i) / 4
        const yv = yMin + ((yMax - yMin) * i) / 4
        return (
          <g key={i}>
            <line x1={xScale(xv)} y1={PADDING.top} x2={xScale(xv)} y2={PADDING.top + plotH} stroke="rgb(51,65,85)" strokeDasharray="2,3" />
            <line x1={PADDING.left} y1={yScale(yv)} x2={PADDING.left + scatterW} y2={yScale(yv)} stroke="rgb(51,65,85)" strokeDasharray="2,3" />
            <text x={xScale(xv)} y={height - 24} textAnchor="middle" className="fill-slate-400 text-[10px]">{xv.toFixed(0)}</text>
            <text x={PADDING.left - 6} y={yScale(yv) + 4} textAnchor="end" className="fill-slate-400 text-[10px]">{yv.toFixed(0)}</text>
          </g>
        )
      })}

      <text x={PADDING.left + scatterW / 2} y={height - 6} textAnchor="middle" className="fill-slate-400 text-[10px]">{objectiveNames[0]}</text>

      {paretoLine && <path d={paretoLine} fill="none" stroke="rgba(250,204,21,0.5)" strokeWidth={1.5} />}

      {solutions.map(s => {
        const cx = xScale(s.objectives[0] ?? 0)
        const cy = yScale(s.objectives[1] ?? 0)
        const r = s.frontRank === 0 ? 5 : 3
        const isActive = s.id === (hoveredId ?? selectedId)
        return (
          <circle
            key={s.id} cx={cx} cy={cy} r={isActive ? r + 2 : r}
            fill={compColor(s.objectives[2] ?? 0)}
            stroke={isActive ? '#fbbf24' : 'none'} strokeWidth={isActive ? 2 : 0}
            opacity={s.frontRank === 0 ? 1 : 0.4}
            className="cursor-pointer"
            onMouseEnter={() => setHoveredId(s.id)}
            onMouseLeave={() => setHoveredId(null)}
            onClick={() => onSelect?.(s.id)}
          />
        )
      })}

      <text x={panelX} y={PADDING.top + 8} className="fill-slate-300 text-[11px] font-medium">
        {active ? (active.label ?? active.id) : 'Hover a point'}
      </text>
      {active && objectiveNames.map((name, i) => {
        const val = active.objectives[i] ?? 0
        const allVals = solutions.map(s => s.objectives[i] ?? 0)
        const mn = Math.min(...allVals), mx = Math.max(...allVals)
        const norm = mx > mn ? (val - mn) / (mx - mn) : 0.5
        const barY = PADDING.top + 28 + i * 36
        return (
          <g key={name}>
            <text x={panelX} y={barY} className="fill-slate-400 text-[10px]">{name}</text>
            <rect x={panelX} y={barY + 4} width={panelW} height={12} rx={2} fill="rgb(30,41,59)" />
            <rect x={panelX} y={barY + 4} width={panelW * norm} height={12} rx={2} fill={i === 2 ? compColor(val) : '#3b82f6'} />
            <text x={panelX + panelW + 4} y={barY + 14} className="fill-slate-300 text-[10px]">{val.toFixed(1)}</text>
          </g>
        )
      })}
    </svg>
  )
}
