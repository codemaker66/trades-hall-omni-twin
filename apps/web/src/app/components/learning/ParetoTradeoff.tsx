'use client'

import { useMemo, useState } from 'react'

/**
 * Pareto Tradeoff (Fairness-Accuracy)
 *
 * Scatter plot showing accuracy vs fairness violation.
 * Pareto-optimal points highlighted and connected.
 * Hover tooltip shows threshold value for each point.
 */

export interface ParetoTradeoffProps {
  points: Array<{
    accuracy: number
    fairnessViolation: number
    threshold: number
    isPareto?: boolean
  }>
  title?: string
  width?: number
  height?: number
}

const PAD = { top: 36, right: 20, bottom: 44, left: 64 }

export function ParetoTradeoff({
  points,
  title = 'Fairness-Accuracy Tradeoff',
  width = 520,
  height = 400,
}: ParetoTradeoffProps) {
  const [hoverIdx, setHoverIdx] = useState<number | null>(null)

  const plotW = width - PAD.left - PAD.right
  const plotH = height - PAD.top - PAD.bottom

  const { xScale, yScale, xMin, xMax, yMin, yMax } = useMemo(() => {
    if (points.length === 0) {
      return { xScale: () => 0, yScale: () => 0, xMin: 0, xMax: 1, yMin: 0, yMax: 1 }
    }
    const xs = points.map(p => p.fairnessViolation)
    const ys = points.map(p => p.accuracy)
    const xMn = Math.min(...xs), xMx = Math.max(...xs)
    const yMn = Math.min(...ys), yMx = Math.max(...ys)
    const xPad = (xMx - xMn) * 0.08 || 0.05
    const yPad = (yMx - yMn) * 0.08 || 0.05
    return {
      xScale: (v: number) => PAD.left + ((v - (xMn - xPad)) / (xMx - xMn + 2 * xPad)) * plotW,
      yScale: (v: number) => PAD.top + plotH - ((v - (yMn - yPad)) / (yMx - yMn + 2 * yPad)) * plotH,
      xMin: xMn - xPad, xMax: xMx + xPad,
      yMin: yMn - yPad, yMax: yMx + yPad,
    }
  }, [points, plotW, plotH])

  // Pareto front line (sorted by fairnessViolation ascending)
  const paretoPath = useMemo(() => {
    const paretoPoints = points
      .filter(p => p.isPareto)
      .sort((a, b) => a.fairnessViolation - b.fairnessViolation)
    if (paretoPoints.length < 2) return ''
    return paretoPoints.map((p, i) =>
      `${i === 0 ? 'M' : 'L'}${xScale(p.fairnessViolation).toFixed(1)},${yScale(p.accuracy).toFixed(1)}`
    ).join(' ')
  }, [points, xScale, yScale])

  return (
    <div className="rounded-lg bg-slate-900 p-3 inline-block">
      <svg width={width} height={height} className="select-none">
        <text x={width / 2} y={18} textAnchor="middle" className="fill-slate-200 text-xs font-medium">
          {title}
        </text>

        {/* Grid */}
        {Array.from({ length: 5 }, (_, i) => {
          const xv = xMin + ((xMax - xMin) * i) / 4
          const yv = yMin + ((yMax - yMin) * i) / 4
          return (
            <g key={i}>
              <line x1={xScale(xv)} y1={PAD.top} x2={xScale(xv)} y2={PAD.top + plotH} stroke="rgb(51,65,85)" strokeDasharray="2,3" />
              <line x1={PAD.left} y1={yScale(yv)} x2={PAD.left + plotW} y2={yScale(yv)} stroke="rgb(51,65,85)" strokeDasharray="2,3" />
              <text x={xScale(xv)} y={height - 22} textAnchor="middle" className="fill-slate-400 text-[10px]">
                {xv.toFixed(2)}
              </text>
              <text x={PAD.left - 6} y={yScale(yv) + 4} textAnchor="end" className="fill-slate-400 text-[10px]">
                {yv.toFixed(2)}
              </text>
            </g>
          )
        })}

        {/* Axis labels */}
        <text x={PAD.left + plotW / 2} y={height - 6} textAnchor="middle" className="fill-slate-400 text-[10px]">
          Fairness Violation
        </text>
        <text x={14} y={PAD.top + plotH / 2} textAnchor="middle" className="fill-slate-400 text-[10px]" transform={`rotate(-90,14,${PAD.top + plotH / 2})`}>
          Accuracy
        </text>

        {/* Pareto front line */}
        {paretoPath && (
          <path d={paretoPath} fill="none" stroke="#fbbf24" strokeWidth={1.5} strokeDasharray="4,2" />
        )}

        {/* Non-Pareto points first (underneath) */}
        {points.map((p, i) => {
          if (p.isPareto) return null
          const cx = xScale(p.fairnessViolation)
          const cy = yScale(p.accuracy)
          const isHovered = hoverIdx === i
          return (
            <circle
              key={i}
              cx={cx} cy={cy}
              r={isHovered ? 5 : 3}
              fill="rgb(100,116,139)"
              opacity={isHovered ? 0.9 : 0.4}
              className="cursor-pointer"
              onMouseEnter={() => setHoverIdx(i)}
              onMouseLeave={() => setHoverIdx(null)}
            />
          )
        })}

        {/* Pareto points on top */}
        {points.map((p, i) => {
          if (!p.isPareto) return null
          const cx = xScale(p.fairnessViolation)
          const cy = yScale(p.accuracy)
          const isHovered = hoverIdx === i
          return (
            <g key={i} onMouseEnter={() => setHoverIdx(i)} onMouseLeave={() => setHoverIdx(null)}>
              <circle
                cx={cx} cy={cy}
                r={isHovered ? 8 : 6}
                fill="#fbbf24"
                stroke={isHovered ? '#fef3c7' : 'none'}
                strokeWidth={2}
                className="cursor-pointer"
              />
            </g>
          )
        })}

        {/* Hover tooltip */}
        {hoverIdx !== null && points[hoverIdx] && (() => {
          const p = points[hoverIdx]!
          const cx = xScale(p.fairnessViolation)
          const cy = yScale(p.accuracy)
          const tx = cx + 14
          const adjustedX = tx + 140 > width ? cx - 154 : tx
          const adjustedY = cy - 10 < PAD.top ? PAD.top + 4 : cy - 30
          return (
            <g>
              <rect x={adjustedX} y={adjustedY} width={140} height={52} rx={4} fill="rgb(30,41,59)" stroke="rgb(51,65,85)" />
              <text x={adjustedX + 8} y={adjustedY + 14} className="fill-slate-300 text-[10px]">
                Accuracy: {p.accuracy.toFixed(4)}
              </text>
              <text x={adjustedX + 8} y={adjustedY + 28} className="fill-slate-300 text-[10px]">
                Fairness: {p.fairnessViolation.toFixed(4)}
              </text>
              <text x={adjustedX + 8} y={adjustedY + 42} className="fill-slate-400 text-[10px]">
                Threshold: {p.threshold.toFixed(3)}{p.isPareto ? ' (Pareto)' : ''}
              </text>
            </g>
          )
        })()}

        {/* Legend */}
        <circle cx={PAD.left + 10} cy={PAD.top + 10} r={4} fill="#fbbf24" />
        <text x={PAD.left + 18} y={PAD.top + 14} className="fill-slate-400 text-[9px]">Pareto-optimal</text>
        <circle cx={PAD.left + 100} cy={PAD.top + 10} r={3} fill="rgb(100,116,139)" opacity={0.5} />
        <text x={PAD.left + 108} y={PAD.top + 14} className="fill-slate-400 text-[9px]">Dominated</text>
      </svg>
    </div>
  )
}
