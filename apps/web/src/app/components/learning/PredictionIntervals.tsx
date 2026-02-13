'use client'

import { useMemo, useState, useCallback, type MouseEvent } from 'react'

/**
 * Prediction Intervals Fan Chart
 *
 * SVG area chart showing point estimates with 50% and 90% confidence bands.
 * Hover tooltip displays exact values at each data point.
 */

export interface PredictionIntervalsProps {
  data: Array<{
    x: number
    y: number
    ci90Lower: number
    ci90Upper: number
    ci50Lower: number
    ci50Upper: number
  }>
  title?: string
  width?: number
  height?: number
}

const PAD = { top: 32, right: 20, bottom: 40, left: 56 }

export function PredictionIntervals({
  data,
  title = 'Prediction Intervals',
  width = 640,
  height = 340,
}: PredictionIntervalsProps) {
  const [hoverIdx, setHoverIdx] = useState<number | null>(null)

  const plotW = width - PAD.left - PAD.right
  const plotH = height - PAD.top - PAD.bottom

  const { xScale, yScale, xMin, xMax, yMin, yMax } = useMemo(() => {
    if (data.length === 0) {
      return { xScale: () => 0, yScale: () => 0, xMin: 0, xMax: 1, yMin: 0, yMax: 1 }
    }
    const xs = data.map(d => d.x)
    const allY = data.flatMap(d => [d.ci90Lower, d.ci90Upper, d.y])
    const xMn = Math.min(...xs), xMx = Math.max(...xs)
    const yMn = Math.min(...allY), yMx = Math.max(...allY)
    const xPad = (xMx - xMn) * 0.02 || 1
    const yPad = (yMx - yMn) * 0.05 || 1
    return {
      xScale: (v: number) => PAD.left + ((v - (xMn - xPad)) / (xMx - xMn + 2 * xPad)) * plotW,
      yScale: (v: number) => PAD.top + plotH - ((v - (yMn - yPad)) / (yMx - yMn + 2 * yPad)) * plotH,
      xMin: xMn - xPad, xMax: xMx + xPad,
      yMin: yMn - yPad, yMax: yMx + yPad,
    }
  }, [data, plotW, plotH])

  const bandPath = useCallback(
    (lower: (d: (typeof data)[0]) => number, upper: (d: (typeof data)[0]) => number) => {
      if (data.length === 0) return ''
      const top = data.map(d => `${xScale(d.x).toFixed(1)},${yScale(upper(d)).toFixed(1)}`)
      const bot = [...data].reverse().map(d => `${xScale(d.x).toFixed(1)},${yScale(lower(d)).toFixed(1)}`)
      return `M${top.join(' L')} L${bot.join(' L')} Z`
    },
    [data, xScale, yScale],
  )

  const linePath = useMemo(() => {
    if (data.length === 0) return ''
    return data.map((d, i) => `${i === 0 ? 'M' : 'L'}${xScale(d.x).toFixed(1)},${yScale(d.y).toFixed(1)}`).join(' ')
  }, [data, xScale, yScale])

  const onMove = useCallback(
    (e: MouseEvent<SVGSVGElement>) => {
      if (data.length === 0) return
      const rect = e.currentTarget.getBoundingClientRect()
      const mx = e.clientX - rect.left
      let closest = 0
      let minDist = Infinity
      for (let i = 0; i < data.length; i++) {
        const dx = Math.abs(xScale(data[i]!.x) - mx)
        if (dx < minDist) { minDist = dx; closest = i }
      }
      setHoverIdx(minDist < 30 ? closest : null)
    },
    [data, xScale],
  )

  const ci90Path = bandPath(d => d.ci90Lower, d => d.ci90Upper)
  const ci50Path = bandPath(d => d.ci50Lower, d => d.ci50Upper)

  return (
    <div className="rounded-lg bg-slate-900 p-3 inline-block">
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

        {/* Grid lines */}
        {Array.from({ length: 5 }, (_, i) => {
          const yv = yMin + ((yMax - yMin) * i) / 4
          return (
            <g key={i}>
              <line x1={PAD.left} y1={yScale(yv)} x2={PAD.left + plotW} y2={yScale(yv)} stroke="rgb(51,65,85)" strokeDasharray="2,3" />
              <text x={PAD.left - 6} y={yScale(yv) + 4} textAnchor="end" className="fill-slate-400 text-[10px]">
                {yv.toFixed(1)}
              </text>
            </g>
          )
        })}
        {Array.from({ length: 5 }, (_, i) => {
          const xv = xMin + ((xMax - xMin) * i) / 4
          return (
            <text key={`x${i}`} x={xScale(xv)} y={height - 10} textAnchor="middle" className="fill-slate-400 text-[10px]">
              {xv.toFixed(1)}
            </text>
          )
        })}

        {/* Confidence bands */}
        <path d={ci90Path} fill="rgba(59,130,246,0.15)" />
        <path d={ci50Path} fill="rgba(59,130,246,0.30)" />

        {/* Point estimate line */}
        <path d={linePath} fill="none" stroke="#3b82f6" strokeWidth={2} />

        {/* Axis labels */}
        <text x={PAD.left + plotW / 2} y={height - 2} textAnchor="middle" className="fill-slate-400 text-[10px]">
          X
        </text>
        <text x={12} y={PAD.top + plotH / 2} textAnchor="middle" className="fill-slate-400 text-[10px]" transform={`rotate(-90,12,${PAD.top + plotH / 2})`}>
          Y
        </text>

        {/* Hover tooltip */}
        {hoverIdx !== null && data[hoverIdx] && (() => {
          const d = data[hoverIdx]!
          const cx = xScale(d.x)
          const cy = yScale(d.y)
          const tooltipX = cx + 12
          const tooltipW = 140
          const adjustedX = tooltipX + tooltipW > width ? cx - tooltipW - 12 : tooltipX
          return (
            <g>
              <line x1={cx} y1={PAD.top} x2={cx} y2={PAD.top + plotH} stroke="rgba(148,163,184,0.4)" />
              <circle cx={cx} cy={cy} r={4} fill="#3b82f6" stroke="#e2e8f0" strokeWidth={1.5} />
              <rect x={adjustedX} y={PAD.top + 4} width={tooltipW} height={80} rx={4} fill="rgb(30,41,59)" stroke="rgb(51,65,85)" />
              <text x={adjustedX + 8} y={PAD.top + 18} className="fill-slate-300 text-[10px]">x: {d.x.toFixed(2)}</text>
              <text x={adjustedX + 8} y={PAD.top + 30} className="fill-blue-400 text-[10px]">y: {d.y.toFixed(2)}</text>
              <text x={adjustedX + 8} y={PAD.top + 44} className="fill-slate-400 text-[10px]">
                90% CI: [{d.ci90Lower.toFixed(2)}, {d.ci90Upper.toFixed(2)}]
              </text>
              <text x={adjustedX + 8} y={PAD.top + 58} className="fill-slate-400 text-[10px]">
                50% CI: [{d.ci50Lower.toFixed(2)}, {d.ci50Upper.toFixed(2)}]
              </text>
              <text x={adjustedX + 8} y={PAD.top + 72} className="fill-slate-500 text-[10px]">
                width(90): {(d.ci90Upper - d.ci90Lower).toFixed(2)}
              </text>
            </g>
          )
        })()}

        {/* Legend */}
        <rect x={PAD.left + 8} y={PAD.top + 4} width={8} height={8} fill="rgba(59,130,246,0.15)" stroke="rgba(59,130,246,0.3)" />
        <text x={PAD.left + 20} y={PAD.top + 12} className="fill-slate-400 text-[9px]">90% CI</text>
        <rect x={PAD.left + 60} y={PAD.top + 4} width={8} height={8} fill="rgba(59,130,246,0.30)" stroke="rgba(59,130,246,0.5)" />
        <text x={PAD.left + 72} y={PAD.top + 12} className="fill-slate-400 text-[9px]">50% CI</text>
        <line x1={PAD.left + 112} y1={PAD.top + 8} x2={PAD.left + 126} y2={PAD.top + 8} stroke="#3b82f6" strokeWidth={2} />
        <text x={PAD.left + 130} y={PAD.top + 12} className="fill-slate-400 text-[9px]">Estimate</text>
      </svg>
    </div>
  )
}
