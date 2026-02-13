'use client'

import { useMemo, useState } from 'react'

/**
 * SHAP Waterfall Chart
 *
 * Horizontal waterfall showing feature contributions from base value to prediction.
 * Red bars for features that increase the prediction, blue for decreases.
 */

export interface SHAPWaterfallProps {
  values: Array<{ feature: string; value: number; direction: 'increases' | 'decreases' }>
  baseValue: number
  prediction: number
  title?: string
  width?: number
  height?: number
}

const PAD = { top: 36, right: 60, bottom: 32, left: 120 }
const ROW_H = 28

export function SHAPWaterfall({
  values,
  baseValue,
  prediction,
  title = 'SHAP Feature Importance',
  width = 560,
  height: heightProp,
}: SHAPWaterfallProps) {
  const [hoverIdx, setHoverIdx] = useState<number | null>(null)

  const sorted = useMemo(
    () => [...values].sort((a, b) => Math.abs(b.value) - Math.abs(a.value)),
    [values],
  )

  const height = heightProp ?? PAD.top + PAD.bottom + (sorted.length + 2) * ROW_H
  const plotW = width - PAD.left - PAD.right

  // Compute cumulative positions
  const { bars, xScale, valMin, valMax } = useMemo(() => {
    let cum = baseValue
    const entries: Array<{ feature: string; start: number; end: number; value: number; direction: string }> = []

    for (const v of sorted) {
      const start = cum
      cum += v.direction === 'increases' ? v.value : -v.value
      entries.push({ feature: v.feature, start, end: cum, value: v.value, direction: v.direction })
    }

    const allVals = [baseValue, prediction, ...entries.flatMap(e => [e.start, e.end])]
    const mn = Math.min(...allVals)
    const mx = Math.max(...allVals)
    const pad = (mx - mn) * 0.08 || 1

    return {
      bars: entries,
      xScale: (v: number) => PAD.left + ((v - (mn - pad)) / (mx - mn + 2 * pad)) * plotW,
      valMin: mn - pad,
      valMax: mx + pad,
    }
  }, [sorted, baseValue, prediction, plotW])

  return (
    <div className="rounded-lg bg-slate-900 p-3 inline-block">
      <svg width={width} height={height} className="select-none">
        <text x={width / 2} y={18} textAnchor="middle" className="fill-slate-200 text-xs font-medium">
          {title}
        </text>

        {/* X-axis grid */}
        {Array.from({ length: 5 }, (_, i) => {
          const xv = valMin + ((valMax - valMin) * i) / 4
          return (
            <g key={i}>
              <line x1={xScale(xv)} y1={PAD.top} x2={xScale(xv)} y2={height - PAD.bottom} stroke="rgb(51,65,85)" strokeDasharray="2,3" />
              <text x={xScale(xv)} y={height - 14} textAnchor="middle" className="fill-slate-400 text-[10px]">
                {xv.toFixed(2)}
              </text>
            </g>
          )
        })}

        {/* Base value row */}
        <text x={PAD.left - 8} y={PAD.top + ROW_H * 0.65} textAnchor="end" className="fill-slate-300 text-[11px] font-medium">
          Base value
        </text>
        <line x1={xScale(baseValue)} y1={PAD.top + 2} x2={xScale(baseValue)} y2={PAD.top + ROW_H - 4} stroke="#94a3b8" strokeWidth={2} />
        <text x={xScale(baseValue) + 6} y={PAD.top + ROW_H * 0.65} className="fill-slate-400 text-[10px]">
          {baseValue.toFixed(3)}
        </text>

        {/* Feature bars */}
        {bars.map((bar, i) => {
          const y = PAD.top + (i + 1) * ROW_H
          const x1 = xScale(bar.start)
          const x2 = xScale(bar.end)
          const bx = Math.min(x1, x2)
          const bw = Math.max(Math.abs(x2 - x1), 1)
          const isIncrease = bar.direction === 'increases'
          const fill = isIncrease ? '#ef4444' : '#3b82f6'
          const isHovered = hoverIdx === i

          return (
            <g
              key={bar.feature}
              onMouseEnter={() => setHoverIdx(i)}
              onMouseLeave={() => setHoverIdx(null)}
            >
              {/* Connector line from previous bar end */}
              <line
                x1={xScale(bar.start)}
                y1={y - 2}
                x2={xScale(bar.start)}
                y2={y + 4}
                stroke="rgb(71,85,105)"
                strokeDasharray="2,2"
              />

              {/* Bar */}
              <rect
                x={bx}
                y={y + 4}
                width={bw}
                height={ROW_H - 10}
                rx={3}
                fill={fill}
                opacity={isHovered ? 1 : 0.8}
              />

              {/* Feature label */}
              <text x={PAD.left - 8} y={y + ROW_H * 0.58} textAnchor="end" className="fill-slate-300 text-[11px]">
                {bar.feature}
              </text>

              {/* Value label */}
              <text
                x={Math.max(x1, x2) + 4}
                y={y + ROW_H * 0.58}
                className="text-[10px]"
                fill={isIncrease ? '#fca5a5' : '#93c5fd'}
              >
                {isIncrease ? '+' : '-'}{bar.value.toFixed(3)}
              </text>
            </g>
          )
        })}

        {/* Prediction row */}
        {(() => {
          const y = PAD.top + (bars.length + 1) * ROW_H
          return (
            <g>
              <text x={PAD.left - 8} y={y + ROW_H * 0.65} textAnchor="end" className="fill-slate-200 text-[11px] font-bold">
                Prediction
              </text>
              <line x1={xScale(prediction)} y1={y + 2} x2={xScale(prediction)} y2={y + ROW_H - 4} stroke="#fbbf24" strokeWidth={2.5} />
              <text x={xScale(prediction) + 6} y={y + ROW_H * 0.65} className="fill-yellow-400 text-[10px] font-medium">
                {prediction.toFixed(3)}
              </text>
            </g>
          )
        })()}
      </svg>
    </div>
  )
}
