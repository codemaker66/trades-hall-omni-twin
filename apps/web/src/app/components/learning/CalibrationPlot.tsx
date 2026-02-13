'use client'

import { useMemo, useState } from 'react'

/**
 * Calibration Plot (Reliability Diagram)
 *
 * Scatter plot of predicted vs observed frequency with perfect calibration diagonal.
 * Bar chart of bin counts shown below. ECE displayed prominently.
 */

export interface CalibrationPlotProps {
  bins: Array<{ predictedMean: number; observedFrequency: number; count: number }>
  ece: number
  title?: string
  width?: number
  height?: number
}

const PAD = { top: 36, right: 20, bottom: 80, left: 56 }
const BAR_AREA_H = 48

export function CalibrationPlot({
  bins,
  ece,
  title = 'Calibration Plot',
  width = 460,
  height = 420,
}: CalibrationPlotProps) {
  const [hoverIdx, setHoverIdx] = useState<number | null>(null)

  const scatterH = height - PAD.top - PAD.bottom - BAR_AREA_H - 16
  const plotW = width - PAD.left - PAD.right

  const scale = useMemo(() => ({
    x: (v: number) => PAD.left + v * plotW,
    y: (v: number) => PAD.top + scatterH - v * scatterH,
  }), [plotW, scatterH])

  const maxCount = useMemo(
    () => Math.max(1, ...bins.map(b => b.count)),
    [bins],
  )

  const barY0 = PAD.top + scatterH + 16
  const barW = bins.length > 0 ? Math.max(4, (plotW / bins.length) - 2) : 20

  return (
    <div className="rounded-lg bg-slate-900 p-3 inline-block">
      <svg width={width} height={height} className="select-none">
        <text x={width / 2} y={18} textAnchor="middle" className="fill-slate-200 text-xs font-medium">
          {title}
        </text>

        {/* ECE badge */}
        <rect x={width - PAD.right - 90} y={PAD.top + 4} width={82} height={22} rx={4} fill="rgb(30,41,59)" stroke="rgb(71,85,105)" />
        <text x={width - PAD.right - 84} y={PAD.top + 19} className="fill-slate-400 text-[10px]">
          ECE:
        </text>
        <text x={width - PAD.right - 58} y={PAD.top + 19} className="fill-yellow-400 text-[11px] font-medium">
          {ece.toFixed(4)}
        </text>

        {/* Grid lines */}
        {Array.from({ length: 6 }, (_, i) => {
          const v = i / 5
          return (
            <g key={i}>
              <line x1={PAD.left} y1={scale.y(v)} x2={PAD.left + plotW} y2={scale.y(v)} stroke="rgb(51,65,85)" strokeDasharray="2,3" />
              <line x1={scale.x(v)} y1={PAD.top} x2={scale.x(v)} y2={PAD.top + scatterH} stroke="rgb(51,65,85)" strokeDasharray="2,3" />
              <text x={PAD.left - 6} y={scale.y(v) + 4} textAnchor="end" className="fill-slate-400 text-[10px]">{v.toFixed(1)}</text>
              <text x={scale.x(v)} y={PAD.top + scatterH + 12} textAnchor="middle" className="fill-slate-400 text-[10px]">{v.toFixed(1)}</text>
            </g>
          )
        })}

        {/* Perfect calibration line */}
        <line
          x1={scale.x(0)} y1={scale.y(0)}
          x2={scale.x(1)} y2={scale.y(1)}
          stroke="rgba(148,163,184,0.4)"
          strokeDasharray="4,3"
          strokeWidth={1.5}
        />

        {/* Deviation shading */}
        {bins.map((bin, i) => {
          const cx = scale.x(bin.predictedMean)
          const cy = scale.y(bin.observedFrequency)
          const perfectY = scale.y(bin.predictedMean)
          return (
            <line
              key={`dev-${i}`}
              x1={cx} y1={cy}
              x2={cx} y2={perfectY}
              stroke="rgba(239,68,68,0.25)"
              strokeWidth={2}
            />
          )
        })}

        {/* Scatter points */}
        {bins.map((bin, i) => {
          const cx = scale.x(bin.predictedMean)
          const cy = scale.y(bin.observedFrequency)
          const isHovered = hoverIdx === i
          return (
            <g key={i} onMouseEnter={() => setHoverIdx(i)} onMouseLeave={() => setHoverIdx(null)}>
              <circle
                cx={cx} cy={cy}
                r={isHovered ? 6 : 4}
                fill="#3b82f6"
                stroke={isHovered ? '#e2e8f0' : 'none'}
                strokeWidth={1.5}
              />
            </g>
          )
        })}

        {/* Axis labels */}
        <text x={PAD.left + plotW / 2} y={PAD.top + scatterH + 12} textAnchor="middle" className="fill-slate-400 text-[10px]">
          Predicted Probability
        </text>
        <text x={12} y={PAD.top + scatterH / 2} textAnchor="middle" className="fill-slate-400 text-[10px]" transform={`rotate(-90,12,${PAD.top + scatterH / 2})`}>
          Observed Frequency
        </text>

        {/* Bin count bars */}
        <text x={PAD.left} y={barY0 - 2} className="fill-slate-500 text-[9px]">Bin counts</text>
        {bins.map((bin, i) => {
          const bx = PAD.left + (i / bins.length) * plotW + 1
          const bh = (bin.count / maxCount) * BAR_AREA_H
          const isHovered = hoverIdx === i
          return (
            <g key={`bar-${i}`} onMouseEnter={() => setHoverIdx(i)} onMouseLeave={() => setHoverIdx(null)}>
              <rect
                x={bx} y={barY0 + BAR_AREA_H - bh}
                width={barW} height={bh}
                rx={2}
                fill={isHovered ? '#3b82f6' : 'rgb(51,65,85)'}
              />
              {isHovered && (
                <text x={bx + barW / 2} y={barY0 + BAR_AREA_H - bh - 4} textAnchor="middle" className="fill-slate-300 text-[9px]">
                  {bin.count}
                </text>
              )}
            </g>
          )
        })}

        {/* Hover tooltip */}
        {hoverIdx !== null && bins[hoverIdx] && (() => {
          const bin = bins[hoverIdx]!
          const tx = scale.x(bin.predictedMean) + 12
          const ty = scale.y(bin.observedFrequency) - 40
          const adjustedX = tx + 130 > width ? tx - 150 : tx
          const adjustedY = ty < PAD.top ? PAD.top + 4 : ty
          return (
            <g>
              <rect x={adjustedX} y={adjustedY} width={125} height={42} rx={4} fill="rgb(30,41,59)" stroke="rgb(51,65,85)" />
              <text x={adjustedX + 6} y={adjustedY + 14} className="fill-slate-300 text-[10px]">
                Pred: {bin.predictedMean.toFixed(3)}
              </text>
              <text x={adjustedX + 6} y={adjustedY + 26} className="fill-slate-300 text-[10px]">
                Obs: {bin.observedFrequency.toFixed(3)}
              </text>
              <text x={adjustedX + 6} y={adjustedY + 38} className="fill-slate-400 text-[10px]">
                n={bin.count}
              </text>
            </g>
          )
        })()}
      </svg>
    </div>
  )
}
