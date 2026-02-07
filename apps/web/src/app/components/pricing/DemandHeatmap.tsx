'use client'

import { useMemo, useState, useCallback, type MouseEvent } from 'react'

/**
 * Demand Intensity Heatmap
 *
 * Time-of-day x day-of-week heatmap showing booking arrival rates.
 * Color intensity maps to Hawkes intensity values.
 *
 * Data: from hawkesIntensityCurve() or aggregated booking data.
 */

export interface DemandHeatmapProps {
  /** Row labels (e.g. days of week) */
  rowLabels: string[]
  /** Column labels (e.g. hours of day) */
  colLabels: string[]
  /** Values: data[row][col] — intensity/count */
  data: number[][]
  /** Chart dimensions */
  width?: number
  height?: number
  title?: string
  /** Color scheme: 'heat' (red-yellow) or 'cool' (blue-cyan) */
  colorScheme?: 'heat' | 'cool'
}

const PAD = { top: 32, right: 80, bottom: 40, left: 80 }

export function DemandHeatmap({
  rowLabels,
  colLabels,
  data,
  width = 640,
  height = 320,
  title = 'Demand Intensity',
  colorScheme = 'heat',
}: DemandHeatmapProps) {
  const [hover, setHover] = useState<{ row: number; col: number } | null>(null)

  const nRows = rowLabels.length
  const nCols = colLabels.length
  const plotW = width - PAD.left - PAD.right
  const plotH = height - PAD.top - PAD.bottom

  const cellW = plotW / nCols
  const cellH = plotH / nRows

  const { minVal, maxVal } = useMemo(() => {
    let mn = Infinity
    let mx = -Infinity
    for (let r = 0; r < nRows; r++) {
      for (let c = 0; c < nCols; c++) {
        const v = data[r]?.[c] ?? 0
        mn = Math.min(mn, v)
        mx = Math.max(mx, v)
      }
    }
    return { minVal: mn, maxVal: mx }
  }, [data, nRows, nCols])

  const colorFn = useCallback(
    (val: number) => {
      const t = maxVal > minVal ? (val - minVal) / (maxVal - minVal) : 0
      if (colorScheme === 'heat') {
        // Black → Red → Yellow → White
        const r = Math.min(255, Math.round(t * 2 * 255))
        const g = Math.min(255, Math.round(Math.max(0, t * 2 - 1) * 255))
        const b = Math.round(Math.max(0, t - 0.8) * 5 * 255)
        return `rgb(${r},${g},${b})`
      }
      // Cool: dark blue → cyan → white
      const r = Math.round(Math.max(0, t - 0.5) * 2 * 255)
      const g = Math.min(255, Math.round(t * 1.5 * 255))
      const b = Math.min(255, Math.round(128 + t * 127))
      return `rgb(${r},${g},${b})`
    },
    [minVal, maxVal, colorScheme],
  )

  const onMouseMove = useCallback(
    (e: MouseEvent<SVGSVGElement>) => {
      const rect = e.currentTarget.getBoundingClientRect()
      const x = e.clientX - rect.left - PAD.left
      const y = e.clientY - rect.top - PAD.top
      const col = Math.floor(x / cellW)
      const row = Math.floor(y / cellH)
      if (row >= 0 && row < nRows && col >= 0 && col < nCols) {
        setHover({ row, col })
      } else {
        setHover(null)
      }
    },
    [cellW, cellH, nRows, nCols],
  )

  return (
    <svg
      width={width}
      height={height}
      className="select-none"
      onMouseMove={onMouseMove}
      onMouseLeave={() => setHover(null)}
    >
      <text x={width / 2} y={16} textAnchor="middle" className="fill-slate-200 text-xs font-medium">
        {title}
      </text>

      {/* Cells */}
      {Array.from({ length: nRows }, (_, r) =>
        Array.from({ length: nCols }, (_, c) => {
          const val = data[r]?.[c] ?? 0
          const isHovered = hover?.row === r && hover?.col === c
          return (
            <rect
              key={`${r}-${c}`}
              x={PAD.left + c * cellW}
              y={PAD.top + r * cellH}
              width={cellW - 1}
              height={cellH - 1}
              fill={colorFn(val)}
              stroke={isHovered ? 'white' : 'none'}
              strokeWidth={isHovered ? 2 : 0}
              rx={2}
            />
          )
        }),
      )}

      {/* Row labels */}
      {rowLabels.map((label, r) => (
        <text
          key={`row-${r}`}
          x={PAD.left - 6}
          y={PAD.top + r * cellH + cellH / 2 + 4}
          textAnchor="end"
          className="fill-slate-400 text-[10px]"
        >
          {label}
        </text>
      ))}

      {/* Column labels */}
      {colLabels.map((label, c) => (
        <text
          key={`col-${c}`}
          x={PAD.left + c * cellW + cellW / 2}
          y={height - 12}
          textAnchor="middle"
          className="fill-slate-400 text-[10px]"
        >
          {label}
        </text>
      ))}

      {/* Legend gradient */}
      <defs>
        <linearGradient id="heatmap-gradient" x1="0" y1="1" x2="0" y2="0">
          <stop offset="0%" stopColor={colorFn(minVal)} />
          <stop offset="50%" stopColor={colorFn((minVal + maxVal) / 2)} />
          <stop offset="100%" stopColor={colorFn(maxVal)} />
        </linearGradient>
      </defs>
      <rect
        x={width - PAD.right + 16}
        y={PAD.top}
        width={12}
        height={plotH}
        rx={3}
        fill="url(#heatmap-gradient)"
      />
      <text x={width - PAD.right + 34} y={PAD.top + 8} className="fill-slate-400 text-[9px]">
        {maxVal.toFixed(1)}
      </text>
      <text x={width - PAD.right + 34} y={PAD.top + plotH} className="fill-slate-400 text-[9px]">
        {minVal.toFixed(1)}
      </text>

      {/* Tooltip */}
      {hover && (
        <g>
          <rect
            x={PAD.left + hover.col * cellW + cellW / 2 - 40}
            y={PAD.top + hover.row * cellH - 22}
            width={80}
            height={18}
            rx={4}
            fill="rgb(30,41,59)"
            stroke="rgb(71,85,105)"
          />
          <text
            x={PAD.left + hover.col * cellW + cellW / 2}
            y={PAD.top + hover.row * cellH - 9}
            textAnchor="middle"
            className="fill-slate-200 text-[10px]"
          >
            {rowLabels[hover.row]}: {(data[hover.row]?.[hover.col] ?? 0).toFixed(2)}
          </text>
        </g>
      )}
    </svg>
  )
}
