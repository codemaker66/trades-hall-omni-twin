'use client'

import { useMemo, useState, useCallback, type MouseEvent } from 'react'

/**
 * Monte Carlo Distribution with VaR / CVaR Overlay
 *
 * Histogram of terminal revenue from MC simulation.
 * VaR line (dashed), CVaR shaded tail region.
 *
 * Data: from simulateRevenueMC() in @omni-twin/pricing-engine.
 */

export interface MonteCarloHistProps {
  /** Simulated terminal values */
  values: number[]
  /** VaR threshold (e.g. 5th percentile value) */
  var95?: number
  /** CVaR (expected shortfall below VaR) */
  cvar95?: number
  /** Number of histogram bins */
  nBins?: number
  /** Chart dimensions */
  width?: number
  height?: number
  title?: string
}

const PAD = { top: 32, right: 16, bottom: 48, left: 64 }

export function MonteCarloHist({
  values,
  var95,
  cvar95,
  nBins = 40,
  width = 640,
  height = 360,
  title = 'Monte Carlo Revenue Distribution',
}: MonteCarloHistProps) {
  const [hoverBin, setHoverBin] = useState<number | null>(null)

  const plotW = width - PAD.left - PAD.right
  const plotH = height - PAD.top - PAD.bottom

  const { bins, maxCount, valMin, valMax, binWidth } = useMemo(() => {
    if (values.length === 0) return { bins: [], maxCount: 0, valMin: 0, valMax: 1, binWidth: 1 }

    const mn = Math.min(...values)
    const mx = Math.max(...values)
    const bw = (mx - mn) / nBins || 1

    const counts = new Array(nBins).fill(0) as number[]
    for (const v of values) {
      const idx = Math.min(nBins - 1, Math.floor((v - mn) / bw))
      counts[idx]!++
    }

    return {
      bins: counts,
      maxCount: Math.max(...counts),
      valMin: mn,
      valMax: mx,
      binWidth: bw,
    }
  }, [values, nBins])

  const xScale = useCallback(
    (val: number) => PAD.left + ((val - valMin) / (valMax - valMin || 1)) * plotW,
    [valMin, valMax, plotW],
  )

  const yScale = useCallback(
    (count: number) => PAD.top + plotH - (count / (maxCount || 1)) * plotH,
    [maxCount, plotH],
  )

  const barW = plotW / nBins

  const onMouseMove = useCallback(
    (e: MouseEvent<SVGSVGElement>) => {
      const rect = e.currentTarget.getBoundingClientRect()
      const x = e.clientX - rect.left - PAD.left
      const bin = Math.floor((x / plotW) * nBins)
      setHoverBin(bin >= 0 && bin < nBins ? bin : null)
    },
    [plotW, nBins],
  )

  // Statistics
  const mean = values.length > 0 ? values.reduce((s, v) => s + v, 0) / values.length : 0

  return (
    <svg
      width={width}
      height={height}
      className="select-none"
      onMouseMove={onMouseMove}
      onMouseLeave={() => setHoverBin(null)}
    >
      <text x={width / 2} y={16} textAnchor="middle" className="fill-slate-200 text-xs font-medium">
        {title}
      </text>

      {/* CVaR shaded tail region */}
      {var95 !== undefined && (
        <rect
          x={PAD.left}
          y={PAD.top}
          width={Math.max(0, xScale(var95) - PAD.left)}
          height={plotH}
          fill="rgba(239,68,68,0.1)"
        />
      )}

      {/* Histogram bars */}
      {bins.map((count, i) => {
        const binStart = valMin + i * binWidth
        const isInTail = var95 !== undefined && binStart + binWidth / 2 < var95
        const isHovered = hoverBin === i
        return (
          <rect
            key={i}
            x={PAD.left + i * barW + 0.5}
            y={yScale(count)}
            width={Math.max(1, barW - 1)}
            height={PAD.top + plotH - yScale(count)}
            fill={isInTail ? 'rgba(239,68,68,0.6)' : 'rgba(59,130,246,0.6)'}
            stroke={isHovered ? 'white' : 'none'}
            strokeWidth={isHovered ? 1 : 0}
            rx={1}
          />
        )
      })}

      {/* VaR line */}
      {var95 !== undefined && (
        <g>
          <line
            x1={xScale(var95)}
            y1={PAD.top}
            x2={xScale(var95)}
            y2={PAD.top + plotH}
            stroke="rgb(239,68,68)"
            strokeWidth={2}
            strokeDasharray="6,3"
          />
          <text
            x={xScale(var95) + 4}
            y={PAD.top + 14}
            className="fill-red-400 text-[10px] font-medium"
          >
            VaR₉₅: {formatCurrency(var95)}
          </text>
        </g>
      )}

      {/* CVaR marker */}
      {cvar95 !== undefined && (
        <g>
          <line
            x1={xScale(cvar95)}
            y1={PAD.top}
            x2={xScale(cvar95)}
            y2={PAD.top + plotH}
            stroke="rgb(251,146,60)"
            strokeWidth={1.5}
            strokeDasharray="3,3"
          />
          <text
            x={xScale(cvar95) + 4}
            y={PAD.top + 28}
            className="fill-orange-400 text-[10px]"
          >
            CVaR₉₅: {formatCurrency(cvar95)}
          </text>
        </g>
      )}

      {/* Mean line */}
      <line
        x1={xScale(mean)}
        y1={PAD.top}
        x2={xScale(mean)}
        y2={PAD.top + plotH}
        stroke="rgb(34,197,94)"
        strokeWidth={1.5}
        strokeDasharray="4,4"
      />
      <text x={xScale(mean) + 4} y={PAD.top + plotH - 6} className="fill-green-400 text-[10px]">
        Mean: {formatCurrency(mean)}
      </text>

      {/* X-axis */}
      <line
        x1={PAD.left}
        y1={PAD.top + plotH}
        x2={width - PAD.right}
        y2={PAD.top + plotH}
        stroke="rgb(51,65,85)"
      />
      <text x={width / 2} y={height - 8} textAnchor="middle" className="fill-slate-400 text-[10px]">
        Revenue ($)
      </text>

      {/* Sample count */}
      <text x={width - PAD.right} y={height - 8} textAnchor="end" className="fill-slate-500 text-[9px]">
        N={values.length.toLocaleString()}
      </text>

      {/* Hover tooltip */}
      {hoverBin !== null && bins[hoverBin] !== undefined && (
        <g>
          <rect
            x={PAD.left + hoverBin * barW + barW / 2 - 50}
            y={yScale(bins[hoverBin]!) - 22}
            width={100}
            height={18}
            rx={4}
            fill="rgb(30,41,59)"
            stroke="rgb(71,85,105)"
          />
          <text
            x={PAD.left + hoverBin * barW + barW / 2}
            y={yScale(bins[hoverBin]!) - 9}
            textAnchor="middle"
            className="fill-slate-200 text-[10px]"
          >
            {formatCurrency(valMin + hoverBin * binWidth)}: {bins[hoverBin]} sims
          </text>
        </g>
      )}
    </svg>
  )
}

function formatCurrency(n: number): string {
  if (Math.abs(n) >= 1_000_000) return `$${(n / 1_000_000).toFixed(1)}M`
  if (Math.abs(n) >= 1_000) return `$${(n / 1_000).toFixed(1)}K`
  return `$${n.toFixed(0)}`
}
