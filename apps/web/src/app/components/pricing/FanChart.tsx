'use client'

import { useMemo, useState, useCallback, type MouseEvent } from 'react'

/**
 * Fan Chart — Price Path Confidence Cones
 *
 * Renders percentile bands (5-95, 25-75) as layered filled SVG areas
 * with a median line. Interactive hover tooltip showing exact values.
 *
 * Data: precomputed percentile bands at each timestep from
 * `computePercentileBands()` in @omni-twin/pricing-engine.
 */

export interface FanChartProps {
  /** Timestep labels (e.g. days, hours) */
  timeLabels: number[]
  /** Percentile bands: { p5, p25, p50, p75, p95 } at each timestep */
  bands: Array<{
    p5: number
    p25: number
    p50: number
    p75: number
    p95: number
  }>
  /** Optional jump event timestamps (vertical markers) */
  jumpEvents?: number[]
  /** Chart dimensions */
  width?: number
  height?: number
  /** Title */
  title?: string
}

const PADDING = { top: 32, right: 16, bottom: 40, left: 64 }

export function FanChart({
  timeLabels,
  bands,
  jumpEvents,
  width = 640,
  height = 360,
  title = 'Price Forecast',
}: FanChartProps) {
  const [hoverIdx, setHoverIdx] = useState<number | null>(null)

  const plotW = width - PADDING.left - PADDING.right
  const plotH = height - PADDING.top - PADDING.bottom

  const { xScale, yScale, yMin, yMax } = useMemo(() => {
    const allVals = bands.flatMap((b) => [b.p5, b.p95])
    const mn = Math.min(...allVals)
    const mx = Math.max(...allVals)
    const pad = (mx - mn) * 0.05 || 1
    return {
      xScale: (i: number) => PADDING.left + (i / Math.max(1, bands.length - 1)) * plotW,
      yScale: (v: number) => PADDING.top + plotH - ((v - (mn - pad)) / (mx - mn + 2 * pad)) * plotH,
      yMin: mn - pad,
      yMax: mx + pad,
    }
  }, [bands, plotW, plotH])

  const buildAreaPath = useCallback(
    (upper: (b: FanChartProps['bands'][number]) => number, lower: (b: FanChartProps['bands'][number]) => number) => {
      const forward = bands.map((b, i) => `${i === 0 ? 'M' : 'L'}${xScale(i).toFixed(1)},${yScale(upper(b)).toFixed(1)}`)
      const backward = bands
        .map((b, i) => `L${xScale(bands.length - 1 - i).toFixed(1)},${yScale(lower(bands[bands.length - 1 - i]!)).toFixed(1)}`)
      return [...forward, ...backward, 'Z'].join(' ')
    },
    [bands, xScale, yScale],
  )

  const medianPath = useMemo(
    () => bands.map((b, i) => `${i === 0 ? 'M' : 'L'}${xScale(i).toFixed(1)},${yScale(b.p50).toFixed(1)}`).join(' '),
    [bands, xScale, yScale],
  )

  const outerArea = useMemo(() => buildAreaPath((b) => b.p95, (b) => b.p5), [buildAreaPath])
  const innerArea = useMemo(() => buildAreaPath((b) => b.p75, (b) => b.p25), [buildAreaPath])

  const onMouseMove = useCallback(
    (e: MouseEvent<SVGSVGElement>) => {
      const rect = e.currentTarget.getBoundingClientRect()
      const x = e.clientX - rect.left - PADDING.left
      const idx = Math.round((x / plotW) * (bands.length - 1))
      setHoverIdx(idx >= 0 && idx < bands.length ? idx : null)
    },
    [bands.length, plotW],
  )

  // Y-axis ticks
  const yTicks = useMemo(() => {
    const range = yMax - yMin
    const step = niceStep(range / 5)
    const ticks: number[] = []
    let t = Math.ceil(yMin / step) * step
    while (t <= yMax) {
      ticks.push(t)
      t += step
    }
    return ticks
  }, [yMin, yMax])

  // X-axis ticks (at most 8)
  const xTicks = useMemo(() => {
    const step = Math.max(1, Math.floor(timeLabels.length / 8))
    return timeLabels.filter((_, i) => i % step === 0).map((label, _i, arr) => ({
      label,
      idx: timeLabels.indexOf(label),
    }))
  }, [timeLabels])

  return (
    <svg
      width={width}
      height={height}
      className="select-none"
      onMouseMove={onMouseMove}
      onMouseLeave={() => setHoverIdx(null)}
    >
      {/* Title */}
      <text x={width / 2} y={16} textAnchor="middle" className="fill-slate-200 text-xs font-medium">
        {title}
      </text>

      {/* Y-axis grid + labels */}
      {yTicks.map((t) => (
        <g key={t}>
          <line
            x1={PADDING.left}
            y1={yScale(t)}
            x2={width - PADDING.right}
            y2={yScale(t)}
            stroke="rgb(51,65,85)"
            strokeDasharray="2,3"
          />
          <text x={PADDING.left - 6} y={yScale(t) + 4} textAnchor="end" className="fill-slate-400 text-[10px]">
            {formatNumber(t)}
          </text>
        </g>
      ))}

      {/* X-axis labels */}
      {xTicks.map(({ label, idx }) => (
        <text
          key={idx}
          x={xScale(idx)}
          y={height - 8}
          textAnchor="middle"
          className="fill-slate-400 text-[10px]"
        >
          {label}
        </text>
      ))}

      {/* 5-95 band */}
      <path d={outerArea} fill="rgba(59,130,246,0.15)" />
      {/* 25-75 band */}
      <path d={innerArea} fill="rgba(59,130,246,0.3)" />
      {/* Median */}
      <path d={medianPath} fill="none" stroke="rgb(59,130,246)" strokeWidth={2} />

      {/* Jump events */}
      {jumpEvents?.map((t, i) => {
        const idx = timeLabels.findIndex((tl) => tl >= t)
        if (idx < 0) return null
        return (
          <line
            key={i}
            x1={xScale(idx)}
            y1={PADDING.top}
            x2={xScale(idx)}
            y2={height - PADDING.bottom}
            stroke="rgba(239,68,68,0.5)"
            strokeDasharray="3,3"
          />
        )
      })}

      {/* Hover crosshair + tooltip */}
      {hoverIdx !== null && bands[hoverIdx] && (
        <g>
          <line
            x1={xScale(hoverIdx)}
            y1={PADDING.top}
            x2={xScale(hoverIdx)}
            y2={height - PADDING.bottom}
            stroke="rgba(148,163,184,0.4)"
          />
          <circle cx={xScale(hoverIdx)} cy={yScale(bands[hoverIdx]!.p50)} r={3} fill="rgb(59,130,246)" />
          <rect
            x={xScale(hoverIdx) + 8}
            y={yScale(bands[hoverIdx]!.p50) - 36}
            width={120}
            height={60}
            rx={4}
            fill="rgb(30,41,59)"
            stroke="rgb(51,65,85)"
          />
          <text x={xScale(hoverIdx) + 16} y={yScale(bands[hoverIdx]!.p50) - 20} className="fill-slate-300 text-[10px]">
            P95: {formatNumber(bands[hoverIdx]!.p95)}
          </text>
          <text x={xScale(hoverIdx) + 16} y={yScale(bands[hoverIdx]!.p50) - 6} className="fill-blue-400 text-[10px] font-medium">
            Med: {formatNumber(bands[hoverIdx]!.p50)}
          </text>
          <text x={xScale(hoverIdx) + 16} y={yScale(bands[hoverIdx]!.p50) + 8} className="fill-slate-300 text-[10px]">
            P5: {formatNumber(bands[hoverIdx]!.p5)}
          </text>
        </g>
      )}
    </svg>
  )
}

function niceStep(rough: number): number {
  const pow = Math.pow(10, Math.floor(Math.log10(rough)))
  const norm = rough / pow
  if (norm <= 1) return pow
  if (norm <= 2) return 2 * pow
  if (norm <= 5) return 5 * pow
  return 10 * pow
}

function formatNumber(n: number): string {
  if (Math.abs(n) >= 1000) return `$${(n / 1000).toFixed(1)}K`
  return `$${n.toFixed(0)}`
}
