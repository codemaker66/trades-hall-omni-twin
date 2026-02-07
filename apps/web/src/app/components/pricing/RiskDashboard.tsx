'use client'

import { useMemo, useState, useCallback, type MouseEvent } from 'react'

/**
 * Risk Dashboard — Efficient Frontier + CVaR + Black-Litterman
 *
 * Panel 1: Markowitz efficient frontier curve (vol vs expected return)
 *   with current portfolio point and optimal (max Sharpe) highlight.
 * Panel 2: CVaR comparison bar chart across strategies.
 * Panel 3: Black-Litterman posterior vs prior returns.
 *
 * Data: from optimizeBookingMix(), optimizeCVaR(), blackLitterman()
 * in @omni-twin/pricing-engine.
 */

export interface RiskDashboardProps {
  /** Efficient frontier: [volatility, expectedReturn][] */
  frontier: Array<[number, number]>
  /** Current portfolio point */
  currentPortfolio?: { volatility: number; expectedReturn: number; label?: string }
  /** Optimal portfolio (max Sharpe) */
  optimalPortfolio?: { volatility: number; expectedReturn: number }
  /** Individual event types */
  eventTypes?: Array<{ name: string; volatility: number; expectedReturn: number }>
  /** CVaR comparison */
  cvarComparison?: Array<{ strategy: string; cvar: number }>
  /** Black-Litterman prior vs posterior */
  blReturns?: Array<{ name: string; prior: number; posterior: number }>
  /** Chart dimensions */
  width?: number
  height?: number
}

const PAD = { top: 32, right: 16, bottom: 40, left: 64 }

export function RiskDashboard({
  frontier,
  currentPortfolio,
  optimalPortfolio,
  eventTypes,
  cvarComparison,
  blReturns,
  width = 640,
  height = 800,
}: RiskDashboardProps) {
  const panelH = 240
  const panelW = width

  return (
    <div style={{ width }} className="space-y-2">
      <FrontierPanel
        frontier={frontier}
        currentPortfolio={currentPortfolio}
        optimalPortfolio={optimalPortfolio}
        eventTypes={eventTypes}
        width={panelW}
        height={panelH}
      />
      {cvarComparison && cvarComparison.length > 0 && (
        <CVaRPanel data={cvarComparison} width={panelW} height={panelH} />
      )}
      {blReturns && blReturns.length > 0 && (
        <BLPanel data={blReturns} width={panelW} height={panelH} />
      )}
    </div>
  )
}

// ── Panel 1: Efficient Frontier ──────────────────────────────────────

function FrontierPanel({
  frontier,
  currentPortfolio,
  optimalPortfolio,
  eventTypes,
  width,
  height,
}: {
  frontier: Array<[number, number]>
  currentPortfolio?: { volatility: number; expectedReturn: number; label?: string }
  optimalPortfolio?: { volatility: number; expectedReturn: number }
  eventTypes?: Array<{ name: string; volatility: number; expectedReturn: number }>
  width: number
  height: number
}) {
  const [hoverIdx, setHoverIdx] = useState<number | null>(null)

  const plotW = width - PAD.left - PAD.right
  const plotH = height - PAD.top - PAD.bottom

  const { xScale, yScale } = useMemo(() => {
    const allVols = [
      ...frontier.map((p) => p[0]),
      ...(eventTypes?.map((e) => e.volatility) ?? []),
      currentPortfolio?.volatility ?? 0,
      optimalPortfolio?.volatility ?? 0,
    ]
    const allRets = [
      ...frontier.map((p) => p[1]),
      ...(eventTypes?.map((e) => e.expectedReturn) ?? []),
      currentPortfolio?.expectedReturn ?? 0,
      optimalPortfolio?.expectedReturn ?? 0,
    ]
    const volMin = Math.min(...allVols.filter((v) => v > 0))
    const volMax = Math.max(...allVols)
    const retMin = Math.min(...allRets.filter((r) => r > 0))
    const retMax = Math.max(...allRets)
    const vPad = (volMax - volMin) * 0.1 || 1
    const rPad = (retMax - retMin) * 0.1 || 1

    return {
      xScale: (vol: number) =>
        PAD.left + ((vol - (volMin - vPad)) / (volMax - volMin + 2 * vPad)) * plotW,
      yScale: (ret: number) =>
        PAD.top + plotH - ((ret - (retMin - rPad)) / (retMax - retMin + 2 * rPad)) * plotH,
    }
  }, [frontier, eventTypes, currentPortfolio, optimalPortfolio, plotW, plotH])

  const frontierPath = useMemo(
    () =>
      frontier
        .map((p, i) => `${i === 0 ? 'M' : 'L'}${xScale(p[0]).toFixed(1)},${yScale(p[1]).toFixed(1)}`)
        .join(' '),
    [frontier, xScale, yScale],
  )

  const onMouseMove = useCallback(
    (e: MouseEvent<SVGSVGElement>) => {
      const rect = e.currentTarget.getBoundingClientRect()
      const mx = e.clientX - rect.left
      let closest = -1
      let closestDist = Infinity
      for (let i = 0; i < frontier.length; i++) {
        const d = Math.abs(xScale(frontier[i]![0]) - mx)
        if (d < closestDist) {
          closestDist = d
          closest = i
        }
      }
      setHoverIdx(closestDist < 20 ? closest : null)
    },
    [frontier, xScale],
  )

  return (
    <svg
      width={width}
      height={height}
      className="select-none"
      onMouseMove={onMouseMove}
      onMouseLeave={() => setHoverIdx(null)}
    >
      <text x={width / 2} y={16} textAnchor="middle" className="fill-slate-200 text-xs font-medium">
        Efficient Frontier
      </text>

      {/* Axes */}
      <line x1={PAD.left} y1={PAD.top + plotH} x2={width - PAD.right} y2={PAD.top + plotH} stroke="rgb(51,65,85)" />
      <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={PAD.top + plotH} stroke="rgb(51,65,85)" />
      <text x={width / 2} y={height - 8} textAnchor="middle" className="fill-slate-400 text-[10px]">
        Volatility ($)
      </text>
      <text
        x={14}
        y={PAD.top + plotH / 2}
        textAnchor="middle"
        className="fill-slate-400 text-[10px]"
        transform={`rotate(-90, 14, ${PAD.top + plotH / 2})`}
      >
        Expected Return ($)
      </text>

      {/* Frontier curve */}
      <path d={frontierPath} fill="none" stroke="rgb(59,130,246)" strokeWidth={2} />

      {/* Event type scatter */}
      {eventTypes?.map((et, i) => (
        <g key={i}>
          <circle
            cx={xScale(et.volatility)}
            cy={yScale(et.expectedReturn)}
            r={5}
            fill="rgba(148,163,184,0.5)"
            stroke="rgb(148,163,184)"
          />
          <text
            x={xScale(et.volatility) + 8}
            y={yScale(et.expectedReturn) + 4}
            className="fill-slate-400 text-[9px]"
          >
            {et.name}
          </text>
        </g>
      ))}

      {/* Current portfolio */}
      {currentPortfolio && (
        <g>
          <circle
            cx={xScale(currentPortfolio.volatility)}
            cy={yScale(currentPortfolio.expectedReturn)}
            r={6}
            fill="rgba(251,146,60,0.7)"
            stroke="rgb(251,146,60)"
            strokeWidth={2}
          />
          <text
            x={xScale(currentPortfolio.volatility) + 10}
            y={yScale(currentPortfolio.expectedReturn) + 4}
            className="fill-orange-400 text-[10px] font-medium"
          >
            {currentPortfolio.label ?? 'Current'}
          </text>
        </g>
      )}

      {/* Optimal portfolio */}
      {optimalPortfolio && (
        <g>
          <circle
            cx={xScale(optimalPortfolio.volatility)}
            cy={yScale(optimalPortfolio.expectedReturn)}
            r={6}
            fill="rgba(34,197,94,0.7)"
            stroke="rgb(34,197,94)"
            strokeWidth={2}
          />
          <text
            x={xScale(optimalPortfolio.volatility) + 10}
            y={yScale(optimalPortfolio.expectedReturn) + 4}
            className="fill-green-400 text-[10px] font-medium"
          >
            Optimal
          </text>
        </g>
      )}

      {/* Hover point */}
      {hoverIdx !== null && frontier[hoverIdx] && (
        <g>
          <circle
            cx={xScale(frontier[hoverIdx]![0])}
            cy={yScale(frontier[hoverIdx]![1])}
            r={4}
            fill="white"
          />
          <rect
            x={xScale(frontier[hoverIdx]![0]) + 8}
            y={yScale(frontier[hoverIdx]![1]) - 24}
            width={110}
            height={30}
            rx={4}
            fill="rgb(30,41,59)"
            stroke="rgb(71,85,105)"
          />
          <text
            x={xScale(frontier[hoverIdx]![0]) + 14}
            y={yScale(frontier[hoverIdx]![1]) - 10}
            className="fill-slate-300 text-[9px]"
          >
            Vol: {formatNum(frontier[hoverIdx]![0])}
          </text>
          <text
            x={xScale(frontier[hoverIdx]![0]) + 14}
            y={yScale(frontier[hoverIdx]![1]) + 2}
            className="fill-slate-300 text-[9px]"
          >
            Ret: {formatNum(frontier[hoverIdx]![1])}
          </text>
        </g>
      )}
    </svg>
  )
}

// ── Panel 2: CVaR Comparison ─────────────────────────────────────────

function CVaRPanel({
  data,
  width,
  height,
}: {
  data: Array<{ strategy: string; cvar: number }>
  width: number
  height: number
}) {
  const plotW = width - PAD.left - PAD.right
  const plotH = height - PAD.top - PAD.bottom

  const maxVal = Math.max(...data.map((d) => Math.abs(d.cvar)))
  const barH = Math.min(28, plotH / data.length - 4)

  return (
    <svg width={width} height={height} className="select-none">
      <text x={width / 2} y={16} textAnchor="middle" className="fill-slate-200 text-xs font-medium">
        CVaR₉₅ by Strategy
      </text>

      {data.map((d, i) => {
        const y = PAD.top + i * (plotH / data.length) + (plotH / data.length - barH) / 2
        const barWidth = maxVal > 0 ? (Math.abs(d.cvar) / maxVal) * plotW : 0
        return (
          <g key={i}>
            <text x={PAD.left - 6} y={y + barH / 2 + 4} textAnchor="end" className="fill-slate-400 text-[10px]">
              {d.strategy}
            </text>
            <rect
              x={PAD.left}
              y={y}
              width={barWidth}
              height={barH}
              rx={3}
              fill={d.cvar >= 0 ? 'rgba(34,197,94,0.6)' : 'rgba(239,68,68,0.6)'}
            />
            <text
              x={PAD.left + barWidth + 6}
              y={y + barH / 2 + 4}
              className="fill-slate-300 text-[10px]"
            >
              {formatNum(d.cvar)}
            </text>
          </g>
        )
      })}
    </svg>
  )
}

// ── Panel 3: Black-Litterman ─────────────────────────────────────────

function BLPanel({
  data,
  width,
  height,
}: {
  data: Array<{ name: string; prior: number; posterior: number }>
  width: number
  height: number
}) {
  const plotW = width - PAD.left - PAD.right
  const plotH = height - PAD.top - PAD.bottom

  const maxVal = Math.max(...data.flatMap((d) => [Math.abs(d.prior), Math.abs(d.posterior)]))
  const groupH = plotH / data.length
  const barH = Math.min(12, groupH / 2 - 2)

  return (
    <svg width={width} height={height} className="select-none">
      <text x={width / 2} y={16} textAnchor="middle" className="fill-slate-200 text-xs font-medium">
        Black-Litterman: Prior vs Posterior Returns
      </text>

      {/* Legend */}
      <rect x={width - 140} y={PAD.top} width={10} height={10} rx={2} fill="rgba(148,163,184,0.5)" />
      <text x={width - 126} y={PAD.top + 9} className="fill-slate-400 text-[9px]">Prior</text>
      <rect x={width - 80} y={PAD.top} width={10} height={10} rx={2} fill="rgba(59,130,246,0.7)" />
      <text x={width - 66} y={PAD.top + 9} className="fill-slate-400 text-[9px]">Posterior</text>

      {data.map((d, i) => {
        const groupY = PAD.top + 16 + i * groupH
        const priorW = maxVal > 0 ? (Math.abs(d.prior) / maxVal) * plotW * 0.7 : 0
        const postW = maxVal > 0 ? (Math.abs(d.posterior) / maxVal) * plotW * 0.7 : 0

        return (
          <g key={i}>
            <text x={PAD.left - 6} y={groupY + groupH / 2} textAnchor="end" className="fill-slate-400 text-[10px]">
              {d.name}
            </text>
            {/* Prior bar */}
            <rect x={PAD.left} y={groupY + groupH / 2 - barH - 1} width={priorW} height={barH} rx={2} fill="rgba(148,163,184,0.5)" />
            <text x={PAD.left + priorW + 4} y={groupY + groupH / 2 - 2} className="fill-slate-500 text-[9px]">
              {formatNum(d.prior)}
            </text>
            {/* Posterior bar */}
            <rect x={PAD.left} y={groupY + groupH / 2 + 1} width={postW} height={barH} rx={2} fill="rgba(59,130,246,0.7)" />
            <text x={PAD.left + postW + 4} y={groupY + groupH / 2 + barH + 1} className="fill-blue-400 text-[9px]">
              {formatNum(d.posterior)}
            </text>
          </g>
        )
      })}
    </svg>
  )
}

function formatNum(n: number): string {
  if (Math.abs(n) >= 1_000_000) return `$${(n / 1_000_000).toFixed(1)}M`
  if (Math.abs(n) >= 1_000) return `$${(n / 1_000).toFixed(1)}K`
  return `$${n.toFixed(0)}`
}
