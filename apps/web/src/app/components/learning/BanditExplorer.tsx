'use client'

import { useMemo, useState, useCallback, type MouseEvent } from 'react'

/**
 * Bandit Explorer
 *
 * Live multi-armed bandit visualization:
 * - Top: horizontal bar chart of arm mean rewards with UCB error bars
 * - Bottom: cumulative regret line chart
 */

export interface BanditExplorerProps {
  arms: Array<{ index: number; pulls: number; meanReward: number; ucb?: number }>
  cumulativeRegret: number[]
  title?: string
  width?: number
  height?: number
}

const PAD = { top: 32, right: 20, bottom: 32, left: 56 }
const ARM_H = 28

export function BanditExplorer({
  arms,
  cumulativeRegret,
  title = 'Bandit Explorer',
  width = 580,
  height: heightProp,
}: BanditExplorerProps) {
  const [hoverArm, setHoverArm] = useState<number | null>(null)
  const [hoverRegretIdx, setHoverRegretIdx] = useState<number | null>(null)

  const barAreaH = arms.length * ARM_H + 16
  const regretAreaTop = PAD.top + barAreaH + 24
  const height = heightProp ?? regretAreaTop + 140 + PAD.bottom
  const regretH = height - regretAreaTop - PAD.bottom
  const plotW = width - PAD.left - PAD.right

  // Best arm
  const bestIdx = useMemo(
    () => arms.reduce((best, arm) => arm.meanReward > (arms[best]?.meanReward ?? -Infinity) ? arm.index : best, 0),
    [arms],
  )

  // Arm bar scales
  const { maxVal, barScale } = useMemo(() => {
    const allVals = arms.flatMap(a => [a.meanReward, a.ucb ?? a.meanReward])
    const mx = Math.max(0.01, ...allVals)
    return {
      maxVal: mx,
      barScale: (v: number) => (v / mx) * plotW,
    }
  }, [arms, plotW])

  // Regret scales
  const { regretXScale, regretYScale, regretMax } = useMemo(() => {
    if (cumulativeRegret.length === 0) {
      return { regretXScale: () => 0, regretYScale: () => 0, regretMax: 1 }
    }
    const mx = Math.max(1, ...cumulativeRegret)
    return {
      regretXScale: (i: number) => PAD.left + (i / Math.max(1, cumulativeRegret.length - 1)) * plotW,
      regretYScale: (v: number) => regretAreaTop + regretH - (v / mx) * regretH,
      regretMax: mx,
    }
  }, [cumulativeRegret, plotW, regretAreaTop, regretH])

  const regretPath = useMemo(() => {
    if (cumulativeRegret.length === 0) return ''
    return cumulativeRegret.map((v, i) =>
      `${i === 0 ? 'M' : 'L'}${regretXScale(i).toFixed(1)},${regretYScale(v).toFixed(1)}`
    ).join(' ')
  }, [cumulativeRegret, regretXScale, regretYScale])

  const onRegretMove = useCallback(
    (e: MouseEvent<SVGRectElement>) => {
      if (cumulativeRegret.length === 0) return
      const rect = e.currentTarget.getBoundingClientRect()
      const mx = e.clientX - rect.left
      const idx = Math.round((mx / plotW) * (cumulativeRegret.length - 1))
      setHoverRegretIdx(idx >= 0 && idx < cumulativeRegret.length ? idx : null)
    },
    [cumulativeRegret, plotW],
  )

  return (
    <div className="rounded-lg bg-slate-900 p-3 inline-block">
      <svg width={width} height={height} className="select-none">
        <text x={width / 2} y={18} textAnchor="middle" className="fill-slate-200 text-xs font-medium">
          {title}
        </text>

        {/* Arm bars */}
        {arms.map((arm, i) => {
          const y = PAD.top + i * ARM_H
          const bw = barScale(arm.meanReward)
          const isBest = arm.index === bestIdx
          const isHovered = hoverArm === arm.index

          return (
            <g
              key={arm.index}
              onMouseEnter={() => setHoverArm(arm.index)}
              onMouseLeave={() => setHoverArm(null)}
            >
              {/* Label */}
              <text x={PAD.left - 8} y={y + ARM_H * 0.6} textAnchor="end" className="fill-slate-300 text-[11px]">
                Arm {arm.index}
              </text>

              {/* Bar */}
              <rect
                x={PAD.left} y={y + 4}
                width={Math.max(1, bw)} height={ARM_H - 10}
                rx={3}
                fill={isBest ? '#22c55e' : '#3b82f6'}
                opacity={isHovered ? 1 : 0.75}
              />

              {/* Best badge */}
              {isBest && (
                <text x={PAD.left + bw + 4} y={y + ARM_H * 0.35} className="fill-green-400 text-[9px] font-medium">
                  BEST
                </text>
              )}

              {/* UCB error bar */}
              {arm.ucb !== undefined && arm.ucb > arm.meanReward && (
                <g>
                  <line
                    x1={PAD.left + bw}
                    y1={y + ARM_H / 2 - 1}
                    x2={PAD.left + barScale(arm.ucb)}
                    y2={y + ARM_H / 2 - 1}
                    stroke="rgba(148,163,184,0.5)"
                    strokeWidth={1.5}
                  />
                  <line
                    x1={PAD.left + barScale(arm.ucb)}
                    y1={y + 6}
                    x2={PAD.left + barScale(arm.ucb)}
                    y2={y + ARM_H - 8}
                    stroke="rgba(148,163,184,0.5)"
                    strokeWidth={1.5}
                  />
                </g>
              )}

              {/* Pull count + value */}
              <text
                x={PAD.left + Math.max(bw, arm.ucb !== undefined ? barScale(arm.ucb) : bw) + (isBest ? 36 : 4)}
                y={y + ARM_H * 0.6}
                className="fill-slate-400 text-[10px]"
              >
                {arm.meanReward.toFixed(3)} ({arm.pulls} pulls)
              </text>
            </g>
          )
        })}

        {/* Regret section divider */}
        <line x1={PAD.left} y1={regretAreaTop - 12} x2={PAD.left + plotW} y2={regretAreaTop - 12} stroke="rgb(51,65,85)" />
        <text x={PAD.left} y={regretAreaTop - 2} className="fill-slate-400 text-[10px] font-medium">
          Cumulative Regret
        </text>

        {/* Regret Y-axis */}
        {Array.from({ length: 4 }, (_, i) => {
          const v = (regretMax * i) / 3
          return (
            <g key={i}>
              <line x1={PAD.left} y1={regretYScale(v)} x2={PAD.left + plotW} y2={regretYScale(v)} stroke="rgb(51,65,85)" strokeDasharray="2,3" />
              <text x={PAD.left - 6} y={regretYScale(v) + 4} textAnchor="end" className="fill-slate-400 text-[10px]">
                {v.toFixed(1)}
              </text>
            </g>
          )
        })}

        {/* Regret line */}
        <path d={regretPath} fill="none" stroke="#ef4444" strokeWidth={1.5} />

        {/* Regret hover area */}
        <rect
          x={PAD.left} y={regretAreaTop}
          width={plotW} height={regretH}
          fill="transparent"
          onMouseMove={onRegretMove}
          onMouseLeave={() => setHoverRegretIdx(null)}
        />

        {/* Regret hover crosshair */}
        {hoverRegretIdx !== null && cumulativeRegret[hoverRegretIdx] !== undefined && (
          <g>
            <line x1={regretXScale(hoverRegretIdx)} y1={regretAreaTop} x2={regretXScale(hoverRegretIdx)} y2={regretAreaTop + regretH} stroke="rgba(148,163,184,0.4)" />
            <circle cx={regretXScale(hoverRegretIdx)} cy={regretYScale(cumulativeRegret[hoverRegretIdx]!)} r={3} fill="#ef4444" stroke="#e2e8f0" strokeWidth={1} />
            <rect x={regretXScale(hoverRegretIdx) + 8} y={regretYScale(cumulativeRegret[hoverRegretIdx]!) - 14} width={72} height={20} rx={4} fill="rgb(30,41,59)" stroke="rgb(51,65,85)" />
            <text x={regretXScale(hoverRegretIdx) + 14} y={regretYScale(cumulativeRegret[hoverRegretIdx]!) + 1} className="fill-slate-300 text-[10px]">
              {cumulativeRegret[hoverRegretIdx]!.toFixed(2)}
            </text>
          </g>
        )}

        {/* X-axis label */}
        <text x={PAD.left + plotW / 2} y={height - 8} textAnchor="middle" className="fill-slate-400 text-[10px]">
          Step
        </text>
      </svg>
    </div>
  )
}
