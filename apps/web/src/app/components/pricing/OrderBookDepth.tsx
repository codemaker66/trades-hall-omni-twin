'use client'

import { useMemo, useState, useCallback, type MouseEvent } from 'react'

/**
 * Order Book Depth Chart
 *
 * Bid side (green, left) and Ask side (red, right) as cumulative
 * staircase curves. Imbalance indicator bar below.
 *
 * Data: from OrderBook.depth() in @omni-twin/pricing-engine.
 */

export interface DepthLevel {
  price: number
  quantity: number
}

export interface OrderBookDepthProps {
  bids: DepthLevel[]
  asks: DepthLevel[]
  /** Chart dimensions */
  width?: number
  height?: number
  title?: string
}

const PAD = { top: 32, right: 16, bottom: 56, left: 56 }

export function OrderBookDepth({
  bids,
  asks,
  width = 640,
  height = 360,
  title = 'Order Book Depth',
}: OrderBookDepthProps) {
  const [hoverPrice, setHoverPrice] = useState<number | null>(null)

  const plotW = width - PAD.left - PAD.right
  const plotH = height - PAD.top - PAD.bottom - 24 // 24px for imbalance bar

  const { cumBids, cumAsks, priceMin, priceMax, qtyMax, xScale, yScale } = useMemo(() => {
    // Cumulative bids (highest price first → accumulate from top)
    const sortedBids = [...bids].sort((a, b) => b.price - a.price)
    const cBids: Array<{ price: number; cumQty: number }> = []
    let cumQ = 0
    for (const b of sortedBids) {
      cumQ += b.quantity
      cBids.push({ price: b.price, cumQty: cumQ })
    }
    cBids.reverse() // Now ascending by price

    // Cumulative asks (lowest price first)
    const sortedAsks = [...asks].sort((a, b) => a.price - b.price)
    const cAsks: Array<{ price: number; cumQty: number }> = []
    cumQ = 0
    for (const a of sortedAsks) {
      cumQ += a.quantity
      cAsks.push({ price: a.price, cumQty: cumQ })
    }

    const allPrices = [...cBids.map((b) => b.price), ...cAsks.map((a) => a.price)]
    const pMin = allPrices.length > 0 ? Math.min(...allPrices) : 0
    const pMax = allPrices.length > 0 ? Math.max(...allPrices) : 1
    const allQty = [...cBids.map((b) => b.cumQty), ...cAsks.map((a) => a.cumQty)]
    const qMax = allQty.length > 0 ? Math.max(...allQty) : 1

    const pPad = (pMax - pMin) * 0.05 || 1

    return {
      cumBids: cBids,
      cumAsks: cAsks,
      priceMin: pMin - pPad,
      priceMax: pMax + pPad,
      qtyMax: qMax * 1.1,
      xScale: (price: number) => PAD.left + ((price - (pMin - pPad)) / (pMax - pMin + 2 * pPad)) * plotW,
      yScale: (qty: number) => PAD.top + plotH - (qty / (qMax * 1.1)) * plotH,
    }
  }, [bids, asks, plotW, plotH])

  // Build staircase paths
  const bidPath = useMemo(() => {
    if (cumBids.length === 0) return ''
    const parts = [`M${xScale(cumBids[0]!.price).toFixed(1)},${yScale(0).toFixed(1)}`]
    for (const { price, cumQty } of cumBids) {
      parts.push(`L${xScale(price).toFixed(1)},${yScale(cumQty).toFixed(1)}`)
    }
    const last = cumBids[cumBids.length - 1]!
    parts.push(`L${xScale(last.price).toFixed(1)},${yScale(0).toFixed(1)}`)
    return parts.join(' ')
  }, [cumBids, xScale, yScale])

  const askPath = useMemo(() => {
    if (cumAsks.length === 0) return ''
    const parts = [`M${xScale(cumAsks[0]!.price).toFixed(1)},${yScale(0).toFixed(1)}`]
    for (const { price, cumQty } of cumAsks) {
      parts.push(`L${xScale(price).toFixed(1)},${yScale(cumQty).toFixed(1)}`)
    }
    const last = cumAsks[cumAsks.length - 1]!
    parts.push(`L${xScale(last.price).toFixed(1)},${yScale(0).toFixed(1)}`)
    return parts.join(' ')
  }, [cumAsks, xScale, yScale])

  // Imbalance
  const totalBidQty = cumBids.length > 0 ? cumBids[cumBids.length - 1]!.cumQty : 0
  const totalAskQty = cumAsks.length > 0 ? cumAsks[cumAsks.length - 1]!.cumQty : 0
  const imbalance = totalBidQty + totalAskQty > 0
    ? (totalBidQty - totalAskQty) / (totalBidQty + totalAskQty)
    : 0

  const onMouseMove = useCallback(
    (e: MouseEvent<SVGSVGElement>) => {
      const rect = e.currentTarget.getBoundingClientRect()
      const x = e.clientX - rect.left - PAD.left
      const price = priceMin + (x / plotW) * (priceMax - priceMin)
      setHoverPrice(price)
    },
    [priceMin, priceMax, plotW],
  )

  const spread = cumAsks.length > 0 && cumBids.length > 0
    ? cumAsks[0]!.price - cumBids[cumBids.length - 1]!.price
    : 0

  return (
    <svg
      width={width}
      height={height}
      className="select-none"
      onMouseMove={onMouseMove}
      onMouseLeave={() => setHoverPrice(null)}
    >
      <text x={width / 2} y={16} textAnchor="middle" className="fill-slate-200 text-xs font-medium">
        {title}
      </text>

      {/* Bid area */}
      <path d={bidPath} fill="rgba(34,197,94,0.2)" stroke="rgb(34,197,94)" strokeWidth={1.5} />

      {/* Ask area */}
      <path d={askPath} fill="rgba(239,68,68,0.2)" stroke="rgb(239,68,68)" strokeWidth={1.5} />

      {/* Spread label */}
      {spread > 0 && (
        <text
          x={xScale((cumBids[cumBids.length - 1]?.price ?? 0 + (cumAsks[0]?.price ?? 0)) / 2)}
          y={PAD.top + 16}
          textAnchor="middle"
          className="fill-slate-400 text-[10px]"
        >
          Spread: ${spread.toFixed(0)}
        </text>
      )}

      {/* X-axis */}
      <line
        x1={PAD.left}
        y1={PAD.top + plotH}
        x2={width - PAD.right}
        y2={PAD.top + plotH}
        stroke="rgb(51,65,85)"
      />
      <text x={width / 2} y={PAD.top + plotH + 16} textAnchor="middle" className="fill-slate-400 text-[10px]">
        Price ($)
      </text>

      {/* Y-axis */}
      <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={PAD.top + plotH} stroke="rgb(51,65,85)" />
      <text
        x={14}
        y={PAD.top + plotH / 2}
        textAnchor="middle"
        className="fill-slate-400 text-[10px]"
        transform={`rotate(-90, 14, ${PAD.top + plotH / 2})`}
      >
        Cumulative Qty
      </text>

      {/* Hover crosshair */}
      {hoverPrice !== null && (
        <line
          x1={xScale(hoverPrice)}
          y1={PAD.top}
          x2={xScale(hoverPrice)}
          y2={PAD.top + plotH}
          stroke="rgba(148,163,184,0.3)"
        />
      )}

      {/* Imbalance bar */}
      <rect
        x={PAD.left}
        y={height - 20}
        width={plotW}
        height={12}
        rx={3}
        fill="rgb(30,41,59)"
      />
      <rect
        x={PAD.left + plotW / 2}
        y={height - 20}
        width={(plotW / 2) * Math.abs(imbalance)}
        height={12}
        rx={3}
        fill={imbalance >= 0 ? 'rgba(34,197,94,0.7)' : 'rgba(239,68,68,0.7)'}
        transform={imbalance < 0 ? `translate(${-(plotW / 2) * Math.abs(imbalance)}, 0)` : undefined}
      />
      <text
        x={PAD.left + plotW / 2}
        y={height - 10}
        textAnchor="middle"
        className="fill-slate-300 text-[9px]"
      >
        Imbalance: {(imbalance * 100).toFixed(1)}%
      </text>
    </svg>
  )
}
