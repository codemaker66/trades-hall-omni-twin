/**
 * Limit Order Book Engine for Venue Availability Market
 *
 * A price-time priority order book that organizes venue supply and demand
 * transparently — like a financial exchange but for venue slots.
 *
 * Features:
 * - Limit and market orders with price-time priority matching
 * - IOC (Immediate-or-Cancel) and FOK (Fill-or-Kill) order types
 * - Order book imbalance: (Q_bid - Q_ask) / (Q_bid + Q_ask)
 * - Kyle's lambda: price impact per booking
 * - Almgren-Chriss optimal inventory release schedule
 * - Glosten-Milgrom spread for platform commission
 *
 * References:
 * - Kyle (1985). "Continuous Auctions and Insider Trading"
 * - Almgren & Chriss (2001). "Optimal Execution of Portfolio Transactions"
 * - Glosten & Milgrom (1985). "Bid, Ask and Transaction Prices"
 */

import type {
  Order,
  OrderSide,
  Fill,
  OrderBookDepthLevel,
  OrderBookSnapshot,
} from './types'

/**
 * Price-time priority limit order book.
 *
 * Bids sorted descending by price (best bid first).
 * Asks sorted ascending by price (best ask first).
 * Within same price level: FIFO (first in, first out).
 */
export class OrderBook {
  /** Bids ordered by price (descending), then time (ascending) */
  private bids: Order[] = []
  /** Asks ordered by price (ascending), then time (ascending) */
  private asks: Order[] = []
  /** All executed trades */
  private trades: Fill[] = []
  /** Next order ID */
  private nextId = 1

  /**
   * Submit a limit order. Returns any fills from crossing.
   */
  addLimitOrder(
    side: OrderSide,
    price: number,
    quantity: number,
    venueId: string,
    slotDate: string,
  ): { orderId: number; fills: Fill[] } {
    const order: Order = {
      id: this.nextId++,
      side,
      price,
      quantity,
      timestamp: Date.now(),
      venueId,
      slotDate,
      orderType: 'limit',
    }

    const fills = this.matchOrder(order)

    // If remaining quantity, add to book
    if (order.quantity > 0) {
      this.insertOrder(order)
    }

    return { orderId: order.id, fills }
  }

  /**
   * Submit a market order. Executes immediately at best available price.
   * Any unfilled quantity is cancelled.
   */
  addMarketOrder(
    side: OrderSide,
    quantity: number,
    venueId: string,
    slotDate: string,
  ): { orderId: number; fills: Fill[] } {
    const order: Order = {
      id: this.nextId++,
      side,
      price: side === 'bid' ? Infinity : 0,
      quantity,
      timestamp: Date.now(),
      venueId,
      slotDate,
      orderType: 'market',
    }

    const fills = this.matchOrder(order)
    // Market order: unfilled quantity is cancelled (not added to book)

    return { orderId: order.id, fills }
  }

  /**
   * Submit an IOC (Immediate-or-Cancel) order.
   * Fills whatever is possible immediately, cancels the rest.
   */
  addIOCOrder(
    side: OrderSide,
    price: number,
    quantity: number,
    venueId: string,
    slotDate: string,
  ): { orderId: number; fills: Fill[] } {
    const order: Order = {
      id: this.nextId++,
      side,
      price,
      quantity,
      timestamp: Date.now(),
      venueId,
      slotDate,
      orderType: 'ioc',
    }

    const fills = this.matchOrder(order)
    // IOC: remaining quantity is cancelled

    return { orderId: order.id, fills }
  }

  /**
   * Submit a FOK (Fill-or-Kill) order.
   * Must be completely filled or completely cancelled.
   */
  addFOKOrder(
    side: OrderSide,
    price: number,
    quantity: number,
    venueId: string,
    slotDate: string,
  ): { orderId: number; fills: Fill[]; filled: boolean } {
    // Check if enough liquidity exists
    const availableQty = this.availableLiquidity(
      side === 'bid' ? 'ask' : 'bid',
      price,
      side === 'bid',
    )

    if (availableQty < quantity) {
      return { orderId: this.nextId++, fills: [], filled: false }
    }

    const { orderId, fills } = this.addIOCOrder(side, price, quantity, venueId, slotDate)
    return { orderId, fills, filled: true }
  }

  /**
   * Cancel an existing order by ID.
   */
  cancelOrder(orderId: number): boolean {
    let idx = this.bids.findIndex((o) => o.id === orderId)
    if (idx >= 0) {
      this.bids.splice(idx, 1)
      return true
    }

    idx = this.asks.findIndex((o) => o.id === orderId)
    if (idx >= 0) {
      this.asks.splice(idx, 1)
      return true
    }

    return false
  }

  /** Best bid price (highest buy order) */
  bestBid(): number | null {
    return this.bids.length > 0 ? this.bids[0]!.price : null
  }

  /** Best ask price (lowest sell order) */
  bestAsk(): number | null {
    return this.asks.length > 0 ? this.asks[0]!.price : null
  }

  /** Mid price: (best_bid + best_ask) / 2 */
  midPrice(): number | null {
    const bid = this.bestBid()
    const ask = this.bestAsk()
    if (bid === null || ask === null) return null
    return (bid + ask) / 2
  }

  /** Bid-ask spread */
  spread(): number | null {
    const bid = this.bestBid()
    const ask = this.bestAsk()
    if (bid === null || ask === null) return null
    return ask - bid
  }

  /**
   * Order book imbalance: (Q_bid - Q_ask) / (Q_bid + Q_ask)
   *
   * Positive = more demand than supply → price should rise
   * Negative = more supply than demand → price should fall
   *
   * THE single most actionable signal for dynamic pricing.
   */
  imbalance(): number {
    const bidQty = this.bids.reduce((s, o) => s + o.quantity, 0)
    const askQty = this.asks.reduce((s, o) => s + o.quantity, 0)
    const total = bidQty + askQty
    if (total === 0) return 0
    return (bidQty - askQty) / total
  }

  /**
   * Kyle's lambda: price impact per unit of order flow.
   *
   * λ = ΔPrice / ΔOrderFlow
   *
   * Estimated from recent trades using OLS regression of
   * price changes on signed order flow.
   *
   * High λ = illiquid market, few substitutes
   * Low λ = competitive market, many alternatives
   */
  kyleLambda(): number {
    if (this.trades.length < 5) return 0

    // Use last 50 trades
    const recent = this.trades.slice(-50)

    // Compute price changes and signed order flow
    const priceChanges: number[] = []
    const orderFlow: number[] = []

    for (let i = 1; i < recent.length; i++) {
      priceChanges.push(recent[i]!.price - recent[i - 1]!.price)
      // Positive = buy-initiated, negative = sell-initiated
      orderFlow.push(recent[i]!.quantity)
    }

    if (priceChanges.length === 0) return 0

    // OLS: ΔP = λ·Q + ε
    let sumXY = 0
    let sumXX = 0
    for (let i = 0; i < priceChanges.length; i++) {
      sumXY += orderFlow[i]! * priceChanges[i]!
      sumXX += orderFlow[i]! * orderFlow[i]!
    }

    return sumXX > 0 ? sumXY / sumXX : 0
  }

  /**
   * Get order book depth at N price levels.
   */
  depth(nLevels: number = 10): OrderBookSnapshot {
    const bidLevels = this.aggregateLevels(this.bids, nLevels)
    const askLevels = this.aggregateLevels(this.asks, nLevels)

    return {
      bids: bidLevels,
      asks: askLevels,
      bestBid: this.bestBid(),
      bestAsk: this.bestAsk(),
      spread: this.spread(),
      imbalance: this.imbalance(),
      midPrice: this.midPrice(),
    }
  }

  /** Get all executed trades */
  getTrades(): Fill[] {
    return [...this.trades]
  }

  /** Total bid volume */
  totalBidVolume(): number {
    return this.bids.reduce((s, o) => s + o.quantity, 0)
  }

  /** Total ask volume */
  totalAskVolume(): number {
    return this.asks.reduce((s, o) => s + o.quantity, 0)
  }

  // ---------------------------------------------------------------------------
  // Private methods
  // ---------------------------------------------------------------------------

  private matchOrder(order: Order): Fill[] {
    const fills: Fill[] = []
    const book = order.side === 'bid' ? this.asks : this.bids

    while (order.quantity > 0 && book.length > 0) {
      const best = book[0]!

      // Check price compatibility
      if (order.side === 'bid' && order.price < best.price) break
      if (order.side === 'ask' && order.price > best.price) break

      // Execute fill
      const fillQty = Math.min(order.quantity, best.quantity)
      const fill: Fill = {
        buyOrderId: order.side === 'bid' ? order.id : best.id,
        sellOrderId: order.side === 'ask' ? order.id : best.id,
        price: best.price, // Passive order determines price
        quantity: fillQty,
        timestamp: Date.now(),
      }

      fills.push(fill)
      this.trades.push(fill)

      order.quantity -= fillQty
      best.quantity -= fillQty

      // Remove exhausted order from book
      if (best.quantity <= 0) {
        book.shift()
      }
    }

    return fills
  }

  private insertOrder(order: Order): void {
    if (order.side === 'bid') {
      // Insert maintaining descending price, ascending time
      const idx = this.bids.findIndex(
        (o) => o.price < order.price || (o.price === order.price && o.timestamp > order.timestamp),
      )
      if (idx === -1) {
        this.bids.push(order)
      } else {
        this.bids.splice(idx, 0, order)
      }
    } else {
      // Insert maintaining ascending price, ascending time
      const idx = this.asks.findIndex(
        (o) => o.price > order.price || (o.price === order.price && o.timestamp > order.timestamp),
      )
      if (idx === -1) {
        this.asks.push(order)
      } else {
        this.asks.splice(idx, 0, order)
      }
    }
  }

  private availableLiquidity(
    side: OrderSide,
    priceLimit: number,
    isBuyer: boolean,
  ): number {
    const book = side === 'bid' ? this.bids : this.asks
    let qty = 0
    for (const order of book) {
      if (isBuyer && order.price > priceLimit) break
      if (!isBuyer && order.price < priceLimit) break
      qty += order.quantity
    }
    return qty
  }

  private aggregateLevels(orders: Order[], nLevels: number): OrderBookDepthLevel[] {
    const levels: OrderBookDepthLevel[] = []
    let currentPrice: number | null = null
    let currentQty = 0
    let currentCount = 0

    for (const order of orders) {
      if (currentPrice !== null && order.price !== currentPrice) {
        levels.push({ price: currentPrice, quantity: currentQty, orderCount: currentCount })
        if (levels.length >= nLevels) break
        currentQty = 0
        currentCount = 0
      }
      currentPrice = order.price
      currentQty += order.quantity
      currentCount++
    }

    if (currentPrice !== null && levels.length < nLevels) {
      levels.push({ price: currentPrice, quantity: currentQty, orderCount: currentCount })
    }

    return levels
  }
}

/**
 * Almgren-Chriss optimal inventory release schedule.
 *
 * Determines how a venue should release slots over time to minimize
 * market impact + timing risk.
 *
 * x_j = sinh(κ(T - t_j)) / sinh(κT) * X₀
 *
 * where:
 * - κ = √(risk_aversion * volatility² / market_impact)
 * - High κ (volatile demand): front-load release
 * - Low κ (stable demand): distribute evenly
 *
 * @param totalSlots - Total inventory to release
 * @param nPeriods - Number of release periods
 * @param riskAversion - How risk-averse the venue is (higher = more front-loading)
 * @param volatility - Demand volatility
 * @param marketImpact - Estimated price impact per slot released
 */
export function optimalReleaseSchedule(
  totalSlots: number,
  nPeriods: number,
  riskAversion: number,
  volatility: number,
  marketImpact: number,
): Array<{ period: number; slotsToRelease: number; cumulativeReleased: number }> {
  // κ = √(risk_aversion * σ² / η)
  const kappa = Math.sqrt((riskAversion * volatility * volatility) / Math.max(marketImpact, 1e-10))
  const T = nPeriods

  const schedule: Array<{ period: number; slotsToRelease: number; cumulativeReleased: number }> = []
  let cumulativeReleased = 0

  for (let j = 0; j < nPeriods; j++) {
    // Remaining inventory trajectory: x_j = sinh(κ(T-j)) / sinh(κT) * X₀
    const xCurrent = (Math.sinh(kappa * (T - j)) / Math.sinh(kappa * T)) * totalSlots
    const xNext = (Math.sinh(kappa * (T - j - 1)) / Math.sinh(kappa * T)) * totalSlots

    // Slots to release this period
    const release = Math.max(0, Math.round(xCurrent - xNext))
    cumulativeReleased += release

    schedule.push({
      period: j,
      slotsToRelease: release,
      cumulativeReleased: Math.min(cumulativeReleased, totalSlots),
    })
  }

  return schedule
}

/**
 * Glosten-Milgrom adverse selection spread.
 *
 * The bid-ask spread compensates for adverse selection — planners who
 * know about unannounced events booking before prices rise.
 *
 * spread = 2 * π * (V_high - V_low)
 *
 * where:
 * - π = fraction of informed traders
 * - V_high = venue value if demand is high
 * - V_low = venue value if demand is low
 *
 * Platform commission should cover at least the adverse selection cost.
 */
export function adverseSelectionSpread(
  informedFraction: number,
  valueHigh: number,
  valueLow: number,
): number {
  return 2 * informedFraction * (valueHigh - valueLow)
}
