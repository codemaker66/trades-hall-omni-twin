import { describe, it, expect } from 'vitest'
import { OrderBook, optimalReleaseSchedule, adverseSelectionSpread } from '../order-book'

describe('SP-5: Order Book', () => {
  it('creates an empty order book', () => {
    const book = new OrderBook()
    expect(book.bestBid()).toBeNull()
    expect(book.bestAsk()).toBeNull()
    expect(book.spread()).toBeNull()
    expect(book.imbalance()).toBe(0)
  })

  it('adds limit orders and maintains price-time priority', () => {
    const book = new OrderBook()

    book.addLimitOrder('bid', 100, 5, 'venue1', '2025-06-01')
    book.addLimitOrder('bid', 105, 3, 'venue1', '2025-06-01')
    book.addLimitOrder('ask', 110, 4, 'venue1', '2025-06-01')
    book.addLimitOrder('ask', 108, 2, 'venue1', '2025-06-01')

    expect(book.bestBid()).toBe(105)
    expect(book.bestAsk()).toBe(108)
    expect(book.spread()).toBe(3)
  })

  it('matches crossing orders', () => {
    const book = new OrderBook()

    book.addLimitOrder('ask', 100, 5, 'venue1', '2025-06-01')
    const { fills } = book.addLimitOrder('bid', 100, 3, 'venue1', '2025-06-01')

    expect(fills.length).toBe(1)
    expect(fills[0]!.price).toBe(100) // Passive order price
    expect(fills[0]!.quantity).toBe(3)
  })

  it('partially fills orders', () => {
    const book = new OrderBook()

    book.addLimitOrder('ask', 100, 2, 'venue1', '2025-06-01')
    const { fills } = book.addLimitOrder('bid', 100, 5, 'venue1', '2025-06-01')

    expect(fills.length).toBe(1)
    expect(fills[0]!.quantity).toBe(2)
    // Remaining 3 units should be on the bid side
    expect(book.bestBid()).toBe(100)
    expect(book.totalBidVolume()).toBe(3)
  })

  it('market order fills immediately', () => {
    const book = new OrderBook()

    book.addLimitOrder('ask', 100, 5, 'venue1', '2025-06-01')
    book.addLimitOrder('ask', 105, 5, 'venue1', '2025-06-01')

    const { fills } = book.addMarketOrder('bid', 7, 'venue1', '2025-06-01')

    expect(fills.length).toBe(2)
    expect(fills[0]!.price).toBe(100)
    expect(fills[0]!.quantity).toBe(5)
    expect(fills[1]!.price).toBe(105)
    expect(fills[1]!.quantity).toBe(2)
  })

  it('IOC order fills partially and cancels rest', () => {
    const book = new OrderBook()

    book.addLimitOrder('ask', 100, 2, 'venue1', '2025-06-01')
    const { fills } = book.addIOCOrder('bid', 100, 5, 'venue1', '2025-06-01')

    expect(fills.length).toBe(1)
    expect(fills[0]!.quantity).toBe(2)
    // No remaining order on the book
    expect(book.bestBid()).toBeNull()
  })

  it('FOK order fills completely or not at all', () => {
    const book = new OrderBook()

    book.addLimitOrder('ask', 100, 2, 'venue1', '2025-06-01')

    // FOK for 5 — not enough liquidity, should not fill
    const { fills, filled } = book.addFOKOrder('bid', 100, 5, 'venue1', '2025-06-01')
    expect(filled).toBe(false)
    expect(fills.length).toBe(0)

    // FOK for 2 — enough liquidity
    const result2 = book.addFOKOrder('bid', 100, 2, 'venue1', '2025-06-01')
    expect(result2.filled).toBe(true)
    expect(result2.fills.length).toBe(1)
  })

  it('cancels orders', () => {
    const book = new OrderBook()

    const { orderId } = book.addLimitOrder('bid', 100, 5, 'venue1', '2025-06-01')
    expect(book.bestBid()).toBe(100)

    const cancelled = book.cancelOrder(orderId)
    expect(cancelled).toBe(true)
    expect(book.bestBid()).toBeNull()
  })

  it('computes imbalance correctly', () => {
    const book = new OrderBook()

    book.addLimitOrder('bid', 100, 10, 'venue1', '2025-06-01')
    book.addLimitOrder('ask', 105, 5, 'venue1', '2025-06-01')

    // (10 - 5) / (10 + 5) = 0.333
    expect(book.imbalance()).toBeCloseTo(1 / 3, 3)
  })

  it('computes mid price', () => {
    const book = new OrderBook()

    book.addLimitOrder('bid', 100, 5, 'venue1', '2025-06-01')
    book.addLimitOrder('ask', 110, 5, 'venue1', '2025-06-01')

    expect(book.midPrice()).toBe(105)
  })

  it('computes depth', () => {
    const book = new OrderBook()

    book.addLimitOrder('bid', 100, 5, 'venue1', '2025-06-01')
    book.addLimitOrder('bid', 100, 3, 'venue1', '2025-06-01')
    book.addLimitOrder('bid', 95, 2, 'venue1', '2025-06-01')
    book.addLimitOrder('ask', 105, 4, 'venue1', '2025-06-01')

    const snapshot = book.depth(5)
    expect(snapshot.bids.length).toBe(2) // Two bid price levels
    expect(snapshot.bids[0]!.price).toBe(100)
    expect(snapshot.bids[0]!.quantity).toBe(8) // 5 + 3
    expect(snapshot.bids[0]!.orderCount).toBe(2)
    expect(snapshot.asks.length).toBe(1)
    expect(snapshot.imbalance).toBeGreaterThan(0) // More bid volume
  })

  it('tracks executed trades', () => {
    const book = new OrderBook()

    book.addLimitOrder('ask', 100, 5, 'venue1', '2025-06-01')
    book.addLimitOrder('bid', 100, 3, 'venue1', '2025-06-01')
    book.addLimitOrder('ask', 105, 2, 'venue1', '2025-06-01')
    book.addLimitOrder('bid', 105, 1, 'venue1', '2025-06-01')

    const trades = book.getTrades()
    expect(trades.length).toBe(2)
  })

  // -----------------------------------------------------------------------
  // Almgren-Chriss Release Schedule
  // -----------------------------------------------------------------------
  describe('Optimal Release Schedule', () => {
    it('releases all slots over the horizon', () => {
      const schedule = optimalReleaseSchedule(20, 10, 1.0, 0.3, 0.5)
      const totalReleased = schedule.reduce((s, p) => s + p.slotsToRelease, 0)
      // Should release approximately all 20 slots
      expect(totalReleased).toBeGreaterThanOrEqual(18) // Allow rounding
      expect(totalReleased).toBeLessThanOrEqual(22)
    })

    it('high risk aversion front-loads releases', () => {
      const conservative = optimalReleaseSchedule(20, 10, 5.0, 0.3, 0.5)
      const aggressive = optimalReleaseSchedule(20, 10, 0.1, 0.3, 0.5)

      // Conservative should release more in early periods
      const conservativeEarly = conservative.slice(0, 3).reduce((s, p) => s + p.slotsToRelease, 0)
      const aggressiveEarly = aggressive.slice(0, 3).reduce((s, p) => s + p.slotsToRelease, 0)
      expect(conservativeEarly).toBeGreaterThanOrEqual(aggressiveEarly)
    })
  })

  // -----------------------------------------------------------------------
  // Glosten-Milgrom Spread
  // -----------------------------------------------------------------------
  describe('Adverse Selection Spread', () => {
    it('spread is proportional to informed fraction', () => {
      const spread10 = adverseSelectionSpread(0.1, 120, 80)
      const spread30 = adverseSelectionSpread(0.3, 120, 80)
      expect(spread30).toBeCloseTo(3 * spread10, 2)
    })

    it('spread increases with value difference', () => {
      const spreadNarrow = adverseSelectionSpread(0.2, 105, 95)
      const spreadWide = adverseSelectionSpread(0.2, 150, 50)
      expect(spreadWide).toBeGreaterThan(spreadNarrow)
    })
  })
})
