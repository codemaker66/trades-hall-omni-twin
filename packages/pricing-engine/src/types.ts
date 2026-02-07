/**
 * Pricing Engine Domain Types
 *
 * Complete type definitions for the quantitative finance pricing engine:
 * stochastic models, options pricing, optimal control, order book mechanics,
 * demand models, Monte Carlo, revenue management, and portfolio theory.
 */

// ---------------------------------------------------------------------------
// SP-1: Stochastic Price Models
// ---------------------------------------------------------------------------

/** Ornstein-Uhlenbeck mean-reversion parameters */
export interface OUParams {
  /** Mean-reversion speed (half-life = ln(2)/theta) */
  theta: number
  /** Long-term mean (base fair value, log-price) */
  mu: number
  /** Diffusion volatility */
  sigma: number
  /** Fourier cosine coefficients for seasonal fair value */
  seasonalA: number[]
  /** Fourier sine coefficients for seasonal fair value */
  seasonalB: number[]
}

/** Compound Poisson jump parameters (Merton model) */
export interface JumpParams {
  /** Jump intensity (expected jumps per year) */
  lambda: number
  /** Mean jump size (log-price) */
  muJ: number
  /** Jump size volatility */
  sigmaJ: number
}

/** Geometric Brownian Motion parameters */
export interface GBMParams {
  /** Drift (annualized expected return) */
  mu: number
  /** Volatility (annualized) */
  sigma: number
}

/** Full composite venue pricing model parameters */
export interface VenuePricingModel {
  ou: OUParams
  jumps: JumpParams
}

/** Simulation configuration */
export interface SimulationConfig {
  /** Initial price */
  s0: number
  /** Time horizon in years */
  tYears: number
  /** Time step size in years */
  dt: number
  /** Number of simulation paths */
  nPaths: number
  /** Optional RNG seed for reproducibility */
  seed?: number
}

/** Result of a multi-path simulation */
export interface SimulationResult {
  /** Row-major price matrix: paths[pathIndex * (nSteps+1) + stepIndex] */
  paths: Float64Array
  /** Number of time steps per path */
  nSteps: number
  /** Number of paths */
  nPaths: number
  /** Time step size */
  dt: number
}

/** Calibration result from historical booking data */
export interface CalibrationResult {
  model: VenuePricingModel
  /** Goodness of fit metrics */
  diagnostics: {
    /** Log-likelihood of the fitted model */
    logLikelihood: number
    /** Akaike Information Criterion */
    aic: number
    /** Bayesian Information Criterion */
    bic: number
    /** Number of detected jumps */
    nJumpsDetected: number
    /** Ljung-Box p-value for residual autocorrelation */
    ljungBoxPValue: number
  }
}

// ---------------------------------------------------------------------------
// SP-2: Black-Scholes Booking Options
// ---------------------------------------------------------------------------

/** Full Black-Scholes result with all Greeks */
export interface OptionResult {
  /** Option price (premium / hold fee) */
  price: number
  /** Delta: sensitivity to underlying price */
  delta: number
  /** Gamma: rate of change of delta */
  gamma: number
  /** Theta: time decay per day */
  theta: number
  /** Vega: sensitivity to volatility (per 1% vol change) */
  vega: number
  /** Rho: sensitivity to interest rate (per 1% rate change) */
  rho: number
}

/** Option type */
export type OptionType = 'call' | 'put'

/** Implied volatility surface point */
export interface VolSurfacePoint {
  strike: number
  expiry: number
  impliedVol: number
}

// ---------------------------------------------------------------------------
// SP-3: HJB Dynamic Pricing
// ---------------------------------------------------------------------------

/** HJB solver result */
export interface HJBResult {
  /** Optimal prices: [capacity][timeStep] flattened row-major */
  optimalPrices: Float64Array
  /** Value function: [capacity][timeStep] flattened row-major */
  valueFunction: Float64Array
  /** Shadow prices (marginal value of capacity): [capacity][timeStep] */
  shadowPrices: Float64Array
  /** Number of capacity levels */
  nCapacity: number
  /** Number of time steps */
  nTimeSteps: number
  /** Number of policy iterations to convergence */
  iterations: number
}

/** HJB solver configuration */
export interface HJBConfig {
  /** Maximum venue capacity (number of slots) */
  maxCapacity: number
  /** Time horizon in days */
  timeHorizonDays: number
  /** Time step in days */
  dt: number
  /** Base demand arrival rate (bookings per day) */
  baseDemandRate: number
  /** Price sensitivity parameter alpha: D(p) = lambda * e^{-alpha*p} */
  priceSensitivity: number
  /** Seasonal demand multipliers per time step */
  seasonalFactors?: number[]
}

// ---------------------------------------------------------------------------
// SP-5: Order Book
// ---------------------------------------------------------------------------

export type OrderSide = 'bid' | 'ask'
export type OrderType = 'limit' | 'market' | 'ioc' | 'fok' | 'gtd'

export interface Order {
  id: number
  side: OrderSide
  price: number
  quantity: number
  timestamp: number
  venueId: string
  slotDate: string
  orderType: OrderType
  /** For GTD orders: expiry timestamp */
  expiresAt?: number
}

export interface Fill {
  buyOrderId: number
  sellOrderId: number
  price: number
  quantity: number
  timestamp: number
}

export interface OrderBookDepthLevel {
  price: number
  quantity: number
  orderCount: number
}

export interface OrderBookSnapshot {
  bids: OrderBookDepthLevel[]
  asks: OrderBookDepthLevel[]
  bestBid: number | null
  bestAsk: number | null
  spread: number | null
  imbalance: number
  midPrice: number | null
}

// ---------------------------------------------------------------------------
// SP-6: Demand Models
// ---------------------------------------------------------------------------

/** Hawkes process parameters */
export interface HawkesParams {
  /** Background (baseline) intensity */
  mu: number
  /** Excitation magnitude */
  alpha: number
  /** Exponential decay rate */
  beta: number
}

/** Hawkes process fit result */
export interface HawkesFitResult extends HawkesParams {
  /** Branching ratio alpha/beta — virality metric. Must be < 1 for stability */
  branchingRatio: number
  /** Half-life of excitation in the time unit of the data */
  halfLife: number
  /** Log-likelihood */
  logLikelihood: number
}

// ---------------------------------------------------------------------------
// SP-7: Monte Carlo
// ---------------------------------------------------------------------------

export interface MCConfig {
  nPaths: number
  /** Use antithetic variates (Z, -Z) pairs for variance reduction */
  useAntithetic: boolean
  /** Use control variate on expected path count */
  useControlVariate: boolean
  /** Use Sobol quasi-random sequences instead of pseudorandom */
  useSobol: boolean
  /** Confidence level for VaR/CVaR (e.g. 0.95) */
  confidenceLevel: number
  /** Optional RNG seed */
  seed?: number
}

export interface MCResult {
  mean: number
  stdError: number
  /** Value at Risk at confidence level */
  var: number
  /** Conditional Value at Risk (Expected Shortfall) */
  cvar: number
  /** [5th, 25th, 50th, 75th, 95th] percentiles */
  percentiles: [number, number, number, number, number]
  /** All terminal values for histogram */
  terminalValues: Float64Array
}

// ---------------------------------------------------------------------------
// SP-8: Revenue Management
// ---------------------------------------------------------------------------

export interface FareClass {
  name: string
  /** Revenue per booking */
  revenue: number
  /** Mean demand (bookings expected) */
  meanDemand: number
  /** Standard deviation of demand */
  stdDemand: number
}

export interface EMSRbResult {
  /** Protection level for each fare class (cumulative seats reserved) */
  protectionLevels: number[]
  /** Booking limit for each fare class */
  bookingLimits: number[]
  /** Expected revenue */
  expectedRevenue: number
}

export interface BidPriceResult {
  /** Shadow price (bid price) for each resource */
  bidPrices: Map<string, number>
  /** Which requests to accept */
  acceptedRequests: number[]
  /** Optimal LP objective value */
  optimalRevenue: number
}

export interface Resource {
  name: string
  capacity: number
}

export interface BookingRequest {
  revenue: number
  /** Resource name -> units consumed */
  resourceUsage: Record<string, number>
}

// ---------------------------------------------------------------------------
// SP-9: Portfolio Theory
// ---------------------------------------------------------------------------

export interface PortfolioResult {
  /** Optimal allocation weight per event type */
  weights: number[]
  expectedRevenue: number
  volatility: number
  sharpeRatio: number
  /** 95% Value at Risk */
  var95: number
  /** 95% Conditional Value at Risk */
  cvar95: number
  /** Efficient frontier: [volatility, return] pairs */
  efficientFrontier: Array<[number, number]>
}
