/**
 * @omni-twin/pricing-engine
 *
 * Quantitative finance pricing engine for venue booking optimization.
 *
 * Sub-domains:
 * - SP-1: Stochastic Price Models (GBM, OU, Jump-Diffusion, Composite)
 * - SP-2: Black-Scholes Booking Options (European/American, Greeks, Hold Fees)
 * - SP-3: HJB Dynamic Pricing (policy iteration, shadow prices)
 * - SP-4: Mean-Field Games (Picard iteration, equilibrium prices)
 * - SP-5: Order Book (limit/market orders, Kyle's lambda, Almgren-Chriss)
 * - SP-6: Hawkes Demand Models (self-exciting, NHPP, fitting)
 * - SP-7: Monte Carlo (antithetic, control variates, Sobol QMC)
 * - SP-8: Revenue Management (EMSRb, bid-price control, choice-based RM)
 * - SP-9: Portfolio Theory (Markowitz, CVaR, Black-Litterman)
 */

// Types
export type {
  OUParams,
  JumpParams,
  GBMParams,
  VenuePricingModel,
  SimulationConfig,
  SimulationResult,
  CalibrationResult,
  OptionResult,
  OptionType,
  VolSurfacePoint,
  HJBConfig,
  HJBResult,
  OrderSide,
  OrderType,
  Order,
  Fill,
  OrderBookDepthLevel,
  OrderBookSnapshot,
  HawkesParams,
  HawkesFitResult,
  MCConfig,
  MCResult,
  FareClass,
  EMSRbResult,
  BidPriceResult,
  Resource,
  BookingRequest,
  PortfolioResult,
} from './types'

// SP-1: Stochastic Price Models
export { simulateGBM, gbmExpectedValue, gbmVariance, calibrateGBM } from './gbm'
export { simulateOU, seasonalMean, calibrateOU, ouHalfLife, ouStationaryVariance } from './ou'
export {
  simulateMertonJD,
  calibrateMertonJD,
  mertonExpectedValue,
  mertonCharacteristicFunction,
} from './merton'
export {
  simulateVenuePrice,
  calibrateVenueModel,
  computePercentileBands,
  computePathStatistics,
} from './composite-model'

// SP-2: Black-Scholes
export {
  blackScholesCall,
  blackScholesPut,
  blackScholes,
  impliedVolatility,
  americanOptionBinomial,
  americanOptionWithBoundary,
  computeFloorPrice,
  computeHoldFee,
  computeVolSurface,
  normCDF,
  normPDF,
} from './black-scholes'

// SP-3: HJB Dynamic Pricing
export {
  solveHJBPricing,
  getOptimalPrice,
  getShadowPrice,
  getValueFunction,
  shouldAcceptBooking,
  generatePricingSchedule,
  solveHJBMultiSegment,
} from './hjb-solver'

// SP-4: Mean-Field Games
export { solveMFG } from './mean-field'
export type { MFGConfig, MFGResult } from './mean-field'

// SP-5: Order Book
export { OrderBook, optimalReleaseSchedule, adverseSelectionSpread } from './order-book'

// SP-6: Hawkes Demand Models
export {
  simulateHawkes,
  hawkesIntensity,
  hawkesIntensityCurve,
  fitHawkes,
  simulateNHPP,
  estimateBranchingRatio,
} from './hawkes'

// SP-7: Monte Carlo
export { simulateRevenueMC, monteCarloGeneric } from './monte-carlo'

// SP-8: Revenue Management
export {
  emsrb,
  bidPriceControl,
  choiceBasedRM,
  gallegoVanRyzinPrice,
} from './revenue-management'

// SP-9: Portfolio Theory
export {
  optimizeBookingMix,
  optimizeCVaR,
  blackLitterman,
} from './portfolio'

// Random utilities
export { Rng, SobolSequence } from './random'
