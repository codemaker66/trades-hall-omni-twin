// ---------------------------------------------------------------------------
// OC-10: Optimal Experiment Design -- Thompson Sampling for Dynamic Pricing
// ---------------------------------------------------------------------------
//
// Thompson sampling maintains a Beta posterior over conversion rates for each
// discrete price point.  At each period:
//
// 1. Sample a conversion rate from Beta(alpha_i, beta_i) for every price i.
// 2. Compute expected revenue = price_i * sampled_rate_i.
// 3. Among prices within the rate-limit window around lastPrice, select the
//    one with the highest expected revenue.
//
// After observing a purchase/no-purchase outcome, the corresponding Beta
// posterior is updated.
// ---------------------------------------------------------------------------

import type { PRNG, ThompsonPricingConfig } from '../types.js';

// ---------------------------------------------------------------------------
// Beta distribution sampling via Joehnk's method
// ---------------------------------------------------------------------------

/**
 * Sample from a Beta(alpha, beta) distribution using the PRNG.
 *
 * Uses the ratio-of-uniforms / Joehnk's method for general (alpha, beta)
 * and a simple transformation for integer parameters.
 *
 * For the common bandit case where alpha, beta >= 1 and not too large,
 * we use the Gamma-ratio method:
 *   X ~ Gamma(alpha, 1),  Y ~ Gamma(beta, 1)  =>  X / (X+Y) ~ Beta(alpha, beta)
 *
 * Gamma samples are produced by Marsaglia and Tsang's method (simplified
 * for the PRNG interface).
 */
function sampleBeta(alpha: number, beta: number, rng: PRNG): number {
  const x = sampleGamma(alpha, rng);
  const y = sampleGamma(beta, rng);
  const sum = x + y;
  if (sum < 1e-30) return 0.5; // degenerate case
  return x / sum;
}

/**
 * Sample from Gamma(shape, 1) using Marsaglia-Tsang when shape >= 1, and
 * the Ahrens-Dieter transformation for shape < 1.
 */
function sampleGamma(shape: number, rng: PRNG): number {
  if (shape < 1) {
    // Gamma(a) = Gamma(a+1) * U^(1/a)  where U ~ Uniform(0,1)
    const g = sampleGamma(shape + 1, rng);
    const u = rng();
    return g * Math.pow(Math.max(u, 1e-30), 1 / shape);
  }

  // Marsaglia-Tsang method for shape >= 1
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);

  for (;;) {
    let x: number;
    let v: number;
    do {
      x = normalSample(rng);
      v = 1 + c * x;
    } while (v <= 0);

    v = v * v * v;
    const u = rng();

    if (u < 1 - 0.0331 * (x * x) * (x * x)) {
      return d * v;
    }
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) {
      return d * v;
    }
  }
}

/**
 * Box-Muller transform for a single standard normal sample.
 */
function normalSample(rng: PRNG): number {
  const u1 = Math.max(rng(), 1e-30);
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// ---------------------------------------------------------------------------
// Thompson price selection
// ---------------------------------------------------------------------------

/**
 * Select a price via Thompson sampling.
 *
 * For each price in the grid, a conversion rate is sampled from the Beta
 * posterior.  The expected revenue at each price is  price * sampled_rate.
 * Subject to the rate-limit constraint (|price - lastPrice| <= rateLimit),
 * the price with the highest expected revenue is returned.
 *
 * @param config    Thompson pricing configuration
 * @param lastPrice The price set in the previous period
 * @param rng       Seedable PRNG
 * @returns Selected price and its expected revenue
 */
export function thompsonPriceSelect(
  config: ThompsonPricingConfig,
  lastPrice: number,
  rng: PRNG,
): { price: number; expectedRevenue: number } {
  const { priceGrid, priorAlpha, priorBeta, priceBounds, rateLimit } = config;
  const nPrices = priceGrid.length;

  let bestPrice = priceBounds.min;
  let bestRevenue = -Infinity;

  for (let i = 0; i < nPrices; i++) {
    const price = priceGrid[i]!;

    // Enforce bounds
    if (price < priceBounds.min || price > priceBounds.max) continue;

    // Enforce rate-limit constraint
    if (Math.abs(price - lastPrice) > rateLimit + 1e-12) continue;

    // Sample conversion rate from Beta posterior
    const alpha = priorAlpha[i]!;
    const beta = priorBeta[i]!;
    const sampledRate = sampleBeta(alpha, beta, rng);

    // Expected revenue = price * conversion rate
    const revenue = price * sampledRate;

    if (revenue > bestRevenue) {
      bestRevenue = revenue;
      bestPrice = price;
    }
  }

  return {
    price: bestPrice,
    expectedRevenue: Math.max(0, bestRevenue),
  };
}

// ---------------------------------------------------------------------------
// Thompson pricing Bayesian update
// ---------------------------------------------------------------------------

/**
 * Update the Beta posterior for a price point after observing whether a
 * customer purchased at that price.
 *
 * @param config    Current Thompson pricing configuration
 * @param priceIdx  Index into the priceGrid of the offered price
 * @param purchased Whether the customer bought (true) or not (false)
 * @returns New ThompsonPricingConfig with updated posteriors
 */
export function thompsonPriceUpdate(
  config: ThompsonPricingConfig,
  priceIdx: number,
  purchased: boolean,
): ThompsonPricingConfig {
  const newAlpha = new Float64Array(config.priorAlpha);
  const newBeta = new Float64Array(config.priorBeta);

  if (purchased) {
    newAlpha[priceIdx] = newAlpha[priceIdx]! + 1;
  } else {
    newBeta[priceIdx] = newBeta[priceIdx]! + 1;
  }

  return {
    priceGrid: config.priceGrid,
    priorAlpha: newAlpha,
    priorBeta: newBeta,
    priceBounds: config.priceBounds,
    rateLimit: config.rateLimit,
  };
}
