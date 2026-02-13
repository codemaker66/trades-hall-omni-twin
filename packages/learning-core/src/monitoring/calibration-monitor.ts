// ---------------------------------------------------------------------------
// PIT Histogram Calibration Monitoring
// Probability Integral Transform for checking distributional calibration.
// ---------------------------------------------------------------------------

import type { PITHistogram } from '../types.js';

/**
 * Compute the Probability Integral Transform (PIT) values.
 *
 * For each observation y_i with CDF value F(y_i), the PIT is:
 *   u_i = F(y_i)
 *
 * If the model is well-calibrated, the PIT values should be Uniform(0,1).
 *
 * @param observations - True observed values
 * @param cdfValues - CDF values F(y_i) for each observation under the predictive distribution
 * @returns PIT values (should be ~Uniform(0,1) if calibrated)
 */
export function computePIT(observations: number[], cdfValues: number[]): number[] {
  const n = Math.min(observations.length, cdfValues.length);
  const pit: number[] = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    // Clamp CDF values to [0, 1]
    pit[i] = Math.max(0, Math.min(1, cdfValues[i] ?? 0));
  }
  return pit;
}

/**
 * Construct a PIT histogram and assess uniformity.
 *
 * Bins the PIT values into nBins equal-width bins. If the model is well-calibrated,
 * the histogram should be approximately uniform (all bins have similar counts).
 *
 * Also computes a uniformity p-value via the chi-squared test.
 *
 * @param pitValues - PIT values in [0, 1]
 * @param nBins - Number of histogram bins
 * @returns PITHistogram with bin centers, counts, and uniformity p-value
 */
export function pitHistogram(pitValues: number[], nBins: number): PITHistogram {
  if (pitValues.length === 0 || nBins <= 0) {
    return { bins: [], counts: [], uniformityPValue: 1 };
  }

  // Create bin centers
  const bins: number[] = new Array<number>(nBins);
  for (let i = 0; i < nBins; i++) {
    bins[i] = (i + 0.5) / nBins;
  }

  // Count values in each bin
  const counts: number[] = new Array<number>(nBins).fill(0);
  const n = pitValues.length;
  for (let i = 0; i < n; i++) {
    const v = pitValues[i] ?? 0;
    let bin = Math.floor(v * nBins);
    bin = Math.max(0, Math.min(nBins - 1, bin));
    counts[bin] = (counts[bin] ?? 0) + 1;
  }

  // Chi-squared test for uniformity
  // Under H0: each bin has expected count = n / nBins
  const expected = n / nBins;
  let chiSquared = 0;
  for (let i = 0; i < nBins; i++) {
    const observed = counts[i] ?? 0;
    const diff = observed - expected;
    chiSquared += (diff * diff) / expected;
  }

  // Approximate chi-squared p-value using the regularized incomplete gamma function
  // Degrees of freedom = nBins - 1
  const df = nBins - 1;
  const pValue = df > 0 ? chiSquaredPValue(chiSquared, df) : 1;

  return { bins, counts, uniformityPValue: pValue };
}

/**
 * Kolmogorov-Smirnov test against the Uniform(0,1) distribution.
 *
 * Computes the KS statistic (maximum absolute deviation between the empirical
 * CDF and the uniform CDF) and returns an approximate p-value.
 *
 * @param pitValues - PIT values in [0, 1]
 * @returns Approximate p-value for the KS test against Uniform(0,1)
 */
export function kolmogorovSmirnovTest(pitValues: number[]): number {
  const n = pitValues.length;
  if (n === 0) return 1;

  // Sort PIT values
  const sorted = [...pitValues].sort((a, b) => a - b);

  // Compute KS statistic: max |F_n(x) - x| where F_n is empirical CDF
  let ksStatistic = 0;
  for (let i = 0; i < n; i++) {
    const empiricalCdf = (i + 1) / n;
    const uniformCdf = sorted[i] ?? 0;

    // D+ = max(F_n(x) - F(x))
    const dPlus = Math.abs(empiricalCdf - uniformCdf);
    // D- = max(F(x) - F_n(x-)), where F_n(x-) = i/n
    const dMinus = Math.abs(uniformCdf - i / n);

    const d = Math.max(dPlus, dMinus);
    if (d > ksStatistic) ksStatistic = d;
  }

  // Approximate p-value using the Kolmogorov distribution asymptotic formula
  // P(D_n > x) ≈ 2 * Σ_{k=1}^∞ (-1)^{k-1} * exp(-2k²n*x²)
  // We use a finite sum approximation
  const lambda = (Math.sqrt(n) + 0.12 + 0.11 / Math.sqrt(n)) * ksStatistic;
  let pValue = 0;
  for (let k = 1; k <= 100; k++) {
    const term = Math.exp(-2 * k * k * lambda * lambda);
    if (k % 2 === 1) {
      pValue += term;
    } else {
      pValue -= term;
    }
    if (term < 1e-15) break;
  }
  pValue = Math.max(0, Math.min(1, 2 * pValue));

  return pValue;
}

// ---------------------------------------------------------------------------
// Helper: Chi-squared p-value approximation
// ---------------------------------------------------------------------------

/**
 * Approximate the upper-tail p-value of a chi-squared distribution.
 * P(X^2 > x) where X^2 ~ chi^2(df)
 *
 * Uses the regularized incomplete gamma function:
 * P(X^2 > x) = 1 - γ(df/2, x/2) / Γ(df/2)
 *            = 1 - P(df/2, x/2)
 *
 * where P(a, x) is the regularized lower incomplete gamma function.
 */
function chiSquaredPValue(x: number, df: number): number {
  if (x <= 0) return 1;
  if (df <= 0) return 0;

  const a = df / 2;
  const z = x / 2;

  // Use series expansion for regularized lower incomplete gamma function
  // P(a, z) = γ(a, z) / Γ(a)
  return 1 - regularizedGammaP(a, z);
}

/**
 * Regularized lower incomplete gamma function P(a, x) = γ(a,x) / Γ(a).
 * Uses series expansion for small x and continued fraction for large x.
 */
function regularizedGammaP(a: number, x: number): number {
  if (x < 0) return 0;
  if (x === 0) return 0;

  if (x < a + 1) {
    // Use series expansion
    return gammaPSeries(a, x);
  } else {
    // Use continued fraction (complement is more efficient)
    return 1 - gammaQCF(a, x);
  }
}

/**
 * Series expansion for regularized lower incomplete gamma function.
 * P(a, x) = e^{-x} * x^a * Σ_{n=0}^∞ x^n / (a*(a+1)*...*(a+n))  / Γ(a)
 */
function gammaPSeries(a: number, x: number): number {
  const maxIter = 200;
  const eps = 1e-12;

  let sum = 1 / a;
  let term = 1 / a;

  for (let n = 1; n < maxIter; n++) {
    term *= x / (a + n);
    sum += term;
    if (Math.abs(term) < eps * Math.abs(sum)) break;
  }

  // P(a,x) = sum * exp(-x + a*ln(x) - lnGamma(a))
  const logResult = -x + a * Math.log(x) - lnGamma(a) + Math.log(sum);
  const result = Math.exp(logResult);
  return Math.max(0, Math.min(1, result));
}

/**
 * Continued fraction for complementary regularized gamma function Q(a, x) = 1 - P(a, x).
 * Uses Lentz's algorithm.
 */
function gammaQCF(a: number, x: number): number {
  const maxIter = 200;
  const eps = 1e-12;
  const tiny = 1e-30;

  // Modified Lentz's algorithm for the continued fraction
  // Q(a,x) = e^{-x} * x^a / Γ(a) * CF
  // CF = 1/(x+1-a+ K_{n=1}^∞ n(a-n)/(x+2n+1-a))

  let f = tiny;
  let c = tiny;
  let d = 1 / (x + 1 - a);
  let h = d;

  for (let n = 1; n < maxIter; n++) {
    const an = -n * (n - a);
    const bn = x + 2 * n + 1 - a;

    d = an * d + bn;
    if (Math.abs(d) < tiny) d = tiny;
    d = 1 / d;

    c = bn + an / c;
    if (Math.abs(c) < tiny) c = tiny;

    const delta = c * d;
    h *= delta;

    if (Math.abs(delta - 1) < eps) break;
  }

  const logResult = -x + a * Math.log(x) - lnGamma(a) + Math.log(h);
  const result = Math.exp(logResult);
  return Math.max(0, Math.min(1, result));
}

/**
 * Log-gamma function using Stirling's approximation + Lanczos.
 */
function lnGamma(x: number): number {
  if (x <= 0) return Infinity;

  // Lanczos approximation (g=7, n=9)
  const g = 7;
  const coef = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
  ];

  if (x < 0.5) {
    // Reflection formula: Γ(x)Γ(1-x) = π/sin(πx)
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - lnGamma(1 - x);
  }

  const xm1 = x - 1;
  let a = coef[0] ?? 0;
  const t = xm1 + g + 0.5;

  for (let i = 1; i < coef.length; i++) {
    a += (coef[i] ?? 0) / (xm1 + i);
  }

  return 0.5 * Math.log(2 * Math.PI) + (xm1 + 0.5) * Math.log(t) - t + Math.log(a);
}
