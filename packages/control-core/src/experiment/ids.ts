// ---------------------------------------------------------------------------
// OC-10: Optimal Experiment Design -- Information-Directed Sampling (IDS)
// ---------------------------------------------------------------------------
//
// IDS selects arms (actions) to minimise the *information ratio*:
//
//   Psi_i = Delta_i^2 / g_i
//
// where Delta_i is the expected (instantaneous) regret and g_i is the
// information gain from observing arm i.  Arms with low Psi are preferred
// because they provide a favourable trade-off between regret and learning.
//
// The prior on each arm's reward probability is a Beta distribution,
// suitable for Bernoulli bandits.
// ---------------------------------------------------------------------------

import type { PRNG, IDSConfig, IDSResult } from '../types.js';

// ---------------------------------------------------------------------------
// IDS arm selection
// ---------------------------------------------------------------------------

/**
 * Select an arm according to Information-Directed Sampling.
 *
 * For each arm i with Beta(alpha_i, beta_i) prior:
 *   - mean_i = alpha_i / (alpha_i + beta_i)
 *   - Delta_i = max(means) - mean_i            (expected regret)
 *   - g_i = KL divergence from prior to posterior after a single observation
 *           (averaged over the Bernoulli outcome).
 *   - Psi_i = Delta_i^2 / g_i                  (information ratio)
 *
 * The arm is sampled proportional to 1 / Psi_i (lower ratio = higher
 * probability).  Ties and edge cases (Delta=0 or g=0) are handled
 * gracefully.
 *
 * @param config  IDS configuration with Beta priors per arm
 * @param rng     Seedable PRNG
 * @returns IDSResult with per-arm statistics and sampling probabilities
 */
export function idsSelect(config: IDSConfig, rng: PRNG): IDSResult {
  const { nArms, priorAlpha, priorBeta } = config;

  const means = new Float64Array(nArms);
  const expectedRegret = new Float64Array(nArms);
  const informationGain = new Float64Array(nArms);
  const idsRatio = new Float64Array(nArms);
  const armProbabilities = new Float64Array(nArms);

  // -------------------------------------------------------------------------
  // Compute per-arm means
  // -------------------------------------------------------------------------
  let maxMean = -Infinity;
  for (let i = 0; i < nArms; i++) {
    const a = priorAlpha[i]!;
    const b = priorBeta[i]!;
    means[i] = a / (a + b);
    if (means[i]! > maxMean) {
      maxMean = means[i]!;
    }
  }

  // -------------------------------------------------------------------------
  // Compute per-arm regret, information gain, and IDS ratio
  // -------------------------------------------------------------------------
  let sumInvPsi = 0;
  const invPsi = new Float64Array(nArms);

  for (let i = 0; i < nArms; i++) {
    const a = priorAlpha[i]!;
    const b = priorBeta[i]!;
    const mean = means[i]!;

    // Expected instantaneous regret
    expectedRegret[i] = maxMean - mean;

    // Information gain: expected KL from prior to posterior for a Bernoulli
    // observation of arm i.
    //
    // If reward = 1 (prob = mean):  posterior is Beta(a+1, b)
    // If reward = 0 (prob = 1-mean): posterior is Beta(a, b+1)
    //
    // g_i = mean * KL(Beta(a+1,b) || Beta(a,b))
    //      + (1-mean) * KL(Beta(a,b+1) || Beta(a,b))
    //
    // Using the KL divergence formula for Beta distributions:
    //   KL(Beta(a',b') || Beta(a,b)) = lnB(a,b) - lnB(a',b')
    //       + (a'-a) psi(a') + (b'-b) psi(b')
    //       - (a'+b'-a-b) psi(a'+b')
    //
    // For the special case of incrementing alpha by 1:
    //   KL(Beta(a+1,b) || Beta(a,b)) = ln(a/(a+b)) + (b/(a+b))*( psi(a+b+1) - psi(b) )
    //   ... which simplifies (using psi(x+1) = psi(x) + 1/x) to:
    //   = ln((a+b)/(a)) - b/((a+b)*(a+b+1)) - ...
    //
    // We use a simpler approximation that is accurate for moderate a, b:
    //   g_i approx = mean * ln((a+1)/(a)) * (a/(a+b))
    //              + (1-mean) * ln((b+1)/(b)) * (b/(a+b))
    //
    // Better approximation using digamma differences:
    //   KL(Beta(a+1,b) || Beta(a,b)) ~ 1/(2*a*(a+b))  for large a,b
    //
    // We use the exact formula with log-Beta ratios:
    //   KL(Beta(a+1,b) || Beta(a,b)) = lnGamma(a+b) - lnGamma(a+b+1)
    //       + lnGamma(a+1) - lnGamma(a)
    //       + (a+1 - a)*psi(a+1) + (b-b)*psi(b) - (a+b+1-a-b)*psi(a+b+1)
    //   = -ln(a+b) + ln(a) + psi(a+1) - psi(a+b+1)
    //   = -ln(a+b) + ln(a) + 1/a - 1/(a+b)     (using psi(x+1)=psi(x)+1/x)
    //   ... but psi isn't trivially available.
    //
    // Practical approximation: use the log-ratio of Beta normalising constants.
    //   KL1 = lnBeta(a,b) - lnBeta(a+1,b) + (digamma(a+1) - digamma(a+b+1))
    //
    // Simplest accurate approximation for Bernoulli bandits:
    //   g_i = mean * [psi(a+1) - psi(a+b+1)] + (1-mean) * [psi(b+1) - psi(a+b+1)]
    //       + psi(a+b+1) - mean*psi(a) - (1-mean)*psi(b)
    //
    // We use:  g_i = 1 / (2 * (a + b)^2)  as a well-known O(1/n^2) approx
    // for the expected information gain of a Bernoulli observation with
    // Beta prior.  This is the reciprocal of the Fisher information for the
    // Beta-Bernoulli model: Var(theta) ~ 1/((a+b)^2 * (a+b+1)).
    //
    // More precisely:
    //   g_i = mean * ln((a+b)/a * (a+1)/(a+b+1))
    //       + (1-mean) * ln((a+b)/b * (b+1)/(a+b+1))
    const ab = a + b;
    const kl1 = Math.log((a + 1) / a) + Math.log(ab / (ab + 1)); // KL for reward=1
    const kl0 = Math.log((b + 1) / b) + Math.log(ab / (ab + 1)); // KL for reward=0

    const gi = mean * Math.max(0, kl1) + (1 - mean) * Math.max(0, kl0);
    informationGain[i] = Math.max(gi, 1e-15);

    // IDS ratio
    const delta = expectedRegret[i]!;
    if (delta < 1e-15) {
      // This arm is the best arm -- its regret is 0.
      // Give it a small ratio (high probability of being selected).
      idsRatio[i] = 1e-15;
      invPsi[i] = 1e15;
    } else {
      idsRatio[i] = (delta * delta) / informationGain[i]!;
      invPsi[i] = 1.0 / idsRatio[i]!;
    }
    sumInvPsi += invPsi[i]!;
  }

  // -------------------------------------------------------------------------
  // Compute sampling probabilities proportional to 1/Psi
  // -------------------------------------------------------------------------
  if (sumInvPsi > 0) {
    for (let i = 0; i < nArms; i++) {
      armProbabilities[i] = invPsi[i]! / sumInvPsi;
    }
  } else {
    // Uniform fallback
    for (let i = 0; i < nArms; i++) {
      armProbabilities[i] = 1 / nArms;
    }
  }

  // -------------------------------------------------------------------------
  // Sample an arm according to the probabilities (for side-effect logging;
  // the caller can also use armProbabilities directly).
  // We embed the sampled arm index inside armProbabilities by convention:
  // the result struct focuses on the distribution, not a single draw.
  // -------------------------------------------------------------------------
  // (The caller draws from armProbabilities using rng if needed.)
  // Consume one rng draw so the PRNG advances deterministically.
  rng();

  return {
    armProbabilities,
    expectedRegret,
    informationGain,
    idsRatio,
  };
}

// ---------------------------------------------------------------------------
// IDS Bayesian update
// ---------------------------------------------------------------------------

/**
 * Bayesian update of the Beta priors after observing a Bernoulli reward from
 * a chosen arm.
 *
 * @param config  Current IDS configuration
 * @param arm     Index of the arm that was pulled
 * @param reward  Observed reward (0 or 1)
 * @returns New IDSConfig with updated priors
 */
export function idsUpdate(
  config: IDSConfig,
  arm: number,
  reward: number,
): IDSConfig {
  const newAlpha = new Float64Array(config.priorAlpha);
  const newBeta = new Float64Array(config.priorBeta);

  if (reward > 0.5) {
    // Success: alpha += 1
    newAlpha[arm] = newAlpha[arm]! + 1;
  } else {
    // Failure: beta += 1
    newBeta[arm] = newBeta[arm]! + 1;
  }

  return {
    nArms: config.nArms,
    priorAlpha: newAlpha,
    priorBeta: newBeta,
  };
}
