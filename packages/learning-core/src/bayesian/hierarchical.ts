// ---------------------------------------------------------------------------
// Hierarchical Bayesian Model with Gibbs Sampling
// ---------------------------------------------------------------------------

import type {
  PRNG,
  HierarchicalConfig,
  HierarchicalPosterior,
} from '../types.js';
import { normalSample, gammaSample } from '../types.js';

/**
 * Gibbs sampler for a two-level hierarchical normal model.
 *
 * Model specification:
 *   Global:
 *     mu_global     ~ Normal(0, 1000)
 *     sigma_venue   ~ HalfNormal(100)
 *
 *   Group level (j = 1..J):
 *     mu_j          ~ Normal(mu_global, sigma_venue)
 *
 *   Observation level:
 *     y_ij          ~ Normal(mu_j + X_i * beta, sigma_obs)
 *
 * Gibbs sweeps sample each parameter from its full conditional:
 *   1. mu_global   | rest  (Normal conjugacy)
 *   2. sigma_venue | rest  (Inverse-Gamma conjugacy)
 *   3. mu_j        | rest  for each group (Normal conjugacy)
 *   4. beta        | rest  (Multivariate Normal conjugacy)
 *   5. sigma_obs   | rest  (Inverse-Gamma conjugacy)
 *
 * @param groupIds  Group assignment for each observation (0-indexed)
 * @param X         Feature matrix (n x d)
 * @param y         Target vector (n)
 * @param config    Sampler configuration
 * @param rng       Seedable PRNG
 * @returns Posterior samples for all parameters
 */
export function gibbsSampleHierarchical(
  groupIds: number[],
  X: number[][],
  y: number[],
  config: HierarchicalConfig,
  rng: PRNG,
): HierarchicalPosterior {
  const n = y.length;
  const d = (X[0] ?? []).length;
  const J = config.nGroups;
  const totalIter = config.tuneSamples + config.nSamples;

  // -----------------------------------------------------------------------
  // Index observations by group
  // -----------------------------------------------------------------------
  const groupIndices: number[][] = Array.from({ length: J }, () => []);
  for (let i = 0; i < n; i++) {
    const gid = groupIds[i] ?? 0;
    if (gid >= 0 && gid < J) {
      groupIndices[gid]!.push(i);
    }
  }

  // -----------------------------------------------------------------------
  // Initialise parameters
  // -----------------------------------------------------------------------
  let muGlobal = 0;
  let sigmaVenue = 1;
  const muGroup = new Array<number>(J).fill(0);
  const beta = new Array<number>(d).fill(0);
  let sigmaObs = 1;

  // Prior hyperparameters
  const priorMuVar = 1000 * 1000; // variance of global mean prior
  const priorSigmaVenueScale = 100; // HalfNormal scale for sigma_venue

  // Storage for posterior samples
  const samples = new Map<string, number[]>();
  samples.set('mu_global', []);
  samples.set('sigma_venue', []);
  samples.set('sigma_obs', []);
  for (let j = 0; j < J; j++) {
    samples.set(`mu_group_${j}`, []);
  }
  for (let k = 0; k < d; k++) {
    samples.set(`beta_${k}`, []);
  }

  // -----------------------------------------------------------------------
  // Gibbs iterations
  // -----------------------------------------------------------------------
  for (let iter = 0; iter < totalIter; iter++) {
    const tau2 = sigmaObs * sigmaObs; // observation variance
    const tauV2 = sigmaVenue * sigmaVenue; // venue-level variance

    // -------------------------------------------------------------------
    // 1. Sample mu_global | rest
    //    Conjugate Normal: prior N(0, priorMuVar), likelihood from mu_j
    // -------------------------------------------------------------------
    {
      const priorPrec = 1 / priorMuVar;
      const likPrec = J / Math.max(tauV2, 1e-12);
      const postPrec = priorPrec + likPrec;
      const postVar = 1 / postPrec;

      let sumMu = 0;
      for (let j = 0; j < J; j++) {
        sumMu += muGroup[j] ?? 0;
      }
      const postMean = postVar * (likPrec * (sumMu / Math.max(J, 1)));
      muGlobal = postMean + Math.sqrt(postVar) * normalSample(rng);
    }

    // -------------------------------------------------------------------
    // 2. Sample sigma_venue | rest
    //    We parameterise sigma_venue^2 ~ InverseGamma.
    //    With HalfNormal(0, priorSigmaVenueScale) prior on sigma_venue,
    //    we approximate the conditional using a Gibbs step:
    //    sigma_venue^2 | mu_j, mu_global ~ IG(a, b) where
    //      a = J/2, b = 0.5 * sum((mu_j - mu_global)^2)
    //    (proper IG posterior from flat/reference prior on variance)
    // -------------------------------------------------------------------
    {
      let ss = 0;
      for (let j = 0; j < J; j++) {
        const diff = (muGroup[j] ?? 0) - muGlobal;
        ss += diff * diff;
      }
      // IG(shape, rate) => sample 1/Gamma(shape, 1/rate)
      const shape = (J + 1) / 2; // +1 from half-normal prior
      const rate = 0.5 * ss + 1 / (2 * priorSigmaVenueScale * priorSigmaVenueScale);
      const sigmaVenue2 = 1 / gammaSample(shape, 1 / Math.max(rate, 1e-12), rng);
      sigmaVenue = Math.sqrt(Math.max(sigmaVenue2, 1e-12));
    }

    // -------------------------------------------------------------------
    // 3. Sample mu_j | rest for each group
    //    Conjugate Normal:
    //      prior: N(mu_global, sigma_venue^2)
    //      likelihood: observations in group j
    //      residual_ij = y_ij - X_i * beta
    // -------------------------------------------------------------------
    {
      const tauV2Now = sigmaVenue * sigmaVenue;
      for (let j = 0; j < J; j++) {
        const idxs = groupIndices[j]!;
        const nj = idxs.length;

        // Sum of residuals in this group
        let sumResid = 0;
        for (const i of idxs) {
          const xi = X[i]!;
          let xb = 0;
          for (let k = 0; k < d; k++) {
            xb += (xi[k] ?? 0) * (beta[k] ?? 0);
          }
          sumResid += (y[i] ?? 0) - xb;
        }

        const priorPrec = 1 / Math.max(tauV2Now, 1e-12);
        const likPrec = nj / Math.max(tau2, 1e-12);
        const postPrec = priorPrec + likPrec;
        const postVar = 1 / postPrec;
        const postMean =
          postVar *
          (priorPrec * muGlobal + likPrec * (sumResid / Math.max(nj, 1)));

        muGroup[j] = postMean + Math.sqrt(postVar) * normalSample(rng);
      }
    }

    // -------------------------------------------------------------------
    // 4. Sample beta | rest
    //    Conjugate Normal (ridge-like):
    //      prior: N(0, 100*I)
    //      residual_i = y_i - mu_{g_i}
    //      likelihood: residual_i ~ N(X_i * beta, sigma_obs^2)
    //
    //    For simplicity we use coordinate-wise Gibbs (one beta_k at a time).
    // -------------------------------------------------------------------
    {
      const priorVarBeta = 100 * 100; // prior variance per coordinate
      for (let k = 0; k < d; k++) {
        // Compute partial residuals: r_i = y_i - mu_{g_i} - sum_{l!=k} X_il * beta_l
        let sumXR = 0;
        let sumX2 = 0;
        for (let i = 0; i < n; i++) {
          const xi = X[i]!;
          const gid = groupIds[i] ?? 0;
          let xb = 0;
          for (let l = 0; l < d; l++) {
            if (l !== k) {
              xb += (xi[l] ?? 0) * (beta[l] ?? 0);
            }
          }
          const ri = (y[i] ?? 0) - (muGroup[gid] ?? 0) - xb;
          const xik = xi[k] ?? 0;
          sumXR += xik * ri;
          sumX2 += xik * xik;
        }

        const priorPrec = 1 / priorVarBeta;
        const likPrec = sumX2 / Math.max(tau2, 1e-12);
        const postPrec = priorPrec + likPrec;
        const postVar = 1 / postPrec;
        const postMean = postVar * (sumXR / Math.max(tau2, 1e-12));

        beta[k] = postMean + Math.sqrt(postVar) * normalSample(rng);
      }
    }

    // -------------------------------------------------------------------
    // 5. Sample sigma_obs | rest
    //    sigma_obs^2 ~ IG(a, b)
    //    a = n/2, b = 0.5 * sum((y_i - mu_{g_i} - X_i * beta)^2)
    // -------------------------------------------------------------------
    {
      let ss = 0;
      for (let i = 0; i < n; i++) {
        const xi = X[i]!;
        const gid = groupIds[i] ?? 0;
        let xb = 0;
        for (let k = 0; k < d; k++) {
          xb += (xi[k] ?? 0) * (beta[k] ?? 0);
        }
        const resid = (y[i] ?? 0) - (muGroup[gid] ?? 0) - xb;
        ss += resid * resid;
      }
      const shape = n / 2;
      const rate = 0.5 * ss;
      const sigmaObs2 = 1 / gammaSample(Math.max(shape, 0.01), 1 / Math.max(rate, 1e-12), rng);
      sigmaObs = Math.sqrt(Math.max(sigmaObs2, 1e-12));
    }

    // -------------------------------------------------------------------
    // Store samples (post-burnin)
    // -------------------------------------------------------------------
    if (iter >= config.tuneSamples) {
      samples.get('mu_global')!.push(muGlobal);
      samples.get('sigma_venue')!.push(sigmaVenue);
      samples.get('sigma_obs')!.push(sigmaObs);
      for (let j = 0; j < J; j++) {
        samples.get(`mu_group_${j}`)!.push(muGroup[j] ?? 0);
      }
      for (let k = 0; k < d; k++) {
        samples.get(`beta_${k}`)!.push(beta[k] ?? 0);
      }
    }
  }

  // -----------------------------------------------------------------------
  // Summarise posterior
  // -----------------------------------------------------------------------
  const muGlobalSamples = samples.get('mu_global')!;
  const sigmaVenueSamples = samples.get('sigma_venue')!;
  const sigmaObsSamples = samples.get('sigma_obs')!;

  const groupMeans: number[] = [];
  const groupStds: number[] = [];
  for (let j = 0; j < J; j++) {
    const s = samples.get(`mu_group_${j}`)!;
    groupMeans.push(arrayMean(s));
    groupStds.push(arrayStd(s));
  }

  return {
    globalMean: arrayMean(muGlobalSamples),
    globalStd: arrayStd(muGlobalSamples),
    groupMeans,
    groupStds,
    observationNoise: arrayMean(sigmaObsSamples),
    samples,
  };
}

/**
 * Posterior predictive distribution for a new observation in a given group.
 *
 * Samples from:
 *   y_new | posterior ~ Normal(mu_j + x * beta, sigma_obs)
 *
 * by drawing from the posterior samples and computing mean + std of the
 * resulting predictive distribution.
 */
export function posteriorPredictive(
  posterior: HierarchicalPosterior,
  groupId: number,
  x: number[],
  rng: PRNG,
): { mean: number; std: number } {
  const groupSamples = posterior.samples.get(`mu_group_${groupId}`);
  const sigmaObsSamples = posterior.samples.get('sigma_obs');

  if (!groupSamples || !sigmaObsSamples || groupSamples.length === 0) {
    return { mean: 0, std: 1 };
  }

  const nSamples = groupSamples.length;
  const d = x.length;
  const predictions: number[] = [];

  for (let s = 0; s < nSamples; s++) {
    const muJ = groupSamples[s] ?? 0;
    const sigmaObs = sigmaObsSamples[s] ?? 1;

    // Compute x * beta using posterior sample s
    let xb = 0;
    for (let k = 0; k < d; k++) {
      const betaSamples = posterior.samples.get(`beta_${k}`);
      const betaK = betaSamples ? (betaSamples[s] ?? 0) : 0;
      xb += (x[k] ?? 0) * betaK;
    }

    // Sample from predictive: y_new ~ Normal(mu_j + x*beta, sigma_obs)
    const pred = muJ + xb + sigmaObs * normalSample(rng);
    predictions.push(pred);
  }

  return {
    mean: arrayMean(predictions),
    std: arrayStd(predictions),
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function arrayMean(arr: number[]): number {
  if (arr.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < arr.length; i++) {
    sum += arr[i] ?? 0;
  }
  return sum / arr.length;
}

function arrayStd(arr: number[]): number {
  if (arr.length < 2) return 0;
  const m = arrayMean(arr);
  let sumSq = 0;
  for (let i = 0; i < arr.length; i++) {
    const diff = (arr[i] ?? 0) - m;
    sumSq += diff * diff;
  }
  return Math.sqrt(sumSq / (arr.length - 1));
}
