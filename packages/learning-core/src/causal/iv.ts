// ---------------------------------------------------------------------------
// Instrumental Variables — Two-Stage Least Squares (2SLS)
// ---------------------------------------------------------------------------
//
// 2SLS estimates the causal effect of T on Y when T is endogenous,
// using instruments Z that affect Y only through T.
//
// Stage 1: Regress T on Z (and X if provided), get fitted values T_hat
// Stage 2: Regress Y on T_hat (and X if provided)
//
// The key assumption: Z affects Y only through T (exclusion restriction),
// and Z is correlated with T (relevance, tested via first-stage F-stat).
// ---------------------------------------------------------------------------

import type { IVResult } from '../types.js';

/**
 * Ordinary Least Squares regression.
 *
 * Solves y = X @ beta + epsilon via the normal equations:
 *   beta = (X'X)^{-1} X'y
 *
 * @param y  Dependent variable (length n)
 * @param X  Design matrix (n x p), where X[i] is the i-th row
 * @returns Coefficients, residuals, and R-squared
 */
export function ols(
  y: number[],
  X: number[][],
): { coefficients: number[]; residuals: number[]; rSquared: number } {
  const n = y.length;
  if (n === 0 || X.length === 0) {
    return { coefficients: [], residuals: [], rSquared: 0 };
  }

  const p = X[0]!.length;

  // Compute X'X (p x p)
  const XtX = new Float64Array(p * p);
  for (let i = 0; i < n; i++) {
    const xi = X[i]!;
    for (let j = 0; j < p; j++) {
      for (let k = 0; k < p; k++) {
        XtX[j * p + k] = (XtX[j * p + k] ?? 0) + (xi[j] ?? 0) * (xi[k] ?? 0);
      }
    }
  }

  // Compute X'y (p x 1)
  const Xty = new Float64Array(p);
  for (let i = 0; i < n; i++) {
    const xi = X[i]!;
    const yi = y[i] ?? 0;
    for (let j = 0; j < p; j++) {
      Xty[j] = (Xty[j] ?? 0) + (xi[j] ?? 0) * yi;
    }
  }

  // Solve (X'X) beta = X'y via Cholesky-like decomposition
  // Add small ridge for numerical stability
  for (let j = 0; j < p; j++) {
    XtX[j * p + j] = (XtX[j * p + j] ?? 0) + 1e-10;
  }

  const beta = solveSymmetricSystem(XtX, Xty, p);

  // Compute residuals and R-squared
  const residuals = new Array<number>(n);
  let ssRes = 0;
  let ySum = 0;
  for (let i = 0; i < n; i++) {
    ySum += y[i] ?? 0;
  }
  const yMean = ySum / n;

  let ssTot = 0;
  for (let i = 0; i < n; i++) {
    const xi = X[i]!;
    let yHat = 0;
    for (let j = 0; j < p; j++) {
      yHat += (xi[j] ?? 0) * (beta[j] ?? 0);
    }
    const ri = (y[i] ?? 0) - yHat;
    residuals[i] = ri;
    ssRes += ri * ri;
    const dev = (y[i] ?? 0) - yMean;
    ssTot += dev * dev;
  }

  const rSquared = ssTot > 0 ? 1 - ssRes / ssTot : 0;

  const coefficients = new Array<number>(p);
  for (let j = 0; j < p; j++) {
    coefficients[j] = beta[j] ?? 0;
  }

  return { coefficients, residuals, rSquared };
}

/**
 * Two-Stage Least Squares (2SLS) Instrumental Variables estimator.
 *
 * @param Y  Outcome variable (length n)
 * @param T  Endogenous treatment variable (length n)
 * @param Z  Instruments matrix (n x q), Z[i] is the i-th row
 * @param X  Optional exogenous covariates (n x k), X[i] is the i-th row
 * @returns IVResult with coefficient, standard error, CI, and first-stage F
 */
export function twoStageLeastSquares(
  Y: number[],
  T: number[],
  Z: number[][],
  X?: number[][],
): IVResult {
  const n = Y.length;

  if (n < 3) {
    return {
      coefficient: 0,
      standardError: Infinity,
      ciLower: -Infinity,
      ciUpper: Infinity,
      firstStageF: 0,
    };
  }

  const q = Z[0]!.length;  // Number of instruments
  const hasX = X !== undefined && X.length > 0;
  const k = hasX ? X![0]!.length : 0;

  // ---- Build first-stage design matrix: [Z, X, 1] ----
  const pFirst = q + k + 1; // instruments + covariates + intercept
  const designFirst: number[][] = new Array(n);

  for (let i = 0; i < n; i++) {
    const row = new Array<number>(pFirst);
    const zi = Z[i]!;
    for (let j = 0; j < q; j++) {
      row[j] = zi[j] ?? 0;
    }
    if (hasX) {
      const xi = X![i]!;
      for (let j = 0; j < k; j++) {
        row[q + j] = xi[j] ?? 0;
      }
    }
    row[pFirst - 1] = 1; // intercept
    designFirst[i] = row;
  }

  // ---- Stage 1: Regress T on [Z, X, 1] ----
  const stage1 = ols(T, designFirst);

  // Compute fitted values T_hat
  const tHat = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    const row = designFirst[i]!;
    let pred = 0;
    for (let j = 0; j < pFirst; j++) {
      pred += (row[j] ?? 0) * (stage1.coefficients[j] ?? 0);
    }
    tHat[i] = pred;
  }

  // ---- First-stage F-statistic ----
  // Test whether instruments Z have explanatory power for T beyond X
  // F = ((RSS_restricted - RSS_unrestricted) / q) / (RSS_unrestricted / (n - p_first))
  //
  // Restricted model: regress T on [X, 1] only (no instruments)
  let firstStageF = 0;
  {
    const pRestricted = k + 1;
    const designRestricted: number[][] = new Array(n);
    for (let i = 0; i < n; i++) {
      const row = new Array<number>(pRestricted);
      if (hasX) {
        const xi = X![i]!;
        for (let j = 0; j < k; j++) {
          row[j] = xi[j] ?? 0;
        }
      }
      row[pRestricted - 1] = 1;
      designRestricted[i] = row;
    }

    const restricted = ols(T, designRestricted);

    // RSS for restricted and unrestricted models
    let rssRestricted = 0;
    for (let i = 0; i < n; i++) {
      const r = restricted.residuals[i] ?? 0;
      rssRestricted += r * r;
    }

    let rssUnrestricted = 0;
    for (let i = 0; i < n; i++) {
      const r = stage1.residuals[i] ?? 0;
      rssUnrestricted += r * r;
    }

    const dfNum = q; // Number of instruments (excluded from restricted)
    const dfDen = n - pFirst;

    if (dfDen > 0 && rssUnrestricted > 0) {
      firstStageF = ((rssRestricted - rssUnrestricted) / dfNum) /
        (rssUnrestricted / dfDen);
    }
  }

  // ---- Stage 2: Regress Y on [T_hat, X, 1] ----
  const pSecond = 1 + k + 1; // T_hat + covariates + intercept
  const designSecond: number[][] = new Array(n);

  for (let i = 0; i < n; i++) {
    const row = new Array<number>(pSecond);
    row[0] = tHat[i] ?? 0;
    if (hasX) {
      const xi = X![i]!;
      for (let j = 0; j < k; j++) {
        row[1 + j] = xi[j] ?? 0;
      }
    }
    row[pSecond - 1] = 1; // intercept
    designSecond[i] = row;
  }

  const stage2 = ols(Y, designSecond);
  const coefficient = stage2.coefficients[0] ?? 0;

  // ---- Standard error computation ----
  // For 2SLS, the correct variance uses the original T (not T_hat) in the
  // residual computation, but T_hat in the bread matrix.
  //
  // Residuals use original T:
  //   e_i = Y_i - beta_2sls * T_i - X_i' * gamma
  const residuals2sls = new Array<number>(n);
  let ssRes2sls = 0;
  for (let i = 0; i < n; i++) {
    const xi = designSecond[i]!;
    // Replace T_hat with actual T for residual calculation
    let yHat = coefficient * (T[i] ?? 0);
    for (let j = 1; j < pSecond; j++) {
      yHat += (xi[j] ?? 0) * (stage2.coefficients[j] ?? 0);
    }
    const ri = (Y[i] ?? 0) - yHat;
    residuals2sls[i] = ri;
    ssRes2sls += ri * ri;
  }

  // sigma^2 = RSS / (n - p)
  const sigma2 = ssRes2sls / Math.max(n - pSecond, 1);

  // Variance of beta_2sls:
  // Var(beta) = sigma^2 * (Z'X * (X'PZ*X)^{-1} * X'Z)^{-1}
  // Simplified for the coefficient on T:
  // Var(beta_T) = sigma^2 / sum(T_hat_i^2 - n * T_hat_bar^2)
  //
  // More precisely, we use the sandwich form with T_hat:
  // (T_hat' M_X T_hat)^{-1} * sigma^2
  // where M_X projects out X.

  // For the coefficient on T, compute:
  // sum of (T_hat - T_hat_bar)^2, accounting for other covariates
  let tHatSum = 0;
  for (let i = 0; i < n; i++) {
    tHatSum += tHat[i] ?? 0;
  }
  const tHatMean = tHatSum / n;

  // If we have covariates, we need to partial out X from T_hat
  let tHatResidSS = 0;
  if (hasX) {
    // Regress T_hat on [X, 1] and get residuals
    const pX = k + 1;
    const designX: number[][] = new Array(n);
    for (let i = 0; i < n; i++) {
      const row = new Array<number>(pX);
      const xi = X![i]!;
      for (let j = 0; j < k; j++) {
        row[j] = xi[j] ?? 0;
      }
      row[pX - 1] = 1;
      designX[i] = row;
    }
    const tHatOnX = ols(tHat, designX);
    for (let i = 0; i < n; i++) {
      const r = tHatOnX.residuals[i] ?? 0;
      tHatResidSS += r * r;
    }
  } else {
    for (let i = 0; i < n; i++) {
      const dev = (tHat[i] ?? 0) - tHatMean;
      tHatResidSS += dev * dev;
    }
  }

  const standardError = tHatResidSS > 0
    ? Math.sqrt(sigma2 / tHatResidSS)
    : Infinity;

  // 95% confidence interval
  const z = 1.96;
  const ciLower = coefficient - z * standardError;
  const ciUpper = coefficient + z * standardError;

  return {
    coefficient,
    standardError,
    ciLower,
    ciUpper,
    firstStageF,
  };
}

// ---- Internal Helpers ----

/**
 * Solve a symmetric positive-definite system A @ x = b using
 * Cholesky decomposition (in-place, flat array layout).
 *
 * @param A Flat p*p array (column-major or row-major, symmetric so it doesn't matter)
 * @param b Right-hand side (length p)
 * @param p Dimension
 * @returns Solution vector x
 */
function solveSymmetricSystem(
  A: Float64Array,
  b: Float64Array,
  p: number,
): Float64Array {
  // Cholesky: A = L * L^T
  const L = new Float64Array(p * p);

  for (let i = 0; i < p; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = 0;
      for (let kk = 0; kk < j; kk++) {
        sum += (L[i * p + kk] ?? 0) * (L[j * p + kk] ?? 0);
      }
      if (i === j) {
        const diag = (A[i * p + i] ?? 0) - sum;
        L[i * p + j] = Math.sqrt(Math.max(diag, 1e-15));
      } else {
        const lj = L[j * p + j] ?? 1;
        L[i * p + j] = ((A[i * p + j] ?? 0) - sum) / lj;
      }
    }
  }

  // Forward substitution: L @ y = b
  const y = new Float64Array(p);
  for (let i = 0; i < p; i++) {
    let sum = 0;
    for (let j = 0; j < i; j++) {
      sum += (L[i * p + j] ?? 0) * (y[j] ?? 0);
    }
    y[i] = ((b[i] ?? 0) - sum) / (L[i * p + i] ?? 1);
  }

  // Back substitution: L^T @ x = y
  const x = new Float64Array(p);
  for (let i = p - 1; i >= 0; i--) {
    let sum = 0;
    for (let j = i + 1; j < p; j++) {
      sum += (L[j * p + i] ?? 0) * (x[j] ?? 0);
    }
    x[i] = ((y[i] ?? 0) - sum) / (L[i * p + i] ?? 1);
  }

  return x;
}
