// ---------------------------------------------------------------------------
// Uplift Modeling — T-Learner (Kunzel et al. 2019)
// ---------------------------------------------------------------------------
//
// The T-learner (Two-learner) approach estimates heterogeneous treatment
// effects by training separate models for the treatment and control groups:
//
//   mu_0(x) = E[Y | T=0, X=x]   (control model)
//   mu_1(x) = E[Y | T=1, X=x]   (treatment model)
//
// The individual treatment effect (ITE) or uplift is:
//   tau(x) = mu_1(x) - mu_0(x)
//
// The Average Treatment Effect (ATE) is:
//   ATE = (1/n) sum_i tau(x_i)
//
// This is the simplest meta-learner approach. It works well when
// treatment and control distributions are similar and sample sizes
// are reasonably balanced.
// ---------------------------------------------------------------------------

/**
 * Estimate uplift (heterogeneous treatment effects) using the T-learner.
 *
 * @param Y    Outcome variable (length n)
 * @param T    Binary treatment indicator (0 = control, 1 = treatment)
 * @param X    Covariate matrix (n x p), X[i] is the i-th row
 * @param predictControl   Function that predicts E[Y|T=0, X] for given X rows.
 *                          The caller is responsible for training the model on
 *                          the control subset before calling this.
 * @param predictTreatment Function that predicts E[Y|T=1, X] for given X rows.
 *                          The caller is responsible for training the model on
 *                          the treatment subset before calling this.
 * @returns Object with per-unit uplift scores and the average treatment effect
 */
export function tLearnerEstimate(
  Y: number[],
  T: number[],
  X: number[][],
  predictControl: (X: number[][]) => number[],
  predictTreatment: (X: number[][]) => number[],
): { uplift: number[]; ate: number } {
  const n = Y.length;

  if (n === 0) {
    return { uplift: [], ate: 0 };
  }

  // ---- Split data by treatment assignment ----
  const controlIndices: number[] = [];
  const treatmentIndices: number[] = [];

  for (let i = 0; i < n; i++) {
    if ((T[i] ?? 0) === 0) {
      controlIndices.push(i);
    } else {
      treatmentIndices.push(i);
    }
  }

  // ---- Extract subsets ----
  const xControl: number[][] = new Array(controlIndices.length);
  for (let j = 0; j < controlIndices.length; j++) {
    xControl[j] = X[controlIndices[j]!]!;
  }

  const xTreatment: number[][] = new Array(treatmentIndices.length);
  for (let j = 0; j < treatmentIndices.length; j++) {
    xTreatment[j] = X[treatmentIndices[j]!]!;
  }

  // ---- Build combined input for prediction ----
  // We need predictions for ALL n units from both models
  // The predictControl and predictTreatment functions are assumed to
  // have been trained on their respective subsets. We pass the full
  // covariate matrix to get predictions for every unit.

  // However, to support the common pattern where the prediction function
  // trains internally, we pass the training data first followed by
  // the full data. The predictions for the full data are at the end.
  //
  // Actually, the simplest and most standard approach: the caller
  // passes functions that already know how to predict. We just pass X.

  // Predict E[Y|T=0, X] and E[Y|T=1, X] for all units
  const mu0 = predictControl(X);
  const mu1 = predictTreatment(X);

  // ---- Compute uplift = mu1(x) - mu0(x) ----
  const uplift = new Array<number>(n);
  let upliftSum = 0;

  for (let i = 0; i < n; i++) {
    const tau = (mu1[i] ?? 0) - (mu0[i] ?? 0);
    uplift[i] = tau;
    upliftSum += tau;
  }

  const ate = upliftSum / n;

  return { uplift, ate };
}
