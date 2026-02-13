// ---------------------------------------------------------------------------
// Elastic Weight Consolidation (Kirkpatrick et al. 2017)
// ---------------------------------------------------------------------------
//
// EWC prevents catastrophic forgetting in continual learning by adding a
// regularization term that penalizes changes to parameters that were
// important for previous tasks. Importance is measured by the diagonal
// of the Fisher information matrix, approximated as the mean of squared
// gradients over the previous task's data.
//
// The EWC loss is:
//   L_EWC(theta) = L_new(theta) + (lambda / 2) * sum_i F_i * (theta_i - theta*_i)^2
//
// where:
//   L_new    = loss on the new task
//   theta*   = parameters after training on the old task
//   F_i      = diagonal Fisher information for parameter i
//   lambda   = importance weight
// ---------------------------------------------------------------------------

/**
 * Compute the diagonal of the Fisher information matrix.
 *
 * The Fisher information diagonal is approximated as the mean of squared
 * gradients over a dataset from the previous task:
 *
 *   F_i = (1/N) sum_{n=1}^{N} (dL/d_theta_i | x_n)^2
 *
 * This is the empirical Fisher, which serves as a good approximation
 * when gradients are computed at the MAP/MLE estimate.
 *
 * @param gradients Array of gradient vectors, one per data point.
 *                  gradients[n][i] = partial derivative of loss w.r.t. param i
 *                  for data point n.
 * @returns Diagonal Fisher information vector of length d (number of parameters)
 */
export function computeFisherDiagonal(gradients: number[][]): number[] {
  const n = gradients.length;
  if (n === 0) return [];

  const d = gradients[0]!.length;
  const fisher = new Array<number>(d);

  for (let i = 0; i < d; i++) {
    fisher[i] = 0;
  }

  // Accumulate squared gradients
  for (let sample = 0; sample < n; sample++) {
    const grad = gradients[sample]!;
    for (let i = 0; i < d; i++) {
      const g = grad[i] ?? 0;
      fisher[i] = (fisher[i] ?? 0) + g * g;
    }
  }

  // Average over samples
  for (let i = 0; i < d; i++) {
    fisher[i] = (fisher[i] ?? 0) / n;
  }

  return fisher;
}

/**
 * Compute the EWC loss.
 *
 * L_EWC = L_new + (lambda / 2) * sum_i F_i * (theta_i - theta*_i)^2
 *
 * @param params    Current parameter vector theta
 * @param oldParams Previous task's optimal parameters theta*
 * @param fisher    Diagonal Fisher information F_i
 * @param lambda    EWC regularization strength
 * @param taskLoss  Loss on the new task L_new(theta)
 * @returns Total EWC loss
 */
export function ewcLoss(
  params: Float64Array,
  oldParams: Float64Array,
  fisher: Float64Array,
  lambda: number,
  taskLoss: number,
): number {
  const d = params.length;
  let penalty = 0;

  for (let i = 0; i < d; i++) {
    const diff = (params[i] ?? 0) - (oldParams[i] ?? 0);
    penalty += (fisher[i] ?? 0) * diff * diff;
  }

  return taskLoss + (lambda / 2) * penalty;
}

/**
 * Compute the gradient of the EWC loss.
 *
 * dL_EWC/d_theta_i = dL_new/d_theta_i + lambda * F_i * (theta_i - theta*_i)
 *
 * @param params       Current parameter vector theta
 * @param oldParams    Previous task's optimal parameters theta*
 * @param fisher       Diagonal Fisher information F_i
 * @param lambda       EWC regularization strength
 * @param taskGradient Gradient of the new task loss dL_new/d_theta
 * @returns Gradient of the full EWC loss
 */
export function ewcGradient(
  params: Float64Array,
  oldParams: Float64Array,
  fisher: Float64Array,
  lambda: number,
  taskGradient: Float64Array,
): Float64Array {
  const d = params.length;
  const grad = new Float64Array(d);

  for (let i = 0; i < d; i++) {
    const taskGrad = taskGradient[i] ?? 0;
    const fisherI = fisher[i] ?? 0;
    const diff = (params[i] ?? 0) - (oldParams[i] ?? 0);
    grad[i] = taskGrad + lambda * fisherI * diff;
  }

  return grad;
}
