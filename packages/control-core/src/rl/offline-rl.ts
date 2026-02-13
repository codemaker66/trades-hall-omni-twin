// ---------------------------------------------------------------------------
// OC-5: Offline Reinforcement Learning (CQL, IQL, Decision Transformer)
// ---------------------------------------------------------------------------

import type { MLPWeights } from '../types.js';
import { mlpForward } from './sac.js';

// ---------------------------------------------------------------------------
// Conservative Q-Learning (CQL) Loss
// ---------------------------------------------------------------------------

/**
 * Compute the Conservative Q-Learning (CQL) regularization loss.
 *
 * CQL adds a penalty that pushes down Q-values on out-of-distribution actions
 * while keeping Q-values for in-data actions high:
 *
 *   L_CQL = alpha * ( logsumexp(Q(s, a)) - E_{a ~ data}[Q(s, a)] )
 *
 * The logsumexp over all Q-values encourages conservative estimation,
 * and the data expectation term anchors the Q-values for observed actions.
 *
 * @param qValues      Q-values for all actions at a given state (nActions).
 * @param dataActions  Indices of actions taken in the dataset (nSamples).
 * @param alpha        CQL conservatism coefficient.
 * @returns            Scalar CQL penalty loss.
 */
export function cqlLoss(
  qValues: Float64Array,
  dataActions: Int32Array,
  alpha: number,
): number {
  // logsumexp(Q) for numerical stability
  let maxQ = -Infinity;
  for (let i = 0; i < qValues.length; i++) {
    if (qValues[i]! > maxQ) {
      maxQ = qValues[i]!;
    }
  }

  let sumExp = 0;
  for (let i = 0; i < qValues.length; i++) {
    sumExp += Math.exp(qValues[i]! - maxQ);
  }
  const logsumexpQ = maxQ + Math.log(sumExp);

  // E_data[Q(s, a_data)]
  let dataQSum = 0;
  for (let i = 0; i < dataActions.length; i++) {
    const actionIdx = dataActions[i]!;
    dataQSum += qValues[actionIdx]!;
  }
  const dataQMean = dataActions.length > 0 ? dataQSum / dataActions.length : 0;

  return alpha * (logsumexpQ - dataQMean);
}

// ---------------------------------------------------------------------------
// Implicit Q-Learning (IQL) Expectile Loss
// ---------------------------------------------------------------------------

/**
 * Compute the Implicit Q-Learning (IQL) asymmetric expectile regression loss.
 *
 * IQL avoids querying out-of-distribution actions by using expectile regression:
 *
 *   L_tau(predicted, target) = E[ |tau - 1{predicted < target}| * (predicted - target)^2 ]
 *
 * When tau > 0.5, the loss penalises under-prediction more heavily,
 * effectively approximating an upper expectile of the target distribution.
 *
 * @param predicted  Predicted values (n).
 * @param target     Target values (n).
 * @param tau        Expectile parameter in (0, 1). tau > 0.5 favours upper quantiles.
 * @returns          Scalar expectile loss.
 */
export function iqlExpectileLoss(
  predicted: Float64Array,
  target: Float64Array,
  tau: number,
): number {
  const n = predicted.length;
  if (n === 0) return 0;

  let totalLoss = 0;

  for (let i = 0; i < n; i++) {
    const diff = predicted[i]! - target[i]!;
    const sqDiff = diff * diff;

    // Asymmetric weight: tau when diff >= 0 (over-prediction),
    // (1 - tau) when diff < 0 (under-prediction)
    const weight = diff >= 0 ? tau : (1 - tau);
    totalLoss += weight * sqDiff;
  }

  return totalLoss / n;
}

// ---------------------------------------------------------------------------
// Decision Transformer Forward Pass
// ---------------------------------------------------------------------------

/**
 * Decision Transformer forward pass through an MLP.
 *
 * The Decision Transformer treats RL as a sequence modelling problem.
 * Given a history of (state, action, reward-to-go) tuples, the model
 * predicts the next action.
 *
 * This simplified version concatenates the most recent sequence of
 * (s_t, a_t, R_t) vectors into a single flat input and runs it through
 * the MLP to produce the predicted action.
 *
 * @param weights    MLP weights for the decision transformer.
 * @param stateSeq   Sequence of state vectors (seqLen).
 * @param actionSeq  Sequence of action vectors (seqLen - 1, for conditioning).
 * @param rewardSeq  Reward-to-go at each timestep (seqLen).
 * @returns          Predicted next action vector.
 */
export function dtForwardPass(
  weights: MLPWeights,
  stateSeq: Float64Array[],
  actionSeq: Float64Array[],
  rewardSeq: Float64Array,
): Float64Array {
  // Compute total input length
  let totalLen = 0;
  for (let i = 0; i < stateSeq.length; i++) {
    totalLen += stateSeq[i]!.length; // state
  }
  for (let i = 0; i < actionSeq.length; i++) {
    totalLen += actionSeq[i]!.length; // action
  }
  totalLen += rewardSeq.length; // reward-to-go scalars

  // Concatenate: [r_0, s_0, a_0, r_1, s_1, a_1, ..., r_T, s_T]
  const input = new Float64Array(totalLen);
  let offset = 0;

  for (let t = 0; t < stateSeq.length; t++) {
    // Reward-to-go for this timestep
    input[offset] = rewardSeq[t]!;
    offset++;

    // State at this timestep
    const st = stateSeq[t]!;
    for (let j = 0; j < st.length; j++) {
      input[offset] = st[j]!;
      offset++;
    }

    // Action at this timestep (if available)
    if (t < actionSeq.length) {
      const at = actionSeq[t]!;
      for (let j = 0; j < at.length; j++) {
        input[offset] = at[j]!;
        offset++;
      }
    }
  }

  // Forward through MLP
  return mlpForward(weights, input);
}
