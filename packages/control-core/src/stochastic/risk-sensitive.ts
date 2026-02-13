// ---------------------------------------------------------------------------
// OC-9  Stochastic Optimal Control -- Risk-Sensitive Control
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Exponential Utility
// ---------------------------------------------------------------------------

/**
 * Exponential utility function.
 *
 *   U(c) = (1 / theta) * (1 - exp(-theta * c))
 *
 * For theta > 0, this is a concave (risk-averse) utility.
 * For theta < 0, this is convex (risk-seeking).
 * As theta -> 0, U(c) -> c (risk-neutral).
 *
 * @param cost  - The cost value
 * @param theta - Risk sensitivity parameter (non-zero)
 * @returns Exponential utility of the cost
 */
export function exponentialUtility(cost: number, theta: number): number {
  if (Math.abs(theta) < 1e-15) {
    // Risk-neutral limit
    return cost;
  }
  return (1 / theta) * (1 - Math.exp(-theta * cost));
}

// ---------------------------------------------------------------------------
// Risk-Sensitive Value Iteration
// ---------------------------------------------------------------------------

/**
 * Risk-sensitive value iteration using exponential cost criteria.
 *
 * The risk-sensitive Bellman equation minimises the expected exponential
 * cost (cost-minimisation setting):
 *
 *   V(s) = min_a [ c(s,a) + (gamma / theta) * ln( E[ exp(theta * V(s')) ] ) ]
 *
 * where the expectation is over transition probabilities P(s'|s,a).
 *
 * For theta > 0 the controller is risk-averse (penalises high-variance
 * outcomes). As theta -> 0 this reduces to the standard risk-neutral
 * Bellman equation.
 *
 * @param nStates     - Number of discrete states
 * @param nActions    - Number of discrete actions
 * @param transitions - (s, a) => array of { nextState, probability }
 * @param cost        - Stage cost c(s, a) (to be minimised)
 * @param theta       - Risk sensitivity parameter (> 0 for risk-averse)
 * @param gamma       - Discount factor in [0, 1]
 * @param tol         - Convergence tolerance (default 1e-8)
 * @param maxIter     - Maximum iterations (default 1000)
 * @returns Value function V and greedy policy
 */
export function riskSensitiveVI(
  nStates: number,
  nActions: number,
  transitions: (
    s: number,
    a: number,
  ) => Array<{ nextState: number; probability: number }>,
  cost: (s: number, a: number) => number,
  theta: number,
  gamma: number,
  tol = 1e-8,
  maxIter = 1000,
): { V: Float64Array; policy: Int32Array } {
  let V = new Float64Array(nStates);
  const policy = new Int32Array(nStates);

  // Handle near-zero theta as risk-neutral case
  const riskNeutral = Math.abs(theta) < 1e-15;

  for (let iter = 0; iter < maxIter; iter++) {
    const Vnext = new Float64Array(nStates);
    let maxDelta = 0;

    for (let s = 0; s < nStates; s++) {
      let bestValue = Infinity;
      let bestAction = 0;

      for (let a = 0; a < nActions; a++) {
        const c = cost(s, a);
        const trans = transitions(s, a);

        let futureValue: number;

        if (riskNeutral) {
          // Standard risk-neutral Bellman (cost minimisation)
          let ev = 0;
          for (const { nextState, probability } of trans) {
            ev += probability * V[nextState]!;
          }
          futureValue = c + gamma * ev;
        } else {
          // Risk-sensitive: (gamma / theta) * ln( E[ exp(theta * V(s')) ] )
          // Use log-sum-exp trick for numerical stability:
          //   ln( sum_i p_i exp(theta * V_i) )
          //     = maxV + ln( sum_i p_i exp(theta * V_i - maxV) )
          //   where maxV = theta * max_i V_i

          // Find max for numerical stability
          let maxVal = -Infinity;
          for (const { nextState } of trans) {
            const val = theta * V[nextState]!;
            if (val > maxVal) maxVal = val;
          }

          let sumExp = 0;
          for (const { nextState, probability } of trans) {
            sumExp += probability * Math.exp(theta * V[nextState]! - maxVal);
          }

          // ln(E[exp(theta*V)]) = maxVal + ln(sumExp)
          const logExpectation = maxVal + Math.log(sumExp);
          futureValue = c + (gamma / theta) * logExpectation;
        }

        if (futureValue < bestValue) {
          bestValue = futureValue;
          bestAction = a;
        }
      }

      Vnext[s] = bestValue;
      policy[s] = bestAction;

      const delta = Math.abs(bestValue - V[s]!);
      if (delta > maxDelta) maxDelta = delta;
    }

    V = Vnext;

    if (maxDelta < tol) break;
  }

  return { V, policy };
}
