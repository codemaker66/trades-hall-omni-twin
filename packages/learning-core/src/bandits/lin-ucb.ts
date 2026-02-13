// ---------------------------------------------------------------------------
// LinUCB (Li et al. 2010) — Contextual Bandits
// ---------------------------------------------------------------------------

import type { Matrix } from '../types.js';
import { choleskySolve, dot, matGet, matSet } from '../types.js';

/**
 * LinUCB contextual bandit algorithm.
 *
 * For each arm a, maintains:
 *   A_a — d x d matrix (initialized to identity)
 *   b_a — d-dimensional vector (initialized to zero)
 *
 * At each step, given context x:
 *   theta_a = A_a^{-1} b_a  (solved via Cholesky)
 *   UCB_a = theta_a^T x + alpha * sqrt(x^T A_a^{-1} x)
 *
 * Select arm with highest UCB_a.
 */
export class LinUCB {
  readonly nArms: number;
  readonly d: number;
  readonly alpha: number;

  /** A_a matrices: one d x d matrix per arm */
  private readonly A: Matrix[];
  /** b_a vectors: one d-vector per arm */
  private readonly b: Float64Array[];

  constructor(nArms: number, d: number, alpha: number) {
    this.nArms = nArms;
    this.d = d;
    this.alpha = alpha;

    // Initialize A_a = I_d for each arm
    this.A = new Array<Matrix>(nArms);
    this.b = new Array<Float64Array>(nArms);
    for (let a = 0; a < nArms; a++) {
      const data = new Float64Array(d * d);
      for (let i = 0; i < d; i++) {
        data[i * d + i] = 1; // Identity matrix
      }
      this.A[a] = { data, rows: d, cols: d };
      this.b[a] = new Float64Array(d);
    }
  }

  /**
   * Select an arm given a context vector.
   *
   * For each arm, computes:
   *   theta_a = A_a^{-1} b_a
   *   UCB_a = theta_a^T x + alpha * sqrt(x^T A_a^{-1} x)
   *
   * @param context Context vector of dimension d
   * @returns Index of the selected arm
   */
  select(context: Float64Array): number {
    let bestArm = 0;
    let bestUCB = -Infinity;

    for (let a = 0; a < this.nArms; a++) {
      const Aa = this.A[a]!;
      const ba = this.b[a]!;

      // theta_a = A_a^{-1} b_a
      const theta = choleskySolve(Aa, ba);

      // A_a^{-1} x  (solve A_a z = x)
      const AinvX = choleskySolve(Aa, context);

      // UCB = theta^T x + alpha * sqrt(x^T A_a^{-1} x)
      const exploit = dot(theta, context);
      const explore = Math.sqrt(Math.max(dot(context, AinvX), 0));
      const ucbValue = exploit + this.alpha * explore;

      if (ucbValue > bestUCB) {
        bestUCB = ucbValue;
        bestArm = a;
      }
    }

    return bestArm;
  }

  /**
   * Update the model for a given arm after observing a reward.
   *
   * A_a += x x^T
   * b_a += reward * x
   *
   * @param arm Index of the arm that was played
   * @param context Context vector that was used for the decision
   * @param reward Observed reward
   */
  update(arm: number, context: Float64Array, reward: number): void {
    const Aa = this.A[arm]!;
    const ba = this.b[arm]!;
    const d = this.d;

    // A_a += x x^T  (rank-1 update)
    for (let i = 0; i < d; i++) {
      for (let j = 0; j < d; j++) {
        const current = matGet(Aa, i, j);
        matSet(Aa, i, j, current + (context[i] ?? 0) * (context[j] ?? 0));
      }
    }

    // b_a += reward * x
    for (let i = 0; i < d; i++) {
      ba[i] = (ba[i] ?? 0) + reward * (context[i] ?? 0);
    }
  }
}
