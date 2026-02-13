// ---------------------------------------------------------------------------
// OC-9  Stochastic Optimal Control -- Chance-Constrained Optimization & MPC
// ---------------------------------------------------------------------------

import type { ChanceConstraintConfig, ScenarioMPCConfig } from '../types.js';

// ---------------------------------------------------------------------------
// Chance-Constrained Optimization
// ---------------------------------------------------------------------------

/**
 * Chance-constrained optimization over a finite set of candidate actions.
 *
 * For each action, evaluates the constraint function g(state, action, scenario)
 * across all scenarios. The action is feasible if the empirical violation rate
 * (fraction of scenarios where g > 0) does not exceed violationProb.
 *
 * Among feasible actions, selects the one with the lowest expected cost
 * (average of constraint function values, treated as a cost proxy, or
 * more precisely, the mean constraint function value -- lower is better).
 *
 * If no action is feasible, selects the action with the lowest violation rate.
 *
 * @param config    - ChanceConstraintConfig with violationProb, nScenarios, constraintFn
 * @param state     - Current state vector
 * @param actions   - Array of candidate action vectors
 * @param scenarios - Array of scenario vectors
 * @returns Best action and its empirical violation rate
 */
export function chanceConstrainedOptimize(
  config: ChanceConstraintConfig,
  state: Float64Array,
  actions: Float64Array[],
  scenarios: Float64Array[],
): { bestAction: Float64Array; violationRate: number } {
  const { violationProb, constraintFn } = config;
  const nScenarios = scenarios.length;
  const nActions = actions.length;

  if (nActions === 0) {
    return { bestAction: new Float64Array(0), violationRate: 1 };
  }

  if (nScenarios === 0) {
    // No scenarios -- treat all actions as feasible
    return { bestAction: new Float64Array(actions[0]!), violationRate: 0 };
  }

  let bestAction = actions[0]!;
  let bestViolationRate = 1;
  let bestCost = Infinity;
  let foundFeasible = false;

  for (let ai = 0; ai < nActions; ai++) {
    const action = actions[ai]!;

    let violations = 0;
    let totalCost = 0;

    for (let wi = 0; wi < nScenarios; wi++) {
      const g = constraintFn(state, action, scenarios[wi]!);
      totalCost += g;
      if (g > 0) {
        violations++;
      }
    }

    const violationRate = violations / nScenarios;
    const avgCost = totalCost / nScenarios;
    const isFeasible = violationRate <= violationProb;

    if (isFeasible) {
      if (!foundFeasible || avgCost < bestCost) {
        foundFeasible = true;
        bestAction = action;
        bestViolationRate = violationRate;
        bestCost = avgCost;
      }
    } else if (!foundFeasible) {
      // No feasible action found yet -- track lowest violation rate
      if (violationRate < bestViolationRate) {
        bestAction = action;
        bestViolationRate = violationRate;
        bestCost = avgCost;
      } else if (
        violationRate === bestViolationRate &&
        avgCost < bestCost
      ) {
        bestAction = action;
        bestCost = avgCost;
      }
    }
  }

  return {
    bestAction: new Float64Array(bestAction),
    violationRate: bestViolationRate,
  };
}

// ---------------------------------------------------------------------------
// Scenario-Based MPC
// ---------------------------------------------------------------------------

/**
 * Solve a scenario-based Model Predictive Control problem.
 *
 * Traverses the scenario tree to enumerate leaf paths (complete scenarios),
 * then solves a simplified MPC for each path using the linear dynamics
 * x_{k+1} = A x_k + B u_k from the base MPCConfig.
 *
 * The first-stage decision is non-anticipative: we average the optimal
 * control sequences across all scenario paths weighted by their
 * probabilities, and return the expected cost.
 *
 * For each scenario path:
 *   1. Roll forward the dynamics along the path's state trajectory
 *   2. Compute a locally optimal control sequence minimising the
 *      quadratic cost x'Qx + u'Ru with optional terminal cost Qf
 *   3. Weight the result by the scenario's probability
 *
 * @param config - ScenarioMPCConfig with scenario tree, dynamics, and costs
 * @param x0     - Initial state vector
 * @returns First-stage control sequence and expected cost
 */
export function solveScenarioMPC(
  config: ScenarioMPCConfig,
  x0: Float64Array,
): { uSequence: Float64Array[][]; cost: number } {
  const { A, B, Q, R, Qf, nx, nu, horizon, scenarioTree, nStages } = config;

  // ---- Enumerate leaf paths through the scenario tree ----
  const paths = enumerateScenarioPaths(scenarioTree);
  const nPaths = paths.length;

  if (nPaths === 0) {
    // Degenerate: no scenarios, return zero control
    const zeroSeq: Float64Array[][] = [];
    for (let t = 0; t < horizon; t++) {
      zeroSeq.push([new Float64Array(nu)]);
    }
    return { uSequence: zeroSeq, cost: 0 };
  }

  // ---- Solve MPC for each scenario path ----
  const pathControls: Float64Array[][] = [];
  const pathCosts: number[] = [];
  const pathProbs: number[] = [];

  for (let pi = 0; pi < nPaths; pi++) {
    const path = paths[pi]!;

    // Compute path probability (product along path, or use leaf probability)
    const leafNode = scenarioTree[path[path.length - 1]!]!;
    const pathProb = leafNode.probability;
    pathProbs.push(pathProb);

    // Forward simulate and compute optimal control via simple
    // quadratic cost minimisation (one-step greedy with linear dynamics)
    const uSeq: Float64Array[] = [];
    let x = new Float64Array(x0);
    let totalCost = 0;

    const stepsToSolve = Math.min(horizon, nStages, path.length);

    for (let t = 0; t < stepsToSolve; t++) {
      // Compute greedy control: u = -inv(R + B'QB) * B'QAx
      // This is a simplified single-step optimal for the quadratic cost.
      const u = computeGreedyControl(x, A, B, Q, R, nx, nu);

      // Clamp controls to bounds if specified
      if (config.uMin && config.uMax) {
        for (let i = 0; i < nu; i++) {
          u[i] = Math.max(
            config.uMin[i]!,
            Math.min(config.uMax[i]!, u[i]!),
          );
        }
      }

      uSeq.push(u);

      // Stage cost: x'Qx + u'Ru
      let stageCost = 0;
      for (let i = 0; i < nx; i++) {
        for (let j = 0; j < nx; j++) {
          stageCost += x[i]! * Q[i * nx + j]! * x[j]!;
        }
      }
      for (let i = 0; i < nu; i++) {
        for (let j = 0; j < nu; j++) {
          stageCost += u[i]! * R[i * nu + j]! * u[j]!;
        }
      }
      totalCost += stageCost;

      // Propagate dynamics: x_{k+1} = A x_k + B u_k
      const xNext = new Float64Array(nx);
      for (let i = 0; i < nx; i++) {
        let sum = 0;
        for (let j = 0; j < nx; j++) {
          sum += A[i * nx + j]! * x[j]!;
        }
        for (let j = 0; j < nu; j++) {
          sum += B[i * nu + j]! * u[j]!;
        }
        xNext[i] = sum;
      }
      x = xNext;
    }

    // Terminal cost if Qf provided
    if (Qf) {
      let termCost = 0;
      for (let i = 0; i < nx; i++) {
        for (let j = 0; j < nx; j++) {
          termCost += x[i]! * Qf[i * nx + j]! * x[j]!;
        }
      }
      totalCost += termCost;
    }

    // Pad with zero controls if path was shorter than horizon
    while (uSeq.length < horizon) {
      uSeq.push(new Float64Array(nu));
    }

    pathControls.push(uSeq);
    pathCosts.push(totalCost);
  }

  // ---- Aggregate: probability-weighted average of control sequences ----
  // Normalise probabilities
  let probSum = 0;
  for (let pi = 0; pi < nPaths; pi++) {
    probSum += pathProbs[pi]!;
  }
  if (probSum < 1e-15) probSum = 1;

  const uSequence: Float64Array[][] = [];
  let expectedCost = 0;

  for (let t = 0; t < horizon; t++) {
    const avgU = new Float64Array(nu);
    for (let pi = 0; pi < nPaths; pi++) {
      const w = pathProbs[pi]! / probSum;
      const upi = pathControls[pi]![t]!;
      for (let i = 0; i < nu; i++) {
        avgU[i]! += w * upi[i]!;
      }
    }
    uSequence.push([avgU]);
  }

  for (let pi = 0; pi < nPaths; pi++) {
    expectedCost += (pathProbs[pi]! / probSum) * pathCosts[pi]!;
  }

  return { uSequence, cost: expectedCost };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Enumerate all root-to-leaf paths in a scenario tree.
 * Each path is an array of node indices from root to a leaf.
 */
function enumerateScenarioPaths(
  tree: readonly { children: number[]; parent: number }[],
): number[][] {
  if (tree.length === 0) return [];

  const paths: number[][] = [];

  // Find root nodes (parent === -1)
  const roots: number[] = [];
  for (let i = 0; i < tree.length; i++) {
    if (tree[i]!.parent === -1) {
      roots.push(i);
    }
  }

  // DFS from each root
  const stack: Array<{ nodeIdx: number; path: number[] }> = [];
  for (const root of roots) {
    stack.push({ nodeIdx: root, path: [root] });
  }

  while (stack.length > 0) {
    const { nodeIdx, path } = stack.pop()!;
    const node = tree[nodeIdx]!;

    if (node.children.length === 0) {
      // Leaf node
      paths.push(path);
    } else {
      for (const childIdx of node.children) {
        stack.push({ nodeIdx: childIdx, path: [...path, childIdx] });
      }
    }
  }

  return paths;
}

/**
 * Compute a single-step greedy control for linear-quadratic cost.
 *
 * Minimises  x'Qx + u'Ru  subject to  x+ = Ax + Bu
 * by solving  u = -(R + B'QB)^{-1} B'QA x
 *
 * Uses a simple direct computation for small nu.
 */
function computeGreedyControl(
  x: Float64Array,
  A: Float64Array,
  B: Float64Array,
  Q: Float64Array,
  R: Float64Array,
  nx: number,
  nu: number,
): Float64Array {
  // Compute B'Q (nu x nx)
  const BtQ = new Float64Array(nu * nx);
  for (let i = 0; i < nu; i++) {
    for (let j = 0; j < nx; j++) {
      let sum = 0;
      for (let k = 0; k < nx; k++) {
        // B'[i,k] = B[k,i] (B is nx x nu)
        sum += B[k * nu + i]! * Q[k * nx + j]!;
      }
      BtQ[i * nx + j] = sum;
    }
  }

  // Compute B'QA (nu x nx)
  const BtQA = new Float64Array(nu * nx);
  for (let i = 0; i < nu; i++) {
    for (let j = 0; j < nx; j++) {
      let sum = 0;
      for (let k = 0; k < nx; k++) {
        sum += BtQ[i * nx + k]! * A[k * nx + j]!;
      }
      BtQA[i * nx + j] = sum;
    }
  }

  // Compute B'QB (nu x nu)
  const BtQB = new Float64Array(nu * nu);
  for (let i = 0; i < nu; i++) {
    for (let j = 0; j < nu; j++) {
      let sum = 0;
      for (let k = 0; k < nx; k++) {
        sum += BtQ[i * nx + k]! * B[k * nu + j]!;
      }
      BtQB[i * nu + j] = sum;
    }
  }

  // Compute M = R + B'QB (nu x nu)
  const M = new Float64Array(nu * nu);
  for (let i = 0; i < nu * nu; i++) {
    M[i] = R[i]! + BtQB[i]!;
  }

  // Compute rhs = B'QA * x (nu)
  const rhs = new Float64Array(nu);
  for (let i = 0; i < nu; i++) {
    let sum = 0;
    for (let j = 0; j < nx; j++) {
      sum += BtQA[i * nx + j]! * x[j]!;
    }
    rhs[i] = sum;
  }

  // Solve M * u = -rhs  via Gauss elimination for small systems
  const u = solveSmallLinearSystem(M, rhs, nu);
  for (let i = 0; i < nu; i++) {
    u[i] = -u[i]!;
  }

  return u;
}

/**
 * Solve a small n x n linear system Ax = b via Gaussian elimination
 * with partial pivoting. Modifies copies internally.
 */
function solveSmallLinearSystem(
  A: Float64Array,
  b: Float64Array,
  n: number,
): Float64Array {
  // Working copies
  const aug = new Float64Array(n * (n + 1));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      aug[i * (n + 1) + j] = A[i * n + j]!;
    }
    aug[i * (n + 1) + n] = b[i]!;
  }

  // Forward elimination with partial pivoting
  for (let col = 0; col < n; col++) {
    // Find pivot
    let maxRow = col;
    let maxVal = Math.abs(aug[col * (n + 1) + col]!);
    for (let row = col + 1; row < n; row++) {
      const val = Math.abs(aug[row * (n + 1) + col]!);
      if (val > maxVal) {
        maxVal = val;
        maxRow = row;
      }
    }

    // Swap rows
    if (maxRow !== col) {
      for (let j = 0; j <= n; j++) {
        const tmp = aug[col * (n + 1) + j]!;
        aug[col * (n + 1) + j] = aug[maxRow * (n + 1) + j]!;
        aug[maxRow * (n + 1) + j] = tmp;
      }
    }

    const pivot = aug[col * (n + 1) + col]!;
    if (Math.abs(pivot) < 1e-15) {
      // Singular -- return zero
      return new Float64Array(n);
    }

    // Scale pivot row
    for (let j = col; j <= n; j++) {
      aug[col * (n + 1) + j] = aug[col * (n + 1) + j]! / pivot;
    }

    // Eliminate below
    for (let row = col + 1; row < n; row++) {
      const factor = aug[row * (n + 1) + col]!;
      for (let j = col; j <= n; j++) {
        aug[row * (n + 1) + j] =
          aug[row * (n + 1) + j]! - factor * aug[col * (n + 1) + j]!;
      }
    }
  }

  // Back substitution
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let sum = aug[i * (n + 1) + n]!;
    for (let j = i + 1; j < n; j++) {
      sum -= aug[i * (n + 1) + j]! * x[j]!;
    }
    x[i] = sum;
  }

  return x;
}
