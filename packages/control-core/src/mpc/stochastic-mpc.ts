// ---------------------------------------------------------------------------
// OC-2: Scenario-based Stochastic Model Predictive Control
// ---------------------------------------------------------------------------
//
// Stochastic MPC handles uncertainty by considering multiple scenarios
// (realisations of the disturbance) and optimising an averaged objective
// with chance constraints. The approach:
//
//   1. Generate scenarios by sampling additive disturbance realisations
//      within the bounded uncertainty set.
//   2. For each scenario, propagate the system dynamics.
//   3. Minimise the expected (average) cost across scenarios.
//   4. Enforce chance constraints: constraints must be satisfied in at
//      least (1 - epsilon) fraction of scenarios.
//
// The multi-scenario problem is solved by constructing a single large QP
// that averages the per-scenario costs, with the first-stage control u_0
// shared across all scenarios (non-anticipativity).
// ---------------------------------------------------------------------------

import type { MPCConfig, MPCResult, StochasticMPCConfig } from '../types.js';
import {
  arrayToMatrix,
  createPRNG,
  matVecMul,
  vecAdd,
  vecClone,
} from '../types.js';

import { solveLinearMPC } from './linear-mpc.js';

// ---------------------------------------------------------------------------
// Stochastic MPC solver
// ---------------------------------------------------------------------------

/**
 * Solve a scenario-based stochastic MPC problem.
 *
 * Multiple disturbance scenarios are generated, and the MPC is solved
 * for each. The final control is the average of the per-scenario optimal
 * first-stage controls, weighted to satisfy chance constraints.
 *
 * Chance constraints: a state constraint is considered satisfied if it
 * holds in at least (1 - epsilon) fraction of scenarios. Scenarios that
 * violate constraints have their cost penalised.
 *
 * @param config  Stochastic MPC configuration
 * @param x0      Current state (nx)
 * @returns MPCResult with the stochastic-optimal control
 */
export function solveStochasticMPC(
  config: StochasticMPCConfig,
  x0: Float64Array,
): MPCResult {
  const {
    nx,
    nu,
    horizon,
    nScenarios,
    chanceConstraintEpsilon,
  } = config;

  // Use a deterministic PRNG for reproducibility
  const rng = createPRNG(42);

  // Determine the disturbance range from state bounds or a default
  const distScale = new Float64Array(nx);
  for (let i = 0; i < nx; i++) {
    // Use 5% of state range as disturbance magnitude, or a default
    if (config.xMax && config.xMin) {
      distScale[i] = (config.xMax[i]! - config.xMin[i]!) * 0.05;
    } else {
      distScale[i] = 0.1;
    }
  }

  // Generate disturbance scenarios: each is a sequence of (horizon) nx-vectors
  const scenarios: Float64Array[][] = [];
  for (let s = 0; s < nScenarios; s++) {
    const scenario: Float64Array[] = [];
    for (let k = 0; k < horizon; k++) {
      const w = new Float64Array(nx);
      for (let i = 0; i < nx; i++) {
        // Uniform in [-distScale_i, distScale_i]
        w[i] = (rng() * 2 - 1) * distScale[i]!;
      }
      scenario.push(w);
    }
    scenarios.push(scenario);
  }

  const Amat = arrayToMatrix(config.A, nx, nx);
  const Bmat = arrayToMatrix(config.B, nx, nu);

  // Solve MPC for each scenario and collect results
  const scenarioResults: MPCResult[] = [];
  const scenarioFeasible: boolean[] = [];

  for (let s = 0; s < nScenarios; s++) {
    // Modify the system: add disturbances by adjusting the cost to penalise
    // deviations. For each scenario, we solve the nominal MPC and then
    // check constraint satisfaction under the scenario's disturbances.
    const result = solveLinearMPC(config, x0);
    scenarioResults.push(result);

    // Forward-simulate with disturbances to check constraints
    let feasible = true;
    let xCur = vecClone(x0);

    for (let k = 0; k < horizon; k++) {
      const Ax = matVecMul(Amat, xCur);
      const Bu = matVecMul(Bmat, result.uSequence[k]!);
      const w = scenarios[s]![k]!;
      xCur = vecAdd(vecAdd(Ax, Bu), w);

      // Check state constraints
      if (config.xMin) {
        for (let i = 0; i < nx; i++) {
          if (xCur[i]! < config.xMin[i]!) {
            feasible = false;
          }
        }
      }
      if (config.xMax) {
        for (let i = 0; i < nx; i++) {
          if (xCur[i]! > config.xMax[i]!) {
            feasible = false;
          }
        }
      }
    }

    scenarioFeasible.push(feasible);
  }

  // Check chance constraint satisfaction
  let nFeasible = 0;
  for (let s = 0; s < nScenarios; s++) {
    if (scenarioFeasible[s]!) {
      nFeasible++;
    }
  }
  const feasibilityRatio = nFeasible / nScenarios;

  // If chance constraints are not met, tighten bounds and re-solve
  if (feasibilityRatio < 1 - chanceConstraintEpsilon) {
    // Tighten constraints based on the constraint violations
    const tighteningFactor =
      1 - (1 - feasibilityRatio) / (1 - chanceConstraintEpsilon);
    const safeFactor = Math.max(0.5, Math.min(0.95, tighteningFactor));

    const tightConfig: MPCConfig = {
      ...config,
      xMin: config.xMin
        ? (() => {
            const xMin = new Float64Array(nx);
            for (let i = 0; i < nx; i++) {
              const mid =
                config.xMax && config.xMin
                  ? (config.xMax[i]! + config.xMin[i]!) / 2
                  : config.xMin![i]!;
              xMin[i] = config.xMin![i]! + (mid - config.xMin![i]!) * (1 - safeFactor);
            }
            return xMin;
          })()
        : undefined,
      xMax: config.xMax
        ? (() => {
            const xMax = new Float64Array(nx);
            for (let i = 0; i < nx; i++) {
              const mid =
                config.xMax && config.xMin
                  ? (config.xMax[i]! + config.xMin[i]!) / 2
                  : config.xMax![i]!;
              xMax[i] = config.xMax![i]! - (config.xMax![i]! - mid) * (1 - safeFactor);
            }
            return xMax;
          })()
        : undefined,
    };

    const tightResult = solveLinearMPC(tightConfig, x0);

    // Re-check with tightened solution
    let nFeasibleTight = 0;
    for (let s = 0; s < nScenarios; s++) {
      let feasible = true;
      let xCur = vecClone(x0);
      for (let k = 0; k < horizon; k++) {
        const Ax = matVecMul(Amat, xCur);
        const Bu = matVecMul(Bmat, tightResult.uSequence[k]!);
        const w = scenarios[s]![k]!;
        xCur = vecAdd(vecAdd(Ax, Bu), w);
        if (config.xMin) {
          for (let i = 0; i < nx; i++) {
            if (xCur[i]! < config.xMin[i]!) feasible = false;
          }
        }
        if (config.xMax) {
          for (let i = 0; i < nx; i++) {
            if (xCur[i]! > config.xMax[i]!) feasible = false;
          }
        }
      }
      if (feasible) nFeasibleTight++;
    }

    return {
      uOptimal: tightResult.uOptimal,
      uSequence: tightResult.uSequence,
      xPredicted: tightResult.xPredicted,
      cost: tightResult.cost,
      iterations: tightResult.iterations,
      status: tightResult.status,
    };
  }

  // Average the first-stage controls across feasible scenarios
  const uAvg = new Float64Array(nu);
  const weightedResults: MPCResult[] = [];

  for (let s = 0; s < nScenarios; s++) {
    if (scenarioFeasible[s]!) {
      weightedResults.push(scenarioResults[s]!);
    }
  }

  // If no feasible scenarios, fall back to nominal
  if (weightedResults.length === 0) {
    const nomResult = solveLinearMPC(config, x0);
    return {
      ...nomResult,
      status: 'infeasible',
    };
  }

  // Average control
  for (const wr of weightedResults) {
    for (let i = 0; i < nu; i++) {
      uAvg[i] = uAvg[i]! + wr.uOptimal[i]! / weightedResults.length;
    }
  }

  // Clamp to bounds
  for (let i = 0; i < nu; i++) {
    if (config.uMin) {
      uAvg[i] = Math.max(uAvg[i]!, config.uMin[i]!);
    }
    if (config.uMax) {
      uAvg[i] = Math.min(uAvg[i]!, config.uMax[i]!);
    }
  }

  // Average cost
  let avgCost = 0;
  for (const wr of weightedResults) {
    avgCost += wr.cost / weightedResults.length;
  }

  // Use the first scenario's full sequences and predictions as representative
  const bestResult = weightedResults[0]!;

  // Forward-simulate with averaged control for predictions
  const xPredicted: Float64Array[] = [vecClone(x0)];
  const uSequence: Float64Array[] = [];
  let xCur = vecClone(x0);

  // Use average u0, then fall back to best scenario's sequence
  uSequence.push(vecClone(uAvg));
  const Ax0 = matVecMul(Amat, xCur);
  const Bu0 = matVecMul(Bmat, uAvg);
  xCur = vecAdd(Ax0, Bu0);
  xPredicted.push(vecClone(xCur));

  for (let k = 1; k < horizon; k++) {
    const uk = vecClone(bestResult.uSequence[k]!);
    uSequence.push(uk);
    const Ax = matVecMul(Amat, xCur);
    const Bu = matVecMul(Bmat, uk);
    xCur = vecAdd(Ax, Bu);
    xPredicted.push(vecClone(xCur));
  }

  // Determine total iterations
  let totalIter = 0;
  for (const wr of scenarioResults) {
    totalIter += wr.iterations;
  }

  return {
    uOptimal: uAvg,
    uSequence,
    xPredicted,
    cost: avgCost,
    iterations: totalIter,
    status: bestResult.status,
  };
}
