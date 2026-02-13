// ---------------------------------------------------------------------------
// OC-7: Decentralized MPC via ADMM Consensus
// ---------------------------------------------------------------------------

import type { DecentralizedMPCConfig } from '../types.js';
import { vecSub, vecNorm, vecScale, vecAdd } from '../types.js';

// ---------------------------------------------------------------------------
// solveDecentralizedMPC
// ---------------------------------------------------------------------------

/**
 * Solve a decentralized model-predictive control problem using ADMM-based
 * consensus.
 *
 * Each agent independently solves a local (simplified) MPC problem, producing
 * a control action via proportional feedback u = -K * x. The agents then
 * participate in an ADMM consensus step to ensure coupling constraints
 * (encoded in `couplingMatrix`) are satisfied.
 *
 * Algorithm:
 *   1. Each agent computes a local control action (proportional controller).
 *   2. Average consensus: compute the global average of all agent actions
 *      weighted by the coupling matrix.
 *   3. Dual variable update: each agent's dual variable is updated based on
 *      the residual between its local action and the consensus.
 *   4. Repeat until the dual residual norm falls below tolerance or
 *      maxConsensusIter is reached.
 *
 * @param config  Decentralized MPC configuration with per-agent MPC configs,
 *                coupling matrix, consensus weight, and iteration limits.
 * @param states  Current state vector for each agent (nAgents arrays).
 * @returns       Actions for each agent, convergence flag, and iteration count.
 */
export function solveDecentralizedMPC(
  config: DecentralizedMPCConfig,
  states: Float64Array[],
): { actions: Float64Array[]; converged: boolean; iterations: number } {
  const { nAgents, agentConfigs, couplingMatrix, consensusWeight, maxConsensusIter } = config;

  // Determine each agent's control dimension from its MPC config
  const nuPerAgent: number[] = [];
  for (let a = 0; a < nAgents; a++) {
    nuPerAgent.push(agentConfigs[a]!.nu);
  }

  // Compute local proportional actions: u_a = -K * x_a
  // Simplified: K is a diagonal gain derived from Q/R ratio per agent
  const localActions: Float64Array[] = [];
  for (let a = 0; a < nAgents; a++) {
    const cfg = agentConfigs[a]!;
    const nu = cfg.nu;
    const nx = cfg.nx;
    const x = states[a]!;
    const u = new Float64Array(nu);

    // Simple proportional gain: K_ij = Q_ii / (Q_ii + R_jj)
    // u = -K * x  (only the first min(nu, nx) components are coupled)
    for (let i = 0; i < nu; i++) {
      let val = 0;
      for (let j = 0; j < nx; j++) {
        const qjj = cfg.Q[j * nx + j]!;
        const rii = cfg.R[i * nu + i]!;
        const gain = qjj / (qjj + rii + 1e-12);
        val += gain * x[j]!;
      }
      u[i] = -val;
    }

    localActions.push(u);
  }

  // Find the maximum nu across agents (for consensus variable dimension)
  let maxNu = 0;
  for (let a = 0; a < nAgents; a++) {
    if (nuPerAgent[a]! > maxNu) {
      maxNu = nuPerAgent[a]!;
    }
  }

  // Pad local actions to maxNu for consensus
  const paddedActions: Float64Array[] = [];
  for (let a = 0; a < nAgents; a++) {
    const padded = new Float64Array(maxNu);
    const la = localActions[a]!;
    for (let i = 0; i < nuPerAgent[a]!; i++) {
      padded[i] = la[i]!;
    }
    paddedActions.push(padded);
  }

  // Dual variables (lambda_a) initialized to zero
  const duals: Float64Array[] = [];
  for (let a = 0; a < nAgents; a++) {
    duals.push(new Float64Array(maxNu));
  }

  // ADMM iterations
  let converged = false;
  let iterations = 0;

  // Tolerance for convergence based on the first agent config (representative)
  const tol = 1e-6;

  for (let iter = 0; iter < maxConsensusIter; iter++) {
    iterations = iter + 1;

    // Step 1: Local update — each agent re-optimizes given its dual variable
    // u_a^{k+1} = localAction_a - (1 / rho) * lambda_a^k
    // where rho = consensusWeight
    const rho = consensusWeight;
    for (let a = 0; a < nAgents; a++) {
      const la = localActions[a]!;
      const pad = paddedActions[a]!;
      const dual = duals[a]!;
      for (let i = 0; i < nuPerAgent[a]!; i++) {
        pad[i] = la[i]! - (1.0 / rho) * dual[i]!;
      }
    }

    // Step 2: Consensus — compute weighted average over coupled neighbors
    const consensus: Float64Array[] = [];
    for (let a = 0; a < nAgents; a++) {
      const avg = new Float64Array(maxNu);
      let totalWeight = 0;

      for (let b = 0; b < nAgents; b++) {
        const coupling = couplingMatrix[a * nAgents + b]!;
        if (coupling > 0 || a === b) {
          const w = a === b ? 1.0 : coupling;
          totalWeight += w;
          const pb = paddedActions[b]!;
          for (let i = 0; i < maxNu; i++) {
            avg[i] = avg[i]! + w * pb[i]!;
          }
        }
      }

      if (totalWeight > 0) {
        for (let i = 0; i < maxNu; i++) {
          avg[i] = avg[i]! / totalWeight;
        }
      }
      consensus.push(avg);
    }

    // Step 3: Dual update — lambda_a += rho * (u_a - z_a)
    let maxDualResidual = 0;
    for (let a = 0; a < nAgents; a++) {
      const residual = vecSub(paddedActions[a]!, consensus[a]!);
      const scaledResidual = vecScale(residual, rho);
      duals[a] = vecAdd(duals[a]!, scaledResidual);

      const resNorm = vecNorm(residual);
      if (resNorm > maxDualResidual) {
        maxDualResidual = resNorm;
      }
    }

    // Update padded actions to consensus result for next iteration
    for (let a = 0; a < nAgents; a++) {
      const ca = consensus[a]!;
      const pa = paddedActions[a]!;
      for (let i = 0; i < maxNu; i++) {
        pa[i] = ca[i]!;
      }
    }

    // Check convergence
    if (maxDualResidual < tol) {
      converged = true;
      break;
    }
  }

  // Extract final actions (unpad)
  const actions: Float64Array[] = [];
  for (let a = 0; a < nAgents; a++) {
    const u = new Float64Array(nuPerAgent[a]!);
    const pa = paddedActions[a]!;
    for (let i = 0; i < nuPerAgent[a]!; i++) {
      u[i] = pa[i]!;
    }
    actions.push(u);
  }

  return { actions, converged, iterations };
}
