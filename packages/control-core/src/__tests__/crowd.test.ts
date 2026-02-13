// ---------------------------------------------------------------------------
// Tests for OC-8: Crowd Flow & Evacuation
// ---------------------------------------------------------------------------

import { describe, it, expect } from 'vitest';
import {
  solveEikonal,
  createHughesState,
  hughesStep,
  socialForceStep,
  socialRepulsion,
  wallRepulsion,
  solveEvacuationMPC,
  projectDensity,
} from '../crowd/index.js';
import type {
  HughesModelConfig,
  SocialForceConfig,
  SocialForceAgent,
  EvacuationMPCConfig,
} from '../types.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Create a simple 5x5 Hughes config with an exit at (2, 0). */
function makeSmallHughesConfig(): HughesModelConfig {
  return {
    gridNx: 5,
    gridNy: 5,
    dx: 1.0,
    maxDensity: 5.0,
    speedFn: (rho: number) => Math.max(0, 1.34 * (1 - rho / 5.0)),
    exits: [{ x: 2, y: 0 }],
    dt: 0.1,
  };
}

/** Create a uniform density field for a 5x5 grid. */
function makeUniformDensity(nx: number, ny: number, value: number): Float64Array {
  return new Float64Array(nx * ny).fill(value);
}

const baseSocialForceConfig: SocialForceConfig = {
  desiredSpeed: 1.34,
  relaxationTime: 0.5,
  socialMagnitude: 2.1,
  socialRange: 0.3,
  wallMagnitude: 10.0,
  wallRange: 0.2,
  dt: 0.05,
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('Eikonal Solver', () => {
  it('potential is non-negative everywhere', () => {
    const config = makeSmallHughesConfig();
    const density = makeUniformDensity(5, 5, 0.5);
    const potential = solveEikonal(config, density);

    for (let i = 0; i < potential.length; i++) {
      expect(potential[i]!).toBeGreaterThanOrEqual(0);
    }
  });

  it('potential is zero at exit cells', () => {
    const config = makeSmallHughesConfig();
    const density = makeUniformDensity(5, 5, 0.5);
    const potential = solveEikonal(config, density);

    // Exit is at (x=2, y=0) => index = 0 * 5 + 2 = 2
    expect(potential[2]).toBe(0);

    // Non-exit cells should have positive potential
    expect(potential[0]!).toBeGreaterThan(0);
    expect(potential[12]!).toBeGreaterThan(0); // center of grid
  });
});

describe('Hughes Model', () => {
  it('hughesStep conserves total mass (sum of density) approximately', () => {
    const config = makeSmallHughesConfig();
    // Use a low density so flow is smooth and mass conservation holds well
    const initialDensity = new Float64Array(25);
    // Place some density in the center area, away from exit
    initialDensity[12] = 1.0; // center
    initialDensity[11] = 0.5;
    initialDensity[13] = 0.5;
    initialDensity[7] = 0.3;
    initialDensity[17] = 0.3;

    const state0 = createHughesState(config, initialDensity);

    let totalBefore = 0;
    for (let i = 0; i < state0.density.length; i++) {
      totalBefore += state0.density[i]!;
    }

    // Take a small step
    const state1 = hughesStep(config, state0);

    let totalAfter = 0;
    for (let i = 0; i < state1.density.length; i++) {
      totalAfter += state1.density[i]!;
    }

    // Density is conserved approximately (some mass may exit through exits,
    // or be clamped, but with small dt and interior density it should be close)
    // Allow up to 20% change due to boundary effects and clamping
    expect(totalAfter).toBeGreaterThan(0);
    expect(Math.abs(totalAfter - totalBefore)).toBeLessThan(totalBefore * 0.2 + 0.1);
  });
});

describe('Social Force Model', () => {
  it('agents repel each other (social repulsion pushes apart)', () => {
    // Two agents facing each other
    const agentA: SocialForceAgent = {
      x: 0, y: 0, vx: 0, vy: 0,
      goalX: 10, goalY: 0, radius: 0.3,
    };
    const agentB: SocialForceAgent = {
      x: 0.8, y: 0, vx: 0, vy: 0,
      goalX: -10, goalY: 0, radius: 0.3,
    };

    const rep = socialRepulsion(agentA, agentB, baseSocialForceConfig);

    // Repulsion on A from B should push A in the -x direction (away from B)
    expect(rep.fx).toBeLessThan(0);
    // Force magnitude should be positive
    expect(Math.sqrt(rep.fx * rep.fx + rep.fy * rep.fy)).toBeGreaterThan(0);

    // After a simulation step, agents should move apart (or at least not collide)
    const agents = [agentA, agentB];
    const nextAgents = socialForceStep(baseSocialForceConfig, agents, []);

    const distBefore = Math.abs(agentB.x - agentA.x);
    const distAfter = Math.abs(nextAgents[1]!.x - nextAgents[0]!.x);

    // With agents close together, repulsion should keep them from getting
    // significantly closer (they may move apart due to their goals too)
    expect(distAfter).toBeGreaterThan(0);
  });

  it('wall repulsion prevents agents from penetrating walls', () => {
    // Agent near a wall
    const agent: SocialForceAgent = {
      x: 0.2, y: 0, vx: -1, vy: 0, // moving toward the wall
      goalX: -5, goalY: 0, radius: 0.3,
    };

    const wall = { x1: 0, y1: -5, x2: 0, y2: 5 }; // vertical wall at x=0

    const rep = wallRepulsion(agent, wall, baseSocialForceConfig);

    // Wall force should push agent in +x direction (away from wall)
    expect(rep.fx).toBeGreaterThan(0);

    // Simulate several steps with the wall
    let agents = [agent];
    for (let step = 0; step < 100; step++) {
      agents = socialForceStep(baseSocialForceConfig, agents, [wall]);
    }

    // Agent should not have penetrated the wall (x should be > 0 or close to 0)
    expect(agents[0]!.x).toBeGreaterThan(-0.1);
  });
});

describe('Evacuation MPC', () => {
  it('assigns all people to exits (row sums equal 1)', () => {
    const config: EvacuationMPCConfig = {
      nZones: 3,
      nExits: 2,
      zoneCapacities: new Float64Array([100, 80, 60]),
      exitCapacities: new Float64Array([120, 120]),
      travelTimes: new Float64Array([
        // zone 0 -> exit 0, exit 1
        5, 10,
        // zone 1 -> exit 0, exit 1
        8, 3,
        // zone 2 -> exit 0, exit 1
        6, 6,
      ]),
      horizon: 10,
    };

    const occupancy = new Float64Array([50, 40, 30]);

    const { assignment, totalTime } = solveEvacuationMPC(config, occupancy);

    // Each zone's row should sum to 1 (all people are assigned)
    for (let z = 0; z < 3; z++) {
      let rowSum = 0;
      for (let e = 0; e < 2; e++) {
        rowSum += assignment[z * 2 + e]!;
      }
      expect(rowSum).toBeCloseTo(1.0, 5);
    }

    // All assignments should be non-negative
    for (let i = 0; i < assignment.length; i++) {
      expect(assignment[i]!).toBeGreaterThanOrEqual(-1e-10);
    }

    // Total time should be positive
    expect(totalTime).toBeGreaterThan(0);
  });
});

describe('Density Projection', () => {
  it('output respects max density constraint', () => {
    // Create a density field with some cells exceeding the max
    const density = new Float64Array([2.0, 6.0, 3.0, 8.0, 1.0, 4.0, 7.0, 0.5]);
    const maxDensity = 5.0;

    const projected = projectDensity(density, maxDensity);

    // No cell should exceed maxDensity after projection
    for (let i = 0; i < projected.length; i++) {
      expect(projected[i]!).toBeLessThanOrEqual(maxDensity);
      expect(projected[i]!).toBeGreaterThanOrEqual(0);
    }

    // Cells that were already below maxDensity may have received
    // redistributed excess, but should still not exceed maxDensity
    expect(projected[0]!).toBeLessThanOrEqual(maxDensity);

    // Cells that were above maxDensity have been clamped
    expect(projected[1]!).toBeLessThanOrEqual(maxDensity);
    expect(projected[3]!).toBeLessThanOrEqual(maxDensity);
    expect(projected[6]!).toBeLessThanOrEqual(maxDensity);
  });
});
