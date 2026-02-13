// ---------------------------------------------------------------------------
// SP-8: Social Force Model for Crowd Flow Simulation
// ---------------------------------------------------------------------------
// Helbing & Molnar (1995): m_i·(dv_i/dt) = F_desire + Σ F_social + Σ F_obstacle
// F_desire = m·(v₀·ê_goal - v)/τ  (relaxation to desired velocity)
// F_social = A·exp((r_ij - d_ij)/B)·n̂_ij  (repulsive)
// F_obstacle = A·exp((r_i - d_iw)/B)·n̂_iw

import type { SocialForceConfig, CrowdAgent, PRNG } from '../types.js';
import { createPRNG } from '../types.js';

const DEFAULT_CONFIG: SocialForceConfig = {
  desiredSpeed: 1.3,           // m/s (typical walking speed)
  relaxationTime: 0.5,         // seconds
  socialForceMagnitude: 2.1,   // N
  socialForceRange: 0.3,       // meters
};

/**
 * Compute desire force: tendency to move towards goal at desired speed.
 * F_desire = (v₀·ê_goal - v) / τ
 */
function desireForce(agent: CrowdAgent, config: SocialForceConfig): [number, number] {
  const dx = agent.goalX - agent.x;
  const dy = agent.goalY - agent.y;
  const dist = Math.sqrt(dx * dx + dy * dy);

  if (dist < 0.1) return [0, 0]; // Already at goal

  const ex = dx / dist;
  const ey = dy / dist;

  const fx = (config.desiredSpeed * ex - agent.vx) / config.relaxationTime;
  const fy = (config.desiredSpeed * ey - agent.vy) / config.relaxationTime;

  return [fx, fy];
}

/**
 * Compute social repulsion force between two agents.
 * F_social = A·exp((r_ij - d_ij)/B)·n̂_ij
 */
function socialForce(
  agentI: CrowdAgent,
  agentJ: CrowdAgent,
  config: SocialForceConfig,
): [number, number] {
  const dx = agentI.x - agentJ.x;
  const dy = agentI.y - agentJ.y;
  const dist = Math.sqrt(dx * dx + dy * dy);

  if (dist < 0.01) return [0, 0]; // Overlapping

  const rij = 0.6; // Combined radii (2 × pedestrian radius ~0.3m)
  const magnitude = config.socialForceMagnitude * Math.exp((rij - dist) / config.socialForceRange);

  const nx = dx / dist;
  const ny = dy / dist;

  return [magnitude * nx, magnitude * ny];
}

/**
 * Compute wall repulsion force.
 */
function wallForce(
  agent: CrowdAgent,
  wallX: number,
  wallY: number,
  wallNx: number,
  wallNy: number,
  config: SocialForceConfig,
): [number, number] {
  // Distance from agent to wall (signed)
  const dist = (agent.x - wallX) * wallNx + (agent.y - wallY) * wallNy;

  if (dist > 3 * config.socialForceRange) return [0, 0]; // Too far

  const magnitude = config.socialForceMagnitude * 2 * Math.exp((0.3 - dist) / config.socialForceRange);

  return [magnitude * wallNx, magnitude * wallNy];
}

/**
 * Initialize crowd agents with random positions and goals.
 */
export function initializeCrowd(
  count: number,
  roomWidth: number,
  roomHeight: number,
  seed: number = 42,
): CrowdAgent[] {
  const rng = createPRNG(seed);
  const agents: CrowdAgent[] = [];

  for (let i = 0; i < count; i++) {
    agents.push({
      x: 0.5 + rng() * (roomWidth - 1),
      y: 0.5 + rng() * (roomHeight - 1),
      vx: 0,
      vy: 0,
      goalX: 0.5 + rng() * (roomWidth - 1),
      goalY: 0.5 + rng() * (roomHeight - 1),
    });
  }

  return agents;
}

/**
 * Step the Social Force Model forward by dt seconds.
 */
export function socialForceStep(
  agents: CrowdAgent[],
  dt: number = 0.05,
  roomWidth: number = 20,
  roomHeight: number = 15,
  config: SocialForceConfig = DEFAULT_CONFIG,
): CrowdAgent[] {
  const n = agents.length;
  const forces: Array<[number, number]> = new Array(n);

  // Compute forces for each agent
  for (let i = 0; i < n; i++) {
    const agent = agents[i]!;
    let fx = 0, fy = 0;

    // Desire force
    const [fdx, fdy] = desireForce(agent, config);
    fx += fdx;
    fy += fdy;

    // Social forces from other agents
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const [fsx, fsy] = socialForce(agent, agents[j]!, config);
      fx += fsx;
      fy += fsy;
    }

    // Wall forces (4 walls of rectangular room)
    const walls: Array<[number, number, number, number]> = [
      [0, 0, 1, 0],           // Left wall
      [roomWidth, 0, -1, 0],  // Right wall
      [0, 0, 0, 1],           // Bottom wall
      [0, roomHeight, 0, -1], // Top wall
    ];

    for (const [wx, wy, wnx, wny] of walls) {
      const [fwx, fwy] = wallForce(agent, wx, wy, wnx, wny, config);
      fx += fwx;
      fy += fwy;
    }

    forces[i] = [fx, fy];
  }

  // Update velocities and positions (Euler integration)
  return agents.map((agent, i) => {
    const [fx, fy] = forces[i]!;
    const newVx = agent.vx + fx * dt;
    const newVy = agent.vy + fy * dt;

    // Clamp velocity to max speed (2× desired)
    const speed = Math.sqrt(newVx * newVx + newVy * newVy);
    const maxSpeed = config.desiredSpeed * 2;
    const vx = speed > maxSpeed ? newVx * maxSpeed / speed : newVx;
    const vy = speed > maxSpeed ? newVy * maxSpeed / speed : newVy;

    // Update position with boundary clamping
    const x = Math.max(0.1, Math.min(roomWidth - 0.1, agent.x + vx * dt));
    const y = Math.max(0.1, Math.min(roomHeight - 0.1, agent.y + vy * dt));

    return { ...agent, x, y, vx, vy };
  });
}

/**
 * Run social force simulation for multiple steps.
 * Returns time series of agent positions.
 */
export function simulateCrowdFlow(
  agents: CrowdAgent[],
  nSteps: number,
  dt: number = 0.05,
  roomWidth: number = 20,
  roomHeight: number = 15,
  config: SocialForceConfig = DEFAULT_CONFIG,
): CrowdAgent[][] {
  const trajectory: CrowdAgent[][] = [agents.map(a => ({ ...a }))];

  let current = agents;
  for (let step = 0; step < nSteps; step++) {
    current = socialForceStep(current, dt, roomWidth, roomHeight, config);
    trajectory.push(current.map(a => ({ ...a })));
  }

  return trajectory;
}
