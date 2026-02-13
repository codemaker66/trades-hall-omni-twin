// ---------------------------------------------------------------------------
// OC-8  Crowd Flow & Evacuation -- Social Force Model
// ---------------------------------------------------------------------------

import type { SocialForceConfig, SocialForceAgent } from '../types.js';

/** A wall segment defined by its two endpoints. */
interface WallSegment {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

// ---------------------------------------------------------------------------
// Social Repulsion
// ---------------------------------------------------------------------------

/**
 * Compute the exponential social repulsion force exerted on `agent`
 * by `other`:
 *
 *   F = A * exp((r_ij - d_ij) / B) * n_ij
 *
 * where:
 *   r_ij = sum of radii (agent.radius + other.radius)
 *   d_ij = Euclidean distance between centres
 *   n_ij = unit vector from other to agent
 *   A    = socialMagnitude
 *   B    = socialRange
 */
export function socialRepulsion(
  agent: SocialForceAgent,
  other: SocialForceAgent,
  config: SocialForceConfig,
): { fx: number; fy: number } {
  const dx = agent.x - other.x;
  const dy = agent.y - other.y;
  const dist = Math.sqrt(dx * dx + dy * dy);

  if (dist < 1e-12) {
    return { fx: 0, fy: 0 };
  }

  const rij = agent.radius + other.radius;
  const magnitude = config.socialMagnitude * Math.exp((rij - dist) / config.socialRange);

  const nx = dx / dist;
  const ny = dy / dist;

  return { fx: magnitude * nx, fy: magnitude * ny };
}

// ---------------------------------------------------------------------------
// Wall Repulsion
// ---------------------------------------------------------------------------

/**
 * Compute repulsion force from a wall segment on an agent.
 *
 * Uses point-to-line-segment closest distance, then applies exponential
 * repulsion perpendicular to the wall:
 *
 *   F = A_w * exp(-d_w / B_w) * n_w
 *
 * where d_w is the distance from the agent centre to the closest point
 * on the wall and n_w is the unit normal pointing from wall to agent.
 */
export function wallRepulsion(
  agent: SocialForceAgent,
  wall: WallSegment,
  config: SocialForceConfig,
): { fx: number; fy: number } {
  // Vector from wall start to wall end
  const wx = wall.x2 - wall.x1;
  const wy = wall.y2 - wall.y1;
  const wallLenSq = wx * wx + wy * wy;

  // Project agent position onto the wall line segment [0, 1]
  let t = 0;
  if (wallLenSq > 1e-12) {
    t = ((agent.x - wall.x1) * wx + (agent.y - wall.y1) * wy) / wallLenSq;
    t = Math.max(0, Math.min(1, t));
  }

  // Closest point on the wall
  const closestX = wall.x1 + t * wx;
  const closestY = wall.y1 + t * wy;

  // Vector from closest wall point to agent
  const dx = agent.x - closestX;
  const dy = agent.y - closestY;
  const dist = Math.sqrt(dx * dx + dy * dy);

  if (dist < 1e-12) {
    return { fx: 0, fy: 0 };
  }

  const magnitude = config.wallMagnitude * Math.exp((agent.radius - dist) / config.wallRange);
  const nx = dx / dist;
  const ny = dy / dist;

  return { fx: magnitude * nx, fy: magnitude * ny };
}

// ---------------------------------------------------------------------------
// Social Force Step
// ---------------------------------------------------------------------------

/**
 * Advance all agents by one time step using the social force model.
 *
 * Each agent experiences:
 *   1. Desired force:  (v_desired - v_current) / tau
 *   2. Social repulsion from all other agents
 *   3. Wall repulsion from all wall segments
 *
 * Velocities and positions are integrated with an explicit Euler step.
 * Returns a new array of agents (does not mutate inputs).
 */
export function socialForceStep(
  config: SocialForceConfig,
  agents: SocialForceAgent[],
  walls: WallSegment[],
): SocialForceAgent[] {
  const { desiredSpeed, relaxationTime, dt } = config;
  const nAgents = agents.length;

  const result: SocialForceAgent[] = new Array(nAgents);

  for (let i = 0; i < nAgents; i++) {
    const agent = agents[i]!;

    // ---------- Desired force ----------
    const dGoalX = agent.goalX - agent.x;
    const dGoalY = agent.goalY - agent.y;
    const distGoal = Math.sqrt(dGoalX * dGoalX + dGoalY * dGoalY);

    let desiredVx = 0;
    let desiredVy = 0;
    if (distGoal > 1e-12) {
      desiredVx = desiredSpeed * dGoalX / distGoal;
      desiredVy = desiredSpeed * dGoalY / distGoal;
    }

    let fx = (desiredVx - agent.vx) / relaxationTime;
    let fy = (desiredVy - agent.vy) / relaxationTime;

    // ---------- Social repulsion ----------
    for (let j = 0; j < nAgents; j++) {
      if (j === i) continue;
      const other = agents[j]!;
      const rep = socialRepulsion(agent, other, config);
      fx += rep.fx;
      fy += rep.fy;
    }

    // ---------- Wall repulsion ----------
    for (let w = 0; w < walls.length; w++) {
      const wall = walls[w]!;
      const rep = wallRepulsion(agent, wall, config);
      fx += rep.fx;
      fy += rep.fy;
    }

    // ---------- Euler integration ----------
    const newVx = agent.vx + fx * dt;
    const newVy = agent.vy + fy * dt;

    // Clamp speed to a reasonable maximum (3x desired speed)
    const speedMag = Math.sqrt(newVx * newVx + newVy * newVy);
    const maxSpeed = desiredSpeed * 3;
    let clampedVx = newVx;
    let clampedVy = newVy;
    if (speedMag > maxSpeed && speedMag > 1e-12) {
      clampedVx = newVx * maxSpeed / speedMag;
      clampedVy = newVy * maxSpeed / speedMag;
    }

    result[i] = {
      x: agent.x + clampedVx * dt,
      y: agent.y + clampedVy * dt,
      vx: clampedVx,
      vy: clampedVy,
      goalX: agent.goalX,
      goalY: agent.goalY,
      radius: agent.radius,
    };
  }

  return result;
}
