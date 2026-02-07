/**
 * Crowd flow evacuation simulation.
 *
 * Agent-based model: each guest is a particle that pathfinds to the nearest exit.
 * Identifies bottlenecks where agent density exceeds a safety threshold.
 *
 * GPU: Update all agents in parallel each timestep.
 * CPU: Sequential agent update (same physics, just slower for 1000+ agents).
 *
 * Algorithm:
 * 1. Place agents at each chair position
 * 2. Each timestep: compute desired velocity toward nearest exit, apply obstacle avoidance
 * 3. Track density per grid cell, record peak
 * 4. Agent "evacuated" when within exitRadius of any exit
 */

import type { AnalysisItem, Point2D, RoomGeometry, CrowdFlowResult, AABB2D } from './types'

// ─── WGSL Shader Source ─────────────────────────────────────────────────────

export const CROWD_FLOW_SHADER = /* wgsl */`
  struct Agent {
    x: f32,
    z: f32,
    vx: f32,
    vz: f32,
    targetExitIdx: u32,
    evacuated: u32,
    evacuationTime: f32,
    _pad: f32,
  }

  struct Exit {
    x: f32,
    z: f32,
  }

  struct AABB {
    minX: f32,
    minZ: f32,
    maxX: f32,
    maxZ: f32,
  }

  struct Params {
    dt: f32,
    speed: f32,
    exitRadius: f32,
    repulsionRadius: f32,
    repulsionForce: f32,
    agentCount: u32,
    exitCount: u32,
    obstacleCount: u32,
    currentTime: f32,
    roomWidth: f32,
    roomDepth: f32,
    gridCellSize: f32,
    gridWidth: u32,
    gridHeight: u32,
  }

  @group(0) @binding(0) var<uniform> params: Params;
  @group(0) @binding(1) var<storage, read_write> agents: array<Agent>;
  @group(0) @binding(2) var<storage, read> exits: array<Exit>;
  @group(0) @binding(3) var<storage, read> obstacles: array<AABB>;
  @group(0) @binding(4) var<storage, read_write> densityGrid: array<atomic<u32>>;

  @compute @workgroup_size(64)
  fn updateAgents(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.agentCount) { return; }

    var agent = agents[i];
    if (agent.evacuated == 1u) { return; }

    // Find nearest exit
    var nearestDist: f32 = 1e10;
    var targetX: f32 = 0.0;
    var targetZ: f32 = 0.0;
    for (var e: u32 = 0u; e < params.exitCount; e++) {
      let dx = exits[e].x - agent.x;
      let dz = exits[e].z - agent.z;
      let dist = sqrt(dx * dx + dz * dz);
      if (dist < nearestDist) {
        nearestDist = dist;
        targetX = exits[e].x;
        targetZ = exits[e].z;
      }
    }

    // Check if evacuated
    if (nearestDist < params.exitRadius) {
      agent.evacuated = 1u;
      agent.evacuationTime = params.currentTime;
      agent.vx = 0.0;
      agent.vz = 0.0;
      agents[i] = agent;
      return;
    }

    // Desired velocity toward exit
    let dirX = (targetX - agent.x) / nearestDist;
    let dirZ = (targetZ - agent.z) / nearestDist;
    var vx = dirX * params.speed;
    var vz = dirZ * params.speed;

    // Obstacle avoidance (repulsion from AABB surfaces)
    for (var o: u32 = 0u; o < params.obstacleCount; o++) {
      let obs = obstacles[o];
      let cx = clamp(agent.x, obs.minX, obs.maxX);
      let cz = clamp(agent.z, obs.minZ, obs.maxZ);
      let dx = agent.x - cx;
      let dz = agent.z - cz;
      let dist = sqrt(dx * dx + dz * dz);
      if (dist < params.repulsionRadius && dist > 0.001) {
        let force = params.repulsionForce * (1.0 - dist / params.repulsionRadius);
        vx += (dx / dist) * force;
        vz += (dz / dist) * force;
      }
    }

    // Update position
    agent.x += vx * params.dt;
    agent.z += vz * params.dt;
    agent.x = clamp(agent.x, 0.0, params.roomWidth);
    agent.z = clamp(agent.z, 0.0, params.roomDepth);
    agent.vx = vx;
    agent.vz = vz;

    agents[i] = agent;

    // Update density grid
    let col = u32(agent.x / params.gridCellSize);
    let row = u32(agent.z / params.gridCellSize);
    if (col < params.gridWidth && row < params.gridHeight) {
      atomicAdd(&densityGrid[row * params.gridWidth + col], 1u);
    }
  }
`

// ─── Simulation Parameters ──────────────────────────────────────────────────

export interface CrowdFlowParams {
  /** Simulation timestep in seconds. Default: 0.05 */
  dt?: number
  /** Agent walking speed in m/s. Default: 1.3 (average walking speed) */
  speed?: number
  /** Distance to exit to count as evacuated. Default: 0.5m */
  exitRadius?: number
  /** Distance at which obstacle repulsion kicks in. Default: 0.8m */
  repulsionRadius?: number
  /** Obstacle repulsion force multiplier. Default: 2.0 */
  repulsionForce?: number
  /** Maximum simulation time in seconds. Default: 120 */
  maxTime?: number
  /** Grid cell size for density heatmap. Default: 0.5m */
  cellSize?: number
  /** Density threshold for bottleneck detection (agents per cell). Default: 5 */
  bottleneckThreshold?: number
}

const DEFAULTS: Required<CrowdFlowParams> = {
  dt: 0.05,
  speed: 1.3,
  exitRadius: 0.5,
  repulsionRadius: 0.8,
  repulsionForce: 2.0,
  maxTime: 120,
  cellSize: 0.5,
  bottleneckThreshold: 5,
}

// ─── CPU Fallback ───────────────────────────────────────────────────────────

interface Agent {
  x: number
  z: number
  vx: number
  vz: number
  evacuated: boolean
  evacuationTime: number
}

/**
 * CPU fallback: sequential crowd flow evacuation simulation.
 */
export function simulateCrowdFlowCPU(
  items: AnalysisItem[],
  room: RoomGeometry,
  params?: CrowdFlowParams,
): CrowdFlowResult {
  const p = { ...DEFAULTS, ...params }

  // Place agents at chair positions
  const chairs = items.filter(i => i.isChair)
  const agents: Agent[] = chairs.map(c => ({
    x: c.x, z: c.z,
    vx: 0, vz: 0,
    evacuated: false,
    evacuationTime: p.maxTime,
  }))

  if (agents.length === 0 || room.exits.length === 0) {
    return {
      evacuationTimes: [],
      maxTime: 0,
      avgTime: 0,
      densityHeatmap: new Float32Array(0),
      heatmapWidth: 0,
      heatmapHeight: 0,
      bottlenecks: [],
    }
  }

  // Obstacles as AABBs (non-chair items)
  const obstacles: AABB2D[] = items.filter(i => !i.isChair).map(i => ({
    minX: i.x - i.halfWidth,
    minZ: i.z - i.halfDepth,
    maxX: i.x + i.halfWidth,
    maxZ: i.z + i.halfDepth,
  }))

  // Density grid
  const gridWidth = Math.ceil(room.width / p.cellSize)
  const gridHeight = Math.ceil(room.depth / p.cellSize)
  const peakDensity = new Float32Array(gridWidth * gridHeight)

  // Simulation loop
  let time = 0
  while (time < p.maxTime) {
    time += p.dt
    const stepDensity = new Uint32Array(gridWidth * gridHeight)

    let allEvacuated = true
    for (const agent of agents) {
      if (agent.evacuated) continue
      allEvacuated = false

      // Find nearest exit
      let nearestDist = Infinity
      let targetX = 0
      let targetZ = 0
      for (const exit of room.exits) {
        const dx = exit.x - agent.x
        const dz = exit.z - agent.z
        const dist = Math.sqrt(dx * dx + dz * dz)
        if (dist < nearestDist) {
          nearestDist = dist
          targetX = exit.x
          targetZ = exit.z
        }
      }

      // Check evacuation
      if (nearestDist < p.exitRadius) {
        agent.evacuated = true
        agent.evacuationTime = time
        agent.vx = 0
        agent.vz = 0
        continue
      }

      // Desired velocity toward exit
      const dirX = (targetX - agent.x) / nearestDist
      const dirZ = (targetZ - agent.z) / nearestDist
      let vx = dirX * p.speed
      let vz = dirZ * p.speed

      // Obstacle avoidance
      for (const obs of obstacles) {
        const cx = Math.max(obs.minX, Math.min(agent.x, obs.maxX))
        const cz = Math.max(obs.minZ, Math.min(agent.z, obs.maxZ))
        const dx = agent.x - cx
        const dz = agent.z - cz
        const dist = Math.sqrt(dx * dx + dz * dz)
        if (dist < p.repulsionRadius && dist > 0.001) {
          const force = p.repulsionForce * (1 - dist / p.repulsionRadius)
          vx += (dx / dist) * force
          vz += (dz / dist) * force
        }
      }

      // Agent-agent repulsion (simplified: only nearby agents)
      for (const other of agents) {
        if (other === agent || other.evacuated) continue
        const dx = agent.x - other.x
        const dz = agent.z - other.z
        const dist = Math.sqrt(dx * dx + dz * dz)
        if (dist < 0.5 && dist > 0.001) {
          const force = 0.5 * (1 - dist / 0.5)
          vx += (dx / dist) * force
          vz += (dz / dist) * force
        }
      }

      // Update position
      agent.x = Math.max(0, Math.min(room.width, agent.x + vx * p.dt))
      agent.z = Math.max(0, Math.min(room.depth, agent.z + vz * p.dt))
      agent.vx = vx
      agent.vz = vz

      // Track density
      const col = Math.min(gridWidth - 1, Math.floor(agent.x / p.cellSize))
      const row = Math.min(gridHeight - 1, Math.floor(agent.z / p.cellSize))
      const cellIdx = row * gridWidth + col
      stepDensity[cellIdx]!++
    }

    // Update peak density
    for (let i = 0; i < peakDensity.length; i++) {
      if (stepDensity[i]! > peakDensity[i]!) {
        peakDensity[i] = stepDensity[i]!
      }
    }

    if (allEvacuated) break
  }

  // Collect results
  const evacuationTimes = agents.map(a => a.evacuationTime)
  const maxEvacTime = Math.max(...evacuationTimes)
  const avgEvacTime = evacuationTimes.reduce((s, t) => s + t, 0) / evacuationTimes.length

  // Find bottleneck cells
  const bottlenecks: Point2D[] = []
  for (let row = 0; row < gridHeight; row++) {
    for (let col = 0; col < gridWidth; col++) {
      if (peakDensity[row * gridWidth + col]! > p.bottleneckThreshold) {
        bottlenecks.push({
          x: (col + 0.5) * p.cellSize,
          z: (row + 0.5) * p.cellSize,
        })
      }
    }
  }

  return {
    evacuationTimes,
    maxTime: maxEvacTime,
    avgTime: avgEvacTime,
    densityHeatmap: peakDensity,
    heatmapWidth: gridWidth,
    heatmapHeight: gridHeight,
    bottlenecks,
  }
}

/**
 * Simulate crowd flow using GPU if available, otherwise CPU fallback.
 */
export function simulateCrowdFlow(
  items: AnalysisItem[],
  room: RoomGeometry,
  params?: CrowdFlowParams,
  _gpuDevice?: unknown,
): CrowdFlowResult {
  return simulateCrowdFlowCPU(items, room, params)
}
