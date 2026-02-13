/**
 * PS-3: QUBO, Ising, and Potts Spin Glass Models
 *
 * Builds Quadratic Unconstrained Binary Optimization (QUBO) matrices for
 * event scheduling, converts between QUBO and Ising formulations,
 * implements Potts model construction, and provides SA/PT solver wrappers.
 */

import type {
  QUBOMatrix, IsingModel, PottsModel,
  EventSpec, RoomSpec, TimeslotSpec, SAConfig, PTConfig,
  PRNG,
} from './types.js'
import { CoolingSchedule, createPRNG } from './types.js'

import { simulatedAnnealing } from './sa.js'
import { parallelTempering } from './parallel-tempering.js'

// ---------------------------------------------------------------------------
// Upper-triangular storage helpers
// ---------------------------------------------------------------------------

/**
 * Index into the flattened upper-triangular array for element (i, j).
 * Canonicalizes so that lo <= hi. Storage order: row-major upper triangle
 * including diagonal.
 *
 *   idx(lo, hi, n) = lo * (2*n - lo - 1) / 2 + hi
 *
 * Total entries: n*(n+1)/2
 */
function upperTriIndex(n: number, i: number, j: number): number {
  const lo = Math.min(i, j)
  const hi = Math.max(i, j)
  return (lo * (2 * n - lo - 1)) / 2 + hi
}

/**
 * Get the value of Q[i][j] from upper-triangular QUBO storage.
 * Handles both (i,j) and (j,i) by canonicalizing to lo<=hi.
 */
export function getQuboEntry(qubo: QUBOMatrix, i: number, j: number): number {
  const idx = upperTriIndex(qubo.n, i, j)
  return qubo.data[idx] ?? 0
}

/**
 * Set the value of Q[i][j] in upper-triangular QUBO storage.
 * Handles both (i,j) and (j,i) by canonicalizing to lo<=hi.
 */
export function setQuboEntry(qubo: QUBOMatrix, i: number, j: number, val: number): void {
  const idx = upperTriIndex(qubo.n, i, j)
  qubo.data[idx] = val
}

/**
 * Add a value to Q[i][j] (accumulate). Canonicalizes to upper-triangular.
 */
function addQuboEntry(qubo: QUBOMatrix, i: number, j: number, val: number): void {
  const idx = upperTriIndex(qubo.n, i, j)
  qubo.data[idx] = (qubo.data[idx] ?? 0) + val
}

// ---------------------------------------------------------------------------
// QUBO Construction for Event Scheduling
// ---------------------------------------------------------------------------

/**
 * Build a QUBO matrix for event-room-timeslot scheduling.
 *
 * Binary variable x_{e,r,s} = 1 iff event e is assigned to room r in
 * timeslot s. Variables are flattened as:
 *   v = e * (nRooms * nSlots) + r * nSlots + s
 *
 * The Hamiltonian is:
 *   H = H_obj + lambda1*H_one_room + lambda2*H_no_conflict + lambda3*H_capacity
 *
 * Where:
 * - H_obj: negative preference weights (we minimize, so negative = good)
 * - H_one_room: (sum_v x_v - 1)^2 per event -- exactly-one constraint
 * - H_no_conflict: sum over (room, slot) pairs of pairwise conflict penalties
 * - H_capacity: max(0, guests - capacity)^2 on diagonal
 *
 * @param events - Array of event specifications
 * @param rooms - Array of room specifications
 * @param timeslots - Array of timeslot specifications
 * @param penaltyWeight - Lambda multiplier for constraint penalties (default 10)
 */
export function buildSchedulingQUBO(
  events: readonly EventSpec[],
  rooms: readonly RoomSpec[],
  timeslots: readonly TimeslotSpec[],
  penaltyWeight: number = 10,
): QUBOMatrix {
  const nEvents = events.length
  const nRooms = rooms.length
  const nSlots = timeslots.length
  const n = nEvents * nRooms * nSlots

  // Upper-triangular storage: n*(n+!))/2 entries
  const size = (n * (n + 1)) / 2
  const data = new Float64Array(size)
  const qubo: QUBOMatrix = { n, data }

  const lambda1 = penaltyWeight  // one-hot (each event assigned exactly once)
  const lambda2 = penaltyWeight  // no room-timeslot conflict
  const lambda3 = penaltyWeight  // capacity

  // Helper to compute variable index from (event, room, slot) triple
  const varIdx = (e: number, r: number, s: number): number =>
    e * (nRooms * nSlots) + r * nSlots + s

  // ----- H_obj: Objective (preference) on diagonal -----
  // Negate because QUBO minimizes: negative weight = preferred assignment.
  for (let e = 0; e < nEvents; e++) {
    const event = events[e]!
    for (let r = 0; r < nRooms; r++) {
      const room = rooms[r]!
      const pref = event.preferences[room.id] ?? 0
      for (let s = 0; s < nSlots; s++) {
        const v = varIdx(e, r, s)
        addQuboEntry(qubo, v, v, -pref)
      }
    }
  }

  // ----- H_one_room: Each event assigned to exactly one (room, timeslot) -----
  // For each event e: (sum_v x_v - 1)^2
  //   = sum_v x_v^2 - 2*sum_v x_v + 1 + 2*sum_{v<w} x_v x_w
  // Since x_v in {0,1}, x_v^2 = x_v:
  //   = -sum_v x_v + 1 + 2*sum_{v<w} x_v x_w
  // Diagonal: -lambda1 per variable (linear penalty)
  // Off-diagonal: +2*lambda1 per pair within same event
  for (let e = 0; e < nEvents; e++) {
    const vars: number[] = []
    for (let r = 0; r < nRooms; r++) {
      for (let s = 0; s < nSlots; s++) {
        vars.push(varIdx(e, r, s))
      }
    }

    // Diagonal contribution: -lambda1
    for (const v of vars) {
      addQuboEntry(qubo, v, v, -lambda1)
    }

    // Off-diagonal contribution: +2*lambda1 for all pairs
    for (let a = 0; a < vars.length; a++) {
      for (let b = a + 1; b < vars.length; b++) {
        addQuboEntry(qubo, vars[a]!, vars[b]!, 2 * lambda1)
      }
    }
  }

  // ----- H_no_conflict: No two events in the same (room, timeslot) -----
  // For each (room, timeslot) pair: penalty for assigning more than one event.
  // sum_{e1<e2} lambda2 * x_{e1,r,s} * x_{e2,r,s}
  for (let r = 0; r < nRooms; r++) {
    for (let s = 0; s < nSlots; s++) {
      for (let e1 = 0; e1 < nEvents; e1++) {
        for (let e2 = e1 + 1; e2 < nEvents; e2++) {
          const v1 = varIdx(e1, r, s)
          const v2 = varIdx(e2, r, s)
          addQuboEntry(qubo, v1, v2, lambda2)
        }
      }
    }
  }

  // ----- H_capacity: Penalty for exceeding room capacity -----
  // Diagonal penalty: lambda3 * max(0, guests - capacity)^2
  for (let e = 0; e < nEvents; e++) {
    const event = events[e]!
    for (let r = 0; r < nRooms; r++) {
      const room = rooms[r]!
      const overflow = Math.max(0, event.guests - room.capacity)
      if (overflow > 0) {
        const penalty = lambda3 * overflow * overflow
        for (let s = 0; s < nSlots; s++) {
          const v = varIdx(e, r, s)
          addQuboEntry(qubo, v, v, penalty)
        }
      }
    }
  }

  return qubo
}

// ---------------------------------------------------------------------------
// QUBO <-> Ising Conversion
// ---------------------------------------------------------------------------

/**
 * Convert a QUBO matrix to an Ising model.
 *
 * Uses the substitution x_i = (sigma_i + 1) / 2 where x in {0,1}
 * and sigma in {-1, +1}.
 *
 * Expanding x^T Q x with the substitution:
 *   Q_ii * x_i = Q_ii * (s_i + 1) / 2
 *   Q_ij * x_i * x_j = Q_ij * (s_i + 1)(s_j + 1) / 4   (i < j)
 *
 * The Ising Hamiltonian is H = -1/2 sum J_ij s_i s_j - sum h_i s_i.
 * Matching coefficients (negating to convert min-QUBO to Ising convention):
 *
 *   J_ij = -Q_ij / 4            (i != j)
 *   h_i  = -Q_ii / 2 - sum_{j!=i} Q_ij / 4
 *
 * The constant offset is discarded (does not affect optimization).
 */
export function quboToIsing(qubo: QUBOMatrix): IsingModel {
  const { n } = qubo
  const couplings = new Float64Array(n * n)
  const field = new Float64Array(n)

  // Diagonal terms contribute to h_i
  for (let i = 0; i < n; i++) {
    field[i] = -getQuboEntry(qubo, i, i) / 2
  }

  // Off-diagonal terms contribute to both J_ij and h_i
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const qij = getQuboEntry(qubo, i, j)
      if (qij === 0) continue

      const jVal = -qij / 4
      couplings[i * n + j] = jVal
      couplings[j * n + i] = jVal

      // Off-diagonal contribution to field
      field[i] = (field[i] ?? 0) + (-qij / 4)
      field[j] = (field[j] ?? 0) + (-qij / 4)
    }
  }

  return { n, couplings, field }
}

// ---------------------------------------------------------------------------
// Potts Model Construction
// ---------------------------------------------------------------------------

/**
 * Build a Potts model for event scheduling.
 *
 * Each event is a Potts spin with K = rooms.length possible states,
 * representing which room the event is assigned to.
 *
 * - Couplings: Negative coupling penalizes two events being assigned the
 *   same room (same Potts state), enforcing no-conflict constraints.
 *   couplings[i * n + j] < 0 means energy increases when s_i == s_j.
 * - Local field: Encodes room preference + capacity compatibility.
 *   field[i * K + q] is the bias for event i to be assigned room q.
 *   Positive = favorable, negative = penalized.
 *
 * @param events - Array of event specifications
 * @param rooms - Array of room specifications
 */
export function buildPottsScheduling(
  events: readonly EventSpec[],
  rooms: readonly RoomSpec[],
): PottsModel {
  const n = events.length
  const k = rooms.length

  // Coupling matrix [n*n]: energy bonus when s_i == s_j (negative = penalty)
  const couplings = new Float64Array(n * n)

  // Local field [n*k]: bias for spin i taking value q
  const field = new Float64Array(n * k)

  // ----- Local field: preference + capacity -----
  for (let i = 0; i < n; i++) {
    const event = events[i]!
    for (let q = 0; q < k; q++) {
      const room = rooms[q]!
      const pref = event.preferences[room.id] ?? 0
      const overflow = Math.max(0, event.guests - room.capacity)
      const capacityPenalty = overflow * overflow
      field[i * k + q] = pref - capacityPenalty
    }
  }

  // ----- Couplings: penalize same-room conflicts -----
  // Negative coupling = energy increases when spins match (same room).
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      couplings[i * n + j] = -1
      couplings[j * n + i] = -1
    }
  }

  return { n, k, couplings, field }
}

// ---------------------------------------------------------------------------
// Energy Evaluation
// ---------------------------------------------------------------------------

/**
 * Evaluate the QUBO energy H = x^T Q x for a binary solution vector.
 *
 * For binary x in {0,1}:
 *   H = sum_i Q_ii x_i + sum_{i<j} Q_ij x_i x_j
 *
 * (Diagonal terms simplify because x_i^2 = x_i for binary.)
 *
 * @param qubo - The QUBO matrix (upper-triangular storage)
 * @param solution - Binary solution vector (entries 0 or 1)
 * @returns The QUBO energy value
 */
export function evaluateQUBO(qubo: QUBOMatrix, solution: Uint8Array): number {
  const { n } = qubo
  let energy = 0

  for (let i = 0; i < n; i++) {
    if (solution[i] !== 1) continue
    // Diagonal: Q_ii * x_i
    energy += getQuboEntry(qubo, i, i)

    // Off-diagonal: Q_ij * x_i * x_j for j > i
    for (let j = i + 1; j < n; j++) {
      if (solution[j] !== 1) continue
      energy += getQuboEntry(qubo, i, j)
    }
  }

  return energy
}

/**
 * Evaluate the Ising energy H = -1/2 sum_{i,j} J_ij s_i s_j - sum_i h_i s_i
 *
 * Since J is symmetric, the sum over upper triangle is doubled:
 *   H = -sum_{i<j} J_ij s_i s_j - sum_i h_i s_i
 *
 * @param model - The Ising model (couplings + field)
 * @param spins - Spin configuration (entries -1 or +1)
 * @returns The Ising energy value
 */
export function evaluateIsing(model: IsingModel, spins: Int8Array): number {
  const { n, couplings, field } = model
  let energy = 0

  // Field term: -sum_i h_i s_i
  for (let i = 0; i < n; i++) {
    const hi = field[i] ?? 0
    const si = spins[i] ?? 0
    energy -= hi * si
  }

  // Coupling term: -sum_{i<j} J_ij s_i s_j
  // (Uses upper triangle only since J is symmetric; factor of 2 cancels
  //  with the 1/2 in the Hamiltonian definition.)
  for (let i = 0; i < n; i++) {
    const si = spins[i] ?? 0
    if (si === 0) continue
    for (let j = i + 1; j < n; j++) {
      const sj = spins[j] ?? 0
      if (sj === 0) continue
      const jij = couplings[i * n + j] ?? 0
      energy -= jij * si * sj
    }
  }

  return energy
}

// ---------------------------------------------------------------------------
// Spin <-> Binary Conversion
// ---------------------------------------------------------------------------

/**
 * Convert Ising spins {-1, +1} to binary variables {0, 1}.
 *
 * Uses the mapping: x_i = (s_i + 1) / 2
 *   s_i = +1  ->  x_i = 1
 *   s_i = -1  ->  x_i = 0
 */
export function spinsToBinary(spins: Int8Array): Uint8Array {
  const n = spins.length
  const binary = new Uint8Array(n)
  for (let i = 0; i < n; i++) {
    binary[i] = spins[i]! >= 0 ? 1 : 0
  }
  return binary
}

// ---------------------------------------------------------------------------
// SA / PT Solver Wrappers
// ---------------------------------------------------------------------------

/**
 * Solve a QUBO problem using Simulated Annealing.
 *
 * Converts the QUBO to an Ising model, runs SA with spin-flip neighbor
 * generation, then maps the best Ising spins back to binary variables.
 *
 * @param qubo   - Upper-triangular QUBO matrix
 * @param config - SA hyperparameters
 * @returns Binary solution as Uint8Array with values in {0, 1}
 */
export function solveQUBOSA(
  qubo: QUBOMatrix,
  config: SAConfig,
): Uint8Array {
  const ising = quboToIsing(qubo)
  const { n } = ising

  // Initial state: random spins encoded as Float64Array (+1 or -1)
  const rng = createPRNG(config.seed ?? 42)
  const initialState = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    initialState[i] = rng.random() < 0.5 ? -1 : 1
  }

  // Energy function: evaluate Ising energy from Float64Array state
  const energyFn = (state: Float64Array): number => {
    const spins = new Int8Array(state.length)
    for (let i = 0; i < state.length; i++) {
      spins[i] = state[i]! >= 0 ? 1 : -1
    }
    return evaluateIsing(ising, spins)
  }

  // Neighbor function: flip a random spin
  const neighborFn = (state: Float64Array, rng: PRNG): Float64Array => {
    const next = new Float64Array(state.length)
    next.set(state)
    const idx = Math.floor(rng.random() * state.length)
    next[idx] = -next[idx]!
    return next
  }

  const result = simulatedAnnealing(initialState, config, energyFn, neighborFn)

  // Convert best state (Float64Array of +/-1) to binary
  const spins = new Int8Array(n)
  for (let i = 0; i < n; i++) {
    spins[i] = result.bestState[i]! >= 0 ? 1 : -1
  }
  return spinsToBinary(spins)
}

/**
 * Solve a QUBO problem using Parallel Tempering.
 *
 * Converts the QUBO to an Ising model, runs PT with spin-flip neighbor
 * generation, then maps the best Ising spins back to binary variables.
 *
 * @param qubo   - Upper-triangular QUBO matrix
 * @param config - PT hyperparameters
 * @returns Binary solution as Uint8Array with values in {0, 1}
 */
export function solveQUBOPT(
  qubo: QUBOMatrix,
  config: PTConfig,
): Uint8Array {
  const ising = quboToIsing(qubo)
  const { n } = ising

  // Initial state: random spins encoded as Float64Array (+1 or -1)
  const rng = createPRNG(config.seed ?? 42)
  const initialState = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    initialState[i] = rng.random() < 0.5 ? -1 : 1
  }

  // Energy function: evaluate Ising energy from Float64Array state
  const energyFn = (state: Float64Array): number => {
    const spins = new Int8Array(state.length)
    for (let i = 0; i < state.length; i++) {
      spins[i] = state[i]! >= 0 ? 1 : -1
    }
    return evaluateIsing(ising, spins)
  }

  // Neighbor function: flip a random spin
  const neighborFn = (state: Float64Array, rng: PRNG): Float64Array => {
    const next = new Float64Array(state.length)
    next.set(state)
    const idx = Math.floor(rng.random() * state.length)
    next[idx] = -next[idx]!
    return next
  }

  const result = parallelTempering(initialState, energyFn, neighborFn, config)

  // Convert best state (Float64Array of +/-1) to binary
  const spins = new Int8Array(n)
  for (let i = 0; i < n; i++) {
    spins[i] = result.bestState[i]! >= 0 ? 1 : -1
  }
  return spinsToBinary(spins)
}
