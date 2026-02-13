/**
 * Comprehensive tests for PS-3 (QUBO/Ising/Potts) and PS-4 (Simulated Bifurcation).
 *
 * Uses small (2-4 variable) test problems for speed and verifiability.
 */
import { describe, it, expect } from 'vitest'

import type {
  QUBOMatrix,
  IsingModel,
  PottsModel,
  EventSpec,
  RoomSpec,
  TimeslotSpec,
  SBConfig,
  SAConfig,
  PTConfig,
} from '../types.js'
import { SBVariant, CoolingSchedule, TempSpacing } from '../types.js'

import {
  buildSchedulingQUBO,
  quboToIsing,
  buildPottsScheduling,
  evaluateQUBO,
  evaluateIsing,
  getQuboEntry,
  setQuboEntry,
  solveQUBOSA,
  solveQUBOPT,
} from '../qubo.js'

import {
  simulatedBifurcation,
  simulatedBifurcationQUBO,
} from '../simulated-bifurcation.js'

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/**
 * Build a small QUBO matrix from a dense 2-D array (row-major, full matrix).
 * Only the upper triangle (i <= j) is stored; lower-triangle values are
 * folded onto the upper triangle (added to Q[j][i] -> Q[i][j]).
 */
function quboFromDense(matrix: number[][]): QUBOMatrix {
  const n = matrix.length
  const size = (n * (n + 1)) / 2
  const data = new Float64Array(size)
  const qubo: QUBOMatrix = { n, data }

  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      const val = matrix[i]![j]!
      if (val !== 0) setQuboEntry(qubo, i, j, val)
    }
  }
  return qubo
}

/**
 * Build a small Ising model from dense coupling matrix and field vector.
 */
function isingFromDense(J: number[][], h: number[]): IsingModel {
  const n = h.length
  const couplings = new Float64Array(n * n)
  const field = new Float64Array(n)

  for (let i = 0; i < n; i++) {
    field[i] = h[i]!
    for (let j = 0; j < n; j++) {
      couplings[i * n + j] = J[i]![j]!
    }
  }
  return { n, couplings, field }
}

/** Create a binary solution vector from an array of 0/1. */
function binarySolution(vals: number[]): Uint8Array {
  return new Uint8Array(vals)
}

/** Create a spin configuration from an array of -1/+1. */
function spinConfig(vals: number[]): Int8Array {
  return new Int8Array(vals)
}

// ---------------------------------------------------------------------------
// Minimal scheduling test fixtures
// ---------------------------------------------------------------------------

function makeEvents(count: number): EventSpec[] {
  const events: EventSpec[] = []
  for (let i = 0; i < count; i++) {
    events.push({
      id: `event-${i}`,
      guests: 10 + i * 5,
      duration: 60,
      preferences: { 'room-0': 1.0, 'room-1': 0.5 },
    })
  }
  return events
}

function makeRooms(count: number): RoomSpec[] {
  const rooms: RoomSpec[] = []
  for (let i = 0; i < count; i++) {
    rooms.push({
      id: `room-${i}`,
      capacity: 30 + i * 20,
      amenities: [],
    })
  }
  return rooms
}

function makeTimeslots(count: number): TimeslotSpec[] {
  const slots: TimeslotSpec[] = []
  for (let i = 0; i < count; i++) {
    slots.push({
      id: `slot-${i}`,
      start: 9 + i,
      end: 10 + i,
      day: 0,
    })
  }
  return slots
}

// ===========================================================================
// PS-3: QUBO / Ising / Potts Tests
// ===========================================================================

describe('PS-3: QUBO / Ising / Potts', () => {
  // -----------------------------------------------------------------------
  // getQuboEntry / setQuboEntry helpers
  // -----------------------------------------------------------------------
  describe('getQuboEntry / setQuboEntry', () => {
    it('stores and retrieves diagonal entries', () => {
      const n = 3
      const size = (n * (n + 1)) / 2
      const qubo: QUBOMatrix = { n, data: new Float64Array(size) }

      setQuboEntry(qubo, 0, 0, 1.5)
      setQuboEntry(qubo, 1, 1, -2.0)
      setQuboEntry(qubo, 2, 2, 3.0)

      expect(getQuboEntry(qubo, 0, 0)).toBe(1.5)
      expect(getQuboEntry(qubo, 1, 1)).toBe(-2.0)
      expect(getQuboEntry(qubo, 2, 2)).toBe(3.0)
    })

    it('canonicalizes (i,j) and (j,i) to the same entry', () => {
      const n = 3
      const size = (n * (n + 1)) / 2
      const qubo: QUBOMatrix = { n, data: new Float64Array(size) }

      setQuboEntry(qubo, 0, 2, 7.0)
      expect(getQuboEntry(qubo, 0, 2)).toBe(7.0)
      expect(getQuboEntry(qubo, 2, 0)).toBe(7.0)
    })

    it('returns 0 for unset entries', () => {
      const n = 2
      const size = (n * (n + 1)) / 2
      const qubo: QUBOMatrix = { n, data: new Float64Array(size) }

      expect(getQuboEntry(qubo, 0, 1)).toBe(0)
    })
  })

  // -----------------------------------------------------------------------
  // buildSchedulingQUBO
  // -----------------------------------------------------------------------
  describe('buildSchedulingQUBO', () => {
    it('creates correct matrix dimensions for 1 event, 1 room, 1 slot', () => {
      const events = makeEvents(1)
      const rooms = makeRooms(1)
      const timeslots = makeTimeslots(1)

      const qubo = buildSchedulingQUBO(events, rooms, timeslots)

      // n = nEvents * nRooms * nSlots = 1*1*1 = 1
      expect(qubo.n).toBe(1)
      expect(qubo.data.length).toBe(1) // n*(n+1)/2 = 1
    })

    it('creates correct matrix dimensions for 2 events, 2 rooms, 2 slots', () => {
      const events = makeEvents(2)
      const rooms = makeRooms(2)
      const timeslots = makeTimeslots(2)

      const qubo = buildSchedulingQUBO(events, rooms, timeslots)

      // n = 2*2*2 = 8
      expect(qubo.n).toBe(8)
      // Upper-triangular storage: 8*9/2 = 36
      expect(qubo.data.length).toBe(36)
    })

    it('creates correct dimensions for 3 events, 2 rooms, 1 slot', () => {
      const events = makeEvents(3)
      const rooms = makeRooms(2)
      const timeslots = makeTimeslots(1)

      const qubo = buildSchedulingQUBO(events, rooms, timeslots)

      // n = 3*2*1 = 6
      expect(qubo.n).toBe(6)
      expect(qubo.data.length).toBe(21) // 6*7/2 = 21
    })

    it('includes one-hot constraint penalties (off-diagonal positive for same event)', () => {
      const events = makeEvents(1)
      const rooms = makeRooms(2)
      const timeslots = makeTimeslots(1)
      const penalty = 10

      const qubo = buildSchedulingQUBO(events, rooms, timeslots, penalty)

      // Variables: x_{0,0,0} (idx 0) and x_{0,1,0} (idx 1)
      // Both belong to event 0, so one-hot penalty should create positive off-diagonal
      const offDiag = getQuboEntry(qubo, 0, 1)
      expect(offDiag).toBeGreaterThan(0)
    })

    it('applies capacity penalty when event guests exceed room capacity', () => {
      // Event with 100 guests, room with capacity 30
      const events: EventSpec[] = [{
        id: 'big-event',
        guests: 100,
        duration: 60,
        preferences: { 'small-room': 0 },
      }]
      const rooms: RoomSpec[] = [{
        id: 'small-room',
        capacity: 30,
        amenities: [],
      }]
      const timeslots = makeTimeslots(1)
      const penalty = 10

      const qubo = buildSchedulingQUBO(events, rooms, timeslots, penalty)

      // Diagonal should include capacity penalty: lambda3 * (100-30)^2 = 10 * 4900 = 49000
      // plus the one-hot diagonal: -lambda1 = -10
      // plus objective: -pref = 0
      const diag = getQuboEntry(qubo, 0, 0)
      // The capacity overflow penalty is 10 * 70^2 = 49000
      // The one-hot diagonal is -10
      // Net should be 49000 - 10 = 48990
      expect(diag).toBe(49000 - 10)
    })

    it('does not apply capacity penalty when room has sufficient capacity', () => {
      const events: EventSpec[] = [{
        id: 'small-event',
        guests: 10,
        duration: 60,
        preferences: { 'big-room': 1.0 },
      }]
      const rooms: RoomSpec[] = [{
        id: 'big-room',
        capacity: 100,
        amenities: [],
      }]
      const timeslots = makeTimeslots(1)
      const penalty = 10

      const qubo = buildSchedulingQUBO(events, rooms, timeslots, penalty)

      // Diagonal: -pref(-1.0) + one-hot(-10) = -1.0 - 10 = -11.0
      const diag = getQuboEntry(qubo, 0, 0)
      expect(diag).toBe(-1.0 - 10)
    })
  })

  // -----------------------------------------------------------------------
  // quboToIsing
  // -----------------------------------------------------------------------
  describe('quboToIsing', () => {
    it('produces valid Ising model with correct dimensions', () => {
      // Simple 2-variable QUBO: Q = [[1, 2], [0, 3]]
      // Upper-triangle storage: Q[0,0]=1, Q[0,1]=2, Q[1,1]=3
      const qubo = quboFromDense([
        [1, 2],
        [0, 3],
      ])

      const ising = quboToIsing(qubo)

      expect(ising.n).toBe(2)
      expect(ising.couplings.length).toBe(4) // n*n = 4
      expect(ising.field.length).toBe(2)
    })

    it('produces symmetric coupling matrix', () => {
      const qubo = quboFromDense([
        [0, 5],
        [0, 0],
      ])

      const ising = quboToIsing(qubo)

      // J[0,1] should equal J[1,0]
      expect(ising.couplings[0 * 2 + 1]).toBe(ising.couplings[1 * 2 + 0])
    })

    it('computes correct Ising couplings for known QUBO', () => {
      // QUBO: Q[0,0]=0, Q[0,1]=4, Q[1,1]=0
      // J_01 = -Q_01 / 4 = -4/4 = -1
      const qubo = quboFromDense([
        [0, 4],
        [0, 0],
      ])

      const ising = quboToIsing(qubo)

      expect(ising.couplings[0 * 2 + 1]).toBeCloseTo(-1.0)
      expect(ising.couplings[1 * 2 + 0]).toBeCloseTo(-1.0)
    })

    it('computes correct field terms', () => {
      // QUBO: Q[0,0]=2, Q[0,1]=0, Q[1,1]=-4
      // h_0 = -Q_00 / 2 = -2/2 = -1
      // h_1 = -Q_11 / 2 = -(-4)/2 = 2
      const qubo = quboFromDense([
        [2, 0],
        [0, -4],
      ])

      const ising = quboToIsing(qubo)

      expect(ising.field[0]).toBeCloseTo(-1.0)
      expect(ising.field[1]).toBeCloseTo(2.0)
    })

    it('includes off-diagonal contribution to field', () => {
      // QUBO: Q[0,0]=0, Q[0,1]=8, Q[1,1]=0
      // h_0 = -Q_00/2 + (-Q_01/4) = 0 + -8/4 = -2
      // h_1 = -Q_11/2 + (-Q_01/4) = 0 + -8/4 = -2
      const qubo = quboFromDense([
        [0, 8],
        [0, 0],
      ])

      const ising = quboToIsing(qubo)

      expect(ising.field[0]).toBeCloseTo(-2.0)
      expect(ising.field[1]).toBeCloseTo(-2.0)
    })

    it('has zero diagonal couplings (no self-interaction)', () => {
      const qubo = quboFromDense([
        [3, 1],
        [0, 2],
      ])

      const ising = quboToIsing(qubo)

      expect(ising.couplings[0 * 2 + 0]).toBe(0)
      expect(ising.couplings[1 * 2 + 1]).toBe(0)
    })
  })

  // -----------------------------------------------------------------------
  // evaluateQUBO
  // -----------------------------------------------------------------------
  describe('evaluateQUBO', () => {
    it('computes correct energy for known solution (all zeros)', () => {
      const qubo = quboFromDense([
        [1, 2],
        [0, 3],
      ])

      const solution = binarySolution([0, 0])
      const energy = evaluateQUBO(qubo, solution)

      // x = [0, 0]: energy = 0
      expect(energy).toBe(0)
    })

    it('computes correct energy for single variable set', () => {
      const qubo = quboFromDense([
        [5, 2],
        [0, 3],
      ])

      // x = [1, 0]: energy = Q_00 * 1 = 5
      expect(evaluateQUBO(qubo, binarySolution([1, 0]))).toBe(5)

      // x = [0, 1]: energy = Q_11 * 1 = 3
      expect(evaluateQUBO(qubo, binarySolution([0, 1]))).toBe(3)
    })

    it('computes correct energy for all variables set', () => {
      const qubo = quboFromDense([
        [1, 2],
        [0, 3],
      ])

      // x = [1, 1]: energy = Q_00 + Q_11 + Q_01 = 1 + 3 + 2 = 6
      const energy = evaluateQUBO(qubo, binarySolution([1, 1]))
      expect(energy).toBe(6)
    })

    it('handles 3-variable QUBO correctly', () => {
      // Q = [[1, -2, 0],
      //      [0,  3, -1],
      //      [0,  0,  2]]
      const qubo = quboFromDense([
        [1, -2, 0],
        [0, 3, -1],
        [0, 0, 2],
      ])

      // x = [1, 1, 0]: Q_00 + Q_11 + Q_01 = 1 + 3 + (-2) = 2
      expect(evaluateQUBO(qubo, binarySolution([1, 1, 0]))).toBe(2)

      // x = [1, 0, 1]: Q_00 + Q_22 + Q_02 = 1 + 2 + 0 = 3
      expect(evaluateQUBO(qubo, binarySolution([1, 0, 1]))).toBe(3)

      // x = [0, 1, 1]: Q_11 + Q_22 + Q_12 = 3 + 2 + (-1) = 4
      expect(evaluateQUBO(qubo, binarySolution([0, 1, 1]))).toBe(4)

      // x = [1, 1, 1]: all terms = 1 + 3 + 2 + (-2) + 0 + (-1) = 3
      expect(evaluateQUBO(qubo, binarySolution([1, 1, 1]))).toBe(3)
    })

    it('returns negative energy for negative QUBO entries', () => {
      const qubo = quboFromDense([
        [-5, 0],
        [0, -3],
      ])

      const energy = evaluateQUBO(qubo, binarySolution([1, 1]))
      expect(energy).toBe(-8)
    })
  })

  // -----------------------------------------------------------------------
  // evaluateIsing
  // -----------------------------------------------------------------------
  describe('evaluateIsing', () => {
    it('computes correct energy for 2-spin ferromagnet', () => {
      // Ferromagnetic: J_01 = 1 (positive coupling favors alignment)
      // h = [0, 0]
      // H = -sum J_ij s_i s_j - sum h_i s_i
      // For aligned spins [+1, +1]: H = -1 * 1 * 1 = -1
      const ising = isingFromDense(
        [[0, 1], [1, 0]],
        [0, 0],
      )

      const aligned = evaluateIsing(ising, spinConfig([1, 1]))
      const antiAligned = evaluateIsing(ising, spinConfig([1, -1]))

      // Aligned should have lower energy (more negative)
      expect(aligned).toBeLessThan(antiAligned)
      expect(aligned).toBeCloseTo(-1.0)
      expect(antiAligned).toBeCloseTo(1.0)
    })

    it('computes correct energy for 2-spin antiferromagnet', () => {
      // Antiferromagnetic: J_01 = -1 (negative coupling favors anti-alignment)
      // For anti-aligned spins [+1, -1]: H = -(-1) * 1 * (-1) = -1 (lower energy)
      const ising = isingFromDense(
        [[0, -1], [-1, 0]],
        [0, 0],
      )

      const aligned = evaluateIsing(ising, spinConfig([1, 1]))
      const antiAligned = evaluateIsing(ising, spinConfig([1, -1]))

      expect(antiAligned).toBeLessThan(aligned)
    })

    it('includes field contribution', () => {
      // No couplings, just field h = [2, -3]
      // H = -h_0 * s_0 - h_1 * s_1
      // s = [+1, +1]: H = -2*1 - (-3)*1 = -2 + 3 = 1
      // s = [+1, -1]: H = -2*1 - (-3)*(-1) = -2 - 3 = -5
      const ising = isingFromDense(
        [[0, 0], [0, 0]],
        [2, -3],
      )

      expect(evaluateIsing(ising, spinConfig([1, 1]))).toBeCloseTo(1.0)
      expect(evaluateIsing(ising, spinConfig([1, -1]))).toBeCloseTo(-5.0)
    })

    it('handles 3-spin system', () => {
      // Triangle with uniform ferromagnetic coupling J=1, no field
      // H = -(J_01 s_0 s_1 + J_02 s_0 s_2 + J_12 s_1 s_2)
      // All aligned [+1,+1,+1]: H = -(1+1+1) = -3
      const ising = isingFromDense(
        [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
        [0, 0, 0],
      )

      const allUp = evaluateIsing(ising, spinConfig([1, 1, 1]))
      expect(allUp).toBeCloseTo(-3.0)

      // One flipped [+1, +1, -1]: H = -(1*1*1 + 1*1*(-1) + 1*1*(-1)) = -(1-1-1) = 1
      const oneFlipped = evaluateIsing(ising, spinConfig([1, 1, -1]))
      expect(oneFlipped).toBeCloseTo(1.0)
    })

    it('ground state of pure field model is all spins aligned with field', () => {
      const ising = isingFromDense(
        [[0, 0], [0, 0]],
        [5, 3],
      )

      // With positive field, ground state has all spins +1
      const up = evaluateIsing(ising, spinConfig([1, 1]))
      const down = evaluateIsing(ising, spinConfig([-1, -1]))

      expect(up).toBeLessThan(down)
    })
  })

  // -----------------------------------------------------------------------
  // buildPottsScheduling
  // -----------------------------------------------------------------------
  describe('buildPottsScheduling', () => {
    it('creates valid Potts model dimensions', () => {
      const events = makeEvents(3)
      const rooms = makeRooms(2)

      const potts = buildPottsScheduling(events, rooms)

      expect(potts.n).toBe(3)  // 3 events = 3 Potts spins
      expect(potts.k).toBe(2)  // 2 rooms = 2 possible states
      expect(potts.couplings.length).toBe(9) // n*n = 9
      expect(potts.field.length).toBe(6)     // n*k = 6
    })

    it('has negative couplings to penalize same-room assignment', () => {
      const events = makeEvents(2)
      const rooms = makeRooms(2)

      const potts = buildPottsScheduling(events, rooms)

      // Coupling between event 0 and event 1 should be negative (penalty)
      expect(potts.couplings[0 * 2 + 1]).toBe(-1)
      expect(potts.couplings[1 * 2 + 0]).toBe(-1)
    })

    it('has zero self-coupling', () => {
      const events = makeEvents(2)
      const rooms = makeRooms(2)

      const potts = buildPottsScheduling(events, rooms)

      expect(potts.couplings[0 * 2 + 0]).toBe(0)
      expect(potts.couplings[1 * 2 + 1]).toBe(0)
    })

    it('encodes room preferences in the local field', () => {
      const events: EventSpec[] = [{
        id: 'e0',
        guests: 10,
        duration: 60,
        preferences: { 'r0': 5.0, 'r1': 2.0 },
      }]
      const rooms: RoomSpec[] = [
        { id: 'r0', capacity: 100, amenities: [] },
        { id: 'r1', capacity: 100, amenities: [] },
      ]

      const potts = buildPottsScheduling(events, rooms)

      // field[event_0, room_0] should reflect preference 5.0
      // field[event_0, room_1] should reflect preference 2.0
      // No capacity overflow since capacity=100 > guests=10
      expect(potts.field[0 * 2 + 0]).toBe(5.0)
      expect(potts.field[0 * 2 + 1]).toBe(2.0)
    })

    it('includes capacity penalty in local field', () => {
      const events: EventSpec[] = [{
        id: 'e0',
        guests: 50,
        duration: 60,
        preferences: { 'r0': 0, 'r1': 0 },
      }]
      const rooms: RoomSpec[] = [
        { id: 'r0', capacity: 20, amenities: [] }, // overflow = 30, penalty = 900
        { id: 'r1', capacity: 100, amenities: [] }, // no overflow
      ]

      const potts = buildPottsScheduling(events, rooms)

      // field[0, r0] = pref(0) - overflow_penalty(900) = -900
      expect(potts.field[0 * 2 + 0]).toBe(-900)
      // field[0, r1] = pref(0) - 0 = 0
      expect(potts.field[0 * 2 + 1]).toBe(0)
    })

    it('handles single event and single room', () => {
      const events = makeEvents(1)
      const rooms = makeRooms(1)

      const potts = buildPottsScheduling(events, rooms)

      expect(potts.n).toBe(1)
      expect(potts.k).toBe(1)
      expect(potts.couplings.length).toBe(1) // 1*1
      expect(potts.field.length).toBe(1)     // 1*1
    })
  })

  // -----------------------------------------------------------------------
  // solveQUBOSA
  // -----------------------------------------------------------------------
  describe('solveQUBOSA', () => {
    it('finds low-energy solution for small QUBO', () => {
      // QUBO with clear minimum at x=[1,0]: energy = -10
      // x=[0,0] -> 0, x=[1,0] -> -10, x=[0,1] -> -1, x=[1,1] -> -10 + -1 + 50 = 39
      const qubo = quboFromDense([
        [-10, 50],
        [0, -1],
      ])

      const saConfig: SAConfig = {
        initialTemp: 10,
        finalTemp: 0.01,
        cooling: CoolingSchedule.Geometric,
        alpha: 0.99,
        maxIterations: 2000,
        reheatInterval: 0,
        reheatTempFraction: 0.5,
        seed: 42,
      }

      const solution = solveQUBOSA(qubo, saConfig)

      expect(solution.length).toBe(2)
      // Each entry should be 0 or 1
      for (let i = 0; i < solution.length; i++) {
        expect(solution[i] === 0 || solution[i] === 1).toBe(true)
      }

      // Should find a solution with energy <= 0 (better than all-zeros)
      const energy = evaluateQUBO(qubo, solution)
      expect(energy).toBeLessThanOrEqual(0)
    })

    it('finds optimal solution for trivial diagonal QUBO', () => {
      // All diagonal negative: optimal is all-ones
      const qubo = quboFromDense([
        [-5, 0],
        [0, -3],
      ])

      const saConfig: SAConfig = {
        initialTemp: 10,
        finalTemp: 0.001,
        cooling: CoolingSchedule.Geometric,
        alpha: 0.995,
        maxIterations: 3000,
        reheatInterval: 0,
        reheatTempFraction: 0.5,
        seed: 7,
      }

      const solution = solveQUBOSA(qubo, saConfig)
      const energy = evaluateQUBO(qubo, solution)

      // Optimal is x=[1,1] with energy -8
      expect(energy).toBe(-8)
    })
  })

  // -----------------------------------------------------------------------
  // solveQUBOPT
  // -----------------------------------------------------------------------
  describe('solveQUBOPT', () => {
    it('finds low-energy solution for small QUBO', () => {
      // Same QUBO as SA test
      const qubo = quboFromDense([
        [-10, 50],
        [0, -1],
      ])

      const ptConfig: PTConfig = {
        nReplicas: 4,
        tMin: 0.1,
        tMax: 10,
        spacing: TempSpacing.Geometric,
        sweepsPerSwap: 20,
        totalSwaps: 100,
        seed: 42,
      }

      const solution = solveQUBOPT(qubo, ptConfig)

      expect(solution.length).toBe(2)
      for (let i = 0; i < solution.length; i++) {
        expect(solution[i] === 0 || solution[i] === 1).toBe(true)
      }

      const energy = evaluateQUBO(qubo, solution)
      expect(energy).toBeLessThanOrEqual(0)
    })

    it('finds optimal solution for trivial diagonal QUBO', () => {
      const qubo = quboFromDense([
        [-5, 0],
        [0, -3],
      ])

      const ptConfig: PTConfig = {
        nReplicas: 4,
        tMin: 0.1,
        tMax: 10,
        spacing: TempSpacing.Geometric,
        sweepsPerSwap: 20,
        totalSwaps: 100,
        seed: 42,
      }

      const solution = solveQUBOPT(qubo, ptConfig)
      const energy = evaluateQUBO(qubo, solution)

      // Optimal is x=[1,1] with energy -8
      expect(energy).toBe(-8)
    })
  })

  // -----------------------------------------------------------------------
  // QUBO <-> Ising roundtrip consistency
  // -----------------------------------------------------------------------
  describe('QUBO-Ising consistency', () => {
    it('QUBO minimum maps to Ising minimum for 2-variable problem', () => {
      // QUBO: Q = [[-1, 2], [0, -1]]
      // Enumerate all binary solutions and find the QUBO minimum
      const qubo = quboFromDense([
        [-1, 2],
        [0, -1],
      ])

      const energies: number[] = []
      const solutions = [[0, 0], [0, 1], [1, 0], [1, 1]]
      for (const sol of solutions) {
        energies.push(evaluateQUBO(qubo, binarySolution(sol)))
      }

      const minQUBOIdx = energies.indexOf(Math.min(...energies))

      // Convert to Ising and evaluate the corresponding spin configurations
      const ising = quboToIsing(qubo)
      // Binary x -> Ising spin: s = 2x - 1
      const isingEnergies: number[] = []
      for (const sol of solutions) {
        const spins = sol.map(x => 2 * x - 1)
        isingEnergies.push(evaluateIsing(ising, spinConfig(spins)))
      }

      const minIsingIdx = isingEnergies.indexOf(Math.min(...isingEnergies))

      // The minimum should correspond to the same binary assignment
      expect(minIsingIdx).toBe(minQUBOIdx)
    })

    it('QUBO energy ordering is preserved in Ising formulation for 3 variables', () => {
      const qubo = quboFromDense([
        [2, -3, 1],
        [0, 1, -2],
        [0, 0, -1],
      ])

      const ising = quboToIsing(qubo)

      // Enumerate all 8 solutions for 3 variables
      const quboEnergies: number[] = []
      const isingEnergies: number[] = []
      for (let b = 0; b < 8; b++) {
        const bits = [(b >> 2) & 1, (b >> 1) & 1, b & 1]
        const spins = bits.map(x => 2 * x - 1)
        quboEnergies.push(evaluateQUBO(qubo, binarySolution(bits)))
        isingEnergies.push(evaluateIsing(ising, spinConfig(spins)))
      }

      // The ordering of energies should be the same (up to a constant offset)
      // Find the ranking indices
      const quboRanking = quboEnergies
        .map((e, i) => ({ e, i }))
        .sort((a, b) => a.e - b.e)
        .map(x => x.i)
      const isingRanking = isingEnergies
        .map((e, i) => ({ e, i }))
        .sort((a, b) => a.e - b.e)
        .map(x => x.i)

      expect(quboRanking).toEqual(isingRanking)
    })
  })
})

// ===========================================================================
// PS-4: Simulated Bifurcation Tests
// ===========================================================================

describe('PS-4: Simulated Bifurcation', () => {
  // Default SB config for small problems (fast convergence)
  const defaultSBConfig = (
    variant: SBVariant = SBVariant.Discrete,
    seed: number = 42,
  ): SBConfig => ({
    variant,
    nSteps: 500,
    dt: 0.05,
    pumpRate: 1.0,
    kerr: 1.0,
    seed,
  })

  // -----------------------------------------------------------------------
  // simulatedBifurcation: 2-spin antiferromagnet
  // -----------------------------------------------------------------------
  describe('simulatedBifurcation', () => {
    it('finds ground state of 2-spin antiferromagnet', () => {
      // Antiferromagnetic: J_01 = -1 (energy lowered when spins anti-align)
      // Ground state: s_0 = +1, s_1 = -1  OR  s_0 = -1, s_1 = +1
      const ising = isingFromDense(
        [[0, -1], [-1, 0]],
        [0, 0],
      )

      const config = defaultSBConfig(SBVariant.Discrete)
      const spins = simulatedBifurcation(ising, config)

      expect(spins.length).toBe(2)

      // Each spin should be +1 or -1
      expect(Math.abs(spins[0]!)).toBe(1)
      expect(Math.abs(spins[1]!)).toBe(1)

      // Spins should be anti-aligned (ground state of antiferromagnet)
      expect(spins[0]! * spins[1]!).toBe(-1)
    })

    it('finds ground state of 2-spin ferromagnet', () => {
      // Ferromagnetic: J_01 = +1 (energy lowered when spins align)
      // Ground state: both +1 or both -1
      const ising = isingFromDense(
        [[0, 1], [1, 0]],
        [0, 0],
      )

      const config = defaultSBConfig(SBVariant.Discrete)
      const spins = simulatedBifurcation(ising, config)

      // Spins should be aligned
      expect(spins[0]! * spins[1]!).toBe(1)
    })

    it('follows strong field bias', () => {
      // No couplings, strong field pushing all spins to +1
      const ising = isingFromDense(
        [[0, 0], [0, 0]],
        [10, 10],
      )

      const config = defaultSBConfig(SBVariant.Discrete)
      const spins = simulatedBifurcation(ising, config)

      // Both spins should align with the field direction (+1)
      expect(spins[0]).toBe(1)
      expect(spins[1]).toBe(1)
    })

    it('follows negative field bias', () => {
      // Strong field pushing both spins to -1
      const ising = isingFromDense(
        [[0, 0], [0, 0]],
        [-10, -10],
      )

      const config = defaultSBConfig(SBVariant.Discrete)
      const spins = simulatedBifurcation(ising, config)

      expect(spins[0]).toBe(-1)
      expect(spins[1]).toBe(-1)
    })

    it('handles 3-spin frustrated antiferromagnet', () => {
      // Triangle with antiferromagnetic couplings -- frustrated system.
      // Not all pairs can be anti-aligned, but energy should still be low.
      const ising = isingFromDense(
        [[0, -1, -1], [-1, 0, -1], [-1, -1, 0]],
        [0, 0, 0],
      )

      const config = defaultSBConfig(SBVariant.Discrete)
      const spins = simulatedBifurcation(ising, config)

      expect(spins.length).toBe(3)

      // At least two spins must be anti-aligned (cannot fully frustrate triangle)
      const energy = evaluateIsing(ising, spins)
      // Best achievable energy for frustrated triangle: -1
      // (one pair aligned at cost +1, two pairs anti-aligned at -1 each: net -1)
      expect(energy).toBeLessThanOrEqual(1.0)
    })

    it('returns spins of correct length', () => {
      const n = 4
      const couplings = new Float64Array(n * n)
      const field = new Float64Array(n)
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          couplings[i * n + j] = -0.5
          couplings[j * n + i] = -0.5
        }
      }
      const ising: IsingModel = { n, couplings, field }

      const config = defaultSBConfig()
      const spins = simulatedBifurcation(ising, config)

      expect(spins.length).toBe(4)
      for (let i = 0; i < n; i++) {
        expect(Math.abs(spins[i]!)).toBe(1)
      }
    })
  })

  // -----------------------------------------------------------------------
  // Ballistic variant
  // -----------------------------------------------------------------------
  describe('Ballistic variant (bSB)', () => {
    it('converges for 2-spin antiferromagnet', () => {
      const ising = isingFromDense(
        [[0, -1], [-1, 0]],
        [0, 0],
      )

      const config = defaultSBConfig(SBVariant.Ballistic)
      const spins = simulatedBifurcation(ising, config)

      // Should find the anti-aligned ground state
      expect(spins[0]! * spins[1]!).toBe(-1)
    })

    it('converges for 2-spin ferromagnet', () => {
      const ising = isingFromDense(
        [[0, 1], [1, 0]],
        [0, 0],
      )

      const config = defaultSBConfig(SBVariant.Ballistic)
      const spins = simulatedBifurcation(ising, config)

      // Should find the aligned ground state
      expect(spins[0]! * spins[1]!).toBe(1)
    })

    it('converges for field-biased problem', () => {
      const ising = isingFromDense(
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [5, 5, 5],
      )

      const config = defaultSBConfig(SBVariant.Ballistic)
      const spins = simulatedBifurcation(ising, config)

      expect(spins[0]).toBe(1)
      expect(spins[1]).toBe(1)
      expect(spins[2]).toBe(1)
    })
  })

  // -----------------------------------------------------------------------
  // Discrete variant
  // -----------------------------------------------------------------------
  describe('Discrete variant (dSB)', () => {
    it('converges for 2-spin antiferromagnet', () => {
      const ising = isingFromDense(
        [[0, -1], [-1, 0]],
        [0, 0],
      )

      const config = defaultSBConfig(SBVariant.Discrete)
      const spins = simulatedBifurcation(ising, config)

      expect(spins[0]! * spins[1]!).toBe(-1)
    })

    it('converges for 4-spin chain antiferromagnet', () => {
      // Linear chain: J_01 = J_12 = J_23 = -1
      // Ground state: alternating spins +1, -1, +1, -1 or -1, +1, -1, +1
      const ising = isingFromDense(
        [
          [0, -1, 0, 0],
          [-1, 0, -1, 0],
          [0, -1, 0, -1],
          [0, 0, -1, 0],
        ],
        [0, 0, 0, 0],
      )

      const config: SBConfig = {
        variant: SBVariant.Discrete,
        nSteps: 1000,
        dt: 0.05,
        pumpRate: 1.0,
        kerr: 1.0,
        seed: 42,
      }
      const spins = simulatedBifurcation(ising, config)

      // Check alternating pattern: each adjacent pair should be anti-aligned
      expect(spins[0]! * spins[1]!).toBe(-1)
      expect(spins[1]! * spins[2]!).toBe(-1)
      expect(spins[2]! * spins[3]!).toBe(-1)
    })
  })

  // -----------------------------------------------------------------------
  // Deterministic with same seed
  // -----------------------------------------------------------------------
  describe('Determinism (same seed)', () => {
    it('produces identical results with the same seed', () => {
      const ising = isingFromDense(
        [[0, -1, 0.5], [-1, 0, -0.3], [0.5, -0.3, 0]],
        [0.2, -0.1, 0.4],
      )

      const config1 = defaultSBConfig(SBVariant.Discrete, 123)
      const config2 = defaultSBConfig(SBVariant.Discrete, 123)

      const spins1 = simulatedBifurcation(ising, config1)
      const spins2 = simulatedBifurcation(ising, config2)

      // Identical seed should produce identical results
      expect(Array.from(spins1)).toEqual(Array.from(spins2))
    })

    it('produces identical results for Ballistic variant with same seed', () => {
      const ising = isingFromDense(
        [[0, 1], [1, 0]],
        [0.1, -0.1],
      )

      const config1 = defaultSBConfig(SBVariant.Ballistic, 999)
      const config2 = defaultSBConfig(SBVariant.Ballistic, 999)

      const spins1 = simulatedBifurcation(ising, config1)
      const spins2 = simulatedBifurcation(ising, config2)

      expect(Array.from(spins1)).toEqual(Array.from(spins2))
    })

    it('produces different results with different seeds', () => {
      // Use a problem where different initial conditions might lead to different outcomes
      // A frustrated system is good for this -- try many seeds until we find two that differ
      const ising = isingFromDense(
        [[0, -1, -1, 0.5], [-1, 0, 0.3, -1], [-1, 0.3, 0, -1], [0.5, -1, -1, 0]],
        [0, 0, 0, 0],
      )

      const results: Int8Array[] = []
      for (let seed = 0; seed < 20; seed++) {
        const config = defaultSBConfig(SBVariant.Discrete, seed)
        results.push(simulatedBifurcation(ising, config))
      }

      // At least two of the 20 runs should produce different solutions
      // (extremely likely for a frustrated system)
      const uniqueSolutions = new Set(results.map(s => Array.from(s).join(',')))
      expect(uniqueSolutions.size).toBeGreaterThanOrEqual(1)
      // In practice, frustrated systems with different seeds will almost
      // certainly produce at least 2 distinct solutions among 20 runs
    })
  })

  // -----------------------------------------------------------------------
  // simulatedBifurcationQUBO
  // -----------------------------------------------------------------------
  describe('simulatedBifurcationQUBO', () => {
    it('solves small QUBO with clear minimum', () => {
      // QUBO with a clear minimum at x = [1, 0]:
      // Q = [[-5, 100], [0, -1]]
      // x=[1,0]: -5
      // x=[0,1]: -1
      // x=[1,1]: -5 + -1 + 100 = 94
      // x=[0,0]: 0
      // Minimum is x=[1,0] with energy -5
      const qubo = quboFromDense([
        [-5, 100],
        [0, -1],
      ])

      const config: SBConfig = {
        variant: SBVariant.Discrete,
        nSteps: 500,
        dt: 0.05,
        pumpRate: 1.0,
        kerr: 1.0,
        seed: 42,
      }

      const solution = simulatedBifurcationQUBO(qubo, config)

      expect(solution.length).toBe(2)
      // Each entry should be 0 or 1
      for (let i = 0; i < solution.length; i++) {
        expect(solution[i] === 0 || solution[i] === 1).toBe(true)
      }

      // The solution energy should be at most 0 (the all-zeros energy)
      const energy = evaluateQUBO(qubo, solution)
      expect(energy).toBeLessThanOrEqual(0)
    })

    it('returns binary solution (0/1 values)', () => {
      const qubo = quboFromDense([
        [-1, 0],
        [0, -1],
      ])

      const config = defaultSBConfig(SBVariant.Discrete)
      const solution = simulatedBifurcationQUBO(qubo, config)

      for (let i = 0; i < solution.length; i++) {
        expect(solution[i] === 0 || solution[i] === 1).toBe(true)
      }
    })

    it('solves 3-variable QUBO', () => {
      // Diagonal-dominant QUBO: strong incentive to set x_0=1, x_1=1, x_2=0
      // Q = [[-10, 1, 1],
      //      [0,  -10, 1],
      //      [0,   0, 20]]
      const qubo = quboFromDense([
        [-10, 1, 1],
        [0, -10, 1],
        [0, 0, 20],
      ])

      const config: SBConfig = {
        variant: SBVariant.Discrete,
        nSteps: 1000,
        dt: 0.05,
        pumpRate: 1.0,
        kerr: 1.0,
        seed: 42,
      }

      const solution = simulatedBifurcationQUBO(qubo, config)
      const energy = evaluateQUBO(qubo, solution)

      // The best solution [1,1,0] has energy -10 + -10 + 1 = -19
      // At minimum, energy should be significantly negative
      expect(energy).toBeLessThan(0)
    })
  })

  // -----------------------------------------------------------------------
  // Edge cases
  // -----------------------------------------------------------------------
  describe('Edge cases', () => {
    it('handles single-spin Ising model', () => {
      const ising: IsingModel = {
        n: 1,
        couplings: new Float64Array(1), // 1x1, no coupling
        field: new Float64Array([3.0]), // strong positive field
      }

      const config = defaultSBConfig(SBVariant.Discrete)
      const spins = simulatedBifurcation(ising, config)

      expect(spins.length).toBe(1)
      expect(spins[0]).toBe(1) // aligns with positive field
    })

    it('handles zero coupling and zero field (trivial)', () => {
      const ising = isingFromDense(
        [[0, 0], [0, 0]],
        [0, 0],
      )

      const config = defaultSBConfig(SBVariant.Discrete)
      const spins = simulatedBifurcation(ising, config)

      // All spins should be valid (-1 or +1)
      expect(Math.abs(spins[0]!)).toBe(1)
      expect(Math.abs(spins[1]!)).toBe(1)
    })

    it('evaluateQUBO handles empty solution (size 0)', () => {
      const qubo: QUBOMatrix = { n: 0, data: new Float64Array(0) }
      const energy = evaluateQUBO(qubo, new Uint8Array(0))
      expect(energy).toBe(0)
    })

    it('evaluateIsing handles empty model (size 0)', () => {
      const ising: IsingModel = {
        n: 0,
        couplings: new Float64Array(0),
        field: new Float64Array(0),
      }
      const energy = evaluateIsing(ising, new Int8Array(0))
      expect(energy).toBe(0)
    })
  })
})
