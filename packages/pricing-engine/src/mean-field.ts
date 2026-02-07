/**
 * Mean-Field Games — Competing Planners
 *
 * When N planners compete for venue slots, the Nash equilibrium is modeled
 * via the coupled HJB-Fokker-Planck system. The equilibrium venue price
 * emerges as a Lagrange multiplier for supply-demand balance.
 *
 * The system:
 * (a) HJB (backward): -∂V/∂t - ν∆V + H(x, ∇V, m) = 0
 *     Each planner optimizes their bidding strategy.
 *
 * (b) Fokker-Planck (forward): ∂m/∂t - ν∆m - div(m · ∇_pH) = 0
 *     Population distribution evolves under optimal strategies.
 *
 * (c) Fixed-point: Distribution m used in HJB = distribution from optimal V.
 *
 * Solved via Picard iteration with damping for stability.
 *
 * References:
 * - Gomes & Saude (2020). "MFG Approach to Price Formation"
 * - Lasry & Lions (2007). "Mean field games"
 */

/**
 * MFG solver configuration.
 */
export interface MFGConfig {
  /** Number of planner types */
  nPlannerTypes: number
  /** Budget distribution per planner type */
  plannerBudgets: number[]
  /** Venue capacity (total slots available) */
  venueCapacity: number
  /** Time horizon (planning periods) */
  timeHorizon: number
  /** Time step size */
  dt: number
  /** Number of state grid points (budget discretization) */
  nGridPoints: number
  /** Congestion parameter: how much competition raises prices */
  congestionParam: number
  /** Viscosity coefficient (noise in planner decisions) */
  viscosity: number
  /** Damping factor for Picard iteration stability (0 < damping < 1) */
  damping: number
  /** Maximum Picard iterations */
  maxIterations: number
  /** Convergence tolerance */
  tolerance: number
}

export interface MFGResult {
  /** Equilibrium price per slot at each time step */
  equilibriumPrices: number[]
  /** Optimal bidding strategy per planner type at each time step */
  plannerStrategies: number[][]
  /** Equilibrium distribution of planner states */
  distribution: number[][]
  /** Number of Picard iterations to convergence */
  iterations: number
  /** Final L2 error between successive distributions */
  finalError: number
}

/**
 * Solve the mean-field game via Picard iteration.
 *
 * Algorithm:
 * 1. Initialize distribution m⁰ (uniform over planner budget states)
 * 2. Solve HJB backward given m^k → value function V^k and optimal strategy α^k
 * 3. Solve FPK forward given α^k → new distribution m^{k+1}
 * 4. Damp: m^{k+1} ← damping·m^{k+1} + (1-damping)·m^k
 * 5. Check convergence: ||m^{k+1} - m^k||₂ < tol
 * 6. Repeat until convergence
 *
 * The Hamiltonian for venue booking:
 *   H(x, p, m) = -max_α { α·(v(m) - x) - C(α) }
 *
 * where:
 * - x is planner's remaining budget
 * - α is booking intensity (how aggressively to bid)
 * - v(m) is venue value given population distribution
 * - C(α) = α²/2 is the effort cost of aggressive bidding
 */
export function solveMFG(config: MFGConfig): MFGResult {
  const {
    nPlannerTypes,
    plannerBudgets,
    venueCapacity,
    timeHorizon,
    dt,
    nGridPoints,
    congestionParam,
    viscosity,
    damping,
    maxIterations,
    tolerance,
  } = config

  const nSteps = Math.floor(timeHorizon / dt)

  // State grid: budget levels from 0 to max budget
  const maxBudget = Math.max(...plannerBudgets) * 1.5
  const dx = maxBudget / (nGridPoints - 1)
  const grid = Array.from({ length: nGridPoints }, (_, i) => i * dx)

  // Initialize distribution: Gaussian around each planner type's budget
  // m[type][gridPoint] = density
  let m: number[][] = []
  for (let type = 0; type < nPlannerTypes; type++) {
    const dist = new Array(nGridPoints).fill(0) as number[]
    const budget = plannerBudgets[type]!
    let sum = 0
    for (let i = 0; i < nGridPoints; i++) {
      dist[i] = Math.exp(-((grid[i]! - budget) ** 2) / (2 * (budget * 0.2) ** 2))
      sum += dist[i]!
    }
    // Normalize
    for (let i = 0; i < nGridPoints; i++) {
      dist[i] = dist[i]! / (sum * dx)
    }
    m.push(dist)
  }

  // Value function V[type][gridPoint][timeStep]
  const V: number[][][] = Array.from({ length: nPlannerTypes }, () =>
    Array.from({ length: nGridPoints }, () =>
      new Array(nSteps + 1).fill(0) as number[],
    ),
  )

  // Optimal strategy α[type][gridPoint][timeStep]
  const alpha: number[][][] = Array.from({ length: nPlannerTypes }, () =>
    Array.from({ length: nGridPoints }, () =>
      new Array(nSteps + 1).fill(0) as number[],
    ),
  )

  let iterations = 0
  let finalError = Infinity

  for (let iter = 0; iter < maxIterations; iter++) {
    iterations = iter + 1
    const mOld = m.map((row) => [...row])

    // Compute aggregate demand at each time step (from distribution)
    const aggregateDemand = new Array(nSteps + 1).fill(0) as number[]
    for (let t = 0; t <= nSteps; t++) {
      let totalDemand = 0
      for (let type = 0; type < nPlannerTypes; type++) {
        for (let i = 0; i < nGridPoints; i++) {
          totalDemand += m[type]![i]! * grid[i]! * dx
        }
      }
      aggregateDemand[t] = totalDemand
    }

    // Step 2: Solve HJB backward for each planner type
    for (let type = 0; type < nPlannerTypes; type++) {
      // Terminal condition: V(x, T) = 0 (no value after horizon)
      for (let i = 0; i < nGridPoints; i++) {
        V[type]![i]![nSteps] = 0
      }

      for (let t = nSteps - 1; t >= 0; t--) {
        // Venue price based on congestion: p(t) = base + congestion * demand/capacity
        const venuePrice = congestionParam * aggregateDemand[t]! / venueCapacity

        for (let i = 1; i < nGridPoints - 1; i++) {
          const x = grid[i]! // Current budget

          // Optimal booking intensity: α* = max(0, venue_value - x_cost) / effort
          // From FOC of Hamiltonian: α* = max(0, (V(x-p, t+1) - V(x, t+1)))
          const valueIfBook = i > 0 && grid[i]! >= venuePrice
            ? V[type]![Math.max(0, Math.floor((x - venuePrice) / dx))]![t + 1]! + venuePrice
            : 0
          const valueIfWait = V[type]![i]![t + 1]!

          const optAlpha = Math.max(0, (valueIfBook - valueIfWait) / (1 + congestionParam))
          alpha[type]![i]![t] = optAlpha

          // Viscous HJB: V(x,t) = V(x,t+1) + dt * (ν·ΔV - H(x,∇V,m))
          const laplacian = (V[type]![i + 1]![t + 1]! - 2 * V[type]![i]![t + 1]! + V[type]![i - 1]![t + 1]!) / (dx * dx)

          V[type]![i]![t] = V[type]![i]![t + 1]!
                          + dt * (viscosity * laplacian + optAlpha * (valueIfBook - valueIfWait) - 0.5 * optAlpha * optAlpha)
        }

        // Boundary conditions: V(0, t) = 0, V(xMax, t) from extrapolation
        V[type]![0]![t] = 0
        V[type]![nGridPoints - 1]![t] = V[type]![nGridPoints - 2]![t]!
      }
    }

    // Step 3: Solve Fokker-Planck forward
    const mNew: number[][] = m.map((row) => [...row])

    for (let type = 0; type < nPlannerTypes; type++) {
      for (let t = 0; t < nSteps; t++) {
        const drift = new Array(nGridPoints).fill(0) as number[]

        for (let i = 0; i < nGridPoints; i++) {
          // Drift from optimal strategy
          drift[i] = -alpha[type]![i]![t]! * congestionParam
        }

        // Forward Euler for FPK
        const mTemp = [...mNew[type]!]
        for (let i = 1; i < nGridPoints - 1; i++) {
          // Diffusion: ν·Δm
          const diffusion = viscosity * (mTemp[i + 1]! - 2 * mTemp[i]! + mTemp[i - 1]!) / (dx * dx)
          // Advection: -div(m·drift)
          const advection = -(mTemp[i + 1]! * drift[i + 1]! - mTemp[i - 1]! * drift[i - 1]!) / (2 * dx)

          mNew[type]![i] = mTemp[i]! + dt * (diffusion + advection)
        }

        // Ensure non-negativity and renormalize
        let sum = 0
        for (let i = 0; i < nGridPoints; i++) {
          mNew[type]![i] = Math.max(0, mNew[type]![i]!)
          sum += mNew[type]![i]!
        }
        if (sum > 0) {
          for (let i = 0; i < nGridPoints; i++) {
            mNew[type]![i] = mNew[type]![i]! / (sum * dx)
          }
        }
      }
    }

    // Step 4: Damped update
    for (let type = 0; type < nPlannerTypes; type++) {
      for (let i = 0; i < nGridPoints; i++) {
        m[type]![i] = damping * mNew[type]![i]! + (1 - damping) * mOld[type]![i]!
      }
    }

    // Step 5: Check convergence
    let error = 0
    for (let type = 0; type < nPlannerTypes; type++) {
      for (let i = 0; i < nGridPoints; i++) {
        error += (m[type]![i]! - mOld[type]![i]!) ** 2
      }
    }
    finalError = Math.sqrt(error * dx)

    if (finalError < tolerance) break
  }

  // Extract equilibrium prices
  const equilibriumPrices: number[] = []
  for (let t = 0; t <= nSteps; t++) {
    let totalDemand = 0
    for (let type = 0; type < nPlannerTypes; type++) {
      for (let i = 0; i < nGridPoints; i++) {
        totalDemand += m[type]![i]! * grid[i]! * dx
      }
    }
    equilibriumPrices.push(congestionParam * totalDemand / venueCapacity)
  }

  // Extract strategies (average over budget grid)
  const plannerStrategies: number[][] = []
  for (let type = 0; type < nPlannerTypes; type++) {
    const strategy: number[] = []
    for (let t = 0; t <= nSteps; t++) {
      let avgAlpha = 0
      let totalWeight = 0
      for (let i = 0; i < nGridPoints; i++) {
        avgAlpha += alpha[type]![i]![t]! * m[type]![i]! * dx
        totalWeight += m[type]![i]! * dx
      }
      strategy.push(totalWeight > 0 ? avgAlpha / totalWeight : 0)
    }
    plannerStrategies.push(strategy)
  }

  return {
    equilibriumPrices,
    plannerStrategies,
    distribution: m,
    iterations,
    finalError,
  }
}
