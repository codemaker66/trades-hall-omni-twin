# TECHNIQUE_02_OPTIMAL_TRANSPORT.md — Optimal Transport for Venue-Event Matching

> **Purpose**: Feed this to Claude Code AFTER the main phases and CT (Category Theory)
> are complete. Implements mathematically principled venue-event matching using Optimal
> Transport — Wasserstein distances, Sinkhorn algorithm, barycenters, partial transport,
> and GPU-accelerated browser computation.
>
> **How to use**: Tell Claude Code: "Read TECHNIQUE_02_OPTIMAL_TRANSPORT.md and implement
> it incrementally, starting from OT-1."
>
> **Key papers**:
> - Cuturi 2013, "Sinkhorn Distances" (arXiv:1306.0895) — entropic regularization
> - Schmitzer 2019, "Stabilized Sparse Scaling" (arXiv:1610.06519) — log-domain Sinkhorn
> - Feydy et al. 2019, "Interpolating between OT and MMD" (arXiv:1810.08278) — Sinkhorn divergence
> - Chizat 2024, "Convergence of Annealed Sinkhorn" (arXiv:2408.11620) — epsilon scheduling
> - Vayer et al. 2019, "Fused Gromov-Wasserstein" (arXiv:1805.09114) — heterogeneous matching
> - Li et al. 2018, "Learning to Match via Inverse OT" (arXiv:1802.03644) — learned cost functions
> - Chapel et al. 2020, "Partial OT" (arXiv:2002.08276) — partial transport
> - Chizat et al. 2016, "Unbalanced OT" (arXiv:1607.05816) — soft marginal relaxation
> - Agueh & Carlier 2011, "Wasserstein Barycenters" (DOI:10.1137/100805741)
> - FlashSinkhorn 2025 (arXiv:2602.03067) — IO-aware GPU Sinkhorn

---

## Why This Matters

Optimal Transport computes the minimum-cost coupling between two distributions. For
venue planning, this means: given a set of venues (supply) and events (demand), OT
finds the mathematically optimal assignment that minimizes total mismatch cost — while
handling the fact that features are heterogeneous (capacity is numeric, amenities are
categorical, location is geographic, price has asymmetric penalties).

No competitor in the venue space uses OT. Traditional matching uses naive scoring
(weighted sums) which ignores the coupling structure. OT considers ALL assignments
jointly, guaranteeing global optimality.

---

## OT-1: Core Sinkhorn Solver in TypeScript + WebGPU

### What to Build

A Sinkhorn solver that runs in the browser via WebGPU, with CPU fallback. This is the
computational engine everything else builds on.

### Implementation

Create `packages/optimal-transport/`:

```
packages/optimal-transport/
  src/
    sinkhorn.ts            — Core Sinkhorn algorithm (CPU, typed arrays)
    sinkhorn-log.ts        — Log-domain stabilized Sinkhorn (for small epsilon)
    sinkhorn-gpu.ts        — WebGPU compute shader Sinkhorn
    cost-matrix.ts         — Cost matrix construction utilities
    types.ts               — TransportPlan, CostMatrix, SinkhornConfig types
    utils.ts               — Normalization, log-sum-exp, numerical helpers
    index.ts               — Public API
  __tests__/
    sinkhorn.test.ts       — Correctness tests against known solutions
    sinkhorn-gpu.test.ts   — GPU vs CPU output equivalence tests
    properties.test.ts     — Property-based tests (marginal constraints, etc.)
  shaders/
    sinkhorn-u.wgsl        — WebGPU compute shader for u-update
    sinkhorn-v.wgsl        — WebGPU compute shader for v-update
    transport-plan.wgsl    — Recover transport plan from dual potentials
```

### CPU Sinkhorn (Standard + Log-Domain)

```typescript
// packages/optimal-transport/src/sinkhorn.ts

export interface SinkhornConfig {
  epsilon: number;          // Regularization parameter (default: 0.01)
  maxIterations: number;    // Max iterations (default: 100)
  tolerance: number;        // Convergence tolerance on marginals (default: 1e-6)
  method: 'standard' | 'log' | 'stabilized';
}

export interface TransportResult {
  plan: Float64Array;       // N×M transport plan (row-major)
  cost: number;             // Total transport cost <T, C>
  dualF: Float64Array;      // Dual potential f (N)
  dualG: Float64Array;      // Dual potential g (M)
  iterations: number;       // Iterations to converge
  converged: boolean;
}

/**
 * Standard Sinkhorn: T* = diag(u) · K · diag(v)
 * where K = exp(-C/ε), iterating u = a/(Kv), v = b/(Kᵀu)
 *
 * Stable for ε > 0.01·median(C). For smaller ε, use log-domain.
 */
export function sinkhorn(
  a: Float64Array,    // Source distribution (N), sums to 1
  b: Float64Array,    // Target distribution (M), sums to 1
  C: Float64Array,    // Cost matrix (N×M, row-major)
  config: Partial<SinkhornConfig> = {}
): TransportResult { ... }

/**
 * Log-domain Sinkhorn (Schmitzer 2019):
 * Works with dual potentials f, g instead of scaling vectors u, v.
 * f_i ← -ε · LSE_j((g_j - C_ij) / ε) + ε·log(a_i)
 * g_j ← -ε · LSE_i((f_i - C_ij) / ε) + ε·log(b_j)
 *
 * Stable for ANY ε, including very small values.
 * Uses log-sum-exp trick: LSE(x) = max(x) + log(Σ exp(x - max(x)))
 */
export function sinkhornLog(
  a: Float64Array,
  b: Float64Array,
  C: Float64Array,
  config: Partial<SinkhornConfig> = {}
): TransportResult { ... }
```

### WebGPU Sinkhorn Compute Shader

The core WGSL shader for the u-update step:

```wgsl
// packages/optimal-transport/shaders/sinkhorn-u.wgsl

struct Params {
  N: u32,          // Number of venues (source)
  M: u32,          // Number of events (target)
  inv_epsilon: f32  // 1.0 / epsilon
}

@group(0) @binding(0) var<storage, read> C: array<f32>;       // [N*M] cost matrix
@group(0) @binding(1) var<storage, read> v: array<f32>;       // [M] current v vector
@group(0) @binding(2) var<storage, read> a: array<f32>;       // [N] source marginal
@group(0) @binding(3) var<storage, read_write> u: array<f32>; // [N] output u vector
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.N) { return; }

  // u_i = a_i / Σ_j exp(-C_ij / ε) * v_j
  var sum: f32 = 0.0;
  for (var j: u32 = 0u; j < params.M; j++) {
    sum += exp(-C[i * params.M + j] * params.inv_epsilon) * v[j];
  }
  u[i] = a[i] / max(sum, 1e-30);
}
```

Symmetric shader for v-update (`sinkhorn-v.wgsl`):

```wgsl
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let j = gid.x;
  if (j >= params.M) { return; }

  // v_j = b_j / Σ_i exp(-C_ij / ε) * u_i
  var sum: f32 = 0.0;
  for (var i: u32 = 0u; i < params.N; i++) {
    sum += exp(-C[i * params.M + j] * params.inv_epsilon) * u[i];
  }
  v[j] = b[j] / max(sum, 1e-30);
}
```

### GPU Orchestrator

```typescript
// packages/optimal-transport/src/sinkhorn-gpu.ts

export class SinkhornGPU {
  private device: GPUDevice;
  private uPipeline: GPUComputePipeline;
  private vPipeline: GPUComputePipeline;

  static async create(): Promise<SinkhornGPU | null> {
    if (!navigator.gpu) return null;
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return null;
    const device = await adapter.requestDevice();
    // Compile shaders, create pipelines...
    return new SinkhornGPU(device, ...);
  }

  async solve(
    a: Float32Array,   // [N]
    b: Float32Array,   // [M]
    C: Float32Array,   // [N*M]
    config: SinkhornConfig
  ): Promise<TransportResult> {
    // 1. Upload a, b, C to GPU storage buffers
    // 2. Initialize u = ones(N), v = ones(M) on GPU
    // 3. For each iteration:
    //    - Dispatch u-update shader (ceil(N/64) workgroups)
    //    - Dispatch v-update shader (ceil(M/64) workgroups)
    // 4. Read back u, v from GPU
    // 5. Compute transport plan T_ij = u_i * exp(-C_ij/ε) * v_j
    // 6. Compute cost = Σ T_ij * C_ij
  }
}

// Auto-select GPU or CPU
export async function createSolver(): Promise<SinkhornSolver> {
  const gpu = await SinkhornGPU.create();
  if (gpu) {
    console.log('OT: Using WebGPU Sinkhorn');
    return gpu;
  }
  console.log('OT: Falling back to CPU Sinkhorn');
  return new SinkhornCPU();
}
```

### Performance Targets

| Problem Size | WebGPU | CPU (TypedArrays) |
|---|---|---|
| 100×100 | < 1ms | < 5ms |
| 1,000×100 | < 10ms | 50-200ms |
| 1,000×1,000 | < 50ms | 500ms-2s |
| 5,000×5,000 | < 500ms | Not practical |

WebGPU crossover point is ~N=200-500. Below that, CPU is fine.

### Property-Based Tests

```typescript
import fc from 'fast-check';

// Transport plan satisfies marginal constraints
test('transport plan marginals match source and target', () => {
  fc.assert(fc.property(
    arbitraryDistribution(50),   // source
    arbitraryDistribution(80),   // target
    arbitraryCostMatrix(50, 80), // cost
    (a, b, C) => {
      const result = sinkhorn(a, b, C, { epsilon: 0.01 });
      const rowSums = computeRowSums(result.plan, 50, 80);
      const colSums = computeColSums(result.plan, 50, 80);
      // Row sums ≈ source distribution (within tolerance)
      expectClose(rowSums, a, 1e-4);
      // Column sums ≈ target distribution (within tolerance)
      expectClose(colSums, b, 1e-4);
    }
  ));
});

// Transport plan is non-negative
test('transport plan entries are non-negative', () => { ... });

// GPU and CPU produce equivalent results
test('GPU Sinkhorn matches CPU Sinkhorn', () => { ... });

// Sinkhorn divergence is positive definite
test('Sinkhorn divergence S(a,b) >= 0 and S(a,a) = 0', () => { ... });

// Cost decreases with better matches
test('lower cost matrix → lower transport cost', () => { ... });
```

---

## OT-2: Heterogeneous Cost Matrix for Venue-Event Matching

### What to Build

A cost matrix builder that handles mixed feature types with asymmetric penalties.
This is where the domain knowledge lives.

### Feature Distance Functions

```typescript
// packages/optimal-transport/src/cost-matrix.ts

export interface VenueFeatures {
  capacity: number;
  amenities: boolean[];   // [projector, stage, wifi, kitchen, AV, outdoor, ...]
  location: { lat: number; lng: number };
  pricePerEvent: number;
  sqFootage: number;
  venueType: string;       // 'hotel' | 'conference' | 'outdoor' | ...
}

export interface EventRequirements {
  guestCount: number;
  requiredAmenities: boolean[];
  preferredLocation: { lat: number; lng: number };
  budget: number;
  minSqFootage: number;
  eventType: string;
}

/**
 * Capacity distance — ASYMMETRIC:
 * Venue too small for event = severe penalty (2.0× normalized shortfall)
 * Venue too large for event = mild penalty (0.3× normalized excess)
 *
 * A 300-seat venue for 500 guests is MUCH worse than 800 seats for 500 guests.
 */
function capacityDistance(eventGuests: number, venueCapacity: number): number {
  if (venueCapacity < eventGuests) {
    return 2.0 * (eventGuests - venueCapacity) / eventGuests;  // severe
  }
  return 0.3 * (venueCapacity - eventGuests) / eventGuests;    // mild waste
}

/**
 * Amenity distance — ASYMMETRIC:
 * Missing REQUIRED amenities = penalty proportional to fraction missing
 * Extra amenities venue has = no penalty (bonus, not cost)
 */
function amenityDistance(required: boolean[], available: boolean[]): number {
  let missing = 0;
  let totalRequired = 0;
  for (let i = 0; i < required.length; i++) {
    if (required[i]) {
      totalRequired++;
      if (!available[i]) missing++;
    }
  }
  return totalRequired > 0 ? missing / totalRequired : 0;
}

/**
 * Location distance — Haversine (great-circle distance in km)
 */
function locationDistance(a: { lat: number; lng: number }, b: { lat: number; lng: number }): number {
  const R = 6371.0;
  const dLat = toRad(b.lat - a.lat);
  const dLon = toRad(b.lng - a.lng);
  const sinLat = Math.sin(dLat / 2);
  const sinLon = Math.sin(dLon / 2);
  const h = sinLat * sinLat + Math.cos(toRad(a.lat)) * Math.cos(toRad(b.lat)) * sinLon * sinLon;
  return 2 * R * Math.asin(Math.sqrt(h));
}

/**
 * Price distance — ASYMMETRIC:
 * Over budget = 1.5× normalized overage (painful)
 * Under budget = 0.1× normalized savings (mild — under budget is fine)
 */
function priceDistance(budget: number, venuePrice: number): number {
  const diff = venuePrice - budget;
  if (diff > 0) return Math.min(1.5 * diff / budget, 3.0);  // over budget, capped
  return 0.1 * Math.abs(diff) / budget;                       // under budget, mild
}

/**
 * Build the full N×M cost matrix.
 *
 * Each per-feature distance matrix is normalized to [0,1], then combined
 * with configurable weights. Default weights reflect domain priority:
 *   capacity: 0.30 (must fit the guests)
 *   price:    0.30 (must fit the budget)
 *   amenity:  0.25 (must have the right equipment)
 *   location: 0.15 (nice to have, less critical)
 */
export interface CostWeights {
  capacity: number;
  price: number;
  amenity: number;
  location: number;
}

export function buildCostMatrix(
  events: EventRequirements[],
  venues: VenueFeatures[],
  weights: CostWeights = { capacity: 0.30, price: 0.30, amenity: 0.25, location: 0.15 }
): Float64Array {
  const N = events.length;
  const M = venues.length;

  // Compute per-feature distance matrices
  const capDist = new Float64Array(N * M);
  const priceDist = new Float64Array(N * M);
  const amenDist = new Float64Array(N * M);
  const locDist = new Float64Array(N * M);

  for (let i = 0; i < N; i++) {
    for (let j = 0; j < M; j++) {
      const idx = i * M + j;
      capDist[idx] = capacityDistance(events[i].guestCount, venues[j].capacity);
      priceDist[idx] = priceDistance(events[i].budget, venues[j].pricePerEvent);
      amenDist[idx] = amenityDistance(events[i].requiredAmenities, venues[j].amenities);
      locDist[idx] = locationDistance(events[i].preferredLocation, venues[j].location);
    }
  }

  // Normalize each to [0, 1]
  normalize01(capDist);
  normalize01(priceDist);
  normalize01(amenDist);
  normalize01(locDist);

  // Weighted combination
  const C = new Float64Array(N * M);
  for (let k = 0; k < N * M; k++) {
    C[k] = weights.capacity * capDist[k]
         + weights.price * priceDist[k]
         + weights.amenity * amenDist[k]
         + weights.location * locDist[k];
  }

  return C;
}
```

### Sinkhorn Divergence (Debiased — Use This, Not Raw Distance)

The Sinkhorn divergence eliminates entropic bias and is positive definite
(Feydy et al. 2019, arXiv:1810.08278):

```typescript
/**
 * Sinkhorn Divergence: S_ε(a,b) = OT_ε(a,b) - ½·OT_ε(a,a) - ½·OT_ε(b,b)
 *
 * Properties:
 * - S_ε(a,a) = 0 (identity of indiscernibles)
 * - S_ε(a,b) ≥ 0 (positive definite)
 * - Converges to W(a,b) as ε → 0
 * - Better behaved as a loss function than raw Sinkhorn distance
 *
 * ALWAYS use this instead of raw sinkhorn2() for scoring/ranking.
 */
export function sinkhornDivergence(
  a: Float64Array,
  b: Float64Array,
  C_ab: Float64Array,  // Cost between a and b
  C_aa: Float64Array,  // Self-cost of a
  C_bb: Float64Array,  // Self-cost of b
  epsilon: number
): number {
  const OT_ab = sinkhornCost(a, b, C_ab, epsilon);
  const OT_aa = sinkhornCost(a, a, C_aa, epsilon);
  const OT_bb = sinkhornCost(b, b, C_bb, epsilon);
  return OT_ab - 0.5 * OT_aa - 0.5 * OT_bb;
}
```

---

## OT-3: Wasserstein Barycenters — The "Ideal Venue" Centroid

### What to Build

Compute the Wasserstein barycenter of past successful venue bookings to create an
"ideal venue" profile. Score new candidate venues by their Wasserstein distance to
this centroid. This is fundamentally better than Euclidean averaging because it
respects the cost structure of the feature space.

### Implementation

```typescript
// packages/optimal-transport/src/barycenter.ts

/**
 * Fixed-support Wasserstein barycenter via Iterative Bregman Projections
 * (Benamou et al. 2015, arXiv:1412.5154)
 *
 * Given N input distributions μ_1, ..., μ_N on the same support,
 * finds the distribution bar(μ) that minimizes:
 *   Σ_i λ_i · W_ε²(bar, μ_i)
 *
 * This is the "average ideal venue" — not the Euclidean mean of features,
 * but the distribution that minimizes total transport cost to all inputs.
 *
 * Complexity: O(N·n²) per iteration, typically converges in 20-50 iterations.
 */
export function fixedSupportBarycenter(
  distributions: Float64Array[], // N distributions, each of length n (support size)
  costMatrix: Float64Array,      // n×n cost matrix on the shared support
  weights: Float64Array,         // N weights (sum to 1), how much each input matters
  epsilon: number,               // Regularization (0.01 default)
  maxIter: number = 100
): Float64Array {
  const N = distributions.length;
  const n = distributions[0].length;

  // Gibbs kernel K = exp(-C/ε)
  const K = new Float64Array(n * n);
  for (let k = 0; k < n * n; k++) {
    K[k] = Math.exp(-costMatrix[k] / epsilon);
  }

  // Initialize barycenter as uniform
  let bary = new Float64Array(n).fill(1.0 / n);

  // Iterative Bregman projections
  // Each iteration: for each input distribution, run one Sinkhorn step
  // Then combine via weighted geometric mean
  for (let iter = 0; iter < maxIter; iter++) {
    const logBary = new Float64Array(n);

    for (let i = 0; i < N; i++) {
      // Sinkhorn step: v_i = μ_i / (Kᵀ · (bary / (K · v_i)))
      // Accumulate log contribution: logBary += weight_i * log(K · v_i)
      // (Implementation details: standard iterative Bregman projection)
    }

    // bary = exp(logBary) then normalize
    // Check convergence: ||bary_new - bary_old|| < tolerance
  }

  return bary;
}

/**
 * Score a candidate venue against the ideal barycenter.
 * Lower score = better match to historical successful bookings.
 */
export function scoreAgainstBarycenter(
  idealBarycenter: Float64Array,
  candidateDistribution: Float64Array,
  costMatrix: Float64Array,
  epsilon: number
): number {
  return sinkhornDivergence(
    idealBarycenter, candidateDistribution,
    costMatrix, costMatrix, costMatrix,
    epsilon
  );
}
```

### Usage in the Venue Platform

```typescript
// When a planner searches for venues:
// 1. Fetch their past N successful bookings from the database
// 2. Convert each past venue to a feature distribution (histogram over capacity
//    bins, amenity presence, location proximity, price range)
// 3. Compute the Wasserstein barycenter = their "ideal venue" profile
// 4. Score all candidate venues against this barycenter
// 5. Return ranked matches
//
// This provides PERSONALIZED recommendations that improve with booking history,
// and the barycenter is interpretable — you can show the planner their "ideal
// venue" as a visual profile.
```

---

## OT-4: Partial & Unbalanced Transport — Real-World Mismatches

### What to Build

Standard OT forces all supply to meet all demand. But in venue matching:
- A 1,000-seat venue for 500 guests shouldn't be penalized for unused capacity
- Not all event requirements need perfect satisfaction (nice-to-have vs must-have)
- The number of venues ≠ number of events

**Partial OT** (Chapel et al. 2020, arXiv:2002.08276) transports only a fraction
m of the total mass. **Unbalanced OT** (Chizat et al. 2016, arXiv:1607.05816) uses
soft KL-divergence penalties instead of hard marginal constraints.

### Implementation

```typescript
// packages/optimal-transport/src/partial.ts

/**
 * Partial Optimal Transport:
 * min_T <T, C>  s.t.  T≥0, T·1 ≤ a, Tᵀ·1 ≤ b, Σ_ij T_ij = m
 *
 * Only transport mass m ≤ min(||a||₁, ||b||₁).
 * Use when you know the fraction to match: "fill 70% of requirements."
 */
export function partialSinkhorn(
  a: Float64Array,
  b: Float64Array,
  C: Float64Array,
  mass: number,          // Total mass to transport (0 < m ≤ min(sum(a), sum(b)))
  epsilon: number
): TransportResult { ... }

/**
 * Unbalanced Optimal Transport:
 * min_T <T,C> + ε·Ω(T) + ρ₁·KL(T·1 || a) + ρ₂·KL(Tᵀ·1 || b)
 *
 * The parameter ρ (reg_m) controls marginal relaxation:
 *   ρ → 0:   nearly ignores marginals (full mass destruction allowed)
 *   ρ → ∞:   enforces marginals exactly (recovers balanced OT)
 *   ρ = 0.1: good default for venue matching with supply/demand mismatch
 *
 * Sinkhorn iterations with modified updates:
 *   u_i = (a_i / (K·v)_i)^(ρ/(ρ+ε))
 *   v_j = (b_j / (Kᵀ·u)_j)^(ρ/(ρ+ε))
 */
export function unbalancedSinkhorn(
  a: Float64Array,
  b: Float64Array,
  C: Float64Array,
  epsilon: number,
  regMarginal: number    // ρ — marginal relaxation strength
): TransportResult { ... }
```

### When to Use Which

```
Standard Sinkhorn:
  Events and venues are 1:1 matched, all must be assigned
  → ot.sinkhorn(a, b, C, ε)

Partial OT:
  "Match the top 70% of requirements, skip the rest"
  "Only assign events to the best-fitting 80% of venues"
  → partialSinkhorn(a, b, C, mass=0.7, ε)

Unbalanced OT:
  Venue capacity >> event demand (don't penalize unused space)
  Some requirements are soft (nice-to-have amenities)
  → unbalancedSinkhorn(a, b, C, ε, ρ=0.1)
```

---

## OT-5: Displacement Interpolation — Smooth Layout Transitions

### What to Build

McCann's displacement interpolation creates the "optimal path" between two
distributions. For venue layouts: given Layout A and Layout B, compute the
smoothest possible transition — the mathematically optimal way to rearrange
furniture from one configuration to another.

### Implementation

```typescript
// packages/optimal-transport/src/interpolation.ts

/**
 * Displacement interpolation between two layouts.
 *
 * Given transport plan T from layout A to layout B:
 * At time t ∈ [0, 1], each furniture piece is at position:
 *   pos(t) = (1-t) · posA + t · posB
 *
 * where the assignment (which piece in A maps to which piece in B)
 * is determined by the optimal transport plan.
 *
 * This produces the smoothest possible rearrangement animation,
 * minimizing total movement distance.
 */
export function displacementInterpolation(
  layoutA: FurniturePosition[],  // Source layout positions
  layoutB: FurniturePosition[],  // Target layout positions
  transportPlan: Float64Array,   // OT plan mapping A → B
  t: number                       // Interpolation parameter [0, 1]
): FurniturePosition[] {
  // 1. Extract the dominant assignment from the transport plan
  //    (each piece in A maps to the piece in B with highest transport mass)
  // 2. For each assigned pair (a_i, b_j):
  //    interpolated_pos = (1-t) * a_i.pos + t * b_j.pos
  //    interpolated_rot = slerp(a_i.rot, b_j.rot, t)
  // 3. Handle unmatched pieces (from partial OT): fade in/out with opacity
  ...
}

/**
 * Generate keyframes for a smooth layout transition animation.
 * Returns positions at N evenly spaced time steps.
 */
export function generateTransitionKeyframes(
  layoutA: FurniturePosition[],
  layoutB: FurniturePosition[],
  steps: number = 60  // 60 frames = 1 second at 60fps
): FurniturePosition[][] {
  // 1. Build cost matrix from A positions to B positions
  // 2. Solve OT to get optimal assignment
  // 3. Generate interpolated layouts at t = 0, 1/steps, 2/steps, ..., 1
  ...
}
```

### Visual Integration

In the 3D editor, when a user switches between two saved layouts (scenarios),
animate the transition using displacement interpolation. Furniture slides along
optimal paths rather than teleporting or using naive linear interpolation.

---

## OT-6: Visualization — Sankey Diagrams & Transport Flow

### What to Build

Interactive visualizations of the transport plan showing which venues match to
which events and why.

### Sankey Diagram (Venue-Event Matching Flow)

```typescript
// apps/web/src/components/ot-visualization/TransportSankey.tsx

/**
 * Bipartite Sankey diagram:
 * - Left nodes: venues (sized by capacity)
 * - Right nodes: events (sized by guest count)
 * - Links: transport plan values (width = match strength)
 * - Color: match quality (green = strong, yellow = moderate, red = poor)
 *
 * Interactive:
 * - Hover a venue → highlight all events it matches to
 * - Hover an event → highlight all venues that could serve it
 * - Click a link → show detailed cost breakdown (capacity, price, amenity, location)
 *
 * Use d3-sankey (v0.12.3) for layout, React for rendering.
 */
```

### Transport Cost Heatmap

```typescript
// apps/web/src/components/ot-visualization/CostHeatmap.tsx

/**
 * Interactive heatmap of the cost matrix C[i,j]:
 * - Rows: events, Columns: venues
 * - Cell color: cost (dark = high cost / poor match, light = low cost / good match)
 * - Overlay: transport plan T[i,j] as circles (size = assignment weight)
 *
 * The heatmap shows WHERE assignments are optimal and WHERE costs are high.
 * Hover cells to see per-feature cost breakdown.
 */
```

### Barycenter Radar Chart

```typescript
// apps/web/src/components/ot-visualization/BarycenterRadar.tsx

/**
 * Radar/spider chart showing the "ideal venue" barycenter profile:
 * - Axes: capacity, price range, amenity coverage, location centrality, size
 * - Filled area: the barycenter distribution
 * - Overlay: individual past bookings as faded outlines
 * - Highlight: how the currently viewed venue compares to the ideal
 *
 * This makes the OT math interpretable: "Based on your booking history,
 * your ideal venue has 200-400 capacity, mid-range pricing, strong AV,
 * and is within 5km of downtown."
 */
```

---

## OT-7: Inverse Optimal Transport — Learn the Cost Function

### What to Build

Instead of manually setting cost weights (capacity: 0.30, price: 0.30, etc.),
**learn the optimal weights from observed successful matchings**.

From Li et al. 2018 (arXiv:1802.03644): given a set of observed venue-event
matchings, solve for the cost matrix C that makes the observed matchings optimal
under OT.

### Implementation

```typescript
// packages/optimal-transport/src/inverse-ot.ts

/**
 * Inverse Optimal Transport (Li et al. 2018):
 *
 * Given observed matchings T_obs (which events booked which venues),
 * find the cost weights w* that minimize:
 *   ||T_OT(C(w)) - T_obs||²
 *
 * where C(w) = w₁·C_capacity + w₂·C_price + w₃·C_amenity + w₄·C_location
 * and T_OT(C) is the Sinkhorn solution for cost matrix C.
 *
 * This learns what planners ACTUALLY value, not what we assume they value.
 * The learned weights update as more bookings accumulate — data network effect.
 */
export function learnCostWeights(
  observedMatchings: Array<{ eventId: string; venueId: string; success: boolean }>,
  events: EventRequirements[],
  venues: VenueFeatures[],
  initialWeights: CostWeights,
  learningRate: number = 0.01,
  iterations: number = 100
): CostWeights {
  // Gradient descent on the weights:
  // 1. Build C(w) from current weights
  // 2. Solve Sinkhorn → T_predicted
  // 3. Compute loss = ||T_predicted - T_observed||²
  // 4. Backpropagate through Sinkhorn (implicit differentiation or unrolling)
  // 5. Update weights: w ← w - lr · ∇w(loss)
  //
  // Use the fact that Sinkhorn is differentiable (the dual potentials f, g
  // are differentiable functions of C, which is a differentiable function of w).
  ...
}
```

This creates a **data network effect** — the more bookings the platform processes,
the better the matching becomes. This is one of the four network effect types from
your research brief (Section 7: data network effects are "nearly impossible to
replicate").

---

## Integration with Category Theory (CT Package)

The OT system connects to the categorical architecture:

- **Cost matrix builder** is a morphism: `(EventSpec, VenueSpec) → CostMatrix`
- **Sinkhorn solver** is a morphism: `CostMatrix → TransportPlan`
- **Barycenter** is a morphism: `List<Distribution> → Distribution`
- **The full pipeline** composes: `buildCost ∘ sinkhorn ∘ extractAssignment`
- **Inverse OT** is a functor from the category of observed matchings to the
  category of cost functions

The SpatialFunctor (CT-2) maps 2D layout positions to the distributions used by
displacement interpolation (OT-5).

---

## Session Management

1. **OT-1** (Sinkhorn CPU + WebGPU + property tests) — 1-2 sessions
2. **OT-2** (cost matrix builder + Sinkhorn divergence) — 1 session
3. **OT-3** (Wasserstein barycenters) — 1 session
4. **OT-4** (partial + unbalanced transport) — 1 session
5. **OT-5** (displacement interpolation) — 1 session
6. **OT-6** (visualization: Sankey, heatmap, radar) — 1 session
7. **OT-7** (inverse OT — learn cost weights) — 1 session

Each session: implement, write tests, update PROGRESS.md. Commit after each section.

---

## Python Backend (ML API)

Some computations are better done server-side (larger datasets, training inverse OT):

```
apps/ml-api/
  src/
    ot/
      solver.py         — POT-based Sinkhorn solver for large-scale problems
      barycenter.py     — Barycenter computation with POT
      inverse_ot.py     — Learn cost weights from booking history
      fgw.py            — Fused Gromov-Wasserstein for structurally incomparable features
    routes/
      matching.py       — FastAPI endpoints: POST /match, POST /barycenter, POST /learn-weights
```

Install: `uv add POT numpy scipy`

The browser (WebGPU Sinkhorn) handles real-time interactive matching for small N.
The Python backend handles batch learning (inverse OT) and large-scale problems.
