// ---------------------------------------------------------------------------
// Statistical Learning Theory — Core Types
// ---------------------------------------------------------------------------

/** Seedable PRNG function (same pattern as physics-solvers). */
export type PRNG = () => number;

// ---------------------------------------------------------------------------
// SLT-1: Conformal Prediction
// ---------------------------------------------------------------------------

export interface PredictionInterval {
  lower: number;
  upper: number;
  confidenceLevel: number;
}

export interface ConformalResult {
  pointEstimate: number[];
  intervals: PredictionInterval[];
}

export interface ConformalConfig {
  alpha: number; // Miscoverage rate (0.1 = 90% confidence)
  method: 'split' | 'jackknife_plus' | 'cqr' | 'aci' | 'enbpi' | 'weighted';
}

export interface ACIState {
  alphaTarget: number;
  alphaT: number;
  gamma: number;
  coverageHistory: number[];
}

export interface WeightedConformalConfig {
  alpha: number;
  weights: number[];  // Importance weights p_test(X_i) / p_train(X_i)
}

// ---------------------------------------------------------------------------
// SLT-2: PAC-Bayes Bounds
// ---------------------------------------------------------------------------

export interface PACBayesBound {
  name: string;
  value: number;
  empRisk: number;
  klDivergence: number;
  n: number;
  delta: number;
}

export interface ModelCandidate {
  name: string;
  empRisk: number;
  klDiv: number;
  vcDimension?: number;
}

// ---------------------------------------------------------------------------
// SLT-3: VC Dimension / Sample Complexity
// ---------------------------------------------------------------------------

export interface SampleComplexityResult {
  minSamples: number;
  vcDimension: number;
  epsilon: number;
  delta: number;
  recommendedModel: string;
}

// ---------------------------------------------------------------------------
// SLT-4: Bandits
// ---------------------------------------------------------------------------

export interface BanditArm {
  index: number;
  pulls: number;
  totalReward: number;
  meanReward: number;
}

export interface BanditState {
  arms: BanditArm[];
  totalPulls: number;
  cumulativeRegret: number;
}

export interface LinUCBConfig {
  nArms: number;
  d: number;       // Context dimension
  alpha: number;   // Exploration coefficient
}

export interface ThompsonState {
  alpha: number[];  // Beta distribution successes + 1
  beta: number[];   // Beta distribution failures + 1
}

export interface HedgeConfig {
  nExperts: number;
  eta: number;  // Learning rate
}

export interface EXP3Config {
  nArms: number;
  gamma: number;  // Exploration parameter
}

// ---------------------------------------------------------------------------
// SLT-5: Gaussian Processes
// ---------------------------------------------------------------------------

export type KernelType = 'rbf' | 'matern32' | 'matern52' | 'periodic' | 'linear' | 'spectral_mixture';

export interface KernelConfig {
  type: KernelType;
  lengthscale: number;
  variance: number;
  period?: number;       // For periodic kernel
  nu?: number;           // For Matérn
  weights?: number[];    // For spectral mixture
  means?: number[];      // For spectral mixture
  variances?: number[];  // For spectral mixture
}

export interface CompositeKernelConfig {
  op: 'add' | 'mul';
  left: KernelConfig | CompositeKernelConfig;
  right: KernelConfig | CompositeKernelConfig;
}

export interface GPConfig {
  kernel: KernelConfig | CompositeKernelConfig;
  noiseVariance: number;
  meanFunction: 'zero' | 'constant';
  meanValue?: number;
}

export interface GPPrediction {
  mean: number[];
  variance: number[];
  lower: number[];
  upper: number[];
  logMarginalLikelihood: number;
}

export interface BayesOptConfig {
  bounds: Array<[number, number]>;
  acquisitionFn: 'ei' | 'ucb' | 'pi';  // Expected Improvement, UCB, Prob. of Improvement
  kappa: number;          // UCB exploration parameter
  xi: number;             // EI/PI exploration parameter
  nInitial: number;       // Initial random evaluations
  maxIterations: number;
}

export interface BayesOptResult {
  bestX: number[];
  bestY: number;
  history: Array<{ x: number[]; y: number; acquisition: number }>;
}

// ---------------------------------------------------------------------------
// SLT-6: Tree Ensembles
// ---------------------------------------------------------------------------

export interface TreeNode {
  featureIndex: number;
  threshold: number;
  left: TreeNode | TreeLeaf;
  right: TreeNode | TreeLeaf;
}

export interface TreeLeaf {
  values: number[];  // All training values that reached this leaf
}

export interface RandomForestConfig {
  nEstimators: number;
  maxDepth: number;
  minSamplesLeaf: number;
  maxFeatures: number | 'sqrt' | 'log2';
  seed?: number;
}

export interface QuantileForestPrediction {
  quantiles: Map<number, number[]>;  // quantile → predicted values
  median: number[];
}

export interface SHAPValue {
  feature: string;
  value: number;
  direction: 'increases' | 'decreases';
}

export interface GradientBoostConfig {
  nEstimators: number;
  learningRate: number;
  maxDepth: number;
  subsample: number;
  loss: 'squared' | 'absolute' | 'huber' | 'quantile';
  quantile?: number;  // For quantile loss
  seed?: number;
}

// ---------------------------------------------------------------------------
// SLT-7: Hierarchical Bayesian
// ---------------------------------------------------------------------------

export interface HierarchicalConfig {
  nGroups: number;
  nSamples: number;
  nChains: number;
  tuneSamples: number;
  targetAccept: number;
}

export interface HierarchicalPosterior {
  globalMean: number;
  globalStd: number;
  groupMeans: number[];
  groupStds: number[];
  observationNoise: number;
  samples: Map<string, number[]>;  // Parameter name → posterior samples
}

export interface MCDropoutConfig {
  dropoutRate: number;
  nForwardPasses: number;
}

export interface MCDropoutPrediction {
  mean: number[];
  epistemicStd: number[];
  aleatoricStd: number[];
  totalStd: number[];
  samples: number[][];
}

// ---------------------------------------------------------------------------
// SLT-8: Drift Detection
// ---------------------------------------------------------------------------

export interface DriftDetectorState {
  driftDetected: boolean;
  warningDetected: boolean;
  nObservations: number;
}

export interface ADWINConfig {
  delta: number;  // Confidence parameter (default 0.002)
}

export interface DDMConfig {
  minInstances: number;  // Minimum instances before detection (default 30)
  warningLevel: number;  // Standard deviations for warning (default 2)
  driftLevel: number;    // Standard deviations for drift (default 3)
}

export interface PageHinkleyConfig {
  delta: number;     // Magnitude threshold
  lambda: number;    // Detection threshold
  alpha: number;     // Forgetting factor
}

export interface DROConfig {
  epsilon: number;   // Wasserstein ball radius
  nIterations: number;
}

export interface EWCConfig {
  lambda: number;    // Importance weight
}

// ---------------------------------------------------------------------------
// SLT-9: Causal Inference
// ---------------------------------------------------------------------------

export interface CausalEffect {
  ate: number;         // Average Treatment Effect
  ciLower: number;     // Confidence interval lower
  ciUpper: number;     // Confidence interval upper
  pValue: number;
}

export interface DMLConfig {
  nFolds: number;      // Cross-fitting folds (default 5)
  nEstimators: number; // Trees in nuisance models
}

export interface IVResult {
  coefficient: number;
  standardError: number;
  ciLower: number;
  ciUpper: number;
  firstStageF: number;  // Weak instrument diagnostic
}

export interface SequentialTestState {
  nObservations: number;
  treatmentSum: number;
  controlSum: number;
  treatmentN: number;
  controlN: number;
  rejected: boolean;
  confidenceSequence: Array<{ n: number; lower: number; upper: number }>;
}

export interface SyntheticControlResult {
  weights: number[];
  preEffect: number;
  postEffect: number;
  ciLower: number;
  ciUpper: number;
  placeboEffects: number[];
}

// ---------------------------------------------------------------------------
// SLT-10: Information Theory
// ---------------------------------------------------------------------------

export interface MutualInformationResult {
  features: string[];
  scores: number[];
  selectedIndices: number[];
}

export interface DivergenceResult {
  kl: number;
  reverseKl: number;
  jsd: number;
  jsdMetric: number;   // √JSD — a true metric
}

export interface MDLResult {
  modelIndex: number;
  modelComplexity: number;
  dataFit: number;
  totalLength: number;
}

export interface FisherOEDResult {
  optimalDesign: number[];  // Optimal allocation of experiments
  fisherInfo: number[][];   // Fisher information matrix
  dOptimality: number;      // det(I(θ))
}

// ---------------------------------------------------------------------------
// SLT-11: Calibration & Fairness
// ---------------------------------------------------------------------------

export interface CalibrationResult {
  bins: Array<{
    predictedMean: number;
    observedFrequency: number;
    count: number;
  }>;
  ece: number;           // Expected Calibration Error
  mce: number;           // Maximum Calibration Error
  brier: number;         // Brier score
}

export interface PlattParams {
  a: number;  // Slope
  b: number;  // Intercept
}

export interface IsotonicParams {
  xs: number[];  // Breakpoints
  ys: number[];  // Values at breakpoints
}

export interface TemperatureParams {
  temperature: number;
}

export interface MultiCalibrationConfig {
  alpha: number;         // Calibration tolerance
  subgroups: string[];   // Subgroup identifiers
  maxIterations: number;
}

// ---------------------------------------------------------------------------
// SLT-12: Production API + Monitoring
// ---------------------------------------------------------------------------

export interface PricePredictionResponse {
  prediction: {
    pointEstimate: number;
    predictionIntervals: PredictionInterval[];
  };
  uncertainty: {
    level: 'low' | 'medium' | 'high';
    intervalWidth: number;
    epistemicStd: number;
    aleatoricStd: number;
    dataSupport: number;
  };
  explanation: {
    topDrivers: SHAPValue[];
  };
  metadata: {
    modelVersion: string;
    calibrationCoverage: number;
    isColdStart: boolean;
    pacBayesBound: number;
  };
}

export interface CoverageMetrics {
  nominal: number;
  empirical: number;
  rolling: number[];
  windowSize: number;
  alert: boolean;
}

export interface PITHistogram {
  bins: number[];
  counts: number[];
  uniformityPValue: number;
}

export interface RetrainingTriggerResult {
  shouldRetrain: boolean;
  reason: string | null;
  driftScore: number;
  coverageDrop: number;
  lastRetrainedAt: number;
}

// ---------------------------------------------------------------------------
// Shared utility types
// ---------------------------------------------------------------------------

/** Dense matrix as flat Float64Array + dimensions. */
export interface Matrix {
  data: Float64Array;
  rows: number;
  cols: number;
}

/** Create a matrix from 2D array. */
export function createMatrix(values: number[][]): Matrix {
  const rows = values.length;
  const cols = values[0]?.length ?? 0;
  const data = new Float64Array(rows * cols);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      data[i * cols + j] = values[i]![j]!;
    }
  }
  return { data, rows, cols };
}

/** Get element at (i, j). */
export function matGet(m: Matrix, i: number, j: number): number {
  return m.data[i * m.cols + j]!;
}

/** Set element at (i, j). */
export function matSet(m: Matrix, i: number, j: number, v: number): void {
  m.data[i * m.cols + j] = v;
}

/** Matrix-vector multiply: A @ x → result. */
export function matVecMul(A: Matrix, x: Float64Array): Float64Array {
  const result = new Float64Array(A.rows);
  for (let i = 0; i < A.rows; i++) {
    let sum = 0;
    for (let j = 0; j < A.cols; j++) {
      sum += (A.data[i * A.cols + j] ?? 0) * (x[j] ?? 0);
    }
    result[i] = sum;
  }
  return result;
}

/** Dot product of two vectors. */
export function dot(a: Float64Array, b: Float64Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += (a[i] ?? 0) * (b[i] ?? 0);
  }
  return sum;
}

/** Cholesky decomposition (lower triangular L such that A = L L^T). */
export function cholesky(A: Matrix): Matrix {
  const n = A.rows;
  const L = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = 0;
      for (let k = 0; k < j; k++) {
        sum += (L[i * n + k] ?? 0) * (L[j * n + k] ?? 0);
      }
      if (i === j) {
        const diag = (A.data[i * n + i] ?? 0) - sum;
        L[i * n + j] = Math.sqrt(Math.max(diag, 1e-12));
      } else {
        L[i * n + j] = ((A.data[i * n + j] ?? 0) - sum) / (L[j * n + j] ?? 1);
      }
    }
  }
  return { data: L, rows: n, cols: n };
}

/** Solve L @ x = b for lower triangular L (forward substitution). */
export function solveLower(L: Matrix, b: Float64Array): Float64Array {
  const n = L.rows;
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < i; j++) {
      sum += (L.data[i * n + j] ?? 0) * (x[j] ?? 0);
    }
    x[i] = ((b[i] ?? 0) - sum) / (L.data[i * n + i] ?? 1);
  }
  return x;
}

/** Solve L^T @ x = b for lower triangular L (back substitution). */
export function solveUpper(L: Matrix, b: Float64Array): Float64Array {
  const n = L.rows;
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let sum = 0;
    for (let j = i + 1; j < n; j++) {
      sum += (L.data[j * n + i] ?? 0) * (x[j] ?? 0);
    }
    x[i] = ((b[i] ?? 0) - sum) / (L.data[i * n + i] ?? 1);
  }
  return x;
}

/** Solve A @ x = b via Cholesky decomposition. */
export function choleskySolve(A: Matrix, b: Float64Array): Float64Array {
  const L = cholesky(A);
  const y = solveLower(L, b);
  return solveUpper(L, y);
}

/** Seedable PRNG — mulberry32. */
export function createPRNG(seed: number): PRNG {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Sample from standard normal using Box-Muller. */
export function normalSample(rng: PRNG): number {
  const u1 = rng();
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(Math.max(u1, 1e-15))) * Math.cos(2 * Math.PI * u2);
}

/** Sample from Beta(a, b) distribution using Jöhnk's algorithm. */
export function betaSample(a: number, b: number, rng: PRNG): number {
  // Use gamma sampling for Beta
  const ga = gammaSample(a, 1, rng);
  const gb = gammaSample(b, 1, rng);
  return ga / (ga + gb);
}

/** Sample from Gamma(shape, scale) distribution (Marsaglia & Tsang). */
export function gammaSample(shape: number, scale: number, rng: PRNG): number {
  if (shape < 1) {
    return gammaSample(shape + 1, scale, rng) * Math.pow(rng(), 1 / shape);
  }
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x: number;
    let v: number;
    do {
      x = normalSample(rng);
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = rng();
    if (u < 1 - 0.0331 * (x * x) * (x * x)) return d * v * scale;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v * scale;
  }
}
