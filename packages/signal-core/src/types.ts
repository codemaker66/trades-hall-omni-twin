// ---------------------------------------------------------------------------
// @omni-twin/signal-core — Signal Processing Types
// ---------------------------------------------------------------------------

/** Seedable PRNG function (same pattern as physics-solvers / learning-core). */
export type PRNG = () => number;

// ---------------------------------------------------------------------------
// Complex number representation
// ---------------------------------------------------------------------------

export interface Complex {
  re: number;
  im: number;
}

// ---------------------------------------------------------------------------
// SP-1: Fourier Analysis
// ---------------------------------------------------------------------------

export interface SpectralResult {
  frequencies: Float64Array;
  magnitudes: Float64Array;
  phases: Float64Array;
}

export interface SeasonalityResult {
  frequencies: Float64Array;
  magnitudes: Float64Array;
  periods: Float64Array;
  dominantPeriods: Array<{ period: number; magnitude: number }>;
}

export interface WelchResult {
  frequencies: Float64Array;
  psd: Float64Array;
}

export interface STFTResult {
  frequencies: Float64Array;
  times: Float64Array;
  /** Row-major: spectrogram[t * nFreqs + f] */
  spectrogram: Float64Array;
  nFreqs: number;
  nTimes: number;
}

export type WindowFunction =
  | 'rectangular'
  | 'hann'
  | 'hamming'
  | 'blackman'
  | 'blackman-harris'
  | 'kaiser';

// ---------------------------------------------------------------------------
// SP-2: Wavelet Analysis
// ---------------------------------------------------------------------------

export type WaveletFamily = 'haar' | 'db4' | 'db8' | 'sym4' | 'coif2';

export interface DWTResult {
  approximation: Float64Array;
  details: Float64Array[];
  wavelet: WaveletFamily;
  levels: number;
}

export interface MODWTResult {
  approximation: Float64Array;
  details: Float64Array[];
  wavelet: WaveletFamily;
  levels: number;
}

export interface DenoiseConfig {
  wavelet: WaveletFamily;
  method: 'soft' | 'hard';
  levels?: number;
  /** 'universal' (Donoho-Johnstone) or 'bayes-shrink' */
  thresholdRule: 'universal' | 'bayes-shrink';
}

export interface MultiscaleForecastResult {
  forecast: Float64Array;
  componentForecasts: Float64Array[];
  levels: number;
}

// ---------------------------------------------------------------------------
// SP-3: Kalman Filtering
// ---------------------------------------------------------------------------

export interface KalmanState {
  /** State vector (dim_x) */
  x: Float64Array;
  /** Covariance matrix (dim_x × dim_x, row-major) */
  P: Float64Array;
  dim: number;
}

export interface KalmanConfig {
  /** State transition matrix F (dim_x × dim_x) */
  F: Float64Array;
  /** Observation matrix H (dim_z × dim_x) */
  H: Float64Array;
  /** Process noise Q (dim_x × dim_x) */
  Q: Float64Array;
  /** Measurement noise R (dim_z × dim_z) */
  R: Float64Array;
  dimX: number;
  dimZ: number;
}

export interface KalmanResult {
  state: KalmanState;
  innovation: Float64Array;
  kalmanGain: Float64Array;
}

export interface UKFConfig extends KalmanConfig {
  alpha: number; // Spread of sigma points (~1e-3 to 1)
  beta: number;  // Prior knowledge (2 for Gaussian)
  kappa: number; // Secondary scaling (usually 0)
  /** Nonlinear state transition f(x) → x */
  stateTransitionFn: (x: Float64Array) => Float64Array;
  /** Nonlinear observation h(x) → z */
  observationFn: (x: Float64Array) => Float64Array;
}

export interface RTSSmootherResult {
  smoothedStates: Float64Array[];
  smoothedCovariances: Float64Array[];
}

export interface AdaptiveKalmanConfig extends KalmanConfig {
  /** Exponential forgetting factor α ∈ [0.95, 0.99] */
  forgettingFactor: number;
}

export interface DemandEstimate {
  demandLevel: number;
  demandVelocity: number;
  seasonal: number;
  uncertainty: number;
}

// ---------------------------------------------------------------------------
// SP-4: Digital Filter Preprocessing
// ---------------------------------------------------------------------------

export type FilterType = 'lowpass' | 'highpass' | 'bandpass' | 'bandstop';

export interface ButterworthConfig {
  order: number;
  cutoff: number | [number, number]; // Single for low/high, pair for band
  type: FilterType;
  fs: number;
}

/** Second-order section: [b0, b1, b2, a0, a1, a2] */
export type SOSSection = [number, number, number, number, number, number];

export interface SavitzkyGolayConfig {
  windowLength: number;
  polyOrder: number;
  deriv?: number;
}

export interface PreprocessingResult {
  cleaned: Float64Array;
  trend: Float64Array;
  weekly: Float64Array;
  monthly: Float64Array;
  annual: Float64Array;
  velocity: Float64Array;
  acceleration: Float64Array;
}

// ---------------------------------------------------------------------------
// SP-5: Cross-Spectral Analysis
// ---------------------------------------------------------------------------

export interface CoherenceResult {
  frequencies: Float64Array;
  coherence: Float64Array; // γ²(f) ∈ [0,1]
  phase: Float64Array;     // radians
}

export interface CepstrumResult {
  quefrencies: Float64Array;
  cepstrum: Float64Array;
  dominantQuefrencies: number[];
}

export interface SpectralCluster {
  clusterId: number;
  label: string;
  memberIndices: number[];
  centroidPSD: Float64Array;
}

// ---------------------------------------------------------------------------
// SP-6: Anomaly Detection
// ---------------------------------------------------------------------------

export interface AnomalyFlags {
  anomalies: boolean[];
  scores: Float64Array;
  method: string;
}

export interface EnsembleAnomalyResult {
  consensus: boolean[];
  stl: AnomalyFlags;
  matrixProfile: AnomalyFlags;
  cusum: AnomalyFlags;
  spectralResidual: AnomalyFlags;
  /** Number of methods flagging each point */
  voteCount: Uint8Array;
}

export interface MatrixProfileResult {
  /** Distance profile */
  profile: Float64Array;
  /** Nearest neighbor indices */
  profileIndex: Int32Array;
  /** Subsequence length */
  windowSize: number;
}

export interface ChangePointResult {
  /** Indices of detected changepoints */
  changepoints: number[];
  /** Cost/penalty used */
  penalty: number;
}

export interface STLResult {
  trend: Float64Array;
  seasonal: Float64Array;
  remainder: Float64Array;
}

export interface CUSUMConfig {
  /** Allowable slack (in σ units, typically 0.5) */
  k: number;
  /** Decision threshold (in σ units, typically 5) */
  h: number;
}

// ---------------------------------------------------------------------------
// SP-7: Acoustic Simulation
// ---------------------------------------------------------------------------

export interface AcousticMaterial {
  name: string;
  /** Absorption coefficients at standard octave bands [125, 250, 500, 1k, 2k, 4k Hz] */
  absorption: [number, number, number, number, number, number];
  /** Scattering coefficient (0-1) */
  scattering: number;
}

export interface RoomGeometry {
  /** Room dimensions in meters [length, width, height] */
  dimensions: [number, number, number];
  /** Surface areas in m² and their materials */
  surfaces: Array<{
    area: number;
    material: AcousticMaterial;
  }>;
  /** Room volume in m³ */
  volume: number;
}

export interface RT60Result {
  /** RT60 per octave band [125, 250, 500, 1k, 2k, 4k Hz] */
  rt60: [number, number, number, number, number, number];
  /** Average across 500-2kHz (speech range) */
  rt60Mid: number;
  formula: 'sabine' | 'eyring' | 'fitzroy';
}

export interface STIResult {
  sti: number;
  rating: 'excellent' | 'good' | 'fair' | 'poor' | 'bad';
  /** Modulation Transfer Function values per octave band */
  mtf: Float64Array;
}

export interface ImpulseResponse {
  /** Time-domain impulse response samples */
  samples: Float64Array;
  sampleRate: number;
  /** Duration in seconds */
  duration: number;
}

// ---------------------------------------------------------------------------
// SP-8: Occupancy Sensing
// ---------------------------------------------------------------------------

export interface CO2Config {
  /** Outdoor CO2 concentration (ppm), typically ~420 */
  outdoorCO2: number;
  /** Ventilation rate (L/s per person) */
  ventilationRate: number;
  /** CO2 generation rate (L/s per person), ~0.005 at rest */
  generationRate: number;
}

export interface OccupancyEstimate {
  count: number;
  uncertainty: number;
  /** Confidence interval */
  lower: number;
  upper: number;
}

export interface ParticleFilterConfig {
  nParticles: number;
  /** Max occupancy capacity */
  maxOccupancy: number;
  /** Process noise std dev */
  processNoise: number;
  /** Measurement noise std dev */
  measurementNoise: number;
}

export interface ParticleState {
  particles: Float64Array;
  weights: Float64Array;
  estimate: OccupancyEstimate;
  effectiveSampleSize: number;
}

export interface SocialForceConfig {
  /** Desired speed (m/s) */
  desiredSpeed: number;
  /** Relaxation time (s) */
  relaxationTime: number;
  /** Social force magnitude */
  socialForceMagnitude: number;
  /** Social force range (m) */
  socialForceRange: number;
}

export interface CrowdAgent {
  x: number;
  y: number;
  vx: number;
  vy: number;
  goalX: number;
  goalY: number;
}

// ---------------------------------------------------------------------------
// SP-9: Compressed Sensing
// ---------------------------------------------------------------------------

export interface OMPConfig {
  /** Number of nonzero coefficients to recover */
  nComponents: number;
  /** Convergence tolerance */
  tolerance?: number;
}

export interface SparseRecoveryResult {
  /** Recovered full signal */
  signal: Float64Array;
  /** Sparse coefficients */
  coefficients: Float64Array;
  /** Support set (indices of nonzero coefficients) */
  support: number[];
  /** Residual norm */
  residualNorm: number;
}

export interface MatrixCompletionConfig {
  /** Number of rows */
  nRows: number;
  /** Number of columns */
  nCols: number;
  /** Regularization strength */
  lambda: number;
  /** Max iterations for ADMM */
  maxIter: number;
  tolerance: number;
}

export interface MatrixCompletionResult {
  /** Completed matrix (row-major) */
  completed: Float64Array;
  /** Estimated rank */
  rank: number;
  /** Convergence residual */
  residual: number;
}

export interface FISTAConfig {
  /** Step size */
  stepSize: number;
  /** Regularization parameter */
  lambda: number;
  maxIter: number;
  tolerance: number;
}

// ---------------------------------------------------------------------------
// SP-10: Time-Frequency Methods
// ---------------------------------------------------------------------------

export interface IMF {
  /** Intrinsic mode function */
  data: Float64Array;
  /** Instantaneous frequency */
  instantFrequency?: Float64Array;
  /** Instantaneous amplitude */
  instantAmplitude?: Float64Array;
}

export interface EMDResult {
  imfs: IMF[];
  residue: Float64Array;
  nIterations: number;
}

export interface VMDConfig {
  /** Number of modes to extract */
  nModes: number;
  /** Penalty parameter (data-fidelity constraint) */
  alpha: number;
  /** Noise tolerance (Lagrangian dual) */
  tau: number;
  /** DC component flag */
  dc: boolean;
  maxIter: number;
  tolerance: number;
}

export interface VMDResult {
  modes: Float64Array[];
  centerFrequencies: Float64Array;
  nIterations: number;
}

export interface SSTResult {
  /** Time-frequency representation (nFreqs × nTimes, row-major) */
  tfr: Float64Array;
  frequencies: Float64Array;
  times: Float64Array;
  nFreqs: number;
  nTimes: number;
}

export interface StockwellResult {
  /** S-transform (nFreqs × nTimes, row-major) */
  stransform: Float64Array;
  frequencies: Float64Array;
  times: Float64Array;
  nFreqs: number;
  nTimes: number;
}

// ---------------------------------------------------------------------------
// SP-11: Streaming Architecture
// ---------------------------------------------------------------------------

export interface SlidingDFTConfig {
  /** Window size N */
  windowSize: number;
  /** Which bins to track (subset of 0..N-1) */
  trackedBins?: number[];
}

export interface GoertzelConfig {
  /** Target frequency in Hz */
  targetFrequency: number;
  /** Sample rate */
  sampleRate: number;
  /** Block size */
  blockSize: number;
}

export interface RingBufferConfig {
  /** Capacity in samples */
  capacity: number;
  /** Number of channels (default 1) */
  channels?: number;
}

export interface StreamProcessorConfig {
  /** Input sample rate */
  sampleRate: number;
  /** FFT window size */
  fftSize: number;
  /** Hop size between windows */
  hopSize: number;
  /** Window function */
  window: WindowFunction;
}

// ---------------------------------------------------------------------------
// SP-12: WASM/WebGPU DSP
// ---------------------------------------------------------------------------

export interface WASMFFTConfig {
  size: number;
  /** Use SIMD if available */
  useSIMD: boolean;
}

export interface WebGPUFFTConfig {
  size: number;
  /** Use Stockham auto-sort (avoids bit-reversal) */
  useStockham: boolean;
}

// ---------------------------------------------------------------------------
// Shared utility types
// ---------------------------------------------------------------------------

/** Row-major matrix stored as flat Float64Array */
export interface Matrix {
  data: Float64Array;
  rows: number;
  cols: number;
}

/** Create a new Matrix */
export function createMatrix(rows: number, cols: number): Matrix {
  return { data: new Float64Array(rows * cols), rows, cols };
}

/** Get element at (i, j) */
export function matGet(m: Matrix, i: number, j: number): number {
  return m.data[i * m.cols + j]!;
}

/** Set element at (i, j) */
export function matSet(m: Matrix, i: number, j: number, val: number): void {
  m.data[i * m.cols + j] = val;
}

/** Matrix-vector multiply: y = A·x */
export function matVecMul(A: Matrix, x: Float64Array): Float64Array {
  const y = new Float64Array(A.rows);
  for (let i = 0; i < A.rows; i++) {
    let sum = 0;
    for (let j = 0; j < A.cols; j++) {
      sum += matGet(A, i, j) * x[j]!;
    }
    y[i] = sum;
  }
  return y;
}

/** Matrix-matrix multiply: C = A·B */
export function matMul(A: Matrix, B: Matrix): Matrix {
  const C = createMatrix(A.rows, B.cols);
  for (let i = 0; i < A.rows; i++) {
    for (let j = 0; j < B.cols; j++) {
      let sum = 0;
      for (let k = 0; k < A.cols; k++) {
        sum += matGet(A, i, k) * matGet(B, k, j);
      }
      matSet(C, i, j, sum);
    }
  }
  return C;
}

/** Matrix transpose */
export function matTranspose(A: Matrix): Matrix {
  const T = createMatrix(A.cols, A.rows);
  for (let i = 0; i < A.rows; i++) {
    for (let j = 0; j < A.cols; j++) {
      matSet(T, j, i, matGet(A, i, j));
    }
  }
  return T;
}

/** Matrix addition: C = A + B */
export function matAdd(A: Matrix, B: Matrix): Matrix {
  const C = createMatrix(A.rows, A.cols);
  for (let i = 0; i < A.data.length; i++) {
    C.data[i] = A.data[i]! + B.data[i]!;
  }
  return C;
}

/** Matrix subtraction: C = A - B */
export function matSub(A: Matrix, B: Matrix): Matrix {
  const C = createMatrix(A.rows, A.cols);
  for (let i = 0; i < A.data.length; i++) {
    C.data[i] = A.data[i]! - B.data[i]!;
  }
  return C;
}

/** Scale matrix: C = s·A */
export function matScale(A: Matrix, s: number): Matrix {
  const C = createMatrix(A.rows, A.cols);
  for (let i = 0; i < A.data.length; i++) {
    C.data[i] = A.data[i]! * s;
  }
  return C;
}

/** Identity matrix */
export function matIdentity(n: number): Matrix {
  const I = createMatrix(n, n);
  for (let i = 0; i < n; i++) {
    matSet(I, i, i, 1);
  }
  return I;
}

/** Invert a small matrix via Gauss-Jordan (for Kalman filter, n ≤ ~10) */
export function matInvert(A: Matrix): Matrix {
  const n = A.rows;
  // Augmented matrix [A | I]
  const aug = createMatrix(n, 2 * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      matSet(aug, i, j, matGet(A, i, j));
    }
    matSet(aug, i, n + i, 1);
  }

  for (let col = 0; col < n; col++) {
    // Partial pivoting
    let maxRow = col;
    let maxVal = Math.abs(matGet(aug, col, col));
    for (let row = col + 1; row < n; row++) {
      const val = Math.abs(matGet(aug, row, col));
      if (val > maxVal) {
        maxVal = val;
        maxRow = row;
      }
    }
    if (maxRow !== col) {
      for (let j = 0; j < 2 * n; j++) {
        const tmp = matGet(aug, col, j);
        matSet(aug, col, j, matGet(aug, maxRow, j));
        matSet(aug, maxRow, j, tmp);
      }
    }

    const pivot = matGet(aug, col, col);
    if (Math.abs(pivot) < 1e-15) {
      throw new Error('Matrix is singular or near-singular');
    }

    // Scale pivot row
    for (let j = 0; j < 2 * n; j++) {
      matSet(aug, col, j, matGet(aug, col, j) / pivot);
    }

    // Eliminate column
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = matGet(aug, row, col);
      for (let j = 0; j < 2 * n; j++) {
        matSet(aug, row, j, matGet(aug, row, j) - factor * matGet(aug, col, j));
      }
    }
  }

  // Extract inverse
  const inv = createMatrix(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      matSet(inv, i, j, matGet(aug, i, n + j));
    }
  }
  return inv;
}

/** Convert flat Float64Array to Matrix */
export function arrayToMatrix(data: Float64Array, rows: number, cols: number): Matrix {
  return { data: new Float64Array(data), rows, cols };
}

/** Diagonal matrix from vector */
export function matDiag(v: Float64Array): Matrix {
  const n = v.length;
  const D = createMatrix(n, n);
  for (let i = 0; i < n; i++) {
    matSet(D, i, i, v[i]!);
  }
  return D;
}

/** Create simple seedable PRNG (xorshift128+) */
export function createPRNG(seed: number): PRNG {
  let s0 = seed | 0 || 1;
  let s1 = (seed >>> 16) ^ 0x5DEECE66D;
  if (s1 === 0) s1 = 0xDEADBEEF;
  return () => {
    let x = s0;
    const y = s1;
    s0 = y;
    x ^= x << 23;
    x ^= x >> 17;
    x ^= y;
    x ^= y >> 26;
    s1 = x;
    return ((s0 + s1) >>> 0) / 0x100000000;
  };
}
