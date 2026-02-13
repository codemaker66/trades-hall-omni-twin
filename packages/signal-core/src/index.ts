// ---------------------------------------------------------------------------
// @omni-twin/signal-core — Barrel Export
// ---------------------------------------------------------------------------
// Pure-TypeScript signal processing for venue intelligence.
// 12 sub-domains: SP-1 through SP-12.

// SP-0: Shared types & matrix utilities
export type {
  Complex,
  SpectralResult,
  SeasonalityResult,
  WelchResult,
  STFTResult,
  WindowFunction,
  WaveletFamily,
  DWTResult,
  MODWTResult,
  DenoiseConfig,
  KalmanState,
  KalmanConfig,
  UKFConfig,
  RTSSmootherResult,
  ButterworthConfig,
  SOSSection,
  SavitzkyGolayConfig,
  PreprocessingResult,
  CoherenceResult,
  CepstrumResult,
  SpectralCluster,
  AnomalyFlags,
  EnsembleAnomalyResult,
  MatrixProfileResult,
  ChangePointResult,
  STLResult,
  CUSUMConfig,
  AcousticMaterial,
  RoomGeometry,
  RT60Result,
  STIResult,
  ImpulseResponse,
  CO2Config,
  OccupancyEstimate,
  ParticleFilterConfig,
  ParticleState,
  SocialForceConfig,
  CrowdAgent,
  OMPConfig,
  SparseRecoveryResult,
  MatrixCompletionConfig,
  FISTAConfig,
  IMF,
  EMDResult,
  VMDConfig,
  VMDResult,
  SSTResult,
  StockwellResult,
  SlidingDFTConfig,
  GoertzelConfig,
  RingBufferConfig,
  StreamProcessorConfig,
  WASMFFTConfig,
  WebGPUFFTConfig,
  Matrix,
} from './types.js';

export {
  createMatrix,
  matGet,
  matSet,
  matVecMul,
  matMul,
  matTranspose,
  matAdd,
  matSub,
  matScale,
  matIdentity,
  matInvert,
  matDiag,
  createPRNG,
} from './types.js';

// SP-1: Fourier Analysis
export * from './fourier/index.js';

// SP-2: Wavelet Multi-Resolution
export * from './wavelets/index.js';

// SP-3: Kalman Filtering
export * from './kalman/index.js';

// SP-4: Digital Filter Preprocessing
export * from './filters/index.js';

// SP-5: Cross-Spectral Analysis
export * from './cross-spectral/index.js';

// SP-6: Anomaly Detection Ensemble
export * from './anomaly/index.js';

// SP-7: Acoustic Simulation
export * from './acoustic/index.js';

// SP-8: Occupancy Sensing
export * from './occupancy/index.js';

// SP-9: Compressed Sensing
export * from './compressed-sensing/index.js';

// SP-10: Time-Frequency Methods
export * from './time-frequency/index.js';

// SP-11: Streaming Architecture
export * from './streaming/index.js';

// SP-12: WASM/WebGPU DSP
export * from './wasm-gpu/index.js';
