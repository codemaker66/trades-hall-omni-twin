// ---------------------------------------------------------------------------
// SP-3: Linear Kalman Filter
// ---------------------------------------------------------------------------
// Predict: x̂(k|k-1) = F·x̂(k-1|k-1) + B·u(k)
//          P(k|k-1) = F·P(k-1|k-1)·Fᵀ + Q
// Update:  K(k) = P(k|k-1)·Hᵀ·(H·P(k|k-1)·Hᵀ + R)⁻¹
//          x̂(k|k) = x̂(k|k-1) + K(k)·(z(k) - H·x̂(k|k-1))
//          P(k|k) = (I - K·H)·P(k|k-1)

import type { KalmanConfig, KalmanState, KalmanResult, DemandEstimate } from '../types.js';
import {
  arrayToMatrix, matMul, matTranspose, matAdd, matSub,
  matInvert, matVecMul, matIdentity, matScale,
  type Matrix,
} from '../types.js';

function toMatrix(data: Float64Array, rows: number, cols: number): Matrix {
  return arrayToMatrix(data, rows, cols);
}

/**
 * Create initial Kalman filter state.
 */
export function createKalmanState(dimX: number, initialState?: Float64Array, initialCovariance?: Float64Array): KalmanState {
  const x = initialState ?? new Float64Array(dimX);
  const P = initialCovariance ?? (() => {
    const p = new Float64Array(dimX * dimX);
    for (let i = 0; i < dimX; i++) p[i * dimX + i] = 100; // Large initial uncertainty
    return p;
  })();
  return { x: new Float64Array(x), P: new Float64Array(P), dim: dimX };
}

/**
 * Kalman filter predict step.
 * x̂(k|k-1) = F·x̂(k-1|k-1)
 * P(k|k-1) = F·P(k-1|k-1)·Fᵀ + Q
 */
export function kalmanPredict(state: KalmanState, config: KalmanConfig): KalmanState {
  const { dimX } = config;
  const F = toMatrix(config.F, dimX, dimX);
  const Q = toMatrix(config.Q, dimX, dimX);
  const P = toMatrix(state.P, dimX, dimX);

  const xPred = matVecMul(F, state.x);
  const FP = matMul(F, P);
  const FT = matTranspose(F);
  const PPred = matAdd(matMul(FP, FT), Q);

  return { x: xPred, P: PPred.data, dim: dimX };
}

/**
 * Kalman filter update step.
 * K = P·Hᵀ·(H·P·Hᵀ + R)⁻¹
 * x̂ = x̂ + K·(z - H·x̂)
 * P = (I - K·H)·P
 */
export function kalmanUpdate(
  state: KalmanState,
  measurement: Float64Array,
  config: KalmanConfig,
): KalmanResult {
  const { dimX, dimZ } = config;
  const H = toMatrix(config.H, dimZ, dimX);
  const R = toMatrix(config.R, dimZ, dimZ);
  const P = toMatrix(state.P, dimX, dimX);

  // Innovation: y = z - H·x
  const Hx = matVecMul(H, state.x);
  const innovation = new Float64Array(dimZ);
  for (let i = 0; i < dimZ; i++) {
    innovation[i] = measurement[i]! - Hx[i]!;
  }

  // Innovation covariance: S = H·P·Hᵀ + R
  const HT = matTranspose(H);
  const HP = matMul(H, P);
  const S = matAdd(matMul(HP, HT), R);

  // Kalman gain: K = P·Hᵀ·S⁻¹
  const SInv = matInvert(S);
  const PHT = matMul(P, HT);
  const K = matMul(PHT, SInv);

  // Updated state: x̂ = x̂ + K·y
  const Ky = matVecMul(K, innovation);
  const xUpd = new Float64Array(dimX);
  for (let i = 0; i < dimX; i++) {
    xUpd[i] = state.x[i]! + Ky[i]!;
  }

  // Updated covariance: P = (I - K·H)·P (Joseph form for numerical stability)
  const I = matIdentity(dimX);
  const KH = matMul(K, H);
  const IminusKH = matSub(I, KH);
  const PUpd = matMul(IminusKH, P);

  return {
    state: { x: xUpd, P: PUpd.data, dim: dimX },
    innovation,
    kalmanGain: K.data,
  };
}

/**
 * Run full Kalman filter (predict + update) for a single step.
 */
export function kalmanStep(
  state: KalmanState,
  measurement: Float64Array,
  config: KalmanConfig,
): KalmanResult {
  const predicted = kalmanPredict(state, config);
  return kalmanUpdate(predicted, measurement, config);
}

/**
 * Run Kalman filter over a batch of measurements.
 * Returns all intermediate states for smoothing.
 */
export function kalmanBatch(
  measurements: Float64Array[],
  config: KalmanConfig,
  initialState?: KalmanState,
): { states: KalmanState[]; predictions: KalmanState[]; innovations: Float64Array[] } {
  const states: KalmanState[] = [];
  const predictions: KalmanState[] = [];
  const innovations: Float64Array[] = [];

  let state = initialState ?? createKalmanState(config.dimX);

  for (const z of measurements) {
    const predicted = kalmanPredict(state, config);
    predictions.push(predicted);

    const result = kalmanUpdate(predicted, z, config);
    states.push(result.state);
    innovations.push(result.innovation);
    state = result.state;
  }

  return { states, predictions, innovations };
}

/**
 * Create a venue demand tracker using Kalman filter.
 * State: [demand_level, demand_velocity, seasonal_component]
 * Observations: [website_visits, inquiries, bookings]
 */
export function createDemandTracker(dt: number = 1): {
  config: KalmanConfig;
  state: KalmanState;
  update: (visits: number, inquiries: number, bookings: number) => DemandEstimate;
} {
  const dimX = 3;
  const dimZ = 3;

  const config: KalmanConfig = {
    F: new Float64Array([
      1, dt, 0,
      0, 1,  0,
      0, 0,  1,
    ]),
    H: new Float64Array([
      1.0, 0, 1.0,    // website visits (demand + seasonal)
      0.5, 0, 0.3,    // inquiries (weaker signal)
      0.2, 0, 0.1,    // bookings (weakest but cleanest)
    ]),
    Q: new Float64Array([
      1.0, 0,   0,
      0,   0.1, 0,
      0,   0,   0.5,
    ]),
    R: new Float64Array([
      25.0, 0,   0,
      0,    4.0, 0,
      0,    0,   1.0,
    ]),
    dimX,
    dimZ,
  };

  let state = createKalmanState(dimX,
    new Float64Array([100, 0, 10]),
    new Float64Array([
      100, 0, 0,
      0, 100, 0,
      0, 0, 100,
    ]),
  );

  return {
    config,
    state,
    update(visits: number, inquiries: number, bookings: number): DemandEstimate {
      const result = kalmanStep(state, new Float64Array([visits, inquiries, bookings]), config);
      state = result.state;
      return {
        demandLevel: state.x[0]!,
        demandVelocity: state.x[1]!,
        seasonal: state.x[2]!,
        uncertainty: Math.sqrt(state.P[0]!), // P[0,0]
      };
    },
  };
}
