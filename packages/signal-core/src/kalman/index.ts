// SP-3: Kalman Filtering
export {
  createKalmanState,
  kalmanPredict,
  kalmanUpdate,
  kalmanStep,
  kalmanBatch,
  createDemandTracker,
} from './kalman-filter.js';

export {
  generateSigmaPoints,
  ukfPredict,
  ukfUpdate,
  ukfStep,
} from './ukf.js';

export { rtsSmooth } from './rts-smoother.js';

export {
  createAdaptiveState,
  adaptiveKalmanStep,
  type AdaptiveKalmanState,
} from './adaptive.js';

export {
  MultiSensorFusion,
  createVenueFusion,
  type SensorConfig,
} from './multi-sensor.js';
