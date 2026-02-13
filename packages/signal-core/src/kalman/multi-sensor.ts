// ---------------------------------------------------------------------------
// SP-3: Multi-Sensor Fusion via Kalman Filter
// ---------------------------------------------------------------------------
// Fuse website visits, inquiries, booking signals into unified demand estimate.
// Sequential update: process each sensor independently for flexibility.

import type { KalmanState, KalmanConfig, DemandEstimate } from '../types.js';
import { kalmanPredict, kalmanUpdate, createKalmanState } from './kalman-filter.js';

export interface SensorConfig {
  name: string;
  /** Observation matrix row (dimX elements) */
  observationRow: Float64Array;
  /** Measurement noise variance */
  noiseVariance: number;
}

/**
 * Multi-sensor fusion: sequential Kalman update with individual sensor models.
 * More flexible than batch update — can handle missing sensors per timestep.
 */
export class MultiSensorFusion {
  private state: KalmanState;
  private readonly config: KalmanConfig;
  private readonly sensors: SensorConfig[];

  constructor(
    dimX: number,
    sensors: SensorConfig[],
    F: Float64Array,
    Q: Float64Array,
    initialState?: Float64Array,
  ) {
    this.sensors = sensors;
    this.config = {
      F,
      H: new Float64Array(dimX), // Placeholder, overridden per sensor
      Q,
      R: new Float64Array(1),    // Placeholder, overridden per sensor
      dimX,
      dimZ: 1, // Sequential: 1 sensor at a time
    };
    this.state = createKalmanState(dimX, initialState);
  }

  /**
   * Predict step (call once per timestep before updates).
   */
  predict(): void {
    this.state = kalmanPredict(this.state, this.config);
  }

  /**
   * Update with a single sensor measurement.
   * Can be called multiple times per timestep for sensor fusion.
   */
  updateSensor(sensorIndex: number, value: number): void {
    const sensor = this.sensors[sensorIndex];
    if (!sensor) return;

    const singleConfig: KalmanConfig = {
      ...this.config,
      H: sensor.observationRow,
      R: new Float64Array([sensor.noiseVariance]),
      dimZ: 1,
    };

    const result = kalmanUpdate(this.state, new Float64Array([value]), singleConfig);
    this.state = result.state;
  }

  /**
   * Update with all available sensors.
   * Pass null for missing measurements.
   */
  updateAll(measurements: (number | null)[]): void {
    for (let i = 0; i < measurements.length; i++) {
      const m = measurements[i];
      if (m !== null && m !== undefined) {
        this.updateSensor(i, m);
      }
    }
  }

  /**
   * Get current demand estimate.
   */
  getEstimate(): DemandEstimate {
    return {
      demandLevel: this.state.x[0]!,
      demandVelocity: this.state.x.length > 1 ? this.state.x[1]! : 0,
      seasonal: this.state.x.length > 2 ? this.state.x[2]! : 0,
      uncertainty: Math.sqrt(this.state.P[0]!),
    };
  }

  getState(): KalmanState {
    return this.state;
  }
}

/**
 * Create a standard 3-sensor venue demand fusion system.
 */
export function createVenueFusion(dt: number = 1): MultiSensorFusion {
  const dimX = 3; // [demand, velocity, seasonal]

  const sensors: SensorConfig[] = [
    {
      name: 'website_visits',
      observationRow: new Float64Array([1.0, 0, 1.0]),
      noiseVariance: 25.0,
    },
    {
      name: 'inquiries',
      observationRow: new Float64Array([0.5, 0, 0.3]),
      noiseVariance: 4.0,
    },
    {
      name: 'bookings',
      observationRow: new Float64Array([0.2, 0, 0.1]),
      noiseVariance: 1.0,
    },
  ];

  const F = new Float64Array([
    1, dt, 0,
    0, 1,  0,
    0, 0,  1,
  ]);

  const Q = new Float64Array([
    1.0, 0,   0,
    0,   0.1, 0,
    0,   0,   0.5,
  ]);

  return new MultiSensorFusion(dimX, sensors, F, Q, new Float64Array([100, 0, 10]));
}
