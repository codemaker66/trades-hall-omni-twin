// ---------------------------------------------------------------------------
// SP-4: Full Preprocessing Pipeline
// ---------------------------------------------------------------------------
// Recommended order:
// 1. Median filter (kernel=5): outlier removal, 50% breakdown point
// 2. Savitzky-Golay (window=15, poly=3): denoise preserving peaks
// 3. Butterworth bandpass: extract weekly/monthly/annual bands
// 4. Low-pass Butterworth (cutoff=1/30): trend extraction
// 5. SG with deriv=1,2: velocity/acceleration features

import type { PreprocessingResult } from '../types.js';
import { medianFilter } from './median.js';
import { savitzkyGolayFilter } from './savitzky-golay.js';
import { designButterworth, sosfiltfilt } from './butterworth.js';

/**
 * Full booking data preprocessing pipeline.
 */
export function preprocessBookings(
  bookings: Float64Array,
  fs: number = 1,
): PreprocessingResult {
  // Step 1: Robust outlier removal
  const step1 = medianFilter(bookings, 5);

  // Step 2: Smooth preserving peaks
  const step2 = savitzkyGolayFilter(step1, {
    windowLength: 15,
    polyOrder: 3,
  });

  // Step 3: Bandpass features
  const weeklyFilter = designButterworth({
    order: 4, cutoff: [1 / 8, 1 / 6], type: 'bandpass', fs,
  });
  const monthlyFilter = designButterworth({
    order: 4, cutoff: [1 / 35, 1 / 25], type: 'bandpass', fs,
  });
  const annualFilter = designButterworth({
    order: 4, cutoff: [1 / 400, 1 / 300], type: 'bandpass', fs,
  });

  const weekly = weeklyFilter.length > 0 ? sosfiltfilt(weeklyFilter, step2) : new Float64Array(step2.length);
  const monthly = monthlyFilter.length > 0 ? sosfiltfilt(monthlyFilter, step2) : new Float64Array(step2.length);
  const annual = annualFilter.length > 0 ? sosfiltfilt(annualFilter, step2) : new Float64Array(step2.length);

  // Step 4: Trend (low-pass)
  const trendFilter = designButterworth({
    order: 4, cutoff: 1 / 30, type: 'lowpass', fs,
  });
  const trend = trendFilter.length > 0 ? sosfiltfilt(trendFilter, step2) : step2;

  // Step 5: Derivative features
  const velocity = savitzkyGolayFilter(step2, {
    windowLength: 15,
    polyOrder: 3,
    deriv: 1,
  });
  const acceleration = savitzkyGolayFilter(step2, {
    windowLength: 15,
    polyOrder: 3,
    deriv: 2,
  });

  return {
    cleaned: step2,
    trend,
    weekly,
    monthly,
    annual,
    velocity,
    acceleration,
  };
}
