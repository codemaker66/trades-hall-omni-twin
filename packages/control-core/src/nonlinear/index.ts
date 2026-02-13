// ---------------------------------------------------------------------------
// OC-6  Nonlinear Control -- Barrel Re-export
// ---------------------------------------------------------------------------

export { lieDerivative, computeRelativeDegree, feedbackLinearize } from './feedback-linearization.js';
export { backsteppingControl, backsteppingLyapunov } from './backstepping.js';
export { slidingModeControl, evaluateSurface, checkReachingCondition } from './sliding-mode.js';
export { mracStep, l1AdaptiveStep, createAdaptiveState } from './adaptive.js';
