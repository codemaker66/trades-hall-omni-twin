// ---------------------------------------------------------------------------
// @omni-twin/control-core — Optimal Control for Venue Operations
// ---------------------------------------------------------------------------
// Barrel re-export for all 12 OC sub-domains (OC-1 through OC-12).
// ---------------------------------------------------------------------------

// Infrastructure: types, matrix utilities, vector ops, PRNG
export * from './types.js';

// OC-1: LQR for Venue Resources
export * from './lqr/index.js';

// OC-2: Model Predictive Control — Dynamic Pricing
export * from './mpc/index.js';

// OC-3: Pontryagin's Maximum Principle
export * from './pmp/index.js';

// OC-4: Dynamic Programming / HJB
export * from './dp/index.js';

// OC-5: Reinforcement Learning as Approximate Control
export * from './rl/index.js';

// OC-6: Feedback Linearization & Nonlinear Control
export * from './nonlinear/index.js';

// OC-7: Multi-Agent Venue Networks
export * from './multi-agent/index.js';

// OC-8: Crowd Flow & Evacuation Control
export * from './crowd/index.js';

// OC-9: Stochastic Optimal Control
export * from './stochastic/index.js';

// OC-10: Optimal Experiment Design
export * from './experiment/index.js';

// OC-11: Real-Time Control Architecture
export * from './architecture/index.js';

// OC-12: Edge/Browser Deployment
export * from './edge/index.js';
