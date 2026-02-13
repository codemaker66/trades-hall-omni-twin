// ---------------------------------------------------------------------------
// OC-12: Edge Deployment — Barrel Re-exports
// ---------------------------------------------------------------------------

export { linearFeedback, loadFeedbackGain } from './linear-feedback.js';
export { explicitMPCQuery, isInsidePolytope } from './explicit-mpc-lookup.js';
export {
  mlpLayer,
  neuralPolicyForward,
  loadNeuralPolicy,
} from './neural-policy.js';
export {
  checkConstraints,
  clipAction,
  checkRateConstraint,
} from './constraint-checker.js';
