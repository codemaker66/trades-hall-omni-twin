// ---------------------------------------------------------------------------
// OC-5: Reinforcement Learning -- Barrel Re-export
// ---------------------------------------------------------------------------

export {
  mlpForward,
  createSACAgent,
  sacSelectAction,
  createReplayBuffer,
  pushReplayBuffer,
  sampleReplayBuffer,
} from './sac.js';

export {
  createPPOAgent,
  ppoSelectAction,
  computeGAE,
} from './ppo.js';

export {
  cqlLoss,
  iqlExpectileLoss,
  dtForwardPass,
} from './offline-rl.js';

export {
  safeProject,
  updateLagrangeMultipliers,
} from './safe-rl.js';

export {
  computeVenueReward,
  potentialShaping,
} from './reward-shaping.js';

export {
  fitLinearDynamics,
  mpcSafetyShield,
} from './model-based.js';
