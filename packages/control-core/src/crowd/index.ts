// ---------------------------------------------------------------------------
// OC-8: Crowd Flow & Evacuation -- Barrel Re-exports
// ---------------------------------------------------------------------------

export {
  createHughesState,
  solveEikonal,
  hughesStep,
  hughesEgressTime,
} from './hughes.js';

export {
  socialForceStep,
  socialRepulsion,
  wallRepulsion,
} from './social-force.js';

export {
  solveEvacuationMPC,
  greedyEvacuation,
} from './evacuation-mpc.js';

export {
  checkDensityConstraints,
  projectDensity,
} from './density-constraints.js';
