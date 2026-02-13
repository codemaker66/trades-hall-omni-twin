// ---------------------------------------------------------------------------
// OC-3: Pontryagin's Maximum Principle — Barrel Re-exports
// ---------------------------------------------------------------------------

export { evaluateHamiltonian, integrateODE, rk4Step } from './hamiltonian.js';
export { singleShooting } from './single-shooting.js';
export { multipleShooting } from './multiple-shooting.js';
export { directCollocation, trapezoidalCollocation } from './collocation.js';
export { analyzeSwitchingFunction, constructBangBang } from './bang-bang.js';
