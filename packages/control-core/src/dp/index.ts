// ---------------------------------------------------------------------------
// OC-4  Dynamic Programming -- Barrel Re-export
// ---------------------------------------------------------------------------

export { valueIteration, policyIteration, backwardInduction } from './bellman.js';
export { solveHJB, queryHJBValue } from './hjb.js';
export { fittedValueIteration, evaluateApproxValue } from './approximate-dp.js';
export { solveBidPriceLP, bidPriceAcceptReject, simplexLP } from './bid-price.js';
export { computeStoppingThresholds, shouldAccept } from './optimal-stopping.js';
