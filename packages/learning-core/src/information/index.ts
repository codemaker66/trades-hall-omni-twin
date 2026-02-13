// ---------------------------------------------------------------------------
// SLT-10: Information Theory — barrel export
// ---------------------------------------------------------------------------

export { entropy, jointEntropy, mutualInformation, mrmrSelect } from './mutual-information.js';
export { klDivergence, reverseKL, jsDivergence, jsdMetric, histogramFromSamples } from './divergence.js';
export { mdlScore, mdlSelect, normalizedMaximumLikelihood } from './mdl.js';
export { fisherInformationMatrix, dOptimality, aOptimality, federovExchange } from './fisher-oed.js';
