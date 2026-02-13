// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — GNN-6: Temporal Graph Networks — Barrel Re-exports
// ---------------------------------------------------------------------------

export { tgnUpdate, tgnEmbed, createTemporalGraph } from './tgn.js';
export { tgatLayer } from './tgat.js';
export type { TemporalNeighbor } from './tgat.js';
export {
  bochnerTimeEncoding,
  positionEncoding,
  relativeTimeEncoding,
} from './time-encoding.js';
