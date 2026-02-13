// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — GNN-12 Production Architecture: Barrel Export
// ---------------------------------------------------------------------------

export { GraphStore } from './graph-store.js';

export { EmbeddingCache } from './cache-manager.js';

export {
  EventProcessor,
  type EventUpdate,
  type ProcessResult,
} from './event-processor.js';

export {
  ServingPipeline,
  type ServingPipelineConfig,
  type ServingRequest,
  type ServingResponse,
  type HealthCheckResult,
} from './serving-pipeline.js';
