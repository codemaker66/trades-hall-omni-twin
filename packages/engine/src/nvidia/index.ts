// NVIDIA integrations: Cosmos, Omniverse, ACE

export type {
  JobStatus,
  JobResult,
  SceneDescription,
  CosmosRequest,
  CosmosResult,
  OmniverseSessionConfig,
  OmniverseSession,
  OmniverseSceneItem,
  OmniverseSceneState,
  ConciergeContext,
  ConciergeMessage,
  ConciergeSession,
  ConciergeResponse,
} from './types'

// Cosmos (T9)
export {
  serializeScene,
  countByType,
  estimateCapacity,
  formatFurnitureSummary,
  detectLayoutStyle,
  type CosmosClient,
  MockCosmosClient,
  createCosmosClient,
} from './cosmos'

// Omniverse (T10)
export {
  toOmniverseScene,
  toOmniverseItem,
  computeSceneDiff,
  type SceneDiffOp,
  type OmniverseStreamingClient,
  MockOmniverseClient,
  createOmniverseClient,
} from './omniverse'

// ACE (T11)
export {
  buildConciergeContext,
  buildSystemPrompt,
  type ConciergeChatClient,
  MockConciergeClient,
  createConciergeClient,
} from './ace'
