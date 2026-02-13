// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — Type Definitions
// Graph Neural Networks for Venue Intelligence
// ---------------------------------------------------------------------------

// ---- PRNG (deterministic seeded random) ----

export type PRNG = () => number;

/** Mulberry32 PRNG — deterministic, seedable, fast. */
export function createPRNG(seed: number): PRNG {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---- Graph Primitives (CSR Sparse Format) ----

/** Compressed Sparse Row graph representation. */
export interface Graph {
  /** Number of nodes. */
  readonly numNodes: number;
  /** Number of edges. */
  readonly numEdges: number;
  /** Row pointer array (length numNodes+1). rowPtr[i]..rowPtr[i+1] are edges from node i. */
  readonly rowPtr: Uint32Array;
  /** Column indices for each edge (length numEdges). */
  readonly colIdx: Uint32Array;
  /** Optional edge weights (length numEdges). */
  readonly edgeWeights?: Float64Array;
  /** Node feature matrix, row-major: node i's features at [i*featureDim .. (i+1)*featureDim). */
  readonly nodeFeatures: Float64Array;
  /** Dimension of each node's feature vector. */
  readonly featureDim: number;
  /** Optional edge feature matrix, row-major (numEdges * edgeFeatureDim). */
  readonly edgeFeatures?: Float64Array;
  /** Dimension of each edge's feature vector (0 if no edge features). */
  readonly edgeFeatureDim?: number;
}

/** Node type info for heterogeneous graphs. */
export interface HeteroNodeStore {
  readonly features: Float64Array;
  readonly count: number;
  readonly featureDim: number;
}

/** Edge type info for heterogeneous graphs (CSR per edge type). */
export interface HeteroEdgeStore {
  readonly rowPtr: Uint32Array;
  readonly colIdx: Uint32Array;
  readonly numEdges: number;
  readonly features?: Float64Array;
  readonly featureDim?: number;
}

/** Heterogeneous graph with typed nodes and edges. */
export interface HeteroGraph {
  readonly nodeTypes: string[];
  /** Edge types as [srcType, relation, dstType] triplets. */
  readonly edgeTypes: readonly [string, string, string][];
  readonly nodes: Map<string, HeteroNodeStore>;
  /** Key is "srcType/relation/dstType". */
  readonly edges: Map<string, HeteroEdgeStore>;
}

/** Batched graphs for graph-level tasks. */
export interface GraphBatch {
  /** Combined graph (all nodes/edges merged). */
  readonly graph: Graph;
  /** Maps each node to its graph index in the batch. */
  readonly batchIndex: Uint32Array;
  readonly numGraphs: number;
  /** Offsets for node indices per graph. */
  readonly nodeOffsets: Uint32Array;
}

/** A single temporal edge (interaction event). */
export interface TemporalEdge {
  readonly src: number;
  readonly dst: number;
  readonly timestamp: number;
  readonly features: Float64Array;
}

/** Temporal graph with streaming edges and per-node memory. */
export interface TemporalGraph {
  readonly numNodes: number;
  readonly temporalEdges: TemporalEdge[];
  /** Per-node memory state (numNodes * memoryDim), row-major. */
  nodeMemory: Float64Array;
  readonly memoryDim: number;
  /** Timestamp of last interaction per node. */
  lastUpdate: Float64Array;
}

// ---- GNN Layer Configs (GNN-1) ----

export type ActivationFn = 'relu' | 'elu' | 'leaky_relu' | 'tanh' | 'sigmoid' | 'none';

export interface GCNConfig {
  readonly inDim: number;
  readonly outDim: number;
  readonly bias: boolean;
  readonly activation: ActivationFn;
}

export interface SAGEConfig {
  readonly inDim: number;
  readonly outDim: number;
  readonly aggregator: 'mean' | 'max' | 'gcn';
  readonly normalize: boolean;
  readonly activation: ActivationFn;
}

export interface GATConfig {
  readonly inDim: number;
  readonly outDim: number;
  readonly heads: number;
  readonly dropout: number;
  readonly negativeSlope: number;
  readonly concat: boolean;
  /** Use GATv2 (Brody et al. 2022) dynamic attention. */
  readonly v2: boolean;
}

export interface GINConfig {
  readonly inDim: number;
  readonly hiddenDim: number;
  readonly outDim: number;
  readonly epsilon: number;
  readonly trainEpsilon: boolean;
}

// ---- GNN Layer Weights ----

export interface GCNWeights {
  W: Float64Array;  // inDim × outDim
  bias?: Float64Array;  // outDim
}

export interface SAGEWeights {
  W_self: Float64Array;  // inDim × outDim
  W_neigh: Float64Array; // inDim × outDim
  bias?: Float64Array;   // outDim
}

export interface GATWeights {
  W: Float64Array;       // inDim × (outDim * heads)
  a_src: Float64Array;   // outDim per head → heads * outDim
  a_dst: Float64Array;   // outDim per head → heads * outDim
  // GATv2 uses W_src, W_dst, a instead
  W_src?: Float64Array;
  W_dst?: Float64Array;
  a?: Float64Array;      // outDim (shared across concat)
}

export interface GINWeights {
  mlp: MLPWeights;
  epsilon: number;
}

export interface MLPWeights {
  layers: { W: Float64Array; bias: Float64Array; inDim: number; outDim: number }[];
}

// ---- Heterogeneous GNN Configs (GNN-2) ----

export interface RGCNConfig {
  readonly inDim: number;
  readonly outDim: number;
  readonly numRelations: number;
  readonly numBases: number;
  readonly bias: boolean;
}

export interface RGCNWeights {
  bases: Float64Array[];    // numBases arrays, each inDim × outDim
  coeffs: Float64Array;     // numRelations × numBases
  bias?: Float64Array;      // outDim
}

export interface HANConfig {
  readonly inDim: number;
  readonly outDim: number;
  readonly heads: number;
  readonly metaPaths: number[][];  // Each meta-path is sequence of edge type indices
}

export interface HANWeights {
  nodeAttn: GATWeights;
  semanticAttnVec: Float64Array;   // outDim (or outDim*heads)
  W_semantic: Float64Array;        // outDim × outDim
}

export interface HGTConfig {
  readonly inDim: number;
  readonly outDim: number;
  readonly heads: number;
  readonly numNodeTypes: number;
  readonly numEdgeTypes: number;
}

export interface HGTWeights {
  W_Q: Float64Array[];   // per node type
  W_K: Float64Array[];   // per node type
  W_V: Float64Array[];   // per node type
  W_ATT: Float64Array[]; // per edge type
  W_MSG: Float64Array[]; // per edge type
  mu: Float64Array;      // per edge type (prior)
}

export interface SimpleHGNWeights {
  W: Float64Array;
  a: Float64Array;
  edgeTypeEmb: Float64Array[];  // per edge type
  residualW?: Float64Array;
}

// ---- Recommendation Types (GNN-3) ----

export interface LightGCNConfig {
  readonly numUsers: number;
  readonly numItems: number;
  readonly embeddingDim: number;
  readonly numLayers: number;
  readonly learningRate: number;
  readonly l2Reg: number;
  readonly epochs: number;
}

export interface LightGCNResult {
  readonly userEmbeddings: Float64Array;
  readonly itemEmbeddings: Float64Array;
  readonly losses: number[];
}

export interface PinSageConfig {
  readonly walkLength: number;
  readonly numWalks: number;
  readonly topK: number;
}

export interface SessionGraphData {
  readonly graph: Graph;
  readonly nodeMapping: Map<number, number>;  // original → local ID
  readonly reverseMapping: number[];          // local → original ID
  readonly sessionItems: number[];
}

export interface RecommendationResult {
  readonly scores: Float64Array;
  readonly topK: number[];
  readonly embeddings?: Float64Array;
}

// ---- Spatial Layout Types (GNN-4) ----

export interface LayoutItem {
  readonly id: number;
  readonly type: number;        // one-hot index (chair=0, table=1, ...)
  readonly numTypes: number;
  readonly width: number;
  readonly depth: number;
  readonly height: number;
  readonly x: number;
  readonly y: number;
  readonly rotation: number;    // radians
}

export interface LayoutGraphConfig {
  readonly distanceThreshold: number;
  readonly functionalEdges: boolean;
  readonly wallEdges: boolean;
}

export interface LayoutQualityResult {
  readonly score: number;
  readonly nodeAttentionWeights?: Float64Array;
}

export interface GraphMatchResult {
  readonly similarity: number;
  readonly crossAttention1: Float64Array;
  readonly crossAttention2: Float64Array;
}

// ---- Graph Generation Types (GNN-5) ----

export interface GraphRNNConfig {
  readonly maxNodes: number;
  readonly hiddenDim: number;
  readonly edgeHorizon: number;  // BFS-ordered edge window
}

export interface GraphRNNWeights {
  nodeGRU: GRUWeights;
  edgeGRU: GRUWeights;
  nodeOutputW: Float64Array;
  nodeOutputBias: Float64Array;
  edgeOutputW: Float64Array;
  edgeOutputBias: Float64Array;
}

export interface GRUWeights {
  W_z: Float64Array;   // update gate
  U_z: Float64Array;
  b_z: Float64Array;
  W_r: Float64Array;   // reset gate
  U_r: Float64Array;
  b_r: Float64Array;
  W_h: Float64Array;   // candidate
  U_h: Float64Array;
  b_h: Float64Array;
}

export interface GRANConfig {
  readonly maxNodes: number;
  readonly blockSize: number;
  readonly hiddenDim: number;
  readonly numMixtures: number;
}

export interface DiGressConfig {
  readonly numNodeTypes: number;
  readonly numEdgeTypes: number;
  readonly diffusionSteps: number;
  readonly hiddenDim: number;
}

export interface DiGressState {
  readonly nodeTypes: Uint8Array;    // categorical node types
  readonly adjMatrix: Uint8Array;    // flattened upper-triangle edge types
  readonly numNodes: number;
}

export interface GeneratedGraph {
  readonly adjacency: Uint8Array;  // n×n flattened
  readonly numNodes: number;
  readonly nodeTypes?: Uint8Array;
}

export interface ForceDirectedConfig {
  readonly iterations: number;
  readonly learningRate: number;
  readonly attractionStrength: number;
  readonly repulsionStrength: number;
  readonly idealEdgeLength: number;
}

export interface SurrogateEnergyModel {
  readonly gatWeights: GATWeights[];
  readonly poolingW: Float64Array;
  readonly headW: Float64Array;
  readonly headBias: Float64Array;
  readonly config: GATConfig;
}

// ---- Temporal Types (GNN-6) ----

export interface TGNConfig {
  readonly numNodes: number;
  readonly memoryDim: number;
  readonly timeDim: number;
  readonly messageDim: number;
  readonly aggregator: 'last' | 'mean';
}

export interface TGNWeights {
  msgW: Float64Array;
  gru: GRUWeights;
}

export interface TGATConfig {
  readonly inDim: number;
  readonly outDim: number;
  readonly heads: number;
  readonly timeDim: number;
}

export interface TimeEncodingConfig {
  readonly dim: number;
  readonly frequencies: Float64Array;
}

// ---- Graph Transformer Types (GNN-7) ----

export interface GPSConfig {
  readonly dim: number;
  readonly heads: number;
  readonly ffnDim: number;
  readonly dropout: number;
  readonly mpnnType: 'gcn' | 'gat' | 'gin';
}

export interface GPSWeights {
  mpnnWeights: GCNWeights | GATWeights | GINWeights;
  attnW_Q: Float64Array;
  attnW_K: Float64Array;
  attnW_V: Float64Array;
  attnW_O: Float64Array;
  ffnW1: Float64Array;
  ffnB1: Float64Array;
  ffnW2: Float64Array;
  ffnB2: Float64Array;
  norm1Gamma: Float64Array;
  norm1Beta: Float64Array;
  norm2Gamma: Float64Array;
  norm2Beta: Float64Array;
}

export interface PositionalEncodingResult {
  readonly pe: Float64Array;  // numNodes × peDim, row-major
  readonly peDim: number;
}

export interface ExphormerConfig {
  readonly dim: number;
  readonly heads: number;
  readonly numVirtualNodes: number;
  readonly expanderDegree: number;
}

// ---- Combinatorial Optimization Types (GNN-8) ----

export interface AttentionModelConfig {
  readonly dim: number;
  readonly heads: number;
  readonly numLayers: number;
  readonly clipC: number;
}

export interface AttentionModelWeights {
  encoderLayers: {
    W_Q: Float64Array; W_K: Float64Array; W_V: Float64Array; W_O: Float64Array;
    ffnW1: Float64Array; ffnB1: Float64Array; ffnW2: Float64Array; ffnB2: Float64Array;
  }[];
  decoderW_Q: Float64Array;
  decoderW_K: Float64Array;
}

export interface AssignmentResult {
  readonly assignment: Uint32Array;  // event[i] → room assignment[i]
  readonly cost: number;
  readonly feasible: boolean;
}

export interface SinkhornConfig {
  readonly iterations: number;
  readonly temperature: number;
  readonly epsilon: number;
}

export interface MIPVariable {
  readonly cost: number;
  readonly lb: number;
  readonly ub: number;
  readonly isInteger: boolean;
}

export interface MIPConstraint {
  readonly coeffs: Float64Array;
  readonly varIndices: Uint32Array;
  readonly rhs: number;
  readonly sense: 'le' | 'ge' | 'eq';
}

export interface MIPEncoding {
  readonly graph: Graph;
  readonly numVariables: number;
  readonly numConstraints: number;
}

// ---- Scalable Inference Types (GNN-9) ----

export interface NeighborSampleResult {
  readonly subgraph: Graph;
  readonly targetNodes: Uint32Array;
  readonly originalIds: Uint32Array;  // subgraph node → original node
}

export interface MiniBatch {
  readonly subgraph: Graph;
  readonly targetMask: Uint8Array;
  readonly originalIds: Uint32Array;
}

export interface ClusterPartition {
  readonly assignment: Uint32Array;  // node → cluster
  readonly numClusters: number;
  readonly clusterSizes: Uint32Array;
}

export interface GLNNConfig {
  readonly hiddenDims: number[];
  readonly lambda: number;       // KL vs CE weight
  readonly temperature: number;  // distillation temperature
  readonly epochs: number;
  readonly learningRate: number;
}

export interface GLNNResult {
  readonly mlpWeights: MLPWeights;
  readonly losses: number[];
  readonly accuracy: number;
}

export interface IVFIndex {
  readonly centroids: Float64Array;   // nClusters × dim, row-major
  readonly assignments: Uint32Array;  // each embedding → cluster
  readonly dim: number;
  readonly nClusters: number;
  readonly embeddings: Float64Array;  // all embeddings, row-major
  readonly numEmbeddings: number;
}

export interface ANNSearchResult {
  readonly indices: Uint32Array;
  readonly distances: Float64Array;
}

// ---- Explainability Types (GNN-10) ----

export interface ExplainerConfig {
  readonly epochs: number;
  readonly lr: number;
  readonly sizeReg: number;
  readonly entropyReg: number;
  readonly edgeMaskThreshold: number;
  readonly featureMaskThreshold: number;
}

export interface ExplanationResult {
  readonly edgeMask: Float64Array;
  readonly featureMask?: Float64Array;
  readonly importantEdges: [number, number][];
  readonly importantFeatures?: number[];
}

export interface CounterfactualResult {
  readonly removedEdges: [number, number][];
  readonly originalPrediction: number;
  readonly newPrediction: number;
  readonly numEditsRequired: number;
}

export interface ExplanationTemplate {
  readonly edgeType: string;
  readonly sourceNodeType: string;
  readonly targetNodeType: string;
  readonly template: string;
}

export interface NaturalLanguageExplanation {
  readonly explanations: string[];
  readonly importanceScores: number[];
}

// ---- Integration Types (GNN-11) ----

export interface WassersteinReadoutConfig {
  readonly numPrototypes: number;
  readonly prototypeDim: number;
  readonly sinkhornIterations: number;
  readonly epsilon: number;
}

export interface WassersteinReadoutResult {
  readonly distances: Float64Array;  // numPrototypes distances
  readonly transportPlans: Float64Array[];
}

export interface PersistenceDiagram {
  readonly births: Float64Array;
  readonly deaths: Float64Array;
  readonly dim: number;  // homology dimension (0=components, 1=cycles)
}

export interface TOGLConfig {
  readonly maxDim: number;
  readonly numFiltrationSteps: number;
  readonly imageResolution: number;
}

export interface SheafConfig {
  readonly stalkDim: number;
  readonly diffusionSteps: number;
  readonly learningRate: number;
}

export interface SheafWeights {
  restrictionMLP: MLPWeights;     // maps edge features → restriction map
  stalkDim: number;
}

export interface DemandForecast {
  readonly mean: Float64Array;
  readonly variance: Float64Array;
  readonly timestamps: Float64Array;
}

export interface PricingResult {
  readonly optimalPrice: number;
  readonly expectedRevenue: number;
  readonly demandAtPrice: number;
}

// ---- Production Types (GNN-12) ----

export interface GraphStoreConfig {
  readonly maxNodes: number;
  readonly featureDim: number;
}

export interface GraphStoreNode {
  id: string;
  type: string;
  features: Float64Array;
}

export interface GraphStoreEdge {
  src: string;
  dst: string;
  type: string;
  weight: number;
}

export interface CacheConfig {
  readonly maxSize: number;
  readonly ttlMs: number;
  readonly embeddingDim: number;
}

export interface CacheEntry<T> {
  value: T;
  timestamp: number;
  accessCount: number;
}


// ---- GNN Forward Function Type ----

/** Generic GNN forward pass signature for use with explainability, distillation, etc. */
export type GNNForwardFn = (graph: Graph, features: Float64Array) => Float64Array;
