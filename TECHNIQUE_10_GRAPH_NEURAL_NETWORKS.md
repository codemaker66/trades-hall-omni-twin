# TECHNIQUE_10_GRAPH_NEURAL_NETWORKS.md — Graph Neural Networks for Venue Intelligence

<!-- COMMAND_TRACK_LINKS_START -->
## Canonical Command Track
Canonical command track: `docs/commands/gnn/`
Execution authority for agent command specs is `docs/commands/**`.
This document remains a research/reference source.

## ID Mapping Notes
GNN-N maps to GNN-N.
Integration checkpoints use IDs `GNN-INT-1`, `GNN-INT-2`, and `GNN-INT-3`.
<!-- COMMAND_TRACK_LINKS_END -->


> ## CRITICAL: NEVER OMIT, SIMPLIFY, OR WEAKEN
>
> Claude Code must NEVER:
> - Omit features, techniques, or subsystems to fit within a single session
> - Simplify implementations to make them "easier" or "more practical"
> - Replace a specified technique with a simpler alternative without explicit approval
> - Skip sections because they seem redundant or low-priority
> - Truncate implementations to keep diffs small — instead, split across commits
>
> If a task is too large for one session: split it into sub-tasks across multiple
> sessions. Document what remains in PROGRESS.md and pick it up next session.
> The project must contain the FULL depth of every technique specified.
> "Good enough" is not good enough. Every feature exists at maximum fidelity.

> **Purpose**: Feed this to Claude Code after the main phases and earlier techniques
> are complete. Implements graph-structured learning for venue-event matching,
> recommendation, spatial layout understanding, temporal booking patterns,
> combinatorial scheduling, and explainable venue suggestions.
>
> **How to use**: Tell Claude Code: "Read TECHNIQUE_10_GRAPH_NEURAL_NETWORKS.md
> and implement incrementally, starting from GNN-1."
>
> **This is 12 sub-domains. Implement ALL of them. Do not skip any.**

---

## Key Papers (Referenced Throughout)

- Gilmer et al. (2017). MPNN framework. arXiv:1704.01212
- Kipf & Welling (2017). GCN. arXiv:1609.02907
- Hamilton et al. (2017). GraphSAGE. arXiv:1706.02216
- Veličković et al. (2018). GAT. arXiv:1710.10903
- Brody, Alon & Yahav (2022). GATv2. ICLR
- Xu et al. (2019). GIN. arXiv:1810.00826
- Schlichtkrull et al. (2018). R-GCN. arXiv:1703.06103
- Wang et al. (2019). HAN. WWW. arXiv:1903.07293
- Hu et al. (2020). HGT. WWW. arXiv:2003.01332
- He et al. (2020). LightGCN. SIGIR. arXiv:2002.02126
- Ying et al. (2018). PinSage. KDD. arXiv:1806.01973
- Wu et al. (2019). SR-GNN. AAAI. arXiv:1811.00855
- Rossi et al. (2020). TGN. arXiv:2006.10637
- Rampášek et al. (2022). GPS Graph Transformer. NeurIPS. arXiv:2205.12454
- Vignac et al. (2023). DiGress. ICLR. arXiv:2209.14734
- Kool et al. (2019). Attention Model for CO. ICLR. arXiv:1803.08475
- Zhang et al. (2022). GLNN distillation. ICLR
- Ying et al. (2019). GNNExplainer. NeurIPS. arXiv:1903.03894
- Luo et al. (2020). PGExplainer. NeurIPS
- Béligneul et al. OT-GNN. arXiv:2006.04804
- Horn et al. (2022). TOGL. ICLR
- Bodnar et al. (2022). Neural Sheaf Diffusion. NeurIPS

---

## Architecture Overview

```
apps/
  ml-api/
    src/
      gnn/
        core/
          mpnn.py                     — Base MPNN framework
          gcn.py                      — GCN (spectral, transductive)
          sage.py                     — GraphSAGE (inductive, sampling)
          gat.py                      — GAT/GATv2 (attention, interpretable)
          gin.py                      — GIN (WL-expressive, graph-level)
        heterogeneous/
          rgcn.py                     — R-GCN with basis decomposition
          han.py                      — HAN (hierarchical meta-path attention)
          hgt.py                      — HGT (type-decomposed attention)
          simple_hgn.py               — Simple-HGN (GAT + type embeddings)
          venue_graph_builder.py      — Construct heterogeneous venue graph
        recommendation/
          lightgcn.py                 — LightGCN collaborative filtering
          pinsage.py                  — PinSage-style random walk sampling
          sr_gnn.py                   — Session-based recommendation
          kgat.py                     — Knowledge-graph-enhanced
          cold_start.py               — Content-based + meta-learning
        spatial/
          layout_gnn.py               — Furniture layout as graph
          layout_quality.py           — GNN layout scoring
          layout_gmn.py               — Graph Matching for layout comparison
          scene_graph.py              — SceneGraphNet for arrangement
        generation/
          graph_rnn.py                — Sequential graph generation
          gran.py                     — Block-wise generation (5K nodes)
          digress.py                  — Discrete diffusion on graphs
          graph_to_layout.py          — Graph topology → spatial coordinates
          surrogate_energy.py         — GNN as differentiable layout energy
        temporal/
          tgn.py                      — Temporal Graph Networks
          tgat.py                     — Temporal Graph Attention
          time_encoding.py            — Bochner's theorem time features
        transformers/
          gps.py                      — GPS framework (MPNN + global attn)
          positional_encoding.py      — Laplacian PE, Random Walk PE
          exphormer.py                — O(N+E) sparse attention
        optimization/
          attention_model.py          — Transformer for CO (warm-start MIP)
          event_room_assignment.py    — Bipartite GNN + Sinkhorn + Hungarian
          mip_gnn.py                  — GNN-guided branch-and-bound
        inference/
          neighbor_loader.py          — GraphSAGE mini-batch sampling
          cluster_gcn.py              — METIS partition training
          glnn_distill.py             — GNN → MLP knowledge distillation
          faiss_index.py              — FAISS IVF+PQ for ANN embedding search
          onnx_export.py              — MLP → ONNX → ONNX Runtime Web
        explainability/
          gnn_explainer.py            — GNNExplainer (subgraph + features)
          pg_explainer.py             — PGExplainer (parameterized, 108× faster)
          counterfactual.py           — CF-GNNExplainer (minimal edge edits)
          template_mapper.py          — Explanation subgraph → natural language
        integration/
          ot_gnn.py                   — OT-GNN Wasserstein graph readout
          togl.py                     — TOGL persistent homology layer
          sheaf.py                    — Neural Sheaf Diffusion
          demand_gnn.py               — GNN → demand prediction → pricing
      routes/
        gnn.py                        — FastAPI endpoints

packages/
  gnn-core/                           — TypeScript types + browser inference
    src/
      types.ts                        — VenueGraph, Recommendation types
      inference/
        mlp_inference.ts              — GLNN-distilled MLP (ONNX Runtime Web)
        embedding_search.ts           — FAISS-style ANN in browser
      visualization/
        GraphExplorer.tsx             — Interactive venue graph visualization
        RecommendationCard.tsx        — Venue recommendation with explanation
        AttentionHeatmap.tsx          — GAT attention weight visualization
        LayoutGraphView.tsx           — Furniture layout as graph overlay
```

### Python Dependencies

```
torch>=2.4                      # PyTorch
torch-geometric>=2.7.0          # PyG (all GNN architectures)
torch-scatter>=2.1               # Efficient scatter operations
torch-sparse>=0.6                # Sparse matrix operations
faiss-gpu>=1.9                   # ANN embedding search
gurobipy>=11.0                   # MIP solver (warm-start from GNN)
onnx>=1.17                       # ONNX export
onnxruntime>=1.20                # ONNX inference
neo4j>=5.0                       # Graph database for venue graph storage
```

---

## GNN-1: MPNN Foundation and Architecture Selection

### What to Build

The core GNN framework instantiating GCN, GraphSAGE, GAT/GATv2, and GIN,
with architecture selection logic for different venue tasks.

```python
# apps/ml-api/src/gnn/core/gat.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class VenueEventMatcher(torch.nn.Module):
    """
    GATv2 (Brody, Alon & Yahav 2022, ICLR) for venue-event matching.

    GATv2 fixes GAT's static attention problem:
    Original GAT: α_{ij} = softmax(LeakyReLU(a^T[Wh_i ∥ Wh_j]))
      Decomposes to a_1^T·Wh_i + a_2^T·Wh_j → ranking is query-independent!
    GATv2: α_{ij} = softmax(a^T LeakyReLU(W[h_i ∥ h_j]))
      Non-decomposable → ranking genuinely depends on query.

    Attention weights reveal WHY a venue was matched:
    One head focuses on capacity, another on location, another on amenities.

    Architecture selection for venue tasks:
    - GCN: static graph classification (~81-85% accuracy, hard to beat when tuned)
    - GraphSAGE: inductive (new venues arrive), AUC >0.99 on link prediction
    - GAT/GATv2: interpretability needed (attention = explanation)
    - GIN: graph-level tasks (layout quality, venue cluster similarity)

    Hyperparameters for 1K-100K node venue graphs:
    - 2-3 layers (deeper → over-smoothing)
    - Hidden: 64-256, LR: 0.001, Dropout: 0.6 (GAT)
    - GAT heads: 8 intermediate, 1 output
    - Early stopping over 100-200 epochs
    """

    def __init__(self, in_channels, hidden=64, heads=8):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden, heads=heads, dropout=0.6)
        self.conv2 = GATv2Conv(hidden * heads, hidden, heads=1,
                               concat=False, dropout=0.6)

    def forward(self, x, edge_index, return_attention=False):
        x = F.dropout(x, p=0.6, training=self.training)
        x, attn1 = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x, attn2 = self.conv2(x, edge_index, return_attention_weights=True)
        if return_attention:
            return x, attn1, attn2
        return x
```

### Over-Smoothing Solutions

```python
"""
2-3 GNN layers suffice for venue-scale graphs.
Deeper → node representations converge to indistinguishable states.

Three proven solutions:
1. Residual connections: h^{l+1} = h^l + GNN_layer(h^l)
2. JKNet (Jumping Knowledge): h_final = CONCAT(h^0, h^1, ..., h^L)
   Model selects most informative layer per node.
3. DropEdge: randomly remove edges during training (p=0.1-0.3)
   Slows over-smoothing and regularizes.
"""
```

### Venue Graph Construction

```python
"""
Node features (~20-50 dim per venue):
  capacity (float, normalized), lat/lon, price_per_hour, rating,
  amenities (multi-hot: WiFi, parking, AV, catering, outdoor),
  venue_type (one-hot), square_footage

Edge construction strategies:
  - k-NN geographic (k=5-20, haversine distance)
  - Event-type compatibility (Jaccard similarity)
  - Historical co-booking (shared planners frequency)
"""
```

---

## GNN-2: Heterogeneous Venue Ecosystem Graph

### What to Build

Multi-type graph with Venue, Event, Planner, Vendor, Date nodes
and typed edges (hosts, books, prefers, available_on, supplies).

```python
# apps/ml-api/src/gnn/heterogeneous/venue_graph_builder.py

import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

def build_venue_hetero_graph(venues, planners, events, bookings, vendors):
    """
    Heterogeneous graph for the full venue marketplace.

    Node types: Venue, Event, Planner, Vendor, Date
    Edge types: hosts, books, prefers, available_on, supplies

    Key insight (HGB benchmark, Lv et al. KDD 2021):
    Simple-HGN (GAT + learnable edge-type embeddings + residual + L2 norm)
    OUTPERFORMS all prior heterogeneous GNNs on ALL benchmark datasets.

    → Start with Simple-HGN as baseline.
    → Move to HGT only at web scale (100M+ nodes).

    Meta-paths for HAN:
    - VEPEV (Venue-Event-Planner-Event-Venue): shared planner preferences
    - VVeV (Venue-Vendor-Venue): shared supply chain
    - VDE (Venue-Date-Event): temporal co-occurrence
    """
    data = HeteroData()
    data['venue'].x = torch.tensor(venues, dtype=torch.float)
    data['planner'].x = torch.tensor(planners, dtype=torch.float)
    data['event'].x = torch.tensor(events, dtype=torch.float)

    data['planner', 'books', 'venue'].edge_index = ...
    data['event', 'held_at', 'venue'].edge_index = ...
    data['vendor', 'supplies', 'venue'].edge_index = ...

    data = T.ToUndirected()(data)  # Reverse edges for message passing
    return data
```

```python
# apps/ml-api/src/gnn/heterogeneous/hgt.py

from torch_geometric.nn import HGTConv, Linear

class VenueHGT(torch.nn.Module):
    """
    HGT (Hu et al., WWW 2020, arXiv:2003.01332):
    Type-decomposed attention via meta-relation triplets ⟨τ(s), φ(e), τ(t)⟩.

    ATT-head^i(s,e,t) = (K^i(s)·W_ATT^{φ(e)}·Q^i(t)^T)·μ/√d

    9-21% improvement on OAG (179M nodes, 2B edges).
    Advantage diminishes on smaller datasets.

    R-GCN (Schlichtkrull et al. 2018, arXiv:1703.06103):
    Relation-specific weights W_r with basis decomposition:
      W_r = Σ_{b=1}^B a_{rb}·V_b  (B << |R| shared bases)
    Controls parameter explosion from many relation types.

    Fastest prototyping: PyG to_hetero() auto-converts homogeneous GNN:
      model = to_hetero(GNN(64, 32), data.metadata(), aggr='sum')
    """

    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, metadata):
        super().__init__()
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)
        self.convs = torch.nn.ModuleList([
            HGTConv(hidden_channels, hidden_channels, metadata,
                    num_heads=num_heads, group='sum')
            for _ in range(num_layers)
        ])
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {k: self.lin_dict[k](x).relu() for k, x in x_dict.items()}
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
        return x_dict
```

---

## GNN-3: Recommendation System (LightGCN + PinSage)

### What to Build

Venue-planner collaborative filtering as link prediction on bipartite graph.

```python
# apps/ml-api/src/gnn/recommendation/lightgcn.py

from torch_geometric.nn.models import LightGCN

class VenueRecommender:
    """
    LightGCN (He et al., SIGIR 2020, arXiv:2002.02126):
    THE most important recommendation result.

    REMOVES feature transformation and nonlinear activation from GCN.
    Keeps ONLY neighborhood aggregation:
      e_u^{(k+1)} = Σ_{i∈N_u} (1/√|N_u|√|N_i|)·e_i^{(k)}

    Final embedding: weighted mean across K+1 layers:
      e_u = Σ_{k=0}^K (1/(K+1))·e_u^{(k)}

    Only 0th-layer embeddings are trainable.
    BPR loss: L = -Σ ln σ(ŷ_{ui} - ŷ_{uj}) + λ||E^{(0)}||²

    WHY simplification works: in collaborative filtering, nodes have
    only ID embeddings (no rich features), so weight matrices just overfit.

    LightGCN outperforms complex NGCF by +16-21% Recall@20.
    3-layer, dim=64: Recall@20=0.1824, NDCG@20=0.1554 on Gowalla.

    Production serving:
    Pre-compute embeddings → FAISS IVF+PQ index.
    100K venues at 128-dim: top-20 in <1ms.
    Full pipeline (lookup → ANN → rerank): <50ms.
    """

    def __init__(self, num_planners, num_venues, embedding_dim=64, num_layers=3):
        self.model = LightGCN(
            num_nodes=num_planners + num_venues,
            embedding_dim=embedding_dim,
            num_layers=num_layers)

    def train_step(self, edge_index, pos_edges, neg_edges, unique_nodes):
        out = self.model.get_embedding(edge_index)
        pos_rank = self.model.predict_link(edge_index, pos_edges)
        neg_rank = self.model.predict_link(edge_index, neg_edges)
        loss = self.model.recommendation_loss(
            pos_rank, neg_rank, node_id=unique_nodes)
        return loss
```

### Session-Based and Knowledge-Graph Recommendations

```python
"""
SR-GNN (Wu et al., AAAI 2019, arXiv:1811.00855):
  Model planner's search session as directed graph.
  Session [venueA, venueB, venueC, venueA, venueD]:
  directed edges capture click transitions.
  Gated GNN + hybrid attention (global preference + current interest).
  Directly applicable to planner browsing sequences.

KGAT (Wang et al., KDD 2019, arXiv:1905.07854):
  Merge bipartite booking graph with venue knowledge graph.
  Venues connect to amenities, neighborhoods, event types, price ranges.
  Planner who booked premium downtown venue with AV →
  recommendations enriched by ALL connected attributes.

Cold-start strategies for new venues:
1. Content-based: k-NN by features → GraphSAGE aggregation
2. Meta-learning (MetaHIN, KDD 2020): adapt with 1-5 interactions
3. Full GNN embedding: after sufficient bookings
"""
```

---

## GNN-4: Spatial Layout Understanding

### What to Build

Furniture layouts as graphs. Score layout quality, compare layouts,
generate layouts from event requirements.

```python
# apps/ml-api/src/gnn/spatial/layout_quality.py

import torch
from torch_geometric.nn import GATConv, global_mean_pool

class LayoutQualityGNN(torch.nn.Module):
    """
    Furniture layout as graph:
    Node features: x_i = [type_onehot, width, depth, height, pos_x, pos_y, sin(θ), cos(θ)]
    Edges: spatial adjacency (distance threshold), functional (chair→table), wall adjacency
    Edge features: [dx, dy, euclidean_distance, Δθ, visibility]

    SceneGraphNet (Zhou et al., ICCV 2019, arXiv:1907.11308):
    Dense graph with multiple edge types (supporting, next-to, facing, symmetrical).
    Each edge type has own learned message function.
    GRU updates node states with attention-weighted messages.

    LayoutGMN (Patil et al., CVPR 2021):
    Graph Matching Network for structural layout comparison.
    Cross-graph attention: μ_i = Σ_{j∈G'} softmax(h_i^T h_j')·(h_i - h_j')
    Compare two venue layouts structurally.

    GNN as differentiable surrogate for layout energy:
    Train E(layout) = GNN_energy(G) on labeled layouts,
    then optimize positions via gradient descent:
      pos_i -= lr · ∂E/∂pos_i
    NRMSE < 10⁻³, inference <0.01s (wind farm/structural optimization results).
    """

    def __init__(self, node_dim, edge_dim, hidden=128, heads=4, layers=4):
        super().__init__()
        self.node_enc = torch.nn.Linear(node_dim, hidden)
        self.edge_enc = torch.nn.Linear(edge_dim, hidden)
        self.convs = torch.nn.ModuleList([
            GATConv(hidden, hidden // heads, heads=heads, edge_dim=hidden)
            for _ in range(layers)
        ])
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Dropout(0.1), torch.nn.Linear(hidden, 1))

    def forward(self, data):
        x = self.node_enc(data.x)
        ea = self.edge_enc(data.edge_attr)
        for conv in self.convs:
            x = torch.relu(conv(x, data.edge_index, edge_attr=ea)) + x  # Residual
        return self.head(global_mean_pool(x, data.batch)).squeeze(-1)
```

---

## GNN-5: Graph Generation for Layout Synthesis

### What to Build

Generate furniture arrangement graphs from event requirements,
then decode to spatial coordinates.

```python
# apps/ml-api/src/gnn/generation/digress.py

"""
DiGress (Vignac et al., ICLR 2023, arXiv:2209.14734):
  Discrete denoising diffusion on graphs.
  Forward: independently corrupt node/edge types via transition matrices
    X_t = X · Q_t^X
  Graph Transformer predicts clean graph from noisy input.
  3× validity improvement on planar graphs.
  Classifier-free guidance for conditional generation:
    p̃(G_0|G_t, c) ∝ p_θ(G_0|G_t, c)^{1+w} / p_θ(G_0|G_t, ∅)^w

DiffuScene (CVPR 2024):
  Object as node with [location, size, orientation, class, shape_code].
  DDPM on fully-connected scene graphs.
  Scene completion, arrangement, text-conditioned synthesis.

InstructScene (ICLR 2024 Spotlight):
  Two-stage: discrete graph diffusion → semantic graph,
  then layout decoder → continuous attributes.

EchoScene (ECCV 2024, arXiv:2405.00915):
  Dual-branch diffusion: each node has own denoising process,
  information shared via graph convolution at every step.
"""

# apps/ml-api/src/gnn/generation/graph_to_layout.py

"""
Graph topology → spatial coordinates (4 approaches):

1. Joint generation (DiffuScene): topology + positions simultaneously
2. Two-stage (PlanIT): topology first, then constrained optimization
   → MOST PRACTICAL for venues
3. Force-directed: F_{ij} = k(||pos_i - pos_j|| - d_target)·(pos_j - pos_i)/||...||
   pos_i += η·Σ_j F_{ij}
4. GNN coordinate decoder: predict (x, y, θ) per node from graph structure

For venues: generate furniture graph conditioned on event type, capacity, style,
then constrained optimization (or GNN surrogate) for exact positions.
"""
```

---

## GNN-6: Temporal Graph Networks for Dynamic Booking

### What to Build

Model the evolving venue-event interaction stream with per-node memory.

```python
# apps/ml-api/src/gnn/temporal/tgn.py

from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator

class VenueTGN:
    """
    TGN (Rossi et al., 2020, arXiv:2006.10637):
    Continuous-time dynamic graph learning with 4 modules:

    1. Message function: msg_i(t) = msg(s_i(t⁻), s_j(t⁻), Δt, e(t))
    2. Message aggregator: most-recent or mean
    3. Memory updater: GRU — s_i(t) = GRU(m̄_i(t), s_i(t⁻))
    4. Embedding module: identity (fastest) or temporal graph attention

    AP: 98.7% Wikipedia, 98.5% Reddit.

    TGAT (Xu et al., 2020, arXiv:2002.07962):
    Time encoding via Bochner's theorem:
      Φ(t) = √(1/d)·[cos(ω₁t), sin(ω₁t), ..., cos(ω_d t), sin(ω_d t)]
    Learnable frequencies ω map timestamps to d-dim features.

    Venue temporal edges:
    - Booking events (planner→venue at time t, {price, event_type, party_size})
    - Availability changes, price updates, cancellations, reviews
    - Memory captures seasonal demand patterns via learned time encoding
    - Anomaly detection: flag low-probability events (unusual cancellation spikes)
    """

    def __init__(self, num_nodes, raw_msg_dim, memory_dim=100, time_dim=100):
        self.memory = TGNMemory(
            num_nodes, raw_msg_dim, memory_dim, time_dim,
            message_module=IdentityMessage(raw_msg_dim, memory_dim, time_dim),
            aggregator_module=LastAggregator())
```

---

## GNN-7: Graph Transformers (GPS Framework)

```python
# apps/ml-api/src/gnn/transformers/gps.py

"""
GPS (Rampášek et al., NeurIPS 2022, arXiv:2205.12454):
  h_i^{(l+1)} = MLP(h_i^{(l)} + MPNN^{(l)}(h_i, G) + GlobalAttn^{(l)}(h_i, H))

Positional encodings are CRITICAL (without them, self-attention
is permutation-invariant and ignores graph structure):
  - Laplacian Eigenvector PE: k smallest eigenvectors of graph Laplacian
    (SignNet handles sign ambiguity)
  - Random Walk PE: landing probs [RW¹(v,v), ..., RW^K(v,v)]
    Sign-invariant by construction.

Scaling:
  - Vanilla: O(N²) → ~500 nodes max
  - Exphormer (Shirzad et al., ICML 2023, arXiv:2303.06147):
    O(N+E) via local + expander graph + virtual global nodes → 160K+ nodes
  - NodeFormer (Wu et al., NeurIPS 2022):
    Kernelized Gumbel-Softmax O(N) → 2M nodes
  - SGFormer (Wu et al., NeurIPS 2023):
    Single-layer global attention matches multi-layer deep GTs → billion-level

CRITICAL NUANCE (Tönshoff et al. 2023):
  After proper tuning, basic MPGNNs CLOSE much of the gap.
  Well-tuned GCN achieves SOTA on Peptides-Struct.
  NeurIPS 2024: tuned GCN matches Graph Transformers on 17/18 datasets.
  → For venues, Graph Transformers NOT yet production-ready.
  → Classic GNNs with proper tuning remain pragmatic choice.
  → Use GTs only when long-range dependencies clearly necessary.
"""

from torch_geometric.transforms import AddLaplacianEigenvectorPE, AddRandomWalkPE

# PE setup
transform_lap = AddLaplacianEigenvectorPE(k=8, attr_name='lap_pe')
transform_rw = AddRandomWalkPE(walk_length=20, attr_name='rw_pe')
```

---

## GNN-8: Combinatorial Optimization for Event Scheduling

```python
# apps/ml-api/src/gnn/optimization/event_room_assignment.py

"""
Event-room assignment: assign m events to n rooms minimizing cost
subject to capacity, amenity, and no-overlap constraints.

Attention Model (Kool et al., ICLR 2019, arXiv:1803.08475):
  Transformer encoder → node embeddings.
  Autoregressive decoder: u_i = C·tanh(q^T k_i/√d_k) clipped at C=10.
  REINFORCE with greedy rollout baseline.
  <1% optimality gap on TSP-100.

For venue scheduling:
  1. Encode event-room bipartite graph with GNN
  2. Soft assignment via Sinkhorn layer
  3. Decode to feasible assignment via Hungarian algorithm
  4. GNN solution WARM-STARTS exact MIP solver:

gnn_assignment = gnn_model(event_features, room_features, graph)
model = gurobipy.Model()
x = model.addVars(events, rooms, vtype=GRB.BINARY)
for e, r in gnn_assignment:
    x[e, r].Start = 1.0  # Warm start from GNN
model.optimize()

MIP-GNN (Khalil et al., AAAI 2022):
  GNN predicts variable biases to guide branch-and-bound.
  MILP as bipartite graph (variables ↔ constraints).
  Half-convolution: variable→constraint and constraint→variable.

Generalization challenge: models trained on n=100 degrade at n=500+.
Recent: GLOP (AAAI 2024) hierarchical partition-and-solve,
Scale-Net (AAAI 2026) U-Net for cross-scale generalization.
"""
```

---

## GNN-9: Scalable Inference (<100ms at 100K Venues)

### What to Build

Mini-batch training, GLNN distillation, FAISS serving, ONNX browser export.

```python
# apps/ml-api/src/gnn/inference/glnn_distill.py

"""
GLNN (Zhang et al., ICLR 2022):
Train MLP student to mimic GNN teacher via knowledge distillation:
  L = λ·KL(z_student, z_teacher) + (1-λ)·CE(y_student, y_true)

Graph structure gets "baked into" soft labels.
Distilled MLP: 146×-273× FASTER than GNN teacher.
Matches accuracy on 6/7 datasets.

For venue graphs: train GNN offline → deploy MLP for real-time inference.
No graph dependency at serving time.

GLNN works best when node features have high MI with labels (homophilous).
Degrades on highly structural tasks.
TINED: 94× speedup with up to 3.21% improvement over basic GLNN.

Browser inference via ONNX Runtime Web:
  Full GNN with dynamic graphs is impractical in browser.
  GLNN → MLP → ONNX export → ONNX Runtime Web (WASM + WebGPU).
  MLPs export cleanly (unlike GNNs: scatter/gather ops unsupported).
  ONNX Runtime Web: ~3.4× speedup from SIMD+threads.
"""

# apps/ml-api/src/gnn/inference/neighbor_loader.py

from torch_geometric.loader import NeighborLoader

"""
Mini-batch training strategies:

1. GraphSAGE NeighborLoader: fanout [25, 10] for 2-layer GNN.
   Trades accuracy for bounded memory.

2. ClusterGCN (Chiang et al., KDD 2019):
   Pre-partition with METIS → train on cluster subgraphs.

3. GraphSAINT (Zeng et al., ICLR 2020):
   Importance-based subgraph sampling + normalization.
   Random walk sampler performs best in practice.

loader = NeighborLoader(
    data, num_neighbors=[25, 10], batch_size=512,
    input_nodes=data.train_mask, shuffle=True)
"""
```

### Production Architecture

```
Three tiers for <100ms latency at 100K venues:

Option A (fastest, RECOMMENDED):
  Train GNN offline → distill to GLNN/MLP → deploy MLP
  <1ms per-node inference. 100K nodes: ~50ms batch on GPU.

Option B:
  Pre-compute all 100K embeddings offline → FAISS IVF+PQ index
  ANN search <5ms for top-100. Total pipeline <20ms.

Option C (real-time GNN):
  2-layer GraphSAGE, hidden_dim=64, fanout [15, 10]
  Single-query: 5-15ms on GPU.

torch.compile (PyG 2.5+): 2-5.4× speedup.
DistDGL: 100M nodes, 3B edges at 13s/epoch on 16 machines.
```

---

## GNN-10: Explainability for Venue Recommendations

```python
# apps/ml-api/src/gnn/explainability/gnn_explainer.py

from torch_geometric.explain import Explainer, GNNExplainer

"""
GNNExplainer (Ying et al., NeurIPS 2019, arXiv:1903.03894):
  max_{G_S, X_S} MI(Y, (G_S, X_S))
  Learnable continuous edge mask → gradient descent (100-300 epochs)
  Size + Laplacian + entropy regularization → discrete mask
  17.1% improvement over gradient/attention baselines.

PGExplainer (Luo et al., NeurIPS 2020):
  Parameterized MLP generates explanations for ANY instance.
  Binary concrete distributions for differentiable edge sampling.
  108× FASTER than GNNExplainer, up to 24.7% AUC improvement.

CF-GNNExplainer (Lucic et al., AISTATS 2022):
  "What minimal change would alter the prediction?"
  "If your venue had a stage, this event would rank higher."
  Finds minimal edge deletions (typically <3 edges), ≥94% accuracy.

Natural language template mapping:
  High-importance edge (venue → event_type) →
    "Your venue has hosted similar event types"
  High-importance edge (venue → location_cluster) →
    "Nearby venues were successful for this event type"
"""

explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(mode='multiclass_classification',
                      task_level='node', return_type='log_probs'))
```

---

## GNN-11: Integration with OT, TDA, Sheaves, and Pricing

```python
# apps/ml-api/src/gnn/integration/ot_gnn.py

"""
OT-GNN (Béligneul et al., arXiv:2006.04804):
  Replace lossy sum/mean readouts with Wasserstein distances to prototypes.
  W_2(H, P_k) = min_{T∈Π} Σ_{ij} T_{ij}·||h_i - p_k^j||²
  Vector [W_2(H,P_1), ..., W_2(H,P_K)] is a universal approximator.
  Strictly more expressive than sum aggregation.

  Fused Gromov-Wasserstein (Vincent-Cuaz et al., NeurIPS 2022):
  Jointly compare feature AND structural dissimilarity.
  POT library integrates with PyG.

  For venues: venue as distribution over node embeddings,
  compare via W₂, match feature distributions to event requirements.

TOGL (Horn et al., ICLR 2022):
  Persistent homology as differentiable GNN layer.
  Learned filtration → persistence diagrams.
  H_0: market segments (connected clusters)
  H_1: redundant paths in accessibility networks
  Strictly more powerful than WL[1] test. +8% on MNIST-type.

Neural Sheaf Diffusion (Bodnar et al., NeurIPS 2022):
  Each node has a vector space (stalk), each edge has restriction maps.
  Sheaf Laplacian L_F = δ^T δ generalizes graph Laplacian.
  Standard GNNs = trivial sheaf (identity restriction maps).
  Non-trivial sheaves separate in heterophilic settings where GCN fails.
  Sheaf4Rec (ACM Trans. RecSys, 2025): SOTA for recommendations.
"""

# apps/ml-api/src/gnn/integration/demand_gnn.py

"""
GNN → demand prediction → pricing pipeline:
  1. GNN encoder processes venue graph → embeddings
  2. Demand predictor (DeepAR) → D(t) ~ N(μ_t, σ_t²)
  3. Stochastic pricing optimizer: max E[revenue] = p·E[D(p)]

GraphDeepAR (Amazon Science, 2024):
  GNN-based spatial correlations significantly improve demand forecasts.
"""
```

---

## GNN-12: Production Data Flow Architecture

```
┌────────────────────────────────────────────────────────────┐
│  OFFLINE PIPELINE (hourly/daily)                           │
│  Graph Builder (Neo4j) → GNN Training (PyG, GPU)          │
│  → GLNN Distillation → Embedding Pre-computation          │
│  → FAISS Index Build → GNNExplainer → Explanation Cache   │
├────────────────────────────────────────────────────────────┤
│  ONLINE SERVING (<50ms)                                    │
│  Request → Feature Lookup (Redis, <2ms)                    │
│  → GLNN/MLP Inference (<1ms) OR FAISS ANN Search (<5ms)   │
│  → Demand Prediction → Price Optimization                  │
│  → Explanation Template Fill → Response                    │
├────────────────────────────────────────────────────────────┤
│  EVENT-DRIVEN UPDATES (Kafka)                              │
│  New booking → Update TGN memory                           │
│  New venue → k-hop subgraph recomputation → Cache update   │
│  Price change → Demand model update → Pricing refresh      │
└────────────────────────────────────────────────────────────┘

Browser-side:
  GLNN MLP via ONNX Runtime Web (WASM + WebGPU backend)
  Embedding lookup for instant recommendations
  Attention heatmap visualization (GAT weights)
  Explanation cards from cached templates
```

---

## Session Management

1. **GNN-1** (MPNN core: GCN, GraphSAGE, GAT/GATv2, GIN + graph construction) — 1-2 sessions
2. **GNN-2** (Heterogeneous: R-GCN, HAN, HGT, Simple-HGN, to_hetero) — 1-2 sessions
3. **GNN-3** (Recommendation: LightGCN, PinSage, SR-GNN, KGAT, cold-start) — 2 sessions
4. **GNN-4** (Spatial layout: layout graph, quality GNN, LayoutGMN, surrogate) — 1-2 sessions
5. **GNN-5** (Generation: GraphRNN, GRAN, DiGress, graph→layout, conditional) — 1-2 sessions
6. **GNN-6** (Temporal: TGN, TGAT, time encoding, booking stream) — 1 session
7. **GNN-7** (Graph Transformers: GPS, PE, Exphormer — future-proofing) — 1 session
8. **GNN-8** (Combinatorial: Attention Model, Sinkhorn+Hungarian, MIP warm-start) — 1 session
9. **GNN-9** (Scalable inference: NeighborLoader, GLNN distill, FAISS, ONNX) — 1-2 sessions
10. **GNN-10** (Explainability: GNNExplainer, PGExplainer, CF, templates) — 1 session
11. **GNN-11** (Integration: OT-GNN, TOGL, Sheaf, demand→pricing pipeline) — 1 session
12. **GNN-12** (Production architecture: Neo4j, Redis, Kafka, ONNX RT Web) — 1-2 sessions

Total: ~14-18 Claude Code sessions.
