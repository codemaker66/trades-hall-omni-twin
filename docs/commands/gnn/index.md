# Graph Neural Networks Command Track

Canonical command track: `docs/commands/gnn/`

Execution authority for agent commands is `docs/commands/**`.

## Commands

- `GNN-1` - MPNN Foundation and Architecture Selection (depends_on: None)
- `GNN-2` - Heterogeneous Venue Ecosystem Graph (depends_on: `GNN-1`)
- `GNN-3` - Recommendation System (LightGCN + PinSage) (depends_on: `GNN-2`)
- `GNN-4` - Spatial Layout Understanding (depends_on: `GNN-3`)
- `GNN-5` - Graph Generation for Layout Synthesis (depends_on: `GNN-INT-1`)
- `GNN-6` - Temporal Graph Networks for Dynamic Booking (depends_on: `GNN-5`)
- `GNN-7` - Graph Transformers (GPS Framework) (depends_on: `GNN-6`)
- `GNN-8` - Combinatorial Optimization for Event Scheduling (depends_on: `GNN-7`)
- `GNN-9` - Scalable Inference (<100ms at 100K Venues) (depends_on: `GNN-INT-2`)
- `GNN-10` - Explainability for Venue Recommendations (depends_on: `GNN-9`)
- `GNN-11` - Integration with OT, TDA, Sheaves, and Pricing (depends_on: `GNN-10`)
- `GNN-12` - Production Data Flow Architecture (depends_on: `GNN-11`)
- `GNN-INT-1` - Graph Neural Networks integration checkpoint 1 (depends_on: `GNN-1`, `GNN-2`, `GNN-3`, `GNN-4`)
- `GNN-INT-2` - Graph Neural Networks integration checkpoint 2 (depends_on: `GNN-5`, `GNN-6`, `GNN-7`, `GNN-8`)
- `GNN-INT-3` - Graph Neural Networks integration checkpoint 3 (depends_on: `GNN-9`, `GNN-10`, `GNN-11`, `GNN-12`)
