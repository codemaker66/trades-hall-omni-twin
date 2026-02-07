"""
Mapper algorithm for venue-event compatibility graphs (TDA-2).

Carrière et al. 2018, "Statistical analysis and parameter selection for Mapper" (JMLR)
Barberi et al. 2025, "Mapper for fraud detection" (arXiv:2508.14136)

Uses KeplerMapper to create an interpretable graph of the venue-event
compatibility space. Nodes = clusters of similar venue-event pairings,
edges = overlap between clusters.

Interpretation:
- Connected components = fundamentally different market segments
- Branches = niche specializations
- Loops = versatile venues suitable for multiple event types
"""

import numpy as np
import kmapper as km
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN


def build_compatibility_mapper(
    venue_features: np.ndarray,
    event_features: np.ndarray,
    compatibility_scores: np.ndarray,
    n_cubes: int = 12,
    overlap: float = 0.4,
    eps: float = 0.3,
    min_samples: int = 5,
) -> dict:
    """
    Build a Mapper graph of the venue-event compatibility space.

    Args:
        venue_features: (N_venues, d_v) feature array
        event_features: (N_events, d_e) feature array
        compatibility_scores: (N_venues, N_events) compatibility/cost matrix
        n_cubes: Number of intervals for the Mapper cover
        overlap: Fractional overlap between intervals
        eps: DBSCAN epsilon parameter
        min_samples: DBSCAN minimum cluster size

    Returns:
        Dict with nodes, edges, stats, and optional HTML visualization
    """
    n_venues, n_events = compatibility_scores.shape

    # Build joint feature space: each row = one venue-event pair
    joint_features = []
    for i in range(n_venues):
        for j in range(n_events):
            row = np.concatenate([
                venue_features[i],
                event_features[j],
                [compatibility_scores[i, j]],
            ])
            joint_features.append(row)
    X = np.array(joint_features)

    mapper = km.KeplerMapper(verbose=0)

    # Filter: PCA projection to 2D (captures max variance)
    lens = mapper.fit_transform(X, projection=PCA(n_components=2))

    # Build graph: DBSCAN clustering within overlapping intervals
    graph = mapper.map(
        lens,
        X,
        clusterer=DBSCAN(eps=eps, min_samples=min_samples),
        cover=km.Cover(n_cubes=n_cubes, perc_overlap=overlap),
    )

    # Extract graph structure for frontend rendering
    nodes = []
    for node_id, members in graph["nodes"].items():
        nodes.append({
            "id": node_id,
            "size": len(members),
            "mean_compatibility": float(np.mean(X[members, -1])),
            "member_indices": members,
            # Decode member indices back to (venue_idx, event_idx) pairs
            "pairs": [
                {"venue_idx": int(m // n_events), "event_idx": int(m % n_events)}
                for m in members
            ],
        })

    edges = []
    for source_id, targets in graph["links"].items():
        for target_id in targets:
            edges.append({"source": source_id, "target": target_id})

    # Count connected components via BFS
    num_components = _count_components(graph)

    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "connected_components": num_components,
            "total_pairs": n_venues * n_events,
        },
    }


def build_mapper_with_html(
    venue_features: np.ndarray,
    event_features: np.ndarray,
    compatibility_scores: np.ndarray,
    **kwargs,
) -> dict:
    """
    Build Mapper graph with self-contained HTML visualization.

    Returns the same dict as build_compatibility_mapper plus an 'html' key
    containing a full HTML page for embedding in an iframe.
    """
    n_venues, n_events = compatibility_scores.shape

    # Build joint space
    joint_features = []
    for i in range(n_venues):
        for j in range(n_events):
            row = np.concatenate([
                venue_features[i],
                event_features[j],
                [compatibility_scores[i, j]],
            ])
            joint_features.append(row)
    X = np.array(joint_features)

    mapper = km.KeplerMapper(verbose=0)
    lens = mapper.fit_transform(X, projection=PCA(n_components=2))

    n_cubes = kwargs.get("n_cubes", 12)
    overlap = kwargs.get("overlap", 0.4)
    eps = kwargs.get("eps", 0.3)
    min_samples = kwargs.get("min_samples", 5)

    graph = mapper.map(
        lens,
        X,
        clusterer=DBSCAN(eps=eps, min_samples=min_samples),
        cover=km.Cover(n_cubes=n_cubes, perc_overlap=overlap),
    )

    html = mapper.visualize(
        graph,
        path_html="mapper_graph.html",
        title="Venue-Event Compatibility Space",
        color_values=X[:, -1],
        color_function_name="Compatibility Score",
    )

    result = build_compatibility_mapper(
        venue_features, event_features, compatibility_scores, **kwargs
    )
    result["html"] = html
    return result


def _count_components(graph: dict) -> int:
    """Count connected components in a Mapper graph via BFS."""
    if not graph["nodes"]:
        return 0

    node_ids = set(graph["nodes"].keys())
    adj: dict[str, set] = {nid: set() for nid in node_ids}
    for source, targets in graph["links"].items():
        for target in targets:
            adj[source].add(target)
            adj[target].add(source)

    visited: set[str] = set()
    components = 0

    for node in node_ids:
        if node in visited:
            continue
        components += 1
        queue = [node]
        while queue:
            current = queue.pop()
            if current in visited:
                continue
            visited.add(current)
            for neighbor in adj.get(current, set()):
                if neighbor not in visited:
                    queue.append(neighbor)

    return components
