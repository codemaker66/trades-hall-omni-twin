"""
Core persistent homology computation (TDA-1).

Uses Ripser for Vietoris-Rips persistence and GUDHI for alpha complexes.
Implements auto-scaling strategy for datasets from 10 to 50K+ points.

Key papers:
- Bauer 2021, "Ripser: efficient computation of VR persistence" (arXiv:1908.02518)
- Ali et al. 2023, "Persistence statistics" (IEEE TPAMI, arXiv:2212.09703)

Interpretation for venue data:
    H₀: Venue CLUSTERS — groups of similar venues that merge as distance grows.
    H₁: LOOPS in venue space — circular trade-off patterns (capacity↑→price↑→location↓).
    H₂: VOIDS — empty regions = market gaps = business opportunities.
"""

import numpy as np
from ripser import ripser


def compute_persistence(
    distance_matrix: np.ndarray,
    max_dim: int = 2,
    threshold: float | None = None,
    n_perm: int | None = None,
) -> dict:
    """
    Compute persistent homology via Vietoris-Rips filtration.

    Args:
        distance_matrix: N×N distance matrix (e.g., Gower distance)
        max_dim: Maximum homology dimension (0=clusters, 1=loops, 2=voids)
        threshold: Max filtration value. Set to 0.8 for Gower distances
                   to prevent exponential complex growth.
        n_perm: Greedy permutation subsampling size (for large datasets)

    Returns:
        Dictionary with persistence diagrams per dimension and statistics.
    """
    kwargs: dict = {
        "maxdim": max_dim,
        "distance_matrix": True,
    }
    if threshold is not None:
        kwargs["thresh"] = threshold
    if n_perm is not None:
        kwargs["n_perm"] = n_perm

    result = ripser(distance_matrix, **kwargs)

    diagrams: dict = {}
    stats: dict = {}

    for dim in range(max_dim + 1):
        key = f"H{dim}"
        dgm = result["dgms"][dim]

        # Filter out infinite features for finite analysis
        finite = dgm[dgm[:, 1] != np.inf] if len(dgm) > 0 else np.empty((0, 2))
        lifespans = finite[:, 1] - finite[:, 0] if len(finite) > 0 else np.array([])

        diagrams[key] = dgm.tolist()

        # Persistence statistics (Ali et al. 2023 — these simple stats
        # outperform complex vectorizations on standard benchmarks)
        if len(lifespans) > 0:
            normed = lifespans / (lifespans.sum() + 1e-10)
            entropy = float(-np.sum(normed * np.log(normed + 1e-10)))
            stats[key] = {
                "count": int(len(lifespans)),
                "mean_lifespan": float(np.mean(lifespans)),
                "std_lifespan": float(np.std(lifespans)),
                "median_lifespan": float(np.median(lifespans)),
                "max_lifespan": float(np.max(lifespans)),
                "iqr_lifespan": float(
                    np.percentile(lifespans, 75) - np.percentile(lifespans, 25)
                ),
                "entropy": entropy,
            }
        else:
            stats[key] = {"count": 0}

    return {
        "diagrams": diagrams,
        "stats": stats,
        "num_points": int(distance_matrix.shape[0]),
    }


def compute_persistence_scaled(
    distance_matrix: np.ndarray,
    max_dim: int = 2,
) -> dict:
    """
    Auto-select the best computation strategy based on dataset size.

    <3,000 venues:  Direct Ripser, H₀–H₂, no approximation
    3K–10K:         Ripser + threshold + edge collapse for H₂
    10K–50K:        GUDHI sparse Rips (ε=0.3) for O(n)-size complex
    50K+:           Multiple subsampling (Cao & Rabadán, arXiv:2204.09155)

    Args:
        distance_matrix: N×N distance matrix
        max_dim: Maximum homology dimension

    Returns:
        Same format as compute_persistence()
    """
    n = distance_matrix.shape[0]

    if n <= 3000:
        return compute_persistence(distance_matrix, max_dim=max_dim, threshold=0.8)

    elif n <= 10000:
        # Threshold + subsample for H₂
        result_h01 = compute_persistence(
            distance_matrix, max_dim=1, threshold=0.8
        )
        result_h2 = compute_persistence(
            distance_matrix, max_dim=2, threshold=0.5, n_perm=min(n, 2000)
        )
        result_h01["diagrams"]["H2"] = result_h2["diagrams"]["H2"]
        result_h01["stats"]["H2"] = result_h2["stats"]["H2"]
        return result_h01

    elif n <= 50000:
        # GUDHI sparse Rips
        return _gudhi_sparse_persistence(distance_matrix, max_dim)

    else:
        # Multiple subsampling
        return _multi_subsample_persistence(
            distance_matrix, max_dim, subsample_size=2000, num_subsamples=20
        )


def _gudhi_sparse_persistence(
    distance_matrix: np.ndarray, max_dim: int
) -> dict:
    """GUDHI sparse Rips computation for medium-large datasets (10K-50K)."""
    import gudhi

    rips = gudhi.RipsComplex(distance_matrix=distance_matrix, sparse=0.3)
    st = rips.create_simplex_tree(max_dimension=max_dim + 1)
    st.compute_persistence()

    diagrams: dict = {}
    stats: dict = {}

    for dim in range(max_dim + 1):
        key = f"H{dim}"
        intervals = st.persistence_intervals_in_dimension(dim)
        if len(intervals) == 0:
            diagrams[key] = []
            stats[key] = {"count": 0}
            continue

        diagrams[key] = intervals.tolist()

        finite = intervals[np.isfinite(intervals[:, 1])]
        lifespans = finite[:, 1] - finite[:, 0] if len(finite) > 0 else np.array([])

        if len(lifespans) > 0:
            normed = lifespans / (lifespans.sum() + 1e-10)
            entropy = float(-np.sum(normed * np.log(normed + 1e-10)))
            stats[key] = {
                "count": int(len(lifespans)),
                "mean_lifespan": float(np.mean(lifespans)),
                "std_lifespan": float(np.std(lifespans)),
                "median_lifespan": float(np.median(lifespans)),
                "max_lifespan": float(np.max(lifespans)),
                "iqr_lifespan": float(
                    np.percentile(lifespans, 75) - np.percentile(lifespans, 25)
                ),
                "entropy": entropy,
            }
        else:
            stats[key] = {"count": 0}

    return {
        "diagrams": diagrams,
        "stats": stats,
        "num_points": int(distance_matrix.shape[0]),
    }


def _multi_subsample_persistence(
    distance_matrix: np.ndarray,
    max_dim: int,
    subsample_size: int = 2000,
    num_subsamples: int = 20,
) -> dict:
    """
    Multiple subsampling strategy for very large datasets (50K+).
    Draw K random subsamples, compute PH on each, aggregate statistics.
    """
    n = distance_matrix.shape[0]
    all_stats: dict[str, list] = {f"H{d}": [] for d in range(max_dim + 1)}

    for _ in range(num_subsamples):
        indices = np.random.choice(n, size=min(subsample_size, n), replace=False)
        sub_matrix = distance_matrix[np.ix_(indices, indices)]
        result = compute_persistence(sub_matrix, max_dim=max_dim, threshold=0.8)

        for dim in range(max_dim + 1):
            key = f"H{dim}"
            if result["stats"][key]["count"] > 0:
                all_stats[key].append(result["stats"][key])

    # Aggregate: average the statistics across subsamples
    diagrams: dict = {}
    stats: dict = {}

    for dim in range(max_dim + 1):
        key = f"H{dim}"
        diagrams[key] = []  # Individual diagrams not meaningful for aggregates

        sub_stats = all_stats[key]
        if len(sub_stats) > 0:
            stats[key] = {
                "count": int(np.mean([s["count"] for s in sub_stats])),
                "mean_lifespan": float(np.mean([s["mean_lifespan"] for s in sub_stats])),
                "std_lifespan": float(np.mean([s["std_lifespan"] for s in sub_stats])),
                "median_lifespan": float(
                    np.mean([s["median_lifespan"] for s in sub_stats])
                ),
                "max_lifespan": float(np.max([s["max_lifespan"] for s in sub_stats])),
                "iqr_lifespan": float(
                    np.mean([s["iqr_lifespan"] for s in sub_stats])
                ),
                "entropy": float(np.mean([s["entropy"] for s in sub_stats])),
                "num_subsamples": num_subsamples,
                "subsample_size": subsample_size,
            }
        else:
            stats[key] = {"count": 0}

    return {
        "diagrams": diagrams,
        "stats": stats,
        "num_points": int(n),
        "method": "multi_subsample",
    }
