"""
Wasserstein barycenter computation using POT.

Computes the barycenter of historical successful bookings to create
an "ideal venue" profile.
"""

import numpy as np
import ot


def compute_barycenter(
    distributions: list[np.ndarray],
    cost_matrix: np.ndarray,
    weights: np.ndarray | None = None,
    reg: float = 0.01,
    max_iter: int = 100,
) -> dict:
    """
    Compute the fixed-support Wasserstein barycenter.

    Args:
        distributions: List of N distributions on the same support of size n
        cost_matrix: n×n cost matrix on the support
        weights: N weights (sum to 1). If None, uniform weights used.
        reg: Entropic regularization
        max_iter: Maximum iterations

    Returns:
        dict with keys: barycenter, iterations
    """
    N = len(distributions)
    n = len(distributions[0])

    if weights is None:
        weights = np.ones(N) / N

    # Stack distributions as columns of a matrix
    A = np.column_stack(distributions)

    # POT's barycenter function
    bary = ot.bregman.barycenter(
        A, cost_matrix, reg, weights=weights, numItermax=max_iter
    )

    return {
        "barycenter": bary.tolist(),
        "support_size": n,
    }


def score_against_barycenter(
    barycenter: np.ndarray,
    candidate: np.ndarray,
    cost_matrix: np.ndarray,
    reg: float = 0.01,
) -> float:
    """
    Score a candidate distribution against the ideal barycenter.
    Uses Sinkhorn divergence (debiased).

    Returns:
        float: Sinkhorn divergence score (lower = better match)
    """
    ot_ab = ot.sinkhorn2(barycenter, candidate, cost_matrix, reg=reg)
    ot_aa = ot.sinkhorn2(barycenter, barycenter, cost_matrix, reg=reg)
    ot_bb = ot.sinkhorn2(candidate, candidate, cost_matrix, reg=reg)

    # Sinkhorn divergence
    divergence = float(ot_ab) - 0.5 * float(ot_aa) - 0.5 * float(ot_bb)
    return max(divergence, 0.0)  # Clamp for numerical stability
