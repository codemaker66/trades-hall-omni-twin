"""
Inverse Optimal Transport: learn cost function weights from observed matchings.

Li et al. 2018 (arXiv:1802.03644): given observed matchings, find the cost
matrix C that makes the observed matchings optimal under OT.
"""

import numpy as np
import ot
from scipy.optimize import minimize


def build_observed_plan(
    matchings: list[dict],
    n_events: int,
    n_venues: int,
) -> np.ndarray:
    """
    Build the observed transport plan from historical matchings.

    Args:
        matchings: List of {event_idx, venue_idx, success} dicts
        n_events: Number of events (rows)
        n_venues: Number of venues (columns)

    Returns:
        N×M transport plan matrix
    """
    T = np.zeros((n_events, n_venues))
    for m in matchings:
        weight = 1.0 if m["success"] else 0.1
        T[m["event_idx"], m["venue_idx"]] += weight

    # Normalize rows
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    T /= row_sums
    return T


def build_weighted_cost(
    weights: np.ndarray,
    cost_components: list[np.ndarray],
) -> np.ndarray:
    """
    Build cost matrix as weighted sum of component matrices.

    Args:
        weights: (K,) weight vector
        cost_components: K component matrices, each (N, M)

    Returns:
        (N, M) weighted cost matrix
    """
    C = np.zeros_like(cost_components[0])
    for w, comp in zip(weights, cost_components):
        C += w * comp
    return C


def learn_cost_weights(
    matchings: list[dict],
    cost_components: list[np.ndarray],
    n_events: int,
    n_venues: int,
    reg: float = 0.05,
    initial_weights: np.ndarray | None = None,
) -> dict:
    """
    Learn optimal cost weights from observed matchings via gradient-free
    optimization (L-BFGS-B).

    Args:
        matchings: Historical matchings
        cost_components: K component cost matrices (capacity, price, amenity, location)
        n_events: Number of events
        n_venues: Number of venues
        reg: Sinkhorn regularization
        initial_weights: Starting weights (default: uniform)

    Returns:
        dict with keys: weights, loss, n_iterations
    """
    K = len(cost_components)
    if initial_weights is None:
        initial_weights = np.ones(K) / K

    # Build observed plan
    T_obs = build_observed_plan(matchings, n_events, n_venues)

    # Marginals
    a = np.ones(n_events) / n_events
    b = np.ones(n_venues) / n_venues

    def objective(w: np.ndarray) -> float:
        # Project to positive and normalize
        w_pos = np.maximum(w, 0.01)
        w_norm = w_pos / w_pos.sum()

        # Build cost matrix
        C = build_weighted_cost(w_norm, cost_components)

        # Solve Sinkhorn
        try:
            T_pred = ot.sinkhorn(a, b, C, reg=reg, numItermax=200)
        except Exception:
            return 1e10

        # Frobenius loss
        return float(np.sum((T_pred - T_obs) ** 2))

    result = minimize(
        objective,
        initial_weights,
        method="L-BFGS-B",
        bounds=[(0.01, 1.0)] * K,
        options={"maxiter": 200},
    )

    # Normalize final weights
    w_final = np.maximum(result.x, 0.01)
    w_final /= w_final.sum()

    return {
        "weights": w_final.tolist(),
        "loss": float(result.fun),
        "n_iterations": int(result.nit),
        "converged": result.success,
    }
