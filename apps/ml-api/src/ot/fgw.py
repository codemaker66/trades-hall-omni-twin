"""
Fused Gromov-Wasserstein for structurally incomparable features.

Vayer et al. 2019 (arXiv:1805.09114): handles matching between venues and
events whose feature spaces have different structures — combining feature
distances (Wasserstein) with structural distances (Gromov-Wasserstein).
"""

import numpy as np
import ot


def fused_gromov_wasserstein(
    C1: np.ndarray,
    C2: np.ndarray,
    M: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    alpha: float = 0.5,
    max_iter: int = 200,
) -> dict:
    """
    Fused Gromov-Wasserstein distance.

    Combines:
    - Feature cost M[i,j] (Wasserstein part, e.g., capacity/price distances)
    - Structure cost from C1, C2 (Gromov-Wasserstein part,
      e.g., internal relationships within venues/events)

    Args:
        C1: (N, N) structural cost matrix for source (venues)
        C2: (M, M) structural cost matrix for target (events)
        M: (N, M) feature cost matrix
        p: (N,) source distribution
        q: (M,) target distribution
        alpha: Trade-off parameter. alpha=0 → pure GW, alpha=1 → pure W
        max_iter: Maximum iterations

    Returns:
        dict with keys: plan, cost, gw_cost, w_cost
    """
    T, log = ot.gromov.fused_gromov_wasserstein(
        M, C1, C2, p, q,
        loss_fun="square_loss",
        alpha=alpha,
        numItermax=max_iter,
        log=True,
    )

    total_cost = float(log.get("fgw_dist", 0.0))
    w_cost = float(np.sum(T * M))

    return {
        "plan": T.tolist(),
        "cost": total_cost,
        "w_cost": w_cost,
        "gw_cost": total_cost - alpha * w_cost if alpha > 0 else total_cost,
    }


def build_structural_cost(
    features: np.ndarray,
    metric: str = "sqeuclidean",
) -> np.ndarray:
    """
    Build a structural (intra-set) cost matrix from feature vectors.

    Args:
        features: (N, D) feature matrix
        metric: Distance metric ('sqeuclidean', 'euclidean', 'cosine')

    Returns:
        (N, N) cost matrix capturing internal structure
    """
    from scipy.spatial.distance import cdist

    C = cdist(features, features, metric=metric)
    # Normalize to [0, 1]
    c_max = C.max()
    if c_max > 0:
        C /= c_max
    return C
