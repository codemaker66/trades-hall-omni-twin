"""
POT-based Sinkhorn solver for large-scale problems.

Uses the Python Optimal Transport (POT) library for batch processing
that exceeds browser capabilities.
"""

import numpy as np
import ot


def sinkhorn_solve(
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    reg: float = 0.01,
    max_iter: int = 1000,
) -> dict:
    """
    Solve an OT problem using POT's Sinkhorn implementation.

    Args:
        a: Source distribution (N,)
        b: Target distribution (M,)
        C: Cost matrix (N, M)
        reg: Entropic regularization (epsilon)
        max_iter: Maximum Sinkhorn iterations

    Returns:
        dict with keys: plan, cost, converged
    """
    T = ot.sinkhorn(a, b, C, reg=reg, numItermax=max_iter)
    cost = float(np.sum(T * C))
    return {
        "plan": T.tolist(),
        "cost": cost,
        "converged": True,
    }


def sinkhorn_unbalanced_solve(
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    reg: float = 0.01,
    reg_m: float = 0.1,
    max_iter: int = 1000,
) -> dict:
    """
    Solve an unbalanced OT problem with marginal relaxation.

    Args:
        a: Source distribution (N,)
        b: Target distribution (M,)
        C: Cost matrix (N, M)
        reg: Entropic regularization
        reg_m: Marginal relaxation (KL penalty)
        max_iter: Maximum iterations

    Returns:
        dict with keys: plan, cost, converged
    """
    T = ot.unbalanced.sinkhorn_unbalanced(
        a, b, C, reg=reg, reg_m=reg_m, numItermax=max_iter
    )
    cost = float(np.sum(T * C))
    return {
        "plan": T.tolist(),
        "cost": cost,
        "converged": True,
    }


def partial_sinkhorn_solve(
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    mass: float = 0.7,
    reg: float = 0.01,
    max_iter: int = 1000,
) -> dict:
    """
    Solve a partial OT problem (transport fraction of total mass).

    Args:
        a: Source distribution (N,)
        b: Target distribution (M,)
        C: Cost matrix (N, M)
        mass: Fraction of mass to transport (0, 1]
        reg: Entropic regularization
        max_iter: Maximum iterations

    Returns:
        dict with keys: plan, cost, mass_transported
    """
    T = ot.partial.entropic_partial_wasserstein(
        a, b, C, reg=reg, m=mass, numItermax=max_iter
    )
    cost = float(np.sum(T * C))
    transported = float(np.sum(T))
    return {
        "plan": T.tolist(),
        "cost": cost,
        "mass_transported": transported,
    }
