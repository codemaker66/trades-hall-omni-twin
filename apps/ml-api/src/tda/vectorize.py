"""
Persistence diagram vectorization for ML features (TDA-5).

Ali et al. 2023 (IEEE TPAMI, arXiv:2212.09703) showed simple persistence
statistics OUTPERFORM complex vectorizations (persistence images, landscapes)
on standard benchmarks. Achieved 0.992 accuracy on Outex.

For each homology dimension (H₀, H₁, H₂), compute 12 features:
count, mean, std, median, IQR, max, range, entropy, p10, p25, p75, p90

Total: 12 features × 3 dimensions = 36-dimensional vector.
"""

import numpy as np


FEATURES_PER_DIM = 12
NUM_DIMS = 3  # H0, H1, H2
TOTAL_FEATURES = FEATURES_PER_DIM * NUM_DIMS  # 36


def persistence_statistics(diagrams: dict) -> np.ndarray:
    """
    Compute persistence statistics vector from persistence diagrams.

    Args:
        diagrams: Dict with keys 'H0', 'H1', 'H2', each a list of [birth, death] pairs

    Returns:
        36-dimensional feature vector (12 features × 3 dimensions)
    """
    features: list[float] = []

    for dim in ["H0", "H1", "H2"]:
        dgm = np.array(diagrams.get(dim, []))

        if len(dgm) == 0:
            features.extend([0.0] * FEATURES_PER_DIM)
            continue

        lifespans = dgm[:, 1] - dgm[:, 0]
        lifespans = lifespans[np.isfinite(lifespans)]

        if len(lifespans) == 0:
            features.extend([0.0] * FEATURES_PER_DIM)
            continue

        normed = lifespans / (lifespans.sum() + 1e-10)
        entropy = float(-np.sum(normed * np.log(normed + 1e-10)))

        features.extend([
            float(len(lifespans)),                                        # count
            float(np.mean(lifespans)),                                    # mean
            float(np.std(lifespans)),                                     # std
            float(np.median(lifespans)),                                  # median
            float(np.percentile(lifespans, 75) - np.percentile(lifespans, 25)),  # IQR
            float(np.max(lifespans)),                                     # max
            float(np.max(lifespans) - np.min(lifespans)),                 # range
            entropy,                                                      # entropy
            float(np.percentile(lifespans, 10)),                          # p10
            float(np.percentile(lifespans, 25)),                          # p25
            float(np.percentile(lifespans, 75)),                          # p75
            float(np.percentile(lifespans, 90)),                          # p90
        ])

    return np.array(features)


def persistence_image(
    diagram: list[list[float]],
    resolution: int = 20,
    sigma: float = 0.1,
    weight_fn: str = "linear",
) -> np.ndarray:
    """
    Compute persistence image from a single-dimension diagram.

    Alternative to statistics — creates a 2D image (resolution × resolution)
    suitable for CNNs. Less effective than statistics for standard benchmarks
    but useful for spatial pattern recognition.

    Args:
        diagram: List of [birth, death] pairs
        resolution: Grid resolution (output is resolution × resolution)
        sigma: Gaussian bandwidth
        weight_fn: 'linear' (persistence-weighted) or 'uniform'

    Returns:
        (resolution, resolution) persistence image
    """
    dgm = np.array(diagram)
    if len(dgm) == 0:
        return np.zeros((resolution, resolution))

    # Convert to birth-persistence coordinates
    births = dgm[:, 0]
    persistences = dgm[:, 1] - dgm[:, 0]
    finite_mask = np.isfinite(persistences)
    births = births[finite_mask]
    persistences = persistences[finite_mask]

    if len(births) == 0:
        return np.zeros((resolution, resolution))

    # Grid bounds
    b_min, b_max = float(births.min()), float(births.max())
    p_min, p_max = 0.0, float(persistences.max())

    # Pad bounds slightly
    b_range = max(b_max - b_min, 1e-6)
    p_range = max(p_max - p_min, 1e-6)
    b_min -= 0.1 * b_range
    b_max += 0.1 * b_range
    p_max += 0.1 * p_range

    # Grid
    b_grid = np.linspace(b_min, b_max, resolution)
    p_grid = np.linspace(p_min, p_max, resolution)
    bb, pp = np.meshgrid(b_grid, p_grid, indexing="ij")

    img = np.zeros((resolution, resolution))

    for k in range(len(births)):
        b_k, p_k = births[k], persistences[k]

        # Weight
        if weight_fn == "linear":
            w = p_k / p_range
        else:
            w = 1.0

        # Gaussian kernel
        g = np.exp(-((bb - b_k) ** 2 + (pp - p_k) ** 2) / (2 * sigma ** 2))
        img += w * g

    return img


def bottleneck_features(
    diagrams_a: dict,
    diagrams_b: dict,
) -> dict[str, float]:
    """
    Compute bottleneck distances between two persistence diagrams (per dimension).

    Useful for comparing two venue configurations or time windows.
    """
    from persim import bottleneck

    distances: dict[str, float] = {}
    for dim in ["H0", "H1", "H2"]:
        dgm_a = np.array(diagrams_a.get(dim, []))
        dgm_b = np.array(diagrams_b.get(dim, []))

        # Filter infinite features
        if len(dgm_a) > 0:
            dgm_a = dgm_a[np.isfinite(dgm_a[:, 1])]
        else:
            dgm_a = np.empty((0, 2))

        if len(dgm_b) > 0:
            dgm_b = dgm_b[np.isfinite(dgm_b[:, 1])]
        else:
            dgm_b = np.empty((0, 2))

        if len(dgm_a) > 0 or len(dgm_b) > 0:
            distances[dim] = float(bottleneck(dgm_a, dgm_b))
        else:
            distances[dim] = 0.0

    return distances
