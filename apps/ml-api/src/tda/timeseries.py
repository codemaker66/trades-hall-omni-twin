"""
Booking time series analysis via Takens embedding + SW1PerS (TDA-4).

Detects periodicity, anomalies, and regime changes in venue booking time
series using topological methods robust to noise and shape-agnostic.

Key papers:
- Perea & Harer 2015, "Sliding windows and persistence" / SW1PerS (arXiv:1307.6188)
- Gidea & Katz 2018, persistence landscape norms for regime detection (arXiv:1703.04385)
- Chazal et al. 2024, "TADA: anomaly detection via TDA" (JMLR 25, arXiv:2406.06168)
- Rivera-Castro et al. 2020, "TDA for hospitality demand forecasting" (arXiv:2009.03661)
"""

import numpy as np
from ripser import ripser


def takens_embedding(
    time_series: np.ndarray,
    delay: int,
    dimension: int,
) -> np.ndarray:
    """
    Takens delay embedding: convert a 1D time series into a point cloud.

    x(t) -> [x(t), x(t+τ), x(t+2τ), ..., x(t+(d-1)τ)]

    The resulting point cloud's topology encodes the underlying dynamics:
    - Periodic signals trace LOOPS (detected as H₁ features)
    - Chaotic signals fill VOLUMES
    - Anomalies appear as outlier points or topology changes

    Args:
        time_series: 1D array of values (e.g., daily booking counts)
        delay: τ — time delay in steps (e.g., 7 for weekly periodicity)
        dimension: d — embedding dimension (typically 2 or 3)

    Returns:
        Point cloud as (N, dimension) array
    """
    n = len(time_series)
    num_points = n - (dimension - 1) * delay
    if num_points <= 0:
        raise ValueError(
            f"Time series too short ({n}) for delay={delay}, dim={dimension}"
        )

    embedded = np.zeros((num_points, dimension))
    for i in range(dimension):
        embedded[:, i] = time_series[i * delay: i * delay + num_points]
    return embedded


def detect_periodicity_sw1pers(
    time_series: np.ndarray,
    window_sizes: list[int] | None = None,
) -> list[dict]:
    """
    SW1PerS periodicity detection (Perea & Harer, arXiv:1307.6188).

    For each candidate period, compute Takens embedding and measure
    maximum H₁ persistence. High persistence = strong periodicity.

    Shape-agnostic: detects weekly booking cycles whether the pattern
    is sinusoidal, has sharp weekend spikes, or follows any waveform.

    Args:
        time_series: 1D booking counts (daily)
        window_sizes: Candidate periods in days (default: [7, 14, 30, 90, 365])

    Returns:
        Ranked periodicities with confidence scores
    """
    if window_sizes is None:
        window_sizes = [7, 14, 30, 90, 365]

    # Normalize to [0, 1]
    ts_min, ts_max = time_series.min(), time_series.max()
    ts = (time_series - ts_min) / (ts_max - ts_min + 1e-10)

    results = []
    for window in window_sizes:
        if len(ts) < 3 * window:
            continue

        # Takens embedding with delay = window
        cloud = takens_embedding(ts, delay=window, dimension=2)

        # Compute H₁ persistence
        dgm = ripser(cloud, maxdim=1)["dgms"][1]
        if len(dgm) == 0:
            results.append({
                "period_days": window,
                "persistence": 0.0,
                "confidence": 0.0,
                "label": _period_label(window),
            })
            continue

        # Maximum H₁ persistence = strength of periodicity
        lifespans = dgm[:, 1] - dgm[:, 0]
        max_persistence = float(np.max(lifespans))

        # Confidence: ratio of max to mean (how dominant is the period)
        confidence = max_persistence / (float(np.mean(lifespans)) + 1e-10)

        results.append({
            "period_days": window,
            "persistence": max_persistence,
            "confidence": float(min(confidence / 5.0, 1.0)),
            "label": _period_label(window),
        })

    return sorted(results, key=lambda r: r["persistence"], reverse=True)


def detect_regime_changes(
    time_series: np.ndarray,
    window_size: int = 30,
    step: int = 1,
) -> list[dict]:
    """
    Detect regime changes using persistence landscape L² norms on sliding windows.

    Based on Gidea & Katz (arXiv:1703.04385) — persistence landscape norms
    grow significantly BEFORE financial crashes.

    For venue bookings: detect when booking patterns fundamentally shift
    (seasonal transitions, market disruptions, competitor entry).

    Args:
        time_series: 1D booking counts (daily)
        window_size: Sliding window size in days
        step: Step size for window sliding

    Returns:
        List of detected change points with severity scores
    """
    norms: list[float] = []
    positions: list[int] = []

    for start in range(0, len(time_series) - window_size, step):
        window = time_series[start: start + window_size]

        # Normalize window
        w_min, w_max = window.min(), window.max()
        if w_max - w_min < 1e-10:
            norms.append(0.0)
            positions.append(start + window_size // 2)
            continue

        w_norm = (window - w_min) / (w_max - w_min)

        delay = max(1, window_size // 7)
        try:
            cloud = takens_embedding(w_norm, delay=delay, dimension=2)
        except ValueError:
            norms.append(0.0)
            positions.append(start + window_size // 2)
            continue

        dgm = ripser(cloud, maxdim=1)["dgms"][1]
        if len(dgm) == 0:
            norms.append(0.0)
        else:
            # Use sum of squared lifespans as a proxy for landscape L² norm
            # (avoids persim dependency for PersistenceLandscapeExact)
            lifespans = dgm[:, 1] - dgm[:, 0]
            lifespans = lifespans[np.isfinite(lifespans)]
            norms.append(float(np.sum(lifespans ** 2)))

        positions.append(start + window_size // 2)

    norms_arr = np.array(norms)

    if len(norms_arr) < 3:
        return []

    # Detect change points: where the norm changes significantly
    diff = np.abs(np.diff(norms_arr))
    threshold = np.mean(diff) + 2 * np.std(diff)  # 2-sigma outliers

    if threshold < 1e-10:
        return []

    change_indices = np.where(diff > threshold)[0]

    return [
        {
            "position": int(positions[i]),
            "severity": float(diff[i] / threshold),
            "norm_before": float(norms_arr[i]),
            "norm_after": float(norms_arr[i + 1]),
        }
        for i in change_indices
    ]


def detect_anomalies_tada(
    booking_channels: dict[str, np.ndarray],
    window_size: int = 14,
    step: int = 1,
) -> list[dict]:
    """
    TADA anomaly detection (Chazal et al. 2024, JMLR 25, arXiv:2406.06168).

    Encode time-dependent correlations between booking channels as a
    dynamic graph. Compute PH via VR filtration. Score anomalies via
    deviation from "normal" topology.

    Detects:
    - Sudden decorrelation between channels = system failure
    - Unusual correlation patterns = coordinated fraud
    - Channel behavior divergence = market disruption

    Args:
        booking_channels: Dict mapping channel name to 1D time series
                         e.g., {"online": [...], "phone": [...], "agent": [...]}
        window_size: Window size for sliding analysis
        step: Step size for window sliding

    Returns:
        List of anomalous windows with scores
    """
    channel_names = list(booking_channels.keys())
    n_channels = len(channel_names)
    if n_channels < 2:
        return []

    # Stack channels into matrix (T, n_channels)
    T = min(len(v) for v in booking_channels.values())
    data = np.column_stack([booking_channels[name][:T] for name in channel_names])

    # Compute PH for each sliding window
    topo_features: list[np.ndarray] = []
    positions: list[int] = []

    for start in range(0, T - window_size, step):
        window = data[start: start + window_size]

        # Correlation distance matrix between channels
        corr = np.corrcoef(window.T)
        dist = 1.0 - np.abs(corr)
        np.fill_diagonal(dist, 0.0)

        # Compute H₀ persistence (cluster structure of channels)
        dgm = ripser(dist, maxdim=0, distance_matrix=True)["dgms"][0]
        lifespans = dgm[:, 1] - dgm[:, 0]
        lifespans = lifespans[np.isfinite(lifespans)]

        if len(lifespans) > 0:
            features = np.array([
                float(np.mean(lifespans)),
                float(np.std(lifespans)),
                float(np.max(lifespans)),
                float(len(lifespans)),
            ])
        else:
            features = np.zeros(4)

        topo_features.append(features)
        positions.append(start + window_size // 2)

    if len(topo_features) < 10:
        return []

    features_matrix = np.vstack(topo_features)

    # Compute "normal" distribution (mean + covariance)
    mean = np.mean(features_matrix, axis=0)
    cov = np.cov(features_matrix.T)

    # Handle singular covariance
    cov += np.eye(cov.shape[0]) * 1e-6

    # Mahalanobis distance from normal
    cov_inv = np.linalg.inv(cov)
    anomalies = []

    for i, feat in enumerate(features_matrix):
        diff = feat - mean
        maha = float(np.sqrt(diff @ cov_inv @ diff))

        # Flag if Mahalanobis > 3 (approximately 3-sigma outlier)
        if maha > 3.0:
            anomalies.append({
                "position": int(positions[i]),
                "mahalanobis_score": maha,
                "severity": "high" if maha > 5.0 else "medium",
                "feature_values": {
                    "mean_lifespan": float(feat[0]),
                    "std_lifespan": float(feat[1]),
                    "max_lifespan": float(feat[2]),
                    "num_components": int(feat[3]),
                },
            })

    return anomalies


def _period_label(days: int) -> str:
    """Convert a period in days to a human-readable label."""
    if days <= 7:
        return "weekly"
    if days <= 14:
        return "biweekly"
    if days <= 31:
        return "monthly"
    if days <= 93:
        return "quarterly"
    if days <= 366:
        return "annual"
    return f"{days}-day"
