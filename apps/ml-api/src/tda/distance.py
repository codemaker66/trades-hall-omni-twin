"""
Gower distance matrix for mixed venue features (TDA-1).

Handles continuous, categorical, binary, and geographic feature types.
Satisfies triangle inequality. Directly feedable to Ripser as a distance matrix.
"""

import numpy as np
import pandas as pd
from gower import gower_matrix

# Standard amenity names for binary feature expansion
AMENITY_NAMES = [
    "projector", "stage", "wifi", "kitchen", "av",
    "outdoor", "parking", "catering", "bar", "dancefloor",
]


def build_venue_distance_matrix(
    venues: list[dict],
    capacity_weight: float = 1.5,
    price_weight: float = 1.5,
) -> np.ndarray:
    """
    Compute Gower distance matrix for mixed venue features.

    Gower distance:
    - Continuous features (capacity, price, sqft): range-normalized Manhattan
    - Categorical features (venue_type): simple match/mismatch (0 or 1)
    - Binary features (amenities): Dice coefficient
    - Geographic (lat/lng): continuous, range-normalized

    All normalized to [0,1], averaged. Satisfies triangle inequality.

    Args:
        venues: List of venue dicts with keys: capacity, price_per_event,
                sq_footage, venue_type, lat, lng, amenities (bool list)
        capacity_weight: Weight for capacity feature (default 1.5)
        price_weight: Weight for price feature (default 1.5)

    Returns:
        N×N symmetric distance matrix with values in [0, 1]
    """
    df = pd.DataFrame(venues)

    # Expand amenities list into binary columns
    for i, name in enumerate(AMENITY_NAMES):
        df[f"amenity_{name}"] = df["amenities"].apply(
            lambda a, idx=i: bool(a[idx]) if idx < len(a) else False
        )

    # Select feature columns with proper dtypes
    feature_cols = [
        "capacity", "price_per_event", "sq_footage",  # continuous
        "venue_type",                                   # categorical
        "lat", "lng",                                   # continuous (geographic)
    ]
    feature_cols += [f"amenity_{n}" for n in AMENITY_NAMES]  # binary

    feature_df = df[feature_cols].copy()

    # Gower handles dtype detection automatically:
    #   float/int -> continuous, object/category -> categorical, bool -> binary
    feature_df["venue_type"] = feature_df["venue_type"].astype("category")
    for col in feature_df.columns:
        if col.startswith("amenity_"):
            feature_df[col] = feature_df[col].astype(bool)

    # Per-feature weights
    weights = np.ones(len(feature_df.columns))
    weights[0] = capacity_weight   # capacity
    weights[1] = price_weight      # price

    return gower_matrix(feature_df, weight=weights)


def build_distance_from_points(points: np.ndarray) -> np.ndarray:
    """
    Build Euclidean distance matrix from a 2D point cloud.
    Used for layout analysis (alpha complexes).

    Args:
        points: (N, 2) array of 2D points

    Returns:
        N×N symmetric distance matrix
    """
    from scipy.spatial.distance import pdist, squareform
    return squareform(pdist(points, metric="euclidean"))
