"""
Alpha complex layout analysis — dead space detection (TDA-3).

Uses GUDHI alpha complexes to analyze venue floor plan layouts.
Detects dead spaces (wasted areas), coverage gaps, and compares
layouts topologically.

Inspired by Hickok et al. 2024 (SIAM Review, arXiv:2206.04834) —
polling-site coverage analysis transferred to venue layouts.

This is the most visually impressive TDA application and has ZERO
prior work in the venue domain.
"""

import numpy as np
import gudhi


def analyze_floor_plan_topology(
    furniture_positions: list[dict],
    room_boundary: list[tuple],
    dead_space_threshold_ft: float = 6.0,
    connectivity_threshold_ft: float = 3.0,
) -> dict:
    """
    Detect dead spaces and coverage gaps in a venue floor plan.

    Method:
    1. Represent furniture as a 2D point cloud (center + corner points)
    2. Build alpha complex filtration (GUDHI — O(n log n) for 2D)
    3. Compute persistent homology:
       H₀: Connected components = isolated furniture groups
       H₁: Loops/holes = empty regions surrounded by furniture
    4. Classify H₁ by persistence (persistence ≈ diameter in feet)

    Args:
        furniture_positions: List of {x, y, type, width, depth} dicts
        room_boundary: Polygon vertices [(x1,y1), (x2,y2), ...]
        dead_space_threshold_ft: H₁ features larger than this = dead spaces
        connectivity_threshold_ft: H₀ features larger than this = disconnected groups

    Returns:
        dead_spaces: detected dead spaces with coordinates and severity
        coverage_score: 0-1 metric of furniture coverage
        connectivity_score: 0-1 metric of layout connectivity
        persistence_diagram: full PH result for visualization
    """
    # 1. Build point cloud from furniture
    points = []
    for item in furniture_positions:
        cx, cy = item["x"], item["y"]
        w = item.get("width", 2.0)
        d = item.get("depth", 2.0)

        # Center point
        points.append([cx, cy])
        # Corner points (for better boundary detection)
        points.extend([
            [cx - w / 2, cy - d / 2],
            [cx + w / 2, cy - d / 2],
            [cx - w / 2, cy + d / 2],
            [cx + w / 2, cy + d / 2],
        ])

    points_arr = np.array(points)

    if len(points_arr) < 4:
        return {
            "dead_spaces": [],
            "coverage_score": 0.0,
            "connectivity_score": 0.0,
            "persistence_diagram": {"H0": [], "H1": []},
            "num_furniture_points": len(points_arr),
        }

    # 2. Alpha complex (optimal for 2D Euclidean — much faster than VR)
    alpha = gudhi.AlphaComplex(points=points_arr.tolist())
    st = alpha.create_simplex_tree()
    st.compute_persistence()

    # 3. Extract persistence diagrams
    h0 = st.persistence_intervals_in_dimension(0)
    h1 = st.persistence_intervals_in_dimension(1)

    if len(h0) == 0:
        h0 = np.empty((0, 2))
    if len(h1) == 0:
        h1 = np.empty((0, 2))

    # 4. Identify dead spaces from H₁
    # Alpha complex filtration values are squared radii
    alpha_threshold = dead_space_threshold_ft ** 2

    dead_spaces = []
    for birth, death in h1:
        persistence = death - birth
        if death != np.inf and persistence > alpha_threshold:
            dead_spaces.append({
                "birth_radius": float(np.sqrt(birth)),
                "death_radius": float(np.sqrt(death)),
                "persistence": float(np.sqrt(persistence)),
                "approx_diameter_ft": float(2 * np.sqrt(death)),
                "severity": "high" if persistence > alpha_threshold * 2 else "medium",
            })

    # 5. Coverage score
    room_area = _polygon_area(room_boundary)
    covered_area = sum(
        item.get("width", 2.0) * item.get("depth", 2.0)
        for item in furniture_positions
    )
    coverage_score = min(covered_area / room_area, 1.0) if room_area > 0 else 0.0

    # 6. Connectivity score
    h0_lifespans = h0[:, 1] - h0[:, 0]
    h0_lifespans = h0_lifespans[np.isfinite(h0_lifespans)]
    conn_threshold_sq = connectivity_threshold_ft ** 2
    long_lived_h0 = int(np.sum(h0_lifespans > conn_threshold_sq))
    connectivity_score = 1.0 / (1.0 + long_lived_h0)

    return {
        "dead_spaces": dead_spaces,
        "coverage_score": float(coverage_score),
        "connectivity_score": float(connectivity_score),
        "persistence_diagram": {
            "H0": h0.tolist(),
            "H1": h1.tolist(),
        },
        "num_furniture_points": len(points_arr),
    }


def compare_layouts_topologically(
    layout_a: list[dict],
    layout_b: list[dict],
    room_boundary: list[tuple],
) -> dict:
    """
    Compare two floor plan layouts using topological distance.

    Uses Wasserstein distance between persistence diagrams (via persim).
    Captures structural similarity that Euclidean metrics miss.

    Args:
        layout_a: First layout furniture positions
        layout_b: Second layout furniture positions
        room_boundary: Room polygon vertices

    Returns:
        topological_distance: per-dimension distances
        analysis_a/analysis_b: full topology analysis
    """
    from persim import wasserstein as wasserstein_distance

    analysis_a = analyze_floor_plan_topology(layout_a, room_boundary)
    analysis_b = analyze_floor_plan_topology(layout_b, room_boundary)

    distances: dict[str, float] = {}
    for dim in ["H0", "H1"]:
        dgm_a = np.array(analysis_a["persistence_diagram"][dim])
        dgm_b = np.array(analysis_b["persistence_diagram"][dim])

        # Filter out infinite features
        if len(dgm_a) > 0:
            dgm_a = dgm_a[np.isfinite(dgm_a[:, 1])]
        else:
            dgm_a = np.empty((0, 2))

        if len(dgm_b) > 0:
            dgm_b = dgm_b[np.isfinite(dgm_b[:, 1])]
        else:
            dgm_b = np.empty((0, 2))

        if len(dgm_a) > 0 or len(dgm_b) > 0:
            distances[dim] = float(wasserstein_distance(dgm_a, dgm_b))
        else:
            distances[dim] = 0.0

    return {
        "topological_distance": distances,
        "analysis_a": analysis_a,
        "analysis_b": analysis_b,
    }


def _polygon_area(vertices: list[tuple]) -> float:
    """Compute area of a polygon using the shoelace formula."""
    n = len(vertices)
    if n < 3:
        return 0.0

    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    return abs(area) / 2.0
