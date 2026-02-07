# TECHNIQUE_03_TDA_PERSISTENT_HOMOLOGY.md — Topological Data Analysis

> **Purpose**: Feed this to Claude Code after the main phases and earlier techniques
> are complete. Implements Topological Data Analysis — persistent homology for market
> gap detection, Mapper for venue-event compatibility graphs, alpha complexes for
> layout dead-space detection, Takens embedding for booking time series, and simplicial
> complexes for higher-order relationship modeling.
>
> **How to use**: Tell Claude Code: "Read TECHNIQUE_03_TDA_PERSISTENT_HOMOLOGY.md and
> implement incrementally, starting from TDA-1."
>
> **Key papers**:
> - Bauer 2021, "Ripser: efficient computation of VR persistence" (arXiv:1908.02518)
> - Carrière et al. 2018, "Statistical analysis and parameter selection for Mapper" (JMLR)
> - Perea & Harer 2015, "Sliding windows and persistence" / SW1PerS (arXiv:1307.6188)
> - Hickok et al. 2024, "Persistent homology for resource coverage" (SIAM Review, arXiv:2206.04834)
> - Ali et al. 2023, "Persistence statistics outperform complex vectorizations" (IEEE TPAMI, arXiv:2212.09703)
> - Rivera-Castro et al. 2020, "TDA for hospitality demand forecasting" (arXiv:2009.03661)
> - Bodnar et al. 2021, "Message Passing Simplicial Networks" (ICML, arXiv:2103.03212)
> - Bick et al. 2023, "Higher-order networks" (SIAM Review, DOI:10.1137/21M1414024)
> - Chazal et al. 2024, "TADA: anomaly detection via TDA" (JMLR 25, arXiv:2406.06168)
> - Damrich et al. 2024, "Spectral distances for high-D PH" (NeurIPS 2024)
> - Barberi et al. 2025, "Mapper for fraud detection" (arXiv:2508.14136)

---

## Why This Matters

TDA is the rarest technique on this list in production software. Almost nobody does
this outside of computational biology and materials science. Implementing it in a
venue planning platform is a genuine novelty — there are zero papers applying TDA
to venue/event layout optimization (confirmed by the deep research). This is original
applied research, not library-wrapper engineering.

What TDA gives you that nothing else can:
- **Market gap detection**: H₂ voids in venue feature space = regions where no venues
  exist = business opportunities
- **Dead space detection in layouts**: H₁ holes in furniture point clouds = wasted
  space that traditional metrics miss
- **Booking pattern analysis**: topological features of time series that survive noise
  where statistical methods fail
- **Higher-order relationships**: simplicial complexes capture venue-vendor-event-timeslot
  relationships that pairwise graphs cannot represent

---

## TDA-1: Python Backend — Persistent Homology on Venue Data

### What to Build

Server-side TDA computation using Ripser + GUDHI for analyzing the venue database.
This runs in the FastAPI ML service, not the browser.

### Architecture

```
apps/ml-api/
  src/
    tda/
      persistence.py        — Core PH computation (Ripser wrapper)
      distance.py           — Gower distance matrix for mixed venue features
      vectorize.py          — Persistence diagram → ML feature vectors
      mapper.py             — Mapper algorithm for compatibility graphs
      timeseries.py         — Takens embedding + SW1PerS for booking data
      layout.py             — Alpha complex analysis for floor plan dead spaces
      simplicial.py         — Simplicial complex construction for relationships
      visualize.py          — Persistence diagram / barcode plotting
    routes/
      tda.py                — FastAPI endpoints for all TDA operations
```

### Dependencies

```
uv add ripser persim gower gudhi keplerMapper scikit-learn scipy numpy
```

### Gower Distance for Mixed Venue Features

The critical first step: computing a distance matrix that handles mixed types
(continuous, categorical, binary, geographic) correctly.

```python
# apps/ml-api/src/tda/distance.py

import numpy as np
from gower import gower_matrix
import pandas as pd

def build_venue_distance_matrix(venues: list[dict]) -> np.ndarray:
    """
    Compute Gower distance matrix for mixed venue features.

    Gower distance:
    - Continuous features (capacity, price, sqft): range-normalized Manhattan
    - Categorical features (venue_type): simple match/mismatch (0 or 1)
    - Binary features (amenities): Dice coefficient
    - Geographic (lat/lng): haversine, then range-normalized

    All normalized to [0,1], averaged. Satisfies triangle inequality.
    Directly feedable to Ripser as a distance matrix.
    """
    # Build DataFrame with typed columns
    df = pd.DataFrame(venues)

    # Expand amenities list into binary columns
    amenity_names = ['projector', 'stage', 'wifi', 'kitchen', 'av', 'outdoor',
                     'parking', 'catering', 'bar', 'dancefloor']
    for i, name in enumerate(amenity_names):
        df[f'amenity_{name}'] = df['amenities'].apply(
            lambda a: bool(a[i]) if i < len(a) else False
        )

    # Select feature columns with proper dtypes
    feature_cols = ['capacity', 'price_per_event', 'sq_footage',  # continuous
                    'venue_type',                                   # categorical
                    'lat', 'lng']                                   # continuous (geographic)
    feature_cols += [f'amenity_{n}' for n in amenity_names]        # binary

    feature_df = df[feature_cols].copy()

    # Gower handles dtype detection automatically:
    #   float/int → continuous, object/category → categorical, bool → binary
    feature_df['venue_type'] = feature_df['venue_type'].astype('category')
    for col in [c for c in feature_df.columns if c.startswith('amenity_')]:
        feature_df[col] = feature_df[col].astype(bool)

    # Optional: per-feature weights emphasizing what matters most
    weights = np.ones(len(feature_df.columns))
    # Increase weight on capacity and price (indices 0, 1)
    weights[0] = 1.5  # capacity
    weights[1] = 1.5  # price

    return gower_matrix(feature_df, weight=weights)
```

### Persistent Homology Computation

```python
# apps/ml-api/src/tda/persistence.py

from ripser import ripser
import numpy as np

def compute_persistence(
    distance_matrix: np.ndarray,
    max_dim: int = 2,
    threshold: float | None = None,
    n_perm: int | None = None
) -> dict:
    """
    Compute persistent homology via Vietoris-Rips filtration.

    Args:
        distance_matrix: N×N Gower distance matrix
        max_dim: Maximum homology dimension (0=clusters, 1=loops, 2=voids)
        threshold: Max filtration value (critical for scaling — prevents
                   exponential complex growth). Set to 0.8 for Gower distances.
        n_perm: Greedy permutation subsampling size (for large datasets)

    Returns:
        Dictionary with persistence diagrams per dimension and interpretation.

    Interpretation for venue data:
        H₀ features: Venue CLUSTERS — groups of similar venues that merge as
                     the distance threshold grows. Long-lived H₀ = well-separated
                     market segments. Short-lived H₀ = venues that are nearly
                     interchangeable.

        H₁ features: LOOPS in venue feature space — circular trade-off patterns.
                     Example: a ring of venues where capacity↑ → price↑ →
                     location_quality↓ → capacity↓ cyclically. Long-lived H₁
                     = robust market structure. Can also reveal pricing loops
                     (arbitrage-like patterns).

        H₂ features: VOIDS — empty regions in venue feature space surrounded by
                     venues on all sides. This is the money: a void means there
                     is DEMAND (venues surround it) but NO SUPPLY (no venue exists
                     in that region). These are market gaps = business opportunities.
    """
    kwargs = {
        'maxdim': max_dim,
        'distance_matrix': True,
    }
    if threshold is not None:
        kwargs['thresh'] = threshold
    if n_perm is not None:
        kwargs['n_perm'] = n_perm

    result = ripser(distance_matrix, **kwargs)

    diagrams = {}
    interpretations = {}

    for dim in range(max_dim + 1):
        dgm = result['dgms'][dim]
        # Filter out infinite features for finite analysis
        finite = dgm[dgm[:, 1] != np.inf] if len(dgm) > 0 else np.empty((0, 2))
        lifespans = finite[:, 1] - finite[:, 0] if len(finite) > 0 else np.array([])

        diagrams[f'H{dim}'] = dgm.tolist()

        # Compute persistence statistics (Ali et al. 2023 — TPAMI)
        # These simple stats outperform complex vectorizations
        if len(lifespans) > 0:
            interpretations[f'H{dim}'] = {
                'count': len(lifespans),
                'mean_lifespan': float(np.mean(lifespans)),
                'std_lifespan': float(np.std(lifespans)),
                'median_lifespan': float(np.median(lifespans)),
                'max_lifespan': float(np.max(lifespans)),
                'iqr_lifespan': float(np.percentile(lifespans, 75) - np.percentile(lifespans, 25)),
                'entropy': float(-np.sum((lifespans / lifespans.sum()) *
                                  np.log(lifespans / lifespans.sum() + 1e-10)))
                           if lifespans.sum() > 0 else 0.0,
            }
        else:
            interpretations[f'H{dim}'] = {'count': 0}

    return {
        'diagrams': diagrams,
        'stats': interpretations,
        'num_points': distance_matrix.shape[0],
    }
```

### Scaling Strategy

```python
def compute_persistence_scaled(distance_matrix: np.ndarray, max_dim: int = 2):
    """
    Auto-select the best computation strategy based on dataset size.

    <3,000 venues:  Direct Ripser, H₀–H₂, no approximation
    3K–10K:         Ripser + threshold + edge collapse for H₂
    10K–50K:        GUDHI sparse Rips (ε=0.3) for O(n)-size complex
    50K+:           Multiple subsampling (Cao & Rabadán, arXiv:2204.09155)
    """
    n = distance_matrix.shape[0]

    if n <= 3000:
        return compute_persistence(distance_matrix, max_dim=max_dim, threshold=0.8)

    elif n <= 10000:
        # Threshold + subsample for H₂
        result_h01 = compute_persistence(distance_matrix, max_dim=1, threshold=0.8)
        result_h2 = compute_persistence(distance_matrix, max_dim=2,
                                         threshold=0.5, n_perm=min(n, 2000))
        result_h01['diagrams']['H2'] = result_h2['diagrams']['H2']
        result_h01['stats']['H2'] = result_h2['stats']['H2']
        return result_h01

    elif n <= 50000:
        # GUDHI sparse Rips
        import gudhi
        rips = gudhi.RipsComplex(distance_matrix=distance_matrix, sparse=0.3)
        st = rips.create_simplex_tree(max_dimension=max_dim + 1)
        st.compute_persistence()
        # Convert to our format...
        return _gudhi_to_result(st, max_dim)

    else:
        # Multiple subsampling: draw K random subsamples, compute PH, average
        return _multi_subsample_persistence(distance_matrix, max_dim,
                                             subsample_size=2000, num_subsamples=20)
```

### Tests for TDA-1

```python
def test_persistence_known_circle():
    """A circle of points should have one long-lived H₁ feature."""
    theta = np.linspace(0, 2*np.pi, 100, endpoint=False)
    points = np.column_stack([np.cos(theta), np.sin(theta)])
    dist = squareform(pdist(points))
    result = compute_persistence(dist, max_dim=1)
    # Exactly one long-lived H₁ feature (the circle)
    h1_lifespans = [b - a for a, b in result['diagrams']['H1'] if b != np.inf]
    long_lived = [l for l in h1_lifespans if l > 0.5]
    assert len(long_lived) == 1

def test_persistence_clusters():
    """Three well-separated clusters should have long-lived H₀ features."""
    ...

def test_gower_distance_symmetry():
    """Gower distance matrix must be symmetric."""
    ...

def test_gower_distance_triangle_inequality():
    """Gower distance must satisfy triangle inequality."""
    ...

def test_scaling_strategy_selects_correctly():
    """Verify correct strategy selected for each dataset size range."""
    ...
```

---

## TDA-2: Mapper Algorithm for Venue-Event Compatibility Graphs

### What to Build

Use the Mapper algorithm to create an interpretable graph of the venue-event
compatibility space. Nodes = clusters of similar venue-event pairings, edges =
overlap between clusters. This graph is the "map" of your market.

### Implementation

```python
# apps/ml-api/src/tda/mapper.py

import kmapper as km
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

def build_compatibility_mapper(
    venue_features: np.ndarray,
    event_features: np.ndarray,
    compatibility_scores: np.ndarray,
    n_cubes: int = 12,
    overlap: float = 0.4,
) -> dict:
    """
    Build a Mapper graph of the venue-event compatibility space.

    The compatibility matrix (from OT cost matrix or simple scoring) defines
    a high-dimensional feature space. Mapper compresses it into an interpretable
    graph.

    Filter function: PCA (captures maximum variance in compatibility patterns)
    Clustering: DBSCAN (no preset cluster count, detects arbitrary shapes)
    Cover: 12 intervals, 40% overlap (start here, adjust for granularity)

    Interpretation:
    - Connected components = fundamentally different market segments
      (e.g., "corporate conference venues" vs "outdoor wedding venues")
    - Branches = niche specializations diverging from a main cluster
    - Loops = versatile venues suitable for multiple event types
    - Node size = number of venue-event pairs in that cluster
    - Node color = mean compatibility score (or success rate, or price tier)
    """
    # Combine venue and event features with compatibility into a joint space
    # Each row = one venue-event pair
    n_venues, n_events = compatibility_scores.shape
    joint_features = []
    for i in range(n_venues):
        for j in range(n_events):
            row = np.concatenate([
                venue_features[i],
                event_features[j],
                [compatibility_scores[i, j]]
            ])
            joint_features.append(row)
    X = np.array(joint_features)

    mapper = km.KeplerMapper(verbose=0)

    # Filter: PCA projection to 2D (captures max variance)
    lens = mapper.fit_transform(X, projection=PCA(n_components=2))

    # Build graph: DBSCAN clustering within overlapping intervals
    graph = mapper.map(
        lens, X,
        clusterer=DBSCAN(eps=0.3, min_samples=5),
        cover=km.Cover(n_cubes=n_cubes, perc_overlap=overlap)
    )

    # Generate interactive HTML visualization
    html = mapper.visualize(
        graph, path_html='mapper_graph.html',
        title='Venue-Event Compatibility Space',
        color_values=X[:, -1],  # Color by compatibility score
        color_function_name='Compatibility Score',
    )

    # Extract graph structure for frontend rendering
    nodes = []
    for node_id, members in graph['nodes'].items():
        nodes.append({
            'id': node_id,
            'size': len(members),
            'mean_compatibility': float(np.mean(X[members, -1])),
            'member_indices': members,
        })

    edges = [{'source': e[0], 'target': e[1]} for e in graph['links'].items()]

    return {
        'nodes': nodes,
        'edges': edges,
        'html': html,  # Self-contained interactive HTML
        'stats': {
            'num_nodes': len(nodes),
            'num_edges': sum(len(v) for v in graph['links'].values()),
            'connected_components': _count_components(graph),
        }
    }
```

### Interactive Visualization (Frontend)

```typescript
// apps/web/src/components/tda/MapperGraph.tsx

/**
 * Render the Mapper graph using D3 force-directed layout or deck.gl.
 *
 * Features:
 * - Nodes sized by cluster size, colored by compatibility/success/price
 * - Hover node → show which venues and events are in that cluster
 * - Click node → filter the venue/event list to show only those members
 * - Toggle color by: compatibility score, average price, success rate,
 *   venue type distribution, event type distribution
 * - Connected components highlighted with distinct background colors
 * - Loops in the graph highlighted (these are the versatile venues)
 *
 * Can also embed KeplerMapper's self-contained HTML in an iframe
 * for quick deployment, then replace with custom D3 rendering later.
 */
```

---

## TDA-3: Alpha Complex Layout Analysis — Dead Space Detection

### What to Build

Use GUDHI's alpha complexes to analyze venue floor plan layouts, detecting dead spaces
(wasted areas), coverage gaps, and comparing layouts topologically. This is the most
visually impressive TDA application and has ZERO prior work in the venue domain.

### Implementation

```python
# apps/ml-api/src/tda/layout.py

import gudhi
import numpy as np

def analyze_floor_plan_topology(
    furniture_positions: list[dict],  # [{x, y, type, width, depth}, ...]
    room_boundary: list[tuple],       # [(x1,y1), (x2,y2), ...] polygon vertices
    analysis_resolution: float = 1.0  # Grid point spacing in feet
) -> dict:
    """
    Detect dead spaces and coverage gaps in a venue floor plan using
    alpha complex persistent homology.

    Inspired by Hickok et al. 2024 (SIAM Review, arXiv:2206.04834) —
    polling-site coverage analysis transferred to venue layouts.

    Method:
    1. Represent furniture as a 2D point cloud (center points + corner points)
    2. Build alpha complex filtration (GUDHI — O(n log n) for 2D Euclidean)
    3. Compute persistent homology:
       H₀: Connected components = isolated furniture groups
           → Long-lived H₀ = disconnected zones (bad: guests can't flow between areas)
       H₁: Loops/holes = empty regions surrounded by furniture
           → Long-lived H₁ = DEAD SPACES (wasted area, blocked zones)
           → Short-lived H₁ = normal gaps between furniture (fine)
    4. Classify H₁ features by persistence:
       - Persistence > threshold → genuine dead space (flag it)
       - Persistence < threshold → normal gap (ignore)

    The threshold is calibrated: persistence ≈ diameter of the dead space in feet.
    A dead space > 6ft across is probably wasted space that could hold furniture.

    Returns:
        dead_spaces: list of detected dead spaces with center coordinates and area
        coverage_score: 0-1 metric of how well furniture covers usable area
        connectivity_score: 0-1 metric of how well-connected the layout is
        persistence_diagram: full PH result for visualization
    """
    # 1. Build point cloud from furniture
    points = []
    for item in furniture_positions:
        cx, cy = item['x'], item['y']
        w, d = item.get('width', 2), item.get('depth', 2)
        # Center point
        points.append([cx, cy])
        # Corner points (for better boundary detection)
        points.extend([
            [cx - w/2, cy - d/2], [cx + w/2, cy - d/2],
            [cx - w/2, cy + d/2], [cx + w/2, cy + d/2],
        ])
    points = np.array(points)

    if len(points) < 4:
        return {'dead_spaces': [], 'coverage_score': 0, 'connectivity_score': 0}

    # 2. Alpha complex (optimal for 2D Euclidean — much faster than VR)
    alpha = gudhi.AlphaComplex(points=points)
    st = alpha.create_simplex_tree()
    st.compute_persistence()

    # 3. Extract persistence diagrams
    h0 = st.persistence_intervals_in_dimension(0)
    h1 = st.persistence_intervals_in_dimension(1)

    # 4. Identify dead spaces from H₁
    dead_space_threshold = 6.0  # feet — holes larger than this are dead spaces
    # Alpha complex filtration values are squared radii, so threshold² = 36
    alpha_threshold = dead_space_threshold ** 2

    dead_spaces = []
    for birth, death in h1:
        persistence = death - birth
        if death != np.inf and persistence > alpha_threshold:
            # This is a genuine dead space
            # Approximate center: centroid of the boundary cycle
            # (For precise center, would need to extract the cycle boundary
            #  from the simplicial complex — use representative cycles)
            dead_spaces.append({
                'birth_radius': float(np.sqrt(birth)),
                'death_radius': float(np.sqrt(death)),
                'persistence': float(np.sqrt(persistence)),
                'approx_diameter_ft': float(2 * np.sqrt(death)),
                'severity': 'high' if persistence > alpha_threshold * 2 else 'medium',
            })

    # 5. Coverage score: ratio of "covered" area to total room area
    room_area = _polygon_area(room_boundary)
    covered_area = sum(item.get('width', 2) * item.get('depth', 2)
                       for item in furniture_positions)
    coverage_score = min(covered_area / room_area, 1.0) if room_area > 0 else 0

    # 6. Connectivity score: 1 if all furniture is connected, lower if fragmented
    # Based on number of long-lived H₀ features (connected components)
    h0_lifespans = h0[:, 1] - h0[:, 0]
    h0_lifespans = h0_lifespans[np.isfinite(h0_lifespans)]
    # Ideal: 1 connected component. Score decreases with fragmentation.
    long_lived_h0 = np.sum(h0_lifespans > 3.0)  # components separated by >3ft
    connectivity_score = 1.0 / (1.0 + long_lived_h0)

    return {
        'dead_spaces': dead_spaces,
        'coverage_score': float(coverage_score),
        'connectivity_score': float(connectivity_score),
        'persistence_diagram': {
            'H0': h0.tolist(),
            'H1': h1.tolist(),
        },
        'num_furniture_points': len(points),
    }


def compare_layouts_topologically(
    layout_a: list[dict],
    layout_b: list[dict],
    room_boundary: list[tuple]
) -> dict:
    """
    Compare two floor plan layouts using topological distance.

    Compute persistence diagrams for both, then measure distance using
    Wasserstein distance between persistence diagrams (via persim).
    This captures structural similarity that Euclidean metrics miss:
    two layouts can have furniture in completely different positions
    but the same topological structure (same connectivity, same dead spaces).

    Returns:
        topological_distance: how structurally different the layouts are
        analysis_a: full topology analysis of layout A
        analysis_b: full topology analysis of layout B
        summary: plain-English comparison
    """
    from persim import wasserstein as wasserstein_distance

    analysis_a = analyze_floor_plan_topology(layout_a, room_boundary)
    analysis_b = analyze_floor_plan_topology(layout_b, room_boundary)

    # Wasserstein distance between persistence diagrams (per dimension)
    distances = {}
    for dim in ['H0', 'H1']:
        dgm_a = np.array(analysis_a['persistence_diagram'][dim])
        dgm_b = np.array(analysis_b['persistence_diagram'][dim])
        # Filter out infinite features
        dgm_a = dgm_a[np.isfinite(dgm_a[:, 1])] if len(dgm_a) > 0 else np.empty((0,2))
        dgm_b = dgm_b[np.isfinite(dgm_b[:, 1])] if len(dgm_b) > 0 else np.empty((0,2))
        if len(dgm_a) > 0 or len(dgm_b) > 0:
            distances[dim] = float(wasserstein_distance(dgm_a, dgm_b))
        else:
            distances[dim] = 0.0

    return {
        'topological_distance': distances,
        'analysis_a': analysis_a,
        'analysis_b': analysis_b,
    }
```

### Frontend Visualization: Dead Space Overlay

```typescript
// apps/web/src/components/tda/DeadSpaceOverlay.tsx

/**
 * Overlay dead space detection results on the 2D floor plan editor.
 *
 * Visual design:
 * - Dead spaces: semi-transparent red circles centered on detected voids,
 *   radius = death_radius from the persistence diagram
 * - Severity: high = solid red, medium = orange, low = yellow
 * - Connectivity issues: dashed lines showing disconnected furniture groups
 * - Tooltip on hover: "Dead space detected: ~12ft diameter. Consider adding
 *   a table or decoration to activate this area."
 *
 * Also show on the 3D view as translucent red columns rising from the floor
 * at dead space locations — impossible to miss.
 *
 * Toggle: "Analyze Layout" button in the editor toolbar runs the analysis
 * (API call to Python backend) and shows/hides the overlay.
 */
```

---

## TDA-4: Booking Time Series Analysis — Takens Embedding + SW1PerS

### What to Build

Detect periodicity, anomalies, and regime changes in venue booking time series using
topological methods that are robust to noise and shape-agnostic (unlike FFT which
assumes sinusoidal patterns).

### Implementation

```python
# apps/ml-api/src/tda/timeseries.py

import numpy as np
from ripser import ripser

def takens_embedding(
    time_series: np.ndarray,
    delay: int,
    dimension: int
) -> np.ndarray:
    """
    Takens delay embedding: convert a 1D time series into a point cloud.

    x(t) → [x(t), x(t+τ), x(t+2τ), ..., x(t+(d-1)τ)]

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
        raise ValueError(f"Time series too short for delay={delay}, dim={dimension}")

    embedded = np.zeros((num_points, dimension))
    for i in range(dimension):
        embedded[:, i] = time_series[i * delay : i * delay + num_points]
    return embedded


def detect_periodicity_sw1pers(
    time_series: np.ndarray,
    window_sizes: list[int] = [7, 14, 30, 90, 365],
) -> list[dict]:
    """
    SW1PerS periodicity detection (Perea & Harer, arXiv:1307.6188).

    For each candidate period (window size), compute the Takens embedding
    and measure the maximum H₁ persistence. High persistence = strong
    periodicity at that frequency.

    Shape-agnostic: detects weekly booking cycles whether the pattern is
    sinusoidal, has sharp weekend spikes, or follows any waveform.

    Returns ranked periodicities with confidence scores.
    """
    # Normalize time series to [0, 1]
    ts = (time_series - time_series.min()) / (time_series.max() - time_series.min() + 1e-10)

    results = []
    for window in window_sizes:
        if len(ts) < 3 * window:
            continue

        # Takens embedding with delay = window
        cloud = takens_embedding(ts, delay=window, dimension=2)

        # Compute H₁ persistence
        dgm = ripser(cloud, maxdim=1)['dgms'][1]
        if len(dgm) == 0:
            results.append({'period': window, 'persistence': 0, 'confidence': 0})
            continue

        # Maximum H₁ persistence = strength of periodicity at this window
        lifespans = dgm[:, 1] - dgm[:, 0]
        max_persistence = float(np.max(lifespans))

        # Confidence: ratio of max persistence to mean (how dominant is the period)
        confidence = max_persistence / (np.mean(lifespans) + 1e-10)

        results.append({
            'period_days': window,
            'persistence': max_persistence,
            'confidence': float(min(confidence / 5.0, 1.0)),  # Normalize to [0,1]
            'label': _period_label(window),
        })

    return sorted(results, key=lambda r: r['persistence'], reverse=True)


def _period_label(days: int) -> str:
    if days <= 7: return 'weekly'
    if days <= 14: return 'biweekly'
    if days <= 31: return 'monthly'
    if days <= 93: return 'quarterly'
    if days <= 366: return 'annual'
    return f'{days}-day'


def detect_regime_changes(
    time_series: np.ndarray,
    window_size: int = 30,
    step: int = 1,
) -> list[dict]:
    """
    Detect regime changes using persistence landscape Lᵖ norms on sliding windows.

    Based on Gidea & Katz (arXiv:1703.04385) — showed that persistence landscape
    norms grow significantly BEFORE financial crashes (250 trading days early).

    For venue bookings: detect when booking patterns fundamentally shift
    (seasonal transitions, market disruptions, competitor entry).

    Method:
    1. Slide a window across the time series
    2. At each position, compute Takens embedding → PH → persistence landscape
    3. Track the L² norm of the landscape over time
    4. Sharp increases in the norm = topology is changing = regime shift

    Returns list of detected change points with severity scores.
    """
    from persim import PersistenceLandscapeExact

    norms = []
    positions = []

    for start in range(0, len(time_series) - window_size, step):
        window = time_series[start:start + window_size]
        cloud = takens_embedding(window, delay=max(1, window_size // 7), dimension=2)

        dgm = ripser(cloud, maxdim=1)['dgms'][1]
        if len(dgm) == 0:
            norms.append(0.0)
        else:
            landscape = PersistenceLandscapeExact(dgms=[dgm], hom_deg=1)
            norms.append(float(landscape.p_norm(p=2)))

        positions.append(start + window_size // 2)

    norms = np.array(norms)

    # Detect change points: where the norm changes significantly
    if len(norms) < 3:
        return []

    diff = np.abs(np.diff(norms))
    threshold = np.mean(diff) + 2 * np.std(diff)  # 2-sigma outliers
    change_indices = np.where(diff > threshold)[0]

    return [{
        'position': int(positions[i]),
        'severity': float(diff[i] / threshold),
        'norm_before': float(norms[i]),
        'norm_after': float(norms[i + 1]),
    } for i in change_indices]


def detect_anomalies_tada(
    booking_channels: dict[str, np.ndarray],
    window_size: int = 14,
) -> list[dict]:
    """
    TADA anomaly detection (Chazal et al. 2024, JMLR 25, arXiv:2406.06168).

    Encode time-dependent correlations between booking channels (online,
    phone, agent, walk-in) as a dynamic graph. Compute PH via VR filtration.
    Score anomalies via Mahalanobis distance from "normal" topology.

    Detects:
    - Sudden decorrelation between channels = system failure
    - Unusual correlation patterns = coordinated fraud
    - Channel behavior divergence = market disruption
    """
    # Implementation: sliding window over multi-channel data,
    # build correlation graph at each step, compute PH, vectorize,
    # fit a "normal" distribution, score outliers via Mahalanobis distance.
    ...
```

---

## TDA-5: Persistence Diagram Vectorization for ML

### What to Build

Convert persistence diagrams into fixed-length feature vectors for use in ML models
(venue recommendation, demand prediction, layout classification).

### Implementation

```python
# apps/ml-api/src/tda/vectorize.py

import numpy as np

def persistence_statistics(diagrams: dict) -> np.ndarray:
    """
    Simple persistence statistics vector.
    Ali et al. 2023 (IEEE TPAMI, arXiv:2212.09703) showed these OUTPERFORM
    complex vectorizations (persistence images, landscapes) on standard benchmarks.
    Achieved 0.992 accuracy on Outex.

    For each homology dimension (H₀, H₁, H₂), compute:
    - count, mean, std, median, IQR, max, range, entropy of lifespans
    - 10th, 25th, 75th, 90th percentile of lifespans

    Total: 12 features × 3 dimensions = 36-dimensional vector.
    Low dimensionality avoids curse of dimensionality. Robust to noise.
    """
    features = []
    for dim in ['H0', 'H1', 'H2']:
        dgm = np.array(diagrams.get(dim, []))
        if len(dgm) == 0:
            features.extend([0.0] * 12)
            continue

        lifespans = dgm[:, 1] - dgm[:, 0]
        lifespans = lifespans[np.isfinite(lifespans)]

        if len(lifespans) == 0:
            features.extend([0.0] * 12)
            continue

        normed = lifespans / (lifespans.sum() + 1e-10)
        entropy = float(-np.sum(normed * np.log(normed + 1e-10)))

        features.extend([
            len(lifespans),
            float(np.mean(lifespans)),
            float(np.std(lifespans)),
            float(np.median(lifespans)),
            float(np.percentile(lifespans, 75) - np.percentile(lifespans, 25)),
            float(np.max(lifespans)),
            float(np.max(lifespans) - np.min(lifespans)),
            entropy,
            float(np.percentile(lifespans, 10)),
            float(np.percentile(lifespans, 25)),
            float(np.percentile(lifespans, 75)),
            float(np.percentile(lifespans, 90)),
        ])

    return np.array(features)
```

---

## TDA-6: Simplicial Complexes for Venue-Vendor-Event Relationships

### What to Build

Model the venue-vendor-event-timeslot network as a simplicial complex — capturing
multi-way relationships that pairwise graphs cannot represent.

### Implementation

```python
# apps/ml-api/src/tda/simplicial.py

import numpy as np

def build_booking_simplicial_complex(bookings: list[dict]) -> dict:
    """
    Construct a simplicial complex from booking data.

    Simplices:
    - 0-simplices: individual entities (venues, vendors, events, timeslots)
    - 1-simplices: pairwise relationships (venue↔vendor contract, event↔venue booking)
    - 2-simplices: 3-way relationships (Vendor V serves Event E at Venue U)
    - 3-simplices: 4-way bookings (Vendor V serves Event E at Venue U during Timeslot T)

    The downward closure property holds naturally:
    if a 4-way booking exists, all subset relationships must also exist.

    Use TopoNetX for construction and analysis, or build manually.

    The Hodge decomposition separates signals on the complex into:
    - Gradient component: preference orderings (A is better than B)
    - Curl component: cyclic preferences (A>B>C>A — reveals intransitivity)
    - Harmonic component: persistent structural patterns (robust affinities)

    This decomposition reveals robust venue-vendor-event affinities that
    pairwise analysis misses.
    """
    # Implementation using TopoNetX or manual construction
    # Build simplicial complex, compute Hodge Laplacians,
    # extract Betti numbers and harmonic representatives
    ...
```

### Integration with Python Backend

```python
# apps/ml-api/src/routes/tda.py — FastAPI endpoints

from fastapi import APIRouter

router = APIRouter(prefix="/tda", tags=["tda"])

@router.post("/persistence")
async def compute_venue_persistence(venue_ids: list[str] | None = None):
    """Compute PH on venue feature space. Returns diagrams + market gap analysis."""

@router.post("/mapper")
async def build_mapper_graph(include_events: bool = True):
    """Build Mapper compatibility graph. Returns nodes/edges for frontend rendering."""

@router.post("/layout-analysis")
async def analyze_layout(floor_plan_id: str):
    """Detect dead spaces in a floor plan. Returns overlay data for the editor."""

@router.post("/layout-compare")
async def compare_layouts(floor_plan_id_a: str, floor_plan_id_b: str):
    """Topological distance between two layouts."""

@router.post("/booking-periodicity")
async def detect_periodicity(venue_id: str):
    """SW1PerS periodicity detection on booking history."""

@router.post("/booking-anomalies")
async def detect_anomalies(venue_id: str, window_days: int = 14):
    """TADA anomaly detection on booking channels."""

@router.post("/regime-changes")
async def detect_regime_changes(venue_id: str):
    """Persistence landscape regime change detection."""
```

---

## TDA-7: Browser-Side TDA via WASM (Small Datasets Only)

### What to Build

Compile Ripser to WASM for small interactive TDA computations in the browser.
Use for: layout analysis of a single venue (<200 furniture points), quick topology
previews, educational visualizations.

### Architecture

```
packages/tda-wasm/
  src/
    ripser.cpp          — Ripser source (from github.com/Ripser/ripser)
    bindings.cpp        — Emscripten bindings exposing ripser() to JS
    index.ts            — TypeScript wrapper with Web Worker support
    worker.ts           — Web Worker for non-blocking computation
  build/
    ripser.wasm         — Compiled WASM binary
    ripser.js           — Emscripten glue code
```

### Build Command

```bash
emcc --bind -O3 -s WASM=1 -s ALLOW_MEMORY_GROWTH=1 \
  -o build/ripser.js ripser.cpp bindings.cpp
```

### TypeScript Wrapper

```typescript
// packages/tda-wasm/src/index.ts

export interface PersistenceResult {
  diagrams: { [dim: string]: [number, number][] };
  computeTimeMs: number;
}

/**
 * Runs Ripser in a Web Worker to avoid blocking the main thread.
 *
 * Performance expectations:
 *   100 points, H₁: < 1 second (interactive)
 *   200 points, H₁: 1-5 seconds (near-real-time)
 *   500 points, H₁: 5-30 seconds (show loading indicator)
 *   H₂: limit to ≤100 points in browser
 */
export async function computePersistenceBrowser(
  distanceMatrix: Float64Array,
  maxDim: number = 1,
  threshold?: number
): Promise<PersistenceResult> {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL('./worker.ts', import.meta.url));
    worker.postMessage({ distanceMatrix, maxDim, threshold });
    worker.onmessage = (e) => { resolve(e.data); worker.terminate(); };
    worker.onerror = (e) => { reject(e); worker.terminate(); };
  });
}
```

### When to Use Browser vs Server

```
Browser (WASM Ripser):
  - Layout analysis of current floor plan (< 200 furniture points)
  - Quick topology preview while editing
  - Educational/demo: show PH computation live

Server (Python Ripser/GUDHI):
  - Full venue database analysis (100s-1000s of venues)
  - Time series analysis (needs numpy/scipy)
  - Mapper graph construction (needs scikit-learn)
  - Simplicial complex analysis (needs TopoNetX)
  - Any computation with > 500 points or H₂
```

---

## Integration with Other Techniques

- **Category Theory** (CT): TDA operations are morphisms in the analysis category.
  `buildDistanceMatrix ∘ computePH ∘ vectorize ∘ classify` is a composed pipeline.
- **Optimal Transport** (OT): Wasserstein distance between persistence diagrams
  (via persim) measures topological similarity. The OT cost matrix and the Gower
  distance matrix share the same heterogeneous distance construction.
- **Stochastic Pricing** (next technique): TDA regime change detection triggers
  price model recalibration.
- **Graph Neural Networks** (later): Simplicial neural networks (TDA-6) are a
  strictly more powerful generalization of GNNs.

---

## Session Management

1. **TDA-1** (Python backend: distance + persistence + scaling) — 1 session
2. **TDA-2** (Mapper algorithm + frontend graph viz) — 1 session
3. **TDA-3** (Alpha complex layout analysis + dead space overlay) — 1-2 sessions
4. **TDA-4** (Time series: Takens + SW1PerS + regime detection) — 1 session
5. **TDA-5** (Vectorization for ML) — 1 session (small)
6. **TDA-6** (Simplicial complexes for relationships) — 1 session
7. **TDA-7** (WASM browser-side Ripser) — 1 session

Each session: implement, write tests, update PROGRESS.md. Commit after each section.
