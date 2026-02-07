"""
FastAPI endpoints for all TDA operations (TDA-1 through TDA-6).
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/tda", tags=["tda"])


# ─── Request/Response Models ───────────────────────────────────────────────


class VenueData(BaseModel):
    capacity: int
    price_per_event: float
    sq_footage: float
    venue_type: str
    lat: float
    lng: float
    amenities: list[bool]


class PersistenceRequest(BaseModel):
    venues: list[VenueData]
    max_dim: int = 2
    threshold: float | None = 0.8


class MapperRequest(BaseModel):
    venue_features: list[list[float]]
    event_features: list[list[float]]
    compatibility_scores: list[list[float]]
    n_cubes: int = 12
    overlap: float = 0.4


class FurnitureItem(BaseModel):
    x: float
    y: float
    type: str = "chair"
    width: float = 2.0
    depth: float = 2.0


class LayoutAnalysisRequest(BaseModel):
    furniture: list[FurnitureItem]
    room_boundary: list[list[float]]  # [[x1,y1], [x2,y2], ...]
    dead_space_threshold_ft: float = 6.0


class LayoutCompareRequest(BaseModel):
    layout_a: list[FurnitureItem]
    layout_b: list[FurnitureItem]
    room_boundary: list[list[float]]


class TimeSeriesRequest(BaseModel):
    values: list[float]
    window_sizes: list[int] | None = None


class RegimeChangeRequest(BaseModel):
    values: list[float]
    window_size: int = 30
    step: int = 1


class AnomalyRequest(BaseModel):
    channels: dict[str, list[float]]
    window_size: int = 14


class BookingData(BaseModel):
    venue_id: str
    event_id: str
    vendor_ids: list[str] = []
    timeslot_id: str = "default"


class SimplicialRequest(BaseModel):
    bookings: list[BookingData]


class VectorizeRequest(BaseModel):
    diagrams: dict[str, list[list[float]]]


# ─── Endpoints ─────────────────────────────────────────────────────────────


@router.post("/persistence")
async def compute_venue_persistence(request: PersistenceRequest):
    """
    Compute persistent homology on venue feature space.

    Returns persistence diagrams + market gap analysis.
    H₀ = clusters, H₁ = loops, H₂ = voids (market gaps).
    """
    import numpy as np
    from ..tda.distance import build_venue_distance_matrix
    from ..tda.persistence import compute_persistence_scaled

    if len(request.venues) < 3:
        raise HTTPException(400, "Need at least 3 venues for TDA")

    venues = [v.model_dump() for v in request.venues]
    dist_matrix = build_venue_distance_matrix(venues)
    result = compute_persistence_scaled(dist_matrix, max_dim=request.max_dim)

    return result


@router.post("/persistence/diagram")
async def persistence_diagram_image(request: PersistenceRequest):
    """
    Generate persistence diagram as a PNG image (base64).
    """
    import numpy as np
    from ..tda.distance import build_venue_distance_matrix
    from ..tda.persistence import compute_persistence_scaled
    from ..tda.visualize import persistence_diagram_to_png

    venues = [v.model_dump() for v in request.venues]
    dist_matrix = build_venue_distance_matrix(venues)
    result = compute_persistence_scaled(dist_matrix, max_dim=request.max_dim)

    image = persistence_diagram_to_png(result["diagrams"])
    return {"image_base64": image, "stats": result["stats"]}


@router.post("/mapper")
async def build_mapper_graph(request: MapperRequest):
    """
    Build Mapper compatibility graph.
    Returns nodes/edges for frontend rendering.
    """
    import numpy as np
    from ..tda.mapper import build_compatibility_mapper

    venue_features = np.array(request.venue_features)
    event_features = np.array(request.event_features)
    compat = np.array(request.compatibility_scores)

    result = build_compatibility_mapper(
        venue_features, event_features, compat,
        n_cubes=request.n_cubes, overlap=request.overlap,
    )

    return result


@router.post("/layout-analysis")
async def analyze_layout(request: LayoutAnalysisRequest):
    """
    Detect dead spaces in a floor plan layout.
    Returns overlay data for the editor.
    """
    from ..tda.layout import analyze_floor_plan_topology

    furniture = [f.model_dump() for f in request.furniture]
    boundary = [tuple(p) for p in request.room_boundary]

    result = analyze_floor_plan_topology(
        furniture, boundary,
        dead_space_threshold_ft=request.dead_space_threshold_ft,
    )

    return result


@router.post("/layout-compare")
async def compare_layouts(request: LayoutCompareRequest):
    """
    Topological distance between two layouts.
    """
    from ..tda.layout import compare_layouts_topologically

    layout_a = [f.model_dump() for f in request.layout_a]
    layout_b = [f.model_dump() for f in request.layout_b]
    boundary = [tuple(p) for p in request.room_boundary]

    result = compare_layouts_topologically(layout_a, layout_b, boundary)
    return result


@router.post("/booking-periodicity")
async def detect_periodicity(request: TimeSeriesRequest):
    """
    SW1PerS periodicity detection on booking time series.
    """
    import numpy as np
    from ..tda.timeseries import detect_periodicity_sw1pers

    ts = np.array(request.values)
    if len(ts) < 30:
        raise HTTPException(400, "Need at least 30 data points")

    result = detect_periodicity_sw1pers(ts, window_sizes=request.window_sizes)
    return {"periodicities": result}


@router.post("/regime-changes")
async def detect_regime_changes_endpoint(request: RegimeChangeRequest):
    """
    Persistence landscape regime change detection.
    """
    import numpy as np
    from ..tda.timeseries import detect_regime_changes

    ts = np.array(request.values)
    if len(ts) < request.window_size * 2:
        raise HTTPException(400, "Time series too short for window size")

    result = detect_regime_changes(ts, request.window_size, request.step)
    return {"change_points": result}


@router.post("/booking-anomalies")
async def detect_anomalies(request: AnomalyRequest):
    """
    TADA anomaly detection on booking channels.
    """
    import numpy as np
    from ..tda.timeseries import detect_anomalies_tada

    channels = {k: np.array(v) for k, v in request.channels.items()}
    result = detect_anomalies_tada(channels, request.window_size)
    return {"anomalies": result}


@router.post("/simplicial")
async def build_simplicial(request: SimplicialRequest):
    """
    Build simplicial complex from booking data.
    """
    from ..tda.simplicial import build_booking_simplicial_complex

    bookings = [b.model_dump() for b in request.bookings]
    result = build_booking_simplicial_complex(bookings)
    return result


@router.post("/vendor-affinities")
async def vendor_affinities(request: SimplicialRequest):
    """
    Analyze vendor-venue affinities using simplicial analysis.
    """
    from ..tda.simplicial import analyze_vendor_venue_affinities

    bookings = [b.model_dump() for b in request.bookings]
    result = analyze_vendor_venue_affinities(bookings)
    return result


@router.post("/vectorize")
async def vectorize_persistence(request: VectorizeRequest):
    """
    Convert persistence diagrams to ML feature vectors.
    """
    from ..tda.vectorize import persistence_statistics

    features = persistence_statistics(request.diagrams)
    return {"features": features.tolist(), "dimensionality": len(features)}
