"""
FastAPI routes for OT-based venue-event matching.

Endpoints:
  POST /match        — Solve OT matching problem
  POST /barycenter   — Compute ideal venue profile
  POST /learn-weights — Learn cost weights from history
  POST /fgw          — Fused Gromov-Wasserstein matching
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

from ..ot.solver import (
    sinkhorn_solve,
    sinkhorn_unbalanced_solve,
    partial_sinkhorn_solve,
)
from ..ot.barycenter import compute_barycenter, score_against_barycenter
from ..ot.inverse_ot import learn_cost_weights
from ..ot.fgw import fused_gromov_wasserstein, build_structural_cost

app = FastAPI(title="OmniTwin ML API", version="0.1.0")


# ─── Request/Response Models ───────────────────────────────────────────────


class MatchRequest(BaseModel):
    a: list[float]
    b: list[float]
    cost_matrix: list[list[float]]
    reg: float = 0.01
    method: str = "balanced"  # "balanced" | "unbalanced" | "partial"
    reg_m: float = 0.1       # for unbalanced
    mass: float = 0.7        # for partial


class MatchResponse(BaseModel):
    plan: list[list[float]]
    cost: float
    converged: bool


class BarycenterRequest(BaseModel):
    distributions: list[list[float]]
    cost_matrix: list[list[float]]
    weights: list[float] | None = None
    reg: float = 0.01


class BarycenterResponse(BaseModel):
    barycenter: list[float]
    support_size: int


class ScoreRequest(BaseModel):
    barycenter: list[float]
    candidate: list[float]
    cost_matrix: list[list[float]]
    reg: float = 0.01


class ScoreResponse(BaseModel):
    divergence: float


class LearnWeightsRequest(BaseModel):
    matchings: list[dict]
    cost_components: list[list[list[float]]]
    n_events: int
    n_venues: int
    reg: float = 0.05
    initial_weights: list[float] | None = None


class LearnWeightsResponse(BaseModel):
    weights: list[float]
    loss: float
    n_iterations: int
    converged: bool


class FGWRequest(BaseModel):
    C1: list[list[float]]
    C2: list[list[float]]
    M: list[list[float]]
    p: list[float]
    q: list[float]
    alpha: float = 0.5


class FGWResponse(BaseModel):
    plan: list[list[float]]
    cost: float
    w_cost: float
    gw_cost: float


# ─── Endpoints ─────────────────────────────────────────────────────────────


@app.post("/match", response_model=MatchResponse)
async def match(req: MatchRequest):
    """Solve OT matching problem."""
    try:
        a = np.array(req.a)
        b = np.array(req.b)
        C = np.array(req.cost_matrix)

        if req.method == "unbalanced":
            result = sinkhorn_unbalanced_solve(a, b, C, reg=req.reg, reg_m=req.reg_m)
        elif req.method == "partial":
            result = partial_sinkhorn_solve(a, b, C, mass=req.mass, reg=req.reg)
        else:
            result = sinkhorn_solve(a, b, C, reg=req.reg)

        return MatchResponse(
            plan=result["plan"],
            cost=result["cost"],
            converged=result.get("converged", True),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/barycenter", response_model=BarycenterResponse)
async def barycenter(req: BarycenterRequest):
    """Compute Wasserstein barycenter."""
    try:
        dists = [np.array(d) for d in req.distributions]
        C = np.array(req.cost_matrix)
        weights = np.array(req.weights) if req.weights else None

        result = compute_barycenter(dists, C, weights=weights, reg=req.reg)

        return BarycenterResponse(
            barycenter=result["barycenter"],
            support_size=result["support_size"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score", response_model=ScoreResponse)
async def score(req: ScoreRequest):
    """Score a candidate against the ideal barycenter."""
    try:
        bary = np.array(req.barycenter)
        candidate = np.array(req.candidate)
        C = np.array(req.cost_matrix)

        divergence = score_against_barycenter(bary, candidate, C, reg=req.reg)

        return ScoreResponse(divergence=divergence)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/learn-weights", response_model=LearnWeightsResponse)
async def learn_weights_endpoint(req: LearnWeightsRequest):
    """Learn optimal cost weights from observed matchings."""
    try:
        cost_components = [np.array(c) for c in req.cost_components]
        initial = np.array(req.initial_weights) if req.initial_weights else None

        result = learn_cost_weights(
            matchings=req.matchings,
            cost_components=cost_components,
            n_events=req.n_events,
            n_venues=req.n_venues,
            reg=req.reg,
            initial_weights=initial,
        )

        return LearnWeightsResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fgw", response_model=FGWResponse)
async def fgw(req: FGWRequest):
    """Fused Gromov-Wasserstein matching."""
    try:
        C1 = np.array(req.C1)
        C2 = np.array(req.C2)
        M = np.array(req.M)
        p = np.array(req.p)
        q = np.array(req.q)

        result = fused_gromov_wasserstein(C1, C2, M, p, q, alpha=req.alpha)

        return FGWResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
