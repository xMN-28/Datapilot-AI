from fastapi import APIRouter, HTTPException

from ..services.analysis_service import run_analysis
from ..services.storage_service import load_json

router = APIRouter(tags=["analysis"])


@router.post("/datasets/{dataset_id}/analyze")
def analyze(dataset_id: str):
    try:
        return run_analysis(dataset_id)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/datasets/{dataset_id}/analysis")
def get_analysis(dataset_id: str):
    analysis = load_json(dataset_id, "analysis")
    if not analysis:
        raise HTTPException(404, "Analysis not found")
    return analysis


@router.get("/datasets/{dataset_id}/charts")
def charts(dataset_id: str):
    return get_analysis(dataset_id)["chart_specs"]


@router.get("/datasets/{dataset_id}/insights")
def insights(dataset_id: str):
    return get_analysis(dataset_id)["insights"]
