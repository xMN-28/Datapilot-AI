from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..services.export_service import analysis_export, html_report

router = APIRouter(tags=["export"])


@router.get("/datasets/{dataset_id}/export/analysis")
def export_analysis(dataset_id: str):
    try:
        return FileResponse(analysis_export(dataset_id), filename="analysis.json")
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/datasets/{dataset_id}/export/report")
def export_report(dataset_id: str):
    try:
        return FileResponse(html_report(dataset_id), filename="datapilot_report.html")
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/datasets/{dataset_id}/export/charts")
def export_charts(dataset_id: str):
    return export_analysis(dataset_id)
