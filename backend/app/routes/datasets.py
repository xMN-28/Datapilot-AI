from fastapi import APIRouter, File, HTTPException, UploadFile

from ..services.analysis_service import profile_dataset
from ..services.storage_service import load_json, new_id, save_upload

router = APIRouter(tags=["datasets"])


@router.post("/datasets/upload")
def upload_dataset(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a CSV file")
    dataset_id = new_id("ds")
    try:
        save_upload(dataset_id, file.filename, file.file)
        profile = profile_dataset(dataset_id, file.filename)
        return profile
    except Exception as exc:
        raise HTTPException(400, f"Invalid CSV: {exc}") from exc


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str):
    profile = load_json(dataset_id, "profile")
    if not profile:
        raise HTTPException(404, "Dataset not found")
    return profile


@router.get("/datasets/{dataset_id}/profile")
def get_profile(dataset_id: str):
    return get_dataset(dataset_id)
