from pydantic import BaseModel
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from ..services.prediction_service import batch_predict, bundle_zip, chat_predict, predict, train_model
from ..services.storage_service import load_json
from ..services.training_job_service import cancel, create_training_job, public_state, start_training_thread

router = APIRouter(tags=["prediction"])


class TrainRequest(BaseModel):
    target: str
    mode: str = "balanced"


class PredictRequest(BaseModel):
    values: dict


class ChatPredictRequest(BaseModel):
    message: str
    conversation_state: dict = {}


@router.post("/datasets/{dataset_id}/models/train")
def train(dataset_id: str, request: TrainRequest):
    try:
        job = create_training_job(dataset_id, request.target, request.mode)
        start_training_thread(job["job_id"], dataset_id, request.target, request.mode)
        return {"job_id": job["job_id"], "status": "queued"}
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/training-jobs/{job_id}")
def training_job(job_id: str):
    state = public_state(job_id)
    if not state:
        raise HTTPException(404, "Training job not found")
    return state


@router.post("/training-jobs/{job_id}/cancel")
def cancel_training_job(job_id: str):
    if not cancel(job_id):
        raise HTTPException(404, "Training job not found")
    return {"job_id": job_id, "status": "cancel_requested"}


@router.get("/datasets/{dataset_id}/models/{model_id}")
def model(dataset_id: str, model_id: str):
    summary = load_json(dataset_id, f"model_{model_id}")
    if not summary:
        raise HTTPException(404, "Model not found")
    return summary


@router.get("/datasets/{dataset_id}/models/{model_id}/stats")
def stats(dataset_id: str, model_id: str):
    return model(dataset_id, model_id)


@router.post("/datasets/{dataset_id}/models/{model_id}/predict")
def make_prediction(dataset_id: str, model_id: str, request: PredictRequest):
    try:
        return predict(dataset_id, model_id, request.values)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/datasets/{dataset_id}/models/{model_id}/chat-predict")
def make_chat_prediction(dataset_id: str, model_id: str, request: ChatPredictRequest):
    try:
        return chat_predict(dataset_id, model_id, request.message, request.conversation_state)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/datasets/{dataset_id}/models/{model_id}/batch-predict")
def make_batch_prediction(dataset_id: str, model_id: str, file: UploadFile = File(...)):
    try:
        path = batch_predict(dataset_id, model_id, file.file)
        return FileResponse(path, filename="batch_predictions.csv")
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/datasets/{dataset_id}/models/{model_id}/download")
def download_model(dataset_id: str, model_id: str):
    try:
        path = bundle_zip(dataset_id, model_id)
        return FileResponse(path, filename="model_bundle.zip")
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
