from pydantic import BaseModel
from fastapi import APIRouter, HTTPException

from ..services.analyst_tool_service import answer_question, get_last_tool_trace
from ..services.storage_service import load_json

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    question: str


@router.post("/datasets/{dataset_id}/chat")
def chat(dataset_id: str, request: ChatRequest):
    analysis = load_json(dataset_id, "analysis")
    if not analysis:
        raise HTTPException(404, "Run analysis before chatting")
    return answer_question(dataset_id, request.question, analysis)


@router.get("/datasets/{dataset_id}/chat/last-trace")
def last_trace(dataset_id: str):
    trace = get_last_tool_trace(dataset_id)
    if trace is None:
        raise HTTPException(404, "No chat trace recorded for this dataset")
    return trace
