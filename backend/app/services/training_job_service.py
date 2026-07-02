import threading
import time
from datetime import datetime, timezone
from typing import Any

from .storage_service import new_id

training_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


STEP_NAMES = [
    "Loading dataset",
    "Task detection",
    "Leakage check",
    "Feature filtering",
    "Feature schema",
    "Preprocessing setup",
    "Candidate selection",
    "Model training",
    "Model comparison",
    "Saving model bundle",
    "Preparing prediction workspace",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clock() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _duration(started_at: str | None) -> float | None:
    if not started_at:
        return None
    return round(datetime.now(timezone.utc).timestamp() - datetime.fromisoformat(started_at).timestamp(), 2)


def create_training_job(dataset_id: str, target: str, mode: str) -> dict[str, Any]:
    job_id = new_id("train")
    now = _now()
    state = {
        "job_id": job_id,
        "dataset_id": dataset_id,
        "target": target,
        "mode": mode,
        "status": "queued",
        "progress": 0,
        "current_step": "Queued",
        "current_model": None,
        "started_at": now,
        "elapsed_seconds": 0,
        "estimated_remaining_seconds": None,
        "logs": [],
        "steps": [{"name": name, "status": "pending", "started_at": None, "completed_at": None, "duration_seconds": None} for name in STEP_NAMES],
        "models": [],
        "model_id": None,
        "error": None,
        "cancel_requested": False,
    }
    with _lock:
        training_jobs[job_id] = state
    log(job_id, "info", f"Queued training job for target {target} in {mode.title()} mode")
    return state


def public_state(job_id: str) -> dict[str, Any] | None:
    with _lock:
        job = training_jobs.get(job_id)
        if not job:
            return None
        copy = {k: v for k, v in job.items() if k != "cancel_requested"}
    copy["elapsed_seconds"] = _duration(copy.get("started_at")) or 0
    return copy


def log(job_id: str, level: str, message: str) -> None:
    line = {"time": _clock(), "level": level, "message": message}
    with _lock:
        if job_id in training_jobs:
            training_jobs[job_id]["logs"].append(line)
            training_jobs[job_id]["logs"] = training_jobs[job_id]["logs"][-300:]
    print(f"[{job_id}] {message}", flush=True)


def set_status(job_id: str, status: str, error: str | None = None) -> None:
    with _lock:
        job = training_jobs.get(job_id)
        if not job:
            return
        job["status"] = status
        if error:
            job["error"] = error


def set_progress(job_id: str, progress: int, current_step: str | None = None) -> None:
    with _lock:
        job = training_jobs.get(job_id)
        if not job:
            return
        job["progress"] = max(0, min(100, int(progress)))
        if current_step:
            job["current_step"] = current_step


def step(job_id: str, name: str, status: str, progress: int | None = None) -> None:
    now = _now()
    with _lock:
        job = training_jobs.get(job_id)
        if not job:
            return
        for item in job["steps"]:
            if item["name"] == name:
                if status == "running" and not item["started_at"]:
                    item["started_at"] = now
                if status in {"complete", "failed"}:
                    item["completed_at"] = now
                    item["duration_seconds"] = _duration(item["started_at"])
                item["status"] = status
        job["current_step"] = name
        if progress is not None:
            job["progress"] = max(0, min(100, int(progress)))


def set_models(job_id: str, names: list[str]) -> None:
    with _lock:
        job = training_jobs.get(job_id)
        if not job:
            return
        job["models"] = [{"name": name, "status": "pending", "started_at": None, "completed_at": None, "duration_seconds": None, "metrics": None, "error": None} for name in names]


def model_update(job_id: str, name: str, status: str, metrics: dict[str, Any] | None = None, error: str | None = None) -> None:
    now = _now()
    with _lock:
        job = training_jobs.get(job_id)
        if not job:
            return
        job["current_model"] = name if status == "running" else None
        for model in job["models"]:
            if model["name"] == name:
                if status == "running" and not model["started_at"]:
                    model["started_at"] = now
                if status in {"complete", "failed", "skipped", "timeout"}:
                    model["completed_at"] = now
                    model["duration_seconds"] = _duration(model["started_at"])
                model["status"] = status
                if metrics is not None:
                    model["metrics"] = metrics
                if error:
                    model["error"] = error


def complete(job_id: str, model_id: str) -> None:
    with _lock:
        job = training_jobs.get(job_id)
        if not job:
            return
        job["model_id"] = model_id
        job["status"] = "complete"
        job["progress"] = 100
        job["current_step"] = "Complete"
        job["current_model"] = None
    log(job_id, "success", f"Training complete. Model ready: {model_id}")


def fail(job_id: str, error: str) -> None:
    set_status(job_id, "failed", error)
    log(job_id, "error", error)


def cancel(job_id: str) -> bool:
    with _lock:
        job = training_jobs.get(job_id)
        if not job:
            return False
        job["cancel_requested"] = True
        job["status"] = "cancelled" if job["status"] == "queued" else job["status"]
    log(job_id, "warning", "Cancellation requested")
    return True


def is_cancelled(job_id: str | None) -> bool:
    if not job_id:
        return False
    with _lock:
        return bool(training_jobs.get(job_id, {}).get("cancel_requested"))


def start_training_thread(job_id: str, dataset_id: str, target: str, mode: str) -> None:
    def runner() -> None:
        try:
            from .prediction_service import train_model

            set_status(job_id, "running")
            summary = train_model(dataset_id, target, mode, job_id=job_id)
            if is_cancelled(job_id):
                set_status(job_id, "cancelled")
                return
            complete(job_id, summary["model_id"])
        except Exception as exc:
            fail(job_id, str(exc))

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
