import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1] / "storage"
DATASETS = ROOT / "datasets"
MODELS = ROOT / "models"
EXPORTS = ROOT / "exports"

for directory in (DATASETS, MODELS, EXPORTS):
    directory.mkdir(parents=True, exist_ok=True)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dataset_dir(dataset_id: str) -> Path:
    path = DATASETS / dataset_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_dir(dataset_id: str, model_id: str) -> Path:
    path = MODELS / dataset_id / model_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_upload(dataset_id: str, filename: str, fileobj: Any) -> Path:
    suffix = Path(filename).suffix or ".csv"
    target = dataset_dir(dataset_id) / f"raw{suffix}"
    with target.open("wb") as out:
        shutil.copyfileobj(fileobj, out)
    return target


def csv_path(dataset_id: str) -> Path:
    candidates = list(dataset_dir(dataset_id).glob("raw.*"))
    if not candidates:
        raise FileNotFoundError("Dataset CSV not found")
    return candidates[0]


def read_frame(dataset_id: str) -> pd.DataFrame:
    return pd.read_csv(csv_path(dataset_id))


def save_json(dataset_id: str, name: str, payload: dict) -> Path:
    path = dataset_dir(dataset_id) / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def load_json(dataset_id: str, name: str) -> dict | None:
    path = dataset_dir(dataset_id) / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_model_artifact(dataset_id: str, model_id: str, name: str, artifact: Any) -> Path:
    path = model_dir(dataset_id, model_id) / name
    joblib.dump(artifact, path)
    return path


def load_model_artifact(dataset_id: str, model_id: str, name: str) -> Any:
    return joblib.load(model_dir(dataset_id, model_id) / name)
