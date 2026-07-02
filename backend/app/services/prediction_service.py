import json
import re
import time
import warnings
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .llm_service import complete_json, complete_text, has_llm
from .storage_service import load_json, model_dir, new_id, read_frame, save_json, save_model_artifact
from ..config import get_settings
from ..utils.missing import clean_categorical_for_chart, is_missing_like

RANDOM_STATE = 42
ID_TOKENS = {"id", "uuid", "user_id", "order_id", "customer_id", "product_id", "transaction_id", "invoice_id", "row_id", "serial_number"}
DATE_PARTS = {"year", "month", "day", "quarter", "week", "day_of_week"}
LEAKAGE_HINTS = {
    "high_value": {"amount", "total", "revenue", "profit", "tax", "discount_amount", "subtotal", "value"},
    "returned": {"return", "refund", "review_rating", "order_status", "return_date"},
    "delivery_delayed": {"actual_delivery", "delivery_days", "delay", "delivered_at"},
}
BOOLEAN_TRUE = {"true", "yes", "1", "y", "t"}
BOOLEAN_FALSE = {"false", "no", "0", "n", "f"}


def _display_name(name: str) -> str:
    return re.sub(r"[_-]+", " ", name).strip().title()


def _is_bool_like(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return True
    values = set(clean_categorical_for_chart(series).dropna().astype(str).str.strip().str.lower().unique())
    return bool(values) and values.issubset(BOOLEAN_TRUE | BOOLEAN_FALSE) and len(values) <= 2


def _is_date_like(name: str, series: pd.Series) -> bool:
    lower = name.lower()
    if lower in DATE_PARTS or lower.endswith(tuple(f"_{part}" for part in DATE_PARTS)):
        return True
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if "date" not in lower and "time" not in lower:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(clean_categorical_for_chart(series).dropna().astype(str).head(80), errors="coerce")
    return not parsed.empty and parsed.notna().mean() > 0.7


def _is_id_like(name: str, series: pd.Series, rows: int) -> bool:
    lower = name.lower()
    if lower in ID_TOKENS or any(lower.endswith(f"_{token}") for token in ID_TOKENS):
        return True
    if any(token in lower for token in ["uuid", "serial", "invoice", "transaction"]) or lower.endswith("_id"):
        return True
    unique_ratio = series.nunique(dropna=True) / max(rows, 1)
    return rows > 25 and unique_ratio > 0.92


def _is_near_constant(series: pd.Series, rows: int) -> bool:
    counts = series.value_counts(dropna=True)
    return series.nunique(dropna=True) <= 1 or (len(counts) > 0 and counts.iloc[0] / max(rows, 1) > 0.985)


def _leakage_reason(target: str, feature: str) -> str | None:
    t = target.lower()
    f = feature.lower()
    if f == t or f.replace("_", "") == t.replace("_", ""):
        return "target column"
    for target_hint, feature_hints in LEAKAGE_HINTS.items():
        if target_hint in t and any(hint in f for hint in feature_hints):
            return f"possible target leakage for {target}"
    target_tokens = {token for token in re.split(r"[_\W]+", t) if len(token) > 3}
    feature_tokens = {token for token in re.split(r"[_\W]+", f) if len(token) > 3}
    if target_tokens and target_tokens.intersection(feature_tokens) and any(token in f for token in ["amount", "status", "date", "flag", "score"]):
        return f"shares outcome-like terms with target {target}"
    return None


def _feature_type(name: str, series: pd.Series) -> str:
    if _is_bool_like(series):
        return "boolean"
    if _is_date_like(name, series):
        return "date"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def _coerce_bool_value(value: Any) -> float | None:
    if is_missing_like(value):
        return np.nan
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    text = str(value).strip().lower()
    if text in BOOLEAN_TRUE:
        return 1.0
    if text in BOOLEAN_FALSE:
        return 0.0
    return np.nan


def _coerce_for_features(df: pd.DataFrame, features: list[dict[str, Any]]) -> pd.DataFrame:
    output = df.copy()
    for feature in features:
        name = feature["feature_name"]
        if name not in output.columns:
            output[name] = np.nan
        if feature["type"] == "boolean":
            output[name] = output[name].map(_coerce_bool_value)
        elif feature["type"] == "numeric":
            output[name] = pd.to_numeric(output[name], errors="coerce")
        else:
            output[name] = output[name].astype("object")
    return output[[f["feature_name"] for f in features]]


def _task_type(y: pd.Series) -> str:
    if _is_bool_like(y) or y.dtype == "object" or str(y.dtype) == "category":
        return "classification"
    numeric = pd.to_numeric(y, errors="coerce").dropna()
    if numeric.empty:
        return "classification"
    unique = numeric.nunique(dropna=True)
    integer_like = bool(np.all(np.isclose(numeric, np.round(numeric))))
    # Numeric targets are regression by default. Only clearly discrete binary/small
    # integer labels should become classification; ratings and salaries stay regression.
    return "classification" if integer_like and unique <= 2 else "regression"


def _candidate_features(df: pd.DataFrame, target: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    rows = len(df)
    features: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    leakage: list[dict[str, str]] = []
    seen_signatures: dict[tuple, str] = {}

    for col in df.columns:
        series = df[col]
        reason = None
        leak = _leakage_reason(target, col)
        ftype = _feature_type(col, series)
        signature = tuple(series.fillna("__NA__").astype(str).head(500).tolist())

        if col == target:
            reason = "target column"
        elif _is_id_like(col, series, rows):
            reason = "id-like or mostly unique column"
        elif _is_near_constant(series, rows):
            reason = "constant or near-constant column"
        elif ftype == "date":
            reason = "raw date/date-part excluded to avoid redundant manual inputs"
        elif ftype == "categorical" and series.nunique(dropna=True) > min(80, max(20, rows * 0.2)):
            reason = "high-cardinality categorical/text column"
        elif signature in seen_signatures:
            reason = f"duplicate of {seen_signatures[signature]}"
        elif leak:
            reason = leak
            leakage.append({"column": col, "reason": leak, "action": "excluded_by_default"})

        if reason:
            excluded.append({"feature_name": col, "display_name": _display_name(col), "type": ftype, "required": False, "source": "original", "excluded_reason": reason})
            continue

        seen_signatures[signature] = col
        feature: dict[str, Any] = {
            "feature_name": col,
            "name": col,
            "display_name": _display_name(col),
            "type": ftype,
            "required": True,
            "source": "original",
            "excluded_reason": None,
        }
        if ftype == "categorical":
            feature["allowed_values"] = sorted([str(v) for v in clean_categorical_for_chart(series).dropna().unique()[:80]])
            feature["categories"] = feature["allowed_values"]
        elif ftype == "boolean":
            feature["allowed_values"] = ["true", "false"]
            feature["categories"] = ["true", "false"]
        else:
            feature["allowed_values"] = None
            feature["categories"] = None
        features.append(feature)

    return features, excluded, leakage


def _pipeline(features: list[dict[str, Any]], estimator: Any) -> Pipeline:
    numeric = [f["feature_name"] for f in features if f["type"] in {"numeric", "boolean"}]
    categorical = [f["feature_name"] for f in features if f["type"] == "categorical"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocessing", preprocessor), ("estimator", estimator)])


def _candidates(task: str, mode: str, rows: int, feature_count: int = 0) -> list[tuple[str, Any]]:
    if task == "classification":
        models = [
            ("Logistic Regression", LogisticRegression(max_iter=1200, random_state=RANDOM_STATE)),
            ("Random Forest Classifier", RandomForestClassifier(n_estimators=120 if mode != "fast" else 70, max_depth=14 if mode != "deep" else None, random_state=RANDOM_STATE, class_weight="balanced_subsample", n_jobs=-1)),
            ("Gradient Boosting Classifier", GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]
        if mode == "deep":
            models.append(("ExtraTrees Classifier", ExtraTreesClassifier(n_estimators=180, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1)))
        if rows <= 15000:
            models.append(("KNN Classifier", KNeighborsClassifier()))
    else:
        models = [
            ("Ridge Regression", Ridge(random_state=RANDOM_STATE)),
            ("Linear Regression", LinearRegression()),
            ("Random Forest Regressor", RandomForestRegressor(n_estimators=120 if mode != "fast" else 70, max_depth=14 if mode != "deep" else None, random_state=RANDOM_STATE, n_jobs=-1)),
            ("Gradient Boosting Regressor", GradientBoostingRegressor(random_state=RANDOM_STATE)),
        ]
        if mode == "deep":
            models.append(("ExtraTrees Regressor", ExtraTreesRegressor(n_estimators=180, random_state=RANDOM_STATE, n_jobs=-1)))
        if rows <= 15000:
            models.append(("KNN Regressor", KNeighborsRegressor()))
    return models[:3] if mode == "fast" else models[:5] if mode == "balanced" else models


def _classification_metrics(y_test: pd.Series, pred: np.ndarray, pipeline: Pipeline, X_test: pd.DataFrame, imbalanced: bool) -> tuple[float, dict[str, Any], str]:
    metrics: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "f1_weighted": round(float(f1_score(y_test, pred, average="weighted", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_test, pred, average="macro", zero_division=0)), 4),
        "precision_weighted": round(float(precision_score(y_test, pred, average="weighted", zero_division=0)), 4),
        "recall_weighted": round(float(recall_score(y_test, pred, average="weighted", zero_division=0)), 4),
    }
    try:
        if hasattr(pipeline[-1], "predict_proba") and y_test.nunique() == 2:
            proba = pipeline.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = round(float(roc_auc_score(y_test, proba)), 4)
    except Exception:
        metrics["roc_auc"] = None
    selected = "f1_weighted" if imbalanced else "accuracy"
    return float(metrics[selected]), metrics, selected


def _regression_metrics(y_test: pd.Series, pred: np.ndarray) -> tuple[float, dict[str, Any], str]:
    metrics = {
        "r2": round(float(r2_score(y_test, pred)), 4),
        "mae": round(float(mean_absolute_error(y_test, pred)), 4),
        "rmse": round(float(root_mean_squared_error(y_test, pred)), 4),
    }
    return float(metrics["r2"]), metrics, "r2"


def _baseline(task: str, y_train: pd.Series, y_test: pd.Series) -> dict[str, Any]:
    if task == "classification":
        majority = y_train.value_counts().idxmax()
        pred = np.array([majority] * len(y_test))
        return {
            "strategy": "majority_class",
            "prediction": str(majority),
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "f1_weighted": round(float(f1_score(y_test, pred, average="weighted", zero_division=0)), 4),
        }
    mean_value = float(pd.to_numeric(y_train, errors="coerce").mean())
    pred = np.array([mean_value] * len(y_test))
    return {
        "strategy": "mean_prediction",
        "prediction": round(mean_value, 4),
        "r2": round(float(r2_score(y_test, pred)), 4),
        "mae": round(float(mean_absolute_error(y_test, pred)), 4),
        "rmse": round(float(root_mean_squared_error(y_test, pred)), 4),
    }


def _job(job_id: str | None):
    if not job_id:
        return None
    from . import training_job_service as jobs

    return jobs


def _check_cancel(job_id: str | None) -> None:
    jobs = _job(job_id)
    if jobs and jobs.is_cancelled(job_id):
        raise RuntimeError("Training cancelled")


def train_model(dataset_id: str, target: str, mode: str = "balanced", job_id: str | None = None) -> dict[str, Any]:
    started = time.time()
    jobs = _job(job_id)
    if jobs:
        jobs.step(job_id, "Loading dataset", "running", 2)
        jobs.log(job_id, "info", f"Loading dataset {dataset_id}")
    df = read_frame(dataset_id).dropna(axis=0, how="all")
    if jobs:
        jobs.log(job_id, "info", f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
        jobs.log(job_id, "info", f"Target column: {target}; selected mode: {mode.title()}")
        jobs.step(job_id, "Loading dataset", "complete", 5)
    if target not in df.columns:
        raise ValueError("Target column not found")
    _check_cancel(job_id)

    if jobs:
        jobs.step(job_id, "Task detection", "running", 6)
    y = df[target]
    valid = y.notna()
    df = df[valid].copy()
    y = y[valid]
    if len(df) < 10 or y.nunique(dropna=True) < 2:
        raise ValueError("Not enough usable target data to train a model")

    task = _task_type(y)
    if jobs:
        jobs.log(job_id, "success", f"Detected {task} task for target {target}")
        jobs.step(job_id, "Task detection", "complete", 10)
        jobs.step(job_id, "Leakage check", "running", 11)
    features, excluded_columns, leakage_warnings = _candidate_features(df, target)
    if jobs:
        for warning in leakage_warnings[:12]:
            jobs.log(job_id, "warning", f"Possible leakage excluded: {warning['column']} ({warning['reason']})")
        jobs.step(job_id, "Leakage check", "complete", 15)
        jobs.step(job_id, "Feature filtering", "running", 16)
    if not features:
        raise ValueError("No suitable feature columns remain after excluding IDs, leakage, dates, constants, and high-cardinality columns")
    if jobs:
        jobs.log(job_id, "info", f"Included {len(features)} features; excluded {len(excluded_columns)} columns")
        for item in excluded_columns[:20]:
            jobs.log(job_id, "info", f"Excluded {item['feature_name']}: {item['excluded_reason']}")
        if len(excluded_columns) > 20:
            jobs.log(job_id, "info", f"...and {len(excluded_columns) - 20} more excluded columns")
        jobs.step(job_id, "Feature filtering", "complete", 22)
        jobs.step(job_id, "Feature schema", "running", 23)

    X = _coerce_for_features(df, features)
    if jobs:
        jobs.step(job_id, "Feature schema", "complete", 27)
        jobs.step(job_id, "Preprocessing setup", "running", 28)
    class_balance = None
    class_imbalance_warning = None
    imbalanced = False
    if task == "classification":
        counts = y.value_counts(dropna=False)
        class_balance = {str(k): int(v) for k, v in counts.items()}
        min_ratio = float(counts.min() / counts.sum())
        imbalanced = min_ratio < 0.2
        if imbalanced:
            class_imbalance_warning = "Target classes are imbalanced. Accuracy may be misleading."
            if jobs:
                jobs.log(job_id, "warning", class_imbalance_warning)

    stratify = y if task == "classification" and y.value_counts().min() >= 2 and y.nunique() < len(y) * 0.5 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=RANDOM_STATE, stratify=stratify)
    baseline = _baseline(task, y_train, y_test)
    if jobs:
        jobs.log(job_id, "info", f"Train/test split: {len(X_train)} train rows, {len(X_test)} test rows")
        jobs.log(job_id, "info", f"Baseline strategy: {baseline['strategy']}")
        jobs.step(job_id, "Preprocessing setup", "complete", 34)
        jobs.step(job_id, "Candidate selection", "running", 35)
    compared: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    best: dict[str, Any] | None = None

    cv_folds = 0 if mode == "fast" else 3 if mode == "balanced" else 5
    validation_method = "stratified train/test split" if stratify is not None else "train/test split"
    if cv_folds:
        validation_method += f" + {cv_folds}-fold cross-validation"

    candidate_models = _candidates(task, mode, len(X), len(features))
    if jobs:
        jobs.set_models(job_id, [name for name, _ in candidate_models])
        jobs.log(job_id, "info", f"Candidate models: {', '.join(name for name, _ in candidate_models)}")
        if mode == "balanced":
            jobs.log(job_id, "info", "ExtraTrees is reserved for Deep mode to avoid slow wide encoded training.")
        jobs.step(job_id, "Candidate selection", "complete", 40)
        jobs.step(job_id, "Model training", "running", 41)

    settings = get_settings()
    model_progress_span = 40
    model_count = max(len(candidate_models), 1)
    for index, (name, estimator) in enumerate(candidate_models, start=1):
        _check_cancel(job_id)
        model_started = time.time()
        try:
            if jobs:
                jobs.model_update(job_id, name, "running")
                jobs.set_progress(job_id, 40 + int((index - 1) / model_count * model_progress_span), f"Training {name} {index}/{model_count}")
                jobs.log(job_id, "info", f"Training {name} {index}/{model_count}...")
            pipeline = _pipeline(features, estimator)
            pipeline.fit(X_train, y_train)
            if settings.enable_model_timeouts and time.time() - model_started > settings.model_train_timeout_seconds:
                raise TimeoutError(f"{name} exceeded timeout of {settings.model_train_timeout_seconds}s")
            if jobs:
                jobs.set_progress(job_id, 40 + int((index - 0.5) / model_count * model_progress_span), f"Evaluating {name} {index}/{model_count}")
            pred = pipeline.predict(X_test)
            if task == "classification":
                score, metrics, selected_metric = _classification_metrics(y_test, pred, pipeline, X_test, imbalanced)
                cv_scoring = "f1_weighted" if imbalanced else "accuracy"
                splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE) if cv_folds and y.value_counts().min() >= cv_folds else None
            else:
                score, metrics, selected_metric = _regression_metrics(y_test, pred)
                cv_scoring = "r2"
                splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE) if cv_folds else None
            if splitter is not None:
                cv_scores = cross_val_score(pipeline, X, y, cv=splitter, scoring=cv_scoring)
                metrics["cv_scores"] = [round(float(v), 4) for v in cv_scores]
                metrics["cv_mean"] = round(float(np.mean(cv_scores)), 4)
                metrics["cv_std"] = round(float(np.std(cv_scores)), 4)
                score = float(np.mean(cv_scores))
            item = {
                "model_name": name,
                "metrics": metrics,
                "selected_metric": selected_metric,
                "selection_score": round(float(score), 4),
                "training_time_seconds": round(time.time() - model_started, 3),
                "status": "complete",
            }
            compared.append(item)
            if jobs:
                metric_bits = ", ".join(f"{k}={v}" for k, v in metrics.items() if isinstance(v, (int, float)) and k in {selected_metric, "r2", "mae", "rmse", "accuracy", "f1_weighted"})
                jobs.model_update(job_id, name, "complete", metrics)
                jobs.log(job_id, "success", f"{name} completed in {item['training_time_seconds']}s; {metric_bits}")
                jobs.set_progress(job_id, 40 + int(index / model_count * model_progress_span), f"Completed {name}")
            if best is None or score > best["score"]:
                best = {"score": score, "name": name, "pipeline": pipeline, "metrics": metrics, "selected_metric": selected_metric}
        except TimeoutError as exc:
            failed.append({"model_name": name, "error": str(exc), "status": "timeout"})
            if jobs:
                jobs.model_update(job_id, name, "timeout", error=str(exc))
                jobs.log(job_id, "warning", f"{name} timed out and was skipped")
        except Exception as exc:
            failed.append({"model_name": name, "error": str(exc), "status": "failed"})
            if jobs:
                jobs.model_update(job_id, name, "failed", error=str(exc))
                jobs.log(job_id, "error", f"{name} failed: {exc}")

    if best is None:
        raise ValueError("All candidate models failed to train")
    if jobs:
        jobs.step(job_id, "Model training", "complete", 82)
        jobs.step(job_id, "Model comparison", "running", 85)

    suspicious: list[str] = []
    metric_values = [v for v in best["metrics"].values() if isinstance(v, (int, float))]
    if any(v >= 0.99 for v in metric_values if v <= 1.0):
        suspicious.append("Suspiciously high score. This may indicate target leakage, duplicate rows, or an easy deterministic target.")

    baseline_metric = baseline.get(best["selected_metric"])
    beats_baseline = None
    if isinstance(baseline_metric, (int, float)):
        beats_baseline = float(best["metrics"].get(best["selected_metric"], best["score"])) > float(baseline_metric) + 0.02
    if jobs:
        jobs.log(job_id, "success", f"Selected best model: {best['name']}")
        if beats_baseline is False:
            jobs.log(job_id, "warning", "Selected model does not beat baseline meaningfully.")
        for msg in suspicious:
            jobs.log(job_id, "warning", msg)
        jobs.step(job_id, "Model comparison", "complete", 88)
        jobs.step(job_id, "Saving model bundle", "running", 90)

    model_id = new_id("model")
    feature_schema = {
        "target": target,
        "task_type": task,
        "features": features,
        "excluded_features": excluded_columns,
        "leakage_warnings": leakage_warnings,
    }
    save_model_artifact(dataset_id, model_id, "model.pkl", best["pipeline"])
    summary = {
        "dataset_id": dataset_id,
        "model_id": model_id,
        "target": target,
        "task_type": task,
        "selected_model": best["name"],
        "mode": mode,
        "metrics": best["metrics"],
        "selected_metric": best["selected_metric"],
        "baseline": baseline,
        "beats_baseline": beats_baseline,
        "compared_models": compared + failed,
        "feature_schema": feature_schema,
        "excluded_columns": excluded_columns,
        "leakage_warnings": leakage_warnings,
        "warnings": [w for w in [class_imbalance_warning, *suspicious] if w],
        "validation_method": validation_method,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "class_balance": class_balance,
        "random_state": RANDOM_STATE,
        "chat_prediction_enabled": has_llm(),
        "training_time_seconds": round(time.time() - started, 3),
        "rows_used": int(len(X)),
        "features_used": int(len(features)),
    }
    save_json(dataset_id, f"model_{model_id}", summary)
    latest = load_json(dataset_id, "profile") or {}
    latest["trained_model_id"] = model_id
    save_json(dataset_id, "profile", latest)
    _write_bundle_files(dataset_id, model_id, summary)
    if jobs:
        jobs.step(job_id, "Saving model bundle", "complete", 95)
        jobs.step(job_id, "Preparing prediction workspace", "running", 96)
        jobs.step(job_id, "Preparing prediction workspace", "complete", 99)
    return summary


def _write_bundle_files(dataset_id: str, model_id: str, summary: dict[str, Any]) -> None:
    directory = model_dir(dataset_id, model_id)
    (directory / "feature_schema.json").write_text(json.dumps(summary["feature_schema"], indent=2), encoding="utf-8")
    (directory / "metrics.json").write_text(json.dumps(summary["metrics"], indent=2), encoding="utf-8")
    (directory / "model_metadata.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    pd.DataFrame([{f["feature_name"]: "" for f in summary["feature_schema"]["features"]}]).to_csv(directory / "example_input.csv", index=False)
    (directory / "README.txt").write_text("Load model.pkl with joblib. Inputs must match feature_schema.json. The bundle includes preprocessing, estimator, metrics, exclusions, leakage warnings, and target metadata.", encoding="utf-8")


def _summary(dataset_id: str, model_id: str) -> dict[str, Any]:
    summary = load_json(dataset_id, f"model_{model_id}")
    if not summary:
        raise ValueError("Model not found")
    return summary


def _validate_values(summary: dict[str, Any], values: dict[str, Any]) -> tuple[pd.DataFrame | None, list[dict[str, str]]]:
    features = summary["feature_schema"]["features"]
    missing = []
    for feature in features:
        name = feature["feature_name"]
        if feature.get("required", True) and (name not in values or values[name] in [None, ""]):
            missing.append({"feature_name": name, "display_name": feature["display_name"], "type": feature["type"]})
    if missing:
        return None, missing
    return _coerce_for_features(pd.DataFrame([{f["feature_name"]: values.get(f["feature_name"]) for f in features}]), features), []


def predict(dataset_id: str, model_id: str, values: dict[str, Any]) -> dict[str, Any]:
    from .storage_service import load_model_artifact

    summary = _summary(dataset_id, model_id)
    frame, missing = _validate_values(summary, values)
    if missing:
        return {"status": "missing_features", "missing_features": missing}
    model = load_model_artifact(dataset_id, model_id, "model.pkl")
    pred = model.predict(frame)[0]
    result: dict[str, Any] = {"status": "complete", "prediction": pred.item() if hasattr(pred, "item") else pred}
    if hasattr(model[-1], "predict_proba"):
        probs = model.predict_proba(frame)[0]
        result["probabilities"] = {str(cls): round(float(prob), 4) for cls, prob in zip(model[-1].classes_, probs)}
    return result


def _heuristic_extract(message: str, features: list[dict[str, Any]]) -> dict[str, Any]:
    extracted: dict[str, Any] = {}
    lower = message.lower()
    for feature in features:
        name = feature["feature_name"]
        aliases = {name.lower(), feature["display_name"].lower(), name.lower().replace("_", " ")}
        if feature["type"] == "boolean":
            for alias in aliases:
                if re.search(rf"{re.escape(alias)}\s*(is|=|:)?\s*(true|yes|false|no|0|1)", lower):
                    match = re.search(rf"{re.escape(alias)}\s*(is|=|:)?\s*(true|yes|false|no|0|1)", lower)
                    extracted[name] = match.group(2)
        elif feature["type"] == "numeric":
            for alias in aliases:
                match = re.search(rf"{re.escape(alias)}\s*(is|=|:)?\s*\$?(-?\d+(?:\.\d+)?)%?", lower)
                if match:
                    extracted[name] = float(match.group(2))
                    break
        else:
            for value in feature.get("allowed_values") or []:
                if str(value).lower() in lower:
                    extracted[name] = value
                    break
    return extracted


def chat_predict(dataset_id: str, model_id: str, message: str, conversation_state: dict[str, Any] | None = None) -> dict[str, Any]:
    summary = _summary(dataset_id, model_id)
    features = summary["feature_schema"]["features"]
    state = dict((conversation_state or {}).get("values", conversation_state or {}))
    extracted: dict[str, Any] = {}

    if not has_llm():
        return {
            "status": "llm_required",
            "assistant_message": "Chat Prediction requires an OpenAI API key because it uses the LLM to extract feature values from natural language.",
            "conversation_state": {"values": state},
            "missing_features": [],
        }

    schema_prompt = [{"feature_name": f["feature_name"], "display_name": f["display_name"], "type": f["type"], "allowed_values": f.get("allowed_values")} for f in features]
    try:
        parsed = complete_json(
            [
                {"role": "system", "content": "Extract prediction feature values from natural language. Return strict JSON only: {\"values\": {...}}. Use feature_name keys exactly. Do not predict."},
                {"role": "user", "content": f"Feature schema: {json.dumps(schema_prompt)[:7000]}\nMessage: {message}"},
            ],
            max_tokens=700,
        )
        extracted = parsed.get("values", {}) if isinstance(parsed, dict) else {}
    except Exception:
        extracted = _heuristic_extract(message, features)

    for key, value in extracted.items():
        if key in {f["feature_name"] for f in features} and value not in [None, ""]:
            state[key] = value

    frame, missing = _validate_values(summary, state)
    if missing:
        names = ", ".join(item["display_name"] for item in missing[:10])
        more = " and a few more fields" if len(missing) > 10 else ""
        return {
            "status": "missing_features",
            "assistant_message": f"I still need {names}{more} before I can run this prediction.",
            "conversation_state": {"values": state},
            "extracted_values": extracted,
            "missing_features": missing,
        }

    result = predict(dataset_id, model_id, state)
    try:
        explanation = complete_text(
            [
                {"role": "system", "content": "Explain a sklearn model prediction briefly. Do not claim causality. Use the provided prediction and inputs only."},
                {"role": "user", "content": f"Target: {summary['target']}. Task: {summary['task_type']}. Prediction result: {json.dumps(result, default=str)}. Inputs: {json.dumps(state, default=str)[:5000]}"},
            ],
            max_tokens=260,
        )
    except Exception:
        explanation = f"The trained model predicts {result.get('prediction')} for {summary['target']} based on the supplied feature values."
    return {
        "status": "complete",
        "assistant_message": explanation,
        "conversation_state": {"values": state},
        "extracted_values": extracted,
        "prediction": result,
        "missing_features": [],
    }


def batch_predict(dataset_id: str, model_id: str, csv_file) -> Path:
    from .storage_service import load_model_artifact

    summary = _summary(dataset_id, model_id)
    features = summary["feature_schema"]["features"]
    required = [f["feature_name"] for f in features]
    frame = pd.read_csv(csv_file)
    missing = [f for f in required if f not in frame.columns]
    if missing:
        raise ValueError(f"Batch CSV is missing columns: {', '.join(missing)}")
    model = load_model_artifact(dataset_id, model_id, "model.pkl")
    output = frame.copy()
    output[f"predicted_{summary['target']}"] = model.predict(_coerce_for_features(frame, features))
    target = model_dir(dataset_id, model_id) / "batch_predictions.csv"
    output.to_csv(target, index=False)
    return target


def bundle_zip(dataset_id: str, model_id: str) -> Path:
    directory = model_dir(dataset_id, model_id)
    target = directory / "model_bundle.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in ["model.pkl", "feature_schema.json", "metrics.json", "model_metadata.json", "example_input.csv", "README.txt"]:
            zf.write(directory / name, arcname=name)
    return target
