from typing import Any

from .chart_service import generate_chart_specs
from .cleaning_service import clean_for_analysis
from .eda_service import run_eda
from .feature_service import engineer_features
from .insight_service import generate_chart_insights
from .rag_service import build_artifacts
from .schema_service import detect_schema
from .storage_service import load_json, read_frame, save_json, utc_now


def profile_dataset(dataset_id: str, filename: str | None = None) -> dict[str, Any]:
    df = read_frame(dataset_id)
    schema = detect_schema(df)
    profile = {
        "dataset_id": dataset_id,
        "filename": filename or load_json(dataset_id, "profile") or "uploaded.csv",
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "schema": schema,
        "analysis_status": "Not started",
        "created_at": utc_now(),
        "analysis_log": [{"step": "CSV uploaded", "status": "complete"}, {"step": "Schema detected", "status": "complete"}],
    }
    save_json(dataset_id, "profile", profile)
    return profile


def run_analysis(dataset_id: str) -> dict[str, Any]:
    raw = read_frame(dataset_id)
    profile = load_json(dataset_id, "profile") or profile_dataset(dataset_id)
    schema = profile["schema"]
    cleaned, cleaning_report = clean_for_analysis(raw, schema)
    eda = run_eda(cleaned, schema, cleaning_report)
    engineered, feature_report = engineer_features(cleaned)
    charts = generate_chart_specs(engineered, eda)
    charts = generate_chart_insights(charts)
    analysis_log = [
        {"step": "Schema detected", "status": "complete"},
        {"step": "Cleaning report generated", "status": "complete"},
        {"step": "EDA completed", "status": "complete"},
        {"step": "Feature engineering completed", "status": "complete"},
        {"step": "Candidate charts generated", "status": "complete"},
        {"step": "Meaningful charts selected", "status": "complete"},
        {"step": "Insights generated", "status": "complete"},
        {"step": "RAG index created", "status": "complete"},
    ]
    analysis = {
        "dataset_id": dataset_id,
        "filename": profile.get("filename", "uploaded.csv"),
        "rows": int(len(raw)),
        "columns": int(len(raw.columns)),
        "schema": schema,
        "cleaning_report": cleaning_report,
        "eda_report": eda,
        "feature_engineering_report": feature_report,
        "chart_specs": charts,
        "insights": [{"chart_id": c["chart_id"], "insight": c.get("insight"), "confidence": c.get("confidence")} for c in charts],
        "analysis_log": analysis_log,
        "rag_index_status": "ready",
        "rag_chunks": build_artifacts({"schema": schema, "cleaning_report": cleaning_report, "eda_report": eda, "feature_engineering_report": feature_report, "chart_specs": charts, "analysis_log": analysis_log}),
        "created_at": profile.get("created_at", utc_now()),
        "updated_at": utc_now(),
    }
    profile["analysis_status"] = "Complete"
    profile["analysis_log"] = [{"step": "CSV uploaded", "status": "complete"}] + analysis_log
    save_json(dataset_id, "profile", profile)
    save_json(dataset_id, "analysis", analysis)
    return analysis
