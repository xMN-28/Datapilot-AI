from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from ..utils.missing import categorical_counts_for_chart, clean_categorical_for_chart, is_missing_like

BOOLEAN_STRINGS = {"true", "false", "yes", "no", "0", "1"}


def _safe_float(value: Any) -> float | None:
    try:
        if pd.isna(value) or np.isinf(value):
            return None
        return round(float(value), 4)
    except Exception:
        return None


def _is_boolean_like(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return True
    cleaned = clean_categorical_for_chart(series).dropna()
    values = set(cleaned.astype(str).str.strip().str.lower().unique())
    return bool(values) and values.issubset(BOOLEAN_STRINGS) and len(values) <= 2


def _schema_type_map(schema: dict[str, Any]) -> dict[str, str]:
    return {col["safe_name"]: col["inferred_type"] for col in schema.get("columns", [])}


def _bool_counts(series: pd.Series) -> dict[str, Any]:
    cleaned = clean_categorical_for_chart(series)
    normalized = cleaned.dropna().astype(str).str.strip().str.lower()
    true_count = int(normalized.isin({"true", "yes", "1"}).sum())
    false_count = int(normalized.isin({"false", "no", "0"}).sum())
    total = max(true_count + false_count, 1)
    missing_count = int(cleaned.isna().sum())
    return {
        "true_count": true_count,
        "false_count": false_count,
        "true_percentage": round(float(true_count / total * 100), 2),
        "false_percentage": round(float(false_count / total * 100), 2),
        "missing_count": missing_count,
        "missing_percentage": _safe_float(missing_count / max(len(series), 1) * 100),
        "unique_values": [str(v) for v in cleaned.dropna().unique().tolist()[:8] if not is_missing_like(v)],
    }


def _clean_category_series(series: pd.Series) -> pd.Series:
    return clean_categorical_for_chart(series).dropna()


def run_eda(df: pd.DataFrame, schema: dict[str, Any], cleaning_report: dict[str, Any]) -> dict[str, Any]:
    schema_types = _schema_type_map(schema)
    boolean_cols = [c for c in df.columns if schema_types.get(c) == "boolean" or _is_boolean_like(df[c])]
    numeric_cols = [
        c
        for c in df.select_dtypes(include=["number"]).columns
        if c not in boolean_cols and not pd.api.types.is_bool_dtype(df[c]) and schema_types.get(c) not in {"boolean", "id-like", "constant"}
    ]
    categorical_cols = [
        c
        for c in df.columns
        if c in boolean_cols or df[c].dtype == "object" or str(df[c].dtype) == "category"
        if not pd.api.types.is_datetime64_any_dtype(df[c]) and schema_types.get(c) not in {"id-like", "text-like"}
    ]
    datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    skipped: list[dict[str, str]] = []

    numeric: dict[str, Any] = {}
    for col in numeric_cols:
        try:
            s = df[col].dropna()
            q1 = s.quantile(0.25) if len(s) else None
            q3 = s.quantile(0.75) if len(s) else None
            iqr = q3 - q1 if q1 is not None and q3 is not None else 0
            outliers = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()) if iqr else 0
            numeric[col] = {
                "count": int(s.count()),
                "mean": _safe_float(s.mean()),
                "median": _safe_float(s.median()),
                "std": _safe_float(s.std()),
                "min": _safe_float(s.min()),
                "max": _safe_float(s.max()),
                "q1": _safe_float(q1),
                "q3": _safe_float(q3),
                "skewness": _safe_float(skew(s)) if len(s) > 2 else None,
                "kurtosis": _safe_float(kurtosis(s)) if len(s) > 3 else None,
                "missing_percentage": _safe_float(df[col].isna().mean() * 100),
                "outlier_count": outliers,
            }
        except Exception as exc:
            skipped.append({"column": str(col), "analysis": "numeric_eda", "error": str(exc), "status": "skipped"})

    boolean: dict[str, Any] = {}
    for col in boolean_cols:
        try:
            boolean[col] = _bool_counts(df[col])
        except Exception as exc:
            skipped.append({"column": str(col), "analysis": "boolean_eda", "error": str(exc), "status": "skipped"})

    categorical: dict[str, Any] = {}
    for col in categorical_cols:
        try:
            clean_values = _clean_category_series(df[col])
            counts, missing_count = categorical_counts_for_chart(df[col])
            vc = counts.head(12)
            total_non_missing = max(len(clean_values), 1)
            categorical[col] = {
                "unique_count": int(clean_values.nunique(dropna=True)),
                "top_values": [{"value": str(k), "count": int(v), "percentage": round(float(v / total_non_missing * 100), 2)} for k, v in vc.items()],
                "missing_count": missing_count,
                "missing_percentage": _safe_float(missing_count / max(len(df), 1) * 100),
                "missing_note": f"{col} has {missing_count} missing values excluded from this category summary." if missing_count else None,
                "cardinality_warning": bool(clean_values.nunique(dropna=True) > 50),
                "imbalance_warning": bool((vc.iloc[0] / total_non_missing) > 0.8) if len(vc) else False,
            }
        except Exception as exc:
            skipped.append({"column": str(col), "analysis": "categorical_eda", "error": str(exc), "status": "skipped"})

    datetimes: dict[str, Any] = {}
    for col in datetime_cols:
        try:
            s = df[col].dropna()
            span = s.max() - s.min() if len(s) else None
            datetimes[col] = {
                "min_date": str(s.min()) if len(s) else None,
                "max_date": str(s.max()) if len(s) else None,
                "time_span_days": int(span.days) if span is not None else None,
                "missing_percentage": _safe_float(df[col].isna().mean() * 100),
                "possible_features": ["year", "month", "quarter", "day_of_week", "is_weekend"],
            }
        except Exception as exc:
            skipped.append({"column": str(col), "analysis": "datetime_eda", "error": str(exc), "status": "skipped"})

    corr = []
    if len(numeric_cols) >= 2:
        try:
            matrix = df[numeric_cols].corr(numeric_only=True).round(4)
            for a in matrix.columns:
                for b in matrix.columns:
                    if a < b and pd.notna(matrix.loc[a, b]):
                        corr.append({"x": a, "y": b, "correlation": float(matrix.loc[a, b])})
            corr = sorted(corr, key=lambda item: abs(item["correlation"]), reverse=True)[:20]
        except Exception as exc:
            skipped.append({"column": "__numeric_columns__", "analysis": "correlation", "error": str(exc), "status": "skipped"})

    warnings = []
    if cleaning_report["duplicate_rows"]:
        warnings.append(f"{cleaning_report['duplicate_rows']} duplicate rows detected.")
    high_missing = [k for k, v in cleaning_report["missing_by_column"].items() if v / max(len(df), 1) > 0.4]
    if high_missing:
        warnings.append(f"High missingness in {', '.join(high_missing[:4])}.")

    return {
        "dataset": {
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "memory_size_bytes": int(df.memory_usage(deep=True).sum()),
            "missing_values_total": int(df.isna().sum().sum()),
            "duplicate_rows": cleaning_report["duplicate_rows"],
            "type_counts": schema["type_counts"],
            "data_quality_warnings": warnings,
        },
        "numeric": numeric,
        "boolean": boolean,
        "categorical": categorical,
        "datetime": datetimes,
        "relationships": {"numeric_correlations": corr},
        "skipped": skipped,
    }
