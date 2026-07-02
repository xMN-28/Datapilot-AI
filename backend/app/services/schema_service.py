import re
import warnings
from typing import Any

import pandas as pd

from ..utils.missing import clean_categorical_for_chart


def clean_column_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip().lower()).strip("_")
    return cleaned or "column"


def _looks_datetime(series: pd.Series) -> bool:
    sample = clean_categorical_for_chart(series).dropna().astype(str).head(80)
    if sample.empty:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().mean() >= 0.75


def _looks_numeric_string(series: pd.Series) -> bool:
    sample = clean_categorical_for_chart(series).dropna().astype(str).head(120)
    if sample.empty:
        return False
    normalized = sample.str.replace(r"[$,%]", "", regex=True).str.replace(",", "", regex=False)
    numeric = pd.to_numeric(normalized, errors="coerce")
    return numeric.notna().mean() >= 0.8


def infer_column(series: pd.Series, rows: int, name: str) -> dict[str, Any]:
    chart_clean = clean_categorical_for_chart(series)
    missing_pct = float(chart_clean.isna().mean() * 100) if rows else 0.0
    unique_count = int(chart_clean.nunique(dropna=True))
    sample_values = [str(v) for v in chart_clean.dropna().head(5).tolist()]
    warnings: list[str] = []
    lower_name = name.lower()

    if unique_count <= 1:
        inferred = "constant"
        warnings.append("Column has one or zero distinct values.")
    elif pd.api.types.is_bool_dtype(series) or set(chart_clean.dropna().astype(str).str.lower().unique()).issubset({"true", "false", "yes", "no", "0", "1"}):
        inferred = "boolean"
    elif pd.api.types.is_numeric_dtype(series) or _looks_numeric_string(series):
        inferred = "numeric"
        if unique_count == rows and ("id" in lower_name or unique_count > max(30, rows * 0.9)):
            inferred = "id-like"
    elif pd.api.types.is_datetime64_any_dtype(series) or _looks_datetime(series):
        inferred = "datetime"
    else:
        cardinality_ratio = unique_count / max(rows, 1)
        avg_len = chart_clean.dropna().astype(str).str.len().mean() if unique_count else 0
        if cardinality_ratio > 0.5 and unique_count > 30:
            inferred = "text-like" if avg_len > 24 else "high-cardinality categorical"
        else:
            inferred = "categorical"

    if missing_pct > 40:
        warnings.append("High missingness may reduce reliability.")
    if inferred in {"categorical", "high-cardinality categorical"} and unique_count > 50:
        warnings.append("High cardinality; grouped charts may be limited to top values.")

    target_candidate = inferred in {"numeric", "categorical", "boolean"} and unique_count > 1 and not inferred == "id-like"
    return {
        "name": name,
        "safe_name": clean_column_name(name),
        "inferred_type": inferred,
        "missing_percentage": round(missing_pct, 2),
        "unique_count": unique_count,
        "sample_values": sample_values,
        "warnings": warnings,
        "target_like_candidate": target_candidate,
    }


def detect_schema(df: pd.DataFrame) -> dict[str, Any]:
    rows = len(df)
    columns = [infer_column(df[col], rows, str(col)) for col in df.columns]
    counts: dict[str, int] = {}
    for col in columns:
        counts[col["inferred_type"]] = counts.get(col["inferred_type"], 0) + 1
    return {"columns": columns, "type_counts": counts}
