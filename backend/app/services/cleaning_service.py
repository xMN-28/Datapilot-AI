from typing import Any

import pandas as pd

from .schema_service import clean_column_name


def clean_for_analysis(df: pd.DataFrame, schema: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    cleaned = df.copy()
    display_to_safe: dict[str, str] = {}
    seen: dict[str, int] = {}
    for col in cleaned.columns:
        base = clean_column_name(str(col))
        count = seen.get(base, 0)
        seen[base] = count + 1
        safe = base if count == 0 else f"{base}_{count + 1}"
        display_to_safe[str(col)] = safe
    cleaned = cleaned.rename(columns=display_to_safe)

    conversions: list[dict[str, str]] = []
    for original, safe in display_to_safe.items():
        detected = next((c for c in schema["columns"] if c["name"] == original), None)
        if not detected:
            continue
        series = cleaned[safe]
        if detected["inferred_type"] == "datetime":
            cleaned[safe] = pd.to_datetime(series, errors="coerce")
            conversions.append({"column": original, "conversion": "parsed datetime"})
        elif detected["inferred_type"] == "numeric" and not pd.api.types.is_numeric_dtype(series):
            normalized = series.astype(str).str.replace(r"[$,%]", "", regex=True).str.replace(",", "", regex=False)
            cleaned[safe] = pd.to_numeric(normalized, errors="coerce")
            conversions.append({"column": original, "conversion": "parsed numeric-like strings"})

    duplicate_rows = int(cleaned.duplicated().sum())
    missing_by_column = {str(k): int(v) for k, v in cleaned.isna().sum().items() if int(v) > 0}
    report = {
        "column_name_map": display_to_safe,
        "duplicate_rows": duplicate_rows,
        "missing_by_column": missing_by_column,
        "conversions": conversions,
        "notes": ["Cleaning is non-destructive; raw upload is preserved."],
    }
    return cleaned, report
