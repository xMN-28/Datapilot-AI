from typing import Any

import pandas as pd


LOW_VALUE_NUMERIC_NAMES = {"year", "month", "day", "quarter", "day_of_week", "week", "weekofyear"}


def _should_bin_numeric(name: str, series: pd.Series) -> bool:
    lower = name.lower()
    if lower in LOW_VALUE_NUMERIC_NAMES or lower.endswith(("_id", "_year", "_month", "_day", "_quarter", "_day_of_week", "_is_weekend")):
        return False
    if "id" in lower or "uuid" in lower or "code" in lower:
        return False
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series) and series.nunique(dropna=True) > 8


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    engineered = df.copy()
    features: list[dict[str, Any]] = []

    for col in [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])][:3]:
        base = col
        additions = {
            f"{base}_year": df[col].dt.year,
            f"{base}_month": df[col].dt.month,
            f"{base}_quarter": df[col].dt.quarter,
            f"{base}_day_of_week": df[col].dt.dayofweek,
            f"{base}_is_weekend": df[col].dt.dayofweek.isin([5, 6]).astype("Int64"),
        }
        for name, values in additions.items():
            engineered[name] = values
            features.append({"new_feature": name, "derived_from": [base], "method": "datetime extraction", "reason": "Useful for time-based grouping and trends."})

    numeric_cols = [c for c in df.columns if _should_bin_numeric(c, df[c])]
    for col in numeric_cols[:4]:
        try:
            binned = pd.qcut(df[col], q=4, duplicates="drop")
            if binned.nunique(dropna=True) > 1:
                name = f"{col}_bucket"
                engineered[name] = binned.astype(str)
                features.append({"new_feature": name, "derived_from": [col], "method": "quantile binning", "reason": "Useful for comparing numeric ranges as groups."})
        except Exception:
            continue

    return engineered, {"features": features, "count": len(features)}
