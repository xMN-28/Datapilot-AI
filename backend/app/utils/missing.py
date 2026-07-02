from typing import Any

import numpy as np
import pandas as pd


MISSING_LIKE_STRINGS = {"", "nan", "none", "null", "na", "n/a", "missing", "undefined"}


def is_missing_like(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip().lower() in MISSING_LIKE_STRINGS
    return False


def normalize_missing_values(series: pd.Series) -> pd.Series:
    def clean_value(value: Any) -> Any:
        if is_missing_like(value):
            return pd.NA
        if isinstance(value, str):
            stripped = value.strip()
            return pd.NA if stripped.lower() in MISSING_LIKE_STRINGS else stripped
        return value

    return series.astype("object").map(clean_value)


def categorical_counts_for_chart(series: pd.Series) -> tuple[pd.Series, int]:
    cleaned = normalize_missing_values(series)
    non_missing = cleaned.dropna()
    return non_missing.value_counts(), int(len(cleaned) - len(non_missing))


def clean_categorical_for_chart(series: pd.Series) -> pd.Series:
    return normalize_missing_values(series)
