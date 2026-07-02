import json
import re
from typing import Any

import numpy as np
import pandas as pd

from .llm_service import complete_json, complete_text, has_llm
from .rag_service import retrieve
from .storage_service import load_json, read_frame, save_json
from ..utils.missing import clean_categorical_for_chart, normalize_missing_values

LAST_CHAT_TRACES: dict[str, dict[str, Any]] = {}

ALIASES = {
    "previous semester gpa": ["previous semester gpa", "previous_semester_gpa", "prev semester gpa", "prior semester gpa", "gpa", "grades", "grade"],
    "semester gpa": ["previous semester gpa", "previous_semester_gpa", "semester gpa", "semester_gpa", "gpa", "grades", "grade"],
    "gpa": ["previous semester gpa", "previous_semester_gpa", "semester gpa", "semester_gpa", "gpa", "grades", "grade"],
    "grades": ["previous semester gpa", "previous_semester_gpa", "semester gpa", "semester_gpa", "gpa", "grades", "grade"],
    "sleep": ["sleep", "sleep hours", "sleep_hours_per_night", "sleep hours per night", "hours slept"],
    "sleep hours": ["sleep", "sleep hours", "sleep_hours_per_night", "sleep hours per night", "hours slept"],
    "productivity": ["productivity", "productivity score", "productivity_score", "performance"],
    "stress": ["stress", "stress level", "stress_level"],
    "screen time": ["screen time", "screen_time_hours", "screen hours", "phone time"],
    "social media": ["social media", "social_media_hours", "social hours"],
    "part time": ["part time", "part_time_job", "part time job", "job"],
    "performance": ["performance", "performance category", "performance_category"],
    "gender": ["gender", "sex"],
    "delivery time": ["delivery time", "delivery days", "delivery_days", "days to deliver"],
    "shipping": ["shipping", "shipping method", "shipping_method", "ship mode", "delivery method"],
    "returned": ["returned", "return", "is returned", "return status", "returned_flag"],
    "order amount": ["order amount", "order_amount", "amount", "revenue", "total"],
    "salary": ["salary", "salary midpoint", "salary_midpoint", "salary_midpoint_lpa", "salary_lpa"],
    "rating": ["rating", "company rating", "company_rating"],
    "experience": ["experience", "min experience", "max experience", "experience tier"],
}

AGGREGATION_WORDS = {
    "average": "mean",
    "avg": "mean",
    "mean": "mean",
    "median": "median",
    "total": "sum",
    "sum": "sum",
    "count": "count",
    "how many": "count",
    "minimum": "min",
    "min": "min",
    "maximum": "max",
    "max": "max",
    "std": "std",
    "standard deviation": "std",
}

QUESTION_STOPWORDS = {
    "what", "is", "the", "of", "for", "who", "with", "without", "and", "or", "by", "in", "to", "from", "used",
    "use", "has", "have", "having", "students", "people", "orders", "users", "rows", "records", "compare", "average",
    "avg", "mean", "median", "count", "how", "many", "which", "highest", "lowest", "between", "versus", "vs",
}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _tokens(text: str) -> set[str]:
    return set(_norm(text).split())


def _column_matches(columns: list[str], phrase: str) -> list[tuple[str, int]]:
    phrase_norm = _norm(phrase)
    if not phrase_norm:
        return []
    phrase_tokens = set(phrase_norm.split())
    expanded = set(phrase_tokens)
    for key, vals in ALIASES.items():
        if key in phrase_norm:
            expanded.update(token for val in vals for token in val.split())
            expanded.update(token for val in vals for token in _norm(val).split())
    matches: list[tuple[str, int]] = []
    for col in columns:
        col_norm = _norm(col)
        col_tokens = set(col_norm.split())
        score = len(expanded.intersection(col_tokens)) * 3
        if phrase_norm in col_norm or col_norm in phrase_norm:
            score += 6
        for alias_key, vals in ALIASES.items():
            if alias_key in phrase_norm:
                if any(_norm(val) == col_norm or set(_norm(val).split()).issubset(col_tokens) for val in vals):
                    score += 10
        for token in expanded:
            if token and token in col_norm:
                score += 1
        if score > 0:
            matches.append((col, score))
    return sorted(matches, key=lambda item: item[1], reverse=True)


def _column_lookup(columns: list[str], phrase: str) -> str | None:
    phrase_norm = _norm(phrase)
    matches = _column_matches(columns, phrase)
    if not matches:
        return None
    if len(matches) > 1 and matches[0][1] == matches[1][1]:
        return None
    if phrase_norm in ALIASES and matches[0][1] < 10:
        return None
    return matches[0][0]


def _specific_column(columns: list[str], question: str, concepts: list[str]) -> tuple[str | None, list[str]]:
    candidates: list[tuple[str, int]] = []
    for concept in concepts:
        candidates.extend(_column_matches(columns, concept))
    if not candidates:
        return None, []
    best_by_col: dict[str, int] = {}
    for col, score in candidates:
        best_by_col[col] = max(best_by_col.get(col, 0), score)
    ranked = sorted(best_by_col.items(), key=lambda item: item[1], reverse=True)
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None, [col for col, score in ranked if score == ranked[0][1]]
    return ranked[0][0], []


def _prefer_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized = {_norm(col): col for col in columns}
    for candidate in candidates:
        if _norm(candidate) in normalized:
            return normalized[_norm(candidate)]
    return None


def _schema_for_llm(df: pd.DataFrame) -> list[dict[str, Any]]:
    schema = []
    for col in df.columns:
        s = df[col]
        clean = normalize_missing_values(s)
        entry = {"column": col, "dtype": str(s.dtype), "sample_values": [str(v) for v in clean.dropna().head(5).tolist()]}
        if pd.api.types.is_numeric_dtype(s):
            entry["numeric"] = True
        else:
            valid = clean.dropna()
            entry["top_values"] = [str(v) for v in valid.value_counts().head(8).index.tolist()]
            entry["missing_count"] = int(clean.isna().sum())
        schema.append(entry)
    return schema


def _aggregation_function(question: str) -> str:
    q = question.lower()
    for phrase, fn in AGGREGATION_WORDS.items():
        if phrase in q:
            return fn
    return "mean"


def _is_numeric_column(df: pd.DataFrame, column: str) -> bool:
    return column in df.columns and pd.api.types.is_numeric_dtype(df[column])


def _categorical_values(df: pd.DataFrame, column: str, limit: int = 80) -> list[Any]:
    clean = normalize_missing_values(df[column]).dropna()
    if clean.nunique(dropna=True) > limit:
        return []
    return clean.astype("object").unique().tolist()


def _resolve_metric_column(df: pd.DataFrame, question: str, exclude: set[str] | None = None) -> str | None:
    columns = list(df.columns)
    exclude = exclude or set()
    q_norm = _norm(question)
    candidates: list[tuple[str, int]] = []
    for col in columns:
        if col in exclude:
            continue
        col_norm = _norm(col)
        col_tokens = set(col_norm.split())
        score = 0
        if col_norm in q_norm:
            score += 18
        score += len((set(q_norm.split()) - QUESTION_STOPWORDS).intersection(col_tokens)) * 5
        for alias, values in ALIASES.items():
            if alias in q_norm and any(_norm(value) == col_norm or set(_norm(value).split()).issubset(col_tokens) for value in values):
                score += 18
        if _is_numeric_column(df, col):
            score += 3
        elif any(word in q_norm for word in ["count", "how many"]):
            score += 1
        else:
            score -= 4
        if any(token in col_norm for token in ["id", "uuid"]):
            score -= 10
        if score > 0:
            candidates.append((col, score))
    if not candidates:
        return None
    ranked = sorted(candidates, key=lambda item: item[1], reverse=True)
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    return ranked[0][0]


def _resolve_group_column(df: pd.DataFrame, question: str) -> str | None:
    q = question.lower()
    if " by " in q:
        phrase = q.split(" by ", 1)[1]
        match = _column_lookup(list(df.columns), phrase)
        if match:
            return match
    q_norm = _norm(question)
    categorical_candidates = []
    for col in df.columns:
        if _is_numeric_column(df, col):
            continue
        col_norm = _norm(col)
        score = 0
        if col_norm in q_norm:
            score += 12
        score += len(set(col_norm.split()).intersection(set(q_norm.split()))) * 4
        for alias, values in ALIASES.items():
            if alias in q_norm and any(_norm(value) == col_norm or set(_norm(value).split()).issubset(set(col_norm.split())) for value in values):
                score += 12
        if score > 0:
            categorical_candidates.append((col, score))
    if categorical_candidates:
        return sorted(categorical_candidates, key=lambda item: item[1], reverse=True)[0][0]
    return _column_lookup(list(df.columns), question)


def _condition_column_from_context(df: pd.DataFrame, question: str, start: int, end: int) -> str | None:
    left = question[max(0, start - 80):start]
    right = question[end:min(len(question), end + 40)]
    left_norm = _norm(left)
    nearest: tuple[str, int] | None = None
    for col in df.columns:
        if not _is_numeric_column(df, col):
            continue
        phrases = {_norm(col), _norm(col).replace(" ", " ")}
        for alias, values in ALIASES.items():
            col_tokens = set(_norm(col).split())
            if any(_norm(value) == _norm(col) or set(_norm(value).split()).issubset(col_tokens) for value in values):
                phrases.add(_norm(alias))
                phrases.update(_norm(value) for value in values)
        for phrase in phrases:
            if not phrase:
                continue
            pos = left_norm.rfind(phrase)
            if pos >= 0 and (nearest is None or pos > nearest[1]):
                nearest = (col, pos)
    if nearest:
        return nearest[0]
    for snippet in [left, f"{left} {right}", question]:
        col = _column_lookup(list(df.columns), snippet)
        if col and _is_numeric_column(df, col):
            return col
    numeric_cols = [col for col in df.columns if _is_numeric_column(df, col)]
    ranked = [(col, len(set(_norm(question).split()).intersection(set(_norm(col).split())))) for col in numeric_cols]
    ranked = sorted([item for item in ranked if item[1] > 0], key=lambda item: item[1], reverse=True)
    return ranked[0][0] if ranked else None


def _extract_numeric_filters(df: pd.DataFrame, question: str) -> list[dict[str, Any]]:
    q = question.lower()
    patterns = [
        (r"<=\s*(\d+(?:\.\d+)?)", "<="),
        (r"<\s*(\d+(?:\.\d+)?)", "<"),
        (r">=\s*(\d+(?:\.\d+)?)", ">="),
        (r">\s*(\d+(?:\.\d+)?)", ">"),
        (r"(?:less than or equal to)\s*(\d+(?:\.\d+)?)", "<="),
        (r"(?:less than|under|below|fewer than)\s*(\d+(?:\.\d+)?)", "<"),
        (r"(\d+(?:\.\d+)?)\s*(?:\w+)?\s*or\s*(?:less|fewer)", "<="),
        (r"(?:at most|no more than)\s*(\d+(?:\.\d+)?)", "<="),
        (r"(?:greater than or equal to)\s*(\d+(?:\.\d+)?)", ">="),
        (r"(?:more than|greater than|above|over)\s*(\d+(?:\.\d+)?)", ">"),
        (r"(\d+(?:\.\d+)?)\s*(?:\w+)?\s*or more", ">="),
        (r"(?:at least|minimum of)\s*(\d+(?:\.\d+)?)", ">="),
        (r"between\s*(\d+(?:\.\d+)?)\s*and\s*(\d+(?:\.\d+)?)", "between"),
    ]
    filters: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for pattern, operator in patterns:
        for match in re.finditer(pattern, q):
            col = _condition_column_from_context(df, question, match.start(), match.end())
            if not col:
                continue
            value: Any = [float(match.group(1)), float(match.group(2))] if operator == "between" else float(match.group(1))
            key = (col, operator, str(value))
            if key in seen:
                continue
            seen.add(key)
            filters.append({"column": col, "operator": operator, "value": value, "source_text": match.group(0), "label": f"{col} {operator} {value}"})
    return filters


def _extract_categorical_filters(df: pd.DataFrame, question: str) -> list[dict[str, Any]]:
    q_norm = _norm(question)
    filters: list[dict[str, Any]] = []
    used_cols: set[str] = set()
    for col in df.columns:
        values = _categorical_values(df, col)
        if not values:
            continue
        col_norm = _norm(col)
        for value in values:
            value_norm = _norm(value)
            if not value_norm or len(value_norm) < 2:
                continue
            if value_norm in q_norm and col not in used_cols:
                filters.append({"column": col, "operator": "==", "value": value, "source_text": str(value)})
                used_cols.add(col)
                break
        if col in used_cols:
            continue
        if any(token in q_norm for token in ["returned", "return"]) and "return" in col_norm:
            truthy = [v for v in values if _norm(v) in {"yes", "true", "1", "returned"}]
            if truthy:
                filters.append({"column": col, "operator": "in", "value": truthy, "source_text": "returned"})
                used_cols.add(col)
        if ("without" in q_norm or "not " in q_norm) and any(token in q_norm for token in col_norm.split()):
            falsy = [v for v in values if _norm(v) in {"no", "false", "0"}]
            if falsy and col not in used_cols:
                filters.append({"column": col, "operator": "in", "value": falsy, "source_text": "without/not"})
                used_cols.add(col)
    return filters


def _extract_filters(df: pd.DataFrame, question: str) -> list[dict[str, Any]]:
    filters = _extract_numeric_filters(df, question)
    existing = {(f["column"], f["operator"], str(f["value"])) for f in filters}
    for filt in _extract_categorical_filters(df, question):
        key = (filt["column"], filt["operator"], str(filt["value"]))
        if key not in existing:
            filters.append(filt)
            existing.add(key)
    return filters


def _categorical_group_specs(df: pd.DataFrame, question: str) -> list[dict[str, Any]]:
    q_norm = _norm(question)
    for col in df.columns:
        values = _categorical_values(df, col)
        matched = []
        for value in values:
            value_norm = _norm(value)
            if value_norm and value_norm in q_norm:
                matched.append((value, q_norm.find(value_norm)))
        if len(matched) >= 2:
            matched = sorted(matched, key=lambda item: item[1])
            return [{"label": f"{col} = {value}", "filters": [{"column": col, "operator": "==", "value": value}]} for value, _ in matched[:2]]
        col_norm = _norm(col)
        if ("with" in q_norm and "without" in q_norm) and any(token in q_norm for token in col_norm.split()):
            truthy = [v for v in values if _norm(v) in {"yes", "true", "1"}]
            falsy = [v for v in values if _norm(v) in {"no", "false", "0"}]
            if truthy and falsy:
                return [
                    {"label": f"{col} = true", "filters": [{"column": col, "operator": "in", "value": truthy}]},
                    {"label": f"{col} = false", "filters": [{"column": col, "operator": "in", "value": falsy}]},
                ]
    return []


def _mentioned_numeric_columns(df: pd.DataFrame, question: str) -> list[str]:
    q_norm = _norm(question)
    scored = []
    for col in df.columns:
        if not _is_numeric_column(df, col):
            continue
        col_norm = _norm(col)
        score = 0
        if col_norm in q_norm:
            score += 10
        score += len(set(col_norm.split()).intersection(set(q_norm.split()))) * 3
        for alias, values in ALIASES.items():
            if alias in q_norm and any(_norm(value) == col_norm or set(_norm(value).split()).issubset(set(col_norm.split())) for value in values):
                score += 12
        if score > 0:
            scored.append((col, score))
    return [col for col, _ in sorted(scored, key=lambda item: item[1], reverse=True)]


def _generic_compare_plan(df: pd.DataFrame, question: str, agg: str) -> dict[str, Any]:
    numeric_groups = [{"label": item.get("label", f"{item['column']} {item['operator']} {item['value']}"), "filters": [{k: v for k, v in item.items() if k not in {"label", "source_text"}}]} for item in _extract_numeric_filters(df, question)]
    categorical_groups = _categorical_group_specs(df, question)
    groups = numeric_groups if len(numeric_groups) >= 2 else categorical_groups
    excluded = {filt["column"] for group in groups for filt in group.get("filters", [])}
    value_col = _resolve_metric_column(df, question, exclude=excluded)
    if value_col and len(groups) >= 2:
        return {"tool": "compare_groups", "value_column": value_col, "aggregation": agg, "groups": groups[:2]}
    return {"tool": "compare_groups", "status": "needs_clarification", "reason": "Could not identify the comparison metric and two groups."}


def _generic_filtered_plan(df: pd.DataFrame, question: str, agg: str) -> dict[str, Any]:
    filters = _extract_filters(df, question)
    excluded = {filt["column"] for filt in filters}
    value_col = _resolve_metric_column(df, question, exclude=excluded)
    if agg == "count" and not value_col:
        value_col = next((col for col in df.columns if col not in excluded), df.columns[0])
    if value_col and (_has_filter_condition(question) or filters):
        if not filters:
            return {"tool": "filtered_aggregation", "status": "needs_clarification", "reason": "Could not determine the filter condition."}
        return {"tool": "filtered_aggregation", "filters": filters, "aggregation": {"column": value_col, "function": agg}}
    if value_col:
        return {"tool": "filtered_aggregation", "filters": [], "aggregation": {"column": value_col, "function": agg}}
    return {"tool": "filtered_aggregation", "status": "needs_clarification", "reason": "Could not identify the aggregation column."}


def _generic_groupby_plan(df: pd.DataFrame, question: str, agg: str) -> dict[str, Any] | None:
    group_col = _resolve_group_column(df, question)
    value_col = _resolve_metric_column(df, question, exclude={group_col} if group_col else set())
    if group_col and value_col:
        return {"tool": "groupby_aggregation", "group_by": group_col, "value_column": value_col, "aggregation": agg}
    return None


def _generic_correlation_plan(df: pd.DataFrame, question: str) -> dict[str, Any] | None:
    cols = _mentioned_numeric_columns(df, question)
    if len(cols) >= 2:
        return {"tool": "correlation_query", "x_column": cols[0], "y_column": cols[1], "method": "pearson"}
    return None


def _classify_intent(question: str) -> dict[str, Any]:
    q = question.lower()
    if any(term in q for term in ["chart", "dashboard", "visual", "pattern"]):
        return {"intent": "chart_lookup", "confidence": 0.78, "reason": "Question asks about dashboard or chart context."}
    if any(term in q for term in ["column", "missing", "rows", "dataset"]):
        return {"intent": "dataset_metadata", "confidence": 0.72, "reason": "Question asks about dataset structure."}
    if any(term in q for term in ["correlation", "correlate", "related", "relationship", "relate", "affect", "higher", "lower"]):
        return {"intent": "correlation_query", "confidence": 0.82, "reason": "Question asks about a relationship between variables."}
    if _is_compare_question(question):
        return {"intent": "compare_groups", "confidence": 0.86, "reason": "Question asks for a comparison between groups."}
    if " by " in q or q.startswith("which "):
        return {"intent": "groupby_aggregation", "confidence": 0.75, "reason": "Question asks for grouped aggregation or ranking."}
    if any(term in q for term in AGGREGATION_WORDS) or _has_filter_condition(question):
        return {"intent": "filtered_aggregation", "confidence": 0.76, "reason": "Question asks for an aggregate or filtered metric."}
    return {"intent": "unsupported", "confidence": 0.3, "reason": "No computable analytical intent detected."}


def _apply_filter(df: pd.DataFrame, filt: dict[str, Any]) -> tuple[pd.Series, dict[str, Any]]:
    col = filt["column"]
    op = filt.get("operator", "==")
    value = filt.get("value")
    s = df[col]
    if op in {"<", "<=", ">", ">=", "between"}:
        numeric = pd.to_numeric(s, errors="coerce")
        if op == "<":
            mask = numeric < float(value)
        elif op == "<=":
            mask = numeric <= float(value)
        elif op == ">":
            mask = numeric > float(value)
        elif op == ">=":
            mask = numeric >= float(value)
        else:
            lo, hi = value
            mask = numeric.between(float(lo), float(hi))
    elif op == "contains":
        clean = clean_categorical_for_chart(s)
        mask = clean.astype("string").str.contains(str(value), case=False, na=False)
    elif op == "in":
        raw_values = value if isinstance(value, list) else [value]
        values = {str(v).strip().lower() for v in raw_values}
        clean = clean_categorical_for_chart(s)
        mask = clean.astype("string").str.strip().str.lower().isin(values)
    elif op == "!=":
        clean = clean_categorical_for_chart(s)
        mask = clean.notna() & (clean.astype("string").str.strip().str.lower() != str(value).strip().lower())
    else:
        clean = clean_categorical_for_chart(s)
        mask = clean.notna() & (clean.astype("string").str.strip().str.lower() == str(value).strip().lower())
    return mask.fillna(False), {"column": col, "operator": op, "value": value}


def _aggregate(series: pd.Series, fn: str) -> Any:
    fn = fn.lower()
    if fn == "count":
        return int(series.dropna().shape[0])
    numeric = pd.to_numeric(series, errors="coerce")
    if fn == "mean":
        return float(numeric.mean())
    if fn == "median":
        return float(numeric.median())
    if fn == "sum":
        return float(numeric.sum())
    if fn == "min":
        return float(numeric.min())
    if fn == "max":
        return float(numeric.max())
    if fn == "std":
        return float(numeric.std())
    raise ValueError(f"Unsupported aggregation {fn}")


def filtered_aggregation(df: pd.DataFrame, filters: list[dict[str, Any]], aggregation: dict[str, Any]) -> dict[str, Any]:
    column = aggregation["column"]
    if column not in df.columns:
        raise ValueError(f"Aggregation column not found: {column}")
    for filt in filters:
        if filt.get("column") not in df.columns:
            raise ValueError(f"Filter column not found: {filt.get('column')}")
        if filt.get("operator", "==") not in {"==", "!=", "<", "<=", ">", ">=", "contains", "in", "between"}:
            raise ValueError(f"Unsupported filter operator: {filt.get('operator')}")
    mask = pd.Series(True, index=df.index)
    applied = []
    for filt in filters:
        current, info = _apply_filter(df, filt)
        mask &= current
        applied.append(info)
    matched = df[mask]
    fn = aggregation.get("function", "mean")
    if len(matched) == 0:
        return {"tool": "filtered_aggregation", "status": "no_matches", "matched_rows": 0, "result": None, "aggregation": fn, "column": column, "filters_applied": applied}
    result = _aggregate(matched[column], fn)
    response = {"tool": "filtered_aggregation", "status": "complete", "matched_rows": int(len(matched)), "result": round(result, 4) if isinstance(result, float) and not np.isnan(result) else result, "aggregation": fn, "column": column, "filters_applied": applied}
    if fn in {"mean", "median"}:
        overall = _aggregate(df[column], fn)
        if isinstance(overall, float) and not np.isnan(overall):
            response["overall_baseline"] = round(overall, 4)
            if isinstance(response["result"], (int, float)):
                response["difference_from_overall"] = round(float(response["result"]) - float(response["overall_baseline"]), 4)
        complement = df[~mask]
        if len(complement):
            comp = _aggregate(complement[column], fn)
            if isinstance(comp, float) and not np.isnan(comp):
                response["comparison_group"] = {"label": "Rows not matching the filter", "matched_rows": int(len(complement)), "result": round(comp, 4), "difference": round(float(response["result"]) - float(comp), 4) if isinstance(response["result"], (int, float)) else None}
    return response


def groupby_aggregation(df: pd.DataFrame, group_by: str, value_column: str, aggregation: str) -> dict[str, Any]:
    clean = df[[group_by, value_column]].copy()
    clean[group_by] = clean_categorical_for_chart(clean[group_by])
    clean = clean.dropna(subset=[group_by])
    if aggregation == "count":
        grouped = clean.groupby(group_by)[value_column].count().sort_values(ascending=False).head(20)
    else:
        clean[value_column] = pd.to_numeric(clean[value_column], errors="coerce")
        grouped = getattr(clean.groupby(group_by)[value_column], aggregation)().sort_values(ascending=False).head(20)
    return {"tool": "groupby_aggregation", "group_by": group_by, "value_column": value_column, "aggregation": aggregation, "matched_rows": int(len(clean)), "result": [{"group": str(k), "value": round(float(v), 4)} for k, v in grouped.items() if pd.notna(v)]}


def compare_groups(df: pd.DataFrame, value_column: str, aggregation: str, groups: list[dict[str, Any]]) -> dict[str, Any]:
    if len(groups) < 2:
        raise ValueError("compare_groups requires at least two groups")
    if value_column not in df.columns:
        raise ValueError(f"Value column not found: {value_column}")
    results = []
    for group in groups:
        mask = pd.Series(True, index=df.index)
        applied = []
        for filt in group.get("filters", []):
            current, info = _apply_filter(df, filt)
            mask &= current
            applied.append(info)
        subset = df[mask]
        result = _aggregate(subset[value_column], aggregation)
        results.append(
            {
                "label": group.get("label") or f"Group {len(results) + 1}",
                "filters": applied,
                "matched_rows": int(len(subset)),
                "result": round(float(result), 4) if isinstance(result, float) and not np.isnan(result) else result,
                "sample_size_warning": int(len(subset)) < 30,
            }
        )
    difference = None
    if len(results) >= 2 and all(isinstance(item.get("result"), (int, float)) for item in results[:2]):
        difference = round(float(results[0]["result"]) - float(results[1]["result"]), 4)
    return {"tool": "compare_groups", "status": "complete", "value_column": value_column, "aggregation": aggregation, "groups": results, "difference": difference}


def correlation_query(df: pd.DataFrame, x_column: str, y_column: str, method: str = "pearson") -> dict[str, Any]:
    data = df[[x_column, y_column]].apply(pd.to_numeric, errors="coerce").dropna()
    corr = data[x_column].corr(data[y_column], method=method)
    return {"tool": "correlation_query", "x_column": x_column, "y_column": y_column, "method": method, "matched_rows": int(len(data)), "correlation": round(float(corr), 4) if pd.notna(corr) else None}


def _target_value_column(columns: list[str], question: str) -> str | None:
    q = question.lower()
    if "previous semester gpa" in q or "prev semester gpa" in q or "prior semester gpa" in q:
        return _prefer_column(columns, ["previous_semester_gpa", "Previous_Semester_GPA"]) or _specific_column(columns, question, ["previous semester gpa", "gpa", "grades"])[0]
    if any(term in q for term in ["semester gpa", " gpa", "gpa", "grade"]):
        preferred = _prefer_column(columns, ["previous_semester_gpa", "Previous_Semester_GPA", "semester_gpa", "Semester_GPA"])
        if preferred:
            return preferred
        return _specific_column(columns, question, ["semester gpa", "gpa", "grades"])[0]
    if "stress" in q:
        return _specific_column(columns, question, ["stress"])[0]
    if "screen time" in q:
        return _specific_column(columns, question, ["screen time"])[0]
    if "social media" in q:
        return _specific_column(columns, question, ["social media"])[0]
    if any(term in q for term in ["productive", "productivity", "performance"]):
        return _specific_column(columns, question, ["productivity"])[0]
    return _column_lookup(columns, question)


def _is_compare_question(question: str) -> bool:
    q = question.lower()
    explicit = any(token in q for token in ["compare", " versus ", " vs ", "difference between", " with students who "])
    comparator_count = sum(1 for token in ["less than", "under", "below", "more than", "greater than", "above", "at least", "at most", " or less", " or more", "<", ">"] if token in q)
    return explicit or comparator_count >= 2


def _has_filter_condition(question: str) -> bool:
    q = question.lower()
    return any(
        token in q
        for token in [
            "less than",
            "below",
            "under",
            "fewer than",
            "or less",
            "or fewer",
            "at most",
            "less than or equal to",
            "more than",
            "above",
            "greater than",
            "or more",
            "at least",
            "greater than or equal to",
            "<",
            ">",
        ]
    )


def _numeric_filter_specs(question: str, columns: list[str]) -> tuple[str | None, list[dict[str, Any]]]:
    q = question.lower()
    filter_col = _specific_column(columns, question, ["sleep", "sleep hours"])[0]
    if not filter_col:
        return None, []
    patterns = [
        (r"<=\s*(\d+(?:\.\d+)?)", "<=", "{col} <= {value:g} hours"),
        (r"<\s*(\d+(?:\.\d+)?)", "<", "{col} < {value:g} hours"),
        (r">=\s*(\d+(?:\.\d+)?)", ">=", "{col} >= {value:g} hours"),
        (r">\s*(\d+(?:\.\d+)?)", ">", "{col} > {value:g} hours"),
        (r"(?:less than or equal to)\s*(\d+(?:\.\d+)?)", "<=", "{col} <= {value:g} hours"),
        (r"(?:less than|under|below|fewer than)\s*(\d+(?:\.\d+)?)", "<", "{col} < {value:g} hours"),
        (r"(\d+(?:\.\d+)?)\s*(?:hours?)?\s*or\s*(?:less|fewer)", "<=", "{col} <= {value:g} hours"),
        (r"(?:at most|no more than)\s*(\d+(?:\.\d+)?)", "<=", "{col} <= {value:g} hours"),
        (r"(?:greater than or equal to)\s*(\d+(?:\.\d+)?)", ">=", "{col} >= {value:g} hours"),
        (r"(?:more than|greater than|above|over)\s*(\d+(?:\.\d+)?)", ">", "{col} > {value:g} hours"),
        (r"(\d+(?:\.\d+)?)\s*(?:hours?)?\s*or more", ">=", "{col} >= {value:g} hours"),
        (r"(?:at least|minimum of)\s*(\d+(?:\.\d+)?)", ">=", "{col} >= {value:g} hours"),
    ]
    filters: list[dict[str, Any]] = []
    seen: set[tuple[str, float]] = set()
    for pattern, operator, label_template in patterns:
        for match in re.finditer(pattern, q):
            value = float(match.group(1))
            key = (operator, value)
            if key in seen:
                continue
            seen.add(key)
            filters.append(
                {
                    "label": label_template.format(col="Sleep", value=value),
                    "filter": {"column": filter_col, "operator": operator, "value": value},
                }
            )
    return filter_col, filters


def _numeric_group_specs(question: str, columns: list[str]) -> tuple[str | None, list[dict[str, Any]]]:
    filter_col, filter_specs = _numeric_filter_specs(question, columns)
    groups = [{"label": item["label"], "filters": [item["filter"]]} for item in filter_specs]
    return filter_col, groups


def _categorical_compare_plan(question: str, columns: list[str], value_col: str, agg: str) -> dict[str, Any] | None:
    q = question.lower()
    if "part" in q and "job" in q and (any(token in q for token in ["true", "false", "yes", "no"]) or ("with" in q and "without" in q)):
        col = _specific_column(columns, question, ["part time"])[0]
        if col:
            return {
                "tool": "compare_groups",
                "value_column": value_col,
                "aggregation": agg,
                "groups": [
                    {"label": "Part-time job = true", "filters": [{"column": col, "operator": "in", "value": ["true", "yes", "1"]}]},
                    {"label": "Part-time job = false", "filters": [{"column": col, "operator": "in", "value": ["false", "no", "0"]}]},
                ],
            }
    if "performance" in q and any(token in q for token in ["high", "low"]):
        col = _specific_column(columns, question, ["performance"])[0]
        if col:
            return {
                "tool": "compare_groups",
                "value_column": value_col,
                "aggregation": agg,
                "groups": [
                    {"label": "Performance = High", "filters": [{"column": col, "operator": "==", "value": "High"}]},
                    {"label": "Performance = Low", "filters": [{"column": col, "operator": "==", "value": "Low"}]},
                ],
            }
    return None


def _resolve_column(columns: list[str], requested: Any) -> str | None:
    if requested in columns:
        return str(requested)
    return _column_lookup(columns, str(requested))


def _canonicalize_plan(plan: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    tool = plan.get("tool")
    clean_plan = dict(plan)
    if tool == "filtered_aggregation":
        aggregation = dict(clean_plan.get("aggregation", {}))
        if aggregation.get("column"):
            aggregation["column"] = _resolve_column(columns, aggregation["column"]) or aggregation["column"]
        clean_plan["aggregation"] = aggregation
        clean_plan["filters"] = [_canonical_filter(filt, columns) for filt in clean_plan.get("filters", [])]
    elif tool == "groupby_aggregation":
        for key in ["group_by", "value_column"]:
            if clean_plan.get(key):
                clean_plan[key] = _resolve_column(columns, clean_plan[key]) or clean_plan[key]
    elif tool == "correlation_query":
        for key in ["x_column", "y_column"]:
            if clean_plan.get(key):
                clean_plan[key] = _resolve_column(columns, clean_plan[key]) or clean_plan[key]
    elif tool == "compare_groups":
        if "groups" not in clean_plan:
            groups = []
            if clean_plan.get("group_a_filter"):
                groups.append({"label": "Group A", "filters": [clean_plan["group_a_filter"]]})
            if clean_plan.get("group_b_filter"):
                groups.append({"label": "Group B", "filters": [clean_plan["group_b_filter"]]})
            if clean_plan.get("group_by_plan") and isinstance(clean_plan["group_by_plan"], list):
                groups.extend(clean_plan["group_by_plan"])
            clean_plan["groups"] = groups
        if clean_plan.get("value_column"):
            clean_plan["value_column"] = _resolve_column(columns, clean_plan["value_column"]) or clean_plan["value_column"]
        clean_plan["groups"] = [
            {
                **group,
                "filters": [_canonical_filter(filt, columns) for filt in group.get("filters", [])],
            }
            for group in clean_plan.get("groups", [])
        ]
    return clean_plan


def _canonical_filter(filt: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    normalized = dict(filt)
    if normalized.get("column"):
        normalized["column"] = _resolve_column(columns, normalized["column"]) or normalized["column"]
    return normalized


def _plan_missing_columns(plan: dict[str, Any], columns: list[str]) -> list[str]:
    missing: list[str] = []
    tool = plan.get("tool")
    if tool == "filtered_aggregation":
        col = plan.get("aggregation", {}).get("column")
        if col and col not in columns:
            missing.append(str(col))
    elif tool == "groupby_aggregation":
        for key in ["group_by", "value_column"]:
            if plan.get(key) and plan[key] not in columns:
                missing.append(str(plan[key]))
    elif tool == "compare_groups":
        if plan.get("value_column") and plan["value_column"] not in columns:
            missing.append(str(plan["value_column"]))
        for group in plan.get("groups", []):
            for filt in group.get("filters", []):
                if filt.get("column") and filt["column"] not in columns:
                    missing.append(str(filt["column"]))
    elif tool == "correlation_query":
        for key in ["x_column", "y_column"]:
            if plan.get(key) and plan[key] not in columns:
                missing.append(str(plan[key]))
    return list(dict.fromkeys(missing))


def _missing_column_message(question: str, columns: list[str]) -> str | None:
    q = question.lower()
    if any(term in q for term in ["semester gpa", " gpa", "gpa", "grade"]):
        if not _specific_column(columns, question, ["semester gpa", "gpa", "grades"])[0]:
            return "I could not find a semester GPA column in this dataset."
    if "sleep" in q:
        if not _specific_column(columns, question, ["sleep", "sleep hours"])[0]:
            return "I could not find a sleep-hours column in this dataset."
    return None


def _heuristic_plan(question: str, df_or_columns: pd.DataFrame | list[str]) -> dict[str, Any] | None:
    df = df_or_columns if isinstance(df_or_columns, pd.DataFrame) else None
    columns = list(df.columns) if df is not None else list(df_or_columns)
    q = question.lower()
    agg = _aggregation_function(question)
    if df is not None:
        classified = _classify_intent(question)
        intent = classified["intent"]
        if intent == "compare_groups":
            return _generic_compare_plan(df, question, agg)
        if intent == "filtered_aggregation":
            return _generic_filtered_plan(df, question, agg)
        if intent == "groupby_aggregation":
            plan = _generic_groupby_plan(df, question, agg)
            if plan:
                return plan
        if intent == "correlation_query":
            plan = _generic_correlation_plan(df, question)
            if plan:
                return plan
        if intent in {"chart_lookup", "dataset_metadata"}:
            return {"tool": "chart_lookup"} if intent == "chart_lookup" else None
    agg = "mean" if any(w in q for w in ["average", "avg", "mean", "how productive"]) else "median" if "median" in q else "count" if any(w in q for w in ["how many", "count"]) else None
    if _is_compare_question(question):
        value_col = _target_value_column(columns, question)
        if value_col:
            categorical_plan = _categorical_compare_plan(question, columns, value_col, agg or "mean")
            if categorical_plan:
                return categorical_plan
        filter_col, numeric_groups = _numeric_group_specs(question, columns)
        if value_col and filter_col and len(numeric_groups) >= 2:
            return {
                "tool": "compare_groups",
                "value_column": value_col,
                "aggregation": agg or "mean",
                "groups": numeric_groups[:2],
            }
        return {"tool": "compare_groups", "status": "needs_clarification", "reason": "Could not identify two comparison groups from the question."}
    if any(w in q for w in ["correlat", "related", "relationship"]):
        mentioned = [_column_lookup(columns, token) for token in re.split(r"\band\b|,|\\?|with|to", question)]
        mentioned = [m for m in mentioned if m]
        if len(set(mentioned)) >= 2:
            x, y = list(dict.fromkeys(mentioned))[:2]
            return {"tool": "correlation_query", "x_column": x, "y_column": y, "method": "pearson"}
    if " by " in q:
        left, right = question.split(" by ", 1)
        value = _column_lookup(columns, left)
        group = _column_lookup(columns, right)
        if value and group:
            return {"tool": "groupby_aggregation", "group_by": group, "value_column": value, "aggregation": agg or "mean"}
    if agg:
        value_col = _target_value_column(columns, question)
        filters = []
        filter_col, filter_specs = _numeric_filter_specs(question, columns)
        if filter_specs:
            filters.append(filter_specs[0]["filter"])
        elif _has_filter_condition(question) and value_col:
            return {"tool": "filtered_aggregation", "status": "needs_clarification", "reason": "I found the aggregation column, but could not determine the filter condition."}
        if value_col:
            return {"tool": "filtered_aggregation", "filters": filters, "aggregation": {"column": value_col, "function": agg}}
    return None


def _llm_plan(question: str, df: pd.DataFrame) -> dict[str, Any] | None:
    if not has_llm():
        return None
    try:
        return complete_json(
            [
                {"role": "system", "content": "Plan one safe dataframe analysis tool call. Return strict JSON. Tools: filtered_aggregation, groupby_aggregation, compare_groups, correlation_query, chart_lookup, none. Use actual column names from schema. Do not write code."},
                {"role": "user", "content": f"Question: {question}\nColumns/schema: {json.dumps(_schema_for_llm(df), default=str)[:10000]}\nReturn e.g. {{\"tool\":\"filtered_aggregation\",\"filters\":[{{\"column\":\"sleep_hours_per_night\",\"operator\":\"<\",\"value\":4}}],\"aggregation\":{{\"column\":\"stress_level\",\"function\":\"mean\"}}}}"},
            ],
            max_tokens=900,
        )
    except Exception:
        return None


def execute_plan(df: pd.DataFrame, plan: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any] | None:
    if plan.get("status") == "needs_clarification":
        return {"tool": plan.get("tool"), "status": "needs_clarification", "message": plan.get("reason", "Please clarify the comparison groups.")}
    plan = _canonicalize_plan(plan, list(df.columns))
    missing = _plan_missing_columns(plan, list(df.columns))
    if missing:
        raise ValueError(f"Could not map columns: {', '.join(missing)}")
    tool = plan.get("tool")
    if tool == "filtered_aggregation":
        return filtered_aggregation(df, plan.get("filters", []), plan["aggregation"])
    if tool == "groupby_aggregation":
        return groupby_aggregation(df, plan["group_by"], plan["value_column"], plan.get("aggregation", "mean"))
    if tool == "compare_groups":
        return compare_groups(df, plan["value_column"], plan.get("aggregation", "mean"), plan.get("groups", []))
    if tool == "correlation_query":
        return correlation_query(df, plan["x_column"], plan["y_column"], plan.get("method", "pearson"))
    if tool == "chart_lookup":
        charts = analysis.get("chart_specs", [])
        return {"tool": "chart_lookup", "matched_charts": [{"title": c["title"], "insight": c.get("insight"), "chart_type": c.get("chart_type")} for c in charts[:8]]}
    return None


def _tool_trace(question: str, intent: str, plan: dict[str, Any] | None, tool_result: dict[str, Any] | None, columns: list[str]) -> dict[str, Any]:
    mapped_columns: dict[str, str] = {}
    for phrase in ["gpa", "previous semester gpa", "sleep", "productivity", "stress", "screen time", "social media", "part time", "performance"]:
        col = _resolve_column(columns, phrase)
        if col:
            mapped_columns[phrase] = col
    actual_tool = tool_result.get("tool") if isinstance(tool_result, dict) else None
    planned_tool = plan.get("tool") if isinstance(plan, dict) else None
    errors = []
    if isinstance(tool_result, dict) and tool_result.get("status") == "failed":
        errors.append(str(tool_result.get("error", "tool failed")))
    route = "rag"
    if tool_result:
        route = "fallback" if tool_result.get("status") in {"failed", "needs_clarification"} else "hybrid"
    elif plan and planned_tool not in {None, "none", "chart_lookup"}:
        route = "fallback"
    rag_used = bool(route in {"rag", "hybrid"} or planned_tool == "chart_lookup")
    return {
        "user_question": question,
        "route": route,
        "detected_intent": intent or "unknown",
        "resolved_columns": mapped_columns,
        "extracted_filters": _filters_from_result(tool_result) or _filters_from_plan(plan),
        "planned_tool_call": plan,
        "tool_result": tool_result,
        "actual_tool_called": actual_tool,
        "rag_used": rag_used,
        "errors": errors,
        # Backward-compatible aliases for the existing frontend/debug display.
        "question": question,
        "intent": intent or "unknown",
        "mapped_columns": mapped_columns,
        "filters_extracted": _filters_from_result(tool_result) or _filters_from_plan(plan),
        "filters_applied": _filters_from_result(tool_result),
        "matched_rows": _matched_rows_from_result(tool_result),
        "result": tool_result,
    }


def _fallback_tool_trace(question: str, error: str | None = None) -> dict[str, Any]:
    errors = [error] if error else []
    return {
        "user_question": question,
        "route": "fallback",
        "detected_intent": "unknown",
        "resolved_columns": {},
        "extracted_filters": [],
        "planned_tool_call": None,
        "actual_tool_called": None,
        "tool_result": None,
        "rag_used": False,
        "errors": errors,
        "question": question,
        "intent": "unknown",
        "mapped_columns": {},
        "filters_extracted": [],
        "filters_applied": [],
        "matched_rows": None,
        "result": None,
    }


def _filters_from_plan(plan: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not plan:
        return []
    if plan.get("tool") == "filtered_aggregation":
        return plan.get("filters", [])
    if plan.get("tool") == "compare_groups":
        return [filt for group in plan.get("groups", []) for filt in group.get("filters", [])]
    return []


def _emit_tool_trace(dataset_id: str, trace: dict[str, Any]) -> None:
    LAST_CHAT_TRACES[dataset_id] = trace
    print("USER QUESTION:", trace.get("user_question"))
    print("DETECTED INTENT:", trace.get("detected_intent"))
    print("RESOLVED COLUMNS:", trace.get("resolved_columns"))
    print("EXTRACTED FILTERS:", trace.get("extracted_filters"))
    print("PLANNED TOOL:", trace.get("planned_tool_call"))
    print("ACTUAL TOOL:", trace.get("actual_tool_called"))
    print("TOOL RESULT:", trace.get("tool_result"))


def get_last_tool_trace(dataset_id: str) -> dict[str, Any] | None:
    return LAST_CHAT_TRACES.get(dataset_id)


def _filters_from_result(tool_result: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not tool_result:
        return []
    if tool_result.get("tool") == "filtered_aggregation":
        return tool_result.get("filters_applied", [])
    if tool_result.get("tool") == "compare_groups":
        return [filt for group in tool_result.get("groups", []) for filt in group.get("filters", [])]
    return []


def _matched_rows_from_result(tool_result: dict[str, Any] | None) -> Any:
    if not tool_result:
        return None
    if "matched_rows" in tool_result:
        return tool_result["matched_rows"]
    if tool_result.get("tool") == "compare_groups":
        return [group.get("matched_rows") for group in tool_result.get("groups", [])]
    return None


def answer_question(dataset_id: str, question: str, analysis: dict[str, Any]) -> dict[str, Any]:
    df = read_frame(dataset_id)
    columns = list(df.columns)
    chunks = analysis.get("rag_chunks", [])
    evidence = retrieve(chunks, question)
    debug_evidence: list[dict[str, str]] = []
    try:
        plan = _heuristic_plan(question, df) or _llm_plan(question, df)
    except Exception as exc:
        trace = _fallback_tool_trace(question, f"planner failed because {exc}")
        _emit_tool_trace(dataset_id, trace)
        return {
            "answer": "I could not plan that analysis request.",
            "evidence": evidence[:5],
            "debug_evidence": [{"title": "Tool trace", "text": json.dumps(trace, default=str)}],
            "tool_trace": trace,
            "tool_result": None,
            "grounding_preview": "",
        }
    if not plan:
        exact_missing = _missing_column_message(question, columns)
        if exact_missing:
            trace = _fallback_tool_trace(question, "planner produced no tool plan")
            _emit_tool_trace(dataset_id, trace)
            return {"answer": exact_missing, "evidence": evidence[:5], "debug_evidence": [{"title": "Tool trace", "text": json.dumps(trace, default=str)}], "tool_trace": trace, "tool_result": None, "grounding_preview": ""}
    tool_result = None
    if plan and plan.get("tool") not in {None, "none"}:
        try:
            if plan.get("tool") == "filtered_aggregation" and _has_filter_condition(question) and not plan.get("filters"):
                plan = {"tool": "filtered_aggregation", "status": "needs_clarification", "reason": "I found the aggregation column, but could not determine the filter condition."}
            tool_result = execute_plan(df, plan, analysis)
        except Exception as exc:
            tool_result = {"tool": plan.get("tool"), "status": "failed", "error": str(exc), "plan": plan}
            debug_evidence.append({"title": f"Failed tool attempt: {plan.get('tool')}", "text": json.dumps(tool_result, default=str)})

    trace_intent = plan.get("tool") if plan else "rag_lookup"
    trace = _tool_trace(question, trace_intent, plan, tool_result, columns)
    _emit_tool_trace(dataset_id, trace)
    debug_evidence.append({"title": "Tool trace", "text": json.dumps(trace, default=str)})

    if tool_result and tool_result.get("status") == "needs_clarification":
        message = tool_result.get("message") or "Please clarify the requested filters."
        if plan and plan.get("tool") == "filtered_aggregation":
            message = "I found the GPA column, but I could not determine the sleep-hours condition. Did you mean sleep_hours_per_night <= 5?"
        return {
            "answer": message if plan and plan.get("tool") == "filtered_aggregation" else "I can compute that comparison, but I need two clear groups. Please specify both groups, for example: sleep 5 hours or less vs sleep 8 or more hours.",
            "evidence": evidence[:5],
            "debug_evidence": debug_evidence,
            "tool_trace": trace,
            "tool_result": tool_result,
            "grounding_preview": "",
        }

    if tool_result and tool_result.get("status") == "failed":
        exact_missing = _missing_column_message(question, columns)
        if exact_missing:
            return {"answer": exact_missing, "evidence": evidence[:5], "debug_evidence": debug_evidence, "tool_trace": trace, "tool_result": tool_result, "grounding_preview": ""}
        if plan and plan.get("tool") == "filtered_aggregation":
            return {"answer": "I could not compute that because the filter tool failed.", "evidence": evidence[:5], "debug_evidence": debug_evidence, "tool_trace": trace, "tool_result": tool_result, "grounding_preview": ""}
        if plan and plan.get("tool") == "compare_groups":
            return {"answer": "I could not compute that comparison because one of the requested columns or groups could not be mapped unambiguously. Please specify the exact value column and filter column.", "evidence": evidence[:5], "debug_evidence": debug_evidence, "tool_trace": trace, "tool_result": tool_result, "grounding_preview": ""}

    tool_artifacts = analysis.get("tool_result_artifacts", [])
    if tool_result and tool_result.get("status") != "failed":
        artifact = {"title": f"Computed result: {tool_result.get('tool')}", "text": json.dumps(tool_result, default=str), "result": tool_result}
        tool_artifacts.append(artifact)
        analysis["tool_result_artifacts"] = tool_artifacts[-20:]
        analysis["rag_chunks"] = chunks + [{"title": artifact["title"], "text": artifact["text"]}]
        save_json(dataset_id, "analysis", analysis)

    context = "\n\n".join(f"[{idx + 1}] {chunk['title']}\n{chunk['text'][:1600]}" for idx, chunk in enumerate(evidence))
    successful_tool_result = tool_result if tool_result and tool_result.get("status") != "failed" else None
    computed = json.dumps(successful_tool_result, default=str) if successful_tool_result else "No computation was run."
    if has_llm():
        try:
            answer = complete_text(
                [
                    {"role": "system", "content": "You are DataPilot AI, a tool-using data analyst. If a computed tool result is provided, answer from it. Mention matched row counts, filters, aggregation, and caveats. Never say context is insufficient when a tool result exists. Correlation is not causation."},
                    {"role": "user", "content": f"Question: {question}\nRetrieved context:\n{context}\nComputed tool result:\n{computed}\nAnswer concisely with evidence."},
                ],
                max_tokens=750,
            )
        except Exception:
            answer = _fallback_answer(question, successful_tool_result, evidence)
    else:
        answer = _fallback_answer(question, successful_tool_result, evidence)

    returned_evidence = evidence[:5]
    if successful_tool_result:
        returned_evidence = [{"title": "Computed Tool Result", "text": json.dumps(successful_tool_result, default=str)}] + returned_evidence
    return {"answer": answer, "evidence": returned_evidence, "debug_evidence": debug_evidence, "tool_trace": trace, "tool_result": tool_result, "grounding_preview": context[:1200]}


def _fallback_answer(question: str, tool_result: dict[str, Any] | None, evidence: list[dict[str, str]]) -> str:
    if tool_result and tool_result.get("status") != "failed":
        if tool_result.get("tool") == "filtered_aggregation":
            if tool_result.get("status") == "no_matches":
                return f"No rows matched the requested filters for {tool_result['column']}, so I could not compute a {tool_result['aggregation']}."
            text = f"{tool_result['aggregation']} of {tool_result['column']} is {tool_result['result']} across {tool_result['matched_rows']} matching rows."
            if "overall_baseline" in tool_result:
                diff = tool_result.get("difference_from_overall")
                text += f" The overall dataset baseline is {tool_result['overall_baseline']}, a difference of {diff}."
            if tool_result.get("comparison_group"):
                comp = tool_result["comparison_group"]
                text += f" Rows outside the filter have {tool_result['aggregation']} {comp['result']} across {comp['matched_rows']} rows."
            return text
        if tool_result.get("tool") == "compare_groups":
            groups = tool_result.get("groups", [])
            if len(groups) >= 2:
                warning = " One group has a small sample size, so treat the comparison cautiously." if any(g.get("sample_size_warning") for g in groups) else ""
                return f"{groups[0]['label']} has average {tool_result['value_column']} of {groups[0]['result']} across {groups[0]['matched_rows']} rows. {groups[1]['label']} has average {tool_result['value_column']} of {groups[1]['result']} across {groups[1]['matched_rows']} rows. The difference is {tool_result.get('difference')} points. This is descriptive, not proof of causation.{warning}"
        return f"Computed result for your question: {json.dumps(tool_result, default=str)}"
    return "I found relevant context, but no safe computation plan was available for this question."
