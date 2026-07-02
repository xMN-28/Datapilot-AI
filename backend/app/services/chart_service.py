from typing import Any

import numpy as np
import pandas as pd

from ..utils.missing import categorical_counts_for_chart, clean_categorical_for_chart, normalize_missing_values


LOW_VALUE_NUMERIC_NAMES = {"year", "month", "day", "quarter", "day_of_week", "week", "weekofyear"}
FORMULA_GROUPS = [
    {"order_amount", "tax_amount", "discount_amount", "profit_amount"},
    {"unit_price", "quantity", "order_amount"},
    {"salary_min_lpa", "salary_max_lpa", "salary_midpoint_lpa"},
]
ID_NAME_TOKENS = {"id", "uuid", "order_id", "customer_id", "product_id", "job_id", "company_id", "transaction_id", "invoice_id", "row_id"}


def _is_low_value_numeric_column(name: str) -> bool:
    lower = name.lower()
    return lower in LOW_VALUE_NUMERIC_NAMES or lower.endswith(("_year", "_month", "_day", "_quarter", "_day_of_week", "_is_weekend")) or _is_id_like_name(lower)


def _is_id_like_name(name: str) -> bool:
    lower = name.lower()
    return lower in ID_NAME_TOKENS or lower.endswith("_id") or any(token in lower for token in ["uuid", "transaction_id", "invoice_id", "serial"])


def _is_mostly_unique(df: pd.DataFrame, name: str) -> bool:
    if name not in df.columns:
        return False
    rows = max(len(df), 1)
    return rows > 25 and df[name].nunique(dropna=True) / rows > 0.9


def _date_part_reason(a: str, b: str) -> str | None:
    low = {a.lower(), b.lower()}
    date_parts = {"year", "month", "day", "quarter", "week", "day_of_week"}
    if all(part in date_parts or any(part.endswith(f"_{suffix}") for suffix in date_parts) for part in low):
        return "date-derived parts are redundant against each other"
    return None


def _salary_reason(a: str, b: str, columns: set[str]) -> str | None:
    pair = {a.lower(), b.lower()}
    salary_set = {"salary_min_lpa", "salary_max_lpa", "salary_midpoint_lpa"}
    if salary_set.issubset(columns) and len(pair.intersection(salary_set)) == 2:
        if "salary_midpoint_lpa" in pair:
            return "salary_midpoint_lpa is derived from salary_min_lpa and salary_max_lpa"
        return "salary_min_lpa and salary_max_lpa are salary range endpoints and usually redundant"
    return None


def _formula_reason(a: str, b: str, columns: set[str]) -> str | None:
    pair = {a.lower(), b.lower()}
    salary = _salary_reason(a, b, columns)
    if salary:
        return salary
    date = _date_part_reason(a, b)
    if date:
        return date
    for group in FORMULA_GROUPS:
        if len(pair.intersection(group)) == 2 and len(columns.intersection(group)) >= 2:
            if {"unit_price", "order_amount"}.issubset(pair) and "quantity" in columns:
                return "unit_price and order_amount are formula-like when quantity exists"
            if "order_amount" in pair:
                return "order_amount relationship is formula-derived with tax, discount, profit, unit price, or quantity"
            return "formula-like financial relationship is redundant"
    return None


def _correlation_candidate(df: pd.DataFrame, item: dict[str, Any], columns: set[str]) -> dict[str, Any]:
    a, b = item["x"], item["y"]
    reason = None
    if _is_id_like_name(a) or _is_id_like_name(b) or _is_mostly_unique(df, a) or _is_mostly_unique(df, b):
        reason = "ID-like or mostly unique numeric columns are not analytically useful for correlation charts"
    else:
        reason = _formula_reason(a, b, columns)
    statistical_strength = min(1.0, abs(float(item["correlation"])))
    redundancy_penalty = 0.85 if reason else 0.0
    leakage_penalty = 0.35 if any(token in f"{a} {b}".lower() for token in ["target", "label", "outcome"]) else 0.0
    analytical_usefulness = 0.82 if not reason else 0.12
    final = max(0.05, statistical_strength * 0.35 + analytical_usefulness * 0.55 - redundancy_penalty - leakage_penalty)
    return {
        **item,
        "statistical_strength": round(statistical_strength, 4),
        "analytical_usefulness": round(analytical_usefulness, 4),
        "redundancy_penalty": round(redundancy_penalty, 4),
        "leakage_penalty": round(leakage_penalty, 4),
        "final_usefulness_score": round(final, 4),
        "should_exclude": bool(reason),
        "redundancy_reason": reason,
        "exclusion_reason": reason,
    }


def _known_excluded_pairs(columns: set[str]) -> list[dict[str, Any]]:
    pairs: list[tuple[str, str, str]] = []
    salary_cols = {"salary_min_lpa", "salary_max_lpa", "salary_midpoint_lpa"}
    if salary_cols.issubset(columns):
        pairs.extend(
            [
                ("salary_midpoint_lpa", "salary_min_lpa", "salary_midpoint_lpa is derived from salary_min_lpa and salary_max_lpa"),
                ("salary_midpoint_lpa", "salary_max_lpa", "salary_midpoint_lpa is derived from salary_min_lpa and salary_max_lpa"),
                ("salary_min_lpa", "salary_max_lpa", "salary_min_lpa and salary_max_lpa are salary range endpoints and usually redundant"),
            ]
        )
    for a, b in [("order_amount", "tax_amount"), ("order_amount", "discount_amount"), ("order_amount", "profit_amount"), ("unit_price", "order_amount"), ("month", "quarter"), ("year", "month"), ("day", "month")]:
        if a in columns and b in columns:
            reason = _formula_reason(a, b, columns) or "redundant or formula-derived relationship"
            pairs.append((a, b, reason))
    return [
        {
            "x": a,
            "y": b,
            "correlation": None,
            "statistical_strength": None,
            "analytical_usefulness": 0.0,
            "redundancy_penalty": 1.0,
            "leakage_penalty": 0.0,
            "final_usefulness_score": 0.01,
            "should_exclude": True,
            "redundancy_reason": reason,
            "exclusion_reason": reason,
        }
        for a, b, reason in pairs
    ]


def _is_low_value_category(name: str, stats: dict[str, Any], rows: int) -> bool:
    lower = name.lower()
    unique = int(stats.get("unique_count", 0))
    if "id" in lower or "uuid" in lower or "code" in lower:
        return True
    return unique > 40 or unique > max(12, rows * 0.15) or bool(stats.get("cardinality_warning"))


def _diagnostics(data: list[dict[str, Any]], chart_type: str) -> dict[str, Any]:
    point_count = len(data)
    estimated_payload_kb = round(max(1, point_count * 0.16), 2)
    fallback = point_count > 4000 or estimated_payload_kb > 900
    return {
        "point_count": point_count,
        "category_count": None,
        "estimated_payload_kb": estimated_payload_kb,
        "fallback_recommended": fallback,
        "fallback_reason": "Large frontend payload" if fallback else None,
        "chart_type": chart_type,
        "frontend_renderer": "echarts",
    }


def _safe_label(value: Any) -> str:
    return str(value).strip()


def _category_chart_data(series: pd.Series, limit: int = 12) -> tuple[list[dict[str, Any]], int, int]:
    counts, missing_count = categorical_counts_for_chart(series)
    counts = counts.head(limit)
    total = max(int(counts.sum()), 1)
    data = [{"value": _safe_label(k), "count": int(v), "percentage": round(float(v / total * 100), 2)} for k, v in counts.items()]
    return data, missing_count, int(total)


def _boolean_chart_data(series: pd.Series) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    cleaned = normalize_missing_values(series)
    normalized = cleaned.dropna().astype("string").str.strip().str.lower()
    true_count = int(normalized.isin({"true", "yes", "1"}).sum())
    false_count = int(normalized.isin({"false", "no", "0"}).sum())
    missing_count = int(cleaned.isna().sum())
    total = max(true_count + false_count, 1)
    stats = {
        "true_count": true_count,
        "false_count": false_count,
        "true_percentage": round(float(true_count / total * 100), 2),
        "false_percentage": round(float(false_count / total * 100), 2),
        "missing_count": missing_count,
        "valid_count": true_count + false_count,
    }
    return [{"value": "True", "count": true_count}, {"value": "False", "count": false_count}], missing_count, stats


def _missing_metadata(column: str, missing_count: int, rows: int) -> dict[str, Any]:
    percent = round(float(missing_count / max(rows, 1) * 100), 2)
    return {
        "excluded_missing_count": int(missing_count),
        "excluded_missing_percent": percent,
        "notes": [f"This chart excludes {int(missing_count)} missing values from {column}."] if missing_count else [],
    }


def _is_negligible_group_difference(grouped: pd.Series, values: pd.Series) -> tuple[bool, dict[str, Any]]:
    if len(grouped) < 2:
        return True, {"group_difference_reason": "Fewer than two groups have usable values."}
    spread = float(grouped.max() - grouped.min())
    std = float(values.std()) if len(values.dropna()) > 1 and pd.notna(values.std()) else 0.0
    mean_abs = float(values.abs().mean()) if len(values.dropna()) else 0.0
    effect_size = spread / std if std > 0 else 0.0
    relative_spread = spread / max(abs(mean_abs), 1e-9)
    negligible = spread == 0 or (effect_size < 0.15 and relative_spread < 0.04)
    return negligible, {
        "group_mean_spread": round(spread, 4),
        "effect_size_approx": round(effect_size, 4),
        "relative_spread": round(relative_spread, 4),
        "group_difference_reason": "Group means are nearly identical, so this comparison has low analytical usefulness." if negligible else None,
    }


def _chart(
    chart_id: str,
    title: str,
    chart_type: str,
    data: list[dict[str, Any]],
    encoding: dict[str, str],
    score: float,
    reason: str,
    stats: dict[str, Any] | None = None,
    statistical_strength: float | None = None,
    analytical_usefulness: float | None = None,
    redundancy_penalty: float = 0.0,
    leakage_penalty: float = 0.0,
    should_exclude: bool = False,
    exclusion_reason: str | None = None,
    chart_intent: str | None = None,
    excluded_missing_count: int = 0,
    excluded_missing_percent: float = 0.0,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    final_score = round(max(0.01, score - redundancy_penalty - leakage_penalty), 2)
    columns_used = [value for value in encoding.values() if value not in {"value", "count", "mean", "missing"}]
    return {
        "chart_id": chart_id,
        "title": title,
        "chart_type": chart_type,
        "intent": chart_intent or chart_type,
        "chart_intent": chart_intent or chart_type,
        "columns_used": columns_used,
        "render_mode": "frontend",
        "data": data,
        "encoding": encoding,
        "insight": "",
        "usefulness_score": final_score,
        "statistical_strength": round(statistical_strength if statistical_strength is not None else score, 4),
        "analytical_usefulness": round(analytical_usefulness if analytical_usefulness is not None else score, 4),
        "redundancy_penalty": round(redundancy_penalty, 4),
        "leakage_penalty": round(leakage_penalty, 4),
        "final_usefulness_score": final_score,
        "should_exclude": should_exclude,
        "redundancy_reason": exclusion_reason,
        "exclusion_reason": exclusion_reason,
        "why": reason,
        "reason_selected": reason,
        "missing_values_excluded": excluded_missing_count,
        "caveats": notes or [],
        "computed_stats": stats or {},
        "excluded_missing_count": excluded_missing_count,
        "excluded_missing_percent": excluded_missing_percent,
        "notes": notes or [],
        "render_diagnostics": _diagnostics(data, chart_type),
    }


def _correlation_heatmap(df: pd.DataFrame, corr: list[dict[str, Any]], allowed: set[str]) -> list[dict[str, Any]]:
    cols = list(dict.fromkeys([item["x"] for item in corr] + [item["y"] for item in corr]))
    cols = [col for col in cols if col in allowed][:8]
    if len(cols) < 3:
        return []
    matrix = df[cols].corr(numeric_only=True)
    data = []
    for y_idx, y in enumerate(cols):
        for x_idx, x in enumerate(cols):
            value = matrix.loc[y, x]
            if pd.notna(value):
                data.append({"x": x_idx, "y": y_idx, "x_label": x, "y_label": y, "value": round(float(value), 4)})
    return data


def _numeric_density_data(series: pd.Series) -> list[dict[str, Any]]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 20 or values.nunique() < 5:
        return []
    counts, edges = np.histogram(values, bins=28, density=True)
    smoothed = pd.Series(counts).rolling(window=3, center=True, min_periods=1).mean()
    return [{"x": round(float((edges[i] + edges[i + 1]) / 2), 4), "density": round(float(smoothed.iloc[i]), 6)} for i in range(len(smoothed))]


def _category_numeric_box_data(df: pd.DataFrame, category: str, numeric: str) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    cat_clean = clean_categorical_for_chart(df[category])
    missing_count = int(cat_clean.isna().sum())
    work = pd.DataFrame({"category": cat_clean, "value": pd.to_numeric(df[numeric], errors="coerce")}).dropna(subset=["category", "value"])
    groups = []
    for label, values in work.groupby("category")["value"]:
        if len(values) < 5:
            continue
        groups.append(
            {
                "label": _safe_label(label),
                "min": round(float(values.min()), 4),
                "q1": round(float(values.quantile(0.25)), 4),
                "median": round(float(values.median()), 4),
                "q3": round(float(values.quantile(0.75)), 4),
                "max": round(float(values.max()), 4),
                "count": int(len(values)),
            }
        )
    grouped_mean = work.groupby("category")["value"].mean() if len(work) else pd.Series(dtype=float)
    negligible, effect_stats = _is_negligible_group_difference(grouped_mean, work["value"]) if len(grouped_mean) else (True, {})
    return groups[:10], missing_count, {"groups": groups[:10], "negligible_difference": negligible, **effect_stats}


def _two_category_chart(df: pd.DataFrame, a: str, b: str, rows: int) -> dict[str, Any] | None:
    a_clean = clean_categorical_for_chart(df[a])
    b_clean = clean_categorical_for_chart(df[b])
    work = pd.DataFrame({"a": a_clean, "b": b_clean}).dropna()
    if len(work) < 10 or work["a"].nunique() < 2 or work["b"].nunique() < 2:
        return None
    if work["a"].nunique() > 8 or work["b"].nunique() > 8:
        return None
    table = pd.crosstab(work["a"], work["b"])
    missing_count = int(rows - len(work))
    data = []
    for row_label, row in table.iterrows():
        row_total = max(int(row.sum()), 1)
        item = {"category": _safe_label(row_label)}
        for col_label, count in row.items():
            item[_safe_label(col_label)] = int(count)
            item[f"{_safe_label(col_label)}_percent"] = round(float(count / row_total * 100), 2)
        data.append(item)
    chi_like = 0.0
    if table.shape[0] > 1 and table.shape[1] > 1:
        expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / max(table.values.sum(), 1)
        chi_like = float(((table.values - expected) ** 2 / np.maximum(expected, 1e-9)).sum() / max(table.values.sum(), 1))
    return _chart(
        f"{a}_{b}_stacked",
        f"{b} mix by {a}",
        "stacked_bar",
        data,
        {"x": "category", "series": b},
        0.77 + min(0.1, chi_like),
        "Shows how one categorical variable is distributed inside another without treating missing values as groups.",
        {"cross_tab": table.to_dict(), "association_strength": round(chi_like, 4)},
        statistical_strength=min(1.0, chi_like),
        analytical_usefulness=0.78 if chi_like > 0.02 else 0.45,
        chart_intent="group comparison",
        **_missing_metadata(f"{a}/{b}", missing_count, rows),
    )


def generate_chart_specs(df: pd.DataFrame, eda: dict[str, Any]) -> list[dict[str, Any]]:
    charts: list[dict[str, Any]] = []
    rows = max(len(df), 1)
    numeric_cols = [c for c in eda["numeric"].keys() if not _is_low_value_numeric_column(c)]
    boolean_cols = list(eda.get("boolean", {}).keys())
    categorical_cols = [c for c in eda["categorical"].keys() if c not in boolean_cols and not _is_low_value_category(c, eda["categorical"][c], rows)]
    datetime_cols = list(eda["datetime"].keys())

    missing = [{"column": c, "missing": int(normalize_missing_values(df[c]).isna().sum())} for c in df.columns if int(normalize_missing_values(df[c]).isna().sum()) > 0]
    if missing:
        charts.append(_chart("missingness", "Missing values by column", "bar", missing, {"x": "column", "y": "missing"}, 0.72, "Highlights data quality issues before interpretation.", chart_intent="missingness"))

    for col in numeric_cols[:5]:
        try:
            s = df[col].dropna()
            if s.nunique() < 2:
                continue
            counts, edges = np.histogram(s, bins=min(24, max(6, int(np.sqrt(len(s))))))
            data = [{"bin": f"{edges[i]:.2f} - {edges[i + 1]:.2f}", "count": int(counts[i])} for i in range(len(counts))]
            score = 0.82 + min(0.1, eda["numeric"][col].get("outlier_count", 0) / rows)
            charts.append(_chart(f"{col}_distribution", f"{col} distribution", "histogram", data, {"x": "bin", "y": "count"}, score, "Shows shape, spread, and possible outliers.", eda["numeric"][col], chart_intent="distribution"))
            density = _numeric_density_data(df[col])
            if density:
                charts.append(_chart(f"{col}_density", f"{col} density curve", "density", density, {"x": "x", "y": "density"}, 0.76, "Smooths the numeric distribution to reveal shape without bin-edge noise.", eda["numeric"][col], statistical_strength=0.66, analytical_usefulness=0.76, chart_intent="distribution"))
            if eda["numeric"][col].get("outlier_count", 0) > 0:
                stats = eda["numeric"][col]
                box_data = [{"label": col, "min": stats.get("min"), "q1": stats.get("q1"), "median": stats.get("median"), "q3": stats.get("q3"), "max": stats.get("max")}]
                charts.append(_chart(f"{col}_boxplot", f"{col} spread and outliers", "boxplot", box_data, {"x": "label", "y": "value"}, 0.81, "Summarizes range, quartiles, and outlier-prone spread.", stats, statistical_strength=0.74, analytical_usefulness=0.84, chart_intent="outlier/spread"))
        except Exception:
            continue

    for col in boolean_cols[:5]:
        data, missing_count, stats = _boolean_chart_data(df[col])
        missing_meta = _missing_metadata(col, missing_count, rows)
        charts.append(_chart(f"{col}_boolean_counts", f"{col} true/false share", "donut", data, {"x": "value", "y": "count"}, 0.8, "Treats this boolean field as a binary categorical share. Missing/null-like values are excluded from the slices.", stats, chart_intent="composition/share", **missing_meta))

    for col in categorical_cols[:6]:
        top, missing_count, valid_count = _category_chart_data(df[col], limit=14)
        if len(top) > 1:
            chart_type = "donut" if len(top) <= 4 else "horizontal_bar" if len(top) > 6 else "bar"
            intent = "composition/share" if len(top) <= 4 else "ranking/comparison"
            missing_meta = _missing_metadata(col, missing_count, rows)
            stats = {**eda["categorical"].get(col, {}), "valid_count": valid_count, "top_values": top, "missing_count": missing_count}
            charts.append(_chart(f"{col}_top_values", f"Top {col} values", chart_type, top, {"x": "value", "y": "count"}, 0.78, "Summarizes dominant categories and imbalance. Missing/null-like values are excluded from category bars.", stats, chart_intent=intent, **missing_meta))

    if len(numeric_cols) >= 2:
        allowed = set(numeric_cols)
        all_columns = {c.lower() for c in df.columns}
        candidates = [_correlation_candidate(df, item, all_columns) for item in eda["relationships"]["numeric_correlations"] if item["x"] in allowed and item["y"] in allowed]
        excluded_candidates = [item for item in candidates if item["should_exclude"]]
        known_keys = {(item["x"].lower(), item["y"].lower()) for item in excluded_candidates}
        for item in _known_excluded_pairs(all_columns):
            key = (item["x"].lower(), item["y"].lower())
            reverse_key = (key[1], key[0])
            if key not in known_keys and reverse_key not in known_keys:
                excluded_candidates.append(item)
        corr = sorted([item for item in candidates if not item["should_exclude"]], key=lambda item: item["final_usefulness_score"], reverse=True)
        if corr:
            heatmap_data = _correlation_heatmap(df, corr, allowed)
            if heatmap_data:
                labels = list(dict.fromkeys([item["x_label"] for item in heatmap_data]))
                charts.append(_chart(
                    "correlation_matrix_heatmap",
                    "Correlation matrix heatmap",
                    "heatmap",
                    heatmap_data,
                    {"x": "x", "y": "y", "value": "value"},
                    0.83,
                    "Shows the broader numeric relationship structure after excluding obvious formula-derived and ID-like pairs.",
                    {"x_labels": labels, "y_labels": labels, "relationships": corr[:20], "excluded_candidates": excluded_candidates[:20]},
                    statistical_strength=max(abs(float(item["correlation"])) for item in corr[: min(8, len(corr))]),
                    analytical_usefulness=0.84,
                    chart_intent="correlation",
                ))
            data = [{"pair": f"{x['x']} / {x['y']}", "correlation": x["correlation"], "analytical_usefulness": x["analytical_usefulness"]} for x in corr[:10]]
            charts.append(_chart(
                "strongest_correlations",
                "Most analytically useful numeric relationships",
                "correlation_bar",
                data,
                {"x": "pair", "y": "correlation"},
                max(0.62, corr[0]["final_usefulness_score"]),
                "Ranks numeric relationships by analytical usefulness, not correlation alone.",
                {"relationships": corr[:10], "excluded_candidates": excluded_candidates[:20]},
                statistical_strength=corr[0]["statistical_strength"],
                analytical_usefulness=corr[0]["analytical_usefulness"],
                chart_intent="relationship",
            ))
            best = corr[0]
            sample = df[[best["x"], best["y"]]].dropna()
            points = sample.rename(columns={best["x"]: "x", best["y"]: "y"}).to_dict(orient="records")
            charts.append(_chart(
                f"{best['x']}_{best['y']}_scatter",
                f"{best['x']} vs {best['y']}",
                "scatter",
                points,
                {"x": "x", "y": "y"},
                best["final_usefulness_score"],
                "Visualizes a non-excluded numeric relationship without random sampling.",
                best,
                statistical_strength=best["statistical_strength"],
                analytical_usefulness=best["analytical_usefulness"],
                redundancy_penalty=best["redundancy_penalty"],
                leakage_penalty=best["leakage_penalty"],
                should_exclude=best["should_exclude"],
                exclusion_reason=best["exclusion_reason"],
                chart_intent="relationship",
            ))

    if categorical_cols and numeric_cols:
        preferred_numeric = [c for c in numeric_cols if any(token in c.lower() for token in ["amount", "salary", "price", "score", "rating", "cost", "days", "age", "value", "margin", "quantity"])]
        cat, num = categorical_cols[0], (preferred_numeric[0] if preferred_numeric else numeric_cols[0])
        cat_clean = clean_categorical_for_chart(df[cat])
        missing_count = int(cat_clean.isna().sum())
        work = pd.DataFrame({"category": cat_clean, "value": pd.to_numeric(df[num], errors="coerce")}).dropna(subset=["category", "value"])
        grouped = work.groupby("category")["value"].mean().sort_values(ascending=False).head(12)
        negligible, effect_stats = _is_negligible_group_difference(grouped, work["value"])
        data = [{"category": str(k), "mean": round(float(v), 4)} for k, v in grouped.items() if pd.notna(v)]
        if len(data) > 1:
            missing_meta = _missing_metadata(cat, missing_count, rows)
            if not negligible:
                charts.append(_chart(
                    f"{cat}_{num}_grouped",
                    f"Average {num} by {cat}",
                    "horizontal_bar" if len(data) > 6 else "bar",
                    data,
                    {"x": "category", "y": "mean"},
                    0.84,
                    "Compares a numeric measure across categories only when group differences are meaningful.",
                    {"groups": data, **effect_stats},
                    statistical_strength=min(1.0, float(effect_stats.get("effect_size_approx", 0)) / 2),
                    analytical_usefulness=0.86,
                    chart_intent="ranking/comparison",
                    **missing_meta,
                ))
        spread_data, spread_missing, spread_stats = _category_numeric_box_data(df, cat, num)
        if len(spread_data) > 1 and not spread_stats.get("negligible_difference"):
            charts.append(_chart(
                f"{cat}_{num}_spread",
                f"{num} spread by {cat}",
                "boxplot",
                spread_data,
                {"x": "label", "y": "value"},
                0.82,
                "Compares distribution spread across categories when differences are large enough to matter.",
                spread_stats,
                statistical_strength=min(1.0, float(spread_stats.get("effect_size_approx", 0)) / 2),
                analytical_usefulness=0.86,
                chart_intent="outlier/spread",
                **_missing_metadata(cat, spread_missing, rows),
            ))

    if len(categorical_cols) >= 2:
        two_cat = _two_category_chart(df, categorical_cols[0], categorical_cols[1], rows)
        if two_cat and two_cat["analytical_usefulness"] >= 0.5:
            charts.append(two_cat)

    if datetime_cols and numeric_cols:
        date, num = datetime_cols[0], numeric_cols[0]
        trend = df[[date, num]].dropna().sort_values(date)
        if len(trend) > 2:
            trend = trend.set_index(date).resample("ME")[num].mean().dropna().tail(60)
            data = [{"date": str(k.date()), "value": round(float(v), 4)} for k, v in trend.items()]
            if len(data) > 1:
                charts.append(_chart(f"{date}_{num}_trend", f"{num} trend over time", "line", data, {"x": "date", "y": "value"}, 0.83, "Uses detected datetime values to surface trend candidates.", chart_intent="trend"))
                charts.append(_chart(f"{date}_{num}_area_trend", f"{num} area trend over time", "area", data, {"x": "date", "y": "value"}, 0.74, "Uses an area view for time-pattern magnitude when a date and metric are available.", chart_intent="time pattern"))

    selected = sorted(charts, key=lambda c: c["usefulness_score"], reverse=True)[:14]
    return selected or [_chart("dataset_shape", "Dataset shape", "bar", [{"metric": "rows", "value": len(df)}, {"metric": "columns", "value": len(df.columns)}], {"x": "metric", "y": "value"}, 0.55, "Fallback chart when no richer visual is meaningful.")]
