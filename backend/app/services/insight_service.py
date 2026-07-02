import json
from typing import Any

from .llm_service import complete_json, has_llm


def _fallback_insight(chart: dict[str, Any]) -> dict[str, str]:
    title = chart["title"]
    ctype = chart["chart_type"]
    if ctype in {"histogram", "bar"} and chart["data"]:
        top = chart["data"][0]
        return {"title": title.replace("_", " ").title(), "insight": f"{title} shows the strongest visible signal around {next(iter(top.values()))}.", "confidence": "medium", "reason": chart.get("why", "Computed chart evidence."), "caveat": "No LLM key was available, so this is a deterministic summary."}
    if ctype == "correlation_bar":
        return {"title": "Most Useful Numeric Relationships", "insight": "The strongest numeric relationships are ranked by analytical usefulness, not correlation alone.", "confidence": "high", "reason": "Computed Pearson correlations with redundancy filtering.", "caveat": "Correlation is not causation."}
    return {"title": title.replace("_", " ").title(), "insight": chart.get("why", "This chart was selected from computed dataset structure."), "confidence": "medium", "reason": "Backend-selected chart evidence.", "caveat": "Review the chart before making decisions."}


def generate_chart_insights(charts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not has_llm():
        for chart in charts:
            chart.update(_fallback_insight(chart))
        return charts

    for chart in charts:
        evidence = {
            "chart_title": chart["title"],
            "chart_type": chart["chart_type"],
            "intent": chart.get("intent") or chart.get("chart_intent"),
            "columns_used": list(chart["encoding"].values()),
            "computed_stats": chart.get("computed_stats", {}),
            "statistical_strength": chart.get("statistical_strength"),
            "analytical_usefulness": chart.get("analytical_usefulness"),
            "redundancy_penalty": chart.get("redundancy_penalty"),
            "leakage_penalty": chart.get("leakage_penalty"),
            "final_usefulness_score": chart.get("final_usefulness_score"),
            "redundancy_reason": chart.get("redundancy_reason"),
            "excluded_missing_count": chart.get("excluded_missing_count", 0),
            "excluded_missing_percent": chart.get("excluded_missing_percent", 0),
            "notes": chart.get("notes", []),
            "data_preview": chart["data"][:8],
            "warnings": [],
        }
        try:
            parsed = complete_json(
                [
                    {
                        "role": "system",
                        "content": "You are a senior data analyst and chart selector. Use only computed evidence. Create a concise expressive chart title and an insight. Choose language that fits the chart intent. Do not praise charts merely because correlation is high. Reject obvious, formula-derived, redundant, ID-based, weak effect-size, or date-part relationships. Prefer non-obvious relationships, distribution shape, outliers, imbalance, meaningful trends, and decision-useful patterns. Category data excludes null-like values; never discuss nan, None, null, Missing, N/A, empty strings, or undefined as real groups. You may mention excluded_missing_count as a caveat. Return strict JSON with keys title, insight, confidence, reason, caveat. No markdown.",
                    },
                    {
                        "role": "user",
                        "content": f"Write one concise, useful chart insight. Avoid generic filler. Evidence: {json.dumps(evidence, default=str)[:5000]}",
                    },
                ],
                max_tokens=350,
            )
            chart.update(parsed)
        except Exception:
            chart.update(_fallback_insight(chart))
    return charts
