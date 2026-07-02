from collections import Counter
from typing import Any

from .llm_service import complete_text, has_llm


def build_artifacts(analysis: dict[str, Any]) -> list[dict[str, str]]:
    chunks: list[dict[str, str]] = []
    eda = analysis.get("eda_report", {})
    chunks.append({"title": "Dataset profile artifact", "text": str(eda.get("dataset", {}))})
    chunks.append({"title": "Schema summary", "text": str(analysis.get("schema", {}))})
    chunks.append({"title": "Cleaning report", "text": str(analysis.get("cleaning_report", {}))})
    chunks.append({"title": "Feature engineering", "text": str(analysis.get("feature_engineering_report", {}))})
    for name, stats in eda.get("numeric", {}).items():
        chunks.append({"title": f"Column profile: {name}", "text": f"type=numeric stats={stats}"})
    for name, stats in eda.get("categorical", {}).items():
        chunks.append({"title": f"Column profile: {name}", "text": f"type=categorical top_categories={stats.get('top_values')} missing_count={stats.get('missing_count')} note={stats.get('missing_note')}"})
    for name, stats in eda.get("boolean", {}).items():
        chunks.append({"title": f"Column profile: {name}", "text": f"type=boolean stats={stats}"})
    chunks.append({"title": "Relationship artifacts", "text": str(eda.get("relationships", {}))})
    chunks.append({"title": "Data quality artifacts", "text": str({"missing": analysis.get("cleaning_report", {}).get("missing_by_column", {}), "warnings": eda.get("dataset", {}).get("data_quality_warnings", []), "skipped": eda.get("skipped", [])})})
    for chart in analysis.get("chart_specs", []):
        chunks.append({"title": f"Chart: {chart['title']}", "text": f"Type: {chart.get('chart_type')}. Intent: {chart.get('intent') or chart.get('chart_intent')}. Usefulness: {chart.get('usefulness_score')}. Columns: {chart.get('columns_used', [])}. Missing excluded: {chart.get('excluded_missing_count', 0)}. Notes: {chart.get('notes', [])}. Why selected: {chart.get('why')}. Insight: {chart.get('insight', '')}. Evidence: {chart.get('computed_stats', {})}"})
    for artifact in analysis.get("tool_result_artifacts", []):
        chunks.append({"title": artifact.get("title", "Tool result artifact"), "text": artifact.get("text", str(artifact))})
    chunks.append({"title": "Analysis log", "text": str(analysis.get("analysis_log", []))})
    return chunks


def retrieve(chunks: list[dict[str, str]], question: str, limit: int = 5) -> list[dict[str, str]]:
    terms = [t.lower() for t in question.split() if len(t) > 2]
    wants_charts = any(term in {"chart", "charts", "visual", "visuals", "visualization", "visualizations", "pattern", "patterns", "useful", "important"} for term in terms)
    scored = []
    for chunk in chunks:
        title = chunk["title"].lower()
        text = f"{chunk['title']} {chunk['text']}".lower()
        counts = Counter(text.split())
        score = sum(counts[t] for t in terms) + sum(1 for t in terms if t in text)
        if wants_charts and title.startswith("chart:"):
            score += 6
        if wants_charts and title in {"feature engineering", "schema summary"}:
            score -= 2
        scored.append((score, chunk))
    return [chunk for score, chunk in sorted(scored, key=lambda x: x[0], reverse=True)[:limit] if score > 0] or chunks[:limit]


def answer_from_context(question: str, chunks: list[dict[str, str]]) -> dict[str, Any]:
    evidence = retrieve(chunks, question)
    combined = " ".join(c["text"] for c in evidence)
    if not evidence:
        answer = "I do not have enough computed analysis to answer that yet. Run the dashboard analysis first."
    elif has_llm():
        try:
            context = "\n\n".join(f"[{idx + 1}] {chunk['title']}\n{chunk['text'][:1800]}" for idx, chunk in enumerate(evidence))
            answer = complete_text(
                [
                    {
                        "role": "system",
                        "content": "You are DataPilot AI's analyst. Answer using only the retrieved computed analysis context. Be specific, cite evidence titles inline, and say when the context is insufficient. Do not invent raw CSV facts.",
                    },
                    {
                        "role": "user",
                        "content": f"Question: {question}\n\nRetrieved computed context:\n{context}\n\nGive a concise but genuinely analytical answer.",
                    },
                ],
                max_tokens=700,
            )
            if not answer:
                raise RuntimeError("Empty LLM response")
        except Exception:
            answer = "The LLM call failed, so I am falling back to retrieved evidence only. The relevant analysis excerpts are shown in the evidence panel."
    elif any(word in question.lower() for word in ["quality", "missing", "duplicate"]):
        answer = "The computed data-quality context points to the missing-value and duplicate summaries. Review the referenced evidence before acting on any row-level assumptions."
    elif any(word in question.lower() for word in ["strongest", "important", "pattern", "first"]):
        answer = "Start with the highest-scored visualizations and the strongest numeric relationships. Those were selected from computed EDA, not guessed from raw rows."
    else:
        answer = "Based on the retrieved computed artifacts, the safest answer is to use the dashboard evidence rather than infer beyond it. The relevant analysis excerpts are shown as evidence."
    return {"answer": answer, "evidence": evidence, "grounding_preview": combined[:1200]}
