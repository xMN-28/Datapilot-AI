import html
import json
from pathlib import Path

from .storage_service import EXPORTS, load_json


def analysis_export(dataset_id: str) -> Path:
    analysis = load_json(dataset_id, "analysis")
    if not analysis:
        raise ValueError("Analysis has not been generated")
    path = EXPORTS / f"{dataset_id}_analysis.json"
    path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    return path


def html_report(dataset_id: str) -> Path:
    analysis = load_json(dataset_id, "analysis")
    if not analysis:
        raise ValueError("Analysis has not been generated")
    charts = "\n".join(f"<section><h2>{html.escape(c['title'])}</h2><p>{html.escape(c.get('insight') or '')}</p></section>" for c in analysis.get("chart_specs", []))
    body = f"""<!doctype html><html><head><meta charset="utf-8"><title>DataPilot AI Report</title>
<style>body{{font-family:Inter,Arial,sans-serif;background:#080b14;color:#edf6ff;padding:48px}}section{{border:1px solid #26334d;border-radius:14px;padding:20px;margin:16px 0;background:#101827}}</style></head>
<body><h1>DataPilot AI Analysis Report</h1><p>{analysis['rows']} rows, {analysis['columns']} columns.</p>{charts}</body></html>"""
    path = EXPORTS / f"{dataset_id}_report.html"
    path.write_text(body, encoding="utf-8")
    return path
