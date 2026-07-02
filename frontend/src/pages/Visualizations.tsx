import { AlertTriangle, Database, FileBarChart, Rows3, WandSparkles } from "lucide-react";
import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { api } from "../api/client";
import ChartCard, { chartGridClass } from "../charts/ChartCard";
import AnalysisLog from "../components/AnalysisLog";
import { PageFade, SectionHeader } from "../components/Premium";
import StatCard from "../components/StatCard";
import { compactNumber } from "../lib/format";
import { useWorkspace } from "../store/useWorkspace";

export default function Visualizations() {
  const { datasetId } = useParams();
  const { analysis, setAnalysis } = useWorkspace();
  const [error, setError] = useState("");

  useEffect(() => {
    if (!datasetId || datasetId === "demo" || analysis?.dataset_id === datasetId) return;
    api.analysis(datasetId).then(setAnalysis).catch(() => setError("No analysis found yet. Return to Overview and generate the dashboard."));
  }, [datasetId, analysis?.dataset_id, setAnalysis]);

  if (!analysis) return <div className="premium-card p-8 text-slate-300">{error || "Load a demo workspace or upload a CSV to generate visualizations."}</div>;

  return (
    <PageFade className="dashboard-shell space-y-6 pb-10">
      <SectionHeader eyebrow="Data Visualizations & Insights" title="Generated analytics dashboard" copy="A scored set of adaptive charts, data-quality signals, and AI-written observations for the uploaded dataset." />
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
        <StatCard label="Rows" value={compactNumber(analysis.rows)} icon={Rows3} />
        <StatCard label="Columns" value={compactNumber(analysis.columns)} icon={Database} tone="violet" />
        <StatCard label="Missing values" value={compactNumber(analysis.eda_report.dataset.missing_values_total)} icon={AlertTriangle} tone="amber" />
        <StatCard label="Duplicate rows" value={compactNumber(analysis.cleaning_report.duplicate_rows)} icon={FileBarChart} tone="emerald" />
        <StatCard label="Engineered features" value={analysis.feature_engineering_report.count} icon={WandSparkles} />
      </div>
      <AnalysisLog items={analysis.analysis_log} />
      <div className="grid gap-5 lg:grid-cols-2 2xl:grid-cols-3 min-[1900px]:grid-cols-4">
        {analysis.chart_specs.map((chart, index) => (
          <div key={chart.chart_id} className={chartGridClass(chart)}>
            <ChartCard chart={chart} index={index} />
          </div>
        ))}
      </div>
    </PageFade>
  );
}
