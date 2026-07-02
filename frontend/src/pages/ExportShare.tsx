import { Download, FileJson, FileText, Link2, Package } from "lucide-react";
import { motion } from "framer-motion";
import { useParams } from "react-router-dom";
import { downloadUrl } from "../api/client";
import { PageFade, PremiumCard, SectionHeader, StatusBadge } from "../components/Premium";
import { useWorkspace } from "../store/useWorkspace";

export default function ExportShare() {
  const { datasetId } = useParams();
  const { analysis, trainedModel } = useWorkspace();
  const realDataset = datasetId && datasetId !== "demo";

  return (
    <PageFade className="page-shell space-y-6 pb-10">
      <SectionHeader eyebrow="Export / Share" title="Reporting center" copy="Package analysis artifacts, dashboard reports, model bundles, and workspace links from one place." />
      <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
        <ExportCard index={0} icon={FileJson} title="Analysis JSON" copy="Full structured profile, EDA report, chart specs, insights, and analysis log." disabled={!realDataset} onClick={() => downloadUrl(`/api/datasets/${datasetId}/export/analysis`)} />
        <ExportCard index={1} icon={FileText} title="Dashboard Report" copy="Clean HTML report containing the dataset summary and generated insight narrative." disabled={!realDataset} onClick={() => downloadUrl(`/api/datasets/${datasetId}/export/report`)} />
        <ExportCard index={2} icon={Package} title="Model Bundle" copy="Model, preprocessing pipeline, metrics, schema, example input, and README." disabled={!realDataset || !trainedModel} onClick={() => downloadUrl(`/api/datasets/${datasetId}/models/${trainedModel?.model_id}/download`)} />
        <ExportCard index={3} icon={Download} title="Prediction Results" copy="Available after batch prediction from Prediction Studio." disabled comingSoon onClick={() => undefined} />
        <ExportCard index={4} icon={FileJson} title="RAG Artifacts" copy="Included in the JSON analysis export as computed analysis chunks." disabled={!analysis} onClick={() => realDataset && downloadUrl(`/api/datasets/${datasetId}/export/analysis`)} />
        <ExportCard index={5} icon={Link2} title="Share Workspace Link" copy="Copy the current workspace route for handoff or review." disabled={false} onClick={() => navigator.clipboard?.writeText(window.location.href)} />
      </div>
      {datasetId === "demo" && <p className="text-sm text-slate-400">Demo exports are preview-only. Upload a real CSV to download generated files.</p>}
    </PageFade>
  );
}

function ExportCard({ icon: Icon, title, copy, disabled, comingSoon, onClick, index }: { icon: typeof Download; title: string; copy: string; disabled?: boolean; comingSoon?: boolean; onClick: () => void; index: number }) {
  return (
    <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: index * 0.05 }}>
    <PremiumCard className="h-full p-5">
      <div className="mb-5 flex items-start justify-between gap-3">
        <span className="rounded-xl bg-cyan-300/10 p-3 text-cyan-200"><Icon size={22} /></span>
        {comingSoon ? <StatusBadge tone="amber">Coming soon</StatusBadge> : <StatusBadge tone={disabled ? "slate" : "emerald"}>{disabled ? "Unavailable" : "Ready"}</StatusBadge>}
      </div>
      <h2 className="text-lg font-semibold">{title}</h2>
      <p className="mt-3 min-h-16 text-sm leading-6 text-slate-400">{copy}</p>
      <button disabled={disabled} onClick={onClick} className="button-secondary mt-5 inline-flex w-full items-center justify-center gap-2 rounded-lg px-4 py-3 text-sm font-medium disabled:cursor-not-allowed disabled:opacity-45">
        <Download size={16} /> Export
      </button>
    </PremiumCard>
    </motion.div>
  );
}
