import { motion } from "framer-motion";
import { ArrowRight, BarChart3, Bot, BrainCircuit, FileUp, ShieldCheck, Sparkles } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { api } from "../api/client";
import { useWorkspace } from "../store/useWorkspace";
import { useState } from "react";
import { GlowButton, PageFade, PremiumCard, StatusBadge } from "../components/Premium";

const features = [
  { icon: BarChart3, title: "Dynamic visual analytics", copy: "Candidate charts are generated, scored, deduplicated, and rendered with frontend chart specs." },
  { icon: Bot, title: "Grounded analyst chat", copy: "Chat retrieves computed analysis artifacts instead of embedding raw CSV rows by default." },
  { icon: BrainCircuit, title: "Tabular ML studio", copy: "Train classical scikit-learn pipelines after choosing a target, then predict from a generated form." },
  { icon: ShieldCheck, title: "Safe AI orchestration", copy: "The LLM explains and interprets; backend services do the math and model selection." },
];

export default function Overview() {
  const navigate = useNavigate();
  const { datasetProfile, setDataset, setAnalysis } = useWorkspace();
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [analysisFailed, setAnalysisFailed] = useState(false);

  async function upload(file?: File) {
    if (!file) return;
    setBusy(true);
    setAnalysisFailed(false);
    setMessage("Uploading CSV...");
    try {
      const profile = await api.upload(file);
      setDataset(profile);
      setMessage("Schema ready. Generating dashboard...");
      const analysis = await api.analyze(profile.dataset_id);
      setAnalysis(analysis);
      navigate(`/datasets/${analysis.dataset_id}/visualizations`);
    } catch (error) {
      setAnalysisFailed(true);
      setMessage(error instanceof Error ? error.message : "Upload or analysis failed");
    } finally {
      setBusy(false);
    }
  }

  async function startAnalysis() {
    if (!datasetProfile) return;
    setBusy(true);
    setAnalysisFailed(false);
    setMessage("Running EDA, chart scoring, insights, and RAG prep...");
    try {
      const analysis = await api.analyze(datasetProfile.dataset_id);
      setAnalysis(analysis);
      navigate(`/datasets/${analysis.dataset_id}/visualizations`);
    } catch (error) {
      setAnalysisFailed(true);
      setMessage(error instanceof Error ? error.message : "Analysis failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <PageFade className="hero-shell space-y-10 pb-10">
      <section className="hero-inner min-h-[680px] py-[clamp(1.5rem,3vw,3rem)]">
        <motion.div initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }}>
          <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-cyan-300/20 bg-cyan-300/8 px-3 py-1 text-sm text-cyan-100"><Sparkles size={15} /> Autonomous CSV analytics workspace</div>
          <h1 className="max-w-4xl text-[clamp(3rem,4.6vw,4.75rem)] font-semibold leading-tight tracking-normal">
            Turn any CSV into an <span className="accent-text">intelligent analytics workspace.</span>
          </h1>
          <p className="mt-6 max-w-2xl text-lg leading-8 text-slate-300">Upload structured data, generate adaptive visualizations, ask grounded questions, and train prediction models without writing code.</p>
          <div className="mt-8 flex flex-wrap gap-3">
            <label className="button-primary glow-button inline-flex cursor-pointer items-center gap-2 rounded-lg px-5 py-3 font-medium">
              <FileUp size={18} />
              {busy ? "Preparing Workspace..." : "Upload CSV"}
              <input disabled={busy} className="hidden" type="file" accept=".csv,text/csv" onChange={(event) => upload(event.target.files?.[0])} />
            </label>
            {datasetProfile && analysisFailed && (
              <GlowButton disabled={busy} onClick={startAnalysis}>
                Generate Dashboard <ArrowRight size={18} />
              </GlowButton>
            )}
          </div>
          {message && <p className="mt-4 text-sm text-cyan-100">{message}</p>}
        </motion.div>

        <motion.div initial={{ opacity: 0, scale: 0.96 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.12, duration: 0.7 }} className="relative mx-auto min-h-[520px] w-full max-w-[760px]">
          <motion.div animate={{ y: [0, -5, 0] }} transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }} className="premium-card absolute left-[10%] right-[8%] top-8 p-5">
            <div className="mb-4 flex items-center gap-2 text-sm text-cyan-100"><span className="h-2 w-2 rounded-full bg-emerald-300" /> CSV preview</div>
            <div className="space-y-3">
              {Array.from({ length: 6 }).map((_, row) => (
                <div key={row} className="grid grid-cols-4 gap-3">
                  {Array.from({ length: 4 }).map((__, col) => <span key={col} className={`h-3 rounded-full ${row === 0 ? "bg-cyan-200/35" : "bg-slate-500/20"}`} />)}
                </div>
              ))}
            </div>
          </motion.div>
          <motion.div animate={{ y: [0, 4, 0] }} transition={{ duration: 7, repeat: Infinity, ease: "easeInOut" }} className="premium-card premium-card-spotlight absolute left-[3%] top-44 w-[38%] p-5">
            <p className="text-sm text-slate-400">Dataset profile</p>
            <div className="mt-4 grid grid-cols-2 gap-3 text-sm"><span>Rows<br /><b>12.4k</b></span><span>Columns<br /><b>28</b></span><span>Missing<br /><b>3.8%</b></span><span>Types<br /><b>6</b></span></div>
          </motion.div>
          <motion.div animate={{ y: [0, -6, 0] }} transition={{ duration: 7.5, repeat: Infinity, ease: "easeInOut" }} className="premium-card absolute left-[32%] top-48 w-[54%] p-5">
            <p className="text-sm text-violet-100">Generated chart</p>
            <div className="mt-5 flex h-28 items-end gap-3">{[46, 76, 52, 95, 68, 84].map((h) => <span key={h} className="w-full rounded-t-md bg-cyan-300/70" style={{ height: `${h}%` }} />)}</div>
          </motion.div>
          <motion.div animate={{ y: [0, 5, 0] }} transition={{ duration: 8.5, repeat: Infinity, ease: "easeInOut" }} className="premium-card absolute bottom-16 left-[16%] w-[56%] p-5">
            <p className="text-sm text-cyan-100">AI insight</p>
            <p className="mt-3 text-sm leading-6 text-slate-300">Three columns explain most of the visible separation; review the top relationship chart first.</p>
          </motion.div>
          <div className="premium-card premium-card-subtle absolute bottom-0 left-[12%] right-[10%] flex flex-wrap items-center justify-center gap-2 px-4 py-3 text-sm text-slate-300">Upload <ArrowRight size={14} /> Analyze <ArrowRight size={14} /> Visualize <ArrowRight size={14} /> Ask <ArrowRight size={14} /> Predict</div>
        </motion.div>
      </section>

      {datasetProfile && analysisFailed && (
        <PremiumCard variant="spotlight" className="p-5">
          <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
            <div>
              <div className="mb-2 flex flex-wrap items-center gap-2"><h2 className="text-xl font-semibold">{datasetProfile.filename}</h2><StatusBadge tone="emerald">{datasetProfile.analysis_status}</StatusBadge></div>
              <p className="mt-1 text-sm text-slate-400">{datasetProfile.rows} rows, {datasetProfile.columns} columns. Analysis status: {datasetProfile.analysis_status}</p>
            </div>
            <GlowButton disabled={busy} onClick={startAnalysis}>
              Generate Dashboard <ArrowRight size={18} />
            </GlowButton>
          </div>
        </PremiumCard>
      )}

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {features.map((feature, index) => (
          <motion.div key={feature.title} initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.06 * index }}>
            <PremiumCard className="h-full p-5">
              <feature.icon className="mb-5 text-cyan-200" />
              <h3 className="font-semibold">{feature.title}</h3>
              <p className="mt-3 text-sm leading-6 text-slate-400">{feature.copy}</p>
              <div className="mt-5 h-px w-16 bg-gradient-to-r from-cyan-300 to-transparent" />
            </PremiumCard>
          </motion.div>
        ))}
      </section>
    </PageFade>
  );
}
