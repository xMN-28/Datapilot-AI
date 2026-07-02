import { Bot, BrainCircuit, CheckCircle2, Circle, Download, Eraser, Play, Send, SlidersHorizontal, Timer, WandSparkles, XCircle } from "lucide-react";
import { AnimatePresence, motion } from "framer-motion";
import { ChangeEvent, FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import { api, downloadUrl, type ChatPredictResponse, type ModelFeature, type TrainingJob } from "../api/client";
import { CustomSelect, GlowButton, PageFade, PremiumCard, SectionHeader, StatusBadge } from "../components/Premium";
import { useWorkspace } from "../store/useWorkspace";

type ChatMessage = {
  role: "user" | "assistant";
  text: string;
  prediction?: Record<string, unknown>;
};

const placeholder = "Would this order be a high value order if the customer is Premium, product category is Electronics, unit price is 149.99, quantity is 5, discount is 10%, payment method is Credit Card, and shipping is Express?";
const trainingModes = [
  { value: "fast", label: "Fast", copy: "Quickest", detail: "Lowest latency model search." },
  { value: "balanced", label: "Balanced", copy: "Recommended", detail: "Multiple models with light validation." },
  { value: "deep", label: "Deep", copy: "Thorough", detail: "More validation and tuning." },
];

export default function PredictionStudio() {
  const { datasetId } = useParams();
  const { datasetProfile, trainedModel, setDataset, setModel, loadDemo, activeTrainingJobId, activeTrainingDatasetId, activeTrainingMode, activeTrainingTarget, setTrainingJob } = useWorkspace();
  const [target, setTarget] = useState("");
  const [mode, setMode] = useState("balanced");
  const [activeTab, setActiveTab] = useState<"chat" | "manual">("chat");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [values, setValues] = useState<Record<string, unknown>>({});
  const [prediction, setPrediction] = useState<Record<string, unknown> | null>(null);
  const [chatInput, setChatInput] = useState("");
  const [chatState, setChatState] = useState<Record<string, unknown>>({});
  const [trainingJob, setTrainingJobState] = useState<TrainingJob | null>(null);
  const [logFilter, setLogFilter] = useState<"all" | "warning" | "error">("all");
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      role: "assistant",
      text: "Describe a case naturally and I will extract feature values, ask only for what is missing, then run the trained model.",
    },
  ]);

  const targets = datasetProfile?.schema.columns.filter((c) => c.target_like_candidate) ?? [];
  const targetOptions = targets.map((col) => ({ value: col.name, label: col.name, description: `${col.unique_count} unique · ${col.missing_percentage.toFixed(1)}% missing`, badge: col.inferred_type }));
  const features = trainedModel?.feature_schema.features ?? [];
  const hasModernFeatureSchema = features.length > 0 && features.every((feature) => feature.feature_name && feature.display_name);
  const samplePrompt = useMemo(() => makeSamplePrompt(features, trainedModel?.target), [features, trainedModel?.target]);
  const visibleTraining = trainingJob && activeTrainingDatasetId === datasetId && trainingJob.status !== "complete";
  const setupCompact = Boolean(visibleTraining || trainedModel);

  useEffect(() => {
    if (!datasetId) return;
    if (datasetId === "demo") {
      if (datasetProfile?.dataset_id !== "demo") loadDemo();
      return;
    }
    if (datasetProfile?.dataset_id === datasetId) return;
    api.dataset(datasetId).then(setDataset).catch(() => undefined);
  }, [datasetId, datasetProfile?.dataset_id, loadDemo, setDataset]);

  useEffect(() => {
    if (!activeTrainingJobId || activeTrainingDatasetId !== datasetId) return;
    let stopped = false;

    async function poll() {
      try {
        const job = await api.trainingJob(activeTrainingJobId!);
        if (stopped) return;
        setTrainingJobState(job);
        if (job.status === "complete" && job.model_id && datasetId) {
          const model = await api.model(datasetId, job.model_id);
          if (!stopped) {
            setModel(model);
            setTrainingJob(null);
            setMessage("Model ready.");
          }
          return;
        }
        if (["failed", "cancelled"].includes(job.status)) return;
      } catch (error) {
        if (!stopped) setMessage(error instanceof Error ? error.message : "Could not poll training job.");
      }
    }

    poll();
    const timer = window.setInterval(poll, 1500);
    return () => {
      stopped = true;
      window.clearInterval(timer);
    };
  }, [activeTrainingJobId, activeTrainingDatasetId, datasetId, setModel, setTrainingJob]);

  async function train() {
    if (!datasetId || datasetId === "demo" || !target) {
      setMessage(datasetId === "demo" ? "Upload a real CSV to train a model. Demo mode previews the workflow." : "Choose a target column first.");
      return;
    }
    setBusy(true);
    setMessage("");
    try {
      const job = await api.train(datasetId, target, mode);
      setTrainingJob(job.job_id, datasetId, target, mode);
      setTrainingJobState(null);
      setActiveTab("chat");
      setChatState({});
      setChatMessages([{ role: "assistant", text: "Training has started. I will be ready for prediction once the model finishes." }]);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Training failed");
    } finally {
      setBusy(false);
    }
  }

  async function cancelTraining() {
    if (!activeTrainingJobId) return;
    await api.cancelTrainingJob(activeTrainingJobId);
  }

  async function predict() {
    if (!datasetId || !trainedModel) return;
    setBusy(true);
    try {
      const response = await api.predict(datasetId, trainedModel.model_id, values);
      setPrediction(response);
    } catch (error) {
      setPrediction({ error: error instanceof Error ? error.message : "Prediction failed" });
    } finally {
      setBusy(false);
    }
  }

  async function sendChat(event?: FormEvent) {
    event?.preventDefault();
    if (!datasetId || !trainedModel || !chatInput.trim()) return;
    const outgoing = chatInput.trim();
    setChatInput("");
    setChatMessages((current) => [...current, { role: "user", text: outgoing }]);
    setBusy(true);
    try {
      const response: ChatPredictResponse = await api.chatPredict(datasetId, trainedModel.model_id, outgoing, chatState);
      setChatState(response.conversation_state ?? {});
      setChatMessages((current) => [...current, { role: "assistant", text: response.assistant_message, prediction: response.prediction }]);
    } catch (error) {
      setChatMessages((current) => [...current, { role: "assistant", text: error instanceof Error ? error.message : "Chat prediction failed." }]);
    } finally {
      setBusy(false);
    }
  }

  async function batchPredict(file?: File) {
    if (!file || !datasetId || !trainedModel) return;
    const body = new FormData();
    body.append("file", file);
    setBusy(true);
    try {
      const response = await fetch(`/api/datasets/${datasetId}/models/${trainedModel.model_id}/batch-predict`, { method: "POST", body });
      if (!response.ok) throw new Error(await response.text());
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "batch_predictions.csv";
      link.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Batch prediction failed");
    } finally {
      setBusy(false);
    }
  }

  function clearConversation() {
    setChatInput("");
    setChatState({});
    setChatMessages([{ role: "assistant", text: "Prediction conversation cleared. Describe a new case when you are ready." }]);
  }

  function setValue(feature: ModelFeature, event: ChangeEvent<HTMLInputElement | HTMLSelectElement>) {
    setValues((current) => ({ ...current, [feature.feature_name]: event.target.value }));
  }

  function setFeatureValue(feature: ModelFeature, value: string) {
    setValues((current) => ({ ...current, [feature.feature_name]: value }));
  }

  return (
    <PageFade className="prediction-shell space-y-6 pb-10">
      <motion.div layout className={setupCompact ? "" : "flex min-h-[calc(100vh-170px)] items-center justify-center"}>
      <motion.div layout transition={{ duration: 0.38, ease: "easeOut" }} className={setupCompact ? "mx-auto w-full max-w-[1450px]" : "w-full max-w-[1180px]"}>
      <SectionHeader eyebrow="Prediction Studio" title="Choose what you want to predict." copy="Train a guarded tabular model, then predict with chat or a precise manual form." className={setupCompact ? "mb-5" : "mb-6 text-center [&>p]:mx-auto"} />

      <PremiumCard variant="spotlight" className="p-5 sm:p-6">
        <div className={setupCompact ? "grid gap-5 xl:grid-cols-[minmax(280px,1fr)_1.4fr_160px] xl:items-end" : "grid gap-6"}>
          <CustomSelect label="Target variable" value={target} onChange={setTarget} placeholder="Select target column" options={targetOptions} searchable />
          <div>
            <span className="mb-2 block text-sm text-slate-400">Training mode</span>
            <div className="grid gap-3 md:grid-cols-3">
              {trainingModes.map((item) => (
                <button key={item.value} onClick={() => setMode(item.value)} className={`rounded-xl border p-3 text-left transition ${mode === item.value ? "border-cyan-300/40 bg-cyan-300/10 shadow-glow" : "border-white/10 bg-black/20 hover:border-cyan-300/25"}`}>
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-medium text-slate-100">{item.label}</span>
                    <StatusBadge tone={item.value === "balanced" ? "emerald" : "slate"}>{item.copy}</StatusBadge>
                  </div>
                  <p className="mt-2 text-xs leading-5 text-slate-500">{item.detail}</p>
                </button>
              ))}
            </div>
          </div>
          <GlowButton disabled={busy} onClick={train} className={setupCompact ? "h-[52px]" : "mx-auto h-[52px] w-full max-w-xs"}>
            <Play size={17} /> Train
          </GlowButton>
        </div>
        {message && <p className="mt-4 text-sm text-cyan-100">{message}</p>}
      </PremiumCard>
      </motion.div>
      </motion.div>

      <AnimatePresence>
        {visibleTraining && (
          <motion.div initial={{ opacity: 0, y: 22 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 12 }} transition={{ duration: 0.32 }}>
            <TrainingMonitor
              job={trainingJob}
              target={activeTrainingTarget ?? target}
              mode={activeTrainingMode ?? mode}
              rows={datasetProfile?.rows}
              columns={datasetProfile?.columns}
              logFilter={logFilter}
              setLogFilter={setLogFilter}
              onCancel={cancelTraining}
              onRetry={train}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {trainedModel && (
        <section className="mx-auto grid max-w-[1500px] gap-5 xl:grid-cols-[390px_minmax(0,1fr)]">
          <PremiumCard className="p-5">
            <BrainCircuit className="mb-4 text-cyan-200" />
            <h2 className="text-xl font-semibold">Model ready</h2>
            <div className="mt-4 space-y-2 text-sm text-slate-300">
              <p>Target: <b>{trainedModel.target}</b></p>
              <p>Task: <b>{trainedModel.task_type}</b></p>
              <p>Selected model: <b>{trainedModel.selected_model}</b></p>
              <p>Validation: <b>{trainedModel.validation_method ?? "train/test split"}</b></p>
              <p>Features used: <b>{trainedModel.features_used}</b></p>
            </div>
            {!!trainedModel.warnings?.length && (
              <div className="mt-4 rounded-lg border border-amber-300/20 bg-amber-300/8 p-3 text-sm text-amber-100">
                {trainedModel.warnings[0]}
              </div>
            )}
            {!!trainedModel.leakage_warnings?.length && (
              <div className="mt-3 rounded-lg border border-rose-300/20 bg-rose-300/8 p-3 text-sm text-rose-100">
                Possible target leakage detected. Suspicious columns were excluded by default.
              </div>
            )}
            <button onClick={() => downloadUrl(`/api/datasets/${datasetId}/models/${trainedModel.model_id}/download`)} className="button-secondary mt-5 inline-flex w-full items-center justify-center gap-2 rounded-lg px-4 py-3">
              <Download size={17} /> Download Model Bundle
            </button>
            <label className="button-secondary mt-3 inline-flex w-full cursor-pointer items-center justify-center gap-2 rounded-lg px-4 py-3 text-sm">
              <Download size={17} /> Batch Prediction CSV
              <input className="hidden" type="file" accept=".csv,text/csv" onChange={(event) => batchPredict(event.target.files?.[0])} />
            </label>
            <details className="mt-5 rounded-lg border border-white/10 bg-black/20 p-4">
              <summary className="flex cursor-pointer items-center gap-2 text-sm font-medium"><SlidersHorizontal size={16} /> Stats for Nerds</summary>
              <pre className="mt-3 max-h-96 overflow-auto text-xs text-slate-400">{JSON.stringify(trainedModel, null, 2)}</pre>
            </details>
          </PremiumCard>

          <PremiumCard className="p-5">
            <div className="relative mb-5 inline-flex rounded-xl border border-white/10 bg-black/25 p-1">
              <motion.span layout className={`absolute bottom-1 top-1 rounded-lg bg-cyan-400/15 ${activeTab === "chat" ? "left-1 w-[132px]" : "left-[137px] w-[148px]"}`} />
              <button onClick={() => setActiveTab("chat")} className={`relative rounded-lg px-4 py-2 text-sm transition ${activeTab === "chat" ? "text-cyan-100" : "text-slate-400 hover:text-slate-100"}`}>Chat Prediction</button>
              <button onClick={() => setActiveTab("manual")} className={`relative rounded-lg px-4 py-2 text-sm transition ${activeTab === "manual" ? "text-cyan-100" : "text-slate-400 hover:text-slate-100"}`}>Manual Prediction</button>
            </div>

            {!hasModernFeatureSchema ? (
              <div className="rounded-lg border border-amber-300/20 bg-amber-300/8 p-4 text-sm text-amber-100">
                This model was trained before the upgraded prediction schema. Retrain the model to enable Chat Prediction and the cleaned manual form.
              </div>
            ) : activeTab === "chat" ? (
              <motion.div key="chat-prediction" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.24 }}>
                <div className="mb-4 flex items-start justify-between gap-4">
                  <div>
                    <h2 className="flex items-center gap-2 text-xl font-semibold"><Bot size={20} className="text-cyan-200" /> Ask DataPilot to predict</h2>
                    <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-400">Describe a case naturally. DataPilot will extract the needed feature values, ask for anything missing, then run the trained model.</p>
                  </div>
                  <button onClick={() => setActiveTab("manual")} className="text-sm text-cyan-100 hover:text-cyan-50">Switch to manual form</button>
                </div>

                {!trainedModel.chat_prediction_enabled ? (
                  <div className="rounded-lg border border-amber-300/20 bg-amber-300/8 p-4 text-sm text-amber-100">
                    Chat Prediction requires an OpenAI API key because it uses the LLM to extract feature values from natural language.
                  </div>
                ) : (
                  <>
                    <div className="mb-4 flex flex-wrap gap-2">
                      <button onClick={() => setChatInput(samplePrompt)} className="button-secondary inline-flex items-center gap-2 rounded-lg px-3 py-2 text-sm"><WandSparkles size={15} /> Use sample case</button>
                      <button onClick={clearConversation} className="button-secondary inline-flex items-center gap-2 rounded-lg px-3 py-2 text-sm"><Eraser size={15} /> Clear prediction conversation</button>
                    </div>
                    <div className="min-h-[380px] space-y-3 rounded-xl border border-white/10 bg-black/25 p-4">
                      {chatMessages.map((item, index) => (
                        <motion.div key={index} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className={`flex ${item.role === "user" ? "justify-end" : "justify-start"}`}>
                          <div className={`max-w-3xl rounded-lg px-4 py-3 text-sm leading-6 ${item.role === "user" ? "bg-violet-500/18 text-violet-50" : "bg-white/[0.05] text-slate-200"}`}>
                            <p>{item.text}</p>
                            {item.prediction && <pre className="mt-3 overflow-auto rounded-md bg-black/25 p-3 text-xs text-cyan-50">{JSON.stringify(item.prediction, null, 2)}</pre>}
                          </div>
                        </motion.div>
                      ))}
                    </div>
                    <form onSubmit={sendChat} className="mt-4 flex gap-3">
                      <input value={chatInput} onChange={(event) => setChatInput(event.target.value)} placeholder={placeholder} className="premium-input min-w-0 flex-1" />
                      <GlowButton disabled={busy} type="submit" className="px-4" aria-label="Send prediction chat"><Send size={18} /></GlowButton>
                    </form>
                  </>
                )}
              </motion.div>
            ) : (
              <motion.div key="manual-prediction" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.24 }}>
                <h2 className="text-xl font-semibold">Manual Prediction</h2>
                <p className="mt-2 text-sm text-slate-400">This form uses only the features included in the final trained pipeline.</p>
                <div className="mt-5 grid gap-4 md:grid-cols-2">
                  {features.map((feature) => (
                    <label key={feature.feature_name} className="space-y-2">
                      <span className="text-sm text-slate-400">{feature.display_name}</span>
                      {feature.type === "numeric" ? (
                        <input type="number" onChange={(event) => setValue(feature, event)} className="premium-input w-full" />
                      ) : (
                        <CustomSelect value={String(values[feature.feature_name] ?? "")} onChange={(value) => setFeatureValue(feature, value)} placeholder="Select value" options={(feature.allowed_values ?? feature.categories ?? []).map((category) => ({ value: category, label: category }))} searchable={(feature.allowed_values ?? feature.categories ?? []).length > 8} />
                      )}
                    </label>
                  ))}
                </div>
                <GlowButton onClick={predict} disabled={busy} className="mt-5">Run Prediction</GlowButton>
                {prediction && <pre className="mt-5 overflow-auto rounded-lg border border-cyan-300/15 bg-cyan-300/5 p-4 text-sm text-cyan-50">{JSON.stringify(prediction, null, 2)}</pre>}
              </motion.div>
            )}
          </PremiumCard>
        </section>
      )}
    </PageFade>
  );
}

function TrainingMonitor({ job, target, mode, rows, columns, logFilter, setLogFilter, onCancel, onRetry }: {
  job: TrainingJob;
  target: string;
  mode: string;
  rows?: number;
  columns?: number;
  logFilter: "all" | "warning" | "error";
  setLogFilter: (value: "all" | "warning" | "error") => void;
  onCancel: () => void;
  onRetry: () => void;
}) {
  const logRef = useRef<HTMLDivElement | null>(null);
  const filteredLogs = job.logs.filter((log) => logFilter === "all" || log.level === logFilter || (logFilter === "error" && log.level === "error"));
  const slowMessage = job.elapsed_seconds > 120 ? "Still training. You can keep waiting or cancel and retry in Fast mode." : job.elapsed_seconds > 30 ? "This is taking longer than usual. Large datasets, high-cardinality categorical columns, or heavy models can increase training time." : null;

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: "smooth" });
  }, [filteredLogs.length]);

  return (
    <section className="premium-card p-0">
      <div className="border-b border-white/10 bg-cyan-300/[0.03] p-6">
        <div className="flex flex-col gap-5 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <p className="text-sm text-cyan-100">Training Monitor</p>
            <h2 className="mt-2 text-3xl font-semibold">Training your prediction model</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-400">DataPilot is preparing features, training candidate models, and selecting the best pipeline.</p>
          </div>
          <div className="flex gap-2">
            {job.status === "failed" && <button onClick={onRetry} className="button-primary rounded-lg px-4 py-2 text-sm">Retry</button>}
            <button onClick={onCancel} disabled={job.status !== "running"} className="button-secondary rounded-lg px-4 py-2 text-sm disabled:cursor-not-allowed disabled:opacity-45">Cancel training</button>
          </div>
        </div>

        <div className="mt-6">
          <div className="mb-2 flex items-center justify-between text-sm">
            <span className="text-slate-300">{job.current_step}</span>
            <span className="font-semibold text-cyan-100">{job.progress}%</span>
          </div>
          <div className="h-3 overflow-hidden rounded-full bg-white/8">
            <div className="h-full rounded-full bg-gradient-to-r from-cyan-300 via-violet-400 to-emerald-300 shadow-glow transition-all duration-500" style={{ width: `${job.progress}%` }} />
          </div>
        </div>

        <div className="mt-5 grid gap-3 md:grid-cols-5">
          <MiniStat label="Target" value={target || "Selected target"} />
          <MiniStat label="Mode" value={modeLabel(mode)} />
          <MiniStat label="Rows" value={rows?.toLocaleString() ?? "-"} />
          <MiniStat label="Columns" value={columns?.toLocaleString() ?? "-"} />
          <MiniStat label="Elapsed" value={`${Math.round(job.elapsed_seconds)}s`} />
        </div>
        {slowMessage && <div className="mt-4 rounded-lg border border-amber-300/20 bg-amber-300/8 p-3 text-sm text-amber-100">{slowMessage}</div>}
      </div>

      <div className="grid gap-5 p-6 xl:grid-cols-[1fr_420px]">
        <div className="space-y-5">
          <div className="grid gap-3 md:grid-cols-3 xl:grid-cols-6">
            {compactSteps(job.steps).map((step, index) => <PipelineNode key={step.name} name={step.name} status={step.status} index={index} />)}
          </div>

          <div>
            <h3 className="mb-3 text-lg font-semibold">Candidate Models</h3>
            <div className="grid gap-3 md:grid-cols-2">
              {job.models.length ? job.models.map((model) => <ModelCard key={model.name} model={model} />) : <div className="rounded-lg border border-white/10 bg-white/[0.035] p-4 text-sm text-slate-400">Candidate models will appear after feature preparation.</div>}
            </div>
          </div>
        </div>

        <div className="rounded-lg border border-white/10 bg-black/30 p-4">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="font-semibold">Live Logs</h3>
            <div className="flex rounded-md border border-white/10 bg-black/25 p-1 text-xs">
              {(["all", "warning", "error"] as const).map((value) => <button key={value} onClick={() => setLogFilter(value)} className={`rounded px-2 py-1 ${logFilter === value ? "bg-cyan-400/15 text-cyan-100" : "text-slate-500"}`}>{value}</button>)}
            </div>
          </div>
          <div ref={logRef} className="h-[430px] overflow-auto rounded-md bg-[#030611] p-3 font-mono text-xs leading-5">
            {filteredLogs.map((log, index) => <div key={index} className={logTone(log.level)}><span className="text-slate-600">{log.time}</span> {log.message}</div>)}
          </div>
        </div>
      </div>
    </section>
  );
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return <div className="rounded-lg border border-white/10 bg-white/[0.035] p-3"><div className="text-xs text-slate-500">{label}</div><div className="mt-1 truncate text-sm font-semibold text-slate-100">{value}</div></div>;
}

function compactSteps(steps: TrainingJob["steps"]) {
  const wanted = ["Task detection", "Leakage check", "Preprocessing setup", "Model training", "Model comparison", "Preparing prediction workspace"];
  return wanted.map((name) => steps.find((step) => step.name === name) ?? { name, status: "pending" as const });
}

function PipelineNode({ name, status, index }: { name: string; status: string; index: number }) {
  const Icon = status === "complete" ? CheckCircle2 : status === "failed" ? XCircle : status === "running" ? Timer : Circle;
  return (
    <motion.div layout className={`rounded-lg border p-3 ${status === "running" ? "border-cyan-300/35 bg-cyan-300/8 shadow-glow" : status === "complete" ? "border-emerald-300/20 bg-emerald-300/6" : status === "failed" ? "border-rose-300/25 bg-rose-300/8" : "border-white/10 bg-white/[0.025]"}`}>
      <div className="mb-3 flex items-center justify-between">
        <Icon size={17} className={status === "complete" ? "text-emerald-200" : status === "failed" ? "text-rose-200" : status === "running" ? "text-cyan-200" : "text-slate-600"} />
        <span className="text-xs text-slate-600">0{index + 1}</span>
      </div>
      <div className="text-sm font-medium">{name}</div>
      <div className="mt-1 text-xs capitalize text-slate-500">{status}</div>
    </motion.div>
  );
}

function ModelCard({ model }: { model: TrainingJob["models"][number] }) {
  const metric = model.metrics ? Object.entries(model.metrics).find(([key, value]) => ["r2", "accuracy", "f1_weighted", "mae", "rmse"].includes(key) && typeof value === "number") : null;
  return (
    <motion.div layout whileHover={{ y: -2 }} className="rounded-lg border border-white/10 bg-white/[0.035] p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h4 className="font-medium">{model.name}</h4>
          <p className="mt-1 text-xs capitalize text-slate-500">Status: {model.status}</p>
        </div>
        <span className={`rounded-full px-2 py-1 text-xs ${model.status === "complete" ? "bg-emerald-300/10 text-emerald-100" : model.status === "running" ? "bg-cyan-300/10 text-cyan-100" : model.status === "failed" || model.status === "timeout" ? "bg-rose-300/10 text-rose-100" : "bg-white/8 text-slate-400"}`}>{model.status}</span>
      </div>
      <div className="mt-4 grid grid-cols-2 gap-2 text-xs text-slate-400">
        <span>Time: {model.duration_seconds ? `${model.duration_seconds}s` : "-"}</span>
        <span>{metric ? `${metric[0]}: ${metric[1]}` : "Metric: -"}</span>
      </div>
      {model.error && <p className="mt-3 text-xs text-rose-200">{model.error}</p>}
    </motion.div>
  );
}

function logTone(level: string) {
  if (level === "error") return "text-rose-200";
  if (level === "warning") return "text-amber-200";
  if (level === "success") return "text-emerald-200";
  return "text-slate-300";
}

function modeLabel(mode: string) {
  if (mode === "fast") return "Fast · Quick model search, lowest latency";
  if (mode === "deep") return "Deep · More validation/tuning";
  return "Balanced · Multiple models with light validation";
}

function makeSamplePrompt(features: ModelFeature[], target?: string) {
  const picked = features.map((feature) => {
    const lower = feature.feature_name.toLowerCase();
    if (feature.type === "numeric") {
      const value = lower.includes("price") ? "149.99" : lower.includes("discount") ? "10" : lower.includes("age") ? "34" : lower.includes("cost") ? "8.50" : lower.includes("rating") ? "4" : "5";
      return `${feature.display_name} is ${value}`;
    }
    if (feature.type === "boolean") return `${feature.display_name} is yes`;
    const value = (feature.allowed_values ?? feature.categories ?? [])[0] ?? "example";
    return `${feature.display_name} is ${value}`;
  });
  return `Would this case be predicted as ${target ?? "the target"} if ${picked.join(", ")}?`;
}
