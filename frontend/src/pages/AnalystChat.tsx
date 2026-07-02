import { AnimatePresence, motion } from "framer-motion";
import { Bot, ChevronDown, Send, Sparkles, UserRound } from "lucide-react";
import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import { api, type ToolTrace } from "../api/client";
import { GlowButton, PageFade, StatusBadge } from "../components/Premium";
import { useWorkspace } from "../store/useWorkspace";

type EvidenceItem = { title: string; text: string };
type Message = { role: "user" | "assistant"; text: string; evidence?: EvidenceItem[]; debugEvidence?: EvidenceItem[]; toolTrace?: ToolTrace };

function parseToolTrace(debugEvidence?: EvidenceItem[]) {
  const trace = debugEvidence?.find((item) => item.title === "Tool trace");
  if (!trace) return null;
  try {
    return JSON.parse(trace.text) as Record<string, unknown>;
  } catch {
    return { raw_trace: trace.text };
  }
}

function pretty(value: unknown) {
  if (value === null || value === undefined) return "None";
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") return String(value);
  return JSON.stringify(value, null, 2);
}

function ToolTracePanel({ trace }: { trace: Record<string, unknown> | null }) {
  if (!trace) {
    return (
      <div className="rounded-lg border border-white/8 bg-white/[0.035] p-4">
        <p className="text-sm font-medium text-cyan-100">Tool Trace</p>
        <p className="mt-2 text-xs leading-5 text-slate-400">No tool trace was returned with this response.</p>
      </div>
    );
  }
  const toolResult = trace.tool_result as Record<string, unknown> | undefined;
  const actualTool = trace.actual_tool_called ?? toolResult?.tool ?? (trace.planned_tool_call as Record<string, unknown> | undefined)?.tool ?? "None";
  const ragUsed = Boolean(trace.rag_used);
  const rows: Array<[string, unknown]> = [
    ["User question", trace.user_question ?? trace.question],
    ["Detected intent", trace.detected_intent ?? trace.intent],
    ["Resolved columns", trace.resolved_columns ?? trace.mapped_columns],
    ["Extracted filters", trace.extracted_filters ?? trace.filters_extracted ?? trace.filters_applied],
    ["Planned tool call", trace.planned_tool_call],
    ["Actual tool executed", actualTool],
    ["Raw tool result", trace.tool_result ?? trace.result],
    ["RAG used", ragUsed ? "Yes" : "No"],
  ];
  return (
    <div className="rounded-lg border border-cyan-300/15 bg-cyan-300/[0.045] p-4">
      <p className="text-sm font-medium text-cyan-100">Tool Trace</p>
      <div className="mt-3 space-y-3">
        {rows.map(([label, value]) => (
          <div key={label}>
            <p className="text-[11px] uppercase tracking-wide text-slate-500">{label}</p>
            <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap rounded-md bg-black/25 p-2 text-xs leading-5 text-slate-300">{pretty(value)}</pre>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function AnalystChat() {
  const { datasetId } = useParams();
  const { datasetProfile, analysis } = useWorkspace();
  const [messages, setMessages] = useState<Message[]>([{ role: "assistant", text: "Ask about the computed dashboard, data quality, chart evidence, or patterns. I will stay grounded in the analysis artifacts." }]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [suggestionsOpen, setSuggestionsOpen] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const suggestions = useMemo(() => {
    const charts = analysis?.chart_specs.slice(0, 4) ?? [];
    if (!charts.length) return [];
    return charts.map((chart) => {
      const cols = chart.columns_used?.join(" and ") || chart.title.toLowerCase();
      if (chart.chart_type === "scatter" || chart.intent === "relationship") return `Explain the relationship between ${cols}`;
      if (chart.intent === "distribution") return `What does the ${chart.title.toLowerCase()} distribution reveal?`;
      if (chart.intent === "group comparison") return `Compare the groups in ${chart.title.toLowerCase()}`;
      return `What should I notice in ${chart.title}?`;
    });
  }, [analysis?.chart_specs]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ block: "end", behavior: "smooth" });
  }, [messages.length]);

  async function send(question: string) {
    if (!question.trim() || !datasetId) return;
    setMessages((m) => [...m, { role: "user", text: question }]);
    setInput("");
    setBusy(true);
    try {
      if (datasetId === "demo") {
        setMessages((m) => [...m, { role: "assistant", text: "In the demo workspace, start with the age distribution and class composition. The strongest evidence is chart-level and profile-level, not row-by-row raw data.", evidence: analysis?.chart_specs.map((c) => ({ title: c.title, text: c.insight })) ?? [] }]);
      } else {
        const response = await api.chat(datasetId, question);
        setMessages((m) => [...m, { role: "assistant", text: response.answer, evidence: response.evidence, debugEvidence: response.debug_evidence, toolTrace: response.tool_trace }]);
      }
    } catch (error) {
      setMessages((m) => [...m, { role: "assistant", text: error instanceof Error ? error.message : "I could not retrieve analysis context." }]);
    } finally {
      setBusy(false);
    }
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    send(input);
  }

  return (
    <PageFade className="chat-shell h-[calc(100vh-80px)] min-h-0 overflow-hidden pb-3">
      <section className="premium-card flex h-full min-h-0 flex-col">
        <div className="shrink-0 border-b border-white/10 bg-slate-950/45 p-5 backdrop-blur">
          <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            <div>
              <p className="text-sm text-cyan-100">AI Analyst Chat</p>
              <h1 className="mt-1 text-2xl font-semibold">{datasetProfile?.filename ?? "Dataset analysis"}</h1>
            </div>
            <StatusBadge tone="cyan"><Sparkles size={13} className="mr-1" /> Grounded analyst mode</StatusBadge>
          </div>
          {suggestions.length > 0 && (
            <div className="mt-4 rounded-xl border border-white/8 bg-black/20 p-3">
              <button onClick={() => setSuggestionsOpen((current) => !current)} className="flex w-full items-center justify-between text-sm text-slate-300">
                <span>Suggested investigations</span>
                <ChevronDown size={16} className={`transition ${suggestionsOpen ? "rotate-180" : ""}`} />
              </button>
              <AnimatePresence>
                {suggestionsOpen && (
                  <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }} className="overflow-hidden">
                    <div className="mt-3 flex flex-wrap gap-2">
                      {suggestions.map((q) => <button key={q} onClick={() => send(q)} className="rounded-full border border-white/10 px-3 py-1.5 text-xs text-slate-300 transition hover:border-cyan-300/30 hover:bg-cyan-300/8 hover:text-cyan-100">{q}</button>)}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          )}
        </div>
        <div className="min-h-0 flex-1 space-y-5 overflow-y-auto overscroll-contain p-5">
          <AnimatePresence initial={false}>
          {messages.map((message, index) => (
            <motion.div key={index} initial={{ opacity: 0, y: 14, scale: 0.98 }} animate={{ opacity: 1, y: 0, scale: 1 }} transition={{ duration: 0.22 }} className={`flex gap-3 ${message.role === "user" ? "justify-end" : ""}`}>
              {message.role === "assistant" && <span className="mt-1 flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-cyan-400/12 text-cyan-100"><Bot size={18} /></span>}
              <div className={`max-w-[880px] rounded-2xl px-4 py-3 text-sm leading-6 shadow-lg ${message.role === "user" ? "bg-violet-500/18 text-violet-50" : "bg-white/[0.055] text-slate-200"}`}>
                <p>{message.text}</p>
                {message.role === "assistant" && (message.evidence?.length || message.debugEvidence?.length || message.toolTrace) ? (
                  <details className="mt-3 rounded-xl border border-white/8 bg-black/20 p-3 text-xs text-slate-400">
                    <summary className="cursor-pointer text-cyan-100">Sources / Tool Trace</summary>
                    <div className="mt-3 space-y-3">
                      {message.evidence?.map((item) => (
                        <div key={item.title} className="rounded-lg border border-white/8 bg-white/[0.035] p-3">
                          <p className="font-medium text-slate-200">{item.title}</p>
                          <p className="mt-1 line-clamp-4 leading-5">{item.text}</p>
                        </div>
                      ))}
                      <ToolTracePanel trace={message.toolTrace ?? parseToolTrace(message.debugEvidence)} />
                      {!!message.debugEvidence?.filter((item) => item.title !== "Tool trace").length && (
                        <details className="rounded-lg border border-white/8 bg-black/20 p-3">
                          <summary className="cursor-pointer text-slate-300">Dev diagnostics</summary>
                          {message.debugEvidence.filter((item) => item.title !== "Tool trace").map((item) => (
                            <div key={item.title} className="mt-3">
                              <p className="font-medium text-amber-100/80">{item.title}</p>
                              <pre className="mt-2 max-h-52 overflow-auto whitespace-pre-wrap rounded-md bg-black/30 p-3">{item.text}</pre>
                            </div>
                          ))}
                        </details>
                      )}
                    </div>
                  </details>
                ) : null}
              </div>
              {message.role === "user" && <span className="mt-1 flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-violet-400/12 text-violet-100"><UserRound size={18} /></span>}
            </motion.div>
          ))}
          </AnimatePresence>
          {busy && (
            <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="flex gap-3">
              <span className="mt-1 flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-cyan-400/12 text-cyan-100"><Bot size={18} /></span>
              <div className="rounded-2xl border border-cyan-300/15 bg-cyan-300/[0.045] px-4 py-3 text-sm text-slate-300">
                <div className="flex items-center gap-2"><span>DataPilot is analyzing</span><span className="typing-dot h-1.5 w-1.5 rounded-full bg-cyan-200" /><span className="typing-dot h-1.5 w-1.5 rounded-full bg-cyan-200" /><span className="typing-dot h-1.5 w-1.5 rounded-full bg-cyan-200" /></div>
                <div className="mt-2 flex flex-wrap gap-2 text-xs text-slate-500"><span>Resolving columns</span><span>Running tools</span><span>Grounding answer</span></div>
              </div>
            </motion.div>
          )}
          <div ref={messagesEndRef} />
        </div>
        <div className="shrink-0 border-t border-white/10 bg-slate-950/60 p-4 backdrop-blur">
          <form onSubmit={submit} className="flex gap-3">
            <input value={input} onChange={(event) => setInput(event.target.value)} placeholder="Ask a grounded question about the analysis..." className="premium-input min-w-0 flex-1" />
            <GlowButton disabled={busy} type="submit" className="px-4" aria-label="Send"><Send size={18} /></GlowButton>
          </form>
        </div>
      </section>
    </PageFade>
  );
}
