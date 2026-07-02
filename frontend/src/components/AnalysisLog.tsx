import { CheckCircle2 } from "lucide-react";
import type { LogItem } from "../api/client";

export default function AnalysisLog({ items }: { items: LogItem[] }) {
  return (
    <details className="premium-card premium-card-subtle p-4">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-3 text-sm font-medium text-slate-200">
        <span className="flex items-center gap-2"><CheckCircle2 className="text-emerald-300" size={17} /> Pipeline completed</span>
        <span className="rounded-full border border-white/10 bg-white/8 px-2 py-1 text-xs text-slate-400">View analysis log</span>
      </summary>
      <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {items.map((item) => (
          <div key={item.step} className="flex items-center gap-3 rounded-lg border border-white/8 bg-black/20 p-3 text-sm text-slate-300">
            <CheckCircle2 className="shrink-0 text-emerald-300" size={17} />
            <span className="min-w-0 flex-1 truncate">{item.step}</span>
            <span className="rounded-full bg-emerald-400/10 px-2 py-1 text-xs text-emerald-200">{item.status}</span>
          </div>
        ))}
      </div>
    </details>
  );
}
