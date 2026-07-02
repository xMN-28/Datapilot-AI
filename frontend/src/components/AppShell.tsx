import { BarChart3, Bot, Boxes, Database, Download, Home, Sparkles } from "lucide-react";
import { motion } from "framer-motion";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { useWorkspace } from "../store/useWorkspace";
import { compactNumber } from "../lib/format";

const nav = [
  { label: "Overview", icon: Home, to: "/" },
  { label: "Visualizations", icon: BarChart3, page: "visualizations" },
  { label: "AI Analyst", icon: Bot, page: "chat" },
  { label: "Prediction Studio", icon: Boxes, page: "prediction" },
  { label: "Export / Share", icon: Download, page: "export" },
];

export default function AppShell() {
  const { datasetProfile, currentDatasetId } = useWorkspace();
  const location = useLocation();

  return (
    <div className="min-h-screen overflow-hidden text-slate-100">
      <div className="animated-grid pointer-events-none fixed inset-0 opacity-60" />
      <aside className="fixed inset-y-0 left-0 z-20 hidden w-72 flex-col border-r border-white/10 bg-[#050815]/88 p-5 shadow-[24px_0_80px_rgba(0,0,0,.28)] backdrop-blur-xl lg:flex">
        <div className="mb-8 flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-cyan-400/15 text-cyan-200 shadow-glow">
            <Sparkles size={22} />
          </div>
          <div>
            <div className="text-lg font-semibold">DataPilot AI</div>
            <div className="text-xs text-slate-400">Autonomous CSV analytics</div>
          </div>
        </div>

        <nav className="space-y-2">
          {nav.map((item) => {
            const Icon = item.icon;
            const to = item.to ?? (currentDatasetId ? `/datasets/${currentDatasetId}/${item.page}` : "/");
            const active = item.to ? location.pathname === "/" : location.pathname.endsWith(`/${item.page}`);
            return (
              <NavLink key={item.label} to={to} className={`group relative flex items-center gap-3 overflow-hidden rounded-xl px-3 py-3 text-sm transition ${active ? "bg-cyan-400/12 text-cyan-100 ring-1 ring-cyan-300/30 shadow-[0_0_24px_rgba(34,211,238,.08)]" : "text-slate-400 hover:bg-white/6 hover:text-slate-100"}`}>
                <span className="absolute inset-y-2 left-0 w-0.5 rounded-full bg-cyan-300 opacity-0 transition group-hover:opacity-60" />
                <Icon size={18} className="transition group-hover:translate-x-0.5" />
                {item.label}
              </NavLink>
            );
          })}
        </nav>

        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="premium-card premium-card-subtle mt-auto max-h-48 p-4">
          <div className="mb-3 flex items-center gap-2 text-sm font-medium text-cyan-100">
            <Database size={16} />
            Current dataset
          </div>
          {datasetProfile ? (
            <div className="space-y-2 text-sm text-slate-300">
              <p className="truncate text-slate-100">{datasetProfile.filename}</p>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <span>{compactNumber(datasetProfile.rows)} rows</span>
                <span>{compactNumber(datasetProfile.columns)} columns</span>
              </div>
              <span className="inline-flex rounded-full border border-cyan-300/25 px-2 py-1 text-xs text-cyan-100">{datasetProfile.analysis_status}</span>
            </div>
          ) : (
            <p className="text-sm text-slate-400">No dataset uploaded.</p>
          )}
        </motion.div>
      </aside>
      <main className="relative z-10 max-h-screen w-full overflow-y-auto lg:ml-72 lg:w-[calc(100vw-18rem)]">
        <div className="app-content">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
