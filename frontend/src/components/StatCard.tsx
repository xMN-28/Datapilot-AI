import type { LucideIcon } from "lucide-react";
import { PremiumCard } from "./Premium";

type Props = {
  label: string;
  value: string | number;
  icon: LucideIcon;
  tone?: "cyan" | "violet" | "emerald" | "amber";
};

const tones = {
  cyan: "text-cyan-200 bg-cyan-400/10",
  violet: "text-violet-200 bg-violet-400/10",
  emerald: "text-emerald-200 bg-emerald-400/10",
  amber: "text-amber-200 bg-amber-400/10",
};

export default function StatCard({ label, value, icon: Icon, tone = "cyan" }: Props) {
  return (
    <PremiumCard variant="compact" className="p-4">
      <div className="mb-4 flex items-center justify-between">
        <span className="text-sm text-slate-400">{label}</span>
        <span className={`rounded-md p-2 ${tones[tone]}`}><Icon size={16} /></span>
      </div>
      <div className="text-2xl font-semibold tracking-normal">{value}</div>
    </PremiumCard>
  );
}
