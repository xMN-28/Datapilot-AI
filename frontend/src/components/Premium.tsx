import { AnimatePresence, motion } from "framer-motion";
import { Check, ChevronDown, Search } from "lucide-react";
import { KeyboardEvent, ReactNode, useEffect, useMemo, useRef, useState } from "react";

export function PageFade({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 8 }} transition={{ duration: 0.35, ease: "easeOut" }} className={className}>
      {children}
    </motion.div>
  );
}

export function PremiumCard({ children, className = "", variant = "elevated" }: { children: ReactNode; className?: string; variant?: "subtle" | "elevated" | "spotlight" | "compact" }) {
  return (
    <motion.div
      whileHover={{ y: variant === "compact" ? -2 : -4 }}
      transition={{ duration: 0.22, ease: "easeOut" }}
      className={`premium-card premium-card-${variant} ${className}`}
    >
      {children}
    </motion.div>
  );
}

export function GlowButton({ children, className = "", disabled, onClick, type = "button", "aria-label": ariaLabel }: { children: ReactNode; className?: string; disabled?: boolean; onClick?: () => void; type?: "button" | "submit"; "aria-label"?: string }) {
  return (
    <motion.button aria-label={ariaLabel} type={type} whileTap={{ scale: 0.98 }} disabled={disabled} onClick={onClick} className={`button-primary glow-button inline-flex items-center justify-center gap-2 rounded-lg px-5 py-3 font-medium disabled:cursor-not-allowed disabled:opacity-55 ${className}`}>
      {children}
    </motion.button>
  );
}

export function SectionHeader({ eyebrow, title, copy, className = "" }: { eyebrow: string; title: string; copy?: string; className?: string }) {
  return (
    <div className={className}>
      <p className="text-sm font-medium text-cyan-100">{eyebrow}</p>
      <h1 className="mt-2 text-3xl font-semibold tracking-normal text-slate-50 sm:text-4xl">{title}</h1>
      {copy && <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-400">{copy}</p>}
    </div>
  );
}

export function StatusBadge({ children, tone = "cyan" }: { children: ReactNode; tone?: "cyan" | "violet" | "emerald" | "amber" | "rose" | "slate" }) {
  const tones = {
    cyan: "border-cyan-300/25 bg-cyan-300/10 text-cyan-100",
    violet: "border-violet-300/25 bg-violet-300/10 text-violet-100",
    emerald: "border-emerald-300/25 bg-emerald-300/10 text-emerald-100",
    amber: "border-amber-300/25 bg-amber-300/10 text-amber-100",
    rose: "border-rose-300/25 bg-rose-300/10 text-rose-100",
    slate: "border-white/10 bg-white/8 text-slate-300",
  };
  return <span className={`inline-flex items-center rounded-full border px-2.5 py-1 text-xs ${tones[tone]}`}>{children}</span>;
}

export type SelectOption = {
  value: string;
  label: string;
  description?: string;
  badge?: string;
};

export function CustomSelect({ label, value, options, placeholder = "Select", onChange, searchable = false }: {
  label?: string;
  value: string;
  options: SelectOption[];
  placeholder?: string;
  onChange: (value: string) => void;
  searchable?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const rootRef = useRef<HTMLDivElement | null>(null);
  const selected = options.find((option) => option.value === value);
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return options;
    return options.filter((option) => `${option.label} ${option.description ?? ""} ${option.badge ?? ""}`.toLowerCase().includes(q));
  }, [options, query]);

  useEffect(() => {
    if (!open) return;
    function onPointerDown(event: PointerEvent) {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    }
    window.addEventListener("pointerdown", onPointerDown);
    return () => window.removeEventListener("pointerdown", onPointerDown);
  }, [open]);

  function commit(nextValue: string) {
    onChange(nextValue);
    setOpen(false);
    setQuery("");
  }

  function onKeyDown(event: KeyboardEvent<HTMLButtonElement>) {
    if (event.key === "Escape") setOpen(false);
    if (event.key === "ArrowDown" || event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      setOpen(true);
    }
  }

  return (
    <div ref={rootRef} className="relative">
      {label && <span className="mb-2 block text-sm text-slate-400">{label}</span>}
      <button type="button" aria-haspopup="listbox" aria-expanded={open} onKeyDown={onKeyDown} onClick={() => setOpen((current) => !current)} className="premium-select-trigger">
        <span className="min-w-0">
          <span className={`block truncate ${selected ? "text-slate-100" : "text-slate-500"}`}>{selected?.label ?? placeholder}</span>
          {selected?.description && <span className="mt-0.5 block truncate text-xs text-slate-500">{selected.description}</span>}
        </span>
        <span className="ml-auto flex items-center gap-2">
          {selected?.badge && <StatusBadge tone="violet">{selected.badge}</StatusBadge>}
          <ChevronDown size={17} className={`text-slate-400 transition ${open ? "rotate-180 text-cyan-100" : ""}`} />
        </span>
      </button>
      <AnimatePresence>
        {open && (
          <motion.div role="listbox" initial={{ opacity: 0, y: 8, scale: 0.98 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: 6, scale: 0.98 }} transition={{ duration: 0.18 }} className="premium-select-menu">
            {searchable && (
              <div className="relative mb-2">
                <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
                <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search columns..." className="premium-input w-full pl-9 text-sm" />
              </div>
            )}
            <div className="max-h-72 overflow-auto pr-1">
              {filtered.map((option) => (
                <button
                  type="button"
                  key={option.value}
                  role="option"
                  aria-selected={option.value === value}
                  onClick={() => commit(option.value)}
                  className="premium-select-option"
                >
                  <span className="min-w-0">
                    <span className="block truncate text-sm text-slate-100">{option.label}</span>
                    {option.description && <span className="mt-0.5 block truncate text-xs text-slate-500">{option.description}</span>}
                  </span>
                  <span className="ml-auto flex items-center gap-2">
                    {option.badge && <StatusBadge tone="slate">{option.badge}</StatusBadge>}
                    {option.value === value && <Check size={16} className="text-cyan-200" />}
                  </span>
                </button>
              ))}
              {!filtered.length && <p className="px-3 py-4 text-sm text-slate-500">No matches.</p>}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
