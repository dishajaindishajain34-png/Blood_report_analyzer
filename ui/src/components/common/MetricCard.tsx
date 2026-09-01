import { motion } from "motion/react";
import type { LucideIcon } from "lucide-react";

import { useCountUp } from "@/hooks/useCountUp";
import { cn } from "@/lib/utils";

export function MetricCard({
  label,
  value,
  suffix = "",
  decimals = 0,
  icon: Icon,
  hint,
  index = 0,
}: {
  label: string;
  value: number;
  suffix?: string;
  decimals?: number;
  icon: LucideIcon;
  hint?: string;
  index?: number;
}) {
  const animated = useCountUp(value, 900, decimals);

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, delay: index * 0.06, ease: "easeOut" }}
      className="panel p-5 transition-shadow hover:shadow-[var(--shadow-raised)]"
    >
      <div className="flex items-start justify-between gap-3">
        <p className="text-xs font-medium tracking-wide text-muted-foreground uppercase">{label}</p>
        <span className="grid size-8 shrink-0 place-items-center rounded-lg bg-secondary/10 text-secondary">
          <Icon className="size-4" aria-hidden="true" />
        </span>
      </div>
      <p className="mt-3 text-3xl font-semibold tracking-tight tabular-nums">
        {animated.toFixed(decimals)}
        <span className="text-xl text-muted-foreground">{suffix}</span>
      </p>
      {hint && <p className="mt-1 text-xs text-muted-foreground">{hint}</p>}
    </motion.div>
  );
}

export function StatTile({
  label,
  value,
  className,
}: {
  label: string;
  value: string;
  className?: string;
}) {
  return (
    <div className={cn("rounded-lg border border-border bg-muted/40 px-4 py-3", className)}>
      <p className="text-[11px] font-medium tracking-wide text-muted-foreground uppercase">
        {label}
      </p>
      <p className="mt-1 text-sm font-semibold tabular-nums">{value}</p>
    </div>
  );
}
