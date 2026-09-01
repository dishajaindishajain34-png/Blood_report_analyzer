import { motion } from "motion/react";
import { ShieldCheck, ArrowRight } from "lucide-react";

import { Button } from "@/components/ui/button";
import { useCountUp } from "@/hooks/useCountUp";
import type { AnemiaRisk } from "@/types";
import { riskLabels } from "@/utils/format";

const tone = {
  low: { color: "var(--color-success)", chip: "bg-success/10 text-success border-success/30" },
  moderate: { color: "var(--color-warning)", chip: "bg-warning/15 text-warning border-warning/40" },
  high: {
    color: "var(--color-destructive)",
    chip: "bg-destructive/10 text-destructive border-destructive/30",
  },
} as const;

export function AnemiaRiskSection({
  risk,
  onDetails,
}: {
  risk: AnemiaRisk;
  onDetails?: () => void;
}) {
  const value = useCountUp(risk.score, 1100);
  const radius = 68;
  const circumference = 2 * Math.PI * radius;
  const t = tone[risk.level];
  const isFallbackNeeded = /insufficient|undetermined|re-scan|inconclusive|not enough RBC signal|consider re-scan/i.test(
    risk.explanation,
  );

  return (
    <section className="panel p-6" aria-labelledby="risk-heading">
      <h3 id="risk-heading" className="text-sm font-semibold">
        Anemia Risk Indication
      </h3>
      <p className="mt-1 text-xs text-muted-foreground">
        Screening indication derived from model output — not a diagnosis.
      </p>

      {isFallbackNeeded && (
        <div className="mt-4 rounded-lg border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-100">
          <span className="font-semibold">Anemia screening is inconclusive for this smear.</span> The current image did not provide enough reliable RBC signal for a confident anemia decision. Please upload a dedicated RBC/anemia-focused image for confirmation.
        </div>
      )}

      <div className="mt-6 grid gap-8 lg:grid-cols-[auto_minmax(0,1fr)] lg:items-center">
        <div className="mx-auto grid size-[180px] place-items-center">
          <svg viewBox="0 0 160 160" className="size-[180px] -rotate-90" role="img" aria-label={`Risk score ${risk.score} percent`}>
            <circle cx="80" cy="80" r={radius} fill="none" stroke="var(--color-muted)" strokeWidth="12" />
            <motion.circle
              cx="80"
              cy="80"
              r={radius}
              fill="none"
              stroke={t.color}
              strokeWidth="12"
              strokeLinecap="round"
              strokeDasharray={circumference}
              initial={{ strokeDashoffset: circumference }}
              animate={{ strokeDashoffset: circumference * (1 - risk.score / 100) }}
              transition={{ duration: 1.1, ease: "easeOut" }}
            />
          </svg>
          <div className="pointer-events-none absolute grid place-items-center text-center">
            <span className="text-3xl font-semibold tabular-nums">{value}%</span>
            <span className="mt-1 text-[11px] font-semibold tracking-[0.14em] uppercase" style={{ color: t.color }}>
              {risk.level} risk
            </span>
          </div>
        </div>

        <div className="min-w-0">
          <span className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-medium ${t.chip}`}>
            <ShieldCheck className="size-3.5" aria-hidden="true" />
            {riskLabels[risk.level]}
          </span>
          <p className="mt-3 text-sm leading-relaxed text-muted-foreground">{risk.explanation}</p>

          <ul className="mt-5 space-y-2.5">
            {risk.contributingFactors.map((f) => (
              <li key={f.factor} className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3">
                <div className="min-w-0">
                  <p className="truncate text-xs font-medium">{f.factor}</p>
                  <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-muted">
                    <motion.div
                      className="h-full rounded-full bg-secondary"
                      initial={{ width: 0 }}
                      animate={{ width: `${f.weight * 100}%` }}
                      transition={{ duration: 0.7, ease: "easeOut" }}
                    />
                  </div>
                </div>
                <span className="text-xs tabular-nums text-muted-foreground">
                  {(f.weight * 100).toFixed(0)}%
                </span>
              </li>
            ))}
          </ul>

          <p className="mt-5 rounded-lg border border-warning/40 bg-warning/10 px-4 py-3 text-xs leading-relaxed text-foreground">
            <span className="font-semibold">Important:</span> This AI screening result is not a
            substitute for CBC testing or professional medical evaluation. Please consult a
            qualified healthcare professional.
          </p>

          {onDetails && (
            <Button variant="outline" size="sm" className="mt-4" onClick={onDetails}>
              View Detailed Analysis
              <ArrowRight className="size-4" aria-hidden="true" />
            </Button>
          )}
        </div>
      </div>
    </section>
  );
}
