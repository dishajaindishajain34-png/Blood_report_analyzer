import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { createFileRoute } from "@tanstack/react-router";
import {
  Download, Clock, Hash, CalendarDays, ChevronDown, ChevronUp,
  Microscope, Activity, Droplets, FlaskConical, AlertTriangle, Info,
  GitMerge, Layers,
} from "lucide-react";

import { ImageViewer } from "@/components/analysis/ImageViewer";
import { ReportPreviewDialog } from "@/components/reports/ReportPreviewDialog";
import { ErrorState, PageHeader } from "@/components/common/States";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { api, queryKeys } from "@/services/api";
import { formatDateTime, formatPercent } from "@/utils/format";

// ──────────────────────────────────────────────────────────────────────────────
// Route
// ──────────────────────────────────────────────────────────────────────────────

export const Route = createFileRoute("/_app/results/$analysisId")({
  head: () => ({
    meta: [
      { title: "Analysis Results — HemaVision AI" },
      { name: "description", content: "Blood cell detection and CBC-based anemia screening results." },
    ],
  }),
  component: ResultsPage,
  errorComponent: () => <ErrorState />,
});

// ──────────────────────────────────────────────────────────────────────────────
// Small helpers
// ──────────────────────────────────────────────────────────────────────────────

function Meta({ icon: Icon, label, value }: { icon: typeof Clock; label: string; value: string }) {
  return (
    <div className="flex items-center gap-2">
      <Icon className="size-4 shrink-0 text-muted-foreground" />
      <div>
        <p className="text-[11px] tracking-wide text-muted-foreground uppercase">{label}</p>
        <p className="text-sm font-medium">{value}</p>
      </div>
    </div>
  );
}

/** Simple horizontal probability bar */
function ProbBar({ value, color }: { value: number; color: string }) {
  const pct = Math.round(value * 100);
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs font-medium" style={{ color }}>
        <span>Anemia probability</span>
        <span className="tabular-nums">{pct}%</span>
      </div>
      <div className="h-2.5 w-full rounded-full bg-white/10 overflow-hidden">
        <div
          className="h-full rounded-full transition-all"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
    </div>
  );
}

// ──────────────────────────────────────────────────────────────────────────────
// Section A — Final Multimodal Anemia Assessment (hero card)
// ──────────────────────────────────────────────────────────────────────────────

function FusionAnemiaPanel({ data }: { data: any }) {
  const f = data?.anemiaFusion;
  const hasResult = f && f.status === "success";

  const isGmu =
    f?.fusionStrategy?.includes("gmu") ||
    f?.fusionMode === "experimental_gmu" ||
    Boolean(f?.isExperimental);
  const isStacking =
    f?.fusionStrategy === "learned_stacking_ensemble" ||
    f?.fusionStrategy === "stacking_ensemble";
  const isLearned =
    isStacking ||
    f?.fusionStrategy === "learned_logistic_regression" ||
    f?.fusionStrategy === "learned_lr";
  const isFallback = f?.fusionStrategy === "average_fallback";
  const isImageOnly = f?.fusionStrategy === "image_only" || f?.fusionStrategy === "experimental_gmu_image_only";
  const isCbcOnly = f?.fusionStrategy === "cbc_only" || f?.fusionStrategy === "experimental_gmu_cbc_only";

  const strategyLabel = isGmu
    ? "Experimental GMU Fusion"
    : isStacking
    ? "Multimodal Stacking Ensemble"
    : isLearned
    ? "Learned Multimodal Fusion"
    : isFallback
    ? "Fallback Fusion"
    : isImageOnly
    ? "Image-Only Mode"
    : isCbcOnly
    ? "CBC-Only Mode"
    : f?.fusionStrategy ?? "Fusion";

  return (
    <section
      className="rounded-2xl overflow-hidden shadow-2xl"
      style={{ background: "linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%)" }}
      aria-labelledby="fusion-heading"
    >
      {/* Header bar */}
      <div className="flex items-center justify-between gap-4 px-6 pt-6 pb-4 border-b border-white/10">
        <div className="flex items-center gap-3">
          <span className="grid size-10 place-items-center rounded-xl bg-violet-500/30">
            <GitMerge className="size-5 text-violet-300" />
          </span>
          <div>
            <h2 id="fusion-heading" className="text-lg font-bold text-white">
              Final Multimodal Anemia Assessment
            </h2>
            <p className="text-sm text-slate-400 mt-0.5">
              Combined assessment from blood-smear image and CBC parameters
            </p>
          </div>
        </div>

        {/* Strategy badge */}
        {hasResult && (
          <div className="shrink-0">
            {isGmu ? (
              <Badge className="bg-violet-600/90 text-white border-0 text-xs px-3 py-1 font-semibold flex items-center gap-1.5">
                ⚡ PyTorch GMU (Experimental)
              </Badge>
            ) : isStacking ? (
              <Badge className="bg-emerald-600/90 text-white border-0 text-xs px-3 py-1 font-semibold">
                ✓ Stacking Ensemble v2.0
              </Badge>
            ) : isLearned ? (
              <Badge className="bg-emerald-600/90 text-white border-0 text-xs px-3 py-1 font-semibold">
                ✓ Learned Multimodal Fusion
              </Badge>
            ) : isFallback ? (
              <Badge className="bg-amber-600/90 text-white border-0 text-xs px-3 py-1 font-semibold">
                ⚠ Fallback Mode
              </Badge>
            ) : (
              <Badge className="bg-blue-600/90 text-white border-0 text-xs px-3 py-1 font-semibold">
                {strategyLabel}
              </Badge>
            )}
          </div>
        )}
      </div>

      <div className="px-6 py-6 space-y-6">
        {!hasResult ? (
          <div className="rounded-xl border border-white/10 bg-white/5 px-5 py-8 text-center space-y-2">
            <p className="text-base font-semibold text-slate-300">Fusion assessment unavailable</p>
            <p className="text-sm text-slate-500">{f?.message ?? "Insufficient inputs — upload an image and/or provide CBC values."}</p>
          </div>
        ) : (
          <>
            {/* Giant prediction + probability */}
            <div className="grid gap-4 sm:grid-cols-2">
              {/* Prediction verdict */}
              <div
                className={`rounded-xl p-6 flex flex-col justify-between gap-4 ${
                  f.isAnemic
                    ? "bg-red-950/60 border border-red-500/40"
                    : "bg-emerald-950/60 border border-emerald-500/40"
                }`}
              >
                <p className="text-xs font-semibold uppercase tracking-widest text-slate-400">
                  Final Verdict
                </p>
                <p
                  className={`text-5xl font-black tracking-tight leading-none ${
                    f.isAnemic ? "text-red-400" : "text-emerald-400"
                  }`}
                >
                  {f.fusionPrediction?.toUpperCase()}
                </p>
                <div
                  className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-semibold w-fit ${
                    f.isAnemic
                      ? "bg-red-500/20 text-red-300"
                      : "bg-emerald-500/20 text-emerald-300"
                  }`}
                >
                  <span className={`size-2 rounded-full ${f.isAnemic ? "bg-red-400" : "bg-emerald-400"}`} />
                  {f.isAnemic ? "Anemia indicators present" : "No anemia indicators detected"}
                </div>
              </div>

              {/* Fused probability */}
              <div className="rounded-xl p-6 flex flex-col justify-between gap-4 bg-violet-950/60 border border-violet-500/30">
                <p className="text-xs font-semibold uppercase tracking-widest text-slate-400">
                  Fused Probability
                </p>
                <p className="text-5xl font-black tabular-nums text-violet-200 leading-none">
                  {(f.fusionProbability * 100).toFixed(1)}%
                </p>
                <ProbBar value={f.fusionProbability} color={f.isAnemic ? "#f87171" : "#34d399"} />
              </div>
            </div>

            {/* Contributing modalities */}
            <div className="rounded-xl border border-white/10 bg-white/5 p-5 space-y-4">
              <p className="text-xs font-semibold uppercase tracking-widest text-slate-400">
                Contributing Modalities
              </p>
              <div className="grid gap-3 sm:grid-cols-2">
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <span className="grid size-6 place-items-center rounded bg-sky-500/20">
                      <Activity className="size-3.5 text-sky-400" />
                    </span>
                    <span className="text-sm font-medium text-slate-300">Image Probability</span>
                  </div>
                  <div className="flex items-baseline gap-2">
                    <span className="text-2xl font-bold tabular-nums text-sky-300">
                      {isCbcOnly ? "N/A" : `${(f.imageProbability * 100).toFixed(1)}%`}
                    </span>
                    <span className="text-xs text-slate-500">MobileNetV3-Small</span>
                  </div>
                  <div className="h-2 w-full rounded-full bg-white/10 overflow-hidden">
                    <div
                      className="h-full rounded-full bg-sky-400"
                      style={{ width: `${isCbcOnly ? 0 : Math.min(100, f.imageProbability * 100)}%` }}
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <span className="grid size-6 place-items-center rounded bg-purple-500/20">
                      <FlaskConical className="size-3.5 text-purple-400" />
                    </span>
                    <span className="text-sm font-medium text-slate-300">CBC Probability</span>
                  </div>
                  <div className="flex items-baseline gap-2">
                    <span className="text-2xl font-bold tabular-nums text-purple-300">
                      {isImageOnly ? "N/A" : `${(f.cbcProbability * 100).toFixed(1)}%`}
                    </span>
                    <span className="text-xs text-slate-500">Random Forest</span>
                  </div>
                  <div className="h-2 w-full rounded-full bg-white/10 overflow-hidden">
                    <div
                      className="h-full rounded-full bg-purple-400"
                      style={{ width: `${isImageOnly ? 0 : Math.min(100, f.cbcProbability * 100)}%` }}
                    />
                  </div>
                </div>
              </div>
              {isGmu && f?.latentGateMean !== undefined && (
                <div className="mt-4 rounded-xl border border-violet-500/30 bg-violet-950/40 p-4 space-y-2">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-semibold text-violet-200">GMU Dynamic Modality Gate (z)</span>
                    <span className="font-mono text-violet-300">
                      Smear: {(f.latentGateMean * 100).toFixed(1)}% · CBC: {((1 - f.latentGateMean) * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-2 w-full rounded-full bg-white/10 overflow-hidden flex">
                    <div
                      className="h-full bg-sky-400 transition-all"
                      style={{ width: `${Math.round(f.latentGateMean * 100)}%` }}
                      title="Image modality contribution"
                    />
                    <div
                      className="h-full bg-purple-400 transition-all"
                      style={{ width: `${100 - Math.round(f.latentGateMean * 100)}%` }}
                      title="CBC modality contribution"
                    />
                  </div>
                  <p className="text-[11px] text-violet-300/80">
                    The GMU unit dynamically weighs latent features from each stream based on modality presence and confidence.
                  </p>
                </div>
              )}
            </div>

            {/* Fusion strategy explanation */}
            <div
              className={`rounded-xl px-5 py-3.5 text-sm flex items-start gap-3 ${
                isGmu
                  ? "bg-violet-950/50 border border-violet-500/30"
                  : isLearned
                  ? "bg-emerald-950/50 border border-emerald-500/30"
                  : isFallback
                  ? "bg-amber-950/50 border border-amber-500/30"
                  : "bg-blue-950/50 border border-blue-500/30"
              }`}
            >
              <Info
                className={`size-4 shrink-0 mt-0.5 ${
                  isGmu ? "text-violet-400" : isLearned ? "text-emerald-400" : isFallback ? "text-amber-400" : "text-blue-400"
                }`}
              />
              <div className="space-y-1">
                <p
                  className={`font-semibold ${
                    isGmu ? "text-violet-300" : isLearned ? "text-emerald-300" : isFallback ? "text-amber-300" : "text-blue-300"
                  }`}
                >
                  {strategyLabel}
                </p>
                {isGmu && (
                  <div className="space-y-1.5 text-xs text-violet-300/80">
                    <p>
                      Inference evaluated using the PyTorch Gated Multimodal Unit (GMU) with adaptive gate routing and modality masking.
                    </p>
                    <div className="rounded-md border border-amber-500/30 bg-amber-500/10 p-2.5 text-[11px] text-amber-200">
                      <span className="font-semibold text-amber-300">Academic Prototype Notice: </span>
                      {f.limitationNotice || "Trained on synthetic image-CBC pairings. Evaluated for dynamic gating; not clinically validated on genuine patient cohorts."}
                    </div>
                  </div>
                )}
                {!isGmu && isStacking && (
                  <p className="text-emerald-400/80 text-xs mt-0.5">
                    Multimodal Stacking Ensemble (HistGB + ExtraTrees + Logistic Regression) applied across 12 clinical & morphological features.
                  </p>
                )}
                {!isGmu && !isStacking && isLearned && (
                  <p className="text-emerald-400/80 text-xs mt-0.5">
                    A trained Logistic Regression model has been applied to combine image and CBC probabilities.
                  </p>
                )}
                {isFallback && (
                  <p className="text-amber-400/80 text-xs mt-0.5">
                    Trained fusion model unavailable — result is the simple average of Image probability and CBC probability.
                  </p>
                )}
                {isImageOnly && !isGmu && (
                  <p className="text-blue-400/80 text-xs mt-0.5">
                    No CBC values were provided — result is based on the blood-smear image alone.
                  </p>
                )}
                {isCbcOnly && !isGmu && (
                  <p className="text-blue-400/80 text-xs mt-0.5">
                    No blood-smear image was available — result is based on CBC parameters alone.
                  </p>
                )}
              </div>
            </div>
          </>
        )}
      </div>
    </section>
  );
}

// ──────────────────────────────────────────────────────────────────────────────
// Section B — Blood Cell Classification
// ──────────────────────────────────────────────────────────────────────────────

const CELL_DISPLAY = [
  { key: "rbc",      label: "RBC",      fullLabel: "Red Blood Cells",  color: "#ef4444", icon: Droplets },
  { key: "wbc",      label: "WBC",      fullLabel: "White Blood Cells", color: "#3b82f6", icon: Activity },
  { key: "platelet", label: "Platelet", fullLabel: "Platelets",         color: "#a855f7", icon: Microscope },
] as const;

function DetectedCellsPanel({ data }: { data: any }) {
  const raw  = data?.rawCounts ?? {};
  const conf = data?.confidence ?? 0;

  return (
    <section className="panel p-6 space-y-5" aria-labelledby="cells-heading">
      <div className="flex items-center gap-3">
        <span className="grid size-10 place-items-center rounded-xl bg-blue-500/15 text-blue-400">
          <Microscope className="size-5" />
        </span>
        <div>
          <h3 id="cells-heading" className="text-base font-bold">Blood Cell Classification</h3>
          <p className="text-xs text-muted-foreground mt-0.5">
            Object counts detected by YOLOv8 in the submitted blood smear image
          </p>
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-3">
        {CELL_DISPLAY.map(({ key, label, fullLabel, color, icon: Icon }) => {
          const count = Number(raw[key] ?? 0);
          return (
            <article
              key={key}
              className="rounded-xl border border-border bg-muted/20 p-5 space-y-3"
            >
              <div className="flex items-center gap-2">
                <span
                  className="grid size-8 place-items-center rounded-lg"
                  style={{ background: `${color}22` }}
                >
                  <Icon className="size-4" style={{ color }} />
                </span>
                <span className="text-sm font-semibold text-foreground">{fullLabel}</span>
              </div>
              <p className="text-4xl font-black tabular-nums" style={{ color }}>{count}</p>
              <p className="text-xs font-semibold tracking-wide text-muted-foreground">
                {label} detected in frame
              </p>
            </article>
          );
        })}
      </div>

      <div className="flex items-center justify-between rounded-lg border border-border bg-muted/10 px-4 py-3">
        <p className="text-sm text-muted-foreground">Average detection confidence</p>
        <p className="text-sm font-bold tabular-nums">{formatPercent(conf, 1)}</p>
      </div>

      <div className="flex items-start gap-3 rounded-lg border border-amber-500/25 bg-amber-500/8 px-4 py-3.5">
        <Info className="size-4 shrink-0 text-amber-500 mt-0.5" />
        <p className="text-xs text-amber-600 dark:text-amber-300 leading-relaxed">
          <strong>Important:</strong> These counts represent objects detected in the submitted
          blood-smear image and are <em>not</em> laboratory CBC measurements (cells/µL).
          A single microscopy field of view covers only a small fraction of total blood volume.
        </p>
      </div>
    </section>
  );
}

// ──────────────────────────────────────────────────────────────────────────────
// WBC Differential
// ──────────────────────────────────────────────────────────────────────────────

const WBC_DIFF_REF: Record<string, { min: number; max: number; color: string }> = {
  Neutrophil: { min: 50, max: 70, color: "#f59e0b" },
  Lymphocyte: { min: 20, max: 40, color: "#06b6d4" },
  Monocyte:   { min: 2,  max: 8,  color: "#10b981" },
  Eosinophil: { min: 1,  max: 4,  color: "#f97316" },
  Basophil:   { min: 0,  max: 1,  color: "#8b5cf6" },
};

function WBCDifferentialPanel({ data }: { data: any }) {
  const diff        = data?.wbcDifferential ?? {};
  const isEstimated = data?.isEstimatedDiff ?? true;
  const wbcCount    = data?.rawCounts?.wbc ?? 0;

  return (
    <section className="panel p-6 space-y-4" aria-labelledby="diff-heading">
      <div className="flex items-start justify-between gap-2">
        <div>
          <h3 id="diff-heading" className="text-base font-bold">WBC Differential Count</h3>
          <p className="text-xs text-muted-foreground mt-0.5">
            {isEstimated
              ? "Estimated from WHO reference ratios — ResNet50 classifier not available"
              : "Classified by ResNet50 sub-type model"}
          </p>
        </div>
        {isEstimated && (
          <Badge variant="outline" className="text-amber-500 border-amber-500/40 shrink-0 text-xs">
            Estimated
          </Badge>
        )}
      </div>

      {wbcCount === 0 ? (
        <p className="text-sm text-muted-foreground italic py-2">
          No WBCs detected in this image — differential unavailable.
        </p>
      ) : (
        <div className="space-y-3.5">
          {Object.entries(WBC_DIFF_REF).map(([subtype, ref]) => {
            const pct    = Number(diff[subtype] ?? 0);
            const status = pct < ref.min ? "low" : pct > ref.max ? "high" : "normal";
            const statusColor =
              status === "normal" ? "text-emerald-500" :
              status === "low"    ? "text-blue-500"    : "text-red-500";
            return (
              <div key={subtype} className="space-y-1.5">
                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-2">
                    <span className="inline-block size-2.5 rounded-full" style={{ background: ref.color }} />
                    <span className="font-medium">{subtype}</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className={`text-[10px] font-bold uppercase ${statusColor}`}>{status}</span>
                    <span className="tabular-nums text-muted-foreground w-10 text-right">{pct.toFixed(1)}%</span>
                  </div>
                </div>
                <div className="relative h-2 rounded-full bg-muted overflow-hidden">
                  <div
                    className="absolute top-0 h-full rounded-full opacity-20"
                    style={{ left: `${ref.min}%`, width: `${ref.max - ref.min}%`, background: ref.color }}
                  />
                  <div
                    className="absolute top-0 h-full rounded-full transition-all"
                    style={{ width: `${Math.min(100, pct)}%`, background: ref.color }}
                  />
                </div>
                <p className="text-[10px] text-muted-foreground">Ref: {ref.min}–{ref.max}%</p>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}

// ──────────────────────────────────────────────────────────────────────────────
// Section C — Image-Based Anemia Screening
// ──────────────────────────────────────────────────────────────────────────────

function ImageAnemiaPanel({ data }: { data: any }) {
  const img       = data?.anemiaImage;
  const hasResult = img && img.status === "success";

  return (
    <section className="panel p-6 space-y-5" aria-labelledby="img-anemia-heading">
      <div className="flex items-center gap-3">
        <span className="grid size-10 place-items-center rounded-xl bg-sky-500/15 text-sky-400">
          <Activity className="size-5" />
        </span>
        <div>
          <h3 id="img-anemia-heading" className="text-base font-bold">Image-Based Anemia Screening</h3>
          <p className="text-xs text-muted-foreground mt-0.5">
            MobileNetV3-Small deep learning model · trained on AneRBC photomicrograph dataset
          </p>
        </div>
      </div>

      {!hasResult ? (
        <div className="rounded-xl border border-dashed border-border bg-muted/5 px-5 py-8 text-center space-y-2">
          <p className="text-sm font-semibold text-muted-foreground">Image anemia model unavailable</p>
          <p className="text-xs text-muted-foreground">{img?.message ?? "Model not loaded or no image provided."}</p>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="grid gap-4 sm:grid-cols-2">
            {/* Prediction */}
            <div
              className={`rounded-xl border p-5 space-y-2 ${
                img.isAnemic
                  ? "border-red-500/40 bg-red-500/8"
                  : "border-emerald-500/40 bg-emerald-500/8"
              }`}
            >
              <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Image Model Prediction
              </p>
              <p className={`text-2xl font-black ${img.isAnemic ? "text-red-500" : "text-emerald-500"}`}>
                {img.prediction}
              </p>
              <p className="text-xs text-muted-foreground">
                Confidence: <span className="font-bold">{formatPercent(img.confidence, 1)}</span>
              </p>
            </div>

            {/* Probability */}
            <div className="rounded-xl border border-sky-500/30 bg-sky-500/8 p-5 space-y-3">
              <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Anemia Probability
              </p>
              <p className="text-3xl font-black tabular-nums text-sky-400">
                {(img.probability * 100).toFixed(1)}%
              </p>
              <div className="h-2.5 w-full rounded-full bg-white/10 overflow-hidden">
                <div
                  className="h-full rounded-full bg-sky-400 transition-all"
                  style={{ width: `${Math.min(100, img.probability * 100)}%` }}
                />
              </div>
            </div>
          </div>

          <p className="text-xs text-muted-foreground">
            <span className="font-semibold text-foreground">Model:</span>{" "}
            {img.modelName} · v{img.modelVersion}
          </p>
        </div>
      )}
    </section>
  );
}

// ──────────────────────────────────────────────────────────────────────────────
// Section D — CBC-Based ML Anemia Screening
// ──────────────────────────────────────────────────────────────────────────────

function AnemiaMLPanel({ data }: { data: any }) {
  const ml        = data?.anemiaMl;
  const hasResult = ml && ml.status === "success";

  return (
    <section className="panel p-6 space-y-5" aria-labelledby="anemia-heading">
      <div className="flex items-center gap-3">
        <span className="grid size-10 place-items-center rounded-xl bg-purple-500/15 text-purple-400">
          <FlaskConical className="size-5" />
        </span>
        <div>
          <h3 id="anemia-heading" className="text-base font-bold">CBC-Based Anemia Screening</h3>
          <p className="text-xs text-muted-foreground mt-0.5">
            Random Forest model · uses Gender, Hemoglobin, MCV, MCH, and MCHC
          </p>
        </div>
      </div>

      {/* ── Laboratory CBC Inputs: always shown when provided, regardless of ML status ── */}
      {data?.cbcInputs ? (
        <div className="rounded-xl border border-teal-500/25 bg-teal-500/5 px-5 py-4 space-y-3">
          <p className="text-xs font-semibold uppercase tracking-wide text-teal-400">
            Laboratory CBC Inputs
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 text-sm">
            <div className="space-y-0.5">
              <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Gender</p>
              <p className="font-bold text-foreground">{data.cbcInputs.Gender ?? "—"}</p>
            </div>
            <div className="space-y-0.5">
              <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Hemoglobin</p>
              <p className="font-bold text-foreground">
                {data.cbcInputs.Hemoglobin != null ? `${data.cbcInputs.Hemoglobin} g/dL` : "—"}
              </p>
            </div>
            <div className="space-y-0.5">
              <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">MCV</p>
              <p className="font-bold text-foreground">
                {data.cbcInputs.MCV != null ? `${data.cbcInputs.MCV} fL` : "—"}
              </p>
            </div>
            <div className="space-y-0.5">
              <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">MCH</p>
              <p className="font-bold text-foreground">
                {data.cbcInputs.MCH != null ? `${data.cbcInputs.MCH} pg` : "—"}
              </p>
            </div>
            <div className="space-y-0.5">
              <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">MCHC</p>
              <p className="font-bold text-foreground">
                {data.cbcInputs.MCHC != null ? `${data.cbcInputs.MCHC} g/dL` : "—"}
              </p>
            </div>
          </div>
        </div>
      ) : null}

      {/* ── ML Prediction result (only when available) ── */}
      {!hasResult ? (
        <div className="rounded-xl border border-dashed border-border bg-muted/5 px-5 py-8 text-center space-y-2">
          {data?.cbcInputs ? (
            <>
              <p className="text-sm font-semibold text-muted-foreground">
                {ml?.status === "invalid_input" ? "Incomplete CBC inputs" : "ML prediction unavailable"}
              </p>
              <p className="text-xs text-muted-foreground">
                {ml?.message || "The Random Forest model could not run — check that all five CBC fields were provided."}
              </p>
            </>
          ) : (
            <>
              <p className="text-sm font-semibold text-muted-foreground">No CBC inputs provided</p>
              <p className="text-xs text-muted-foreground">
                To run CBC-based anemia screening, enter Gender, Hemoglobin, MCV, MCH, and MCHC
                in the Analyze page before uploading the image.
              </p>
            </>
          )}
        </div>
      ) : (
        <div className="space-y-4">
          <div className="grid gap-4 sm:grid-cols-2">
            {/* Prediction */}
            <div
              className={`rounded-xl border p-5 space-y-2 ${
                ml.isAnemic
                  ? "border-red-500/40 bg-red-500/8"
                  : "border-emerald-500/40 bg-emerald-500/8"
              }`}
            >
              <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                CBC Model Prediction
              </p>
              <p className={`text-2xl font-black ${ml.isAnemic ? "text-red-500" : "text-emerald-500"}`}>
                {ml.prediction}
              </p>
              <p className="text-xs text-muted-foreground">
                Confidence: <span className="font-bold">{formatPercent(ml.confidence, 1)}</span>
              </p>
            </div>

            {/* Probability */}
            <div className="rounded-xl border border-purple-500/30 bg-purple-500/8 p-5 space-y-3">
              <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Anemia Probability
              </p>
              <p className="text-3xl font-black tabular-nums text-purple-400">
                {(ml.probability * 100).toFixed(1)}%
              </p>
              <div className="h-2.5 w-full rounded-full bg-white/10 overflow-hidden">
                <div
                  className="h-full rounded-full bg-purple-400 transition-all"
                  style={{ width: `${Math.min(100, ml.probability * 100)}%` }}
                />
              </div>
            </div>
          </div>

          <p className="text-xs text-muted-foreground">
            <span className="font-semibold text-foreground">Model:</span>{" "}
            {ml.modelName} · v{ml.modelVersion}
          </p>

          {/* Dataset limitation disclosure */}
          <div className="flex items-start gap-3 rounded-lg border border-amber-500/25 bg-amber-500/8 px-4 py-3.5">
            <AlertTriangle className="size-4 shrink-0 text-amber-500 mt-0.5" />
            <p className="text-xs text-amber-600 dark:text-amber-300 leading-relaxed">
              <strong>Dataset limitation:</strong> The CBC model's training dataset derives anemia
              labels from Hemoglobin thresholds (Kaggle, 1,421 records). Since Hemoglobin is both
              a training feature and the label source, reported accuracy is inflated. Treat results
              as a <em>Hemoglobin-threshold-based indicator</em>, not an independent clinical predictor.
            </p>
          </div>
        </div>
      )}
    </section>
  );
}

// ──────────────────────────────────────────────────────────────────────────────
// Section E — Advanced / Optional Analysis (collapsible)
// ──────────────────────────────────────────────────────────────────────────────

function AdvancedSection({ data }: { data: any }) {
  const [open, setOpen] = useState(false);

  const MorphologySection = open
    ? require("@/components/analysis/MorphologySection").MorphologySection
    : null;
  const CellStatisticsSection = open
    ? require("@/components/analysis/CellStatisticsSection").CellStatisticsSection
    : null;
  const ExplainabilitySection = open
    ? require("@/components/analysis/ExplainabilitySection").ExplainabilitySection
    : null;

  return (
    <div className="panel overflow-hidden">
      <button
        className="flex w-full items-center justify-between px-5 py-4 text-sm font-semibold hover:bg-muted/30 transition-colors"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        <div className="flex items-center gap-2">
          <Layers className="size-4 text-muted-foreground" />
          <span>Advanced Analysis</span>
          <span className="text-xs font-normal text-muted-foreground">(Morphology, Cell Statistics, Explainability)</span>
        </div>
        {open ? <ChevronUp className="size-4" /> : <ChevronDown className="size-4" />}
      </button>

      {open && (
        <div className="border-t border-border p-5 space-y-6">
          <p className="text-xs text-muted-foreground">
            The following sections are supplementary. Morphology metrics are derived
            from bounding box geometry and are approximate. Explainability data reflects
            model-level confidence — not per-pixel attention maps.
          </p>
          {CellStatisticsSection && <CellStatisticsSection stats={data.cellStatistics} />}
          {MorphologySection && <MorphologySection morphology={data.morphology} />}
          {ExplainabilitySection && <ExplainabilitySection analysis={data} />}
        </div>
      )}
    </div>
  );
}

// ──────────────────────────────────────────────────────────────────────────────
// Section F — Medical Disclaimer
// ──────────────────────────────────────────────────────────────────────────────

function Disclaimer() {
  return (
    <div className="rounded-xl border border-red-500/25 bg-red-500/5 px-5 py-4 space-y-1.5">
      <p className="text-xs font-bold text-red-500 uppercase tracking-wide">
        ⚕ Research Prototype — Medical Disclaimer
      </p>
      <p className="text-xs text-muted-foreground leading-relaxed">
        This system is an <strong>AI-based research prototype</strong> for educational and
        experimental purposes only. Image-based cell detection and ML-based anemia screening do
        not replace laboratory CBC testing, microscopic examination by a trained haematologist,
        or professional medical diagnosis. Do not use these results to make clinical decisions.
        All outputs must be reviewed and validated by a qualified medical professional.
      </p>
    </div>
  );
}

// ──────────────────────────────────────────────────────────────────────────────
// Main Results Page
// ──────────────────────────────────────────────────────────────────────────────

function ResultsPage() {
  const { analysisId } = Route.useParams();
  const [reportOpen, setReportOpen] = useState(false);

  const { data, isLoading, isError, refetch } = useQuery({
    queryKey: queryKeys.analysis(analysisId),
    queryFn: () => api.getAnalysis(analysisId),
  });

  if (isError) return <ErrorState onRetry={() => refetch()} />;

  if (isLoading || !data) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-10 w-72" />
        <div className="grid gap-6 lg:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
          <Skeleton className="h-[460px] rounded-xl" />
          <div className="space-y-4">
            {Array.from({ length: 4 }).map((_, i) => (
              <Skeleton key={i} className="h-24 rounded-xl" />
            ))}
          </div>
        </div>
        <Skeleton className="h-72 rounded-xl" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="Blood Smear Analysis"
        subtitle="AI-based cell detection and CBC-based anemia screening."
        actions={
          <Button onClick={() => setReportOpen(true)}>
            <Download className="size-4" />
            Download Report
          </Button>
        }
      />

      {/* Meta row */}
      <div className="panel flex flex-wrap items-center gap-x-8 gap-y-3 px-5 py-4">
        <span className="flex items-center gap-2 text-sm font-medium text-success">
          <span className="size-2 rounded-full bg-success" />
          Analysis Complete
        </span>
        <Meta icon={Hash}         label="Analysis ID"         value={data.id} />
        <Meta icon={CalendarDays} label="Date / time"         value={formatDateTime(data.createdAt)} />
        <Meta icon={Clock}        label="Overall confidence"  value={formatPercent(data.confidence, 1)} />
      </div>

      {/* Image viewer + quick sidebar counts */}
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)] lg:items-start">
        <ImageViewer analysis={data} />

        {/* Quick cell count summary cards */}
        <div className="space-y-3">
          {CELL_DISPLAY.map(({ key, label, color, icon: Icon }) => {
            const count = Number((data.rawCounts as any)?.[key] ?? 0);
            return (
              <article key={key} className="panel p-4 flex items-center gap-4">
                <span
                  className="grid size-10 shrink-0 place-items-center rounded-xl"
                  style={{ background: `${color}20` }}
                >
                  <Icon className="size-5" style={{ color }} />
                </span>
                <div className="min-w-0 flex-1">
                  <p className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">
                    {label} detected
                  </p>
                  <p className="mt-0.5 text-3xl font-black tabular-nums" style={{ color }}>{count}</p>
                </div>
              </article>
            );
          })}
          <article className="panel p-4">
            <p className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">
              Total objects in frame
            </p>
            <p className="mt-0.5 text-3xl font-black tabular-nums">{data.detections.length}</p>
            <p className="mt-1 text-xs text-muted-foreground">
              Avg confidence {formatPercent(data.confidence, 0)}
            </p>
          </article>
        </div>
      </div>

      {/* A — Final Multimodal Anemia Assessment (hero, most prominent) */}
      {data.anemiaFusion && <FusionAnemiaPanel data={data} />}

      {/* B — Blood Cell Classification detail */}
      <DetectedCellsPanel data={data} />

      {/* WBC Differential */}
      <WBCDifferentialPanel data={data} />

      {/* C — Image-Based Anemia Analysis */}
      <ImageAnemiaPanel data={data} />

      {/* D — CBC-Based Anemia Screening */}
      <AnemiaMLPanel data={data} />

      {/* E — Advanced (collapsible) */}
      <AdvancedSection data={data} />

      {/* F — Disclaimer */}
      <Disclaimer />

      <ReportPreviewDialog
        open={reportOpen}
        onOpenChange={setReportOpen}
        analysis={data}
        report={{
          id: `RPT-${data.id.replace(/\D/g, "")}`,
          analysisId: data.id,
          createdAt: data.createdAt,
          result: "AI analysis completed",
          riskLevel: "low",
          status: "ready",
        }}
      />
    </div>
  );
}
