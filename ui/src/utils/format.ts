import type { AnalysisStatus, RiskLevel } from "@/types";

export function formatDateTime(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
  });
}

export function formatPercent(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export const statusLabels: Record<AnalysisStatus, string> = {
  completed: "Completed",
  processing: "Processing",
  review_required: "Review Required",
  failed: "Failed",
};

export const riskLabels: Record<RiskLevel, string> = {
  low: "Low Indication",
  moderate: "Moderate Indication",
  high: "High Indication",
};
