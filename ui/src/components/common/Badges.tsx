import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import type { AnalysisStatus, RangeStatus, RiskLevel } from "@/types";
import { riskLabels, statusLabels } from "@/utils/format";

const statusStyles: Record<AnalysisStatus, string> = {
  completed: "border-success/30 bg-success/10 text-success",
  processing: "border-accent/40 bg-accent/10 text-secondary",
  review_required: "border-warning/40 bg-warning/15 text-warning",
  failed: "border-destructive/30 bg-destructive/10 text-destructive",
};

export function StatusBadge({ status }: { status: AnalysisStatus }) {
  return (
    <Badge variant="outline" className={cn("gap-1.5 font-medium", statusStyles[status])}>
      <span className="size-1.5 rounded-full bg-current" aria-hidden="true" />
      {statusLabels[status]}
    </Badge>
  );
}

const riskStyles: Record<RiskLevel, string> = {
  low: "border-success/30 bg-success/10 text-success",
  moderate: "border-warning/40 bg-warning/15 text-warning",
  high: "border-destructive/30 bg-destructive/10 text-destructive",
};

export function RiskBadge({ level }: { level: RiskLevel }) {
  return (
    <Badge variant="outline" className={cn("font-medium", riskStyles[level])}>
      {riskLabels[level]}
    </Badge>
  );
}

const rangeStyles: Record<RangeStatus, string> = {
  normal: "border-success/30 bg-success/10 text-success",
  low: "border-warning/40 bg-warning/15 text-warning",
  high: "border-warning/40 bg-warning/15 text-warning",
  unknown: "border-border bg-muted text-muted-foreground",
};

const rangeLabels: Record<RangeStatus, string> = {
  normal: "Within reference range",
  low: "Below reference range",
  high: "Above reference range",
  unknown: "Not determined",
};

export function RangeBadge({ status }: { status: RangeStatus }) {
  return (
    <Badge variant="outline" className={cn("font-medium", rangeStyles[status])}>
      {rangeLabels[status]}
    </Badge>
  );
}
