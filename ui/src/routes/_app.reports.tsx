import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { createFileRoute } from "@tanstack/react-router";
import { Download, Eye, FileText } from "lucide-react";

import { RiskBadge } from "@/components/common/Badges";
import { EmptyState, ErrorState, PageHeader } from "@/components/common/States";
import { ReportPreviewDialog } from "@/components/reports/ReportPreviewDialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { api, queryKeys } from "@/services/api";
import { mockAnalysis } from "@/data/mockData";
import { formatDate } from "@/utils/format";
import type { Report } from "@/types";

export const Route = createFileRoute("/_app/reports")({
  head: () => ({
    meta: [
      { title: "Analysis Reports — HemaVision AI" },
      { name: "description", content: "Preview and download structured blood smear analysis reports." },
      { property: "og:title", content: "Analysis Reports — HemaVision AI" },
      {
        property: "og:description",
        content: "Preview and download structured blood smear analysis reports.",
      },
    ],
  }),
  component: ReportsPage,
});

function ReportsPage() {
  const { data, isLoading, isError, refetch } = useQuery({
    queryKey: queryKeys.reports,
    queryFn: api.listReports,
  });
  const [active, setActive] = useState<Report | null>(null);

  if (isError) return <ErrorState title="Report could not be generated." onRetry={() => refetch()} />;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Analysis Reports"
        subtitle="Structured reports generated from model results by the analysis backend."
      />

      <section className="panel overflow-hidden">
        {isLoading ? (
          <div className="space-y-3 p-4">
            {Array.from({ length: 6 }).map((_, i) => (
              <Skeleton key={i} className="h-10 w-full" />
            ))}
          </div>
        ) : (data ?? []).length === 0 ? (
          <EmptyState icon={FileText} title="No reports yet." description="Reports appear here once an analysis completes." />
        ) : (
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Report ID</TableHead>
                  <TableHead>Analysis ID</TableHead>
                  <TableHead>Created</TableHead>
                  <TableHead>Result</TableHead>
                  <TableHead>Risk indication</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {(data ?? []).map((report) => (
                  <TableRow key={report.id}>
                    <TableCell className="font-medium">{report.id}</TableCell>
                    <TableCell className="text-muted-foreground">{report.analysisId}</TableCell>
                    <TableCell className="whitespace-nowrap text-muted-foreground">
                      {formatDate(report.createdAt)}
                    </TableCell>
                    <TableCell className="max-w-[220px] truncate">{report.result}</TableCell>
                    <TableCell>
                      <RiskBadge level={report.riskLevel} />
                    </TableCell>
                    <TableCell>
                      <Badge variant="outline" className="capitalize">
                        {report.status}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right">
                      <div className="flex justify-end gap-1">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => setActive(report)}
                          disabled={report.status !== "ready"}
                        >
                          <Eye className="size-4" aria-hidden="true" />
                          View
                        </Button>
                        <Button asChild variant="ghost" size="sm" disabled={report.status !== "ready"}>
                          <a href={api.downloadReportUrl(report.id)} download aria-label={`Download ${report.id} as PDF`}>
                            <Download className="size-4" aria-hidden="true" />
                            PDF
                          </a>
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </section>

      <ReportPreviewDialog
        report={active}
        analysis={mockAnalysis}
        open={active !== null}
        onOpenChange={(open) => !open && setActive(null)}
      />
    </div>
  );
}
