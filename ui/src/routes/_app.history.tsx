import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { createFileRoute, Link } from "@tanstack/react-router";
import { ArrowUpDown, Eye, Search } from "lucide-react";

import { RiskBadge, StatusBadge } from "@/components/common/Badges";
import { EmptyState, ErrorState, PageHeader } from "@/components/common/States";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { api, queryKeys } from "@/services/api";
import { formatDateTime, formatPercent } from "@/utils/format";

export const Route = createFileRoute("/_app/history")({
  head: () => ({
    meta: [
      { title: "Analysis History — HemaVision AI" },
      { name: "description", content: "Search, filter and revisit previous blood smear analyses." },
      { property: "og:title", content: "Analysis History — HemaVision AI" },
      {
        property: "og:description",
        content: "Search, filter and revisit previous blood smear analyses.",
      },
    ],
  }),
  component: HistoryPage,
});

const PAGE_SIZE = 8;

function HistoryPage() {
  const { data, isLoading, isError, refetch } = useQuery({
    queryKey: queryKeys.analyses,
    queryFn: api.listAnalyses,
  });

  const [query, setQuery] = useState("");
  const [status, setStatus] = useState("all");
  const [risk, setRisk] = useState("all");
  const [sortAsc, setSortAsc] = useState(false);
  const [page, setPage] = useState(1);

  const rows = useMemo(() => {
    const filtered = (data ?? []).filter((row) => {
      const matchesQuery =
        !query ||
        row.id.toLowerCase().includes(query.toLowerCase()) ||
        row.imageName.toLowerCase().includes(query.toLowerCase());
      const matchesStatus = status === "all" || row.status === status;
      const matchesRisk = risk === "all" || row.riskLevel === risk;
      return matchesQuery && matchesStatus && matchesRisk;
    });
    return filtered.sort((a, b) =>
      sortAsc
        ? a.createdAt.localeCompare(b.createdAt)
        : b.createdAt.localeCompare(a.createdAt),
    );
  }, [data, query, status, risk, sortAsc]);

  const pageCount = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
  const current = rows.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  if (isError) return <ErrorState onRetry={() => refetch()} />;

  return (
    <div className="space-y-6">
      <PageHeader title="Analysis History" subtitle="All previous blood smear submissions." />

      <section className="panel overflow-hidden">
        <div className="flex flex-wrap items-center gap-3 border-b border-border px-4 py-3">
          <div className="relative min-w-[200px] flex-1">
            <Search
              className="pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground"
              aria-hidden="true"
            />
            <Input
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                setPage(1);
              }}
              placeholder="Search by analysis ID or image"
              aria-label="Search analyses"
              className="pl-8"
            />
          </div>
          <Select
            value={status}
            onValueChange={(v) => {
              setStatus(v);
              setPage(1);
            }}
          >
            <SelectTrigger className="w-[170px]" aria-label="Filter by status">
              <SelectValue placeholder="Status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All statuses</SelectItem>
              <SelectItem value="completed">Completed</SelectItem>
              <SelectItem value="processing">Processing</SelectItem>
              <SelectItem value="review_required">Review Required</SelectItem>
              <SelectItem value="failed">Failed</SelectItem>
            </SelectContent>
          </Select>
          <Select
            value={risk}
            onValueChange={(v) => {
              setRisk(v);
              setPage(1);
            }}
          >
            <SelectTrigger className="w-[180px]" aria-label="Filter by risk indication">
              <SelectValue placeholder="Risk indication" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All indications</SelectItem>
              <SelectItem value="low">Low Indication</SelectItem>
              <SelectItem value="moderate">Moderate Indication</SelectItem>
              <SelectItem value="high">High Indication</SelectItem>
            </SelectContent>
          </Select>
          <Button variant="outline" size="sm" onClick={() => setSortAsc((s) => !s)}>
            <ArrowUpDown className="size-4" aria-hidden="true" />
            Date {sortAsc ? "ascending" : "descending"}
          </Button>
        </div>

        {isLoading ? (
          <div className="space-y-3 p-4">
            {Array.from({ length: 6 }).map((_, i) => (
              <Skeleton key={i} className="h-10 w-full" />
            ))}
          </div>
        ) : current.length === 0 ? (
          <EmptyState title="No analyses found." description="Try adjusting your search or filters." />
        ) : (
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Analysis ID</TableHead>
                  <TableHead>Date</TableHead>
                  <TableHead>Image</TableHead>
                  <TableHead className="text-right">Detected Cells</TableHead>
                  <TableHead>Risk</TableHead>
                  <TableHead className="text-right">Confidence</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead className="text-right">Action</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {current.map((row) => (
                  <TableRow key={row.id}>
                    <TableCell className="font-medium">{row.id}</TableCell>
                    <TableCell className="whitespace-nowrap text-muted-foreground">
                      {formatDateTime(row.createdAt)}
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <img
                          src={row.imageUrl}
                          alt=""
                          loading="lazy"
                          width={32}
                          height={32}
                          className="size-8 shrink-0 rounded-md object-cover"
                        />
                        <span className="max-w-[150px] truncate text-muted-foreground">
                          {row.imageName}
                        </span>
                      </div>
                    </TableCell>
                    <TableCell className="text-right tabular-nums">{row.cellsDetected}</TableCell>
                    <TableCell>
                      <RiskBadge level={row.riskLevel} />
                    </TableCell>
                    <TableCell className="text-right tabular-nums">
                      {formatPercent(row.confidence, 0)}
                    </TableCell>
                    <TableCell>
                      <StatusBadge status={row.status} />
                    </TableCell>
                    <TableCell className="text-right">
                      <Button asChild variant="ghost" size="sm">
                        <Link to="/results/$analysisId" params={{ analysisId: row.id }}>
                          <Eye className="size-4" aria-hidden="true" />
                          View Results
                        </Link>
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}

        {current.length > 0 && (
          <div className="flex items-center justify-between gap-3 border-t border-border px-4 py-3 text-xs text-muted-foreground">
            <span>
              Page {page} of {pageCount} · {rows.length} results
            </span>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                disabled={page === 1}
                onClick={() => setPage((p) => Math.max(1, p - 1))}
              >
                Previous
              </Button>
              <Button
                variant="outline"
                size="sm"
                disabled={page === pageCount}
                onClick={() => setPage((p) => Math.min(pageCount, p + 1))}
              >
                Next
              </Button>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
