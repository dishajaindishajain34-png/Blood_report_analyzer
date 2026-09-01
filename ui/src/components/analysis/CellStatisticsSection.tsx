import { Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { CellStatistics } from "@/types";
import { formatPercent } from "@/utils/format";

const colors = ["var(--color-chart-1)", "var(--color-chart-2)", "var(--color-chart-3)"];

export function CellStatisticsSection({ stats }: { stats: CellStatistics }) {
  return (
    <section className="panel overflow-hidden" aria-labelledby="cell-stats-heading">
      <div className="border-b border-border px-5 py-4">
        <h3 id="cell-stats-heading" className="text-sm font-semibold">
          Cell Statistics
        </h3>
        <p className="text-xs text-muted-foreground">
          Counts and classification confidence per detected cell type.
        </p>
      </div>
      <div className="grid gap-0 lg:grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)]">
        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Cell Type</TableHead>
                <TableHead className="text-right">Detected Count</TableHead>
                <TableHead className="text-right">Avg. Confidence</TableHead>
                <TableHead className="text-right">Percentage</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {stats.breakdown.map((row, i) => (
                <TableRow key={row.cellType}>
                  <TableCell className="font-medium">
                    <span className="flex items-center gap-2">
                      <span
                        className="size-2.5 rounded-[2px]"
                        style={{ backgroundColor: colors[i % colors.length] }}
                        aria-hidden="true"
                      />
                      {row.cellType}
                    </span>
                  </TableCell>
                  <TableCell className="text-right tabular-nums">{row.count}</TableCell>
                  <TableCell className="text-right tabular-nums">
                    {formatPercent(row.averageConfidence, 1)}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">{row.percentage}%</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
        <div className="h-64 border-t border-border p-3 lg:border-t-0 lg:border-l">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={stats.breakdown}
                dataKey="count"
                nameKey="cellType"
                innerRadius={52}
                outerRadius={80}
                paddingAngle={3}
                stroke="var(--color-background)"
              >
                {stats.breakdown.map((entry, i) => (
                  <Cell key={entry.cellType} fill={colors[i % colors.length]} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  borderRadius: 10,
                  border: "1px solid var(--color-border)",
                  fontSize: 12,
                }}
              />
              <Legend iconType="circle" wrapperStyle={{ fontSize: 12 }} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>
    </section>
  );
}
