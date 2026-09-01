import {
  Bar,
  BarChart,
  CartesianGrid,
  PolarAngleAxis,
  PolarGrid,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { StatTile } from "@/components/common/MetricCard";
import type { MorphologyMetrics } from "@/types";

export function MorphologySection({ morphology }: { morphology: MorphologyMetrics }) {
  return (
    <section className="panel p-5" aria-labelledby="morphology-heading">
      <h3 id="morphology-heading" className="text-sm font-semibold">
        RBC Morphology
      </h3>
      <p className="text-xs text-muted-foreground">
        Image-derived shape and intensity metrics computed by the analysis backend.
      </p>

      <div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
        <StatTile label="Average Cell Area" value={`${morphology.averageCellArea} px²`} />
        <StatTile label="Circularity" value={morphology.circularity.toFixed(2)} />
        <StatTile label="Aspect Ratio" value={morphology.aspectRatio.toFixed(2)} />
        <StatTile label="Mean Intensity" value={morphology.meanIntensity.toFixed(1)} />
        <StatTile label="Shape Variation" value={morphology.shapeVariation} />
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <div>
          <p className="text-xs font-medium text-muted-foreground">Cell area distribution (px²)</p>
          <div className="mt-3 h-56">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={morphology.areaDistribution}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--color-border)" />
                <XAxis dataKey="bin" tickLine={false} axisLine={false} fontSize={11} />
                <YAxis tickLine={false} axisLine={false} fontSize={11} width={32} />
                <Tooltip
                  cursor={{ fill: "var(--color-muted)" }}
                  contentStyle={{
                    borderRadius: 10,
                    border: "1px solid var(--color-border)",
                    fontSize: 12,
                  }}
                />
                <Bar dataKey="count" fill="var(--color-secondary)" radius={[5, 5, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div>
          <p className="text-xs font-medium text-muted-foreground">Morphology profile</p>
          <div className="mt-3 h-56">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={morphology.profile} outerRadius="72%">
                <PolarGrid stroke="var(--color-border)" />
                <PolarAngleAxis dataKey="metric" fontSize={11} />
                <Radar
                  dataKey="value"
                  stroke="var(--color-accent)"
                  fill="var(--color-accent)"
                  fillOpacity={0.25}
                />
                <Tooltip
                  contentStyle={{
                    borderRadius: 10,
                    border: "1px solid var(--color-border)",
                    fontSize: 12,
                  }}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </section>
  );
}
