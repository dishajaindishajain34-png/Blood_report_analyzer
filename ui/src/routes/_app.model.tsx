import { useQuery } from "@tanstack/react-query";
import { createFileRoute } from "@tanstack/react-router";
import { ArrowDown } from "lucide-react";

import { PageHeader, Disclaimer } from "@/components/common/States";
import { StatTile } from "@/components/common/MetricCard";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { api, queryKeys } from "@/services/api";

export const Route = createFileRoute("/_app/model")({
  head: () => ({
    meta: [
      { title: "Model Information — HemaVision AI" },
      {
        name: "description",
        content: "Detection, classification and processing components behind the analysis pipeline.",
      },
      { property: "og:title", content: "Model Information — HemaVision AI" },
      {
        property: "og:description",
        content: "Detection, classification and processing components behind the analysis pipeline.",
      },
    ],
  }),
  component: ModelPage,
});

const pipeline = [
  "Image",
  "Preprocessing",
  "YOLOv8 Detection",
  "Cell Cropping",
  "EfficientNet Classification",
  "Morphology Analysis",
  "Risk Indication",
  "Report Generation",
];

function ModelPage() {
  const { data, isLoading } = useQuery({ queryKey: queryKeys.health, queryFn: api.health });

  return (
    <div className="space-y-6">
      <PageHeader
        title="Model Information"
        subtitle="Components of the deep learning analysis pipeline running in the backend."
      />

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatTile label="Detection Model" value="YOLOv8" />
        <StatTile label="Classification Model" value="EfficientNet" />
        <StatTile label="Image Processing" value="OpenCV" />
        <StatTile label="Data Augmentation" value="GAN-based pipeline" />
      </div>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)] lg:items-start">
        <section className="panel p-5">
          <h3 className="text-sm font-semibold">Analysis pipeline</h3>
          <p className="text-xs text-muted-foreground">
            Each stage executes in the existing Python analysis service.
          </p>
          <ol className="mt-5 space-y-1.5">
            {pipeline.map((stage, i) => (
              <li key={stage}>
                <div className="flex items-center gap-3 rounded-lg border border-border bg-muted/40 px-4 py-3">
                  <span className="grid size-6 shrink-0 place-items-center rounded-full bg-secondary/10 text-[11px] font-semibold text-secondary">
                    {i + 1}
                  </span>
                  <span className="text-sm font-medium">{stage}</span>
                </div>
                {i < pipeline.length - 1 && (
                  <ArrowDown
                    className="mx-auto my-1 size-3.5 text-muted-foreground"
                    aria-hidden="true"
                  />
                )}
              </li>
            ))}
          </ol>
        </section>

        <div className="space-y-4">
          <section className="panel p-5">
            <h3 className="text-sm font-semibold">Model status</h3>
            {isLoading || !data ? (
              <div className="mt-4 space-y-3">
                {Array.from({ length: 3 }).map((_, i) => (
                  <Skeleton key={i} className="h-10 w-full" />
                ))}
              </div>
            ) : (
              <ul className="mt-4 space-y-3 text-sm">
                <StatusRow
                  label="Detection Model"
                  detail={`${data.detectionModel.name} · ${data.detectionModel.version}`}
                  ok={data.detectionModel.loaded}
                  okLabel="Loaded"
                />
                <StatusRow
                  label="Classification Model"
                  detail={`${data.classificationModel.name} · ${data.classificationModel.version}`}
                  ok={data.classificationModel.loaded}
                  okLabel="Loaded"
                />
                <StatusRow
                  label="Inference Engine"
                  detail={data.inferenceEngine.name}
                  ok={data.inferenceEngine.online}
                  okLabel="Online"
                />
              </ul>
            )}
          </section>

          <section className="panel p-5">
            <h3 className="text-sm font-semibold">Analysis capabilities</h3>
            <ul className="mt-3 space-y-2 text-sm text-muted-foreground">
              {[
                "Blood cell detection",
                "Cell classification",
                "Morphological analysis",
                "Anemia risk indication",
              ].map((item) => (
                <li key={item} className="flex items-center gap-2">
                  <span className="size-1.5 rounded-full bg-accent" aria-hidden="true" />
                  {item}
                </li>
              ))}
            </ul>
          </section>
        </div>
      </div>

      <Disclaimer />
    </div>
  );
}

function StatusRow({
  label,
  detail,
  ok,
  okLabel,
}: {
  label: string;
  detail: string;
  ok: boolean;
  okLabel: string;
}) {
  return (
    <li className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3 border-b border-dashed border-border pb-3 last:border-0 last:pb-0">
      <div className="min-w-0">
        <p className="truncate font-medium">{label}</p>
        <p className="truncate text-xs text-muted-foreground">{detail}</p>
      </div>
      <Badge
        variant="outline"
        className={
          ok
            ? "gap-1.5 border-success/30 bg-success/10 text-success"
            : "gap-1.5 border-destructive/30 bg-destructive/10 text-destructive"
        }
      >
        <span className="size-1.5 rounded-full bg-current" aria-hidden="true" />
        {ok ? okLabel : "Unavailable"}
      </Badge>
    </li>
  );
}
