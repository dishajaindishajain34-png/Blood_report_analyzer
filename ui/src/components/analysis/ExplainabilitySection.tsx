import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import type { Analysis } from "@/types";
import { formatPercent } from "@/utils/format";

export function ExplainabilitySection({ analysis }: { analysis: Analysis }) {
  const { explainability } = analysis;

  return (
    <section className="panel p-5" aria-labelledby="explainability-heading">
      <h3 id="explainability-heading" className="text-sm font-semibold">
        Model Explainability
      </h3>
      <p className="text-xs text-muted-foreground">{explainability.note}</p>

      <Tabs defaultValue="attention" className="mt-4">
        <TabsList>
          <TabsTrigger value="attention">Attention Map</TabsTrigger>
          <TabsTrigger value="confidence">Detection Confidence</TabsTrigger>
          <TabsTrigger value="features">Feature Summary</TabsTrigger>
        </TabsList>

        <TabsContent value="attention" className="mt-4">
          <div className="relative overflow-hidden rounded-xl border border-border">
            <img
              src={explainability.attentionMapUrl ?? analysis.imageUrl}
              alt="Blood smear with model attention overlay"
              loading="lazy"
              className="w-full grayscale"
            />
            <div
              aria-hidden="true"
              className="absolute inset-0 mix-blend-multiply"
              style={{
                background:
                  "radial-gradient(circle at 32% 40%, oklch(0.62 0.17 20 / 55%) 0%, transparent 26%), radial-gradient(circle at 68% 62%, oklch(0.78 0.15 78 / 50%) 0%, transparent 24%), radial-gradient(circle at 52% 25%, oklch(0.704 0.14 182 / 40%) 0%, transparent 22%)",
              }}
            />
            <span className="absolute right-3 bottom-3 rounded-md bg-primary/85 px-2 py-1 text-[11px] text-primary-foreground">
              Placeholder — Grad-CAM output slots in here
            </span>
          </div>
        </TabsContent>

        <TabsContent value="confidence" className="mt-4 space-y-4">
          {explainability.confidenceByClass.map((c) => (
            <div key={c.cellType}>
              <div className="flex items-center justify-between text-xs">
                <span className="font-medium">{c.cellType}</span>
                <span className="tabular-nums text-muted-foreground">
                  {formatPercent(c.confidence, 1)}
                </span>
              </div>
              <Progress value={c.confidence * 100} className="mt-2 h-2" />
            </div>
          ))}
        </TabsContent>

        <TabsContent value="features" className="mt-4">
          <dl className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {explainability.featureSummary.map((f) => (
              <div key={f.feature} className="rounded-lg border border-border bg-muted/40 px-4 py-3">
                <dt className="text-[11px] tracking-wide text-muted-foreground uppercase">
                  {f.feature}
                </dt>
                <dd className="mt-1 text-sm font-semibold">{f.value}</dd>
              </div>
            ))}
          </dl>
        </TabsContent>
      </Tabs>
    </section>
  );
}
