import { useRef, useState } from "react";
import { Maximize2, ZoomIn, ZoomOut, RotateCcw } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";
import type { Analysis, CellType } from "@/types";

const boxColor: Record<CellType, string> = {
  RBC: "var(--color-chart-1)",
  WBC: "var(--color-chart-2)",
  Platelet: "var(--color-chart-3)",
};

export function ImageViewer({ analysis }: { analysis: Analysis }) {
  const [zoom, setZoom] = useState(1);
  const [showLabels, setShowLabels] = useState(true);
  const containerRef = useRef<HTMLDivElement>(null);

  const clamp = (z: number) => Math.min(Math.max(z, 1), 4);

  const requestFullscreen = () => {
    void containerRef.current?.requestFullscreen?.();
  };

  return (
    <section className="panel overflow-hidden" aria-label="Blood smear image viewer">
      <Tabs defaultValue="detected">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-4 py-3">
          <TabsList>
            <TabsTrigger value="original">Original Image</TabsTrigger>
            <TabsTrigger value="detected">Detected Cells</TabsTrigger>
            <TabsTrigger value="processed">Processed Image</TabsTrigger>
          </TabsList>
          <div className="flex items-center gap-1">
            <Button
              variant="outline"
              size="icon"
              aria-label="Zoom out"
              onClick={() => setZoom((z) => clamp(z - 0.25))}
            >
              <ZoomOut className="size-4" />
            </Button>
            <span className="w-12 text-center text-xs tabular-nums text-muted-foreground">
              {Math.round(zoom * 100)}%
            </span>
            <Button
              variant="outline"
              size="icon"
              aria-label="Zoom in"
              onClick={() => setZoom((z) => clamp(z + 0.25))}
            >
              <ZoomIn className="size-4" />
            </Button>
            <Button variant="outline" size="icon" aria-label="Reset zoom" onClick={() => setZoom(1)}>
              <RotateCcw className="size-4" />
            </Button>
            <Button
              variant="outline"
              size="icon"
              aria-label="Fullscreen"
              onClick={requestFullscreen}
            >
              <Maximize2 className="size-4" />
            </Button>
          </div>
        </div>

        <div ref={containerRef} className="relative overflow-auto bg-primary/[0.03]">
          <TabsContent value="original" className="m-0">
            <Frame zoom={zoom}>
              <img
                src={analysis.imageUrl}
                alt="Original blood smear microscopy image"
                className="w-full select-none"
              />
            </Frame>
          </TabsContent>

          <TabsContent value="detected" className="m-0">
            <Frame zoom={zoom}>
              <img
                src={analysis.imageUrl}
                alt="Blood smear with AI detection overlays"
                className="w-full select-none"
              />
              {analysis.detections.map((d) => (
                <div
                  key={d.id}
                  className="absolute rounded-[3px] border-2"
                  style={{
                    left: `${d.bbox.x * 100}%`,
                    top: `${d.bbox.y * 100}%`,
                    width: `${d.bbox.width * 100}%`,
                    height: `${d.bbox.height * 100}%`,
                    borderColor: boxColor[d.cellType],
                    boxShadow: `0 0 0 1px oklch(1 0 0 / 35%)`,
                  }}
                  title={`${d.cellType} · ${(d.confidence * 100).toFixed(0)}%`}
                >
                  {showLabels && (
                    <span
                      className="absolute -top-4 left-0 rounded-[3px] px-1 text-[9px] leading-4 font-semibold whitespace-nowrap text-white"
                      style={{ backgroundColor: boxColor[d.cellType] }}
                    >
                      {d.cellType} {(d.confidence * 100).toFixed(0)}%
                    </span>
                  )}
                </div>
              ))}
            </Frame>
          </TabsContent>

          <TabsContent value="processed" className="m-0">
            <Frame zoom={zoom}>
              <img
                src={analysis.processedImageUrl ?? analysis.imageUrl}
                alt="Preprocessed blood smear image"
                className="w-full select-none contrast-125 saturate-50"
              />
            </Frame>
          </TabsContent>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3 border-t border-border px-4 py-3">
          <ul className="flex flex-wrap items-center gap-4 text-xs text-muted-foreground">
            {(Object.keys(boxColor) as CellType[]).map((type) => (
              <li key={type} className="flex items-center gap-1.5">
                <span
                  className="size-2.5 rounded-[2px]"
                  style={{ backgroundColor: boxColor[type] }}
                  aria-hidden="true"
                />
                {type}
              </li>
            ))}
          </ul>
          <div className="flex items-center gap-2">
            <Switch id="labels" checked={showLabels} onCheckedChange={setShowLabels} />
            <Label htmlFor="labels" className="text-xs text-muted-foreground">
              Show labels
            </Label>
          </div>
        </div>
      </Tabs>
    </section>
  );
}

function Frame({ zoom, children }: { zoom: number; children: React.ReactNode }) {
  return (
    <div className={cn("origin-top-left transition-transform duration-200")} style={{ transform: `scale(${zoom})`, width: `${100}%` }}>
      <div className="relative">{children}</div>
    </div>
  );
}
