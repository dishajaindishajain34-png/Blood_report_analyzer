import { useCallback, useRef, useState } from "react";
import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useQueryClient } from "@tanstack/react-query";
import { motion, AnimatePresence } from "motion/react";
import { Check, Loader2, Microscope, Trash2, UploadCloud, Sparkles } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Progress } from "@/components/ui/progress";
import { Disclaimer, ErrorState, PageHeader } from "@/components/common/States";
import { api, queryKeys } from "@/services/api";
import { formatBytes } from "@/utils/format";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/_app/analyze")({
  head: () => ({
    meta: [
      { title: "Analyze Blood Smear — HemaVision AI" },
      {
        name: "description",
        content: "Upload a microscopy image for AI-assisted blood cell detection and analysis.",
      },
      { property: "og:title", content: "Analyze Blood Smear — HemaVision AI" },
      {
        property: "og:description",
        content: "Upload a microscopy image for AI-assisted blood cell detection and analysis.",
      },
    ],
  }),
  component: AnalyzePage,
});

const STEPS = [
  "Uploading image",
  "Image preprocessing",
  "Detecting blood cells",
  "Classifying cells",
  "Performing morphology analysis",
  "Generating results",
];

const MAX_MB = 15;
const ACCEPTED = ["image/jpeg", "image/jpg", "image/png"];

function AnalyzePage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const inputRef = useRef<HTMLInputElement>(null);

  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [dimensions, setDimensions] = useState<string>("—");
  const [dragging, setDragging] = useState(false);
  const [patientId, setPatientId] = useState("");
  const [sampleId, setSampleId] = useState("");
  const [notes, setNotes] = useState("");
  
  const [gender, setGender] = useState("");
  const [hb, setHb] = useState("");
  const [mcv, setMcv] = useState("");
  const [mch, setMch] = useState("");
  const [mchc, setMchc] = useState("");
  
  const [step, setStep] = useState(0);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const accept = useCallback((incoming: File) => {
    if (!ACCEPTED.includes(incoming.type)) {
      toast.error("Unsupported file type. Please upload a JPG, JPEG or PNG image.");
      return;
    }
    if (incoming.size > MAX_MB * 1024 * 1024) {
      toast.error(`Image is larger than ${MAX_MB} MB.`);
      return;
    }
    const url = URL.createObjectURL(incoming);
    const img = new Image();
    img.onload = () => setDimensions(`${img.naturalWidth} × ${img.naturalHeight} px`);
    img.src = url;
    setFile(incoming);
    setPreviewUrl(url);
    setError(null);
  }, []);

  const reset = () => {
    setFile(null);
    setPreviewUrl(null);
    setDimensions("—");
    setStep(0);
  };

  const runAnalysis = async () => {
    if (!file) return;
    setRunning(true);
    setError(null);
    setStep(0);
    try {
      const payload: any = { file };
      if (patientId) payload.patientId = patientId;
      if (sampleId) payload.sampleId = sampleId;
      if (notes) payload.notes = notes;
      
      if (gender) payload.gender = gender;
      if (hb) payload.hb = parseFloat(hb);
      if (mcv) payload.mcv = parseFloat(mcv);
      if (mch) payload.mch = parseFloat(mch);
      if (mchc) payload.mchc = parseFloat(mchc);

      const analysis = await api.analyze(payload, (s) => setStep(s));
      queryClient.setQueryData(queryKeys.analysis(analysis.id), analysis);
      toast.success("Analysis complete");
      void navigate({ to: "/results/$analysisId", params: { analysisId: analysis.id } });
    } catch (err: any) {
      const msg = err?.message || "Image analysis failed. Please check that the FastAPI backend server is running on http://127.0.0.1:8000.";
      setError(msg);
      toast.error(msg);
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="space-y-6">
      <PageHeader
        title="Analyze Blood Smear"
        subtitle="Upload a microscopy image for AI-assisted blood cell analysis."
      />

      <div className="panel border-dashed border-primary/25 bg-primary/[0.02] p-4 text-sm text-muted-foreground">
        <span className="font-medium text-foreground">Single-image workflow:</span> the same smear image is used first for CBC component analysis and anemia screening. If anemia is inconclusive or the image quality is poor, the system will prompt for a dedicated RBC/anemia image for confirmation.
      </div>

      {error ? (
        <ErrorState
          title="Image analysis failed."
          description={error}
          onRetry={() => void runAnalysis()}
        />
      ) : (
        <div className="grid gap-6 lg:grid-cols-[minmax(0,1.55fr)_minmax(0,1fr)] lg:items-start">
          <section className="panel p-5">
            <AnimatePresence mode="wait">
              {!file ? (
                <motion.div
                  key="dropzone"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                >
                  <div
                    role="button"
                    tabIndex={0}
                    aria-label="Upload blood smear image"
                    onClick={() => inputRef.current?.click()}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        inputRef.current?.click();
                      }
                    }}
                    onDragOver={(e) => {
                      e.preventDefault();
                      setDragging(true);
                    }}
                    onDragLeave={() => setDragging(false)}
                    onDrop={(e) => {
                      e.preventDefault();
                      setDragging(false);
                      const dropped = e.dataTransfer.files?.[0];
                      if (dropped) accept(dropped);
                    }}
                    className={cn(
                      "grid cursor-pointer place-items-center rounded-xl border-2 border-dashed px-6 py-16 text-center transition-colors",
                      dragging
                        ? "border-accent bg-accent/8"
                        : "border-border bg-muted/30 hover:border-secondary/50 hover:bg-muted/50",
                    )}
                  >
                    <motion.span
                      animate={{ scale: dragging ? 1.08 : 1 }}
                      transition={{ type: "spring", stiffness: 260, damping: 18 }}
                      className="grid size-14 place-items-center rounded-full bg-secondary/10 text-secondary"
                    >
                      <Microscope className="size-6" aria-hidden="true" />
                    </motion.span>
                    <p className="mt-4 text-sm font-semibold">Drop blood smear image here</p>
                    <p className="mt-1 text-sm text-muted-foreground">or browse from your device</p>
                    <p className="mt-3 max-w-md text-xs text-muted-foreground">
                      One upload is used for CBC detection, RBC/WBC/platelet counts, Hb/Hct estimation, and anemia screening. If the anemia result is uncertain, a dedicated anemia-focused image can be uploaded next.
                    </p>
                    <Button type="button" variant="outline" size="sm" className="mt-5">
                      <UploadCloud className="size-4" aria-hidden="true" />
                      Browse files
                    </Button>
                    <p className="mt-5 text-xs text-muted-foreground">
                      Supported: JPG · JPEG · PNG — maximum {MAX_MB} MB
                    </p>
                  </div>
                  <input
                    ref={inputRef}
                    type="file"
                    accept="image/jpeg,image/png"
                    className="sr-only"
                    onChange={(e) => {
                      const picked = e.target.files?.[0];
                      if (picked) accept(picked);
                    }}
                  />
                </motion.div>
              ) : (
                <motion.div
                  key="preview"
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="space-y-4"
                >
                  <div className="overflow-hidden rounded-xl border border-border">
                    <img
                      src={previewUrl ?? ""}
                      alt="Uploaded blood smear preview"
                      className="max-h-[420px] w-full object-contain bg-primary/[0.03]"
                    />
                  </div>
                  <dl className="grid gap-3 sm:grid-cols-3">
                    {[
                      ["File name", file.name],
                      ["File size", formatBytes(file.size)],
                      ["Dimensions", dimensions],
                    ].map(([label, value]) => (
                      <div key={label} className="rounded-lg border border-border bg-muted/40 px-4 py-3">
                        <dt className="text-[11px] tracking-wide text-muted-foreground uppercase">
                          {label}
                        </dt>
                        <dd className="mt-1 truncate text-sm font-medium">{value}</dd>
                      </div>
                    ))}
                  </dl>
                  <div className="flex flex-wrap gap-2">
                    <Button variant="outline" onClick={reset} disabled={running}>
                      <Trash2 className="size-4" aria-hidden="true" />
                      Remove
                    </Button>
                    <Button onClick={() => void runAnalysis()} disabled={running}>
                      {running ? (
                        <Loader2 className="size-4 animate-spin" aria-hidden="true" />
                      ) : (
                        <Sparkles className="size-4" aria-hidden="true" />
                      )}
                      {running ? "Analyzing…" : "Analyze Image"}
                    </Button>
                  </div>

                  {running && (
                    <div className="rounded-xl border border-border bg-muted/30 p-5">
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-semibold">AI pipeline in progress</span>
                        <span className="tabular-nums text-muted-foreground">
                          {Math.round((step / STEPS.length) * 100)}%
                        </span>
                      </div>
                      <Progress value={(step / STEPS.length) * 100} className="mt-3 h-2" />
                      <ol className="mt-4 space-y-2.5">
                        {STEPS.map((label, i) => {
                          const done = step > i;
                          const active = step === i;
                          return (
                            <li key={label} className="flex items-center gap-2.5 text-xs">
                              <span
                                className={cn(
                                  "grid size-5 shrink-0 place-items-center rounded-full border",
                                  done
                                    ? "border-success bg-success text-success-foreground"
                                    : active
                                      ? "border-accent text-accent"
                                      : "border-border text-muted-foreground",
                                )}
                              >
                                {done ? (
                                  <Check className="size-3" aria-hidden="true" />
                                ) : active ? (
                                  <Loader2 className="size-3 animate-spin" aria-hidden="true" />
                                ) : (
                                  <span className="text-[10px]">{i + 1}</span>
                                )}
                              </span>
                              <span className={done || active ? "font-medium" : "text-muted-foreground"}>
                                {label}
                              </span>
                            </li>
                          );
                        })}
                      </ol>
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </section>

          <aside className="space-y-4">
            <section className="panel p-5">
              <h3 className="text-sm font-semibold">Sample reference (optional)</h3>
              <p className="mt-1 text-xs text-muted-foreground">
                Use non-identifying references only. Do not enter personal medical information.
              </p>
              <div className="mt-4 space-y-4">
                <div className="space-y-1.5">
                  <Label htmlFor="patientId">Patient ID</Label>
                  <Input
                    id="patientId"
                    placeholder="PT-0192"
                    value={patientId}
                    onChange={(e) => setPatientId(e.target.value)}
                  />
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="sampleId">Sample ID</Label>
                  <Input
                    id="sampleId"
                    placeholder="SMP-77341"
                    value={sampleId}
                    onChange={(e) => setSampleId(e.target.value)}
                  />
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="notes">Notes</Label>
                  <Textarea
                    id="notes"
                    rows={3}
                    placeholder="Staining, magnification, field notes…"
                    value={notes}
                    onChange={(e) => setNotes(e.target.value)}
                  />
                </div>
              </div>
            </section>
            
            <section className="panel p-5">
              <h3 className="text-sm font-semibold">Laboratory CBC Inputs</h3>
              <p className="mt-1 text-xs text-muted-foreground">
                Optional: Enter patient CBC values to run the Secondary ML Anemia Screening model.
              </p>
              <div className="mt-4 space-y-4">
                <div className="space-y-1.5">
                  <Label>Gender</Label>
                  <select
                    className="flex h-9 w-full items-center justify-between whitespace-nowrap rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
                    value={gender}
                    onChange={(e) => setGender(e.target.value)}
                  >
                    <option value="">Select Gender</option>
                    <option value="Female">Female</option>
                    <option value="Male">Male</option>
                  </select>
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="hb">Hemoglobin (g/dL)</Label>
                  <Input
                    id="hb"
                    type="number"
                    step="0.1"
                    placeholder="e.g. 12.5"
                    value={hb}
                    onChange={(e) => setHb(e.target.value)}
                  />
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1.5">
                    <Label htmlFor="mcv">MCV (fL)</Label>
                    <Input
                      id="mcv"
                      type="number"
                      step="0.1"
                      placeholder="e.g. 85.0"
                      value={mcv}
                      onChange={(e) => setMcv(e.target.value)}
                    />
                  </div>
                  <div className="space-y-1.5">
                    <Label htmlFor="mch">MCH (pg)</Label>
                    <Input
                      id="mch"
                      type="number"
                      step="0.1"
                      placeholder="e.g. 28.0"
                      value={mch}
                      onChange={(e) => setMch(e.target.value)}
                    />
                  </div>
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="mchc">MCHC (g/dL)</Label>
                  <Input
                    id="mchc"
                    type="number"
                    step="0.1"
                    placeholder="e.g. 33.0"
                    value={mchc}
                    onChange={(e) => setMchc(e.target.value)}
                  />
                </div>
              </div>
            </section>
            
            <Disclaimer />
          </aside>
        </div>
      )}
    </div>
  );
}
