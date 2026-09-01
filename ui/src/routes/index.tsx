import { createFileRoute, Link } from "@tanstack/react-router";
import { motion } from "motion/react";
import { ScanSearch, Boxes, Shapes, FileBarChart, ArrowRight, Activity } from "lucide-react";

import heroVisual from "@/assets/hero-visual.jpg";
import { Logo } from "@/components/brand/Logo";
import { Button } from "@/components/ui/button";
import { Disclaimer } from "@/components/common/States";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "HemaVision AI — Intelligent Blood Smear Analysis" },
      {
        name: "description",
        content:
          "AI-powered blood smear analysis: deep learning cell detection, classification, morphology analysis and anemia risk indication for research and screening assistance.",
      },
      { property: "og:title", content: "HemaVision AI — Intelligent Blood Smear Analysis" },
      {
        property: "og:description",
        content:
          "Deep learning-based blood cell detection, classification and morphological analysis with structured AI reporting.",
      },
    ],
  }),
  component: Landing,
});

const features = [
  {
    icon: ScanSearch,
    title: "Cell Detection",
    body: "Detect and localize blood cells from microscopy images.",
  },
  {
    icon: Boxes,
    title: "Cell Classification",
    body: "Classify detected blood cells using deep learning.",
  },
  {
    icon: Shapes,
    title: "Morphology Analysis",
    body: "Analyze cell morphology and visual characteristics.",
  },
  {
    icon: FileBarChart,
    title: "AI Report",
    body: "Generate a structured analysis report from model results.",
  },
];

function Landing() {
  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-background/85 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-5 py-4">
          <Logo />
          <nav className="flex items-center gap-2">
            <Button asChild variant="ghost" size="sm">
              <Link to="/model">Model</Link>
            </Button>
            <Button asChild size="sm">
              <Link to="/dashboard">Open App</Link>
            </Button>
          </nav>
        </div>
      </header>

      <main>
        <section className="relative overflow-hidden border-b border-border">
          <div className="grid-lab absolute inset-0" aria-hidden="true" />
          <div className="relative mx-auto grid max-w-6xl gap-10 px-5 py-16 lg:grid-cols-[1.05fr_1fr] lg:items-center lg:py-24">
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, ease: "easeOut" }}
            >
              <span className="inline-flex items-center gap-2 rounded-full border border-secondary/25 bg-secondary/8 px-3 py-1 text-xs font-medium text-secondary">
                <Activity className="size-3.5" aria-hidden="true" />
                YOLOv8 detection · EfficientNet classification
              </span>
              <h1 className="mt-5 text-4xl font-bold tracking-tight text-balance sm:text-5xl lg:text-[3.4rem] lg:leading-[1.05]">
                Intelligent Blood Smear Analysis
              </h1>
              <p className="mt-5 max-w-xl text-base leading-relaxed text-muted-foreground">
                Analyze blood smear images using deep learning-based cell detection, classification
                and morphological analysis.
              </p>
              <div className="mt-8 flex flex-wrap gap-3">
                <Button asChild size="lg">
                  <Link to="/analyze">
                    Analyze Blood Smear
                    <ArrowRight className="size-4" aria-hidden="true" />
                  </Link>
                </Button>
                <Button asChild size="lg" variant="outline">
                  <Link to="/dashboard">Explore System</Link>
                </Button>
              </div>
              <dl className="mt-10 grid max-w-lg grid-cols-3 gap-4 border-t border-border pt-6">
                {[
                  ["Detection classes", "RBC · WBC · Platelet"],
                  ["Avg. inference", "2.4 s"],
                  ["Pipeline stages", "6"],
                ].map(([label, value]) => (
                  <div key={label}>
                    <dt className="text-[11px] tracking-wide text-muted-foreground uppercase">
                      {label}
                    </dt>
                    <dd className="mt-1 text-sm font-semibold">{value}</dd>
                  </div>
                ))}
              </dl>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, scale: 0.97 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.6, ease: "easeOut", delay: 0.1 }}
              className="relative overflow-hidden rounded-2xl border border-border shadow-[var(--shadow-raised)]"
            >
              <img
                src={heroVisual}
                width={1200}
                height={912}
                alt="Abstract visualization of blood cells with computer-vision detection boxes"
                className="h-full w-full object-cover"
              />
              <div className="absolute inset-x-0 bottom-0 flex items-center justify-between gap-3 bg-primary/85 px-4 py-3 text-xs text-primary-foreground backdrop-blur">
                <span className="font-medium">Detection overlay preview</span>
                <span className="font-mono text-accent">conf ≥ 0.72</span>
              </div>
            </motion.div>
          </div>
        </section>

        <section className="mx-auto max-w-6xl px-5 py-16">
          <h2 className="text-2xl font-semibold tracking-tight">What the system does</h2>
          <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
            Each stage of the pipeline runs in the analysis backend; the interface collects the
            input and presents the model output.
          </p>
          <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {features.map((feature, i) => (
              <motion.article
                key={feature.title}
                initial={{ opacity: 0, y: 14 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-60px" }}
                transition={{ duration: 0.35, delay: i * 0.06 }}
                className="panel p-5 transition-shadow hover:shadow-[var(--shadow-raised)]"
              >
                <span className="grid size-9 place-items-center rounded-lg bg-secondary/10 text-secondary">
                  <feature.icon className="size-4.5" aria-hidden="true" />
                </span>
                <h3 className="mt-4 text-sm font-semibold">{feature.title}</h3>
                <p className="mt-1.5 text-sm leading-relaxed text-muted-foreground">
                  {feature.body}
                </p>
              </motion.article>
            ))}
          </div>
          <Disclaimer className="mt-8" />
        </section>
      </main>

      <footer className="border-t border-border">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 px-5 py-6 text-xs text-muted-foreground">
          <span>HemaVision AI · AI-powered blood smear analysis</span>
          <span>Research prototype · v1.0.0</span>
        </div>
      </footer>
    </div>
  );
}
