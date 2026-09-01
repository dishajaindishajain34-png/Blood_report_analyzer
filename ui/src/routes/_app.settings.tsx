import { useQuery } from "@tanstack/react-query";
import { createFileRoute } from "@tanstack/react-router";

import { PageHeader } from "@/components/common/States";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { api, API_BASE_URL, USE_MOCK_DATA, queryKeys } from "@/services/api";

export const Route = createFileRoute("/_app/settings")({
  head: () => ({
    meta: [
      { title: "Settings — HemaVision AI" },
      { name: "description", content: "Profile, analysis preferences and system information." },
      { property: "og:title", content: "Settings — HemaVision AI" },
      {
        property: "og:description",
        content: "Profile, analysis preferences and system information.",
      },
    ],
  }),
  component: SettingsPage,
});

function SettingsPage() {
  const { data } = useQuery({ queryKey: queryKeys.health, queryFn: api.health });

  return (
    <div className="space-y-6">
      <PageHeader title="Settings" subtitle="Manage your profile, preferences and system details." />

      <div className="grid gap-6 lg:grid-cols-2">
        <section className="panel p-5">
          <h3 className="text-sm font-semibold">Profile</h3>
          <div className="mt-4 space-y-4">
            <div className="space-y-1.5">
              <Label htmlFor="name">Full name</Label>
              <Input id="name" defaultValue="Trupti G. C." />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="role">Role</Label>
              <Input id="role" defaultValue="Lab Researcher" />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="email">Email</Label>
              <Input id="email" type="email" defaultValue="researcher@hemavision.ai" />
            </div>
            <Button size="sm">Save profile</Button>
          </div>
        </section>

        <section className="panel p-5">
          <h3 className="text-sm font-semibold">Application Preferences</h3>
          <div className="mt-4 space-y-4">
            <ToggleRow id="compact" label="Compact tables" description="Reduce table row height." />
            <ToggleRow id="autoreport" label="Auto-open report" description="Open the report preview after each analysis." defaultChecked />
            <div className="space-y-1.5">
              <Label htmlFor="units">Measurement units</Label>
              <Select defaultValue="si">
                <SelectTrigger id="units">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="si">SI units</SelectItem>
                  <SelectItem value="conventional">Conventional</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </section>

        <section className="panel p-5">
          <h3 className="text-sm font-semibold">Analysis Preferences</h3>
          <div className="mt-4 space-y-4">
            <div className="space-y-1.5">
              <Label htmlFor="threshold">Detection confidence threshold</Label>
              <Input id="threshold" type="number" min={0} max={1} step={0.05} defaultValue={0.7} />
            </div>
            <ToggleRow id="overlay" label="Show detection labels by default" defaultChecked />
            <ToggleRow id="morph" label="Run morphology analysis" description="Requires backend support." defaultChecked />
          </div>
        </section>

        <section className="panel p-5">
          <h3 className="text-sm font-semibold">Notifications</h3>
          <div className="mt-4 space-y-4">
            <ToggleRow id="n-complete" label="Analysis complete" defaultChecked />
            <ToggleRow id="n-review" label="Review required" defaultChecked />
            <ToggleRow id="n-failed" label="Analysis failed" />
          </div>
        </section>
      </div>

      <section className="panel p-5">
        <h3 className="text-sm font-semibold">System Information</h3>
        <dl className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <Info label="Frontend Version" value={data?.frontendVersion ?? "1.0.0"} />
          <Info
            label="Backend Status"
            value={<StatusPill ok={Boolean(data?.apiOnline)} okLabel="Connected" />}
          />
          <Info
            label="AI Model Status"
            value={<StatusPill ok={Boolean(data?.detectionModel.loaded)} okLabel="Loaded" />}
          />
          <Info
            label="API Endpoint"
            value={
              <span className="font-mono text-xs break-all">
                {API_BASE_URL}
                {USE_MOCK_DATA && " (mock mode)"}
              </span>
            }
          />
        </dl>
      </section>
    </div>
  );
}

function ToggleRow({
  id,
  label,
  description,
  defaultChecked,
}: {
  id: string;
  label: string;
  description?: string;
  defaultChecked?: boolean | undefined;
}) {
  return (
    <div className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3">
      <div className="min-w-0">
        <Label htmlFor={id} className="text-sm font-medium">
          {label}
        </Label>
        {description && <p className="mt-0.5 text-xs text-muted-foreground">{description}</p>}
      </div>
      <Switch id={id} defaultChecked={defaultChecked ?? false} />
    </div>
  );
}

function Info({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-border bg-muted/40 px-4 py-3">
      <dt className="text-[11px] tracking-wide text-muted-foreground uppercase">{label}</dt>
      <dd className="mt-1 text-sm font-medium">{value}</dd>
    </div>
  );
}

function StatusPill({ ok, okLabel }: { ok: boolean; okLabel: string }) {
  return (
    <Badge
      variant="outline"
      className={
        ok
          ? "gap-1.5 border-success/30 bg-success/10 text-success"
          : "gap-1.5 border-warning/40 bg-warning/15 text-warning"
      }
    >
      <span className="size-1.5 rounded-full bg-current" aria-hidden="true" />
      {ok ? okLabel : "Checking…"}
    </Badge>
  );
}
