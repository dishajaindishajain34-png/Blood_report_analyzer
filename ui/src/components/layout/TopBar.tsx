import { Link, useRouterState } from "@tanstack/react-router";
import { Bell, Search, ChevronRight } from "lucide-react";

import { SidebarTrigger } from "@/components/ui/sidebar";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

const titles: Record<string, string> = {
  "/dashboard": "AI Analysis Dashboard",
  "/analyze": "Analyze Blood Smear",
  "/history": "Analysis History",
  "/reports": "Analysis Reports",
  "/model": "Model Information",
  "/settings": "Settings",
};

export function TopBar() {
  const pathname = useRouterState({ select: (r) => r.location.pathname });
  const segments = pathname.split("/").filter(Boolean);
  const title = titles[`/${segments[0] ?? ""}`] ?? "Analysis Results";

  return (
    <header className="sticky top-0 z-30 border-b border-border bg-background/85 backdrop-blur-md">
      <div className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3 px-4 py-3 sm:px-6">
        <div className="flex min-w-0 items-center gap-3">
          <SidebarTrigger aria-label="Toggle navigation" className="shrink-0" />
          <div className="min-w-0">
            <h1 className="truncate text-base font-semibold tracking-tight sm:text-lg">{title}</h1>
            <nav aria-label="Breadcrumb" className="hidden items-center gap-1 text-xs text-muted-foreground sm:flex">
              <Link to="/dashboard" className="hover:text-foreground">
                Home
              </Link>
              {segments.map((segment, i) => (
                <span key={segment + i} className="flex items-center gap-1">
                  <ChevronRight className="size-3" aria-hidden="true" />
                  <span className="capitalize">{segment}</span>
                </span>
              ))}
            </nav>
          </div>
        </div>

        <div className="flex items-center gap-1.5">
          <div className="relative hidden lg:block">
            <Search
              className="pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground"
              aria-hidden="true"
            />
            <Input
              type="search"
              placeholder="Search analyses..."
              aria-label="Search analyses"
              className="h-9 w-56 bg-card pl-8"
            />
          </div>
          <Button variant="ghost" size="icon" aria-label="Search" className="lg:hidden min-h-11 min-w-11">
            <Search className="size-4" />
          </Button>
          <Button variant="ghost" size="icon" aria-label="Notifications" className="relative min-h-11 min-w-11">
            <Bell className="size-4" />
            <span className="absolute top-2.5 right-2.5 size-1.5 rounded-full bg-accent" />
          </Button>
          <span className="ml-1 grid size-9 shrink-0 place-items-center rounded-full bg-primary text-xs font-semibold text-primary-foreground">
            TG
          </span>
        </div>
      </div>
    </header>
  );
}
