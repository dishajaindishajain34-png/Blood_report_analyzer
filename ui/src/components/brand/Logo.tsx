import { cn } from "@/lib/utils";

/** HemaVision mark: an erythrocyte ring fused with a neural-network graph. */
export function LogoMark({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 32 32"
      role="img"
      aria-label="HemaVision AI logo"
      className={cn("size-8", className)}
    >
      <circle cx="16" cy="16" r="14" fill="none" stroke="currentColor" strokeWidth="2" opacity="0.9" />
      <circle cx="16" cy="16" r="6.5" fill="none" stroke="currentColor" strokeWidth="1.5" opacity="0.45" />
      <g stroke="currentColor" strokeWidth="1.2" opacity="0.75">
        <line x1="16" y1="16" x2="6.5" y2="9.5" />
        <line x1="16" y1="16" x2="25.5" y2="9.5" />
        <line x1="16" y1="16" x2="8" y2="24" />
        <line x1="16" y1="16" x2="24" y2="24" />
      </g>
      <g fill="currentColor">
        <circle cx="6.5" cy="9.5" r="2.1" />
        <circle cx="25.5" cy="9.5" r="2.1" />
        <circle cx="8" cy="24" r="2.1" />
        <circle cx="24" cy="24" r="2.1" />
        <circle cx="16" cy="16" r="2.4" />
      </g>
    </svg>
  );
}

export function Logo({
  className,
  compact = false,
  tone = "light",
}: {
  className?: string;
  compact?: boolean;
  tone?: "light" | "dark";
}) {
  return (
    <span className={cn("flex items-center gap-2.5", className)}>
      <LogoMark className={tone === "dark" ? "size-8 text-accent" : "size-8 text-secondary"} />
      {!compact && (
        <span className="flex flex-col leading-none">
          <span
            className={cn(
              "text-[15px] font-bold tracking-tight",
              tone === "dark" ? "text-sidebar-foreground" : "text-foreground",
            )}
          >
            HemaVision
            <span className={tone === "dark" ? "text-accent" : "text-secondary"}> AI</span>
          </span>
          <span className="mt-1 text-[10px] font-medium tracking-[0.14em] text-muted-foreground uppercase">
            Smear Analysis
          </span>
        </span>
      )}
    </span>
  );
}
