import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Download, FileText } from "lucide-react";
import { api } from "@/services/api";

export function ReportPreviewDialog({
  open,
  onOpenChange,
  analysis,
  report,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  analysis: any;
  report: any;
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Download Report</DialogTitle>
          <DialogDescription>
            Your analysis report is ready to download.
          </DialogDescription>
        </DialogHeader>
        <div className="flex items-center justify-center p-6 bg-muted/20 rounded-md border border-dashed">
          <div className="flex flex-col items-center gap-2 text-muted-foreground">
            <FileText className="size-10" />
            <p className="text-sm font-medium text-foreground">Clinical Report (PDF)</p>
            <p className="text-xs">ID: {report?.id}</p>
          </div>
        </div>
        <DialogFooter>
          <Button variant="secondary" onClick={() => onOpenChange(false)}>
            Close
          </Button>
          <Button asChild disabled={!report?.analysisId}>
            <a href={api.downloadReportUrl(report?.analysisId ?? "")} download>
              <Download className="mr-2 size-4" />
              Download PDF
            </a>
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
