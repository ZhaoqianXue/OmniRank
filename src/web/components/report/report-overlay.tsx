"use client";

import { useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Download, FileText, BarChart3 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { RankingChart } from "@/components/visualizations";
import type { RankingResults } from "@/lib/api";

interface ReportOverlayProps {
  isVisible: boolean;
  results: RankingResults | null;
  onClose: () => void;
  className?: string;
}

export function ReportOverlay({
  isVisible,
  results,
  onClose,
  className,
}: ReportOverlayProps) {
  const reportRef = useRef<HTMLDivElement>(null);

  // Export report as PDF
  const handleExportPDF = async () => {
    if (!reportRef.current) return;

    try {
      // Dynamic import to avoid SSR issues
      const html2canvas = (await import("html2canvas")).default;
      const jsPDF = (await import("jspdf")).default;

      const canvas = await html2canvas(reportRef.current, {
        scale: 2,
        useCORS: true,
        logging: false,
        backgroundColor: "#ffffff",
      });

      const imgData = canvas.toDataURL("image/png");
      const pdf = new jsPDF({
        orientation: "portrait",
        unit: "mm",
        format: "a4",
      });

      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = pdf.internal.pageSize.getHeight();
      const imgWidth = canvas.width;
      const imgHeight = canvas.height;
      const ratio = Math.min(pdfWidth / imgWidth, pdfHeight / imgHeight);
      const imgX = (pdfWidth - imgWidth * ratio) / 2;
      const imgY = 10;

      pdf.addImage(imgData, "PNG", imgX, imgY, imgWidth * ratio, imgHeight * ratio);
      pdf.save("omnirank-report.pdf");
    } catch (error) {
      console.error("Failed to export PDF:", error);
      // Fallback: print the page
      window.print();
    }
  };

  if (!results) return null;

  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          className={cn(
            "absolute inset-0 z-50 bg-background/95 backdrop-blur-sm rounded-lg overflow-hidden",
            className
          )}
        >
          {/* Header with action buttons */}
          <div className="absolute top-0 left-0 right-0 flex items-center justify-between px-4 py-3 bg-background/80 backdrop-blur-sm border-b border-border/40 z-10">
            <div className="flex items-center gap-2">
              <FileText className="h-5 w-5 text-primary" />
              <h2 className="text-lg font-semibold">Ranking Report</h2>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={handleExportPDF}
                className="h-8"
              >
                <Download className="h-4 w-4 mr-1.5" />
                Export PDF
              </Button>
              <Button
                variant="ghost"
                size="icon"
                onClick={onClose}
                className="h-8 w-8 hover:bg-destructive/10 hover:text-destructive"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          </div>

          {/* Report content - scrollable */}
          <div className="absolute inset-0 top-14 overflow-auto">
            <div
              ref={reportRef}
              className="min-h-full p-6 bg-white dark:bg-zinc-900"
            >
              {/* Report placeholder - content will be provided later */}
              <div className="max-w-4xl mx-auto space-y-6">
                {/* Title section */}
                <div className="text-center border-b border-border/40 pb-6">
                  <h1 className="text-2xl font-bold mb-2">OmniRank Analysis Report</h1>
                  <p className="text-muted-foreground text-sm">
                    Generated on {new Date().toLocaleDateString("en-US", {
                      year: "numeric",
                      month: "long",
                      day: "numeric",
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </p>
                </div>

                {/* Summary section */}
                <div className="space-y-4">
                  <h2 className="text-lg font-semibold">Summary</h2>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="p-4 bg-muted/50 rounded-lg">
                      <p className="text-xs text-muted-foreground uppercase tracking-wide">Total Items</p>
                      <p className="text-2xl font-bold mt-1">{results.items.length}</p>
                    </div>
                    <div className="p-4 bg-muted/50 rounded-lg">
                      <p className="text-xs text-muted-foreground uppercase tracking-wide">Top Ranked</p>
                      <p className="text-2xl font-bold mt-1 truncate">
                        {results.items.find(i => i.rank === 1)?.name || "-"}
                      </p>
                    </div>
                    <div className="p-4 bg-muted/50 rounded-lg">
                      <p className="text-xs text-muted-foreground uppercase tracking-wide">Comparisons</p>
                      <p className="text-2xl font-bold mt-1">{results.n_pairs}</p>
                    </div>
                    <div className="p-4 bg-muted/50 rounded-lg">
                      <p className="text-xs text-muted-foreground uppercase tracking-wide">Direction</p>
                      <p className="text-2xl font-bold mt-1">
                        {results.bigbetter === 1 ? "↑ Higher" : "↓ Lower"}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Rankings table */}
                <div className="space-y-4">
                  <h2 className="text-lg font-semibold">Ranking Results</h2>
                  <div className="border border-border rounded-lg overflow-hidden">
                    <table className="w-full text-sm">
                      <thead className="bg-muted/50">
                        <tr>
                          <th className="px-4 py-3 text-left font-medium">Rank</th>
                          <th className="px-4 py-3 text-left font-medium">Item</th>
                          <th className="px-4 py-3 text-right font-medium">Score (θ̂)</th>
                          <th className="px-4 py-3 text-right font-medium">95% CI</th>
                        </tr>
                      </thead>
                      <tbody>
                        {[...results.items]
                          .sort((a, b) => a.rank - b.rank)
                          .map((item, index) => (
                            <tr
                              key={item.name}
                              className={cn(
                                "border-t border-border/40",
                                index % 2 === 0 ? "bg-background" : "bg-muted/20"
                              )}
                            >
                              <td className="px-4 py-3 font-mono">#{item.rank}</td>
                              <td className="px-4 py-3 font-medium">{item.name}</td>
                              <td className="px-4 py-3 text-right font-mono">
                                {item.theta_hat.toFixed(4)}
                              </td>
                              <td className="px-4 py-3 text-right font-mono text-muted-foreground">
                                [{item.ci_two_sided[0]}, {item.ci_two_sided[1]}]
                              </td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Ranking Chart Visualization */}
                <div className="space-y-4">
                  <div className="flex items-center gap-2">
                    <BarChart3 className="h-5 w-5 text-primary" />
                    <h2 className="text-lg font-semibold">Rankings Chart</h2>
                  </div>
                  <div className="border border-border rounded-lg p-4 bg-card">
                    <RankingChart items={results.items} className="w-full" />
                  </div>
                </div>

                {/* Footer */}
                <div className="text-center text-muted-foreground text-xs py-6 border-t border-border/40">
                  <p>Generated by OmniRank - Spectral Ranking Analysis Platform</p>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
