"use client";

import { Fragment } from "react";
import { motion } from "framer-motion";
import { Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

interface ProgressIndicatorProps {
  progress: number; // 0-1
  message?: string;
  className?: string;
}

const STAGES = [
  { key: "schema", label: "Schema" },
  { key: "queue", label: "Queue" },
  { key: "ranking", label: "Ranking" },
  { key: "report", label: "Report" },
] as const;

function resolveStageIndex(progress: number, message?: string): number {
  const safeProgress = Number.isFinite(progress) ? Math.max(0, Math.min(progress, 1)) : 0;
  const lowerMessage = (message || "").toLowerCase();

  if (safeProgress >= 1 || lowerMessage.includes("completed")) return 3;
  if (safeProgress >= 0.9 || lowerMessage.includes("finalizing") || lowerMessage.includes("report")) return 3;
  if (
    safeProgress >= 0.45 ||
    lowerMessage.includes("executing spectral") ||
    lowerMessage.includes("visualization")
  ) {
    return 2;
  }
  if (safeProgress >= 0.12 || lowerMessage.includes("queued") || lowerMessage.includes("submitting")) {
    return 1;
  }
  return 0;
}

export function ProgressIndicator({ progress, message, className }: ProgressIndicatorProps) {
  const stageIndex = resolveStageIndex(progress, message);

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn("space-y-3", className)}
    >
      <div className="flex items-center gap-2 text-sm">
        <Loader2 className="h-4 w-4 animate-spin text-primary" />
        <span className="text-muted-foreground">{message || "Processing..."}</span>
      </div>

      <div className="rounded-md border border-border/40 bg-card/50 p-3">
        <div className="flex items-center">
          {STAGES.map((stage, index) => {
            const isCompleted = index < stageIndex;
            const isActive = index === stageIndex;
            return (
              <Fragment key={stage.key}>
                <div className="flex flex-col items-center gap-1 min-w-[56px]">
                  <span
                    className={cn(
                      "h-2.5 w-2.5 rounded-full border transition-colors",
                      isCompleted && "bg-primary border-primary",
                      isActive && "bg-amber-500 border-amber-500",
                      !isCompleted && !isActive && "bg-background border-border/70",
                    )}
                  />
                  <span
                    className={cn(
                      "text-[10px]",
                      isCompleted && "text-primary",
                      isActive && "text-amber-600 dark:text-amber-400",
                      !isCompleted && !isActive && "text-muted-foreground",
                    )}
                  >
                    {stage.label}
                  </span>
                </div>
                {index < STAGES.length - 1 && (
                  <span
                    className={cn(
                      "h-px flex-1 mx-1 transition-colors",
                      index < stageIndex ? "bg-primary" : "bg-border/50",
                    )}
                  />
                )}
              </Fragment>
            );
          })}
        </div>
      </div>
    </motion.div>
  );
}
