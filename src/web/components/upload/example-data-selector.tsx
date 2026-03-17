"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { UsersRound, Trophy, Loader2, ChevronRight, ArrowLeftRight, Gauge } from "lucide-react";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import type { ExampleDataInfo } from "@/lib/api";

interface ExampleDataSelectorProps {
  examples: ExampleDataInfo[];
  onSelect: (exampleId: string) => Promise<unknown>;
  disabled?: boolean;
  className?: string;
}

const iconMap: Record<string, React.ComponentType<{ className?: string }>> = {
  pairwise: UsersRound,
  pairwise_human_logs: UsersRound,
  multiway_scores: Gauge,
  multiway_latency: Gauge,
  multiway_rank_columns: Trophy,
  multiway: Trophy,
  multiway_phenotype: Gauge,
};

const formatLabels: Record<string, string> = {
  pairwise: "Pairwise",
  multiway: "Multiway",
};

const primaryExampleOrder = ["pairwise", "multiway_phenotype"] as const;
type PrimaryExampleId = (typeof primaryExampleOrder)[number];

const primaryIconMap: Record<PrimaryExampleId, React.ComponentType<{ className?: string }>> = {
  pairwise: ArrowLeftRight,
  multiway_phenotype: Gauge,
};

const primaryDisplayTitles: Record<PrimaryExampleId, string> = {
  pairwise: "LLM Comparison",
  multiway_phenotype: "PRS Phenotype Matrix",
};

type ExampleCardDetails = {
  summary: string;
  eachRow: string;
  values: string;
  bestFor: string;
};

const cardDetailsByExampleId: Record<string, ExampleCardDetails> = {
  pairwise: {
    summary: "Direct win-loss comparisons between two models on the same evaluation task.",
    eachRow: "One comparison between two models",
    values: "Winner/loser outcome (1/0)",
    bestFor: "Pairwise evaluations and preference-based ranking",
  },
  multiway_scores: {
    summary: "Per-sample performance scores for multiple models evaluated on the same input.",
    eachRow: "One sample with scores for all models",
    values: "Metric from 0 to 1 (higher is better)",
    bestFor: "Sample-level benchmark comparisons",
  },
  pairwise_human_logs: {
    summary: "Raw human preference logs with one row per assistant in each comparison.",
    eachRow: "One assistant outcome within a comparison",
    values: "value = 1 (winner), 0 (loser)",
    bestFor: "Long-format annotation logs that need auto-conversion",
  },
  multiway_latency: {
    summary: "Latency measurements for multiple systems under the same workload scenario.",
    eachRow: "One workload with latency for all systems",
    values: "Latency in ms (lower is better)",
    bestFor: "Speed and systems benchmarking",
  },
  multiway_rank_columns: {
    summary: "Full ranking outcomes stored explicitly in rank_1 through rank_6 columns.",
    eachRow: "One event with a complete rank order",
    values: "Rank positions by candidate name",
    bestFor: "Full-order rankings and tournament results",
  },
  multiway: {
    summary: "Finish-order results for all competitors within the same race.",
    eachRow: "One race with all competitors",
    values: "Finish positions (1 = best, lower is better)",
    bestFor: "Event-level ranking data",
  },
  multiway_phenotype: {
    summary: "AUC comparisons for multiple PRS methods across phenotype cohorts.",
    eachRow: "One phenotype cohort with all methods",
    values: "AUC from 0 to 1 (higher is better)",
    bestFor: "Method comparison, heatmaps, and phenotype-wise ranking",
  },
};

// Short descriptions for each example
const shortDescriptions: Record<string, string> = {
  pairwise: "Direct win-loss comparisons between two models on the same task",
  pairwise_human_logs: "Raw human preference logs with one row per assistant outcome",
  multiway_scores: "Per-sample performance scores for multiple models",
  multiway_latency: "Latency measurements for multiple systems under shared workloads",
  multiway_rank_columns: "Full ranking outcomes stored in rank_1 through rank_6 columns",
  multiway: "Finish-order results for all competitors within each race",
  multiway_phenotype: "AUC comparisons for PRS methods across phenotype records",
};

export function ExampleDataSelector({
  examples,
  onSelect,
  disabled = false,
  className,
}: ExampleDataSelectorProps) {
  const [loadingId, setLoadingId] = useState<string | null>(null);
  const [showMoreExamples, setShowMoreExamples] = useState(false);

  const handleSelect = async (exampleId: string) => {
    if (disabled || loadingId) return;

    setLoadingId(exampleId);
    try {
      await onSelect(exampleId);
    } finally {
      setLoadingId(null);
    }
  };

  const primaryExamples = primaryExampleOrder
    .map((id) => examples.find((example) => example.id === id))
    .filter((example): example is ExampleDataInfo => Boolean(example));
  const secondaryExamples = examples.filter(
    (example) => !primaryExampleOrder.includes(example.id as PrimaryExampleId)
  );

  return (
    <div className={cn("space-y-4", className)}>
      {/* Header with lightweight hierarchy */}
      <div className="space-y-1.5">
        <div className="flex items-center gap-3">
          <div className="h-px flex-1 bg-border/60" />
          <p className="text-sm sm:text-base font-semibold text-foreground whitespace-nowrap">
            Use Example Data
          </p>
          <div className="h-px flex-1 bg-border/60" />
        </div>
        <p className="text-xs sm:text-sm text-muted-foreground text-center leading-relaxed">
          Preview expected input formats before uploading your own dataset.
        </p>
      </div>

      {/* Primary examples */}
      <div className="grid grid-cols-2 gap-2.5 sm:gap-3 auto-rows-fr">
        {primaryExamples.map((example) => {
          const details = cardDetailsByExampleId[example.id];
          const Icon = primaryIconMap[example.id as PrimaryExampleId] || UsersRound;
          const isLoading = loadingId === example.id;
          const isButtonDisabled = disabled || loadingId !== null;

          return (
            <motion.div
              key={example.id}
              whileHover={isButtonDisabled ? {} : { y: -2 }}
              whileTap={isButtonDisabled ? {} : { scale: 0.99 }}
              className="h-full"
            >
              <button
                type="button"
                className={cn(
                  "h-full w-full text-left rounded-xl border p-3 sm:p-4 transition-all bg-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/60 focus-visible:ring-offset-2 focus-visible:ring-offset-background",
                  isLoading && "border-primary bg-background",
                  isButtonDisabled && !isLoading && "opacity-50 cursor-not-allowed",
                  !isButtonDisabled && !isLoading && "hover:border-primary/50 hover:bg-background"
                )}
                onClick={() => handleSelect(example.id)}
                disabled={isButtonDisabled}
                aria-busy={isLoading}
                aria-label={`Load ${example.title} example dataset`}
              >
                <div className="flex flex-col gap-2.5">
                  <div className="flex items-start justify-between gap-2">
                    <div
                      className={cn(
                        "w-9 h-9 rounded-lg flex items-center justify-center",
                        isLoading ? "bg-primary/20 text-primary" : "bg-muted"
                      )}
                    >
                      {isLoading ? (
                        <Loader2 className="h-4 w-4 text-primary animate-spin" />
                      ) : (
                        <Icon className="h-4 w-4 text-muted-foreground" />
                      )}
                    </div>
                    <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                      {formatLabels[example.format] || example.format}
                    </Badge>
                  </div>

                  <div className="space-y-1.5">
                    <h4 className="text-sm sm:text-base font-semibold leading-tight line-clamp-2 text-foreground">
                      {primaryDisplayTitles[example.id as PrimaryExampleId] || example.title}
                    </h4>
                    <p className="text-xs sm:text-sm text-muted-foreground leading-snug">
                      {details?.summary || shortDescriptions[example.id] || ""}
                    </p>
                  </div>

                  <div className="grid gap-1.5">
                    <div className="rounded-md border border-border/60 bg-muted/20 px-2 py-1.5">
                      <p className="text-[11px] font-medium text-muted-foreground">
                        Each Row
                      </p>
                      <p className="text-[11px] sm:text-xs text-foreground leading-tight">
                        {details?.eachRow || "Comparison record"}
                      </p>
                    </div>
                    <div className="rounded-md border border-border/60 bg-muted/20 px-2 py-1.5">
                      <p className="text-[11px] font-medium text-muted-foreground">
                        Values
                      </p>
                      <p className="text-[11px] sm:text-xs text-foreground leading-tight">
                        {details?.values || "See dataset"}
                      </p>
                    </div>
                    <div className="rounded-md border border-border/60 bg-muted/20 px-2 py-1.5">
                      <p className="text-[11px] font-medium text-muted-foreground">
                        Best For
                      </p>
                      <p className="text-[11px] sm:text-xs text-foreground leading-tight">
                        {details?.bestFor || "General ranking"}
                      </p>
                    </div>
                  </div>
                </div>
              </button>
            </motion.div>
          );
        })}
      </div>

      {/* Secondary examples */}
      {secondaryExamples.length > 0 && (
        <div className="space-y-2">
          <button
            type="button"
            className={cn(
              "w-full flex items-center justify-between rounded-lg border border-dashed border-border/70 px-3 py-2 text-left transition-colors",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/50 focus-visible:ring-offset-2 focus-visible:ring-offset-background",
              !disabled && "hover:border-primary/40 hover:bg-primary/[0.03]"
            )}
            onClick={() => setShowMoreExamples((prev) => !prev)}
            aria-expanded={showMoreExamples}
            aria-controls="more-example-data"
          >
            <span className="text-sm font-medium text-foreground">More Example Data</span>
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <span>{secondaryExamples.length} dataset{secondaryExamples.length > 1 ? "s" : ""}</span>
              <ChevronRight
                className={cn("h-4 w-4 transition-transform", showMoreExamples && "rotate-90")}
              />
            </div>
          </button>

          {showMoreExamples && (
            <div
              id="more-example-data"
              className="grid grid-cols-2 gap-2.5 sm:gap-3 auto-rows-fr"
            >
              {secondaryExamples.map((example) => {
                const details = cardDetailsByExampleId[example.id];
                const Icon = iconMap[example.id] || Trophy;
                const isLoading = loadingId === example.id;
                const isButtonDisabled = disabled || loadingId !== null;

                return (
                  <motion.div
                    key={example.id}
                    whileHover={isButtonDisabled ? {} : { y: -2 }}
                    whileTap={isButtonDisabled ? {} : { scale: 0.99 }}
                    className="h-full"
                  >
                    <button
                      type="button"
                      className={cn(
                        "h-full w-full text-left rounded-xl border p-3 sm:p-4 transition-all bg-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/60 focus-visible:ring-offset-2 focus-visible:ring-offset-background",
                        isLoading && "border-primary bg-background",
                        isButtonDisabled && !isLoading && "opacity-50 cursor-not-allowed",
                        !isButtonDisabled && !isLoading && "hover:border-primary/50 hover:bg-background"
                      )}
                      onClick={() => handleSelect(example.id)}
                      disabled={isButtonDisabled}
                      aria-busy={isLoading}
                      aria-label={`Load ${example.title} example dataset`}
                    >
                      <div className="flex flex-col gap-2.5">
                        <div className="flex items-start justify-between gap-2">
                          <div
                            className={cn(
                              "w-9 h-9 rounded-lg flex items-center justify-center",
                              isLoading ? "bg-primary/20 text-primary" : "bg-muted"
                            )}
                          >
                            {isLoading ? (
                              <Loader2 className="h-4 w-4 text-primary animate-spin" />
                            ) : (
                              <Icon className="h-4 w-4 text-muted-foreground" />
                            )}
                          </div>
                          <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                            {formatLabels[example.format] || example.format}
                          </Badge>
                        </div>

                        <div className="space-y-1.5">
                          <h4 className="text-sm sm:text-base font-semibold leading-tight line-clamp-2 text-foreground">
                            {example.title}
                          </h4>
                          <p className="text-xs sm:text-sm text-muted-foreground leading-snug">
                            {details?.summary || shortDescriptions[example.id] || ""}
                          </p>
                        </div>

                        <div className="grid gap-1.5">
                          <div className="rounded-md border border-border/60 bg-muted/20 px-2 py-1.5">
                            <p className="text-[11px] font-medium text-muted-foreground">
                              Each Row
                            </p>
                            <p className="text-[11px] sm:text-xs text-foreground leading-tight">
                              {details?.eachRow || "Comparison record"}
                            </p>
                          </div>
                          <div className="rounded-md border border-border/60 bg-muted/20 px-2 py-1.5">
                            <p className="text-[11px] font-medium text-muted-foreground">
                              Values
                            </p>
                            <p className="text-[11px] sm:text-xs text-foreground leading-tight">
                              {details?.values || "See dataset"}
                            </p>
                          </div>
                          <div className="rounded-md border border-border/60 bg-muted/20 px-2 py-1.5">
                            <p className="text-[11px] font-medium text-muted-foreground">
                              Best For
                            </p>
                            <p className="text-[11px] sm:text-xs text-foreground leading-tight">
                              {details?.bestFor || "General ranking"}
                            </p>
                          </div>
                        </div>
                      </div>
                    </button>
                  </motion.div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
