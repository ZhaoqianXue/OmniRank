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

const primaryExampleOrder = ["pairwise", "multiway_scores"] as const;
type PrimaryExampleId = (typeof primaryExampleOrder)[number];

const primaryIconMap: Record<PrimaryExampleId, React.ComponentType<{ className?: string }>> = {
  pairwise: ArrowLeftRight,
  multiway_scores: Gauge,
};

const primaryDisplayTitles: Record<PrimaryExampleId, string> = {
  pairwise: "LLM Comparison",
  multiway_scores: "Model Performance Matrix",
};

type ExampleCardDetails = {
  summary: string;
  eachRow: string;
  values: string;
  bestFor: string;
};

const cardDetailsByExampleId: Record<string, ExampleCardDetails> = {
  pairwise: {
    summary: "Head-to-head battles between two models on the same prompt.",
    eachRow: "Model A vs Model B",
    values: "Winner/loser (1/0)",
    bestFor: "Head-to-head tests and preference battles",
  },
  multiway_scores: {
    summary: "Side-by-side performance comparisons across all models on the same sample.",
    eachRow: "One sample with all models",
    values: "Metric from 0 to 1 (higher is better)",
    bestFor: "Sample-level benchmark metrics",
  },
  pairwise_human_logs: {
    summary: "Raw human preference logs where each comparison is recorded as two item/value rows.",
    eachRow: "One assistant outcome in a comparison",
    values: "value = 1 (winner), 0 (loser)",
    bestFor: "Raw annotation exports requiring automatic pivot",
  },
  multiway_latency: {
    summary: "Latency comparisons across all systems on the same workload.",
    eachRow: "One workload with all systems",
    values: "Latency in ms (lower is better)",
    bestFor: "System speed and infrastructure benchmarks",
  },
  multiway_rank_columns: {
    summary: "Complete rank-order outcomes recorded as rank_1 to rank_6.",
    eachRow: "One match with full ordering",
    values: "Rank positions by candidate name",
    bestFor: "Tournament-style full ranking outcomes",
  },
  multiway: {
    summary: "Relative finish-order comparisons across all horses in the same race.",
    eachRow: "One race with all horses",
    values: "Finish positions (1 = best, lower is better)",
    bestFor: "Complete ranking data within each event",
  },
  multiway_phenotype: {
    summary:
      "PRS method scores stratified by phenotype. Side-by-side comparisons across 14 methods on each sample within 32 phenotype groups.",
    eachRow: "One sample within a phenotype (phenotype + sample_id)",
    values: "PRS score 0–1 per method (higher is better)",
    bestFor: "Overall and phenotype-wise PRS ranking; forest plots, violin, heatmap",
  },
};

// Short descriptions for each example
const shortDescriptions: Record<string, string> = {
  pairwise: "AI chatbots competing head-to-head on coding, math, and writing tasks",
  pairwise_human_logs: "Raw pairwise logs auto-converted to ranking-ready matrix",
  multiway_scores: "ML models evaluated side by side on each sample",
  multiway_latency: "System latency benchmarks across realistic workload scenarios",
  multiway_rank_columns: "Tournament outcomes in rank_1 through rank_6 format",
  multiway: "Horses ranked by finish position across multiple races",
  multiway_phenotype: "14 PRS methods scored across 32 phenotypes (~1.7k samples)",
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
