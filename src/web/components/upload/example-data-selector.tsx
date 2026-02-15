"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { UsersRound, ListOrdered, Trophy, Loader2, ChevronRight, ArrowLeftRight, Gauge } from "lucide-react";
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
  pointwise: ListOrdered,
  multiway: Trophy,
};

const formatLabels: Record<string, string> = {
  pairwise: "Pairwise",
  pointwise: "Pointwise",
  multiway: "Multiway",
};

const primaryExampleOrder = ["pairwise", "pointwise"] as const;
type PrimaryExampleId = (typeof primaryExampleOrder)[number];

const primaryIconMap: Record<PrimaryExampleId, React.ComponentType<{ className?: string }>> = {
  pairwise: ArrowLeftRight,
  pointwise: Gauge,
};

const primaryDisplayTitles: Record<PrimaryExampleId, string> = {
  pairwise: "Pairwise: LLM Comparison",
  pointwise: "Pointwise: Success Rate by Question",
};

const primaryCardDetails: Record<
  PrimaryExampleId,
  {
    summary: string;
    eachRow: string;
    values: string;
    bestFor: string;
  }
> = {
  pairwise: {
    summary: "Head-to-head battles between two models on the same prompt.",
    eachRow: "Model A vs Model B",
    values: "Winner/loser (1/0)",
    bestFor: "Head-to-head tests and preference battles",
  },
  pointwise: {
    summary:
      "Each row is one test question. Every model has a success-rate style score, and OmniRank combines all rows into final ranks.",
    eachRow: "One question with all models side by side",
    values: "Score from 0 to 1 (1 = better result)",
    bestFor: "Benchmark tables with question-level outcomes",
  },
};

// Short descriptions for each example
const shortDescriptions: Record<string, string> = {
  pairwise: "AI chatbots competing head-to-head on coding, math, and writing tasks",
  pointwise: "ML models evaluated with accuracy scores across test samples",
  multiway: "Horses ranked by finish position across multiple races",
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
      <div className="grid grid-cols-2 gap-2.5 sm:gap-3">
        {primaryExamples.map((example) => {
          const details = primaryCardDetails[example.id as PrimaryExampleId];
          const Icon = primaryIconMap[example.id as PrimaryExampleId] || UsersRound;
          const isLoading = loadingId === example.id;
          const isButtonDisabled = disabled || loadingId !== null;

          return (
            <motion.div
              key={example.id}
              whileHover={isButtonDisabled ? {} : { y: -2 }}
              whileTap={isButtonDisabled ? {} : { scale: 0.99 }}
            >
              <button
                type="button"
                className={cn(
                  "w-full text-left rounded-xl border p-3 sm:p-4 transition-all bg-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/60 focus-visible:ring-offset-2 focus-visible:ring-offset-background",
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
                    <p className="text-xs sm:text-sm text-muted-foreground leading-snug line-clamp-2">
                      {details?.summary || shortDescriptions[example.id] || ""}
                    </p>
                  </div>

                  <div className="grid gap-1.5">
                    <div className="rounded-md border border-border/60 bg-muted/20 px-2 py-1.5">
                      <p className="text-[11px] font-medium text-muted-foreground">
                        Each Row
                      </p>
                      <p className="text-xs sm:text-sm text-foreground leading-tight line-clamp-1">
                        {details?.eachRow || "Comparison record"}
                      </p>
                    </div>
                    <div className="rounded-md border border-border/60 bg-muted/20 px-2 py-1.5">
                      <p className="text-[11px] font-medium text-muted-foreground">
                        Values
                      </p>
                      <p className="text-xs sm:text-sm text-foreground leading-tight line-clamp-1">
                        {details?.values || "See dataset"}
                      </p>
                    </div>
                    <div className="rounded-md border border-border/60 bg-muted/20 px-2 py-1.5">
                      <p className="text-[11px] font-medium text-muted-foreground">
                        Best For
                      </p>
                      <p className="text-xs sm:text-sm text-foreground leading-tight line-clamp-1">
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
            <div id="more-example-data" className="space-y-2.5">
              {secondaryExamples.map((example) => {
                const Icon = iconMap[example.id] || ListOrdered;
                const isLoading = loadingId === example.id;
                const isButtonDisabled = disabled || loadingId !== null;

                return (
                  <motion.div
                    key={example.id}
                    whileHover={isButtonDisabled ? {} : { x: 4 }}
                    whileTap={isButtonDisabled ? {} : { scale: 0.99 }}
                  >
                    <button
                      type="button"
                      className={cn(
                        "w-full text-left flex items-center gap-3 p-3.5 rounded-lg border transition-all bg-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/60 focus-visible:ring-offset-2 focus-visible:ring-offset-background",
                        isLoading && "border-primary bg-background",
                        isButtonDisabled && !isLoading && "opacity-50 cursor-not-allowed",
                        !isButtonDisabled && !isLoading && "hover:border-primary/50 hover:bg-background"
                      )}
                      onClick={() => handleSelect(example.id)}
                      disabled={isButtonDisabled}
                      aria-busy={isLoading}
                      aria-label={`Load ${example.title} example dataset`}
                    >
                      <div
                        className={cn(
                          "flex-shrink-0 w-9 h-9 rounded-lg flex items-center justify-center",
                          isLoading ? "bg-primary/20" : "bg-muted"
                        )}
                      >
                        {isLoading ? (
                          <Loader2 className="h-4 w-4 text-primary animate-spin" />
                        ) : (
                          <Icon className="h-4 w-4 text-muted-foreground" />
                        )}
                      </div>

                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <h4 className="text-sm font-medium">{example.title}</h4>
                          <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                            {formatLabels[example.format] || example.format}
                          </Badge>
                        </div>
                        <p className="text-xs text-muted-foreground mt-0.5 line-clamp-2">
                          {shortDescriptions[example.id] || ""}
                        </p>
                      </div>

                      <ChevronRight className="h-4 w-4 text-muted-foreground flex-shrink-0" />
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
