"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { UsersRound, Trophy, Loader2, ArrowLeftRight, Gauge } from "lucide-react";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import type { ExampleDataInfo } from "@/lib/api";

interface ExampleDataSelectorProps {
  examples: ExampleDataInfo[];
  onSelect: (exampleId: string) => Promise<unknown>;
  disabled?: boolean;
  className?: string;
}

const formatLabels: Record<string, string> = {
  pairwise: "Pairwise",
  multiway: "Multiway",
};

const primaryExampleOrder = ["pairwise", "multiway_phenotype", "multiway_f1"] as const;
type PrimaryExampleId = (typeof primaryExampleOrder)[number];

const primaryIconMap: Record<PrimaryExampleId, React.ComponentType<{ className?: string }>> = {
  pairwise: ArrowLeftRight,
  multiway_phenotype: Gauge,
  multiway_f1: Trophy,
};

const primaryDisplayTitles: Record<PrimaryExampleId, string> = {
  pairwise: "AI Chatbot Rankings",
  multiway_phenotype: "PRS Method Comparison",
  multiway_f1: "Formula 1 Driver Rankings",
};

type ExampleCardDetails = {
  question: string;
  data: string;
};

const cardDetailsByExampleId: Record<string, ExampleCardDetails> = {
  pairwise: {
    question: "Which AI assistant wins more often in head-to-head comparisons?",
    data: "Chatbot Arena battles",
  },
  multiway_scores: {
    question: "Which model performs best across shared benchmark samples?",
    data: "Model performance scores",
  },
  multiway_f1: {
    question: "Which drivers perform best across Grand Prix races?",
    data: "2025 race results",
  },
  pairwise_human_logs: {
    question: "Which assistant wins more often in human preference logs?",
    data: "Human preference annotations",
  },
  multiway_latency: {
    question: "Which system is fastest across shared workloads?",
    data: "Latency measurements",
  },
  multiway_rank_columns: {
    question: "Which candidate ranks highest across full-order results?",
    data: "Tournament rank columns",
  },
  multiway: {
    question: "Which competitor performs best across races?",
    data: "Race finish positions",
  },
  multiway_phenotype: {
    question: "Which genetic risk score method performs best across phenotypes?",
    data: "Phenotype AUC scores",
  },
};

export function ExampleDataSelector({
  examples,
  onSelect,
  disabled = false,
  className,
}: ExampleDataSelectorProps) {
  const [loadingId, setLoadingId] = useState<string | null>(null);

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
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5 sm:gap-3 auto-rows-fr">
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
                      {details?.question || ""}
                    </p>
                  </div>

                  <div className="rounded-md border border-border/60 bg-muted/20 px-2 py-1.5">
                    <p className="text-[11px] sm:text-xs text-foreground leading-tight">
                      <span className="font-medium text-muted-foreground">Data:</span>{" "}
                      {details?.data || example.filename}
                    </p>
                  </div>
                </div>
              </button>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
