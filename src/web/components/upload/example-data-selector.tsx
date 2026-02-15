"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { UsersRound, ListOrdered, Trophy, Loader2, ChevronRight } from "lucide-react";
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

  const handleSelect = async (exampleId: string) => {
    if (disabled || loadingId) return;

    setLoadingId(exampleId);
    try {
      await onSelect(exampleId);
    } finally {
      setLoadingId(null);
    }
  };

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

      {/* Vertical list of examples */}
      <div className="space-y-2.5">
        {examples.map((example) => {
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
                {/* Icon */}
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

                {/* Content */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <h4 className="text-sm font-medium">{example.title}</h4>
                    <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                      {formatLabels[example.format] || example.format}
                    </Badge>
                  </div>
                  <p className="text-xs text-muted-foreground mt-0.5 line-clamp-1">
                    {shortDescriptions[example.id] || ""}
                  </p>
                </div>

                {/* Arrow */}
                <ChevronRight className="h-4 w-4 text-muted-foreground flex-shrink-0" />
              </button>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
