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
      {/* Header with introduction */}
      <div className="text-center space-y-2">
        <div className="flex items-center gap-3">
          <div className="flex-1 h-px bg-border/50" />
          <span className="text-xs text-muted-foreground font-medium uppercase tracking-wider">
            or try an example
          </span>
          <div className="flex-1 h-px bg-border/50" />
        </div>
        <p className="text-xs text-muted-foreground">
          See how OmniRank transforms comparison data into statistically rigorous rankings
        </p>
      </div>

      {/* Vertical list of examples */}
      <div className="space-y-2">
        {examples.map((example) => {
          const Icon = iconMap[example.id] || ListOrdered;
          const isLoading = loadingId === example.id;
          const isDisabled = disabled || (loadingId !== null && loadingId !== example.id);

          return (
            <motion.div
              key={example.id}
              whileHover={isDisabled ? {} : { x: 4 }}
              whileTap={isDisabled ? {} : { scale: 0.99 }}
            >
              <div
                className={cn(
                  "flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-all bg-background",
                  isLoading && "border-primary bg-background",
                  isDisabled && !isLoading && "opacity-50 cursor-not-allowed",
                  !isDisabled && !isLoading && "hover:border-primary/50 hover:bg-background"
                )}
                onClick={() => handleSelect(example.id)}
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
              </div>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
