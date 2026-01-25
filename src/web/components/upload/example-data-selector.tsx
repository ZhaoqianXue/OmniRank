"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { UsersRound, ListOrdered, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { Card, CardContent } from "@/components/ui/card";
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
    <div className={cn("space-y-3", className)}>
      <div className="flex items-center gap-3">
        <div className="flex-1 h-px bg-border/50" />
        <span className="text-xs text-muted-foreground font-medium uppercase tracking-wider">
          or try an example
        </span>
        <div className="flex-1 h-px bg-border/50" />
      </div>

      <div className="grid grid-cols-2 gap-3">
        {examples.map((example) => {
          const Icon = iconMap[example.id] || ListOrdered;
          const isLoading = loadingId === example.id;
          const isDisabled = disabled || (loadingId !== null && loadingId !== example.id);

          return (
            <motion.div
              key={example.id}
              whileHover={isDisabled ? {} : { scale: 1.02 }}
              whileTap={isDisabled ? {} : { scale: 0.98 }}
            >
              <Card
                className={cn(
                  "cursor-pointer transition-all border-2",
                  isLoading && "border-primary bg-primary/5",
                  isDisabled && !isLoading && "opacity-50 cursor-not-allowed",
                  !isDisabled && !isLoading && "hover:border-primary/50 hover:bg-primary/5"
                )}
                onClick={() => handleSelect(example.id)}
              >
                <CardContent className="p-4">
                  <div className="flex items-start gap-3">
                    <div
                      className={cn(
                        "flex-shrink-0 w-10 h-10 rounded-lg flex items-center justify-center",
                        isLoading ? "bg-primary/20" : "bg-muted"
                      )}
                    >
                      {isLoading ? (
                        <Loader2 className="h-5 w-5 text-primary animate-spin" />
                      ) : (
                        <Icon className="h-5 w-5 text-muted-foreground" />
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <h4 className="text-sm font-medium truncate">{example.title}</h4>
                      <p className="text-xs text-muted-foreground mt-0.5 line-clamp-2">
                        {example.format === "pairwise" ? "Pairwise comparison" : "Pointwise scores"}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
