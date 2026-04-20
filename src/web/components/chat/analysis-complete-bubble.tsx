"use client";

import { memo } from "react";
import { motion } from "framer-motion";
import { CheckCircle, MessageCircleQuestion, Play } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

type RankingMode = "flash" | "deep";

interface AnalysisCompleteBubbleProps {
  suggestedQuestions: string[];
  onAskQuestion?: (question: string) => void;
  alternativeMode?: RankingMode | null;
  onRunAlternative?: () => void;
  isRerunDisabled?: boolean;
  className?: string;
}

/**
 * Special bubble shown when analysis completes successfully.
 * Displays a success message and suggested questions for the user to explore.
 */
export const AnalysisCompleteBubble = memo(function AnalysisCompleteBubble({
  suggestedQuestions,
  onAskQuestion,
  alternativeMode,
  onRunAlternative,
  isRerunDisabled = false,
  className,
}: AnalysisCompleteBubbleProps) {
  const alternativeLabel =
    alternativeMode === "flash"
      ? "Also run Overall Ranking"
      : alternativeMode === "deep"
        ? "Also run Stratified Ranking"
        : null;

  return (
    <div
      className={cn(
        "relative min-w-0 w-full max-w-sm px-4 py-4 rounded-2xl shadow-sm border text-sm bg-background border-primary/30 rounded-bl-sm",
        className
      )}
    >
      {/* Success header */}
      <div className="flex items-center gap-2 mb-3">
        <div className="p-1.5 rounded-full bg-primary/20">
          <CheckCircle className="h-4 w-4 text-primary" />
        </div>
        <span className="font-semibold text-primary">
          Analysis Complete!
        </span>
      </div>

      {/* Description */}
      <p className="text-muted-foreground mb-4">
        Your ranking has finished successfully. You can now ask me questions about your analysis results!
      </p>

      {/* Run alternative mode */}
      {alternativeLabel && onRunAlternative && (
        <motion.div
          initial={{ opacity: 0, y: -4 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-4"
        >
          <Button
            variant="outline"
            size="sm"
            onClick={onRunAlternative}
            disabled={isRerunDisabled}
            className="w-full justify-center h-auto py-2 px-3 text-xs font-medium bg-background hover:bg-primary/10 hover:text-foreground border-primary/30 hover:border-primary/50"
          >
            <Play className="h-3.5 w-3.5 mr-2 shrink-0" />
            <span className="min-w-0 break-words whitespace-normal leading-relaxed">
              {alternativeLabel}
            </span>
          </Button>
        </motion.div>
      )}

      {/* Suggested questions */}
      {suggestedQuestions.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <MessageCircleQuestion className="h-3.5 w-3.5" />
            <span>Try one of these questions:</span>
          </div>
          <div className="flex flex-col gap-2">
            {suggestedQuestions.map((question, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.1 * (index + 1) }}
              >
                <Button
                  variant="outline"
                  size="sm"
                  className="w-full min-w-0 items-start justify-start h-auto py-2 px-3 text-left text-xs font-normal whitespace-normal bg-background hover:bg-primary/10 hover:text-foreground border-primary/25 hover:border-primary/40"
                  onClick={() => onAskQuestion?.(question)}
                >
                  <span className="text-primary mr-2 mt-0.5 shrink-0">→</span>
                  <span className="min-w-0 break-words whitespace-normal leading-relaxed">
                    {question}
                  </span>
                </Button>
              </motion.div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
});
