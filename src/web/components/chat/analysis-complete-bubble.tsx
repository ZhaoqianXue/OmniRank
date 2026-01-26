"use client";

import { memo } from "react";
import { motion } from "framer-motion";
import { CheckCircle, MessageCircleQuestion } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface AnalysisCompleteBubbleProps {
  suggestedQuestions: string[];
  onAskQuestion?: (question: string) => void;
  className?: string;
}

/**
 * Special bubble shown when analysis completes successfully.
 * Displays a success message and suggested questions for the user to explore.
 */
export const AnalysisCompleteBubble = memo(function AnalysisCompleteBubble({
  suggestedQuestions,
  onAskQuestion,
  className,
}: AnalysisCompleteBubbleProps) {
  return (
    <div
      className={cn(
        "relative max-w-[90%] px-4 py-4 rounded-2xl shadow-sm border text-sm bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-950/30 dark:to-emerald-950/30 border-green-200/50 dark:border-green-800/50 rounded-bl-sm",
        className
      )}
    >
      {/* Success header */}
      <div className="flex items-center gap-2 mb-3">
        <div className="p-1.5 rounded-full bg-green-100 dark:bg-green-900/50">
          <CheckCircle className="h-4 w-4 text-green-600 dark:text-green-400" />
        </div>
        <span className="font-semibold text-green-700 dark:text-green-300">
          Analysis Complete!
        </span>
      </div>
      
      {/* Description */}
      <p className="text-muted-foreground mb-4">
        Your ranking has finished successfully. You can now ask me questions about your analysis results!
      </p>
      
      {/* Suggested questions */}
      {suggestedQuestions.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <MessageCircleQuestion className="h-3.5 w-3.5" />
            <span>Try one of these:</span>
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
                  className="w-full justify-start h-auto py-2 px-3 text-left text-xs font-normal bg-white dark:bg-zinc-900 hover:bg-green-50 dark:hover:bg-green-950/50 border-green-200/50 dark:border-green-800/30 hover:border-green-300 dark:hover:border-green-700"
                  onClick={() => onAskQuestion?.(question)}
                >
                  <span className="text-primary mr-2">→</span>
                  {question}
                </Button>
              </motion.div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
});
