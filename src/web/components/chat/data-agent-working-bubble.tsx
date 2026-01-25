"use client";
import { Bot, Check, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

interface DataAgentWorkingBubbleProps {
  isComplete?: boolean;
  className?: string;
}

// Steps that the Data Agent performs
const DATA_AGENT_STEPS = [
  "Analyzing your dataset structure...",
  "Detecting data format...",
  "Recognizing ranking items...",
  "Determining ranking direction...",
  "Validating data quality...",
];

export function DataAgentWorkingBubble({
  isComplete = false,
  className,
}: DataAgentWorkingBubbleProps) {
  const completedSteps = isComplete ? DATA_AGENT_STEPS.map((_, i) => i) : [];

  return (
    <div className={cn(
      "bg-white dark:bg-zinc-800 border border-border/50 rounded-2xl rounded-bl-sm shadow-sm max-w-sm w-full px-4 py-3",
      className
    )}>
      {/* Header */}
      <div className="flex items-center gap-2 mb-3">
        <div className="w-6 h-6 rounded-full bg-primary/10 flex items-center justify-center">
          <Bot className="h-3.5 w-3.5 text-primary" />
        </div>
        <span className="text-sm font-semibold">OmniRank&apos;s Data Agent is Working</span>
      </div>

      {/* Steps */}
      <div className="space-y-2">
        {DATA_AGENT_STEPS.map((step, index) => {
          const isCompleted = completedSteps.includes(index);

          return (
            <div key={index} className="flex items-center gap-2">
              {isCompleted ? (
                <Check className="h-4 w-4 text-green-500 flex-shrink-0" />
              ) : (
                <Loader2 className="h-4 w-4 text-primary animate-spin flex-shrink-0" />
              )}
              <span className={cn(
                "text-sm",
                isCompleted ? "text-foreground" : "text-muted-foreground"
              )}>
                {step}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
