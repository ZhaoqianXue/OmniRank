"use client";
import { Bot, Check, Circle, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { DATA_AGENT_STEPS, DATA_AGENT_TOTAL_STEPS } from "@/lib/data-agent-steps";

interface DataAgentWorkingBubbleProps {
  completedSteps?: number;
  totalSteps?: number;
  className?: string;
}

export function DataAgentWorkingBubble({
  completedSteps = 0,
  totalSteps = DATA_AGENT_TOTAL_STEPS,
  className,
}: DataAgentWorkingBubbleProps) {
  const cappedTotalSteps = Math.max(1, Math.min(totalSteps, DATA_AGENT_TOTAL_STEPS));
  const safeCompletedSteps = Math.max(0, Math.min(completedSteps, cappedTotalSteps));

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
        <span className="text-sm font-semibold">OmniRank Data Agent Processing</span>
      </div>

      {/* Steps */}
      <div className="space-y-2">
        {DATA_AGENT_STEPS.map((step, index) => {
          const isCompleted = index < safeCompletedSteps;
          const isActive = index === safeCompletedSteps && safeCompletedSteps < cappedTotalSteps;

          return (
            <div key={index} className="flex items-center gap-2">
              {isCompleted ? (
                <Check className="h-4 w-4 text-green-500 flex-shrink-0" />
              ) : isActive ? (
                <Loader2 className="h-4 w-4 text-primary animate-spin flex-shrink-0" />
              ) : (
                <Circle className="h-4 w-4 text-muted-foreground/50 flex-shrink-0" />
              )}
              <span className={cn(
                "text-sm",
                isCompleted || isActive ? "text-foreground" : "text-muted-foreground"
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
