"use client";

import { useState, useEffect } from "react";
import { Bot, Check, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

interface DataAgentWorkingBubbleProps {
  isComplete?: boolean;
  className?: string;
}

// Steps that the Data Agent performs
const DATA_AGENT_STEPS = [
  "Verifying data format...",
  "Analyzing your dataset structure...",
  "Recognizing ranking items...",
  "Determining ranking direction...",
  "Estimating analysis time...",
];

export function DataAgentWorkingBubble({
  isComplete = false,
  className,
}: DataAgentWorkingBubbleProps) {
  const [completedSteps, setCompletedSteps] = useState<number[]>([]);
  const [currentStep, setCurrentStep] = useState(0);

  // Simulate step completion animation
  useEffect(() => {
    if (isComplete) {
      // If complete, show all steps as done
      setCompletedSteps(DATA_AGENT_STEPS.map((_, i) => i));
      setCurrentStep(DATA_AGENT_STEPS.length);
      return;
    }

    // Animate through steps
    const interval = setInterval(() => {
      setCurrentStep((prev) => {
        if (prev < DATA_AGENT_STEPS.length) {
          setCompletedSteps((completed) => [...completed, prev]);
          return prev + 1;
        }
        clearInterval(interval);
        return prev;
      });
    }, 600);

    return () => clearInterval(interval);
  }, [isComplete]);

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
          const isCurrent = currentStep === index && !isComplete;

          return (
            <div key={index} className="flex items-center gap-2">
              {isCompleted ? (
                <Check className="h-4 w-4 text-green-500 flex-shrink-0" />
              ) : isCurrent ? (
                <Loader2 className="h-4 w-4 text-primary animate-spin flex-shrink-0" />
              ) : (
                <div className="h-4 w-4 flex-shrink-0" />
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
