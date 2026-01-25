"use client";

import { motion } from "framer-motion";
import { Loader2 } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";

interface ProgressIndicatorProps {
  progress: number; // 0-1
  message?: string;
  className?: string;
}

export function ProgressIndicator({ progress, message, className }: ProgressIndicatorProps) {
  const percentage = Math.round(progress * 100);

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn("space-y-2", className)}
    >
      <div className="flex items-center justify-between text-sm">
        <div className="flex items-center gap-2">
          <Loader2 className="h-4 w-4 animate-spin text-primary" />
          <span className="text-muted-foreground">{message || "Processing..."}</span>
        </div>
        <span className="font-mono text-primary">{percentage}%</span>
      </div>
      <Progress value={percentage} className="h-2" />
    </motion.div>
  );
}
