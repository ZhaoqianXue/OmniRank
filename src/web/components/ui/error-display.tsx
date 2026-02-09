"use client";

import { motion } from "framer-motion";
import { AlertCircle, RefreshCw, XCircle, AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface ErrorDisplayProps {
  title?: string;
  message: string;
  type?: "error" | "warning" | "info";
  onRetry?: () => void;
  onDismiss?: () => void;
  className?: string;
}

export function ErrorDisplay({
  title,
  message,
  type = "error",
  onRetry,
  onDismiss,
  className,
}: ErrorDisplayProps) {
  const icons = {
    error: XCircle,
    warning: AlertTriangle,
    info: AlertCircle,
  };

  const colors = {
    error: {
      bg: "bg-destructive/10",
      border: "border-destructive/30",
      icon: "text-destructive",
      title: "text-destructive",
    },
    warning: {
      bg: "bg-yellow-500/10",
      border: "border-yellow-500/30",
      icon: "text-yellow-500",
      title: "text-yellow-600 dark:text-yellow-400",
    },
    info: {
      bg: "bg-primary/10",
      border: "border-primary/30",
      icon: "text-primary",
      title: "text-primary",
    },
  };

  const Icon = icons[type];
  const colorSet = colors[type];

  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className={cn(
        "rounded-lg border p-4",
        colorSet.bg,
        colorSet.border,
        className
      )}
    >
      <div className="flex items-start gap-3">
        <Icon className={cn("h-5 w-5 mt-0.5 flex-shrink-0", colorSet.icon)} />
        <div className="flex-1 min-w-0">
          {title && (
            <h4 className={cn("font-medium mb-1", colorSet.title)}>{title}</h4>
          )}
          <p className="text-sm text-muted-foreground">{message}</p>
          
          {(onRetry || onDismiss) && (
            <div className="flex items-center gap-2 mt-3">
              {onRetry && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={onRetry}
                  className="h-8"
                >
                  <RefreshCw className="h-3 w-3 mr-1" />
                  Retry
                </Button>
              )}
              {onDismiss && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={onDismiss}
                  className="h-8"
                >
                  Dismiss
                </Button>
              )}
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}

// Compact inline error for form fields
interface InlineErrorProps {
  message: string;
  className?: string;
}

export function InlineError({ message, className }: InlineErrorProps) {
  return (
    <motion.p
      initial={{ opacity: 0, height: 0 }}
      animate={{ opacity: 1, height: "auto" }}
      exit={{ opacity: 0, height: 0 }}
      className={cn("text-sm text-destructive flex items-center gap-1 mt-1", className)}
    >
      <AlertCircle className="h-3 w-3" />
      {message}
    </motion.p>
  );
}
