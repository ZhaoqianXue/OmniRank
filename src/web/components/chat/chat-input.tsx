"use client";

import { useState, useRef, useMemo, KeyboardEvent } from "react";
import { Send, Loader2, Zap, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import type { InferredSchema, RankingResults } from "@/lib/api";
import type { AnalysisStatus } from "@/hooks/use-omnirank";

// Analysis stage for quick start questions
type AnalysisStage = "pre-upload" | "post-schema" | "post-analysis";

interface ChatInputProps {
  onSend: (message: string) => Promise<void>;
  disabled?: boolean;
  placeholder?: string;
  className?: string;
  // Context for generating quick start questions
  status?: AnalysisStatus;
  schema?: InferredSchema | null;
  results?: RankingResults | null;
}

/**
 * Generate quick start questions based on the current analysis stage.
 * These questions combine session context with spectral ranking knowledge.
 */
function getQuickStartQuestions(
  stage: AnalysisStage,
  schema?: InferredSchema | null,
  results?: RankingResults | null
): string[] {
  switch (stage) {
    case "pre-upload":
      // Before data upload: General questions about system and methodology
      return [
        "What is spectral ranking and how does it work?",
        "What data formats does OmniRank support?",
        "Can you explain how confidence intervals are calculated?",
      ];

    case "post-schema":
      // After schema inference: Questions about data and configuration
      if (schema) {
        const items = schema.ranking_items?.slice(0, 3).join(", ") || "items";
        return [
          `What does the "${schema.format}" data format mean for my analysis?`,
          schema.indicator_col
            ? `How will the "${schema.indicator_col}" indicator affect the ranking?`
            : "How should I choose the ranking direction?",
          `What can you tell me about comparing ${items}?`,
        ];
      }
      return [
        "How should I configure my analysis settings?",
        "What does 'higher is better' mean for ranking direction?",
        "What happens during the analysis process?",
      ];

    case "post-analysis":
      // After analysis complete: Use LLM-generated questions from results if available
      if (results?.section_questions?.insights?.length) {
        // Use insights questions as they're most relevant for general Q&A
        return results.section_questions.insights.slice(0, 3);
      }

      // Fallback to context-aware questions
      if (results?.items?.length) {
        const topItem = results.items[0]?.name || "the top item";
        const secondItem = results.items[1]?.name || "the second item";
        const metadata = results.metadata;

        return [
          `Is ${topItem} significantly better than ${secondItem}?`,
          "Which rankings can I trust for decision-making?",
          "How should I interpret the confidence intervals?",
        ];
      }
      return [
        "How should I interpret these ranking results?",
        "Which items have statistically significant differences?",
        "What are the limitations of this analysis?",
      ];

    default:
      return [];
  }
}

export function ChatInput({
  onSend,
  disabled = false,
  placeholder = "Type your message...",
  className,
  status,
  schema,
  results,
}: ChatInputProps) {
  const [message, setMessage] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [showQuickStart, setShowQuickStart] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Determine the current analysis stage
  const stage: AnalysisStage = useMemo(() => {
    if (status === "completed" && results) {
      return "post-analysis";
    } else if (schema) {
      return "post-schema";
    }
    return "pre-upload";
  }, [status, schema, results]);

  // Generate context-aware quick start questions
  const quickStartQuestions = useMemo(() => {
    return getQuickStartQuestions(stage, schema, results);
  }, [stage, schema, results]);

  const handleSend = async () => {
    const trimmed = message.trim();
    if (!trimmed || isSending || disabled) return;

    setIsSending(true);
    setShowQuickStart(false);
    try {
      await onSend(trimmed);
      setMessage("");
      // Reset textarea height
      if (textareaRef.current) {
        textareaRef.current.style.height = "auto";
      }
    } finally {
      setIsSending(false);
    }
  };

  const handleQuickQuestion = async (question: string) => {
    if (isSending || disabled) return;

    setShowQuickStart(false);
    setIsSending(true);
    try {
      await onSend(question);
    } finally {
      setIsSending(false);
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // Send on Enter (without Shift)
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
    // Close quick start on Escape
    if (e.key === "Escape") {
      setShowQuickStart(false);
    }
  };

  const handleInput = () => {
    // Auto-resize textarea
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 120)}px`;
    }
  };

  const handleFocus = () => {
    // Show quick start when input is focused and empty
    if (!message.trim()) {
      setShowQuickStart(true);
    }
  };

  const handleBlur = (e: React.FocusEvent) => {
    // Don't hide if clicking on quick start panel
    const relatedTarget = e.relatedTarget as HTMLElement;
    if (relatedTarget?.closest('[data-quick-start-panel]')) {
      return;
    }
    // Delay hiding to allow button clicks
    setTimeout(() => setShowQuickStart(false), 150);
  };

  return (
    <div className={cn("relative", className)}>
      {/* Quick Start Panel */}
      {showQuickStart && quickStartQuestions.length > 0 && (
        <div
          data-quick-start-panel
          className="absolute bottom-full left-0 right-0 mb-2 bg-card border border-border rounded-lg shadow-lg overflow-hidden animate-in slide-in-from-bottom-2 duration-200"
        >
          <div className="flex items-center gap-2 px-3 py-2 border-b border-border/50 bg-muted/30">
            <Zap className="h-3.5 w-3.5 text-primary" />
            <span className="text-xs font-medium text-muted-foreground">Quick Start</span>
          </div>
          <div className="p-1.5">
            {quickStartQuestions.map((question, index) => (
              <button
                key={index}
                onClick={() => handleQuickQuestion(question)}
                disabled={isSending || disabled}
                className="w-full flex items-center gap-2 px-3 py-2 text-left text-sm rounded-md hover:bg-muted/50 transition-colors group disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <ChevronRight className="h-3.5 w-3.5 text-muted-foreground group-hover:text-primary group-hover:translate-x-0.5 transition-all shrink-0" />
                <span className="line-clamp-1">{question}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Input Area */}
      <div className="flex items-end gap-2">
        <Textarea
          ref={textareaRef}
          value={message}
          onChange={(e) => {
            setMessage(e.target.value);
            // Hide quick start when typing
            if (e.target.value.trim()) {
              setShowQuickStart(false);
            }
          }}
          onKeyDown={handleKeyDown}
          onInput={handleInput}
          onFocus={handleFocus}
          onBlur={handleBlur}
          placeholder={placeholder}
          disabled={disabled || isSending}
          className="min-h-[40px] max-h-[120px] resize-none bg-card/50 border-border/50 focus:border-primary"
          rows={1}
        />
        <Button
          onClick={handleSend}
          disabled={!message.trim() || disabled || isSending}
          size="icon"
          className="h-10 w-10 shrink-0"
        >
          {isSending ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Send className="h-4 w-4" />
          )}
        </Button>
      </div>
    </div>
  );
}
