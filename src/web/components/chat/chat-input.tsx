"use client";

import { useMemo, useRef, useState, KeyboardEvent } from "react";
import { Send, Loader2, Zap, ChevronRight, MessageSquareQuote, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import type { QuotePayload, RankingResults, SemanticSchema } from "@/lib/api";
import type { AnalysisStatus } from "@/hooks/use-omnirank";

// Analysis stage for suggest-question generation fallback
type AnalysisStage = "pre-upload" | "post-schema" | "post-analysis";
type SuggestIntent = "comparison" | "uncertainty" | "method" | "data-prep" | "error-fix" | "general";

interface SuggestMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

interface ChatInputProps {
  onSend: (message: string, quotes?: QuotePayload[]) => Promise<void>;
  disabled?: boolean;
  placeholder?: string;
  className?: string;
  quoteDrafts?: QuotePayload[];
  onQuoteDraftsChange?: (quotes: QuotePayload[]) => void;
  // Local context for generating hardcoded suggest questions
  recentMessages?: SuggestMessage[];
  status?: AnalysisStatus;
  schema?: SemanticSchema | null;
  results?: RankingResults | null;
}

function normalizeQuestion(question: string): string {
  const cleaned = question.trim().replace(/\s+/g, " ");
  if (!cleaned) return "";

  let normalized = cleaned.replace(/(?:\.{3}|…)\s*$/u, "").replace(/[.!。！]+$/u, "");
  if (!normalized) return "";
  return normalized.endsWith("?") || normalized.endsWith("？") ? normalized : `${normalized}?`;
}

function pickUniqueQuestions(candidates: string[], limit = 2): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const candidate of candidates) {
    const normalized = normalizeQuestion(candidate);
    if (!normalized) continue;
    const key = normalized.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(normalized);
    if (out.length >= limit) break;
  }
  return out;
}

/**
 * Hardcoded, instant suggestion strategy (no LLM, no network).
 */
function inferIntent(raw: string): SuggestIntent {
  const text = raw.toLowerCase();
  if (!text.trim()) return "general";
  if (text.includes("compare") || text.includes("vs") || text.includes("better than") || text.includes("对比")) {
    return "comparison";
  }
  if (text.includes("ci") || text.includes("confidence") || text.includes("uncert") || text.includes("置信区间")) {
    return "uncertainty";
  }
  if (text.includes("method") || text.includes("bootstrap") || text.includes("spectral") || text.includes("原理")) {
    return "method";
  }
  if (text.includes("schema") || text.includes("format") || text.includes("upload") || text.includes("列")) {
    return "data-prep";
  }
  if (text.includes("error") || text.includes("failed") || text.includes("invalid") || text.includes("报错")) {
    return "error-fix";
  }
  return "general";
}

function draftDrivenQuestion(
  draft: string,
  intent: SuggestIntent,
  topItem: string,
  secondItem: string
): string {
  const trimmed = draft.trim();
  if (!trimmed) return "";
  if (trimmed.includes("?") || trimmed.includes("？")) return trimmed;

  switch (intent) {
    case "comparison":
      return `Can you evaluate "${trimmed}" with rank evidence and integer CI bounds`;
    case "uncertainty":
      return `What uncertainty should I account for regarding "${trimmed}"`;
    case "method":
      return `Can you explain "${trimmed}" and cite the key spectral ranking paper`;
    case "data-prep":
      return `For "${trimmed}", what data/schema setup should I confirm first`;
    case "error-fix":
      return `For "${trimmed}", what is the fastest concrete fix`;
    case "general":
      if (topItem && secondItem) {
        return `Can you connect "${trimmed}" to whether ${topItem} is truly above ${secondItem}`;
      }
      return `Can you help me make "${trimmed}" decision-ready`;
    default:
      return trimmed;
  }
}

function getHardcodedSuggestQuestions(
  stage: AnalysisStage,
  quoteDrafts: QuotePayload[],
  status: AnalysisStatus | undefined,
  draft: string,
  recentMessages: SuggestMessage[],
  schema?: SemanticSchema | null,
  results?: RankingResults | null
): string[] {
  const trimmedDraft = draft.trim();
  const intent = inferIntent(trimmedDraft);
  const topItem = results?.items?.[0]?.name || "";
  const secondItem = results?.items?.[1]?.name || "";
  const lastUserMessage = [...recentMessages]
    .reverse()
    .find((message) => message.role === "user" && message.content.trim().length > 0)?.content;
  const draftQuestion = draftDrivenQuestion(trimmedDraft, intent, topItem, secondItem);
  const candidates: string[] = [];

  if (draftQuestion) {
    candidates.push(draftQuestion);
  }

  if (quoteDrafts.length > 0) {
    const quoted = quoteDrafts[0]?.quoted_text?.trim() || "the quoted content";
    const preview = quoted.length > 68 ? quoted.slice(0, 68).trimEnd() : quoted;
    candidates.push(`What decision implication should I draw from this quote: "${preview}"`);
    if (quoteDrafts.length > 1) {
      candidates.push("How do my quoted sections align or conflict in their conclusions");
    } else if (results?.items?.length) {
      candidates.push("Does this quoted claim match the rank and integer CI evidence");
    } else {
      candidates.push("What uncertainty caveat should I attach when citing this quote");
    }
    return pickUniqueQuestions(candidates, 2);
  }

  if (status === "error") {
    candidates.push("What caused the current error and what exact step should I fix first");
    candidates.push("Which questions can still be answered reliably despite this error");
    return pickUniqueQuestions(candidates, 2);
  }

  if (status === "analyzing") {
    candidates.push("When this run finishes, which two items should I compare first");
    candidates.push("How will CI overlap affect my decision once results are ready");
    return pickUniqueQuestions(candidates, 2);
  }

  if (stage === "post-analysis" && results?.items?.length) {
    candidates.push(`Is ${topItem || "the top item"} truly above ${secondItem || "the runner-up"} after integer CI overlap`);
    candidates.push("Which items should I treat as the same practical tier");
    if (intent === "method") {
      candidates.push("Can you explain the bootstrap CI method and cite the key paper");
    }
    return pickUniqueQuestions(candidates, 2);
  }

  if (stage === "post-schema") {
    if (schema?.indicator_col) {
      candidates.push(`Should I keep indicator "${schema.indicator_col}" for segmented ranking or run overall first`);
    } else {
      candidates.push("How should I set ranking direction before running analysis");
    }
    candidates.push("What should I verify before I click Start Ranking");
    return pickUniqueQuestions(candidates, 2);
  }

  if (stage === "pre-upload") {
    candidates.push("What data format should I prepare for reliable ranking inference");
    candidates.push("Before upload, what assumptions should I verify about my comparisons");
    return pickUniqueQuestions(candidates, 2);
  }

  if (lastUserMessage) {
    candidates.push(`Based on "${lastUserMessage}", what decision-critical evidence is still missing`);
  }
  candidates.push("What should I ask next for my current analysis stage");
  candidates.push("What decision risk should I clarify before moving forward");
  return pickUniqueQuestions(candidates, 2);
}

export function ChatInput({
  onSend,
  disabled = false,
  placeholder = "Type your message...",
  className,
  quoteDrafts = [],
  onQuoteDraftsChange,
  recentMessages = [],
  status,
  schema,
  results,
}: ChatInputProps) {
  const [message, setMessage] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [showSuggestQuestions, setShowSuggestQuestions] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const stage: AnalysisStage = useMemo(() => {
    if (status === "completed" && results) {
      return "post-analysis";
    } else if (schema) {
      return "post-schema";
    }
    return "pre-upload";
  }, [status, schema, results]);

  const suggestQuestionsList = useMemo(
    () => getHardcodedSuggestQuestions(stage, quoteDrafts, status, message, recentMessages, schema, results),
    [stage, quoteDrafts, status, message, recentMessages, schema, results]
  );

  const removeQuote = (index: number) => {
    if (!onQuoteDraftsChange) return;
    onQuoteDraftsChange(quoteDrafts.filter((_, i) => i !== index));
  };

  const clearQuotes = () => {
    onQuoteDraftsChange?.([]);
  };

  const handleSend = async () => {
    const trimmed = message.trim();
    if ((!trimmed && quoteDrafts.length === 0) || isSending || disabled) return;

    const quotesToSend = [...quoteDrafts];
    setIsSending(true);
    setShowSuggestQuestions(false);
    clearQuotes();
    try {
      await onSend(trimmed || "Please answer based on the quoted report excerpt.", quotesToSend);
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

    const quotesToSend = [...quoteDrafts];
    setShowSuggestQuestions(false);
    setIsSending(true);
    clearQuotes();
    try {
      await onSend(question, quotesToSend);
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
    // Close suggest panel on Escape
    if (e.key === "Escape") {
      setShowSuggestQuestions(false);
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
    setShowSuggestQuestions(true);
  };

  const handleBlur = (e: React.FocusEvent) => {
    // Don't hide if clicking on suggest-question panel
    const relatedTarget = e.relatedTarget as HTMLElement;
    if (relatedTarget?.closest('[data-suggest-question-panel]')) {
      return;
    }
    // Delay hiding to allow button clicks
    setTimeout(() => setShowSuggestQuestions(false), 150);
  };

  return (
    <div className={cn("relative", className)}>
      {quoteDrafts.length > 0 && (
        <div className="mb-2 space-y-2 rounded-lg border border-primary/25 bg-primary/[0.06] p-2.5">
          {quoteDrafts.map((quote, index) => (
            <div
              key={`${quote.block_id || "quote"}-${index}`}
              className="flex items-start gap-2 rounded-md border border-primary/20 bg-background/80 px-2.5 py-2"
            >
              <MessageSquareQuote className="h-3.5 w-3.5 mt-0.5 text-primary/80 shrink-0" />
              <div className="min-w-0 flex-1">
                <p className="text-[11px] font-medium text-primary/80 mb-0.5">Quoted from report</p>
                <p className="text-xs text-foreground/85 line-clamp-2 break-words">
                  &ldquo;{quote.quoted_text}&rdquo;
                </p>
              </div>
              <button
                type="button"
                onClick={() => removeQuote(index)}
                className="text-muted-foreground hover:text-foreground transition-colors"
                aria-label="Remove quote"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </div>
          ))}
          {quoteDrafts.length > 1 && (
            <div className="flex justify-end">
              <button
                type="button"
                onClick={clearQuotes}
                className="text-[11px] text-muted-foreground hover:text-foreground transition-colors"
              >
                Clear all quotes
              </button>
            </div>
          )}
        </div>
      )}

      {/* Suggest Question Panel */}
      {showSuggestQuestions && suggestQuestionsList.length > 0 && (
        <div
          data-suggest-question-panel
          className="absolute bottom-full left-0 right-0 mb-2 bg-background border border-border rounded-lg shadow-lg overflow-hidden animate-in slide-in-from-bottom-2 duration-200"
        >
          <div className="flex items-center gap-2 px-3 py-2 border-b border-border/50 bg-background">
            <Zap className="h-3.5 w-3.5 text-primary" />
            <span className="text-xs font-medium text-muted-foreground">Suggest Question</span>
          </div>
          <div className="p-1.5 bg-background">
            {suggestQuestionsList.slice(0, 2).map((question, index) => (
              <button
                key={index}
                onClick={() => handleQuickQuestion(question)}
                disabled={isSending || disabled}
                className="w-full flex items-start gap-2 px-3 py-2 text-left text-xs rounded-md bg-background hover:bg-background transition-colors group disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <ChevronRight className="h-3.5 w-3.5 text-muted-foreground group-hover:text-primary group-hover:translate-x-0.5 transition-all shrink-0" />
                <span className="whitespace-normal break-words leading-snug">{question}</span>
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
            if (!showSuggestQuestions) {
              setShowSuggestQuestions(true);
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
          disabled={(!message.trim() && quoteDrafts.length === 0) || disabled || isSending}
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
