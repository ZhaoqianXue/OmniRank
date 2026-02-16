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
const MAX_SUGGEST_QUESTIONS = 4;

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

  const normalized = cleaned.replace(/(?:\.{3}|…)\s*$/u, "").replace(/[.!]+$/u, "");
  if (!normalized) return "";
  return normalized.endsWith("?") ? normalized : `${normalized}?`;
}

function pickUniqueQuestions(candidates: string[], limit = MAX_SUGGEST_QUESTIONS): string[] {
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
  if (text.includes("compare") || text.includes("vs") || text.includes("better than")) {
    return "comparison";
  }
  if (text.includes("ci") || text.includes("confidence") || text.includes("uncert")) {
    return "uncertainty";
  }
  if (text.includes("method") || text.includes("bootstrap") || text.includes("spectral")) {
    return "method";
  }
  if (text.includes("schema") || text.includes("format") || text.includes("upload")) {
    return "data-prep";
  }
  if (text.includes("error") || text.includes("failed") || text.includes("invalid")) {
    return "error-fix";
  }
  return "general";
}

/**
 * Returns intent-appropriate pre-written questions without embedding raw draft text.
 * Uses detected intent to surface the most relevant question from a curated pool,
 * following SOP rule R12: never output the raw unfinished fragment unchanged.
 */
function getIntentQuestions(
  intent: SuggestIntent,
  topItem: string,
  secondItem: string,
): string[] {
  switch (intent) {
    case "comparison":
      return topItem && secondItem
        ? [`Is ${topItem} really better than ${secondItem}, or is it too close to tell`]
        : ["How do I compare two items in my ranking"];
    case "uncertainty":
      return ["How sure can I be about the ranking order"];
    case "method":
      return ["How does the spectral ranking method work in simple terms"];
    case "data-prep":
      return ["What data format does OmniRank need"];
    case "error-fix":
      return ["What went wrong and how can I fix it"];
    case "general":
    default:
      return [];
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
  // Draft-driven: use intent to select from curated pool, not embed raw text
  const intentCandidates = trimmedDraft ? getIntentQuestions(intent, topItem, secondItem) : [];

  // --- Quote context (highest priority) ---
  if (quoteDrafts.length > 0) {
    const quoted = quoteDrafts[0]?.quoted_text?.trim() || "the quoted content";
    const preview = quoted.length > 68 ? quoted.slice(0, 68).trimEnd() : quoted;
    const anchors: string[] = [
      `What decision implication should I draw from this quote: "${preview}"`,
    ];
    if (results?.items?.length) {
      anchors.push("Does this quote align with the ranking and confidence interval (CI) evidence");
    }
    if (quoteDrafts.length > 1) {
      anchors.push("Do these quotes agree or contradict each other");
    } else if (!results?.items?.length) {
      anchors.push("What's the caveat in this statement");
    }
    anchors.push("What should I do next based on this");
    anchors.push("What assumption here could be wrong");
    return pickUniqueQuestions([anchors[0], ...intentCandidates, ...anchors.slice(1)], MAX_SUGGEST_QUESTIONS);
  }

  // --- Error status (SOP R5: Q2 must address what can still be done while blocked) ---
  if (status === "error") {
    const anchors = [
      "What went wrong and how can I fix it",
      "What can I still do while this is being resolved",
      "Is the problem with my data or the settings",
    ];
    return pickUniqueQuestions([anchors[0], ...intentCandidates, ...anchors.slice(1)], MAX_SUGGEST_QUESTIONS);
  }

  // --- Analyzing status (SOP R6: address CI overlap and decision threshold) ---
  if (status === "analyzing") {
    const anchors = [
      "What should I look at first when results are ready",
      "How will I know if one item is clearly better than another",
    ];
    return pickUniqueQuestions([anchors[0], ...intentCandidates, ...anchors.slice(1)], MAX_SUGGEST_QUESTIONS);
  }

  // --- Post-analysis (differentiated from bubble: deeper exploration) ---
  if (stage === "post-analysis" && results?.items?.length) {
    const anchors = [
      topItem && secondItem
        ? `What drives the difference between ${topItem} and ${secondItem}`
        : "What drives the biggest ranking differences",
      "Can you explain the uncertainty in simpler terms",
      "How confident should I be in the overall ordering",
    ];
    return pickUniqueQuestions([anchors[0], ...intentCandidates, ...anchors.slice(1)], MAX_SUGGEST_QUESTIONS);
  }

  // --- Post-schema (SOP R7: de-risk configuration, indicator usage) ---
  if (stage === "post-schema") {
    const anchors: string[] = [
      "Does this detected schema look right for my data",
    ];
    if (schema?.indicator_col) {
      anchors.push(`Should I rank everything together or separately by "${schema.indicator_col}"`);
    } else {
      anchors.push("Is 'higher is better' the correct direction for my metric");
    }
    anchors.push("What should I verify before starting");
    anchors.push("What will I get when the analysis finishes");
    return pickUniqueQuestions([anchors[0], ...intentCandidates, ...anchors.slice(1)], MAX_SUGGEST_QUESTIONS);
  }

  // --- Pre-upload (SOP R8: at least one product orientation question) ---
  if (stage === "pre-upload") {
    const anchors = [
      "What is OmniRank and how can it help me",
      "What CSV format does OmniRank need",
      "What types of ranking data can I analyze",
    ];
    return pickUniqueQuestions([anchors[0], ...intentCandidates, ...anchors.slice(1)], MAX_SUGGEST_QUESTIONS);
  }

  // --- Fallback (no meta-prompts per SOP R4) ---
  const candidates: string[] = [];
  if (lastUserMessage) {
    const preview = lastUserMessage.length > 50 ? lastUserMessage.slice(0, 50).trimEnd() + "..." : lastUserMessage;
    candidates.push(`What else should I know about "${preview}"`);
  }
  const anchors = [
    "Can you summarize what we know so far",
    "What's the biggest risk I should watch for",
    "What would make me more confident in this",
    "What should I do next",
  ];
  return pickUniqueQuestions([anchors[0], ...intentCandidates, ...candidates, ...anchors.slice(1)], MAX_SUGGEST_QUESTIONS);
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
            <span className="text-xs font-medium text-muted-foreground">Suggested Questions</span>
          </div>
          <div className="p-1.5 bg-background">
            {suggestQuestionsList.slice(0, MAX_SUGGEST_QUESTIONS).map((question, index) => (
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
