"use client";

import { useRef, useEffect, memo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { MessageSquareQuote, User } from "lucide-react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import { MaterialSymbol } from "@/components/ui/material-symbol";
import { RankingPreviewBubble } from "./ranking-preview-bubble";
import { DataAgentWorkingBubble } from "./data-agent-working-bubble";
import { AnalysisCompleteBubble } from "./analysis-complete-bubble";
import type { ChatMessage } from "@/hooks/use-omnirank";
import type { AnalysisConfig } from "@/lib/api";

interface ChatInterfaceProps {
  messages: ChatMessage[];
  onStartAnalysis?: (config: AnalysisConfig) => void;
  onSendMessage?: (message: string) => void | Promise<void>;
  isAnalyzing?: boolean;
  isCompleted?: boolean;
  isReportVisible?: boolean;
  onToggleReport?: () => void;
  className?: string;
}

const CHAT_MARKDOWN_COMPONENTS: Components = {
  p: ({ children }) => <p className="mb-1.5 last:mb-0 leading-relaxed">{children}</p>,
  ul: ({ children }) => <ul className="my-1.5 list-disc pl-5 space-y-1">{children}</ul>,
  ol: ({ children }) => <ol className="my-1.5 list-decimal pl-5 space-y-1">{children}</ol>,
  li: ({ children }) => <li className="leading-relaxed">{children}</li>,
  strong: ({ children }) => <strong className="font-semibold text-foreground">{children}</strong>,
  em: ({ children }) => <em className="italic">{children}</em>,
  blockquote: ({ children }) => (
    <blockquote className="my-2 border-l-2 border-primary/30 pl-3 text-muted-foreground">{children}</blockquote>
  ),
  pre: ({ children }) => (
    <pre className="my-2 overflow-x-auto rounded-md bg-muted/40 p-2.5 text-xs leading-relaxed">{children}</pre>
  ),
  code: ({ children, className }) => (
    <code className={cn("rounded bg-muted/50 px-1 py-0.5 font-mono text-[12px]", className)}>
      {children}
    </code>
  ),
  a: ({ href, children }) => (
    <a href={href} target="_blank" rel="noreferrer" className="text-primary underline underline-offset-2">
      {children}
    </a>
  ),
  table: ({ children }) => (
    <div className="my-2 overflow-x-auto rounded-md border border-border/50">
      <table className="w-full border-collapse text-xs">{children}</table>
    </div>
  ),
  thead: ({ children }) => <thead className="bg-muted/60">{children}</thead>,
  th: ({ children }) => <th className="border-b px-2 py-1.5 text-left font-semibold">{children}</th>,
  tr: ({ children }) => <tr className="border-b last:border-0">{children}</tr>,
  td: ({ children }) => <td className="px-2 py-1.5 align-top">{children}</td>,
};

const MessageIcon = memo(function MessageIcon({ role }: { role: ChatMessage["role"] }) {
  if (role === "user") {
    return (
      <div className="flex-shrink-0 w-8 h-8 rounded-full border border-primary/40 bg-primary text-primary-foreground flex items-center justify-center shadow-sm">
        <User className="h-5 w-5" />
      </div>
    );
  }

  return (
    <div className="flex-shrink-0 w-8 h-8 rounded-full border border-primary/40 bg-primary/10 text-primary flex items-center justify-center shadow-sm">
      <MaterialSymbol 
        icon="robot_2" 
        className="select-none text-primary" 
        style={{ fontSize: '20px' }}
        aria-hidden="true"
      />
    </div>
  );
});

interface ChatMessageItemProps {
  message: ChatMessage;
  onStartAnalysis?: (config: AnalysisConfig) => void;
  onSendMessage?: (message: string) => void | Promise<void>;
  isAnalyzing?: boolean;
  isCompleted?: boolean;
  isReportVisible?: boolean;
  onToggleReport?: () => void;
}

const ChatMessageItem = memo(function ChatMessageItem({ 
  message, 
  onStartAnalysis,
  onSendMessage,
  isAnalyzing = false,
  isCompleted = false,
  isReportVisible = true,
  onToggleReport,
}: ChatMessageItemProps) {
  const isUser = message.role === "user";
  const isSystem = message.role === "system";
  const isConfigMessage = message.type === "ranking-config" && message.configData;
  const isWorkingMessage = message.type === "data-agent-working";
  const isAnalysisCompleteMessage = message.type === "analysis-complete" && message.analysisCompleteData;
  const isThinkingMessage = message.type === "assistant-thinking";
  const attachedQuotes = isUser ? (message.quotes || []) : [];

  // Render Data Agent working bubble
  if (isWorkingMessage) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -10 }}
        layout
        className="flex w-full gap-2 mb-4 items-end flex-row"
      >
        <MessageIcon role="assistant" />
        <DataAgentWorkingBubble
          completedSteps={message.workingData?.completedSteps}
          totalSteps={message.workingData?.totalSteps}
        />
      </motion.div>
    );
  }

  // Render analysis complete bubble with suggested questions
  if (isAnalysisCompleteMessage && message.analysisCompleteData) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -10 }}
        layout
        className="flex w-full gap-2 mb-4 items-end flex-row"
      >
        <MessageIcon role="assistant" />
        <AnalysisCompleteBubble
          suggestedQuestions={message.analysisCompleteData.suggestedQuestions}
          onAskQuestion={onSendMessage}
        />
      </motion.div>
    );
  }

  // Render special ranking config bubble
  if (isConfigMessage && message.configData && onStartAnalysis) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -10 }}
        layout
        className="flex w-full gap-2 mb-4 items-end flex-row"
      >
        <MessageIcon role="assistant" />
        <RankingPreviewBubble
          schema={message.configData.schema}
          detectedFormat={message.configData.detectedFormat}
          formatResult={message.configData.formatResult}
          qualityResult={message.configData.qualityResult}
          warnings={message.configData.warnings}
          onStartAnalysis={onStartAnalysis}
          isAnalyzing={isAnalyzing}
          isCompleted={isCompleted}
          isReportVisible={isReportVisible}
          onToggleReport={onToggleReport}
        />
      </motion.div>
    );
  }

  if (isThinkingMessage) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -10 }}
        layout
        className="flex w-full gap-2 mb-4 items-end flex-row"
      >
        <MessageIcon role="assistant" />
        <div className="relative max-w-[90%] px-4 py-3 rounded-2xl rounded-bl-sm shadow-sm border text-sm bg-background border-border/60 text-foreground">
          <div className="flex items-center gap-1 py-0.5" aria-label="Agent is thinking">
            <span
              className="h-1.5 w-1.5 rounded-full bg-primary/70 animate-bounce"
              style={{ animationDelay: "0ms" }}
            />
            <span
              className="h-1.5 w-1.5 rounded-full bg-primary/70 animate-bounce"
              style={{ animationDelay: "120ms" }}
            />
            <span
              className="h-1.5 w-1.5 rounded-full bg-primary/70 animate-bounce"
              style={{ animationDelay: "240ms" }}
            />
          </div>
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      layout
      className={cn(
        "flex w-full gap-2 mb-4 items-end",
        isUser ? "flex-row-reverse" : "flex-row"
      )}
    >
      <MessageIcon role={message.role} />

      <div className={cn(
        "relative max-w-[90%] px-4 py-3 rounded-2xl shadow-sm border text-sm",
        isUser
          ? "bg-primary/10 border-primary/25 text-foreground rounded-br-sm"
          : "bg-background border-border/60 text-foreground rounded-bl-sm",
        isSystem && "border-yellow-500/20"
      )}>
        {attachedQuotes.length > 0 && (
          <div className="mb-2 space-y-1.5 rounded-lg border border-primary/20 bg-primary/[0.06] p-2">
            <div className="flex items-center gap-1.5 text-[11px] font-medium text-primary/80">
              <MessageSquareQuote className="h-3.5 w-3.5" />
              <span>{attachedQuotes.length} quote{attachedQuotes.length > 1 ? "s" : ""} sent</span>
            </div>
            {attachedQuotes.map((quote, index) => (
              <div
                key={`${quote.block_id || "quote"}-${index}`}
                className="rounded-md border border-primary/15 bg-background/80 px-2 py-1.5"
              >
                <p className="text-xs text-foreground/85 line-clamp-2 break-words">
                  &ldquo;{quote.quoted_text}&rdquo;
                </p>
              </div>
            ))}
          </div>
        )}
        <div className={cn(
          "prose prose-sm dark:prose-invert max-w-none break-words",
          isSystem && "text-yellow-600 dark:text-yellow-400"
        )}>
          <ReactMarkdown remarkPlugins={[remarkGfm]} components={CHAT_MARKDOWN_COMPONENTS}>
            {message.content}
          </ReactMarkdown>
        </div>
      </div>
    </motion.div>
  );
});

export function ChatInterface({ 
  messages, 
  onStartAnalysis,
  onSendMessage,
  isAnalyzing = false,
  isCompleted = false,
  isReportVisible = true,
  onToggleReport,
  className,
}: ChatInterfaceProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    const rafId = window.requestAnimationFrame(() => {
      messagesEndRef.current?.scrollIntoView({ block: "end" });
    });

    return () => window.cancelAnimationFrame(rafId);
  }, [messages]);

  return (
    <ScrollArea className={cn("", className)}>
      <div className="space-y-2 px-2 py-2">
        <AnimatePresence initial={false}>
          {messages.map((message) => (
            <ChatMessageItem 
              key={message.id} 
              message={message}
              onStartAnalysis={onStartAnalysis}
              onSendMessage={onSendMessage}
              isAnalyzing={isAnalyzing}
              isCompleted={isCompleted}
              isReportVisible={isReportVisible}
              onToggleReport={onToggleReport}
            />
          ))}
        </AnimatePresence>
        <div ref={messagesEndRef} aria-hidden="true" className="h-px" />
      </div>
    </ScrollArea>
  );
}
