"use client";

import { useRef, useEffect, memo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { User } from "lucide-react";
import { Fragment } from "react";
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

  const renderInlineBold = (line: string) => {
    const parts = line.split("**");
    return parts.map((part, index) => {
      if (index % 2 === 1) {
        return <strong key={`bold-${index}`}>{part}</strong>;
      }
      return <Fragment key={`text-${index}`}>{part}</Fragment>;
    });
  };

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
        <div className={cn(
          "prose prose-sm dark:prose-invert max-w-none break-words",
          isSystem && "text-yellow-600 dark:text-yellow-400"
        )}>
              {message.content.split("\n").map((line, i) => (
                <p key={i} className="mb-1 last:mb-0 leading-relaxed">
                  {line.startsWith("- ") ? (
                    <span className="flex gap-2">
                      <span className="text-muted-foreground">•</span>
                      <span>{line.slice(2)}</span>
                    </span>
                  ) : line.includes("**") ? (
                    <span>{renderInlineBold(line)}</span>
                  ) : (
                    line
                  )}
                </p>
              ))}
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
