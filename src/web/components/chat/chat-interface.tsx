"use client";

import { useRef, useEffect, memo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Bot, User } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import { RankingPreviewBubble } from "./ranking-preview-bubble";
import { DataAgentWorkingBubble } from "./data-agent-working-bubble";
import type { ChatMessage } from "@/hooks/use-omnirank";
import type { AnalysisConfig } from "@/lib/api";

interface ChatInterfaceProps {
  messages: ChatMessage[];
  onStartAnalysis?: (config: AnalysisConfig) => void;
  isAnalyzing?: boolean;
  className?: string;
}

const MessageIcon = memo(function MessageIcon({ role }: { role: ChatMessage["role"] }) {
  if (role === "user") {
    return (
      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-[#0F172A] text-white flex items-center justify-center shadow-sm">
        <User className="h-5 w-5" />
      </div>
    );
  }

  return (
    <div className="flex-shrink-0 w-8 h-8 rounded-full border-2 border-[#0F172A] bg-white text-[#0F172A] flex items-center justify-center shadow-sm">
      <Bot className="h-5 w-5" />
    </div>
  );
});

interface ChatMessageItemProps {
  message: ChatMessage;
  onStartAnalysis?: (config: AnalysisConfig) => void;
  isAnalyzing?: boolean;
}

const ChatMessageItem = memo(function ChatMessageItem({ 
  message, 
  onStartAnalysis,
  isAnalyzing = false,
}: ChatMessageItemProps) {
  const isUser = message.role === "user";
  const isSystem = message.role === "system";
  const isConfigMessage = message.type === "ranking-config" && message.configData;
  const isWorkingMessage = message.type === "data-agent-working";

  // Render Data Agent working bubble
  if (isWorkingMessage) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -10 }}
        layout
        className="flex w-full gap-3 mb-4 items-end flex-row"
      >
        <MessageIcon role="assistant" />
        <DataAgentWorkingBubble
          isComplete={message.workingData?.isComplete}
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
        className="flex w-full gap-3 mb-4 items-end flex-row"
      >
        <MessageIcon role="assistant" />
        <RankingPreviewBubble
          schema={message.configData.schema}
          onStartAnalysis={onStartAnalysis}
          isAnalyzing={isAnalyzing}
        />
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
        "flex w-full gap-3 mb-4 items-end",
        isUser ? "flex-row-reverse" : "flex-row"
      )}
    >
      <MessageIcon role={message.role} />

      <div className={cn(
        "relative max-w-[90%] px-4 py-3 rounded-2xl shadow-sm border text-sm",
        isUser
          ? "bg-white dark:bg-zinc-800 border-border/50 text-foreground rounded-br-sm"
          : "bg-white dark:bg-zinc-800 border-border/50 text-foreground rounded-bl-sm",
        isSystem && "bg-yellow-500/5 border-yellow-500/20"
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
                <span
                  dangerouslySetInnerHTML={{
                    __html: line.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>"),
                  }}
                />
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
  isAnalyzing = false,
  className,
}: ChatInterfaceProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <ScrollArea ref={scrollRef} className={cn("pr-2", className)}>
      <div className="space-y-2 px-1 py-2">
        <AnimatePresence initial={false}>
          {messages.map((message) => (
            <ChatMessageItem 
              key={message.id} 
              message={message}
              onStartAnalysis={onStartAnalysis}
              isAnalyzing={isAnalyzing}
            />
          ))}
        </AnimatePresence>
      </div>
    </ScrollArea>
  );
}
