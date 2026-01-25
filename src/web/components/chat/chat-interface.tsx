"use client";

import { useRef, useEffect, memo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { AlertTriangle, Sparkles, Bot, User } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import type { ChatMessage } from "@/hooks/use-omnirank";

interface ChatInterfaceProps {
  messages: ChatMessage[];
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

const ChatMessageItem = memo(function ChatMessageItem({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  const isSystem = message.role === "system";

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

export function ChatInterface({ messages, className }: ChatInterfaceProps) {
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
            <ChatMessageItem key={message.id} message={message} />
          ))}
        </AnimatePresence>
      </div>
    </ScrollArea>
  );
}
