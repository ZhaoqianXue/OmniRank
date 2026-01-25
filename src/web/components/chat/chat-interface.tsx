"use client";

import { useRef, useEffect, memo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Bot, User, AlertTriangle, Sparkles } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import type { ChatMessage } from "@/hooks/use-omnirank";

interface ChatInterfaceProps {
  messages: ChatMessage[];
  className?: string;
}

function MessageIcon({ role, agent }: { role: ChatMessage["role"]; agent?: ChatMessage["agent"] }) {
  if (role === "system") {
    return (
      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-yellow-500/10 flex items-center justify-center">
        <AlertTriangle className="h-4 w-4 text-yellow-500" />
      </div>
    );
  }
  
  if (role === "user") {
    return (
      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
        <User className="h-4 w-4 text-primary" />
      </div>
    );
  }

  // Assistant with different agent types
  const agentColors = {
    data: "bg-blue-500/10 text-blue-500",
    orchestrator: "bg-purple-500/10 text-purple-500",
    analyst: "bg-green-500/10 text-green-500",
  };

  const colorClass = agent ? agentColors[agent] : "bg-cyan-500/10 text-cyan-500";

  return (
    <div className={cn("flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center", colorClass.split(" ")[0])}>
      {agent === "analyst" ? (
        <Sparkles className={cn("h-4 w-4", colorClass.split(" ")[1])} />
      ) : (
        <Bot className={cn("h-4 w-4", colorClass.split(" ")[1])} />
      )}
    </div>
  );
}

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
        "flex gap-3 p-3 rounded-lg",
        isUser && "bg-primary/5",
        isSystem && "bg-yellow-500/5 border border-yellow-500/20",
        !isUser && !isSystem && "bg-card/50"
      )}
    >
      <MessageIcon role={message.role} agent={message.agent} />
      
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-sm font-medium">
            {isUser ? "You" : isSystem ? "System" : message.agent ? `${message.agent.charAt(0).toUpperCase() + message.agent.slice(1)} Agent` : "OmniRank"}
          </span>
          <span className="text-xs text-muted-foreground">
            {new Date(message.timestamp).toLocaleTimeString()}
          </span>
        </div>
        <div className={cn(
          "text-sm prose prose-sm dark:prose-invert max-w-none",
          isSystem && "text-yellow-600 dark:text-yellow-400"
        )}>
          {message.content.split("\n").map((line, i) => (
            <p key={i} className="mb-1 last:mb-0">
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
    <ScrollArea ref={scrollRef} className={cn("pr-4", className)}>
      <div className="space-y-3">
        <AnimatePresence initial={false}>
          {messages.length === 0 ? (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex flex-col items-center justify-center h-[200px] text-center text-muted-foreground"
            >
              <Bot className="h-12 w-12 mb-4 opacity-30" />
              <p>Upload a file to start the analysis</p>
              <p className="text-sm">OmniRank will guide you through the process</p>
            </motion.div>
          ) : (
            messages.map((message) => (
              <ChatMessageItem key={message.id} message={message} />
            ))
          )}
        </AnimatePresence>
      </div>
    </ScrollArea>
  );
}
