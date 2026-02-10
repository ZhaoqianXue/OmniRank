"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import { ChevronLeft, ChevronRight, History, House, Loader2, LogIn, LogOut, Plus, Trash2, Trophy } from "lucide-react";
import { fadeInLeft, fadeInRight } from "@/lib/animations";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { FileUpload } from "@/components/upload/file-upload";
import { ExampleDataSelector } from "@/components/upload/example-data-selector";
import { DataPreviewComponent } from "@/components/upload/data-preview";
import { ChatInterface } from "@/components/chat/chat-interface";
import { ChatInput } from "@/components/chat/chat-input";
import { ProgressIndicator } from "@/components/ui/progress-indicator";
import { ErrorDisplay } from "@/components/ui/error-display";
import { ReportOverlay } from "@/components/report";
import { useGoogleAuth } from "@/hooks/use-google-auth";
import { createInitialOmniRankState, useOmniRank, type OmniRankState } from "@/hooks/use-omnirank";
import { cn } from "@/lib/utils";
import type { QuotePayload } from "@/lib/api";

const HISTORY_STORAGE_PREFIX = "omnirank_chat_history_v1";
const MAX_HISTORY_ENTRIES = 20;

interface ChatHistoryEntry {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  lastMessage: string;
  state: OmniRankState;
  quoteDrafts: QuotePayload[];
}

interface ChatHistoryStorePayload {
  activeChatId: string | null;
  entries: ChatHistoryEntry[];
}

function historyStorageKey(userSub: string): string {
  return `${HISTORY_STORAGE_PREFIX}:${userSub}`;
}

function deepClone<T>(value: T): T {
  return typeof structuredClone === "function"
    ? structuredClone(value)
    : (JSON.parse(JSON.stringify(value)) as T);
}

function createHistorySnapshot(state: OmniRankState): OmniRankState {
  const snapshot = deepClone(state);
  snapshot.dataPreview = null;
  return snapshot;
}

function truncateText(value: string, maxLength: number): string {
  if (value.length <= maxLength) return value;
  return `${value.slice(0, maxLength - 3)}...`;
}

function deriveConversationTitle(state: OmniRankState): string {
  if (state.filename) {
    return truncateText(state.filename, 48);
  }

  const firstUserMessage = state.messages.find(
    (message) => message.role === "user" && message.content.trim().length > 0
  );

  if (firstUserMessage) {
    return truncateText(firstUserMessage.content.trim(), 48);
  }

  return "New Chat";
}

function deriveLastMessage(state: OmniRankState): string {
  const lastMessage = [...state.messages]
    .reverse()
    .find((message) => message.content.trim().length > 0);

  return lastMessage ? truncateText(lastMessage.content.trim(), 120) : "";
}

function sortHistoryEntries(entries: ChatHistoryEntry[]): ChatHistoryEntry[] {
  return [...entries].sort((a, b) => b.updatedAt - a.updatedAt);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isQuotePayload(value: unknown): value is QuotePayload {
  if (!isRecord(value)) return false;
  if (typeof value.quoted_text !== "string") return false;
  if (typeof value.source !== "string") return false;
  return value.source === "report" || value.source === "user_upload" || value.source === "external";
}

function parseHistoryEntry(value: unknown): ChatHistoryEntry | null {
  if (!isRecord(value)) return null;

  const {
    id,
    title,
    createdAt,
    updatedAt,
    lastMessage,
    state,
    quoteDrafts,
  } = value;

  if (
    typeof id !== "string" ||
    typeof title !== "string" ||
    typeof createdAt !== "number" ||
    !Number.isFinite(createdAt) ||
    typeof updatedAt !== "number" ||
    !Number.isFinite(updatedAt) ||
    typeof lastMessage !== "string" ||
    !isRecord(state)
  ) {
    return null;
  }

  const rawState = createHistorySnapshot(state as unknown as OmniRankState);
  const baseState = createInitialOmniRankState();
  const parsedState: OmniRankState = {
    ...baseState,
    ...rawState,
    warnings: Array.isArray(rawState.warnings) ? rawState.warnings : [],
    plots: Array.isArray(rawState.plots) ? rawState.plots : [],
    artifacts: Array.isArray(rawState.artifacts) ? rawState.artifacts : [],
    messages:
      Array.isArray(rawState.messages) && rawState.messages.length > 0
        ? rawState.messages
        : baseState.messages,
  };

  if (typeof parsedState.status !== "string") {
    return null;
  }

  const parsedQuoteDrafts = Array.isArray(quoteDrafts)
    ? quoteDrafts.filter(isQuotePayload)
    : [];

  return {
    id,
    title,
    createdAt,
    updatedAt,
    lastMessage,
    state: parsedState,
    quoteDrafts: parsedQuoteDrafts,
  };
}

function loadHistoryStore(userSub: string): ChatHistoryStorePayload {
  if (typeof window === "undefined") {
    return { activeChatId: null, entries: [] };
  }

  try {
    const raw = window.localStorage.getItem(historyStorageKey(userSub));
    if (!raw) {
      return { activeChatId: null, entries: [] };
    }

    const parsed = JSON.parse(raw) as {
      activeChatId?: unknown;
      entries?: unknown;
    };

    const entries = Array.isArray(parsed.entries)
      ? parsed.entries
          .map(parseHistoryEntry)
          .filter((entry): entry is ChatHistoryEntry => entry !== null)
      : [];

    return {
      activeChatId: typeof parsed.activeChatId === "string" ? parsed.activeChatId : null,
      entries: sortHistoryEntries(entries).slice(0, MAX_HISTORY_ENTRIES),
    };
  } catch {
    return { activeChatId: null, entries: [] };
  }
}

function persistHistoryStore(userSub: string, payload: ChatHistoryStorePayload): void {
  if (typeof window === "undefined") return;

  window.localStorage.setItem(
    historyStorageKey(userSub),
    JSON.stringify({
      activeChatId: payload.activeChatId,
      entries: sortHistoryEntries(payload.entries).slice(0, MAX_HISTORY_ENTRIES),
    })
  );
}

function formatTimestamp(timestamp: number): string {
  return new Date(timestamp).toLocaleString();
}

export default function Home() {
  const [isSidebarExpanded, setIsSidebarExpanded] = useState(false);
  const [isHistoryDialogOpen, setIsHistoryDialogOpen] = useState(false);
  const [quoteDrafts, setQuoteDrafts] = useState<QuotePayload[]>([]);
  const [historyEntries, setHistoryEntries] = useState<ChatHistoryEntry[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [isHistoryBootstrapping, setIsHistoryBootstrapping] = useState(false);

  const historySyncPausedRef = useRef(false);

  const {
    user,
    isLoggedIn,
    isLoading: isAuthLoading,
    isConfigured: isGoogleConfigured,
    error: googleAuthError,
    login,
    logout,
  } = useGoogleAuth();

  const {
    state,
    handleUpload,
    loadExampleData,
    cancelData,
    startAnalysis,
    sendMessage,
    reset,
    hydrateState,
    refreshDataPreview,
    toggleReportVisibility,
    hideReport,
    exampleDatasets,
  } = useOmniRank();

  const isIdle = state.status === "idle";
  const isUploading = state.status === "uploading";
  const isPreviewLoading = isUploading && !state.dataPreview;
  const hasData = state.filename && (state.status === "uploading" || state.status === "configuring" || state.status === "analyzing" || state.status === "completed");
  const isAnalyzing = state.status === "analyzing";
  const showProgress = state.status === "analyzing";
  const showResults = state.status === "completed" && state.results;

  const sortedHistoryEntries = useMemo(
    () => sortHistoryEntries(historyEntries),
    [historyEntries]
  );

  const activeHistoryEntry = useMemo(
    () => sortedHistoryEntries.find((entry) => entry.id === activeChatId) || null,
    [activeChatId, sortedHistoryEntries]
  );

  const userInitial = useMemo(() => {
    if (!isLoggedIn || !user) return "?";
    const fromName = user.name.trim().charAt(0);
    const fromEmail = user.email.trim().charAt(0);
    const initial = fromName || fromEmail || "U";
    return initial.toUpperCase();
  }, [isLoggedIn, user]);

  const sidebarItems = useMemo(
    () => [
      { id: "home", label: "Home", icon: House },
      { id: "new-chat", label: "New Chat", icon: Plus },
      { id: "history", label: "Chat History", icon: History },
      { id: "leaderboard", label: "LLM Leaderboard", icon: Trophy },
    ] as const,
    []
  );

  const runWithHistorySyncPaused = useCallback(async (task: () => Promise<void> | void) => {
    historySyncPausedRef.current = true;
    try {
      await task();
    } finally {
      window.setTimeout(() => {
        historySyncPausedRef.current = false;
      }, 0);
    }
  }, []);

  const createHistoryEntry = useCallback(
    (
      chatId: string,
      sourceState: OmniRankState,
      sourceQuoteDrafts: QuotePayload[],
      now = Date.now()
    ): ChatHistoryEntry => ({
      id: chatId,
      title: deriveConversationTitle(sourceState),
      createdAt: now,
      updatedAt: now,
      lastMessage: deriveLastMessage(sourceState),
      state: createHistorySnapshot(sourceState),
      quoteDrafts: deepClone(sourceQuoteDrafts),
    }),
    []
  );

  const upsertHistoryEntry = useCallback((entry: ChatHistoryEntry) => {
    setHistoryEntries((prev) => {
      const existing = prev.find((item) => item.id === entry.id);
      const normalizedEntry = existing
        ? {
            ...entry,
            createdAt: existing.createdAt,
            updatedAt: Date.now(),
          }
        : entry;

      const next = [normalizedEntry, ...prev.filter((item) => item.id !== entry.id)];
      return sortHistoryEntries(next).slice(0, MAX_HISTORY_ENTRIES);
    });
  }, []);

  const hydrateFromHistoryEntry = useCallback(
    async (entry: ChatHistoryEntry) => {
      await runWithHistorySyncPaused(async () => {
        setActiveChatId(entry.id);
        setQuoteDrafts(deepClone(entry.quoteDrafts));
        hydrateState(entry.state);

        if (entry.state.sessionId) {
          try {
            await refreshDataPreview(entry.state.sessionId);
          } catch {
            // Keep saved state if preview cannot be refreshed.
          }
        }
      });
    },
    [hydrateState, refreshDataPreview, runWithHistorySyncPaused]
  );

  useEffect(() => {
    if (!isLoggedIn || !user) {
      setHistoryEntries([]);
      setActiveChatId(null);
      setIsHistoryBootstrapping(false);
      return;
    }

    const store = loadHistoryStore(user.sub);
    setIsHistoryBootstrapping(true);
    setHistoryEntries(store.entries);
    setActiveChatId(store.activeChatId);
  }, [isLoggedIn, user]);

  useEffect(() => {
    if (!isLoggedIn || !user || isHistoryBootstrapping) return;

    persistHistoryStore(user.sub, {
      activeChatId,
      entries: historyEntries,
    });
  }, [activeChatId, historyEntries, isHistoryBootstrapping, isLoggedIn, user]);

  useEffect(() => {
    if (!isLoggedIn || !user || !isHistoryBootstrapping) return;

    if (!activeChatId) {
      setIsHistoryBootstrapping(false);
      return;
    }

    const activeEntry = sortedHistoryEntries.find((entry) => entry.id === activeChatId);
    if (!activeEntry) {
      setIsHistoryBootstrapping(false);
      return;
    }

    void (async () => {
      try {
        await hydrateFromHistoryEntry(activeEntry);
      } finally {
        setIsHistoryBootstrapping(false);
      }
    })();
  }, [
    activeChatId,
    hydrateFromHistoryEntry,
    isHistoryBootstrapping,
    isLoggedIn,
    sortedHistoryEntries,
    user,
  ]);

  useEffect(() => {
    if (!isLoggedIn || !user) return;
    if (isHistoryBootstrapping) return;
    if (activeChatId) return;

    const newChatId = crypto.randomUUID();
    const seededEntry = createHistoryEntry(newChatId, state, quoteDrafts);
    setActiveChatId(newChatId);
    setHistoryEntries((prev) => sortHistoryEntries([seededEntry, ...prev]).slice(0, MAX_HISTORY_ENTRIES));
  }, [activeChatId, createHistoryEntry, isHistoryBootstrapping, isLoggedIn, quoteDrafts, state, user]);

  useEffect(() => {
    if (!isLoggedIn || !user || !activeChatId) return;
    if (isHistoryBootstrapping) return;
    if (historySyncPausedRef.current) return;

    const historyEntry = createHistoryEntry(activeChatId, state, quoteDrafts, Date.now());
    upsertHistoryEntry(historyEntry);
  }, [
    activeChatId,
    createHistoryEntry,
    isHistoryBootstrapping,
    isLoggedIn,
    quoteDrafts,
    state,
    upsertHistoryEntry,
    user,
  ]);

  const handleCreateNewChat = useCallback(async () => {
    if (!isLoggedIn) {
      setQuoteDrafts([]);
      await reset();
      return;
    }

    const newChatId = crypto.randomUUID();
    const initialState = createInitialOmniRankState();
    const freshEntry = createHistoryEntry(newChatId, initialState, []);

    await runWithHistorySyncPaused(async () => {
      setIsHistoryDialogOpen(false);
      setActiveChatId(newChatId);
      setQuoteDrafts([]);
      setHistoryEntries((prev) =>
        sortHistoryEntries([freshEntry, ...prev.filter((entry) => entry.id !== newChatId)]).slice(0, MAX_HISTORY_ENTRIES)
      );
      await reset({ deleteCurrentSession: false });
    });
  }, [createHistoryEntry, isLoggedIn, reset, runWithHistorySyncPaused]);

  const handleSelectHistoryEntry = useCallback(
    async (entryId: string) => {
      const entry = sortedHistoryEntries.find((item) => item.id === entryId);
      if (!entry) return;

      setIsHistoryDialogOpen(false);
      await hydrateFromHistoryEntry(entry);
    },
    [hydrateFromHistoryEntry, sortedHistoryEntries]
  );

  const handleDeleteHistoryEntry = useCallback(
    async (entryId: string) => {
      const remaining = sortedHistoryEntries.filter((entry) => entry.id !== entryId);
      setHistoryEntries(remaining);

      if (entryId !== activeChatId) {
        return;
      }

      if (remaining.length === 0) {
        await handleCreateNewChat();
        return;
      }

      await hydrateFromHistoryEntry(remaining[0]);
    },
    [activeChatId, handleCreateNewChat, hydrateFromHistoryEntry, sortedHistoryEntries]
  );

  const handleSidebarAction = useCallback(
    async (menuId: string) => {
      if (menuId === "home") {
        window.open("/", "_blank", "noopener,noreferrer");
        return;
      }

      if (menuId === "leaderboard") {
        window.open("/leaderboard", "_blank", "noopener,noreferrer");
        return;
      }

      if (menuId === "new-chat") {
        await handleCreateNewChat();
        return;
      }

      if (menuId === "history") {
        if (!isLoggedIn) {
          return;
        }
        setIsHistoryDialogOpen(true);
      }
    },
    [handleCreateNewChat, isLoggedIn]
  );

  const handleLoginToggle = useCallback(async () => {
    if (isLoggedIn) {
      await runWithHistorySyncPaused(async () => {
        setIsHistoryDialogOpen(false);
        setQuoteDrafts([]);
        setHistoryEntries([]);
        setActiveChatId(null);
        await reset({ deleteCurrentSession: false });
      });
      logout();
      return;
    }

    await login();
  }, [isLoggedIn, login, logout, reset, runWithHistorySyncPaused]);

  const handleSendMessage = async (message: string, quotes: QuotePayload[] = []) => {
    const effectiveQuotes = quotes.length > 0 ? quotes : quoteDrafts;
    if (effectiveQuotes.length > 0) {
      setQuoteDrafts([]);
    }
    await sendMessage(message, effectiveQuotes);
  };

  const handleQuoteToInput = (quote: QuotePayload) => {
    setQuoteDrafts((prev) => {
      const exists = prev.some(
        (q) =>
          q.quoted_text === quote.quoted_text &&
          q.block_id === quote.block_id &&
          q.kind === quote.kind
      );
      if (exists) return prev;
      return [...prev, quote];
    });
  };

  return (
    <main className="min-h-screen relative overflow-hidden">
      <div className="fixed inset-0 grid-pattern opacity-50" />
      <div className="fixed inset-0 bg-gradient-to-br from-background via-background to-accent/5" />

      <Dialog open={isHistoryDialogOpen} onOpenChange={setIsHistoryDialogOpen}>
        <DialogContent className="sm:max-w-xl">
          <DialogHeader>
            <DialogTitle>Chat History</DialogTitle>
            <DialogDescription>
              {isLoggedIn
                ? "Switch between saved chats under your current Google account."
                : "Sign in with Google to unlock persistent chat history."}
            </DialogDescription>
          </DialogHeader>

          {!isLoggedIn ? (
            <div className="rounded-md border border-border/60 bg-muted/30 p-4 text-sm text-muted-foreground">
              Login is required to view chat history.
            </div>
          ) : sortedHistoryEntries.length === 0 ? (
            <div className="rounded-md border border-border/60 bg-muted/30 p-4 text-sm text-muted-foreground">
              No saved chats yet.
            </div>
          ) : (
            <ScrollArea className="h-[420px] pr-3">
              <div className="space-y-2">
                {sortedHistoryEntries.map((entry) => (
                  <button
                    key={entry.id}
                    type="button"
                    onClick={() => {
                      void handleSelectHistoryEntry(entry.id);
                    }}
                    className={cn(
                      "w-full rounded-md border p-3 text-left transition-colors",
                      entry.id === activeChatId
                        ? "border-primary/40 bg-primary/10"
                        : "border-border/60 hover:bg-muted/40"
                    )}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2">
                          <p className="truncate text-sm font-medium">{entry.title}</p>
                          <span className="shrink-0 rounded bg-muted px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-muted-foreground">
                            {entry.state.status}
                          </span>
                        </div>
                        <p className="mt-1 text-[11px] text-muted-foreground">
                          {formatTimestamp(entry.updatedAt)}
                        </p>
                        {entry.lastMessage && (
                          <p className="mt-1 truncate text-xs text-muted-foreground">{entry.lastMessage}</p>
                        )}
                      </div>

                      <Button
                        type="button"
                        variant="ghost"
                        size="icon-xs"
                        className="h-7 w-7"
                        onClick={(event) => {
                          event.stopPropagation();
                          void handleDeleteHistoryEntry(entry.id);
                        }}
                        aria-label="Delete chat history entry"
                      >
                        <Trash2 className="h-3.5 w-3.5 text-muted-foreground" />
                      </Button>
                    </div>
                  </button>
                ))}
              </div>
            </ScrollArea>
          )}
        </DialogContent>
      </Dialog>

      <div className="relative z-10 flex min-h-screen">
        <aside
          className={cn(
            "shrink-0 border-r border-border/40 bg-background backdrop-blur-sm flex flex-col justify-between transition-all duration-300 ease-in-out",
            isSidebarExpanded ? "w-56" : "w-12"
          )}
        >
          <div className="p-2">
            <Button
              variant="ghost"
              size="icon-sm"
              className="h-8 w-8"
              onClick={() => setIsSidebarExpanded((prev) => !prev)}
              aria-label={isSidebarExpanded ? "Collapse sidebar" : "Expand sidebar"}
            >
              {isSidebarExpanded ? (
                <ChevronLeft className="h-4 w-4" />
              ) : (
                <ChevronRight className="h-4 w-4" />
              )}
            </Button>
            <div className="my-2 border-b border-border/40" />
            <nav className="space-y-1">
              {sidebarItems.map((item) => {
                const Icon = item.icon;
                const isHistoryDisabled = item.id === "history" && !isLoggedIn;

                return (
                  <Button
                    key={item.id}
                    variant="ghost"
                    size="sm"
                    disabled={isHistoryDisabled}
                    onClick={() => {
                      void handleSidebarAction(item.id);
                    }}
                    className={cn(
                      "h-9 w-full justify-start px-2",
                      !isSidebarExpanded && "justify-center px-0",
                      isHistoryDisabled && "opacity-60"
                    )}
                  >
                    <Icon className="h-4 w-4 shrink-0" />
                    {isSidebarExpanded && <span className="truncate">{item.label}</span>}
                  </Button>
                );
              })}
            </nav>
          </div>

          <div className="p-2 pt-0 border-t border-border/40">
            <Button
              variant="outline"
              onClick={() => {
                void handleLoginToggle();
              }}
              aria-label={isLoggedIn ? "Logout" : "Login"}
              className={cn(
                "mt-2 h-9 w-full justify-start gap-2 px-2",
                !isSidebarExpanded && "h-8 w-8 justify-center p-0"
              )}
              disabled={isAuthLoading}
            >
              <div
                className={cn(
                  "h-6 w-6 rounded-full border flex items-center justify-center text-[10px] font-semibold",
                  isLoggedIn
                    ? "border-primary/40 bg-primary text-primary-foreground"
                    : "border-border/80 bg-muted text-muted-foreground"
                )}
              >
                {userInitial}
              </div>

              {isSidebarExpanded && (
                <>
                  <span className="text-xs font-medium truncate">
                    {isLoggedIn ? user?.name || user?.email || "Logout" : "Login with Google"}
                  </span>
                  {isAuthLoading ? (
                    <Loader2 className="h-3.5 w-3.5 ml-auto animate-spin text-muted-foreground" />
                  ) : isLoggedIn ? (
                    <LogOut className="h-3.5 w-3.5 ml-auto text-muted-foreground" />
                  ) : (
                    <LogIn className="h-3.5 w-3.5 ml-auto text-muted-foreground" />
                  )}
                </>
              )}
            </Button>

            {isSidebarExpanded && !isGoogleConfigured && (
              <p className="mt-2 text-[10px] leading-relaxed text-amber-500">
                Set NEXT_PUBLIC_GOOGLE_CLIENT_ID to enable Google login.
              </p>
            )}

            {isSidebarExpanded && googleAuthError && (
              <p className="mt-2 text-[10px] leading-relaxed text-red-400">{googleAuthError}</p>
            )}
          </div>
        </aside>

        <div className="flex-1 min-w-0 px-4 pb-4 pt-4 md:px-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <motion.div
              variants={fadeInLeft}
              initial="hidden"
              animate="show"
              className="lg:col-span-2"
            >
              <Card className="h-[calc(100vh-32px)] flex flex-col bg-card backdrop-blur-sm glow-border py-4 relative overflow-hidden">
                {showResults && (
                  <ReportOverlay
                    isVisible={state.isReportVisible}
                    results={state.results}
                    reportOutput={state.reportOutput}
                    plots={state.plots}
                    artifacts={state.artifacts}
                    sessionId={state.sessionId}
                    schema={state.schema}
                    config={state.config}
                    onClose={hideReport}
                    onQuoteToInput={handleQuoteToInput}
                  />
                )}

                <CardContent className="flex-1 flex flex-col min-h-0">
                  {isIdle && !hasData && (
                    <div className="space-y-4 mb-4">
                      <FileUpload
                        onUpload={handleUpload}
                        mode="dropzone"
                        isUploading={false}
                        isUploaded={false}
                        filename={null}
                      />
                      <ExampleDataSelector
                        examples={exampleDatasets}
                        onSelect={loadExampleData}
                        disabled={false}
                      />
                    </div>
                  )}

                  {hasData && (
                    <div className="space-y-4 flex-1 flex flex-col min-h-0">
                      <FileUpload
                        onUpload={handleUpload}
                        onCancel={!isUploading ? cancelData : undefined}
                        mode="sticker"
                        filename={state.filename}
                        isExample={state.dataSource === "example"}
                        isUploading={isUploading}
                      />
                      <div className="flex-1 min-h-0">
                        <DataPreviewComponent
                          preview={state.dataPreview}
                          exampleInfo={state.exampleDataInfo}
                          isLoading={isPreviewLoading}
                          className="h-full"
                        />
                      </div>
                    </div>
                  )}

                  {showProgress && (
                    <div className="mb-4">
                      <ProgressIndicator
                        progress={state.progress}
                        message={state.progressMessage}
                      />
                    </div>
                  )}

                  {state.status === "error" && state.error && (
                    <div className="mb-4">
                      <ErrorDisplay
                        title="Analysis Error"
                        message={state.error}
                        type="error"
                        onRetry={reset}
                      />
                    </div>
                  )}
                </CardContent>
              </Card>
            </motion.div>

            <motion.div
              variants={fadeInRight}
              initial="hidden"
              animate="show"
            >
              <Card className="h-[calc(100vh-32px)] flex flex-col bg-card backdrop-blur-sm glow-border gap-0 p-0 overflow-hidden">
                <div className="flex items-center justify-center py-2 px-3 border-b border-border/40 min-h-[48px] shrink-0">
                  <div className="text-sm font-bold flex items-center justify-center gap-2 min-w-0">
                    <div className="w-1.5 h-1.5 rounded-full bg-yellow-500 animate-pulse" />
                    <span className="truncate">{activeHistoryEntry?.title || "OmniRank Agent"}</span>
                  </div>
                </div>

                <CardContent className="flex-1 min-h-0 p-0">
                  <ChatInterface
                    messages={state.messages}
                    onStartAnalysis={startAnalysis}
                    onSendMessage={(message) => handleSendMessage(message)}
                    isAnalyzing={isAnalyzing}
                    isCompleted={!!showResults}
                    isReportVisible={state.isReportVisible}
                    onToggleReport={toggleReportVisibility}
                    className="h-full"
                  />
                </CardContent>

                <div className="p-2 border-t border-border/40">
                  <ChatInput
                    onSend={handleSendMessage}
                    disabled={false}
                    placeholder="Type your message..."
                    quoteDrafts={quoteDrafts}
                    onQuoteDraftsChange={setQuoteDrafts}
                    recentMessages={state.messages
                      .filter((message) => message.content.trim().length > 0)
                      .slice(-8)
                      .map((message) => ({ role: message.role, content: message.content }))}
                    status={state.status}
                    schema={state.schema}
                    results={state.results}
                  />
                </div>
              </Card>
            </motion.div>
          </div>
        </div>
      </div>
    </main>
  );
}
