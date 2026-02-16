"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import { ChevronLeft, ChevronRight, History, House, Loader2, LogOut, Plus, Trash2 } from "lucide-react";
import { fadeInLeft, fadeInRight } from "@/lib/animations";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Progress } from "@/components/ui/progress";
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
import { getDailyUsage, setApiUserSub, type AnalysisConfig, type QuotePayload } from "@/lib/api";

const HISTORY_STORAGE_PREFIX = "omnirank_chat_history_v1";
const GUEST_HISTORY_ENTRY_ID = "__current_guest_chat__";

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

interface SidebarHistoryItem {
  id: string;
  title: string;
  updatedAt: number;
  isRemovable: boolean;
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
      entries: sortHistoryEntries(entries),
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
      entries: sortHistoryEntries(payload.entries),
    })
  );
}

function formatTimestamp(timestamp: number): string {
  return new Date(timestamp).toLocaleString();
}

function GoogleLogo({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      className={cn("h-4 w-4", className)}
      role="img"
      aria-label="Google"
    >
      <path
        fill="#4285F4"
        d="M23.49 12.27c0-.79-.07-1.54-.2-2.27H12v4.29h6.44a5.5 5.5 0 0 1-2.39 3.61v2.99h3.86c2.26-2.08 3.58-5.15 3.58-8.62Z"
      />
      <path
        fill="#34A853"
        d="M12 24c3.24 0 5.95-1.07 7.93-2.9l-3.86-2.99c-1.07.72-2.43 1.14-4.07 1.14-3.12 0-5.76-2.11-6.7-4.95H1.3v3.11A11.99 11.99 0 0 0 12 24Z"
      />
      <path
        fill="#FBBC05"
        d="M5.3 14.3A7.2 7.2 0 0 1 4.92 12c0-.8.14-1.57.39-2.3V6.59H1.3A11.99 11.99 0 0 0 0 12c0 1.93.46 3.76 1.3 5.41L5.3 14.3Z"
      />
      <path
        fill="#EA4335"
        d="M12 4.75c1.76 0 3.34.61 4.58 1.81l3.43-3.43C17.94 1.19 15.24 0 12 0A11.99 11.99 0 0 0 1.3 6.59L5.3 9.7c.94-2.84 3.58-4.95 6.7-4.95Z"
      />
    </svg>
  );
}

export default function Home() {
  const [isSidebarExpanded, setIsSidebarExpanded] = useState(true);
  const [quoteDrafts, setQuoteDrafts] = useState<QuotePayload[]>([]);
  const [historyEntries, setHistoryEntries] = useState<ChatHistoryEntry[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [isHistoryBootstrapping, setIsHistoryBootstrapping] = useState(false);
  const [dailyQuotaProgress, setDailyQuotaProgress] = useState(0);

  const historySyncPausedRef = useRef(false);
  const pendingGuestEntryRef = useRef<ChatHistoryEntry | null>(null);

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

  const guestUpdatedAt = useMemo(() => {
    const lastMessage = [...state.messages]
      .reverse()
      .find((message) => message.content.trim().length > 0);
    return lastMessage?.timestamp ?? Date.now();
  }, [state.messages]);

  const sidebarHistoryItems = useMemo<SidebarHistoryItem[]>(() => {
    if (!isLoggedIn) {
      return [
        {
          id: GUEST_HISTORY_ENTRY_ID,
          title: deriveConversationTitle(state),
          updatedAt: guestUpdatedAt,
          isRemovable: false,
        },
      ];
    }

    return sortedHistoryEntries.map((entry) => ({
      id: entry.id,
      title: entry.title,
      updatedAt: entry.updatedAt,
      isRemovable: true,
    }));
  }, [guestUpdatedAt, isLoggedIn, sortedHistoryEntries, state]);

  const activeSidebarHistoryId = isLoggedIn ? activeChatId : GUEST_HISTORY_ENTRY_ID;
  const activeConversationTitle = useMemo(() => {
    if (isLoggedIn) {
      return activeHistoryEntry?.title || "OmniRank Agent";
    }
    return deriveConversationTitle(state);
  }, [activeHistoryEntry, isLoggedIn, state]);
  const loginButtonDisabled = isAuthLoading || (!isLoggedIn && !isGoogleConfigured);
  const visibleGoogleAuthError = useMemo(() => {
    if (!googleAuthError) return null;
    if (googleAuthError.toLowerCase().includes("popup_closed")) {
      return null;
    }
    return googleAuthError;
  }, [googleAuthError]);

  const refreshDailyQuota = useCallback(async () => {
    if (!isLoggedIn || !user) {
      setDailyQuotaProgress(0);
      return;
    }

    try {
      const usage = await getDailyUsage();
      const progress = Number.isFinite(usage.progress_percent)
        ? Math.max(0, Math.min(100, usage.progress_percent))
        : 0;
      setDailyQuotaProgress(progress);
    } catch {
      // Keep current UI state when usage endpoint is temporarily unavailable.
    }
  }, [isLoggedIn, user]);

  const sidebarItems = useMemo(
    () => [
      { id: "home", label: "Home", icon: House },
      { id: "new-chat", label: "New Chat", icon: Plus },
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

  const buildGuestCarryoverEntry = useCallback((): ChatHistoryEntry | null => {
    const hasMeaningfulWork =
      !!state.sessionId ||
      !!state.filename ||
      state.status !== "idle" ||
      quoteDrafts.length > 0 ||
      state.messages.some(
        (message) => message.id !== "welcome-message" && message.content.trim().length > 0
      );

    if (!hasMeaningfulWork) {
      return null;
    }

    return createHistoryEntry(crypto.randomUUID(), state, quoteDrafts, Date.now());
  }, [createHistoryEntry, quoteDrafts, state]);

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
      return sortHistoryEntries(next);
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
    const pendingGuestEntry = pendingGuestEntryRef.current;
    const mergedEntries = pendingGuestEntry
      ? sortHistoryEntries([pendingGuestEntry, ...store.entries.filter((entry) => entry.id !== pendingGuestEntry.id)])
      : store.entries;
    const nextActiveChatId = pendingGuestEntry?.id ?? store.activeChatId ?? (mergedEntries[0]?.id ?? null);

    pendingGuestEntryRef.current = null;

    setIsHistoryBootstrapping(true);
    setHistoryEntries(mergedEntries);
    setActiveChatId(nextActiveChatId);
    persistHistoryStore(user.sub, {
      activeChatId: nextActiveChatId,
      entries: mergedEntries,
    });
  }, [isLoggedIn, user]);

  useEffect(() => {
    if (!isLoggedIn || !user) {
      setApiUserSub(null);
    } else {
      setApiUserSub(user.sub);
    }
  }, [isLoggedIn, user]);

  useEffect(() => {
    void refreshDailyQuota();
  }, [refreshDailyQuota]);

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
    setHistoryEntries((prev) => sortHistoryEntries([seededEntry, ...prev]));
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
      pendingGuestEntryRef.current = buildGuestCarryoverEntry();
      await login();
      return;
    }

    const newChatId = crypto.randomUUID();
    const initialState = createInitialOmniRankState();
    const freshEntry = createHistoryEntry(newChatId, initialState, []);

    await runWithHistorySyncPaused(async () => {
      setActiveChatId(newChatId);
      setQuoteDrafts([]);
      setHistoryEntries((prev) =>
        sortHistoryEntries([freshEntry, ...prev.filter((entry) => entry.id !== newChatId)])
      );
      await reset({ deleteCurrentSession: false });
    });
  }, [buildGuestCarryoverEntry, createHistoryEntry, isLoggedIn, login, reset, runWithHistorySyncPaused]);

  const handleSelectHistoryEntry = useCallback(
    async (entryId: string) => {
      const entry = sortedHistoryEntries.find((item) => item.id === entryId);
      if (!entry) return;

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

      if (menuId === "new-chat") {
        await handleCreateNewChat();
      }
    },
    [handleCreateNewChat]
  );

  const handleLogin = useCallback(async () => {
    pendingGuestEntryRef.current = buildGuestCarryoverEntry();
    await login();
  }, [buildGuestCarryoverEntry, login]);

  const handleLogout = useCallback(async () => {
    await runWithHistorySyncPaused(async () => {
      setHistoryEntries([]);
      setActiveChatId(null);
    });
    logout();
  }, [logout, runWithHistorySyncPaused]);

  const handleStartAnalysis = useCallback(async (config: AnalysisConfig) => {
    await startAnalysis(config);
    await refreshDailyQuota();
  }, [refreshDailyQuota, startAnalysis]);

  const handleSendMessage = async (message: string, quotes: QuotePayload[] = []) => {
    const effectiveQuotes = quotes.length > 0 ? quotes : quoteDrafts;
    if (effectiveQuotes.length > 0) {
      setQuoteDrafts([]);
    }
    await sendMessage(message, effectiveQuotes);
    await refreshDailyQuota();
  };

  const handleClearCurrentData = useCallback(async () => {
    setQuoteDrafts([]);
    await cancelData();
  }, [cancelData]);

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

      <div className="relative z-10 flex min-h-screen">
        <aside
          className={cn(
            "shrink-0 border-r border-border/40 bg-background backdrop-blur-sm flex flex-col justify-between transition-all duration-300 ease-in-out",
            isSidebarExpanded ? "w-48" : "w-12"
          )}
        >
          <div className="p-2 min-h-0 flex-1 flex flex-col">
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

                return (
                  <Button
                    key={item.id}
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      void handleSidebarAction(item.id);
                    }}
                    className={cn(
                      "h-9 w-full justify-start px-2",
                      !isSidebarExpanded && "justify-center px-0"
                    )}
                  >
                    <Icon className="h-4 w-4 shrink-0" />
                    {isSidebarExpanded && <span className="truncate">{item.label}</span>}
                  </Button>
                );
              })}
            </nav>

            {isSidebarExpanded && (
              <>
                <div className="my-2 border-b border-border/40" />
                <div className="flex items-center gap-2 px-2 py-1 text-[10px] uppercase tracking-[0.12em] text-muted-foreground">
                  <History className="h-3 w-3" />
                  <span>History</span>
                </div>
                <ScrollArea className="mt-1 min-h-0 min-w-0 flex-1">
                  <div className="space-y-1">
                    {sidebarHistoryItems.map((entry) => (
                      <div
                        key={entry.id}
                        role="button"
                        tabIndex={isLoggedIn ? 0 : -1}
                        aria-disabled={!isLoggedIn}
                        onClick={() => {
                          if (!isLoggedIn) return;
                          void handleSelectHistoryEntry(entry.id);
                        }}
                        onKeyDown={(event) => {
                          if (!isLoggedIn) return;
                          if (event.key === "Enter" || event.key === " ") {
                            event.preventDefault();
                            void handleSelectHistoryEntry(entry.id);
                          }
                        }}
                        className={cn(
                          "w-full max-w-full overflow-hidden rounded-md border border-border/60 px-2 py-2 text-left transition-colors",
                          entry.id === activeSidebarHistoryId
                            ? "bg-primary/10 border-primary/40"
                            : "hover:bg-muted/40",
                          !isLoggedIn && "cursor-default"
                        )}
                      >
                        <div
                          className={cn(
                            "grid min-w-0 items-start gap-1.5",
                            entry.isRemovable
                              ? "grid-cols-[minmax(0,1fr)_auto]"
                              : "grid-cols-[minmax(0,1fr)]"
                          )}
                        >
                          <div className="min-w-0 max-w-full overflow-hidden">
                            <p className="block w-full overflow-hidden text-ellipsis whitespace-nowrap text-[11px] font-medium leading-4">
                              {entry.title}
                            </p>
                            <p className="mt-0.5 block w-full overflow-hidden text-ellipsis whitespace-nowrap text-[10px] text-muted-foreground">
                              {formatTimestamp(entry.updatedAt)}
                            </p>
                          </div>

                          {entry.isRemovable && (
                            <Button
                              type="button"
                              variant="ghost"
                              size="icon-xs"
                              className="h-5 w-5 shrink-0"
                              onClick={(event) => {
                                event.stopPropagation();
                                void handleDeleteHistoryEntry(entry.id);
                              }}
                              aria-label="Delete chat history entry"
                            >
                              <Trash2 className="h-3 w-3 text-muted-foreground" />
                            </Button>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
                {!isLoggedIn && (
                  <p className="mt-2 px-2 text-[10px] leading-relaxed text-muted-foreground">
                    Sign in to unlock full history management. Before login, only the current chat is shown.
                  </p>
                )}
              </>
            )}
          </div>

          <div className="p-2 pt-0 border-t border-border/40">
            {isLoggedIn ? (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    variant="outline"
                    aria-label="Open account menu"
                    className={cn(
                      "mt-2 w-full justify-start gap-2 px-2.5 py-2 text-left transition-colors border-[#0b1a30] bg-[#0b1a30] text-white shadow-sm hover:bg-[#132845] hover:text-white disabled:opacity-100 disabled:border-[#0b1a30] disabled:bg-[#0b1a30] disabled:text-white",
                      isSidebarExpanded ? "h-auto min-h-10" : "h-9 w-9 justify-center p-0"
                    )}
                    style={{ backgroundColor: "#0b1a30", borderColor: "#0b1a30", color: "#ffffff" }}
                    disabled={loginButtonDisabled}
                  >
                    <span className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-white ring-1 ring-white/70 shadow-sm">
                      {isAuthLoading ? (
                        <Loader2 className="h-4 w-4 animate-spin text-[#0b1a30]" />
                      ) : (
                        <GoogleLogo className="h-6 w-6 shrink-0" />
                      )}
                    </span>

                    {isSidebarExpanded && (
                      <>
                        <div className="min-w-0 flex-1">
                          <p className="truncate text-xs font-semibold leading-tight">
                            {user?.name || user?.email || "Google account"}
                          </p>
                        </div>
                        {!isAuthLoading && (
                          <ChevronRight className="ml-auto h-3.5 w-3.5 -rotate-90 text-white/80" />
                        )}
                      </>
                    )}
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent
                  side="top"
                  align={isSidebarExpanded ? "start" : "center"}
                  sideOffset={8}
                  className="min-w-0 w-[var(--radix-dropdown-menu-trigger-width)] rounded-lg border-border/60 bg-popover/95 p-2 backdrop-blur"
                >
                  <div className="px-1 py-1.5">
                    <div className="flex items-center justify-between gap-2">
                      <p className="text-[11px] font-medium leading-none text-popover-foreground/90">
                        Daily Usage Limit
                      </p>
                      <span className="text-[11px] tabular-nums text-popover-foreground/75">
                        {dailyQuotaProgress.toFixed(0)}%
                      </span>
                    </div>
                    <Progress value={dailyQuotaProgress} className="mt-2 h-1.5 bg-white/15" />
                  </div>
                  <DropdownMenuSeparator className="my-1.5" />
                  <DropdownMenuItem
                    onSelect={() => {
                      void handleLogout();
                    }}
                    className="rounded-md"
                  >
                    <LogOut className="h-3.5 w-3.5" />
                    Sign out
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            ) : (
              <Button
                variant="outline"
                onClick={() => {
                  void handleLogin();
                }}
                aria-label="Sign-in with Google"
                className={cn(
                  "mt-2 w-full justify-start gap-2 px-2.5 py-2 text-left transition-colors border-[#0b1a30] bg-[#0b1a30] text-white shadow-sm hover:bg-[#132845] hover:text-white disabled:opacity-100 disabled:border-[#0b1a30] disabled:bg-[#0b1a30] disabled:text-white",
                  isSidebarExpanded ? "h-auto min-h-10" : "h-9 w-9 justify-center p-0"
                )}
                style={{ backgroundColor: "#0b1a30", borderColor: "#0b1a30", color: "#ffffff" }}
                disabled={loginButtonDisabled}
              >
                <span className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-white ring-1 ring-white/70 shadow-sm">
                  {isAuthLoading ? (
                    <Loader2 className="h-4 w-4 animate-spin text-white" />
                  ) : (
                    <GoogleLogo className="h-6 w-6 shrink-0" />
                  )}
                </span>

                {isSidebarExpanded && (
                  <div className="min-w-0 flex-1">
                    <p className="whitespace-normal text-xs font-semibold leading-tight">Sign-in with Google</p>
                  </div>
                )}
              </Button>
            )}

            {isSidebarExpanded && !isGoogleConfigured && (
              <p className="mt-2 text-[10px] leading-relaxed text-[#FFD700]">
                Set NEXT_PUBLIC_GOOGLE_CLIENT_ID to enable Google login.
              </p>
            )}

            {isSidebarExpanded && visibleGoogleAuthError && (
              <p className="mt-2 text-[10px] leading-relaxed text-red-400">{visibleGoogleAuthError}</p>
            )}
          </div>
        </aside>

        <div className="flex-1 min-w-0 px-4 pb-4 pt-4 md:px-6">
          <div className="flex flex-col gap-6 lg:flex-row">
            <motion.div
              variants={fadeInLeft}
              initial="hidden"
              animate="show"
              className="min-w-0 flex-1"
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
                    <div className="flex-1 min-h-0 overflow-y-auto pr-1">
                      <div className="space-y-4 pb-4">
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
                    </div>
                  )}

                  {hasData && (
                    <div className="space-y-4 flex-1 flex flex-col min-h-0">
                      <FileUpload
                        onUpload={handleUpload}
                        onCancel={
                          !isUploading
                            ? () => {
                                void handleClearCurrentData();
                              }
                            : undefined
                        }
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
              className="w-full lg:w-[calc((100vw-120px)/3)] lg:shrink-0"
            >
              <Card className="h-[calc(100vh-32px)] flex flex-col bg-card backdrop-blur-sm glow-border gap-0 p-0 overflow-hidden">
                <div className="flex items-center justify-center py-2 px-3 border-b border-border/40 min-h-[48px] shrink-0">
                  <div className="text-sm font-bold flex items-center justify-center gap-2 min-w-0">
                    <div className="w-1.5 h-1.5 rounded-full bg-[#FFD700] animate-pulse" />
                    <span className="truncate" title={activeConversationTitle}>OmniRank Agent</span>
                  </div>
                </div>

                <CardContent className="flex-1 min-h-0 p-0">
                  <ChatInterface
                    messages={state.messages}
                    onStartAnalysis={handleStartAnalysis}
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
