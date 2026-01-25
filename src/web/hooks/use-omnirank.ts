"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import {
  uploadFile,
  analyze,
  getResults,
  createWebSocket,
  askQuestion,
  type UploadResponse,
  type AnalysisConfig,
  type RankingResults,
  type ValidationWarning,
  type InferredSchema,
} from "@/lib/api";

// ============================================================================
// Types
// ============================================================================

export type AnalysisStatus = "idle" | "uploading" | "configuring" | "analyzing" | "completed" | "error";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: number;
  agent?: "data" | "orchestrator" | "analyst";
}

export interface OmniRankState {
  // Session
  sessionId: string | null;
  status: AnalysisStatus;

  // File
  filename: string | null;
  schema: InferredSchema | null;
  warnings: ValidationWarning[];

  // Config
  config: AnalysisConfig | null;

  // Results
  results: RankingResults | null;

  // Chat
  messages: ChatMessage[];

  // Progress
  progress: number;
  progressMessage: string;

  // Error
  error: string | null;
}

// ============================================================================
// Initial State
// ============================================================================

const WELCOME_MESSAGE: ChatMessage = {
  id: "welcome-message",
  role: "assistant",
  content: "Welcome to OmniRank! I'm OmniRank Assistant — here to help you navigate and use this platform. I can answer questions, perform ranking analysis, and analyze results. Let me know what you need help with!",
  timestamp: 0, // Stable timestamp to prevent SSR hydration mismatch
  agent: "analyst",
};

const initialState: OmniRankState = {
  sessionId: null,
  status: "idle",
  filename: null,
  schema: null,
  warnings: [],
  config: null,
  results: null,
  messages: [WELCOME_MESSAGE],
  progress: 0,
  progressMessage: "",
  error: null,
};

// ============================================================================
// Hook
// ============================================================================

export function useOmniRank() {
  const [state, setState] = useState<OmniRankState>(initialState);
  const wsRef = useRef<WebSocket | null>(null);
  const pollingRef = useRef<NodeJS.Timeout | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
      }
    };
  }, []);

  // Add a chat message
  const addMessage = useCallback((role: ChatMessage["role"], content: string, agent?: ChatMessage["agent"]) => {
    const message: ChatMessage = {
      id: crypto.randomUUID(),
      role,
      content,
      timestamp: Date.now(),
      agent,
    };
    setState((prev) => ({
      ...prev,
      messages: [...prev.messages, message],
    }));
    return message;
  }, []);

  // Handle file upload
  const handleUpload = useCallback(async (file: File) => {
    setState((prev) => ({
      ...prev,
      status: "uploading",
      error: null,
      progress: 0,
      progressMessage: "Uploading file...",
    }));

    addMessage("user", `Uploading ${file.name}...`);

    try {
      const response = await uploadFile(file);

      // Connect WebSocket for this session
      wsRef.current = createWebSocket(
        response.session_id,
        (msg) => {
          // Handle WebSocket messages
          if (msg.type === "progress") {
            const payload = msg.payload as { progress: number; message: string };
            setState((prev) => ({
              ...prev,
              progress: payload.progress,
              progressMessage: payload.message,
            }));
          } else if (msg.type === "agent_message") {
            const payload = msg.payload as { agent: string; message: string };
            addMessage("assistant", payload.message, payload.agent as ChatMessage["agent"]);
          } else if (msg.type === "result") {
            const payload = msg.payload as { results: RankingResults };
            setState((prev) => ({
              ...prev,
              results: payload.results,
              status: "completed",
              progress: 1,
              progressMessage: "Complete!",
            }));
          } else if (msg.type === "error") {
            const payload = msg.payload as { error: string };
            setState((prev) => ({
              ...prev,
              error: payload.error,
              status: "error",
            }));
            addMessage("system", `Error: ${payload.error}`);
          }
        },
        () => console.error("WebSocket error"),
        () => console.log("WebSocket closed")
      );

      setState((prev) => ({
        ...prev,
        sessionId: response.session_id,
        filename: response.filename,
        schema: response.inferred_schema,
        warnings: response.warnings,
        status: "configuring",
        progress: 0,
        progressMessage: "",
      }));

      // Add success message
      addMessage("assistant", `File uploaded successfully! Detected format: **${response.inferred_schema.format}** with ${response.inferred_schema.ranking_items.length} items to rank.`, "data");

      // Add warnings if any
      if (response.warnings.length > 0) {
        const warningText = response.warnings.map((w) => `- ${w.message}`).join("\n");
        addMessage("system", `Warnings:\n${warningText}`);
      }

      return response;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Upload failed";
      setState((prev) => ({
        ...prev,
        status: "error",
        error: errorMessage,
      }));
      addMessage("system", `Error: ${errorMessage}`);
      throw error;
    }
  }, [addMessage]);

  // Start analysis
  const startAnalysis = useCallback(async (config: AnalysisConfig) => {
    if (!state.sessionId) {
      throw new Error("No session ID");
    }

    setState((prev) => ({
      ...prev,
      config,
      status: "analyzing",
      progress: 0,
      progressMessage: "Starting analysis...",
      error: null,
    }));

    addMessage("user", "Starting analysis with the configured parameters...");

    try {
      await analyze(state.sessionId, config);

      // Start polling for results (fallback if WebSocket doesn't deliver)
      pollingRef.current = setInterval(async () => {
        try {
          const result = await getResults(state.sessionId!);
          if (result.status === "completed" && result.results) {
            clearInterval(pollingRef.current!);
            pollingRef.current = null;

            setState((prev) => {
              // Only update if not already completed (WebSocket might have delivered first)
              if (prev.status !== "completed") {
                return {
                  ...prev,
                  results: result.results!,
                  status: "completed",
                  progress: 1,
                  progressMessage: "Complete!",
                };
              }
              return prev;
            });
          } else if (result.status === "error") {
            clearInterval(pollingRef.current!);
            pollingRef.current = null;
            setState((prev) => ({
              ...prev,
              error: result.error || "Unknown error",
              status: "error",
            }));
          }
        } catch (e) {
          // Ignore polling errors
        }
      }, 2000);

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Analysis failed";
      setState((prev) => ({
        ...prev,
        status: "error",
        error: errorMessage,
      }));
      addMessage("system", `Error: ${errorMessage}`);
      throw error;
    }
  }, [state.sessionId, addMessage]);

  // Send a follow-up question to the Analyst Agent
  const sendMessage = useCallback(async (message: string) => {
    if (!state.sessionId) {
      addMessage("assistant", "Please upload a dataset first so I can help you analyze it.", "analyst");
      return;
    }

    if (!state.results) {
      addMessage("assistant", "I'm still analyzing your data. Please wait a moment for the results.", "analyst");
      return;
    }

    // Add user message
    addMessage("user", message);

    try {
      const response = await askQuestion(state.sessionId, message);

      // Add assistant response
      addMessage("assistant", response.answer, "analyst");
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Failed to get answer";
      addMessage("system", `Error: ${errorMessage}`);
      throw error;
    }
  }, [state.sessionId, state.results, addMessage]);

  // Reset state
  const reset = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
    setState(initialState);
  }, []);

  return {
    state,
    handleUpload,
    startAnalysis,
    sendMessage,
    addMessage,
    reset,
  };
}
