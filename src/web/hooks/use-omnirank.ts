"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import {
  uploadFile,
  startDataAgent,
  analyze,
  getResults,
  createWebSocket,
  askQuestion,
  loadExampleData as apiLoadExampleData,
  getDataPreview,
  EXAMPLE_DATASETS,
  type UploadResponse,
  type SchemaReadyPayload,
  type AnalysisConfig,
  type RankingResults,
  type ValidationWarning,
  type InferredSchema,
  type DataPreview,
  type ExampleDataInfo,
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
  // Special message types for interactive components
  type?: "text" | "data-agent-working" | "ranking-config";
  // Data for ranking-config type
  configData?: {
    schema: InferredSchema;
    warnings: ValidationWarning[];
  };
  // Data for data-agent-working type
  workingData?: {
    isComplete: boolean;
  };
}

export interface OmniRankState {
  // Session
  sessionId: string | null;
  status: AnalysisStatus;

  // Data source tracking
  dataSource: "upload" | "example" | null;
  exampleDataInfo: ExampleDataInfo | null;
  dataPreview: DataPreview | null;

  // File
  filename: string | null;
  schema: InferredSchema | null;
  warnings: ValidationWarning[];

  // Config
  config: AnalysisConfig | null;

  // Results
  results: RankingResults | null;

  // Report visibility
  isReportVisible: boolean;

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
  dataSource: null,
  exampleDataInfo: null,
  dataPreview: null,
  filename: null,
  schema: null,
  warnings: [],
  config: null,
  results: null,
  isReportVisible: true,  // Report is visible by default when results are ready
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
  const addMessage = useCallback((
    role: ChatMessage["role"], 
    content: string, 
    agent?: ChatMessage["agent"],
    options?: { 
      type?: ChatMessage["type"]; 
      configData?: ChatMessage["configData"];
      workingData?: ChatMessage["workingData"];
    }
  ) => {
    const message: ChatMessage = {
      id: crypto.randomUUID(),
      role,
      content,
      timestamp: Date.now(),
      agent,
      type: options?.type,
      configData: options?.configData,
      workingData: options?.workingData,
    };
    setState((prev) => ({
      ...prev,
      messages: [...prev.messages, message],
    }));
    return message;
  }, []);

  // Update an existing message by ID
  const updateMessage = useCallback((
    messageId: string,
    updates: Partial<Pick<ChatMessage, "content" | "workingData" | "configData" | "type">>
  ) => {
    setState((prev) => ({
      ...prev,
      messages: prev.messages.map((msg) =>
        msg.id === messageId ? { ...msg, ...updates } : msg
      ),
    }));
  }, []);

  // Handle file upload - Two phase process
  // Phase 1: Upload file, show preview immediately
  // Phase 2: Start Data Agent after preview, results via WebSocket
  const handleUpload = useCallback(async (file: File) => {
    // Set uploading state with filename for immediate UI feedback
    setState((prev) => ({
      ...prev,
      status: "uploading",
      filename: file.name,
      dataSource: "upload",
      dataPreview: null,
      error: null,
      progress: 0,
      progressMessage: "Uploading file...",
    }));

    // 1. User message showing what was uploaded
    addMessage("user", `Uploaded: ${file.name}`);

    // Track working message ID for later update
    let workingMessageId: string | null = null;

    try {
      // Phase 1: Upload file (returns immediately)
      const response = await uploadFile(file);

      // File uploaded! Update state to show data preview
      setState((prev) => ({
        ...prev,
        sessionId: response.session_id,
        filename: response.filename,
        status: "uploading",  // Keep uploading until schema is ready
        dataSource: "upload",
        exampleDataInfo: null,
        dataPreview: null,
      }));

      // Fetch data preview before starting Data Agent
      try {
        const preview = await getDataPreview(response.session_id);
        setState((prev) => ({
          ...prev,
          dataPreview: preview,
        }));
      } catch {
        console.warn("Failed to fetch data preview");
      }

      // 2. Show Data Agent working message after preview is ready
      const workingMessage = addMessage(
        "assistant",
        "",
        "data",
        { type: "data-agent-working", workingData: { isComplete: false } }
      );
      workingMessageId = workingMessage.id;

      // Connect WebSocket to receive Data Agent results
      wsRef.current = createWebSocket(
        response.session_id,
        (msg) => {
          // Handle WebSocket messages
          if (msg.type === "schema_ready") {
            // Data Agent completed! 
            const payload = msg.payload as SchemaReadyPayload;
            setState((prev) => ({
              ...prev,
              schema: payload.inferred_schema,
              warnings: payload.warnings,
              status: "configuring",
            }));

            // Add ranking config bubble
            addMessage(
              "assistant",
              "",
              "data",
              { 
                type: "ranking-config", 
                configData: { schema: payload.inferred_schema, warnings: payload.warnings } 
              }
            );

            // Mark working message as complete after preview is shown
            if (workingMessageId) {
              const messageId = workingMessageId;
              setTimeout(() => {
                updateMessage(messageId, { workingData: { isComplete: true } });
              }, 0);
            }
          } else if (msg.type === "progress") {
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
              isReportVisible: true,  // Auto-show report when ranking completes
            }));
          } else if (msg.type === "error") {
            const payload = msg.payload as { error: string };
            setState((prev) => ({
              ...prev,
              error: payload.error,
              status: "error",
            }));
            // Mark working as complete on error
            if (workingMessageId) {
              updateMessage(workingMessageId, { workingData: { isComplete: true } });
            }
            addMessage("system", `Error: ${payload.error}`);
          }
        },
        () => console.error("WebSocket error"),
        () => console.log("WebSocket closed"),
        async () => {
          try {
            await startDataAgent(response.session_id);
          } catch (error) {
            const errorMessage = error instanceof Error ? error.message : "Failed to start Data Agent";
            setState((prev) => ({
              ...prev,
              status: "error",
              error: errorMessage,
            }));
            if (workingMessageId) {
              updateMessage(workingMessageId, { workingData: { isComplete: true } });
            }
            addMessage("system", `Error: ${errorMessage}`);
          }
        }
      );

      return response;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Upload failed";
      setState((prev) => ({
        ...prev,
        status: "error",
        error: errorMessage,
        filename: null,
      }));
      if (workingMessageId) {
        updateMessage(workingMessageId, { workingData: { isComplete: true } });
      }
      addMessage("system", `Error: ${errorMessage}`);
      throw error;
    }
  }, [addMessage, updateMessage]);

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
                  isReportVisible: true,  // Auto-show report when ranking completes
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

  // Load example data - Two phase process (same as file upload)
  const loadExampleData = useCallback(async (exampleId: string) => {
    const exampleInfo = EXAMPLE_DATASETS.find((d) => d.id === exampleId);
    if (!exampleInfo) {
      throw new Error("Unknown example dataset");
    }

    // Set uploading state with filename for immediate UI feedback
    setState((prev) => ({
      ...prev,
      status: "uploading",
      filename: exampleInfo.filename,
      dataSource: "example",
      exampleDataInfo: exampleInfo,
      dataPreview: null,
      error: null,
      progress: 0,
      progressMessage: "Loading example data...",
    }));

    // 1. User message showing what was selected
    addMessage("user", `Selected example: ${exampleInfo.title}`);

    // Track working message ID for later update
    let workingMessageId: string | null = null;

    try {
      // Phase 1: Load example file (returns immediately)
      const response = await apiLoadExampleData(exampleId);

      // File loaded! Update state
      setState((prev) => ({
        ...prev,
        sessionId: response.session_id,
        filename: response.filename,
        status: "uploading",  // Keep uploading until schema is ready
        dataSource: "example",
        exampleDataInfo: exampleInfo,
        dataPreview: null,
      }));

      // Fetch data preview before starting Data Agent
      try {
        const preview = await getDataPreview(response.session_id);
        setState((prev) => ({
          ...prev,
          dataPreview: preview,
        }));
      } catch {
        console.warn("Failed to fetch data preview");
      }

      // 2. Show Data Agent working message after preview is ready
      const workingMessage = addMessage(
        "assistant",
        "",
        "data",
        { type: "data-agent-working", workingData: { isComplete: false } }
      );
      workingMessageId = workingMessage.id;

      // Connect WebSocket to receive Data Agent results
      wsRef.current = createWebSocket(
        response.session_id,
        (msg) => {
          if (msg.type === "schema_ready") {
            // Data Agent completed!
            const payload = msg.payload as SchemaReadyPayload;
            setState((prev) => ({
              ...prev,
              schema: payload.inferred_schema,
              warnings: payload.warnings,
              status: "configuring",
            }));

            // Add ranking config bubble
            addMessage(
              "assistant",
              "",
              "data",
              { 
                type: "ranking-config", 
                configData: { schema: payload.inferred_schema, warnings: payload.warnings } 
              }
            );

            // Mark working message as complete after preview is shown
            if (workingMessageId) {
              const messageId = workingMessageId;
              setTimeout(() => {
                updateMessage(messageId, { workingData: { isComplete: true } });
              }, 0);
            }
          } else if (msg.type === "progress") {
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
              isReportVisible: true,  // Auto-show report when ranking completes
            }));
          } else if (msg.type === "error") {
            const payload = msg.payload as { error: string };
            setState((prev) => ({
              ...prev,
              error: payload.error,
              status: "error",
            }));
            if (workingMessageId) {
              updateMessage(workingMessageId, { workingData: { isComplete: true } });
            }
            addMessage("system", `Error: ${payload.error}`);
          }
        },
        () => console.error("WebSocket error"),
        () => console.log("WebSocket closed"),
        async () => {
          try {
            await startDataAgent(response.session_id);
          } catch (error) {
            const errorMessage = error instanceof Error ? error.message : "Failed to start Data Agent";
            setState((prev) => ({
              ...prev,
              status: "error",
              error: errorMessage,
            }));
            if (workingMessageId) {
              updateMessage(workingMessageId, { workingData: { isComplete: true } });
            }
            addMessage("system", `Error: ${errorMessage}`);
          }
        }
      );

      return response;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Failed to load example data";
      setState((prev) => ({
        ...prev,
        status: "error",
        error: errorMessage,
        filename: null,
      }));
      if (workingMessageId) {
        updateMessage(workingMessageId, { workingData: { isComplete: true } });
      }
      addMessage("system", `Error: ${errorMessage}`);
      throw error;
    }
  }, [addMessage, updateMessage]);

  // Cancel data selection and return to idle state
  const cancelData = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
    setState((prev) => ({
      ...prev,
      sessionId: null,
      status: "idle",
      dataSource: null,
      exampleDataInfo: null,
      dataPreview: null,
      filename: null,
      schema: null,
      warnings: [],
      config: null,
      results: null,
      progress: 0,
      progressMessage: "",
      error: null,
    }));
    addMessage("system", "Data selection cancelled. Please upload a new file or select an example dataset.");
  }, [addMessage]);

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

  // Toggle report visibility
  const toggleReportVisibility = useCallback(() => {
    setState((prev) => ({
      ...prev,
      isReportVisible: !prev.isReportVisible,
    }));
  }, []);

  // Hide report
  const hideReport = useCallback(() => {
    setState((prev) => ({
      ...prev,
      isReportVisible: false,
    }));
  }, []);

  // Show report
  const showReport = useCallback(() => {
    setState((prev) => ({
      ...prev,
      isReportVisible: true,
    }));
  }, []);

  return {
    state,
    handleUpload,
    loadExampleData,
    cancelData,
    startAnalysis,
    sendMessage,
    addMessage,
    reset,
    toggleReportVisibility,
    hideReport,
    showReport,
    exampleDatasets: EXAMPLE_DATASETS,
  };
}
