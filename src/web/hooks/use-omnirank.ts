"use client";

import { useCallback, useState } from "react";
import {
  askQuestion,
  confirmSession,
  deleteSession,
  EXAMPLE_DATASETS,
  getSessionSnapshot,
  getDataPreview,
  getRunJobStatus,
  inferSession,
  loadExampleData as apiLoadExampleData,
  normalizeRunResponse,
  startRunSession,
  toValidationWarnings,
  uploadFile,
  type AnalysisConfig,
  type ArtifactDescriptor,
  type DataPreview,
  type ExampleDataInfo,
  type FormatValidationResult,
  type PlotSpec,
  type QualityValidationResult,
  type QuotePayload,
  type RankingResults,
  type ReportOutput,
  type RunResponse,
  type SemanticSchema,
  type ValidationWarning,
} from "@/lib/api";
import { DATA_AGENT_TOTAL_STEPS } from "@/lib/data-agent-steps";

export type AnalysisStatus =
  | "idle"
  | "uploading"
  | "configuring"
  | "analyzing"
  | "completed"
  | "error";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: number;
  agent?: "data" | "orchestrator" | "analyst";
  type?: "text" | "data-agent-working" | "ranking-config" | "analysis-complete";
  configData?: {
      schema: SemanticSchema;
      warnings: ValidationWarning[];
      formatResult?: FormatValidationResult | null;
      qualityResult?: QualityValidationResult | null;
      detectedFormat?: "pointwise" | "pairwise" | "multiway";
    };
  workingData?: {
    completedSteps: number;
    totalSteps: number;
  };
  analysisCompleteData?: {
    suggestedQuestions: string[];
  };
}

export interface OmniRankState {
  sessionId: string | null;
  status: AnalysisStatus;

  dataSource: "upload" | "example" | null;
  exampleDataInfo: ExampleDataInfo | null;
  dataPreview: DataPreview | null;

  filename: string | null;
  schema: SemanticSchema | null;
  warnings: ValidationWarning[];
  formatResult: FormatValidationResult | null;
  qualityResult: QualityValidationResult | null;

  config: AnalysisConfig | null;

  results: RankingResults | null;
  reportOutput: ReportOutput | null;
  plots: PlotSpec[];
  artifacts: ArtifactDescriptor[];

  isReportVisible: boolean;

  messages: ChatMessage[];

  progress: number;
  progressMessage: string;

  error: string | null;
}

const WELCOME_MESSAGE: ChatMessage = {
  id: "welcome-message",
  role: "assistant",
  content:
    "Welcome to OmniRank. Upload data or select an example dataset, then confirm schema and run spectral ranking.",
  timestamp: 0,
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
  formatResult: null,
  qualityResult: null,
  config: null,
  results: null,
  reportOutput: null,
  plots: [],
  artifacts: [],
  isReportVisible: true,
  messages: [WELCOME_MESSAGE],
  progress: 0,
  progressMessage: "",
  error: null,
};

const STEP_TRANSITION_DELAY_MS = 220;
const RUN_JOB_POLL_INTERVAL_MS = 600;
const RUN_JOB_TIMEOUT_MS = 10 * 60 * 1000;
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export function useOmniRank() {
  const [state, setState] = useState<OmniRankState>(initialState);

  const addMessage = useCallback(
    (
      role: ChatMessage["role"],
      content: string,
      agent?: ChatMessage["agent"],
      options?: {
        type?: ChatMessage["type"];
        configData?: ChatMessage["configData"];
        workingData?: ChatMessage["workingData"];
        analysisCompleteData?: ChatMessage["analysisCompleteData"];
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
        analysisCompleteData: options?.analysisCompleteData,
      };
      setState((prev) => ({ ...prev, messages: [...prev.messages, message] }));
      return message;
    },
    []
  );

  const updateWorkingMessageProgress = useCallback((messageId: string, completedSteps: number) => {
    setState((prev) => ({
      ...prev,
      messages: prev.messages.map((message) => {
        if (message.id !== messageId || message.type !== "data-agent-working") {
          return message;
        }

        return {
          ...message,
          workingData: {
            completedSteps: Math.max(0, Math.min(completedSteps, DATA_AGENT_TOTAL_STEPS)),
            totalSteps: DATA_AGENT_TOTAL_STEPS,
          },
        };
      }),
    }));
  }, []);

  const prepareSession = useCallback(
    async (
      sessionId: string,
      filename: string,
      source: "upload" | "example",
      exampleInfo: ExampleDataInfo | null,
      workingMessageId?: string
    ) => {
      const completeStep = async (stepNumber: number, delayMs = 0) => {
        if (workingMessageId) {
          updateWorkingMessageProgress(workingMessageId, stepNumber);
        }
        if (delayMs > 0) {
          await sleep(delayMs);
        }
      };

      setState((prev) => ({
        ...prev,
        sessionId,
        filename,
        dataSource: source,
        exampleDataInfo: exampleInfo,
        status: "uploading",
        error: null,
        progress: 0.2,
        progressMessage: "Loading preview...",
      }));

      const preview = await getDataPreview(sessionId);
      setState((prev) => ({ ...prev, dataPreview: preview, progress: 0.35, progressMessage: "Inferring schema..." }));
      await completeStep(1);

      const infer = await inferSession(sessionId);
      if (!infer.success || !infer.schema_result?.schema) {
        throw new Error(infer.error || "Failed to infer schema");
      }
      await completeStep(2);

      const warnings = toValidationWarnings(infer.format_result, infer.quality_result);

      setState((prev) => ({
        ...prev,
        schema: infer.schema_result?.schema || null,
        warnings,
        formatResult: infer.format_result || null,
        qualityResult: infer.quality_result || null,
        status: "configuring",
        progress: 0.5,
        progressMessage: "Schema inferred. Awaiting confirmation.",
      }));

      await completeStep(3, STEP_TRANSITION_DELAY_MS);
      await completeStep(4, STEP_TRANSITION_DELAY_MS);
      await completeStep(5, STEP_TRANSITION_DELAY_MS);

      addMessage("assistant", "", "data", {
        type: "ranking-config",
        configData: {
          schema: infer.schema_result.schema,
          warnings,
          formatResult: infer.format_result || null,
          qualityResult: infer.quality_result || null,
          detectedFormat: infer.schema_result.format,
        },
      });
    },
    [addMessage, updateWorkingMessageProgress]
  );

  const handleUpload = useCallback(
    async (file: File) => {
      setState((prev) => ({
        ...prev,
        status: "uploading",
        filename: file.name,
        dataSource: "upload",
        exampleDataInfo: null,
        dataPreview: null,
        error: null,
        progress: 0.1,
        progressMessage: "Uploading file...",
      }));
      addMessage("user", `Uploaded: ${file.name}`);
      const workingMessage = addMessage("assistant", "", "data", {
        type: "data-agent-working",
        workingData: {
          completedSteps: 0,
          totalSteps: DATA_AGENT_TOTAL_STEPS,
        },
      });

      try {
        const upload = await uploadFile(file);
        await prepareSession(upload.session_id, upload.filename, "upload", null, workingMessage.id);
        setState((prev) => ({ ...prev, progress: 0.6 }));
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : "Upload failed";
        setState((prev) => ({ ...prev, status: "error", error: errorMessage }));
        addMessage("system", `Error: ${errorMessage}`);
        return;
      }
    },
    [addMessage, prepareSession]
  );

  const loadExampleData = useCallback(
    async (exampleId: string) => {
      const exampleInfo = EXAMPLE_DATASETS.find((item) => item.id === exampleId) || null;
      if (!exampleInfo) {
        const errorMessage = "Unknown example dataset";
        setState((prev) => ({ ...prev, status: "error", error: errorMessage }));
        addMessage("system", `Error: ${errorMessage}`);
        return;
      }

      setState((prev) => ({
        ...prev,
        status: "uploading",
        filename: exampleInfo.filename,
        dataSource: "example",
        exampleDataInfo: exampleInfo,
        dataPreview: null,
        error: null,
        progress: 0.1,
        progressMessage: "Loading example dataset...",
      }));
      addMessage("user", `Selected example: ${exampleInfo.title}`);
      const workingMessage = addMessage("assistant", "", "data", {
        type: "data-agent-working",
        workingData: {
          completedSteps: 0,
          totalSteps: DATA_AGENT_TOTAL_STEPS,
        },
      });

      try {
        const upload = await apiLoadExampleData(exampleId);
        await prepareSession(upload.session_id, upload.filename, "example", exampleInfo, workingMessage.id);
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : "Failed to load example data";
        setState((prev) => ({ ...prev, status: "error", error: errorMessage }));
        addMessage("system", `Error: ${errorMessage}`);
        return;
      }
    },
    [addMessage, prepareSession]
  );

  const startAnalysis = useCallback(
    async (config: AnalysisConfig) => {
      if (!state.sessionId || !state.schema) {
        const errorMessage = "Session and schema are required before analysis";
        setState((prev) => ({ ...prev, status: "error", error: errorMessage }));
        addMessage("system", `Error: ${errorMessage}`);
        return;
      }

      setState((prev) => ({
        ...prev,
        status: "analyzing",
        progress: 0.05,
        progressMessage: "Confirming schema...",
        config,
        error: null,
      }));

      addMessage("user", "Starting analysis with confirmed schema and parameters...");

      const selectedItems = config.selected_items || state.schema.ranking_items;
      const effectiveIndicatorCol =
        config.indicator_col === undefined ? state.schema.indicator_col : config.indicator_col;
      const selectedIndicators = effectiveIndicatorCol
        ? config.selected_indicator_values || state.schema.indicator_values
        : [];
      const confirmedSchema: SemanticSchema = {
        bigbetter: config.bigbetter,
        ranking_items: selectedItems,
        indicator_col: effectiveIndicatorCol,
        indicator_values: selectedIndicators,
      };

      try {
        await confirmSession(state.sessionId, {
          confirmed: true,
          confirmed_schema: confirmedSchema,
          user_modifications: [],
          B: config.bootstrap_iterations,
          seed: config.random_seed,
        });

        setState((prev) => ({ ...prev, progress: 0.1, progressMessage: "Submitting analysis job..." }));

        const runStart = await startRunSession(state.sessionId, {
          selected_items: selectedItems,
          selected_indicator_values: effectiveIndicatorCol ? selectedIndicators : undefined,
        });
        setState((prev) => ({
          ...prev,
          progress: Math.max(prev.progress, runStart.progress),
          progressMessage: runStart.message || "Analysis job queued...",
        }));

        let run: RunResponse | null = null;
        const pollStartedAt = Date.now();

        while (Date.now() - pollStartedAt < RUN_JOB_TIMEOUT_MS) {
          const runStatus = await getRunJobStatus(state.sessionId, runStart.job_id);

          setState((prev) => ({
            ...prev,
            progress: Math.max(prev.progress, runStatus.progress),
            progressMessage: runStatus.message || prev.progressMessage,
          }));

          if (runStatus.status === "completed") {
            run = runStatus.result ?? null;
            break;
          }
          if (runStatus.status === "failed") {
            throw new Error(runStatus.error || runStatus.message || "Analysis failed");
          }

          await sleep(RUN_JOB_POLL_INTERVAL_MS);
        }

        if (!run) {
          throw new Error("Analysis timed out while waiting for completion.");
        }
        if (!run.success) {
          throw new Error(run.error || "Analysis failed");
        }

        const normalized = normalizeRunResponse(run);
        const snapshot = await getSessionSnapshot(state.sessionId);

        setState((prev) => ({
          ...prev,
          status: "completed",
          progress: 1,
          progressMessage: "Complete",
          results: normalized.rankingResults,
          reportOutput: snapshot.session.report_output || normalized.reportOutput,
          plots: normalized.plots,
          artifacts: snapshot.artifacts,
          isReportVisible: true,
        }));

        addMessage("assistant", "", "analyst", {
          type: "analysis-complete",
          analysisCompleteData: {
            suggestedQuestions: [
              "What is the top-ranked item and how certain is it?",
              "Which items are statistically close?",
              "What do the confidence intervals imply?",
            ],
          },
        });
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : "Analysis failed";
        setState((prev) => ({ ...prev, status: "error", error: errorMessage }));
        addMessage("system", `Error: ${errorMessage}`);
        return;
      }
    },
    [addMessage, state.schema, state.sessionId]
  );

  const sendMessage = useCallback(
    async (message: string, quotes: QuotePayload[] = []) => {
      addMessage("user", message);

      if (!state.sessionId || state.status !== "completed") {
        const fallback =
          "Run the full analysis first. After completion I can answer quote-aware questions using report blocks and ranking results.";
        addMessage("assistant", fallback, "analyst");
        return;
      }

      try {
        const response = await askQuestion(state.sessionId, message, quotes);
        addMessage("assistant", response.answer.answer, "analyst");
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : "Failed to answer question";
        addMessage("system", `Error: ${errorMessage}`);
        return;
      }
    },
    [addMessage, state.sessionId, state.status]
  );

  const cancelData = useCallback(() => {
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
      formatResult: null,
      qualityResult: null,
      config: null,
      results: null,
      reportOutput: null,
      plots: [],
      artifacts: [],
      progress: 0,
      progressMessage: "",
      error: null,
    }));
    addMessage("system", "Data selection cancelled.");
  }, [addMessage]);

  const reset = useCallback(async () => {
    if (state.sessionId) {
      try {
        await deleteSession(state.sessionId);
      } catch {
        // Ignore cleanup failures.
      }
    }
    setState(initialState);
  }, [state.sessionId]);

  const toggleReportVisibility = useCallback(() => {
    setState((prev) => ({ ...prev, isReportVisible: !prev.isReportVisible }));
  }, []);

  const hideReport = useCallback(() => {
    setState((prev) => ({ ...prev, isReportVisible: false }));
  }, []);

  const showReport = useCallback(() => {
    setState((prev) => ({ ...prev, isReportVisible: true }));
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
