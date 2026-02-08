"use client";

import { useCallback, useState } from "react";
import {
  askQuestion,
  confirmSession,
  deleteSession,
  EXAMPLE_DATASETS,
  getSessionSnapshot,
  getDataPreview,
  inferSession,
  loadExampleData as apiLoadExampleData,
  normalizeRunResponse,
  runSession,
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
  type SemanticSchema,
  type ValidationWarning,
} from "@/lib/api";

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
    isComplete: boolean;
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

  const prepareSession = useCallback(
    async (sessionId: string, filename: string, source: "upload" | "example", exampleInfo: ExampleDataInfo | null) => {
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

      const infer = await inferSession(sessionId);
      if (!infer.success || !infer.schema_result?.schema) {
        throw new Error(infer.error || "Failed to infer schema");
      }

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
    [addMessage]
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
      addMessage("assistant", "", "data", { type: "data-agent-working", workingData: { isComplete: false } });

      try {
        const upload = await uploadFile(file);
        await prepareSession(upload.session_id, upload.filename, "upload", null);
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
      addMessage("assistant", "", "data", { type: "data-agent-working", workingData: { isComplete: false } });

      try {
        const upload = await apiLoadExampleData(exampleId);
        await prepareSession(upload.session_id, upload.filename, "example", exampleInfo);
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
        progress: 0.7,
        progressMessage: "Confirming schema and running engine...",
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

        setState((prev) => ({ ...prev, progress: 0.82, progressMessage: "Executing spectral ranking..." }));

        const run = await runSession(state.sessionId, {
          selected_items: selectedItems,
          selected_indicator_values: effectiveIndicatorCol ? selectedIndicators : undefined,
        });

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
