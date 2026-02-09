/**
 * OmniRank API client (SOP single-agent pipeline).
 */

import type {
  ArtifactDescriptor,
  ConfirmResponse,
  DataPreview,
  DataSummary,
  FormatValidationResult,
  HintSpec,
  InferResponse,
  PlotSpec,
  QualityValidationResult,
  QuotePayload,
  QuestionResponse,
  ReportOutput,
  RunJobStatusResponse,
  RunStartResponse,
  RunResponse,
  SemanticSchema,
  SessionSnapshotResponse,
  UploadResponse,
} from "@shared/types";

export type {
  ArtifactDescriptor,
  ConfirmResponse,
  DataPreview,
  DataSummary,
  FormatValidationResult,
  HintSpec,
  InferResponse,
  PlotSpec,
  QualityValidationResult,
  QuotePayload,
  QuestionResponse,
  ReportOutput,
  RunJobStatusResponse,
  RunStartResponse,
  RunResponse,
  SemanticSchema,
  SessionSnapshotResponse,
  UploadResponse,
};

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

function apiEndpoint(path: string): string {
  return `${API_URL}${path}`;
}

function networkFailureMessage(path: string, error: unknown): string {
  const cause = error instanceof Error ? error.message : String(error);
  const endpoint = apiEndpoint(path);
  let message =
    `Cannot reach OmniRank API at ${endpoint}. ` +
    "Start backend with: cd src/api && uvicorn main:app --reload --port 8000.";

  if (typeof window !== "undefined" && window.location.protocol === "https:" && API_URL.startsWith("http://")) {
    message += " Mixed-content is likely blocked (frontend HTTPS, backend HTTP).";
  }

  if (API_URL.includes("localhost")) {
    message += " If backend runs on another host/port, set NEXT_PUBLIC_API_URL in src/web/.env.local.";
  }

  return `${message} Original error: ${cause}`;
}

async function fetchApi(path: string, init: RequestInit): Promise<Response> {
  try {
    return await fetch(apiEndpoint(path), init);
  } catch (error) {
    throw new Error(networkFailureMessage(path, error));
  }
}

export interface ValidationWarning {
  type: "format" | "quality";
  severity: "warning" | "error";
  message: string;
}

export interface AnalysisConfig {
  bigbetter: 0 | 1;
  indicator_col?: string | null;
  selected_items?: string[];
  selected_indicator_values?: string[];
  bootstrap_iterations: number;
  random_seed: number;
}

export interface RankingItem {
  name: string;
  theta_hat: number;
  rank: number;
  ci_lower: number;
  ci_upper: number;
  ci_two_sided: [number, number];
}

export interface RankingMetadata {
  n_items: number;
  n_samples: number;
  k_methods: number;
  n_comparisons: number;
  heterogeneity_index: number;
  spectral_gap: number;
  sparsity_ratio: number;
  mean_ci_width_top_5: number;
  runtime_sec: number;
}

export interface RankingResults {
  items: RankingItem[];
  metadata: RankingMetadata;
  pairwise_matrix: Array<{
    item_a: string;
    item_b: string;
    win_rate_a: number;
    n_comparisons: number;
  }>;
  report?: string;
  section_questions?: {
    rankings: string[];
    insights: string[];
    score_distribution: string[];
    confidence_intervals: string[];
  };
}

export interface ExampleDataInfo {
  id: string;
  filename: string;
  title: string;
  description: string;
  format: "pointwise" | "pairwise" | "multiway";
}

export const EXAMPLE_DATASETS: ExampleDataInfo[] = [
  {
    id: "pairwise",
    filename: "example_data_pairwise.csv",
    title: "LLM Pairwise Comparison",
    description:
      "Head-to-head comparisons between AI chatbots. " +
      "Each row records one comparison where two models competed on a task. " +
      "The winner gets 1, the loser gets 0. " +
      "• Items to rank: Your Model, ChatGPT, Claude, Gemini, Llama, Qwen (6 AI models) " +
      "• Task types: code, math, writing (can filter rankings by task) " +
      "• 3,000 comparison records " +
      "• Goal: Find which AI performs best overall or per task type",
    format: "pairwise",
  },
  {
    id: "pointwise",
    filename: "example_data_pointwise.csv",
    title: "Model Performance Scores",
    description:
      "Performance scores from testing 6 machine learning models. " +
      "Each row is one test sample, showing how well each model performed (0-1 scale, higher is better). " +
      "• Items to rank: model_1 through model_6 (6 ML models) " +
      "• 165 test samples " +
      "• Values: Accuracy scores from 0 to 1 " +
      "• Goal: Find which model performs best across all test cases",
    format: "pointwise",
  },
  {
    id: "multiway",
    filename: "example_data_multiway.csv",
    title: "Horse Racing Results",
    description:
      "Finish positions from horse races. " +
      "Each row is one race, showing where each horse finished (1st, 2nd, 3rd, etc.). " +
      "• Items to rank: Horse_A through Horse_F (6 horses) " +
      "• Track types: grass, dirt (can filter rankings by track) " +
      "• 150 races " +
      "• Values: Finish positions (1 = first place, lower is better) " +
      "• Goal: Find which horse is the best overall or per track type",
    format: "multiway",
  },
];

async function parseResponse<T>(response: Response, fallbackMessage: string): Promise<T> {
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || payload.error || fallbackMessage);
  }
  return response.json();
}

export async function uploadFile(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetchApi("/api/upload", {
    method: "POST",
    body: formData,
  });
  return parseResponse<UploadResponse>(response, "Upload failed");
}

export async function loadExampleData(exampleId: string): Promise<UploadResponse> {
  const response = await fetchApi(`/api/upload/example/${exampleId}`, {
    method: "POST",
  });
  return parseResponse<UploadResponse>(response, "Failed to load example");
}

export async function getDataPreview(sessionId: string): Promise<DataPreview> {
  const response = await fetchApi(`/api/preview/${sessionId}`, {});
  return parseResponse<DataPreview>(response, "Failed to fetch data preview");
}

export async function inferSession(sessionId: string, userHints?: string): Promise<InferResponse> {
  const response = await fetchApi(`/api/sessions/${sessionId}/infer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_hints: userHints ?? null }),
  });
  return parseResponse<InferResponse>(response, "Failed to infer schema");
}

export async function confirmSession(
  sessionId: string,
  payload: {
    confirmed: boolean;
    confirmed_schema: SemanticSchema;
    user_modifications: string[];
    B: number;
    seed: number;
  }
): Promise<ConfirmResponse> {
  const response = await fetchApi(`/api/sessions/${sessionId}/confirm`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return parseResponse<ConfirmResponse>(response, "Failed to confirm schema");
}

export async function runSession(
  sessionId: string,
  payload: { selected_items?: string[]; selected_indicator_values?: string[] }
): Promise<RunResponse> {
  const response = await fetchApi(`/api/sessions/${sessionId}/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return parseResponse<RunResponse>(response, "Failed to run analysis");
}

export async function startRunSession(
  sessionId: string,
  payload: { selected_items?: string[]; selected_indicator_values?: string[] }
): Promise<RunStartResponse> {
  const response = await fetchApi(`/api/sessions/${sessionId}/run/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return parseResponse<RunStartResponse>(response, "Failed to start analysis job");
}

export async function getRunJobStatus(sessionId: string, jobId: string): Promise<RunJobStatusResponse> {
  const response = await fetchApi(`/api/sessions/${sessionId}/run/${jobId}`, {});
  return parseResponse<RunJobStatusResponse>(response, "Failed to fetch analysis job status");
}

export async function askQuestion(
  sessionId: string,
  question: string,
  quotes: QuotePayload[] = []
): Promise<QuestionResponse> {
  const response = await fetchApi(`/api/sessions/${sessionId}/question`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, quotes }),
  });
  return parseResponse<QuestionResponse>(response, "Failed to answer question");
}

export async function getSessionSnapshot(sessionId: string): Promise<SessionSnapshotResponse> {
  const response = await fetchApi(`/api/sessions/${sessionId}`, {});
  return parseResponse<SessionSnapshotResponse>(response, "Failed to fetch session snapshot");
}

export function artifactUrl(sessionId: string, artifactId: string): string {
  return `${API_URL}/api/sessions/${sessionId}/artifacts/${artifactId}`;
}

export async function deleteSession(sessionId: string): Promise<void> {
  const response = await fetchApi(`/api/sessions/${sessionId}`, {
    method: "DELETE",
  });
  await parseResponse(response, "Failed to delete session");
}

export async function checkHealth(): Promise<{ status: string; version: string; r_available: boolean }> {
  const response = await fetchApi("/health", {});
  return parseResponse(response, "Health check failed");
}

export function toValidationWarnings(
  formatResult?: FormatValidationResult,
  qualityResult?: QualityValidationResult
): ValidationWarning[] {
  const warnings: ValidationWarning[] = [];

  if (formatResult) {
    for (const issue of formatResult.issues) {
      warnings.push({ type: "format", severity: formatResult.fixable ? "warning" : "error", message: issue });
    }
  }

  if (qualityResult) {
    for (const warning of qualityResult.warnings) {
      warnings.push({ type: "quality", severity: "warning", message: warning });
    }
    for (const error of qualityResult.errors) {
      warnings.push({ type: "quality", severity: "error", message: error });
    }
  }

  return warnings;
}

export function normalizeRunResponse(run: RunResponse): {
  rankingResults: RankingResults | null;
  reportOutput: ReportOutput | null;
  plots: PlotSpec[];
} {
  if (!run.execution?.results) {
    return { rankingResults: null, reportOutput: run.report ?? null, plots: run.visualizations?.plots ?? [] };
  }

  const raw = run.execution.results;
  const items: RankingItem[] = raw.items.map((name, index) => ({
    // Keep CI bounds as integers for consistent UI interpretation.
    ci_lower: Math.round(raw.ci_lower[index]),
    ci_upper: Math.round(raw.ci_upper[index]),
    name,
    theta_hat: raw.theta_hat[index],
    rank: raw.ranks[index],
    ci_two_sided: [Math.round(raw.ci_lower[index]), Math.round(raw.ci_upper[index])],
  }));

  const rawMetadata = raw.metadata;
  const rankingMetadata: RankingMetadata = {
    n_items: items.length,
    n_samples: rawMetadata?.n_samples ?? 0,
    k_methods: rawMetadata?.k_methods ?? items.length,
    n_comparisons: rawMetadata?.n_comparisons ?? rawMetadata?.n_samples ?? 0,
    heterogeneity_index: rawMetadata?.heterogeneity_index ?? 0,
    spectral_gap: rawMetadata?.spectral_gap ?? 0,
    sparsity_ratio: rawMetadata?.sparsity_ratio ?? 0,
    mean_ci_width_top_5: rawMetadata?.mean_ci_width_top_5 ?? 0,
    runtime_sec: rawMetadata?.runtime_sec ?? run.execution.trace.duration_seconds,
  };

  const rankingResults: RankingResults = {
    items,
    metadata: rankingMetadata,
    pairwise_matrix: [],
    report: run.report?.markdown,
    section_questions: {
      rankings: [],
      insights: [],
      score_distribution: [],
      confidence_intervals: [],
    },
  };

  return {
    rankingResults,
    reportOutput: run.report ?? null,
    plots: run.visualizations?.plots ?? [],
  };
}
