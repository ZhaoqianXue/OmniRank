/**
 * OmniRank API client (SOP single-agent pipeline).
 */

import type {
  ArtifactDescriptor,
  ConfirmResponse,
  DataPreview,
  DataSummary,
  DailyUsageResponse,
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
  DailyUsageResponse,
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
let activeUserSub: string | null = null;

function apiEndpoint(path: string): string {
  return `${API_URL}${path}`;
}

function withUserHeader(init: RequestInit): RequestInit {
  const headers = new Headers(init.headers ?? undefined);
  if (activeUserSub) {
    headers.set("X-OmniRank-User-Sub", activeUserSub);
  }
  return { ...init, headers };
}

export function setApiUserSub(userSub: string | null | undefined): void {
  const normalized = userSub?.trim();
  activeUserSub = normalized ? normalized : null;
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
    return await fetch(apiEndpoint(path), withUserHeader(init));
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
  ranking_mode?: "flash" | "deep";
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
  format: "pairwise" | "multiway";
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
    id: "pairwise_human_logs",
    filename: "example_data_pairwise_human_logs.csv",
    title: "Human Preference Logs",
    description:
      "Raw human preference logs in long format. " +
      "Each comparison has two rows (one per assistant) with value=1 for winner and value=0 for loser. " +
      "• Items to rank: Orion, Astra, Nova, Pulse, Zenith, Nimbus (6 assistants) " +
      "• Task types: coding, math, reasoning, safety " +
      "• 5,040 comparisons / 10,080 log rows " +
      "• OmniRank auto-preprocesses long logs to ranking-ready pairwise matrix " +
      "• Goal: Rank assistants from preference outcomes",
    format: "pairwise",
  },
  {
    id: "multiway_phenotype",
    filename: "example_data_multiway_phenotype.csv",
    title: "PRS Phenotype Matrix",
    description:
      "Trait-stratified count matrix for PRS methods. " +
      "Each row is one phenotype with method counts (number of GWAS per method). " +
      "• Items to rank: C+T, SCT, LDpred, LDpred2, LDpred2-auto, LDpred2-inf, LDpred-funct, AnnoPred, lassosum, lassosum2, PRS-CS, PRS-CS-auto, SBayesR, DBSLMM " +
      "• Indicator: phenotype (33 traits) " +
      "• 33 rows " +
      "• Supports Flash ranking (overall) and Deep ranking (phenotype-wise)",
    format: "multiway",
  },
  {
    id: "multiway_scores",
    filename: "example_data_multiway_scores.csv",
    title: "Model Performance Matrix",
    description:
      "Side-by-side performance comparisons across 6 machine learning models on shared samples. " +
      "Each row is one sample with all model values (0-1 scale, higher is better). " +
      "• Items to rank: model_1 through model_6 (6 ML models) " +
      "• 164 test samples " +
      "• Values: Per-sample evaluation metrics from 0 to 1 " +
      "• Goal: Find which model performs best overall",
    format: "multiway",
  },
  {
    id: "multiway_latency",
    filename: "example_data_multiway_latency.csv",
    title: "System Latency Matrix",
    description:
      "Latency benchmarks across six systems under multiple workload scenarios. " +
      "Each row contains all system latencies for one workload (milliseconds, lower is better). " +
      "• Items to rank: latency_alpha through latency_zeta (6 systems) " +
      "• Scenarios: short_context, medium_context, long_context, tool_heavy " +
      "• 920 workloads " +
      "• Goal: Identify the fastest system overall and by scenario",
    format: "multiway",
  },
  {
    id: "multiway_rank_columns",
    filename: "example_data_multiway_rank_columns.csv",
    title: "Rank-Order Tournaments",
    description:
      "Tournament outcomes stored as rank_1 to rank_6 columns. " +
      "Each row records a full ordering of six candidates for one match. " +
      "• Items to rank: Alpha, Beta, Gamma, Delta, Epsilon, Zeta " +
      "• Domains: coding, math, reasoning, safety " +
      "• 560 matches " +
      "• Goal: Estimate robust ordering from full rank outcomes",
    format: "multiway",
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
      "• 620 races " +
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

export async function getDailyUsage(): Promise<DailyUsageResponse> {
  const response = await fetchApi("/api/usage/daily", {});
  return parseResponse<DailyUsageResponse>(response, "Failed to fetch daily usage");
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
  payload: { selected_items?: string[]; selected_indicator_values?: string[]; ranking_mode?: "flash" | "deep" }
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
  payload: { selected_items?: string[]; selected_indicator_values?: string[]; ranking_mode?: "flash" | "deep" }
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
  sessionId: string | null,
  question: string,
  quotes: QuotePayload[] = []
): Promise<QuestionResponse> {
  const path = sessionId ? `/api/sessions/${sessionId}/question` : "/api/question";
  const response = await fetchApi(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId ?? undefined, question, quotes }),
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
