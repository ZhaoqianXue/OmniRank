/**
 * OmniRank API client (SOP single-agent pipeline).
 */

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

export interface SemanticSchema {
  bigbetter: 0 | 1;
  ranking_items: string[];
  indicator_col: string | null;
  indicator_values: string[];
}

export interface DataSummary {
  columns: string[];
  sample_rows: Array<Record<string, unknown>>;
  row_count: number;
  column_types: Record<string, string>;
}

export interface FormatValidationResult {
  is_ready: boolean;
  fixable: boolean;
  issues: string[];
  suggested_fixes: string[];
}

export interface QualityValidationResult {
  is_valid: boolean;
  warnings: string[];
  errors: string[];
}

export interface ValidationWarning {
  type: "format" | "quality";
  severity: "warning" | "error";
  message: string;
}

export interface UploadResponse {
  session_id: string;
  filename: string;
}

export interface DataPreview {
  columns: string[];
  rows: Array<Record<string, string | number | null>>;
  totalRows: number;
}

export interface InferResponse {
  success: boolean;
  data_summary?: DataSummary;
  schema_result?: {
    success: boolean;
    format: "pointwise" | "pairwise" | "multiway";
    format_evidence: string;
    schema?: SemanticSchema;
    error?: string;
  };
  format_result?: FormatValidationResult;
  quality_result?: QualityValidationResult;
  preprocessed_path?: string;
  requires_confirmation: boolean;
  error?: string;
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
  n_comparisons: number;
  heterogeneity_index: number;
  sparsity_ratio: number;
  runtime_sec: number;
}

export interface PlotSpec {
  type: string;
  data: Record<string, unknown>;
  config: Record<string, unknown>;
  svg_path: string;
  block_id: string;
  caption_plain: string;
  caption_academic: string;
  hint_ids: string[];
}

export interface HintSpec {
  hint_id: string;
  title: string;
  body: string;
  kind: "definition" | "assumption" | "caveat" | "method";
  sources: string[];
}

export interface CitationBlock {
  block_id: string;
  kind: "summary" | "result" | "comparison" | "method" | "limitation" | "repro" | "figure" | "table";
  markdown: string;
  text: string;
  hint_ids: string[];
  artifact_paths: string[];
}

export interface ReportOutput {
  markdown: string;
  key_findings: Record<string, unknown>;
  artifacts: Array<{ kind: string; path: string; title: string; mime_type: string }>;
  hints: HintSpec[];
  citation_blocks: CitationBlock[];
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

export interface RunResponse {
  success: boolean;
  config?: {
    csv_path: string;
    bigbetter: 0 | 1;
    selected_items?: string[];
    selected_indicator_values?: string[];
    B: number;
    seed: number;
    r_script_path: string;
  };
  execution?: {
    success: boolean;
    results?: {
      items: string[];
      theta_hat: number[];
      ranks: number[];
      ci_lower: number[];
      ci_upper: number[];
      indicator_value?: string | null;
    };
    error?: string;
    trace: {
      command: string;
      stdout: string;
      stderr: string;
      exit_code: number;
      duration_seconds: number;
      timestamp: string;
    };
  };
  visualizations?: {
    plots: PlotSpec[];
    errors: string[];
  };
  report?: ReportOutput;
  error?: string;
}

export interface QuotePayload {
  quoted_text: string;
  block_id?: string;
  kind?: string;
  source: "report" | "user_upload" | "external";
}

export interface ArtifactDescriptor {
  artifact_id: string;
  kind: string;
  title: string;
  mime_type: string;
}

export interface QuestionResponse {
  answer: {
    answer: string;
    supporting_evidence: string[];
    used_citation_block_ids: string[];
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
): Promise<{ confirmation: unknown; session_status: string }> {
  const response = await fetchApi(`/api/sessions/${sessionId}/confirm`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return parseResponse(response, "Failed to confirm schema");
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

export async function getSessionSnapshot(sessionId: string): Promise<{
  session: {
    status: string;
    report_output?: ReportOutput;
    visualization_output?: { plots: PlotSpec[]; errors: string[] };
    citation_blocks?: CitationBlock[];
  };
  artifacts: ArtifactDescriptor[];
}> {
  const response = await fetchApi(`/api/sessions/${sessionId}`, {});
  return parseResponse(response, "Failed to fetch session snapshot");
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
    name,
    theta_hat: raw.theta_hat[index],
    rank: raw.ranks[index],
    ci_lower: raw.ci_lower[index],
    ci_upper: raw.ci_upper[index],
    ci_two_sided: [raw.ci_lower[index], raw.ci_upper[index]],
  }));

  const rankingResults: RankingResults = {
    items,
    metadata: {
      n_items: items.length,
      n_comparisons: 0,
      heterogeneity_index: 0,
      sparsity_ratio: 0,
      runtime_sec: run.execution.trace.duration_seconds,
    },
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
