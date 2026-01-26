/**
 * OmniRank API Client
 * Handles communication with the FastAPI backend.
 */

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";

// ============================================================================
// Types (imported from shared/types for type safety)
// ============================================================================

export interface InferredSchema {
  format: "pointwise" | "pairwise" | "multiway";
  bigbetter: 0 | 1;
  ranking_items: string[];
  indicator_col: string | null;
  indicator_values: string[];
}

export interface ValidationWarning {
  type: string;
  message: string;
  severity: "warning" | "error";
}

// Phase 1 response: file uploaded, Data Agent processing in background
export interface UploadResponse {
  session_id: string;
  filename: string;
}

export type DataAgentStartStatus = "started" | "already_started" | "already_completed";

export interface DataAgentStartResponse {
  session_id: string;
  status: DataAgentStartStatus;
}

// Phase 2: Data Agent results (received via WebSocket)
export interface SchemaReadyPayload {
  inferred_schema: InferredSchema;
  warnings: ValidationWarning[];
}

export interface AnalysisConfig {
  bigbetter: 0 | 1;
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
  step2_triggered: boolean;
  runtime_sec: number;
}

export interface SectionQuestions {
  rankings: string[];
  insights: string[];
  score_distribution: string[];
  confidence_intervals: string[];
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
  report?: string;  // LLM-generated analysis report (markdown)
  section_questions?: SectionQuestions;  // LLM-generated questions for each section
}

export interface AnalyzeResponse {
  status: "processing" | "completed" | "error";
  results?: RankingResults;
  error?: string;
}

// ============================================================================
// API Functions
// ============================================================================

/**
 * Upload a file for analysis.
 */
export async function uploadFile(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_URL}/api/upload`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Upload failed");
  }

  return response.json();
}

/**
 * Start Data Agent processing for a session.
 */
export async function startDataAgent(sessionId: string): Promise<DataAgentStartResponse> {
  const response = await fetch(`${API_URL}/api/data-agent/start`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ session_id: sessionId }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Failed to start Data Agent");
  }

  return response.json();
}

/**
 * Trigger analysis with user configuration.
 */
export async function analyze(
  sessionId: string,
  config: AnalysisConfig
): Promise<AnalyzeResponse> {
  const response = await fetch(`${API_URL}/api/analyze`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      session_id: sessionId,
      config,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Analysis failed");
  }

  return response.json();
}

/**
 * Get results for a session.
 */
export async function getResults(sessionId: string): Promise<AnalyzeResponse> {
  const response = await fetch(`${API_URL}/api/results/${sessionId}`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Failed to get results");
  }

  return response.json();
}

/**
 * Delete a session.
 */
export async function deleteSession(sessionId: string): Promise<void> {
  const response = await fetch(`${API_URL}/api/session/${sessionId}`, {
    method: "DELETE",
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Failed to delete session");
  }
}

/**
 * Check API health.
 */
export async function checkHealth(): Promise<{
  status: string;
  version: string;
  r_available: boolean;
}> {
  const response = await fetch(`${API_URL}/health`);
  return response.json();
}

/**
 * Chat response type.
 */
export interface ChatResponse {
  answer: string;
  agent: string;
}

/**
 * Example data metadata.
 */
export interface ExampleDataInfo {
  id: string;
  filename: string;
  title: string;
  description: string;
  format: "pointwise" | "pairwise" | "multiway";
}

/**
 * Data preview information.
 */
export interface DataPreview {
  columns: string[];
  rows: Array<Record<string, string | number>>;
  totalRows: number;
}

/**
 * Available example datasets.
 */
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

/**
 * Send a follow-up question about the analysis.
 */
export async function askQuestion(
  sessionId: string,
  question: string
): Promise<ChatResponse> {
  const response = await fetch(`${API_URL}/api/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      session_id: sessionId,
      question,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Failed to get answer");
  }

  return response.json();
}

/**
 * Send a general question without session context.
 * Used in pre-upload stage for questions about system and methodology.
 */
export async function askGeneralQuestion(
  question: string
): Promise<ChatResponse> {
  const response = await fetch(`${API_URL}/api/chat/general`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ question }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Failed to get answer");
  }

  return response.json();
}

/**
 * Load example data from the server.
 * This uses the FastAPI backend's example endpoint.
 */
export async function loadExampleData(exampleId: string): Promise<UploadResponse> {
  const response = await fetch(`${API_URL}/api/upload/example/${exampleId}`, {
    method: "POST",
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Failed to load example data");
  }

  return response.json();
}

/**
 * Fetch data preview for a session.
 */
export async function getDataPreview(sessionId: string): Promise<DataPreview> {
  const response = await fetch(`${API_URL}/api/preview/${sessionId}`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Failed to get data preview");
  }

  return response.json();
}

// ============================================================================
// WebSocket Connection
// ============================================================================

export type WSMessageHandler = (message: {
  type: string;
  payload: unknown;
}) => void;

/**
 * Create a WebSocket connection for real-time updates.
 */
export function createWebSocket(
  sessionId: string,
  onMessage: WSMessageHandler,
  onError?: (error: Event) => void,
  onClose?: () => void,
  onOpen?: () => void
): WebSocket {
  const ws = new WebSocket(`${WS_URL}/api/ws/${sessionId}`);

  ws.onopen = () => {
    onOpen?.();
  };

  ws.onmessage = (event) => {
    try {
      const message = JSON.parse(event.data);
      onMessage(message);
    } catch (e) {
      console.error("Failed to parse WebSocket message:", e);
    }
  };

  ws.onerror = (error) => {
    console.error("WebSocket error:", error);
    onError?.(error);
  };

  ws.onclose = () => {
    console.log("WebSocket closed");
    onClose?.();
  };

  return ws;
}
