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
  confidence: number;
}

export interface ValidationWarning {
  type: string;
  message: string;
  severity: "warning" | "error";
}

export interface UploadResponse {
  session_id: string;
  filename: string;
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

export interface RankingResults {
  items: RankingItem[];
  metadata: RankingMetadata;
  pairwise_matrix: Array<{
    item_a: string;
    item_b: string;
    win_rate_a: number;
    n_comparisons: number;
  }>;
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
  onClose?: () => void
): WebSocket {
  const ws = new WebSocket(`${WS_URL}/api/ws/${sessionId}`);

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
