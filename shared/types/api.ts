/**
 * OmniRank API Type Definitions
 * Contract-First: These types define the API contract between frontend and backend.
 */

// ============================================================================
// Common Types
// ============================================================================

export type DataFormat = "pointwise" | "pairwise" | "multiway";

export interface InferredSchema {
  format: DataFormat;
  bigbetter: 0 | 1;
  ranking_items: string[];
  indicator_col: string | null;
  indicator_values: string[];
  confidence: number;
}

export interface ValidationWarning {
  type: "sparsity" | "connectivity" | "format";
  message: string;
  severity: "warning" | "error";
}

// ============================================================================
// Upload API
// ============================================================================

export interface UploadResponse {
  session_id: string;
  filename: string;
  inferred_schema: InferredSchema;
  warnings: ValidationWarning[];
}

// ============================================================================
// Data Agent API
// ============================================================================

export type DataAgentStartStatus = "started" | "already_started" | "already_completed";

export interface DataAgentStartRequest {
  session_id: string;
}

export interface DataAgentStartResponse {
  session_id: string;
  status: DataAgentStartStatus;
}

// ============================================================================
// Analyze API
// ============================================================================

export interface AnalysisConfig {
  bigbetter: 0 | 1;
  selected_items?: string[];
  selected_indicator_values?: string[];
  bootstrap_iterations: number;
  random_seed: number;
}

export interface AnalyzeRequest {
  session_id: string;
  config: AnalysisConfig;
}

export type AnalysisStatus = "processing" | "completed" | "error";

export interface AnalyzeResponse {
  status: AnalysisStatus;
  results?: RankingResults;
  error?: string;
}

// ============================================================================
// Ranking Results
// ============================================================================

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

export interface PairwiseComparison {
  item_a: string;
  item_b: string;
  win_rate_a: number;
  n_comparisons: number;
}

export interface RankingResults {
  items: RankingItem[];
  metadata: RankingMetadata;
  pairwise_matrix: PairwiseComparison[];
}

// ============================================================================
// WebSocket Messages
// ============================================================================

export type WSMessageType = "progress" | "agent_message" | "result" | "error";
export type AgentType = "data" | "orchestrator" | "analyst";

export interface WSProgressPayload {
  progress: number;
  message: string;
}

export interface WSAgentMessagePayload {
  agent: AgentType;
  message: string;
  thinking?: string;
}

export interface WSResultPayload {
  results: RankingResults;
}

export interface WSErrorPayload {
  error: string;
  agent?: AgentType;
}

export interface WSMessage {
  type: WSMessageType;
  payload: WSProgressPayload | WSAgentMessagePayload | WSResultPayload | WSErrorPayload;
}

// ============================================================================
// Chat Messages
// ============================================================================

export type MessageRole = "user" | "assistant" | "system";

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: number;
  agent?: AgentType;
}

// ============================================================================
// Session State
// ============================================================================

export interface SessionState {
  session_id: string;
  filename: string | null;
  schema: InferredSchema | null;
  config: AnalysisConfig | null;
  results: RankingResults | null;
  messages: ChatMessage[];
  status: "idle" | "uploading" | "configuring" | "analyzing" | "completed" | "error";
}
