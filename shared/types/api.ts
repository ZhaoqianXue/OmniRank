/**
 * OmniRank Shared API Contracts
 * Contract-first shared types for frontend and backend.
 */

export type ComparisonFormat = "pointwise" | "pairwise" | "multiway";

export interface DataSummary {
  columns: string[];
  sample_rows: Array<Record<string, unknown>>;
  row_count: number;
  column_types: Record<string, "numeric" | "categorical" | "datetime" | "unknown">;
}

export interface ReadDataFileResult {
  success: boolean;
  data?: DataSummary;
  error?: string;
}

export interface SemanticSchema {
  bigbetter: 0 | 1;
  ranking_items: string[];
  indicator_col: string | null;
  indicator_values: string[];
}

export interface SemanticSchemaResult {
  success: boolean;
  format: ComparisonFormat;
  format_evidence: string;
  schema?: SemanticSchema;
  error?: string;
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

export interface PreprocessResult {
  preprocessed_csv_path: string;
  transformation_log: string[];
  row_count: number;
  dropped_rows: number;
}

export interface EngineConfig {
  csv_path: string;
  bigbetter: 0 | 1;
  selected_items?: string[];
  selected_indicator_values?: string[];
  B: number;
  seed: number;
  r_script_path: string;
}

export interface RankingMetadata {
  n_samples: number;
  k_methods: number;
  runtime_sec: number;
  heterogeneity_index: number;
  spectral_gap: number;
  sparsity_ratio: number;
  mean_ci_width_top_5: number;
  n_comparisons?: number | null;
}

export interface RankingResults {
  items: string[];
  theta_hat: number[];
  ranks: number[];
  ci_lower: number[];
  ci_upper: number[];
  indicator_value?: string | null;
  metadata?: RankingMetadata | null;
}

export interface ExecutionTrace {
  command: string;
  stdout: string;
  stderr: string;
  exit_code: number;
  duration_seconds: number;
  timestamp: string;
}

export interface ExecutionResult {
  success: boolean;
  results?: RankingResults;
  error?: string;
  trace: ExecutionTrace;
}

export interface ToolCallRecord {
  tool_name: string;
  inputs: Record<string, unknown>;
  outputs: Record<string, unknown>;
  timestamp: string;
  success: boolean;
  error?: string;
}

export interface SessionMemorySnapshot {
  session_id: string;
  status:
    | "idle"
    | "uploaded"
    | "inferred"
    | "awaiting_confirmation"
    | "confirmed"
    | "running"
    | "completed"
    | "error";
  original_file_path?: string | null;
  current_file_path?: string | null;
  filename?: string | null;
  data_summary?: DataSummary | null;
  inferred_schema?: SemanticSchema | null;
  format_validation_result?: FormatValidationResult | null;
  quality_validation_result?: QualityValidationResult | null;
  confirmed_schema?: SemanticSchema | null;
  config?: EngineConfig | null;
  current_results?: RankingResults | null;
  execution_trace: ExecutionTrace[];
  tool_call_history: ToolCallRecord[];
  report_output?: ReportOutput | null;
  visualization_output?: VisualizationOutput | null;
  citation_blocks?: CitationBlock[];
  created_at: string;
  updated_at: string;
  error?: string | null;
}

export type SessionMemory = SessionMemorySnapshot;

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

export interface ArtifactRef {
  kind: string;
  path: string;
  title: string;
  mime_type: string;
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

export interface QuotePayload {
  quoted_text: string;
  block_id?: string;
  kind?: string;
  source: "report" | "user_upload" | "external";
}

export interface VisualizationOutput {
  plots: PlotSpec[];
  errors: string[];
}

export interface ReportOutput {
  markdown: string;
  key_findings: Record<string, unknown>;
  artifacts: ArtifactRef[];
  hints: HintSpec[];
  citation_blocks: CitationBlock[];
}

export interface AnswerOutput {
  answer: string;
  supporting_evidence: string[];
  used_citation_block_ids: string[];
}

export interface ConfirmationResult {
  confirmed: boolean;
  confirmed_schema: SemanticSchema;
  user_modifications: string[];
  B: number;
  seed: number;
}

export interface DataPreview {
  columns: string[];
  rows: Array<Record<string, string | number | null>>;
  totalRows: number;
}

export interface UploadResponse {
  session_id: string;
  filename: string;
}

export interface InferRequest {
  user_hints?: string;
}

export interface InferResponse {
  success: boolean;
  data_summary?: DataSummary;
  schema_result?: SemanticSchemaResult;
  format_result?: FormatValidationResult;
  quality_result?: QualityValidationResult;
  preprocessed_path?: string;
  requires_confirmation: boolean;
  error?: string;
}

export interface ConfirmRequest {
  confirmed: boolean;
  confirmed_schema: SemanticSchema;
  user_modifications?: string[];
  B?: number;
  seed?: number;
}

export interface ConfirmResponse {
  confirmation: ConfirmationResult;
  session_status: SessionMemorySnapshot["status"];
}

export interface RunRequest {
  selected_items?: string[];
  selected_indicator_values?: string[];
}

export interface RunResponse {
  success: boolean;
  config?: EngineConfig;
  execution?: ExecutionResult;
  visualizations?: VisualizationOutput;
  report?: ReportOutput;
  error?: string;
}

export type RunJobStatus = "queued" | "running" | "completed" | "failed";

export interface RunStartResponse {
  job_id: string;
  status: RunJobStatus;
  progress: number;
  message: string;
}

export interface RunJobStatusResponse {
  job_id: string;
  session_id: string;
  status: RunJobStatus;
  progress: number;
  message: string;
  result?: RunResponse;
  error?: string;
}

export interface QuestionRequest {
  session_id?: string;
  question: string;
  quotes?: QuotePayload[];
}

export interface QuestionResponse {
  answer: AnswerOutput;
}

export interface ArtifactDescriptor {
  artifact_id: string;
  kind: string;
  title: string;
  mime_type: string;
}

export interface SessionSnapshotResponse {
  session: SessionMemorySnapshot;
  artifacts: ArtifactDescriptor[];
}
