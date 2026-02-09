export type LeaderboardMode = "arena" | "huggingface";

export interface SpectralMethod {
  name: string;
  theta_hat: number;
  rank: number;
  ci_two_sided: [number, number];
  ci_left: number;
  ci_uniform_left: number;
  benchmark_scores?: Record<string, number>;
  model_url?: string;
}

export interface LeaderboardDataset {
  mode: LeaderboardMode;
  methods: SpectralMethod[];
}

export interface ExampleArenaPreview {
  headers: string[];
  rows: string[][];
  rowCount: number;
  colCount: number;
  tasks: string[];
}

export interface LeaderboardPageData {
  huggingFace: LeaderboardDataset;
  arena: LeaderboardDataset;
  exampleArena: ExampleArenaPreview;
}

export const ARENA_BENCHMARK_LABELS = [
  "Creative Writing",
  "Math",
  "Instruction Following",
  "Coding",
  "Hard Prompt",
  "Longer Query",
  "Multi-turn",
] as const;

export type ArenaBenchmarkLabel = (typeof ARENA_BENCHMARK_LABELS)[number];

export const HF_BENCHMARK_LABELS = [
  "IFEval",
  "BBH",
  "MATH",
  "GPQA",
  "MUSR",
  "MMLU-Pro",
] as const;

export type HuggingFaceBenchmarkLabel = (typeof HF_BENCHMARK_LABELS)[number];

export const ARENA_LABEL_TO_VIRTUAL: Record<ArenaBenchmarkLabel, string> = {
  "Creative Writing": "creative_writing_bt_prob",
  Math: "math_bt_prob",
  "Instruction Following": "instruction_following_bt_prob",
  Coding: "coding_bt_prob",
  "Hard Prompt": "hard_prompt_bt_prob",
  "Longer Query": "longer_query_bt_prob",
  "Multi-turn": "multi_turn_bt_prob",
};

export const ARENA_VIRTUAL_TO_FIELD: Record<string, string> = {
  creative_writing_bt_prob: "creative_writing",
  math_bt_prob: "math",
  instruction_following_bt_prob: "instruction_following",
  coding_bt_prob: "coding",
  hard_prompt_bt_prob: "hard_prompt",
  longer_query_bt_prob: "longer_query",
  multi_turn_bt_prob: "multi_turn",
};

export const HF_LABEL_TO_KEY: Record<HuggingFaceBenchmarkLabel, string> = {
  IFEval: "ifeval",
  BBH: "bbh",
  MATH: "math",
  GPQA: "gpqa",
  MUSR: "musr",
  "MMLU-Pro": "mmlu_pro",
};

export const HF_KEY_TO_LABEL: Record<string, HuggingFaceBenchmarkLabel> = {
  ifeval: "IFEval",
  bbh: "BBH",
  math: "MATH",
  gpqa: "GPQA",
  musr: "MUSR",
  mmlu_pro: "MMLU-Pro",
};

export const HF_KEY_ORDER = ["ifeval", "bbh", "math", "gpqa", "musr", "mmlu_pro"] as const;

