import fs from "node:fs";
import path from "node:path";
import {
  ARENA_BENCHMARK_LABELS,
  ARENA_LABEL_TO_VIRTUAL,
  ARENA_VIRTUAL_TO_FIELD,
  HF_KEY_ORDER,
  type ExampleArenaPreview,
  type LeaderboardPageData,
  type SpectralMethod,
} from "@/lib/leaderboard-types";

interface BenchmarkMatrix {
  benchmarkKeys: string[];
  modelNames: string[];
  values: number[][]; // modelIndex x benchmarkIndex
}

interface RawArenaMethod {
  model: string;
  theta_hat: number;
  rank: number;
  ci_two_left: number;
  ci_two_right: number;
  ci_left: number;
  ci_uniform_left: number;
}

interface RawHfResponse {
  methods: Array<Record<string, unknown>>;
}

const jsonCache = new Map<string, unknown>();
const csvCache = new Map<string, string[][]>();

const REPO_ROOT = detectRepoRoot();
const LEGACY_ROOT = path.join(REPO_ROOT, "Ranking");

const HF_BASE_FILE = path.join(
  LEGACY_ROOT,
  "data_llm",
  "data_huggingface",
  "data_ranking",
  "current",
  "huggingface_ranking_result_enhanced.json",
);

const HF_COMBINATIONS_DIR = path.join(
  LEGACY_ROOT,
  "data_llm",
  "data_huggingface",
  "data_ranking",
  "current",
  "all_combinations",
);

const HF_MATRIX_FILE = path.join(
  LEGACY_ROOT,
  "data_llm",
  "data_huggingface",
  "data_processing",
  "huggingface_processed_top100.csv",
);

const ARENA_BASE_FILE = path.join(
  LEGACY_ROOT,
  "data_llm",
  "data_arena",
  "data_ranking",
  "current",
  "ranking_results.json",
);

const ARENA_COMBINATIONS_DIR = path.join(
  LEGACY_ROOT,
  "data_llm",
  "data_arena",
  "data_ranking",
  "current",
  "all_combinations",
);

const ARENA_MATRIX_FILE = path.join(
  LEGACY_ROOT,
  "data_llm",
  "data_arena",
  "data_processing",
  "arena_elo_full.csv",
);

const EXAMPLE_ARENA_FILE = path.join(LEGACY_ROOT, "demo_r", "example_arena_style.csv");

let cachedHfBaseMethods: SpectralMethod[] | null = null;
let cachedArenaBaseMethods: SpectralMethod[] | null = null;
let cachedHfMatrix: BenchmarkMatrix | null = null;
let cachedArenaMatrix: BenchmarkMatrix | null = null;
let cachedExampleArenaPreview: ExampleArenaPreview | null = null;

function detectRepoRoot(): string {
  const cwd = process.cwd();
  const candidates = [cwd, path.resolve(cwd, ".."), path.resolve(cwd, "..", "..")];

  for (const candidate of candidates) {
    const hasRanking = fs.existsSync(path.join(candidate, "Ranking"));
    const hasWebApp = fs.existsSync(path.join(candidate, "src", "web", "app"));
    if (hasRanking && hasWebApp) {
      return candidate;
    }
  }

  return path.resolve(cwd, "..", "..");
}

function parseCsv(content: string): string[][] {
  const rows: string[][] = [];
  let currentRow: string[] = [];
  let currentCell = "";
  let inQuotes = false;

  const normalized = content.replace(/^\uFEFF/, "");

  for (let i = 0; i < normalized.length; i += 1) {
    const char = normalized[i];

    if (char === '"') {
      if (inQuotes && normalized[i + 1] === '"') {
        currentCell += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (char === "," && !inQuotes) {
      currentRow.push(currentCell);
      currentCell = "";
      continue;
    }

    if ((char === "\n" || char === "\r") && !inQuotes) {
      if (char === "\r" && normalized[i + 1] === "\n") {
        i += 1;
      }

      currentRow.push(currentCell);
      if (!(currentRow.length === 1 && currentRow[0] === "")) {
        rows.push(currentRow);
      }

      currentRow = [];
      currentCell = "";
      continue;
    }

    currentCell += char;
  }

  if (currentCell.length > 0 || currentRow.length > 0) {
    currentRow.push(currentCell);
    rows.push(currentRow);
  }

  return rows;
}

function readCsvRows(filePath: string): string[][] {
  const cached = csvCache.get(filePath);
  if (cached) {
    return cached;
  }

  const content = fs.readFileSync(filePath, "utf8");
  const rows = parseCsv(content);
  csvCache.set(filePath, rows);
  return rows;
}

function readJsonFile<T>(filePath: string): T {
  const cached = jsonCache.get(filePath);
  if (cached !== undefined) {
    return cached as T;
  }

  const parsed = JSON.parse(fs.readFileSync(filePath, "utf8")) as T;
  jsonCache.set(filePath, parsed);
  return parsed;
}

function toNumber(value: unknown, fallback = 0): number {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }

  if (typeof value === "string") {
    const parsed = Number.parseFloat(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }

  return fallback;
}

function fuzzyFindModelIndex(modelName: string, modelNames: string[]): number {
  const exact = modelNames.indexOf(modelName);
  if (exact >= 0) {
    return exact;
  }

  if (modelName.includes("...")) {
    const base = modelName.split("...")[0] ?? "";
    const startsWith = modelNames.findIndex((name) => name.startsWith(base));
    if (startsWith >= 0) {
      return startsWith;
    }
  }

  if (modelName.includes("Linkbricks-Horizon-AI-Ave")) {
    const index = modelNames.findIndex((name) => name.includes("Linkbricks-Horizon-AI-Ave"));
    if (index >= 0) {
      return index;
    }
  }

  if (modelName.includes("NQLSG-Qwen2.5-14B-MegaFus")) {
    const index = modelNames.findIndex((name) => name.includes("NQLSG-Qwen2.5-14B-MegaFus"));
    if (index >= 0) {
      return index;
    }
  }

  if (modelName.includes("Qwen2.5-72B-Instruct-abli")) {
    const index = modelNames.findIndex((name) => name.includes("Qwen2.5-72B-Instruct"));
    if (index >= 0) {
      return index;
    }
  }

  if (modelName.length > 10) {
    const startsWithEither = modelNames.findIndex(
      (name) => modelName.startsWith(name) || name.startsWith(modelName),
    );
    if (startsWithEither >= 0) {
      return startsWithEither;
    }
  }

  return -1;
}

function cloneMethod(method: SpectralMethod): SpectralMethod {
  return {
    ...method,
    benchmark_scores: method.benchmark_scores ? { ...method.benchmark_scores } : undefined,
  };
}

function sortMethodsByRank(methods: SpectralMethod[]): SpectralMethod[] {
  return methods.slice().sort((a, b) => a.rank - b.rank);
}

function loadHfMatrix(): BenchmarkMatrix {
  if (cachedHfMatrix) {
    return cachedHfMatrix;
  }

  const rows = readCsvRows(HF_MATRIX_FILE);
  if (rows.length < 2) {
    throw new Error("Hugging Face benchmark matrix is empty.");
  }

  const header = rows[0] ?? [];
  const modelNames = header.slice(1);

  const benchmarkKeys: string[] = [];
  const values: number[][] = modelNames.map(() => []);

  for (const row of rows.slice(1)) {
    const benchmarkKey = (row[0] ?? "").trim();
    if (!benchmarkKey) {
      continue;
    }

    benchmarkKeys.push(benchmarkKey);
    for (let modelIndex = 0; modelIndex < modelNames.length; modelIndex += 1) {
      const value = toNumber(row[modelIndex + 1], 0);
      values[modelIndex]?.push(value);
    }
  }

  cachedHfMatrix = { benchmarkKeys, modelNames, values };
  return cachedHfMatrix;
}

function loadArenaMatrix(): BenchmarkMatrix {
  if (cachedArenaMatrix) {
    return cachedArenaMatrix;
  }

  const rows = readCsvRows(ARENA_MATRIX_FILE);
  if (rows.length < 2) {
    throw new Error("Arena benchmark matrix is empty.");
  }

  const header = rows[0] ?? [];
  const modelNames = header.slice(1);

  const benchmarkKeys: string[] = [];
  const values: number[][] = modelNames.map(() => []);

  for (const row of rows.slice(1)) {
    const benchmarkKey = (row[0] ?? "").trim();
    if (!benchmarkKey) {
      continue;
    }

    benchmarkKeys.push(benchmarkKey);
    for (let modelIndex = 0; modelIndex < modelNames.length; modelIndex += 1) {
      const value = toNumber(row[modelIndex + 1], 0);
      values[modelIndex]?.push(value);
    }
  }

  cachedArenaMatrix = { benchmarkKeys, modelNames, values };
  return cachedArenaMatrix;
}

function normalizeHfMethod(item: Record<string, unknown>): SpectralMethod {
  const ciTwoSidedRaw = item.ci_two_sided;
  const ciTwoSided = Array.isArray(ciTwoSidedRaw) && ciTwoSidedRaw.length >= 2
    ? [toNumber(ciTwoSidedRaw[0]), toNumber(ciTwoSidedRaw[1])]
    : [toNumber(item.ci_two_left), toNumber(item.ci_two_right)];

  const benchmarkScoresRaw = item.benchmark_scores;
  const benchmark_scores: Record<string, number> = {};

  if (benchmarkScoresRaw && typeof benchmarkScoresRaw === "object") {
    for (const [key, value] of Object.entries(benchmarkScoresRaw)) {
      benchmark_scores[key] = toNumber(value, 0);
    }
  }

  return {
    name: String(item.name ?? "unknown"),
    theta_hat: toNumber(item.theta_hat),
    rank: Math.max(1, Math.round(toNumber(item.rank, 1))),
    ci_two_sided: [Math.round(ciTwoSided[0]), Math.round(ciTwoSided[1])],
    ci_left: Math.round(toNumber(item.ci_left, 1)),
    ci_uniform_left: Math.round(toNumber(item.ci_uniform_left, 1)),
    model_url: typeof item.model_url === "string" ? item.model_url : undefined,
    benchmark_scores: Object.keys(benchmark_scores).length > 0 ? benchmark_scores : undefined,
  };
}

function normalizeArenaMethod(item: RawArenaMethod): SpectralMethod {
  return {
    name: item.model,
    theta_hat: toNumber(item.theta_hat),
    rank: Math.max(1, Math.round(toNumber(item.rank, 1))),
    ci_two_sided: [Math.round(toNumber(item.ci_two_left, 1)), Math.round(toNumber(item.ci_two_right, 1))],
    ci_left: Math.round(toNumber(item.ci_left, 1)),
    ci_uniform_left: Math.round(toNumber(item.ci_uniform_left, 1)),
  };
}

function attachHfBenchmarkScores(
  methods: SpectralMethod[],
  selectedKeys: string[],
  baseMethods?: SpectralMethod[],
): SpectralMethod[] {
  const matrix = loadHfMatrix();
  const benchmarkIndex = new Map(matrix.benchmarkKeys.map((key, idx) => [key, idx]));
  const selectedIndices = selectedKeys
    .map((key) => benchmarkIndex.get(key))
    .filter((value): value is number => value !== undefined);

  const baseMethodsSafe = baseMethods ?? [];
  const baseNames = baseMethodsSafe.map((method) => method.name);

  return methods.map((method) => {
    const next = cloneMethod(method);
    const modelIndex = fuzzyFindModelIndex(next.name, matrix.modelNames);

    if (modelIndex >= 0) {
      const scores: Record<string, number> = {};

      for (const key of HF_KEY_ORDER) {
        const keyIndex = benchmarkIndex.get(key);
        if (keyIndex !== undefined) {
          scores[key] = matrix.values[modelIndex]?.[keyIndex] ?? 0;
        }
      }

      const selectedValues = selectedIndices
        .map((index) => matrix.values[modelIndex]?.[index])
        .filter((value): value is number => Number.isFinite(value));

      const average = selectedValues.length > 0
        ? selectedValues.reduce((sum, value) => sum + value, 0) / selectedValues.length
        : 0;

      scores.average_score = average;
      next.benchmark_scores = scores;
    }

    if (!next.model_url && baseMethodsSafe.length > 0) {
      const baseIndex = fuzzyFindModelIndex(next.name, baseNames);
      if (baseIndex >= 0) {
        next.model_url = baseMethodsSafe[baseIndex]?.model_url;
      }
    }

    return next;
  });
}

function attachArenaBenchmarkScores(methods: SpectralMethod[], selectedVirtualKeys: string[]): SpectralMethod[] {
  const matrix = loadArenaMatrix();
  const benchmarkIndex = new Map(matrix.benchmarkKeys.map((key, idx) => [key, idx]));
  const selectedIndices = selectedVirtualKeys
    .map((key) => benchmarkIndex.get(key))
    .filter((value): value is number => value !== undefined);

  return methods.map((method) => {
    const next = cloneMethod(method);
    const modelIndex = fuzzyFindModelIndex(next.name, matrix.modelNames);

    if (modelIndex >= 0) {
      const scores: Record<string, number> = {};

      for (const [virtualKey, fieldKey] of Object.entries(ARENA_VIRTUAL_TO_FIELD)) {
        const keyIndex = benchmarkIndex.get(virtualKey);
        if (keyIndex !== undefined) {
          scores[fieldKey] = matrix.values[modelIndex]?.[keyIndex] ?? 0;
        }
      }

      const selectedValues = selectedIndices
        .map((index) => matrix.values[modelIndex]?.[index])
        .filter((value): value is number => Number.isFinite(value));

      const average = selectedValues.length > 0
        ? selectedValues.reduce((sum, value) => sum + value, 0) / selectedValues.length
        : 0;

      scores.average_score = average;
      next.benchmark_scores = scores;
    }

    return next;
  });
}

export function loadHuggingFaceBaseMethods(): SpectralMethod[] {
  if (cachedHfBaseMethods) {
    return cachedHfBaseMethods.map(cloneMethod);
  }

  const response = readJsonFile<RawHfResponse>(HF_BASE_FILE);
  const methods = Array.isArray(response.methods)
    ? response.methods.map((item) => normalizeHfMethod(item))
    : [];

  const hasBenchmarks = methods.some((method) => method.benchmark_scores && method.benchmark_scores.average_score !== undefined);
  const normalized = hasBenchmarks
    ? sortMethodsByRank(methods)
    : sortMethodsByRank(attachHfBenchmarkScores(methods, [...HF_KEY_ORDER]));

  cachedHfBaseMethods = normalized;
  return cachedHfBaseMethods.map(cloneMethod);
}

export function loadArenaBaseMethods(): SpectralMethod[] {
  if (cachedArenaBaseMethods) {
    return cachedArenaBaseMethods.map(cloneMethod);
  }

  const rawMethods = readJsonFile<RawArenaMethod[]>(ARENA_BASE_FILE);
  const methods = Array.isArray(rawMethods) ? rawMethods.map(normalizeArenaMethod) : [];

  const selectedVirtualKeys = ARENA_BENCHMARK_LABELS.map((label) => ARENA_LABEL_TO_VIRTUAL[label]);
  const normalized = sortMethodsByRank(attachArenaBenchmarkScores(methods, selectedVirtualKeys));

  cachedArenaBaseMethods = normalized;
  return cachedArenaBaseMethods.map(cloneMethod);
}

function readCombinationMethods(filePath: string): SpectralMethod[] {
  const rawMethods = readJsonFile<RawArenaMethod[]>(filePath);
  if (!Array.isArray(rawMethods)) {
    return [];
  }

  return rawMethods.map(normalizeArenaMethod);
}

export function loadHuggingFaceCombinationMethods(selectedKeysInput: string[]): SpectralMethod[] {
  const selectedKeys = HF_KEY_ORDER.filter((key) => selectedKeysInput.includes(key));

  if (selectedKeys.length < 2) {
    throw new Error("Please select at least two Hugging Face benchmarks.");
  }

  if (selectedKeys.length >= HF_KEY_ORDER.length) {
    return loadHuggingFaceBaseMethods();
  }

  const combinationName = selectedKeys.join("_");
  const combinationFile = path.join(HF_COMBINATIONS_DIR, combinationName, "ranking_results.json");

  if (!fs.existsSync(combinationFile)) {
    throw new Error(`Combination not found: ${combinationName}`);
  }

  const baseMethods = loadHuggingFaceBaseMethods();
  const methods = readCombinationMethods(combinationFile);

  return sortMethodsByRank(attachHfBenchmarkScores(methods, selectedKeys, baseMethods));
}

function toArenaFieldKey(virtualKey: string): string {
  const field = ARENA_VIRTUAL_TO_FIELD[virtualKey];
  if (!field) {
    throw new Error(`Unrecognized arena benchmark key: ${virtualKey}`);
  }
  return field;
}

export function loadArenaCombinationMethods(selectedVirtualKeysInput: string[]): SpectralMethod[] {
  const normalized = [...new Set(selectedVirtualKeysInput)]
    .filter((key) => Boolean(ARENA_VIRTUAL_TO_FIELD[key]));

  if (normalized.length < 1) {
    throw new Error("Please select at least one Arena benchmark.");
  }

  if (normalized.length >= ARENA_BENCHMARK_LABELS.length) {
    return loadArenaBaseMethods();
  }

  const fields = normalized.map(toArenaFieldKey).sort((a, b) => a.localeCompare(b));
  const combinationName = fields.join("_");
  const combinationFile = path.join(ARENA_COMBINATIONS_DIR, combinationName, "ranking_results.json");

  if (!fs.existsSync(combinationFile)) {
    throw new Error(`Combination not found: ${combinationName}`);
  }

  const methods = readCombinationMethods(combinationFile);
  return sortMethodsByRank(attachArenaBenchmarkScores(methods, normalized));
}

export function loadExampleArenaPreview(): ExampleArenaPreview {
  if (cachedExampleArenaPreview) {
    return {
      ...cachedExampleArenaPreview,
      headers: [...cachedExampleArenaPreview.headers],
      rows: cachedExampleArenaPreview.rows.map((row) => [...row]),
      tasks: [...cachedExampleArenaPreview.tasks],
    };
  }

  const rows = readCsvRows(EXAMPLE_ARENA_FILE);
  const headers = rows[0] ?? [];
  const dataRows = rows.slice(1);

  const taskIndex = headers.findIndex((header) => header.trim().toLowerCase() === "task");
  const taskSet = new Set<string>();

  for (const row of dataRows) {
    if (taskIndex < 0) {
      break;
    }
    const value = (row[taskIndex] ?? "").trim();
    if (value) {
      taskSet.add(value);
    }
  }

  cachedExampleArenaPreview = {
    headers,
    rows: dataRows,
    rowCount: dataRows.length,
    colCount: headers.length,
    tasks: [...taskSet],
  };

  return {
    ...cachedExampleArenaPreview,
    headers: [...cachedExampleArenaPreview.headers],
    rows: cachedExampleArenaPreview.rows.map((row) => [...row]),
    tasks: [...cachedExampleArenaPreview.tasks],
  };
}

export function loadLeaderboardPageData(): LeaderboardPageData {
  return {
    huggingFace: {
      mode: "huggingface",
      methods: loadHuggingFaceBaseMethods(),
    },
    arena: {
      mode: "arena",
      methods: loadArenaBaseMethods(),
    },
    exampleArena: loadExampleArenaPreview(),
  };
}
