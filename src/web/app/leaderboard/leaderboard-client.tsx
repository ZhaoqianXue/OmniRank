"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import {
  ArrowUpRight,
  BarChart3,
  ChevronDown,
  Code2,
  Database,
  Download,
  ExternalLink,
  FileSpreadsheet,
  Medal,
  Rocket,
  Scale,
  Search,
  Sparkles,
  Trophy,
  Upload,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { SiteNavbar } from "@/components/layout/site-navbar";
import { cn } from "@/lib/utils";
import {
  ARENA_BENCHMARK_LABELS,
  ARENA_LABEL_TO_VIRTUAL,
  HF_BENCHMARK_LABELS,
  HF_LABEL_TO_KEY,
  type ArenaBenchmarkLabel,
  type HuggingFaceBenchmarkLabel,
  type LeaderboardMode,
  type LeaderboardPageData,
  type SpectralMethod,
} from "@/lib/leaderboard-types";

interface LeaderboardClientProps {
  initialData: LeaderboardPageData;
}

type SortDirection = "asc" | "desc";

interface TableColumn {
  key: string;
  label: string;
  tooltip?: string;
  sortable?: boolean;
  toggleable?: boolean;
  rankLike?: boolean;
  className?: string;
  style?: CSSProperties;
}

interface CustomSummary {
  modelName: string;
  rank: number;
  scoreRank: number;
  thetaHat: number;
  ciTwoSided: [number, number];
  ciLeft: number;
  ciUniformLeft: number;
  benchmarkScores: Record<string, number>;
}

const sectionWrapperClass = "mx-auto w-full max-w-[1400px] px-4 md:px-8";

function scrollToSection(id: string): void {
  const target = document.getElementById(id);
  if (!target) {
    return;
  }

  const nav = document.getElementById("leaderboard-top-nav");
  const navHeight = nav?.getBoundingClientRect().height ?? 64;
  const top = window.scrollY + target.getBoundingClientRect().top - navHeight - 18;
  window.scrollTo({ top: Math.max(0, top), behavior: "smooth" });
}

function toNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }

  if (typeof value === "string") {
    const cleaned = value.replace(/[^0-9.-]/g, "");
    if (!cleaned) {
      return null;
    }

    const parsed = Number.parseFloat(cleaned);
    return Number.isFinite(parsed) ? parsed : null;
  }

  return null;
}

function formatTopModelName(methods: SpectralMethod[]): string {
  const top = methods.slice().sort((a, b) => a.rank - b.rank)[0];
  return top?.name ?? "N/A";
}

function getAverageScore(method: SpectralMethod): number {
  return method.benchmark_scores?.average_score ?? 0;
}

function cloneMethod(method: SpectralMethod): SpectralMethod {
  return {
    ...method,
    benchmark_scores: method.benchmark_scores ? { ...method.benchmark_scores } : undefined,
  };
}

function buildCustomRankingResult(
  baselineMethods: SpectralMethod[],
  modelName: string,
  benchmarkScores: Record<string, number>,
): { methods: SpectralMethod[]; summary: CustomSummary } {
  const methods = baselineMethods.map(cloneMethod);
  const averageScore = Object.values(benchmarkScores).reduce((sum, value) => sum + value, 0) / 6;

  const existingAverageScores = methods.map((method) => getAverageScore(method));
  const extendedAverageScores = [...existingAverageScores, averageScore].sort((a, b) => b - a);
  const scoreRank = extendedAverageScores.findIndex((value) => value === averageScore) + 1;

  const thetaValues = methods.map((method) => method.theta_hat);
  const minTheta = Math.min(...thetaValues);
  const maxTheta = Math.max(...thetaValues);

  const minAvg = Math.min(...existingAverageScores);
  const maxAvg = Math.max(...existingAverageScores);
  const avgRange = Math.max(maxAvg - minAvg, 0.000001);
  const thetaRange = Math.max(maxTheta - minTheta, 0.000001);

  const normalizedPosition = (averageScore - minAvg) / avgRange;
  const thetaHat = minTheta + normalizedPosition * thetaRange;

  const spectralRank = scoreRank;

  const shiftedMethods = methods.map((method) => {
    if (method.rank >= spectralRank) {
      return { ...method, rank: method.rank + 1 };
    }
    return method;
  });

  const totalModels = shiftedMethods.length + 1;
  const ciLeft = Math.max(1, spectralRank - 1);
  const ciRight = Math.min(totalModels, spectralRank + 3);
  const ciUniformLeft = Math.max(1, spectralRank - 2);

  const customMethod: SpectralMethod = {
    name: modelName,
    rank: spectralRank,
    theta_hat: thetaHat,
    ci_two_sided: [ciLeft, ciRight],
    ci_left: ciLeft,
    ci_uniform_left: ciUniformLeft,
    benchmark_scores: {
      ifeval: benchmarkScores.ifeval,
      bbh: benchmarkScores.bbh,
      math: benchmarkScores.math,
      gpqa: benchmarkScores.gpqa,
      musr: benchmarkScores.musr,
      mmlu_pro: benchmarkScores.mmlu_pro,
      average_score: averageScore,
    },
  };

  const finalMethods = [...shiftedMethods, customMethod].sort((a, b) => a.rank - b.rank);

  return {
    methods: finalMethods,
    summary: {
      modelName,
      rank: spectralRank,
      scoreRank,
      thetaHat,
      ciTwoSided: [ciLeft, ciRight],
      ciLeft,
      ciUniformLeft,
      benchmarkScores: customMethod.benchmark_scores ?? {},
    },
  };
}

function MetricOverview({
  methods,
  benchmarkCount,
  sourceName,
  sourceUrl,
  benchmarkLabel,
}: {
  methods: SpectralMethod[];
  benchmarkCount: number;
  sourceName: string;
  sourceUrl: string;
  benchmarkLabel: string;
}) {
  return (
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      <Card className="gap-2 border-border/70 bg-card/70 py-4">
        <CardHeader className="px-4 pb-1">
          <CardDescription>Ranked Models</CardDescription>
          <CardTitle className="text-2xl">{methods.length}</CardTitle>
        </CardHeader>
      </Card>
      <Card className="gap-2 border-border/70 bg-card/70 py-4">
        <CardHeader className="px-4 pb-1">
          <CardDescription>{benchmarkLabel}</CardDescription>
          <CardTitle className="text-2xl">{benchmarkCount}</CardTitle>
        </CardHeader>
      </Card>
      <Card className="gap-2 border-border/70 bg-card/70 py-4">
        <CardHeader className="px-4 pb-1">
          <CardDescription>Top Ranked Model</CardDescription>
          <CardTitle className="truncate text-xl">{formatTopModelName(methods)}</CardTitle>
        </CardHeader>
      </Card>
      <Card className="gap-2 border-border/70 bg-card/70 py-4">
        <CardHeader className="px-4 pb-1">
          <CardDescription>Data Source</CardDescription>
          <CardTitle className="text-xl">
            <a
              href={sourceUrl}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1 text-primary underline underline-offset-4"
            >
              {sourceName}
              <ExternalLink className="h-4 w-4" />
            </a>
          </CardTitle>
        </CardHeader>
      </Card>
    </div>
  );
}

function WhatIsOmniRankSection() {
  return (
    <section id="what-is-omnirank" className={cn(sectionWrapperClass, "mb-8")}>
      <h2 className="mb-4 text-2xl font-semibold">What is OmniRank LLM Leaderboard?</h2>
      <div className="grid gap-4 lg:grid-cols-3">
        <Card className="border-border/70 bg-card/65 py-5">
          <CardHeader className="px-5 pb-0">
            <CardTitle className="flex items-center gap-2 text-lg"><BarChart3 className="h-5 w-5 text-primary" /> What This Leaderboard Does</CardTitle>
          </CardHeader>
          <CardContent className="px-5 text-sm text-muted-foreground">
            <p>
              This leaderboard provides a statistically robust ranking of Large Language Models (LLMs)
              using the OmniRank algorithm, beyond simple score averaging.
            </p>
            <ul className="mt-3 space-y-2">
              <li>Comprehensive rankings across LMSYS Arena and Hugging Face sources.</li>
              <li>Confidence intervals for every model rank.</li>
              <li>Customizable analysis by selecting benchmark subsets.</li>
              <li>Head-to-head contextual comparison across benchmark categories.</li>
            </ul>
          </CardContent>
        </Card>

        <Card className="border-border/70 bg-card/65 py-5">
          <CardHeader className="px-5 pb-0">
            <CardTitle className="flex items-center gap-2 text-lg"><Scale className="h-5 w-5 text-primary" /> OmniRank vs Regular Ranking</CardTitle>
          </CardHeader>
          <CardContent className="px-5 text-sm text-muted-foreground">
            <p className="font-medium text-foreground">Regular Ranking Limitations:</p>
            <ul className="mt-2 space-y-2">
              <li>Ignores context: simple averaging treats all benchmarks equally.</li>
              <li>No confidence information for ranking stability.</li>
            </ul>
            <p className="mt-3 font-medium text-foreground">OmniRank Advantages:</p>
            <ul className="mt-2 space-y-2">
              <li>Context-aware tournament network modeling.</li>
              <li>Bootstrap-based confidence intervals.</li>
              <li>More robust against outliers and benchmark bias.</li>
            </ul>
          </CardContent>
        </Card>

        <Card className="border-border/70 bg-card/65 py-5">
          <CardHeader className="px-5 pb-0">
            <CardTitle className="flex items-center gap-2 text-lg"><Sparkles className="h-5 w-5 text-primary" /> Key Features</CardTitle>
          </CardHeader>
          <CardContent className="px-5 text-sm text-muted-foreground">
            <ul className="space-y-2">
              <li>Multi data sources: Arena + Hugging Face.</li>
              <li>Custom rankings with benchmark selection.</li>
              <li>Tournament-based OmniRank scoring.</li>
              <li>95% confidence intervals for all models.</li>
              <li>OmniRank vs Average Score Rank side-by-side.</li>
              <li>Rank your own model against top leaderboard models.</li>
            </ul>
          </CardContent>
        </Card>
      </div>
    </section>
  );
}

function RankingTable({
  mode,
  methods,
  selectedLabels,
  highlightModel,
}: {
  mode: LeaderboardMode;
  methods: SpectralMethod[];
  selectedLabels: string[];
  highlightModel: string | null;
}) {
  const [sortKey, setSortKey] = useState<string>("rank");
  const [sortDirection, setSortDirection] = useState<SortDirection>("asc");

  const columns = useMemo<TableColumn[]>(() => {
    if (mode === "arena") {
      const omniRankTooltip = "The model's rank calculated using the OmniRank algorithm. This method provides a more robust result by considering pairwise comparisons across 7 virtual benchmarks derived from human preference data (e.g., Creative Writing, Math, Coding).";
      const scoreRankTooltip = "The model's rank based on the simple average score across the selected virtual benchmarks. Used for comparison against the more robust OmniRank.";

      const baseColumns: TableColumn[] = [
        {
          key: "model",
          label: "Model",
          sortable: true,
          className: "core-column",
          style: {
            width: "250px",
            maxWidth: "250px",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          },
        },
        {
          key: "rank",
          label: "OmniRank",
          sortable: true,
          rankLike: true,
          tooltip: omniRankTooltip,
          className: "core-column rank-header",
        },
        {
          key: "theta_hat",
          label: "θ-hat Score",
          sortable: false,
          toggleable: true,
          tooltip: "The estimated performance score from the OmniRank algorithm. Higher is better.",
          className: "core-column theta-hat-col",
          style: {
            width: "90px",
            minWidth: "90px",
            maxWidth: "90px",
          },
        },
        {
          key: "ci_95",
          label: "95% CI",
          sortable: false,
          toggleable: true,
          tooltip: "The 95% two-sided confidence interval for the rank. For example, an interval of [1, 3] means we are 95% confident the model's true rank is between 1 and 3.",
          className: "core-column ci-95-col",
          style: {
            width: "80px",
            minWidth: "80px",
            maxWidth: "80px",
          },
        },
        {
          key: "ci_uniform",
          label: "Uniform CI",
          sortable: false,
          toggleable: true,
          tooltip: "A more conservative, uniform one-sided confidence interval for the rank that holds simultaneously for all models with 95% confidence.",
          className: "core-column uniform-ci-col",
          style: {
            width: "80px",
            minWidth: "80px",
            maxWidth: "80px",
          },
        },
        {
          key: "avg_rank",
          label: "Average Score Rank",
          sortable: true,
          toggleable: true,
          rankLike: true,
          tooltip: scoreRankTooltip,
          className: "core-column avg-rank-col",
        },
      ];

      const allArenaColumns: TableColumn[] = [
        {
          key: "creative_writing",
          label: "Creative Writing",
          sortable: true,
          rankLike: true,
          className: "benchmark-column",
          tooltip: "The model's rank in the \"Creative Writing\" category. This category evaluates the ability to generate original, imaginative, and emotionally resonant content based on human preference votes.",
        },
        {
          key: "math",
          label: "Math",
          sortable: true,
          rankLike: true,
          className: "benchmark-column",
          tooltip: "The model's rank in the \"Math\" category. This category evaluates the ability to apply mathematical reasoning and problem-solving skills based on human preference votes.",
        },
        {
          key: "instruction_following",
          label: "Instruction Following",
          sortable: true,
          rankLike: true,
          className: "benchmark-column",
          tooltip: "The model's rank in the \"Instruction Following\" category. This category evaluates the ability to accurately follow specific and detailed user instructions based on human preference votes.",
        },
        {
          key: "coding",
          label: "Coding",
          sortable: true,
          rankLike: true,
          className: "benchmark-column",
          tooltip: "The model's rank in the \"Coding\" category. This category evaluates the ability to understand, generate, and debug code based on human preference votes.",
        },
        {
          key: "hard_prompt",
          label: "Hard Prompt",
          sortable: true,
          rankLike: true,
          className: "benchmark-column",
          tooltip: "The model's rank in the \"Hard Prompt\" category. This category evaluates the ability to handle complex, rigorously designed prompts that require multiple skills, based on human preference votes.",
        },
        {
          key: "longer_query",
          label: "Longer Query",
          sortable: true,
          rankLike: true,
          className: "benchmark-column",
          tooltip: "The model's rank in the \"Longer Query\" category. This category includes user prompts that are longer than 500 tokens, testing long-context understanding.",
        },
        {
          key: "multi_turn",
          label: "Multi-turn",
          sortable: true,
          rankLike: true,
          className: "benchmark-column",
          tooltip: "The model's rank in the \"Multi-Turn\" category. This category evaluates performance in conversational interactions that involve more than one turn.",
        },
      ];

      const filtered = allArenaColumns.filter((column) => selectedLabels.includes(column.label));
      const benchmarkColumns = filtered.map((column, index) => ({
        ...column,
        className: cn(column.className, index === 0 && "first-benchmark-col"),
      }));
      return [...baseColumns, ...benchmarkColumns];
    }

    const omniRankTooltip = "The model's rank calculated using the OmniRank algorithm. This method provides a more robust result by considering pairwise comparisons based on scores from 6 key benchmarks: IFEval, BBH, MATH, GPQA, MUSR, and MMLU-Pro.";
    const scoreRankTooltip = "The model's rank based on its average score across the selected benchmarks. Used for comparison against OmniRank.";

    const baseColumns: TableColumn[] = [
      {
        key: "model",
        label: "Model",
        sortable: true,
        className: "core-column",
        style: {
          width: "125px",
          maxWidth: "125px",
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
        },
      },
      {
        key: "rank",
        label: "OmniRank",
        sortable: true,
        rankLike: true,
        tooltip: omniRankTooltip,
        className: "core-column rank-header",
        style: { width: "70px" },
      },
      {
        key: "theta_hat",
        label: "θ-hat Score",
        sortable: true,
        toggleable: true,
        tooltip: "The estimated performance score from the OmniRank algorithm. Higher is better.",
        className: "core-column theta-hat-col",
        style: { width: "80px" },
      },
      {
        key: "ci_95",
        label: "95% CI",
        sortable: false,
        toggleable: true,
        tooltip: "The 95% two-sided confidence interval for the rank. For example, an interval of [1, 3] means we are 95% confident the model's true rank is between 1 and 3.",
        className: "core-column",
        style: { width: "85px" },
      },
      {
        key: "ci_uniform",
        label: "Uniform CI",
        sortable: false,
        toggleable: true,
        tooltip: "A more conservative, uniform one-sided confidence interval for the rank that holds simultaneously for all models with 95% confidence.",
        className: "core-column uniform-ci-col",
        style: { width: "75px" },
      },
      {
        key: "avg_rank",
        label: "Average Score Rank",
        sortable: true,
        rankLike: true,
        toggleable: true,
        tooltip: scoreRankTooltip,
        className: "core-column avg-rank-col",
        style: { width: "70px" },
      },
    ];

    const allHfColumns: TableColumn[] = [
      {
        key: "ifeval",
        label: "IFEval",
        sortable: true,
        className: "benchmark-column",
        tooltip: "Instruction Following Evaluation (IFEval): Assesses the model's ability to follow complex and detailed instructions, focusing on precision and adherence to constraints, not creativity.",
        style: { width: "70px" },
      },
      {
        key: "bbh",
        label: "BBH",
        sortable: true,
        className: "benchmark-column",
        tooltip: "Big-Bench Hard (BBH): A challenging subset of the Big-Bench benchmark, featuring 23 tasks that require significant multi-step reasoning abilities from the language models.",
        style: { width: "70px" },
      },
      {
        key: "math",
        label: "MATH",
        sortable: true,
        className: "benchmark-column",
        tooltip: "A benchmark consisting of 12,500 challenging competition mathematics problems from high school level contests, designed to test mathematical problem-solving and reasoning.",
        style: { width: "70px" },
      },
      {
        key: "gpqa",
        label: "GPQA",
        sortable: true,
        className: "benchmark-column",
        tooltip: "Graduate-Level Google-Proof Q&A (GPQA): A difficult dataset of questions written by domain experts that are hard to find answers for using search engines, testing deep domain knowledge.",
        style: { width: "70px" },
      },
      {
        key: "musr",
        label: "MUSR",
        sortable: true,
        className: "benchmark-column",
        tooltip: "Multi-Step Reasoning (MuSR): Evaluates the model's ability to perform complex, multi-step reasoning by solving problems that require chaining together facts and inferences.",
        style: { width: "70px" },
      },
      {
        key: "mmlu_pro",
        label: "MMLU-Pro",
        sortable: true,
        className: "benchmark-column",
        tooltip: "An advanced version of the MMLU benchmark that features more challenging questions requiring deeper knowledge and reasoning, curated by subject matter experts.",
        style: { width: "70px" },
      },
    ];

    const filtered = allHfColumns.filter((column) => selectedLabels.includes(column.label));
    const benchmarkColumns = filtered.map((column, index) => ({
      ...column,
      className: cn(column.className, index === 0 && "first-benchmark-col"),
    }));

    return [...baseColumns, ...benchmarkColumns];
  }, [mode, selectedLabels]);

  const benchmarkColumnKeys = useMemo(
    () => columns.filter((column) => column.className?.includes("benchmark-column")).map((column) => column.key),
    [columns],
  );

  const avgRankMap = useMemo(() => {
    const map = new Map<string, number>();
    const sorted = methods.slice().sort((a, b) => getAverageScore(b) - getAverageScore(a));
    sorted.forEach((method, index) => {
      map.set(method.name, index + 1);
    });
    return map;
  }, [methods]);

  const arenaRankMap = useMemo(() => {
    const map = new Map<string, Map<string, number>>();
    if (mode !== "arena") {
      return map;
    }

    for (const benchmarkKey of benchmarkColumnKeys) {
      const rankMap = new Map<string, number>();
      const sorted = methods
        .slice()
        .sort((a, b) => (b.benchmark_scores?.[benchmarkKey] ?? 0) - (a.benchmark_scores?.[benchmarkKey] ?? 0));
      sorted.forEach((method, index) => {
        rankMap.set(method.name, index + 1);
      });
      map.set(benchmarkKey, rankMap);
    }

    return map;
  }, [benchmarkColumnKeys, methods, mode]);

  const rows = useMemo(() => {
    return methods
      .slice()
      .sort((a, b) => a.rank - b.rank)
      .map((method) => {
        const values: Record<string, number | string> = {
          rank: method.rank,
          theta_hat: method.theta_hat.toFixed(4),
          ci_95: `[${method.ci_two_sided[0]}, ${method.ci_two_sided[1]}]`,
          ci_uniform: `≤${method.ci_uniform_left}`,
          avg_rank: avgRankMap.get(method.name) ?? "N/A",
        };

        if (mode === "arena") {
          for (const key of benchmarkColumnKeys) {
            values[key] = arenaRankMap.get(key)?.get(method.name) ?? "N/A";
          }
        } else {
          for (const key of benchmarkColumnKeys) {
            const value = method.benchmark_scores?.[key];
            values[key] = value !== undefined ? Number(value).toFixed(2) : "N/A";
          }
        }

        return {
          method,
          values,
        };
      });
  }, [arenaRankMap, avgRankMap, benchmarkColumnKeys, methods, mode]);

  const topThreeByColumn = useMemo(() => {
    const map = new Map<string, { first?: string; second?: string; third?: string }>();

    for (const column of columns) {
      if (column.key === "model" || column.key === "ci_95" || column.key === "ci_uniform") {
        continue;
      }

      const isRankColumn = column.rankLike || (mode === "arena" && benchmarkColumnKeys.includes(column.key));
      const ranked = rows
        .map((row) => {
          const value = toNumber(row.values[column.key]);
          if (value === null) {
            return null;
          }
          return { modelName: row.method.name, value };
        })
        .filter((entry): entry is { modelName: string; value: number } => entry !== null)
        .sort((a, b) => {
          if (isRankColumn) {
            return a.value - b.value;
          }
          return b.value - a.value;
        })
        .slice(0, 3);

      map.set(column.key, {
        first: ranked[0]?.modelName,
        second: ranked[1]?.modelName,
        third: ranked[2]?.modelName,
      });
    }

    return map;
  }, [benchmarkColumnKeys, columns, mode, rows]);

  const sortedRows = useMemo(() => {
    const activeColumn = columns.find((column) => column.key === sortKey);
    if (!activeColumn) {
      return rows.slice().sort((left, right) => Number(left.values.rank) - Number(right.values.rank));
    }

    const direction = sortDirection === "asc" ? 1 : -1;
    return rows.slice().sort((left, right) => {
      const leftValue = left.values[sortKey];
      const rightValue = right.values[sortKey];

      const leftRaw = String(leftValue ?? "");
      const rightRaw = String(rightValue ?? "");
      const hasNumeric = toNumber(leftRaw) !== null || toNumber(rightRaw) !== null;

      if (hasNumeric) {
        const leftNumber = leftRaw === "N/A" ? Number.NEGATIVE_INFINITY * direction : (toNumber(leftRaw) ?? 0);
        const rightNumber = rightRaw === "N/A" ? Number.NEGATIVE_INFINITY * direction : (toNumber(rightRaw) ?? 0);
        return (leftNumber - rightNumber) * direction;
      }

      return leftRaw.localeCompare(rightRaw) * direction;
    });
  }, [columns, rows, sortDirection, sortKey]);

  const handleSort = (column: TableColumn) => {
    if (!column.sortable || column.key === "model") {
      return;
    }

    if (sortKey === column.key) {
      setSortDirection((previous) => (previous === "asc" ? "desc" : "asc"));
      return;
    }

    setSortKey(column.key);
    const isRankColumn = column.rankLike || (mode === "arena" && benchmarkColumnKeys.includes(column.key));
    setSortDirection(isRankColumn ? "asc" : "desc");
  };

  return (
    <div className="leaderboard-spectral-table-html">
      <div className="leaderboard-spectral-table-container">
        <table className={cn("leaderboard-spectral-table show-details", mode === "arena" && "arena-table-layout")}>
          <thead>
            <tr>
              {columns.map((column) => {
                const sortable = Boolean(column.sortable && column.key !== "model");
                const isSorted = sortable && sortKey === column.key;
                const headerClassName = cn(
                  column.className,
                  column.toggleable && "toggleable-col",
                  sortable && "sortable-header",
                  isSorted && sortDirection === "asc" && "sorted-asc",
                  isSorted && sortDirection === "desc" && "sorted-desc",
                );

                const labelContent = (
                  <>
                    <span>{column.label}</span>
                    {sortable ? (
                      <span className="sort-icons">
                        <span className="sort-icon-up">▲</span>
                        <span className="sort-icon-down">▼</span>
                      </span>
                    ) : null}
                  </>
                );

                return (
                  <th
                    key={column.key}
                    className={headerClassName}
                    style={{ ...column.style, textAlign: "left" }}
                    onClick={sortable ? () => handleSort(column) : undefined}
                  >
                    {column.tooltip ? (
                      <span className="tooltip-container">
                        {labelContent}
                        <span className="tooltip-text">{column.tooltip}</span>
                      </span>
                    ) : (
                      labelContent
                    )}
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {sortedRows.map((row) => {
              const rowClassName = cn(
                highlightModel && row.method.name === highlightModel && "user-model-highlight",
              );

              return (
                <tr key={row.method.name} className={rowClassName}>
                  {columns.map((column) => {
                    let topClassName = "";
                    const topThree = topThreeByColumn.get(column.key);
                    if (topThree?.first === row.method.name) {
                      topClassName = "first-place-cell";
                    } else if (topThree?.second === row.method.name) {
                      topClassName = "second-place-cell";
                    } else if (topThree?.third === row.method.name) {
                      topClassName = "third-place-cell";
                    }

                    const cellClassName = cn(
                      column.key === "model" && "model-cell",
                      column.key === "rank" && "rank-cell",
                      column.className,
                      column.toggleable && "toggleable-col",
                      topClassName,
                    );

                    if (column.key === "model") {
                      return (
                        <td key={`${row.method.name}-${column.key}`} className={cellClassName} style={{ ...column.style, textAlign: "left" }}>
                          {row.method.model_url ? (
                            <a href={row.method.model_url} target="_blank" rel="noreferrer">
                              {row.method.name}
                            </a>
                          ) : (
                            row.method.name
                          )}
                        </td>
                      );
                    }

                    const rawValue = row.values[column.key] ?? "N/A";
                    return (
                      <td key={`${row.method.name}-${column.key}`} className={cellClassName} style={{ ...column.style, textAlign: "left" }}>
                        {rawValue}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function TopThreeLegend() {
  return (
    <div className="leaderboard-top-3-legend">
      <div className="legend-item">
        <div className="legend-color first" />
        <span>First Place</span>
      </div>
      <div className="legend-item">
        <div className="legend-color second" />
        <span>Second Place</span>
      </div>
      <div className="legend-item">
        <div className="legend-color third" />
        <span>Third Place</span>
      </div>
    </div>
  );
}

function SpectralTableStyles() {
  return (
    <style jsx global>{`
      .leaderboard-spectral-table-container {
        max-height: 600px;
        overflow: auto;
        border: 1px solid rgba(39, 51, 83, 0.88);
        border-radius: 10px;
        position: relative;
        background: linear-gradient(180deg, rgba(16, 25, 46, 0.93) 0%, rgba(11, 16, 30, 0.93) 100%);
        box-shadow: 0 18px 36px rgba(2, 6, 23, 0.38);
        backdrop-filter: blur(8px);
      }

      .leaderboard-spectral-table-html table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.82em;
      }

      .leaderboard-spectral-table-html thead th {
        position: sticky !important;
        top: 0 !important;
        background: rgba(20, 30, 53, 0.98) !important;
        z-index: 1001 !important;
        border-bottom: 2px solid rgba(60, 80, 124, 0.9);
        border-right: 1px solid rgba(56, 73, 113, 0.84);
        padding: 10px 7px;
        font-weight: 600;
        color: #d6dcfb;
      }

      .leaderboard-spectral-table-html thead th:hover {
        z-index: 1012 !important;
      }

      .leaderboard-spectral-table-html thead th:last-child {
        border-right: none;
      }

      .leaderboard-spectral-table-html thead th.core-column {
        background: rgba(39, 58, 95, 0.96) !important;
        border-right: 1px solid rgba(109, 134, 190, 0.62) !important;
      }

      .leaderboard-spectral-table-html thead th.benchmark-column {
        background: rgba(57, 50, 26, 0.96) !important;
        border-right: 1px solid rgba(153, 116, 45, 0.64) !important;
      }

      .leaderboard-spectral-table-html thead th.core-column + th.benchmark-column {
        border-left: 4px solid #5f729f !important;
      }

      .leaderboard-spectral-table-html thead th.benchmark-column.first-benchmark-col {
        border-left: 4px solid #5f729f !important;
      }

      .leaderboard-spectral-table-html tbody td.core-column + td.benchmark-column {
        border-left: 4px solid #5f729f !important;
      }

      .leaderboard-spectral-table-html tbody td.benchmark-column.first-benchmark-col {
        border-left: 4px solid #5f729f !important;
      }

      .leaderboard-spectral-table-html tbody tr:nth-child(even) {
        background-color: rgba(16, 25, 46, 0.9);
      }

      .leaderboard-spectral-table-html tbody tr:hover {
        background-color: rgba(27, 40, 69, 0.94);
      }

      .leaderboard-spectral-table-html tbody td {
        padding: 9px 22px 9px 7px;
        border-bottom: 1px solid rgba(56, 73, 113, 0.72);
        border-right: 1px solid rgba(56, 73, 113, 0.72);
        color: #d2daf6;
        font-weight: 600;
        position: relative;
      }

      .leaderboard-spectral-table-html tbody td:last-child {
        border-right: none;
      }

      .leaderboard-spectral-table-html .model-cell {
        font-weight: 700;
        color: #e7ebff;
      }

      .leaderboard-spectral-table-html .model-cell a {
        color: #c5b8f6;
        text-decoration: underline;
      }

      .leaderboard-spectral-table-html .rank-cell {
        font-weight: 600;
        color: #9cc9ff;
      }

      .leaderboard-spectral-table-html .rank-header {
        font-weight: 600;
        color: #e7ebff;
      }

      .leaderboard-spectral-table-html .first-place-cell {
        background-color: rgba(146, 64, 14, 0.34) !important;
        color: #fde68a !important;
        font-weight: 600 !important;
        position: relative !important;
      }

      .leaderboard-spectral-table-html .first-place-cell::after {
        content: "🥇";
        position: absolute;
        right: 4px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 1.05em;
        opacity: 0.7;
      }

      .leaderboard-spectral-table-html .second-place-cell {
        background-color: rgba(51, 65, 85, 0.72) !important;
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        position: relative !important;
      }

      .leaderboard-spectral-table-html .second-place-cell::after {
        content: "🥈";
        position: absolute;
        right: 4px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 1.05em;
        opacity: 0.7;
      }

      .leaderboard-spectral-table-html .third-place-cell {
        background-color: rgba(127, 29, 29, 0.34) !important;
        color: #fecaca !important;
        font-weight: 600 !important;
        position: relative !important;
      }

      .leaderboard-spectral-table-html .third-place-cell::after {
        content: "🥉";
        position: absolute;
        right: 4px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 1.05em;
        opacity: 0.7;
      }

      .leaderboard-top-3-legend {
        display: flex;
        gap: 0.875rem;
        margin-bottom: 0.25rem;
        padding: 0.625rem 0.75rem;
        background: rgba(16, 25, 46, 0.76);
        border-radius: 10px;
        border: 1px solid rgba(39, 51, 83, 0.78);
      }

      .leaderboard-top-3-legend .legend-item {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        font-size: 0.8125rem;
        font-weight: 500;
        color: #d1daf8;
      }

      .leaderboard-top-3-legend .legend-color {
        width: 14px;
        height: 14px;
        border-radius: 3px;
        border: 1.5px solid rgba(0, 0, 0, 0.1);
      }

      .leaderboard-top-3-legend .legend-color.first {
        background-color: rgba(146, 64, 14, 0.42);
        border-color: #fbbf24;
      }

      .leaderboard-top-3-legend .legend-color.second {
        background-color: rgba(51, 65, 85, 0.78);
        border-color: #94a3b8;
      }

      .leaderboard-top-3-legend .legend-color.third {
        background-color: rgba(127, 29, 29, 0.42);
        border-color: #f87171;
      }

      .leaderboard-spectral-table-html table .toggleable-col {
        display: none;
      }

      .leaderboard-spectral-table-html table.show-details .toggleable-col {
        display: table-cell;
      }

      .leaderboard-spectral-table-html table.show-details .theta-hat-col,
      .leaderboard-spectral-table-html table.show-details .uniform-ci-col,
      .leaderboard-spectral-table-html table.show-details .avg-rank-col {
        display: none !important;
      }

      .leaderboard-spectral-table-html table .toggleable-col:not(.ci-95-col) {
        display: none !important;
      }

      .leaderboard-spectral-table-html table .ci-95-col {
        display: table-cell !important;
      }

      .leaderboard-spectral-table-html .sortable-header {
        cursor: pointer;
        position: relative;
        white-space: nowrap;
      }

      .leaderboard-spectral-table-html .sort-icons {
        display: inline-flex;
        flex-direction: column;
        margin-left: 6px;
        position: relative;
        top: 2px;
      }

      .leaderboard-spectral-table-html .sort-icons .sort-icon-up,
      .leaderboard-spectral-table-html .sort-icons .sort-icon-down {
        font-size: 0.6875rem;
        line-height: 0.7;
        color: #8a96bc;
      }

      .leaderboard-spectral-table-html .sortable-header:hover .sort-icons .sort-icon-up,
      .leaderboard-spectral-table-html .sortable-header:hover .sort-icons .sort-icon-down {
        color: #bcc7ec;
      }

      .leaderboard-spectral-table-html .sortable-header.sorted-asc .sort-icon-up,
      .leaderboard-spectral-table-html .sortable-header.sorted-desc .sort-icon-down {
        color: #dce4ff;
      }

      .leaderboard-spectral-table-html .tooltip-container {
        position: relative;
        display: inline-flex;
        align-items: center;
      }

      .leaderboard-spectral-table-html .tooltip-container .tooltip-text {
        white-space: normal;
        visibility: hidden;
        width: 240px;
        background-color: rgba(11, 16, 30, 0.98);
        color: #e7ebff;
        text-align: left;
        border-radius: 6px;
        padding: 9px 10px;
        position: absolute;
        z-index: 1010;
        top: 140%;
        left: 50%;
        margin-left: -120px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.8125rem;
        font-weight: 400;
        line-height: 1.55;
        box-shadow: 0 8px 18px rgba(2, 6, 23, 0.52);
      }

      .leaderboard-spectral-table-html .tooltip-container:hover .tooltip-text {
        visibility: visible;
        opacity: 1;
      }

      .leaderboard-spectral-table-html .tooltip-container .tooltip-text::after {
        content: "";
        position: absolute;
        bottom: 100%;
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: transparent transparent rgba(11, 16, 30, 0.98) transparent;
      }

      .leaderboard-spectral-table-html thead th:nth-last-child(1) .tooltip-container .tooltip-text,
      .leaderboard-spectral-table-html thead th:nth-last-child(2) .tooltip-container .tooltip-text {
        left: auto;
        right: 0;
        margin-left: 0;
      }

      .leaderboard-spectral-table-html thead th:nth-last-child(1) .tooltip-container .tooltip-text::after,
      .leaderboard-spectral-table-html thead th:nth-last-child(2) .tooltip-container .tooltip-text::after {
        left: auto;
        right: 20px;
      }

      .leaderboard-spectral-table-html .user-model-highlight td {
        background-color: rgba(22, 101, 52, 0.35) !important;
        color: #bbf7d0 !important;
        font-weight: 700 !important;
        border-top: 2px solid #22c55e;
        border-bottom: 2px solid #22c55e;
        position: relative;
      }

      .leaderboard-spectral-table-html .user-model-highlight:hover td {
        background-color: rgba(22, 101, 52, 0.5) !important;
        transition: all 0.2s ease;
      }

      .leaderboard-spectral-table.arena-table-layout th,
      .leaderboard-spectral-table.arena-table-layout td {
        word-wrap: break-word;
      }

      .leaderboard-spectral-table.arena-table-layout th:not(:first-child),
      .leaderboard-spectral-table.arena-table-layout td:not(:first-child) {
        flex: 1;
      }
    `}</style>
  );
}

export default function LeaderboardClient({ initialData }: LeaderboardClientProps) {
  const [activeMode, setActiveMode] = useState<LeaderboardMode>("arena");

  const [arenaMethods, setArenaMethods] = useState<SpectralMethod[]>(initialData.arena.methods);
  const [huggingFaceBaselineMethods, setHuggingFaceBaselineMethods] = useState<SpectralMethod[]>(
    initialData.huggingFace.methods,
  );
  const [huggingFaceMethods, setHuggingFaceMethods] = useState<SpectralMethod[]>(
    initialData.huggingFace.methods,
  );

  const [arenaSelection, setArenaSelection] = useState<Record<ArenaBenchmarkLabel, boolean>>({
    "Creative Writing": true,
    Math: true,
    "Instruction Following": true,
    Coding: true,
    "Hard Prompt": true,
    "Longer Query": true,
    "Multi-turn": true,
  });

  const [hfSelection, setHfSelection] = useState<Record<HuggingFaceBenchmarkLabel, boolean>>({
    IFEval: true,
    BBH: true,
    MATH: true,
    GPQA: true,
    MUSR: true,
    "MMLU-Pro": true,
  });

  const [arenaLoading, setArenaLoading] = useState(false);
  const [hfLoading, setHfLoading] = useState(false);
  const [arenaError, setArenaError] = useState<string | null>(null);
  const [hfError, setHfError] = useState<string | null>(null);

  const [highlightModel, setHighlightModel] = useState<string | null>(null);

  const [customModelName, setCustomModelName] = useState("");
  const [customScores, setCustomScores] = useState<Record<keyof Omit<CustomSummary["benchmarkScores"], "average_score">, string>>({
    ifeval: "50",
    bbh: "50",
    math: "50",
    gpqa: "50",
    musr: "50",
    mmlu_pro: "50",
  });
  const [customRunning, setCustomRunning] = useState(false);
  const [customProgress, setCustomProgress] = useState(0);
  const [customSummary, setCustomSummary] = useState<CustomSummary | null>(null);
  const [customError, setCustomError] = useState<string | null>(null);
  const didInitArenaSelectionRef = useRef(false);
  const didInitHfSelectionRef = useRef(false);

  const selectedArenaLabels = useMemo(
    () => ARENA_BENCHMARK_LABELS.filter((label) => arenaSelection[label]),
    [arenaSelection],
  );
  const selectedHfLabels = useMemo(
    () => HF_BENCHMARK_LABELS.filter((label) => hfSelection[label]),
    [hfSelection],
  );

  useEffect(() => {
    const syncHashNavigation = () => {
      const hash = window.location.hash;
      if (hash === "#compare-with-your-model") {
        setActiveMode("huggingface");
      }

      if (hash.startsWith("#")) {
        const targetId = hash.slice(1);
        window.setTimeout(() => {
          scrollToSection(targetId);
        }, 220);
      }
    };

    syncHashNavigation();
    window.addEventListener("hashchange", syncHashNavigation);
    return () => {
      window.removeEventListener("hashchange", syncHashNavigation);
    };
  }, []);

  const selectedArenaKey = selectedArenaLabels.join("|");
  const selectedHfKey = selectedHfLabels.join("|");

  const fetchArenaSelection = useCallback(async (labels: ArenaBenchmarkLabel[]) => {
    if (labels.length < 1) {
      setArenaError("Please select at least one benchmark.");
      return;
    }

    setArenaError(null);
    setArenaLoading(true);

    try {
      const keys = labels.map((label) => ARENA_LABEL_TO_VIRTUAL[label]).join(",");
      const response = await fetch(`/api/leaderboard/combination?mode=arena&keys=${encodeURIComponent(keys)}`);
      const payload = (await response.json()) as { methods?: SpectralMethod[]; error?: string };

      if (!response.ok || !payload.methods) {
        throw new Error(payload.error ?? "Failed to load Arena combination results.");
      }

      setArenaMethods(payload.methods);
    } catch (error) {
      setArenaError(error instanceof Error ? error.message : "Failed to load Arena combination results.");
    } finally {
      setArenaLoading(false);
    }
  }, []);

  const fetchHfSelection = useCallback(async (labels: HuggingFaceBenchmarkLabel[]) => {
    if (labels.length < 2) {
      setHfError("Please select at least two benchmarks.");
      return;
    }

    setHfError(null);
    setHfLoading(true);

    try {
      const keys = labels.map((label) => HF_LABEL_TO_KEY[label]).join(",");
      const response = await fetch(`/api/leaderboard/combination?mode=huggingface&keys=${encodeURIComponent(keys)}`);
      const payload = (await response.json()) as { methods?: SpectralMethod[]; error?: string };

      if (!response.ok || !payload.methods) {
        throw new Error(payload.error ?? "Failed to load Hugging Face combination results.");
      }

      setHuggingFaceBaselineMethods(payload.methods);
      setHuggingFaceMethods(payload.methods);
      setHighlightModel(null);
      setCustomSummary(null);
    } catch (error) {
      setHfError(error instanceof Error ? error.message : "Failed to load Hugging Face combination results.");
    } finally {
      setHfLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!didInitArenaSelectionRef.current) {
      didInitArenaSelectionRef.current = true;
      return;
    }

    const timer = window.setTimeout(() => {
      void fetchArenaSelection(selectedArenaLabels);
    }, 220);

    return () => {
      window.clearTimeout(timer);
    };
  }, [fetchArenaSelection, selectedArenaKey, selectedArenaLabels]);

  useEffect(() => {
    if (!didInitHfSelectionRef.current) {
      didInitHfSelectionRef.current = true;
      return;
    }

    const timer = window.setTimeout(() => {
      void fetchHfSelection(selectedHfLabels);
    }, 220);

    return () => {
      window.clearTimeout(timer);
    };
  }, [fetchHfSelection, selectedHfKey, selectedHfLabels]);

  const clearCustomInputs = () => {
    setCustomModelName("");
    setCustomScores({ ifeval: "", bbh: "", math: "", gpqa: "", musr: "", mmlu_pro: "" });
    setCustomError(null);
  };

  const runCustomRanking = async () => {
    const modelName = customModelName.trim();
    if (!modelName) {
      setCustomError("Please fill in your model name and all benchmark scores.");
      return;
    }

    const duplicate = huggingFaceBaselineMethods.some(
      (method) => method.name.toLowerCase() === modelName.toLowerCase(),
    );
    if (duplicate) {
      setCustomError("Model name already exists in the leaderboard. Please use a different name.");
      return;
    }

    const parsedScores: Record<string, number> = {};
    for (const [key, value] of Object.entries(customScores)) {
      const parsed = Number.parseFloat(value);
      if (!Number.isFinite(parsed)) {
        setCustomError("Please fill in your model name and all benchmark scores.");
        return;
      }
      parsedScores[key] = parsed;
    }

    setCustomError(null);
    setCustomRunning(true);
    setCustomProgress(0);

    await new Promise<void>((resolve) => {
      let current = 0;
      const interval = window.setInterval(() => {
        current = Math.min(current + 9, 95);
        setCustomProgress(current);
      }, 120);

      window.setTimeout(() => {
        window.clearInterval(interval);
        resolve();
      }, 1500);
    });

    const { methods, summary } = buildCustomRankingResult(
      huggingFaceBaselineMethods,
      modelName,
      parsedScores,
    );

    setHuggingFaceMethods(methods);
    setHighlightModel(modelName);
    setCustomSummary(summary);
    setCustomProgress(100);
    setCustomRunning(false);
  };

  return (
    <main className="relative min-h-screen overflow-x-hidden bg-background text-foreground">
      <SpectralTableStyles />
      <div className="pointer-events-none fixed inset-0 -z-30 grid-pattern opacity-40" />
      <div className="pointer-events-none fixed inset-0 -z-20 bg-[radial-gradient(circle_at_15%_15%,rgba(152,132,229,0.24),transparent_42%),radial-gradient(circle_at_80%_12%,rgba(197,184,246,0.18),transparent_46%),radial-gradient(circle_at_45%_85%,rgba(16,25,46,0.9),transparent_62%)]" />
      <div className="pointer-events-none fixed inset-0 -z-10 bg-gradient-to-b from-background/85 via-background/95 to-background" />
      <SiteNavbar id="leaderboard-top-nav" />

      <div className="relative z-10 pb-24 pt-28 md:pt-32">
        <section className={cn(sectionWrapperClass, "mb-10")}>
          <div className="rounded-3xl border border-border/70 bg-card/35 p-6 shadow-[0_20px_50px_rgba(0,0,0,0.35)] md:p-10">
            <button
              type="button"
              onClick={() => scrollToSection("choose-data-source")}
              className="inline-flex items-center gap-2 rounded-full border border-primary/50 bg-primary/15 px-4 py-1 text-xs font-semibold tracking-[0.15em] text-primary"
            >
              <Medal className="h-4 w-4" />
              OmniRank LLM Leaderboard
            </button>

            <h1 className="mt-4 text-3xl font-bold tracking-tight md:text-5xl">
              OmniRank LLM Leaderboard
            </h1>
            <p className="mt-3 max-w-3xl text-sm text-muted-foreground md:text-base">
              Ranking top LLMs with OmniRank algorithm.
              <br />
              LMSYS Arena (human preferences) and Hugging Face (standardized benchmarks).
            </p>

            <div className="mt-6 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
              <Card className="cursor-pointer gap-3 border-border/70 bg-background/45 py-4" onClick={() => scrollToSection("what-is-omnirank")}>
                <CardHeader className="px-4 pb-0">
                  <CardTitle className="flex items-center gap-2 text-base"><Trophy className="h-4 w-4 text-primary" /> OmniRank</CardTitle>
                </CardHeader>
                <CardContent className="px-4 text-sm text-muted-foreground">
                  Delivers statistically robust LLM rankings via tournament-style comparison structure.
                </CardContent>
              </Card>

              <Card className="cursor-pointer gap-3 border-border/70 bg-background/45 py-4" onClick={() => scrollToSection("choose-data-source")}>
                <CardHeader className="px-4 pb-0">
                  <CardTitle className="flex items-center gap-2 text-base"><Database className="h-4 w-4 text-primary" /> Multi Data Sources</CardTitle>
                </CardHeader>
                <CardContent className="px-4 text-sm text-muted-foreground">
                  Access rankings from LMSYS Arena and Hugging Face for complementary evaluation perspectives.
                </CardContent>
              </Card>

              <Card
                className="cursor-pointer gap-3 border-border/70 bg-background/45 py-4"
                onClick={() => {
                  setActiveMode("arena");
                  window.setTimeout(() => scrollToSection("arena-rankings"), 120);
                }}
              >
                <CardHeader className="px-4 pb-0">
                  <CardTitle className="flex items-center gap-2 text-base"><Sparkles className="h-4 w-4 text-primary" /> Custom Rankings</CardTitle>
                </CardHeader>
                <CardContent className="px-4 text-sm text-muted-foreground">
                  Select benchmark subsets (1 to 7 in Arena, 2 to 6 in HF) and generate focused ranking views.
                </CardContent>
              </Card>

              <Card
                className="cursor-pointer gap-3 border-border/70 bg-background/45 py-4"
                onClick={() => {
                  setActiveMode("huggingface");
                  window.setTimeout(() => scrollToSection("compare-with-your-model"), 120);
                }}
              >
                <CardHeader className="px-4 pb-0">
                  <CardTitle className="flex items-center gap-2 text-base"><Scale className="h-4 w-4 text-primary" /> Rank Your Model</CardTitle>
                </CardHeader>
                <CardContent className="px-4 text-sm text-muted-foreground">
                  Enter your own benchmark scores and compare against the current Top 100 leaderboard.
                </CardContent>
              </Card>
            </div>
          </div>
        </section>

        <section id="choose-data-source" className={cn(sectionWrapperClass, "mb-3")}>
          <h2 className="text-2xl font-semibold">Choose a Data Source</h2>
        </section>

        <section className={cn(sectionWrapperClass, "mb-8")}>
          <div className="grid gap-4 lg:grid-cols-2">
            <Card
              id="arena-mode-card"
              className={cn(
                "cursor-pointer border-2 py-6 transition-all",
                activeMode === "arena"
                  ? "border-primary bg-card/80 shadow-[0_16px_36px_rgba(0,0,0,0.35)]"
                  : "border-border/70 bg-card/45 opacity-90",
              )}
              onClick={() => setActiveMode("arena")}
            >
              <CardHeader className="px-6 pb-0">
                <CardTitle className="flex items-center gap-3 text-2xl">
                  <span className="rounded-full bg-primary/15 p-3 text-xl">⚔️</span>
                  LMSYS Arena Leaderboard
                </CardTitle>
                <CardDescription>
                  Crowdsourced human preference battles from real-world interactions.
                </CardDescription>
              </CardHeader>
              <CardContent className="px-6 text-sm text-muted-foreground">
                <ul className="space-y-1">
                  <li>Human Preference Data</li>
                  <li>Head-to-Head Battles</li>
                  <li>Real-World Performance</li>
                </ul>
              </CardContent>
            </Card>

            <Card
              id="huggingface-mode-card"
              className={cn(
                "cursor-pointer border-2 py-6 transition-all",
                activeMode === "huggingface"
                  ? "border-primary bg-card/80 shadow-[0_16px_36px_rgba(0,0,0,0.35)]"
                  : "border-border/70 bg-card/45 opacity-90",
              )}
              onClick={() => setActiveMode("huggingface")}
            >
              <CardHeader className="px-6 pb-0">
                <CardTitle className="flex items-center gap-3 text-2xl">
                  <span className="rounded-full bg-primary/15 p-3 text-xl">🤗</span>
                  Hugging Face Leaderboard
                </CardTitle>
                <CardDescription>
                  Standardized academic benchmark evaluation for top open LLMs.
                </CardDescription>
              </CardHeader>
              <CardContent className="px-6 text-sm text-muted-foreground">
                <ul className="space-y-1">
                  <li>6 Core Benchmarks</li>
                  <li>Automated Evaluation</li>
                  <li>Top 100 Models</li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </section>

        {activeMode === "arena" ? (
          <>
            <section id="arena-rankings" className={cn(sectionWrapperClass, "mb-8")}>
              <h2 className="mb-4 text-2xl font-semibold">LMSYS Arena LLM Rankings</h2>
              <div className="mb-4">
                <MetricOverview
                  methods={arenaMethods}
                  benchmarkCount={selectedArenaLabels.length}
                  benchmarkLabel="Task Categories"
                  sourceName="LMSYS Arena"
                  sourceUrl="https://lmarena.ai/leaderboard/"
                />
              </div>

              <Card className="mb-3 border-border/70 bg-card/65 py-4">
                <CardHeader className="px-4 pb-0">
                  <CardTitle className="text-lg">Select Benchmarks</CardTitle>
                  <CardDescription>
                    Choose which virtual benchmarks to include in your OmniRank analysis. You can select 1 to 7 benchmarks.
                  </CardDescription>
                </CardHeader>
                <CardContent className="px-4">
                  <div className="flex flex-wrap gap-3">
                    {ARENA_BENCHMARK_LABELS.map((label) => (
                      <label key={label} className="inline-flex items-center gap-2 rounded-md border border-border/60 bg-background/50 px-3 py-1.5 text-sm">
                        <input
                          type="checkbox"
                          checked={arenaSelection[label]}
                          onChange={(event) => {
                            setArenaSelection((previous) => ({
                              ...previous,
                              [label]: event.target.checked,
                            }));
                          }}
                        />
                        {label}
                      </label>
                    ))}
                  </div>
                  <div className="mt-3 text-xs text-muted-foreground">
                    Selection updates table automatically.
                    {arenaLoading ? " Updating..." : ""}
                  </div>
                  {arenaError ? <div className="mt-2 text-sm text-red-300">{arenaError}</div> : null}
                </CardContent>
              </Card>

              <TopThreeLegend />
              <RankingTable
                key={`arena-${selectedArenaLabels.join("|")}-${arenaMethods.length}-${highlightModel ?? ""}`}
                mode="arena"
                methods={arenaMethods}
                selectedLabels={selectedArenaLabels}
                highlightModel={highlightModel}
              />
            </section>

            <section className={cn(sectionWrapperClass, "mb-8")}>
              <h2 className="mb-4 text-2xl font-semibold">Upload Your LLMs Arena Results for Ranking</h2>
              <Card className="border-border/70 bg-card/65 py-5">
                <CardHeader className="px-5 pb-0">
                  <CardTitle className="flex items-center gap-2 text-xl"><Upload className="h-5 w-5 text-primary" /> Upload Your LLMs Arena Results</CardTitle>
                  <CardDescription>
                    Run a standalone OmniRank leaderboard on collected Arena-style battles. Results remain separate from the built-in LMSYS leaderboard.
                  </CardDescription>
                </CardHeader>
                <CardContent className="px-5 text-sm text-muted-foreground">
                  <ul className="space-y-2">
                    <li><strong>File format:</strong> Arena-style CSV of pairwise battles. Include a task tag column (e.g., <code>Task</code>) and consistent model columns.</li>
                    <li><strong>Row data:</strong> One battle per row. Winner = <code>1.0</code>, loser = <code>0.0</code>, all other models = <code>NaN</code>.</li>
                    <li><strong>Result:</strong> OmniRank scores, ranks, and confidence intervals for models in your file only.</li>
                    <li><strong>Quick example:</strong> Uploading the sample file runs on two tasks (<code>code</code> and <code>math</code>) and outputs OmniRank ranks for ChatGPT, Claude, Gemini, Llama, Qwen, and Your Model.</li>
                  </ul>

                  <div className="mt-4 rounded-xl border border-border/60 bg-background/35 p-3">
                    <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                      <div className="flex items-center gap-2 font-semibold text-foreground">
                        <FileSpreadsheet className="h-4 w-4 text-primary" />
                        Example Arena-style CSV
                        {initialData.exampleArena.tasks.length > 0 ? (
                          <Badge variant="secondary">Tasks: {initialData.exampleArena.tasks.join(", ")}</Badge>
                        ) : null}
                      </div>
                      <span className="text-xs text-muted-foreground">
                        {initialData.exampleArena.rowCount} rows x {initialData.exampleArena.colCount} cols
                      </span>
                    </div>

                    <ScrollArea className="h-[290px] rounded-md border border-border/50">
                      <table className="w-full min-w-[720px] border-collapse text-xs">
                        <thead className="sticky top-0 z-10 bg-muted">
                          <tr>
                            {initialData.exampleArena.headers.map((header) => (
                              <th key={header} className="border-b border-border/50 px-2 py-1.5 text-left font-semibold">
                                {header}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {initialData.exampleArena.rows.map((row, index) => (
                            <tr key={`${row.join("|")}-${index}`} className="even:bg-background/40">
                              {row.map((cell, cellIndex) => (
                                <td key={`${cellIndex}-${cell}`} className="border-b border-border/30 px-2 py-1.5">
                                  {cell}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </ScrollArea>
                  </div>

                  <div className="mt-4">
                    <Button variant="outline" asChild>
                      <Link href="/#mode-selection" target="_blank" rel="noopener noreferrer">
                        Go to Ranking Page (Use Example or Upload Data)
                        <ArrowUpRight className="h-4 w-4" />
                      </Link>
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </section>

            <WhatIsOmniRankSection />

            <section className={cn(sectionWrapperClass, "mb-8")}> 
              <h2 className="mb-4 text-2xl font-semibold">How This Leaderboard is Calculated</h2>
              <div className="grid gap-4 lg:grid-cols-2">
                <Card className="border-border/70 bg-card/65 py-5">
                  <CardHeader className="px-5 pb-0">
                    <CardTitle className="flex items-center gap-2 text-lg"><Database className="h-5 w-5 text-primary" /> Step 1: Data Source</CardTitle>
                  </CardHeader>
                  <CardContent className="px-5 text-sm text-muted-foreground">
                    <ul className="space-y-2">
                      <li><strong>Dataset:</strong> <a className="text-primary underline" href="https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k" target="_blank" rel="noreferrer">lmarena-ai/arena-human-preference-140k</a></li>
                      <li><strong>Data Scale:</strong> 135,634 battle records, 53 unique models, about 1.61 GB.</li>
                      <li><strong>Collection:</strong> Anonymous crowd preferences on Chatbot Arena.</li>
                      <li><strong>Mechanism:</strong> Blind chat between <code>model_a</code> and <code>model_b</code> with votes (win/tie/both bad).</li>
                      <li><strong>Rich Content:</strong> Includes conversation history, winner field, and category tags.</li>
                      <li><strong>License:</strong> User prompts under CC-BY-4.0.</li>
                    </ul>
                  </CardContent>
                </Card>

                <Card className="border-border/70 bg-card/65 py-5">
                  <CardHeader className="px-5 pb-0">
                    <CardTitle className="flex items-center gap-2 text-lg"><Search className="h-5 w-5 text-primary" /> Step 2: Virtual Benchmarks</CardTitle>
                  </CardHeader>
                  <CardContent className="px-5 text-sm text-muted-foreground">
                    <p>Each battle is categorized into 7 virtual benchmarks based on content, metadata, and Arena definitions:</p>
                    <ul className="mt-2 space-y-2">
                      <li><strong>Creative Writing:</strong> tag <code>creative_writing</code>.</li>
                      <li><strong>Math:</strong> tag <code>math</code>.</li>
                      <li><strong>Instruction Following:</strong> tag <code>if</code>.</li>
                      <li><strong>Coding:</strong> <code>is_code == True</code>.</li>
                      <li><strong>Hard Prompt:</strong> at least 6 of 7 complexity criteria.</li>
                      <li><strong>Longer Query:</strong> prompts over 500 tokens.</li>
                      <li><strong>Multi-Turn:</strong> conversations with more than one turn.</li>
                    </ul>
                    <p className="mt-3 text-xs">
                      Source: <a className="text-primary underline" href="https://news.lmarena.ai/arena-category/" target="_blank" rel="noreferrer">Chatbot Arena Categories</a>
                    </p>
                  </CardContent>
                </Card>

                <Card className="border-border/70 bg-card/65 py-5">
                  <CardHeader className="px-5 pb-0">
                    <CardTitle className="flex items-center gap-2 text-lg"><Code2 className="h-5 w-5 text-primary" /> Step 3: BT-MLE Modeling</CardTitle>
                  </CardHeader>
                  <CardContent className="px-5 text-sm text-muted-foreground">
                    <p>
                      We use the Bradley-Terry model (MLE of Elo) for robust static-model scoring.
                    </p>
                    <ul className="mt-2 space-y-2">
                      <li><strong>Why not simple win rate:</strong> ignores opponent strength.</li>
                      <li><strong>Why BT-MLE over online Elo:</strong> online Elo assumes time-varying player skill; BT-MLE is more stable for static LLMs.</li>
                      <li><strong>Core Formula:</strong> Pr(i &gt; j) = sigma(theta_i - theta_j) = exp(theta_i) / (exp(theta_i) + exp(theta_j)).</li>
                      <li><strong>Output:</strong> BT probabilities per category as OmniRank inputs.</li>
                    </ul>
                    <p className="mt-3 text-xs">
                      Reference: <a className="text-primary underline" href="https://lmsys.org/blog/2023-12-07-leaderboard/" target="_blank" rel="noreferrer">Chatbot Arena Elo system update</a>
                    </p>
                  </CardContent>
                </Card>

                <Card className="border-border/70 bg-card/65 py-5">
                  <CardHeader className="px-5 pb-0">
                    <CardTitle className="flex items-center gap-2 text-lg"><Rocket className="h-5 w-5 text-primary" /> Step 4: OmniRank</CardTitle>
                  </CardHeader>
                  <CardContent className="px-5 text-sm text-muted-foreground">
                    <p>
                      Selected virtual benchmark scores (1 to 7) are combined using the OmniRank method into one final robust leaderboard.
                    </p>
                    <ul className="mt-2 space-y-2">
                      <li><strong>Core Idea:</strong> a tournament network estimates a global power score (<code>theta.hat</code>).</li>
                      <li><strong>Uncertainty:</strong> weighted bootstrap simulations generate confidence intervals.</li>
                      <li><strong>Final Output:</strong> OmniRank with CI for statistically sound model comparison.</li>
                    </ul>
                  </CardContent>
                </Card>
              </div>
            </section>
          </>
        ) : (
          <>
            <section className={cn(sectionWrapperClass, "mb-8")}>
              <h2 className="mb-4 text-2xl font-semibold">Hugging Face LLM Rankings</h2>
              <div className="mb-4">
                <MetricOverview
                  methods={huggingFaceMethods}
                  benchmarkCount={selectedHfLabels.length}
                  benchmarkLabel="Benchmarks"
                  sourceName="Hugging Face"
                  sourceUrl="https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard"
                />
              </div>

              <Card className="mb-3 border-border/70 bg-card/65 py-4">
                <CardHeader className="px-4 pb-0">
                  <CardTitle className="text-lg">Select Benchmarks</CardTitle>
                  <CardDescription>
                    Choose which benchmarks to include in your OmniRank analysis. You must select between 2 and 6 benchmarks.
                  </CardDescription>
                </CardHeader>
                <CardContent className="px-4">
                  <div className="flex flex-wrap gap-3">
                    {HF_BENCHMARK_LABELS.map((label) => (
                      <label key={label} className="inline-flex items-center gap-2 rounded-md border border-border/60 bg-background/50 px-3 py-1.5 text-sm">
                        <input
                          type="checkbox"
                          checked={hfSelection[label]}
                          onChange={(event) => {
                            setHfSelection((previous) => ({
                              ...previous,
                              [label]: event.target.checked,
                            }));
                          }}
                        />
                        {label}
                      </label>
                    ))}
                  </div>
                  <div className="mt-3 text-xs text-muted-foreground">
                    Selection updates table automatically.
                    {hfLoading ? " Updating..." : ""}
                  </div>
                  {hfError ? <div className="mt-2 text-sm text-red-300">{hfError}</div> : null}
                </CardContent>
              </Card>

              <TopThreeLegend />
              <RankingTable
                key={`huggingface-${selectedHfLabels.join("|")}-${huggingFaceMethods.length}-${highlightModel ?? ""}`}
                mode="huggingface"
                methods={huggingFaceMethods}
                selectedLabels={selectedHfLabels}
                highlightModel={highlightModel}
              />
            </section>

            <section id="compare-with-your-model" className={cn(sectionWrapperClass, "mb-8")}>
              <h2 className="mb-4 text-2xl font-semibold">Compare With Your Model</h2>
              <Card className="border-border/70 bg-card/65 py-5">
                <CardHeader className="px-5 pb-0">
                  <CardTitle className="flex items-center gap-2 text-xl"><ChevronDown className="h-5 w-5 text-primary" /> Add Your Model</CardTitle>
                  <CardDescription>
                    Enter a model name and six benchmark scores (0-100). We re-rank against the current Top 100 locally in this page.
                  </CardDescription>
                </CardHeader>
                <CardContent className="px-5">
                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="md:col-span-2">
                      <Label htmlFor="custom-model-name">Model Name</Label>
                      <Input
                        id="custom-model-name"
                        placeholder="e.g., My-Awesome-LLM-7B"
                        value={customModelName}
                        onChange={(event) => setCustomModelName(event.target.value)}
                      />
                    </div>

                    <div>
                      <Label htmlFor="custom-ifeval">IFEval (%)</Label>
                      <Input
                        id="custom-ifeval"
                        value={customScores.ifeval}
                        onChange={(event) => setCustomScores((previous) => ({ ...previous, ifeval: event.target.value }))}
                      />
                    </div>
                    <div>
                      <Label htmlFor="custom-bbh">BBH (%)</Label>
                      <Input
                        id="custom-bbh"
                        value={customScores.bbh}
                        onChange={(event) => setCustomScores((previous) => ({ ...previous, bbh: event.target.value }))}
                      />
                    </div>
                    <div>
                      <Label htmlFor="custom-math">MATH (%)</Label>
                      <Input
                        id="custom-math"
                        value={customScores.math}
                        onChange={(event) => setCustomScores((previous) => ({ ...previous, math: event.target.value }))}
                      />
                    </div>
                    <div>
                      <Label htmlFor="custom-gpqa">GPQA (%)</Label>
                      <Input
                        id="custom-gpqa"
                        value={customScores.gpqa}
                        onChange={(event) => setCustomScores((previous) => ({ ...previous, gpqa: event.target.value }))}
                      />
                    </div>
                    <div>
                      <Label htmlFor="custom-musr">MUSR (%)</Label>
                      <Input
                        id="custom-musr"
                        value={customScores.musr}
                        onChange={(event) => setCustomScores((previous) => ({ ...previous, musr: event.target.value }))}
                      />
                    </div>
                    <div>
                      <Label htmlFor="custom-mmlu-pro">MMLU-Pro (%)</Label>
                      <Input
                        id="custom-mmlu-pro"
                        value={customScores.mmlu_pro}
                        onChange={(event) => setCustomScores((previous) => ({ ...previous, mmlu_pro: event.target.value }))}
                      />
                    </div>
                  </div>

                  <div className="mt-4 flex items-center gap-2">
                    <Button variant="outline" onClick={clearCustomInputs}>Clear</Button>
                    <Button onClick={runCustomRanking} disabled={customRunning}>Run OmniRank</Button>
                    {customError ? <span className="text-sm text-red-300">{customError}</span> : null}
                  </div>

                  {customRunning ? (
                    <Card className="mt-4 border-border/70 bg-background/40 py-4">
                      <CardHeader className="px-4 pb-0">
                        <CardTitle className="text-lg">Analyzing Your Model</CardTitle>
                        <CardDescription>
                          Running OmniRank algorithm... Progress: {customProgress.toFixed(0)}%
                        </CardDescription>
                      </CardHeader>
                      <CardContent className="px-4">
                        <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
                          <div className="h-full rounded-full bg-primary transition-all" style={{ width: `${customProgress}%` }} />
                        </div>
                      </CardContent>
                    </Card>
                  ) : null}

                  {customSummary ? (
                    <Card className="mt-4 border-border/70 bg-background/35 py-4">
                      <CardHeader className="px-4 pb-0">
                        <CardTitle className="text-xl">Your Model Summary</CardTitle>
                        <CardDescription>{customSummary.modelName}</CardDescription>
                      </CardHeader>
                      <CardContent className="px-4">
                        <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-4">
                          <Card className="gap-2 border-border/60 bg-card/60 py-4">
                            <CardHeader className="px-4 pb-0">
                              <CardDescription>OmniRank</CardDescription>
                              <CardTitle className="text-3xl text-primary">{customSummary.rank}</CardTitle>
                            </CardHeader>
                          </Card>
                          <Card className="gap-2 border-border/60 bg-card/60 py-4">
                            <CardHeader className="px-4 pb-0">
                              <CardDescription>Average Score Rank</CardDescription>
                              <CardTitle className="text-3xl">{customSummary.scoreRank}</CardTitle>
                            </CardHeader>
                          </Card>
                          <Card className="gap-2 border-border/60 bg-card/60 py-4">
                            <CardHeader className="px-4 pb-0">
                              <CardDescription>θ-hat Score</CardDescription>
                              <CardTitle className="text-2xl">{customSummary.thetaHat.toFixed(4)}</CardTitle>
                            </CardHeader>
                          </Card>
                          <Card className="gap-2 border-border/60 bg-card/60 py-4">
                            <CardHeader className="px-4 pb-0">
                              <CardDescription>95% CI</CardDescription>
                              <CardTitle className="text-2xl">[{customSummary.ciTwoSided[0]}, {customSummary.ciTwoSided[1]}]</CardTitle>
                            </CardHeader>
                          </Card>
                        </div>

                        <h4 className="mt-4 text-base font-semibold">Benchmark Performance</h4>
                        <div className="mt-2 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
                          {[
                            ["IFEval", customSummary.benchmarkScores.ifeval],
                            ["BBH", customSummary.benchmarkScores.bbh],
                            ["MATH", customSummary.benchmarkScores.math],
                            ["GPQA", customSummary.benchmarkScores.gpqa],
                            ["MUSR", customSummary.benchmarkScores.musr],
                            ["MMLU-Pro", customSummary.benchmarkScores.mmlu_pro],
                            ["Average", customSummary.benchmarkScores.average_score],
                          ].map(([label, value]) => (
                            <div key={label} className="rounded-md border border-border/50 bg-background/50 px-3 py-2 text-sm">
                              <div className="text-muted-foreground">{label}</div>
                              <div className="font-semibold text-foreground">{Number(value).toFixed(2)}%</div>
                            </div>
                          ))}
                        </div>
                      </CardContent>
                    </Card>
                  ) : null}
                </CardContent>
              </Card>
            </section>

            <WhatIsOmniRankSection />

            <section className={cn(sectionWrapperClass, "mb-8")}> 
              <h2 className="mb-4 text-2xl font-semibold">How This Leaderboard is Calculated</h2>
              <div className="grid gap-4 lg:grid-cols-2">
                <Card className="border-border/70 bg-card/65 py-5">
                  <CardHeader className="px-5 pb-0">
                    <CardTitle className="flex items-center gap-2 text-lg"><Download className="h-5 w-5 text-primary" /> Step 1: Data Collection & Preparation</CardTitle>
                  </CardHeader>
                  <CardContent className="px-5 text-sm text-muted-foreground">
                    <ul className="space-y-2">
                      <li><strong>Data Source:</strong> <a className="text-primary underline" href="https://huggingface.co/datasets/open-llm-leaderboard/requests" target="_blank" rel="noreferrer">Open LLM Leaderboard Dataset</a>.</li>
                      <li><strong>Data Cleaning & Selection:</strong> keep 6 core benchmark scores and key metadata, filter incomplete models, then take Top 100.</li>
                      <li><strong>Data Transformation:</strong> transform from model-per-row to a 6xN benchmark-vs-model matrix for OmniRank.</li>
                    </ul>
                  </CardContent>
                </Card>

                <Card className="border-border/70 bg-card/65 py-5">
                  <CardHeader className="px-5 pb-0">
                    <CardTitle className="flex items-center gap-2 text-lg"><Rocket className="h-5 w-5 text-primary" /> Step 2: OmniRank</CardTitle>
                  </CardHeader>
                  <CardContent className="px-5 text-sm text-muted-foreground">
                    <p>
                      Scores from selected benchmarks (1 to 6) are merged by the OmniRank method for a robust final ranking.
                    </p>
                    <ul className="mt-2 space-y-2">
                      <li><strong>Core Idea:</strong> benchmark comparisons form a tournament network that estimates model power score <code>theta.hat</code>.</li>
                      <li><strong>Uncertainty & Confidence:</strong> weighted bootstrap simulations generate rank confidence intervals.</li>
                      <li><strong>Final Output:</strong> OmniRank + CI provides statistically grounded leaderboard ordering.</li>
                    </ul>
                  </CardContent>
                </Card>
              </div>
            </section>
          </>
        )}

      </div>
    </main>
  );
}
