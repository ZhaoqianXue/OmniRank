"use client";

import { motion } from "framer-motion";
import { useMemo, useRef, useState, type MouseEvent } from "react";
import { cn } from "@/lib/utils";
import type { PlotSpec } from "@/lib/api";

interface NormalizedRankingPlotProps {
  plot: PlotSpec;
  className?: string;
  theme?: "dark" | "light";
}

interface PointDatum {
  method: string;
  indicator: string;
  rank: number;
}

interface MethodSummary {
  method: string;
  values: number[];
  q1: number;
  median: number;
  q3: number;
  low: number;
  high: number;
  mean: number;
  k: number;
}

interface TooltipState {
  x: number;
  y: number;
  title: string;
  lines: string[];
}

const METHOD_COLORS = [
  "#000000",
  "#0f766e",
  "#0ea5b7",
  "#ec4899",
  "#f472b6",
  "#6d28d9",
  "#2563eb",
  "#a855f7",
  "#60a5fa",
  "#93c5fd",
  "#b91c1c",
  "#8b5a2b",
  "#d97706",
  "#f59e0b",
  "#22c55e",
  "#06b6d4",
];

function parseStringArray(value: unknown): string[] | null {
  if (!Array.isArray(value)) return null;
  const out: string[] = [];
  for (const entry of value) {
    if (typeof entry !== "string") return null;
    out.push(entry);
  }
  return out;
}

function parseFiniteNumber(value: unknown): number | null {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function parseMatrix(value: unknown): Record<string, Record<string, number>> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const matrix: Record<string, Record<string, number>> = {};

  for (const [rowKey, rowValue] of Object.entries(value)) {
    if (!rowValue || typeof rowValue !== "object" || Array.isArray(rowValue)) continue;
    const parsedRow: Record<string, number> = {};
    for (const [colKey, cellValue] of Object.entries(rowValue)) {
      const parsed = parseFiniteNumber(cellValue);
      if (parsed === null) continue;
      parsedRow[colKey] = parsed;
    }
    matrix[rowKey] = parsedRow;
  }

  return matrix;
}

function quantile(values: number[], q: number): number {
  if (values.length === 0) return Number.NaN;
  if (values.length === 1) return values[0];
  const sorted = [...values].sort((a, b) => a - b);
  const index = (sorted.length - 1) * q;
  const low = Math.floor(index);
  const high = Math.ceil(index);
  if (low === high) return sorted[low];
  const ratio = index - low;
  return sorted[low] * (1 - ratio) + sorted[high] * ratio;
}

function stableJitter(key: string, amplitude: number): number {
  let hash = 2166136261 >>> 0;
  for (let i = 0; i < key.length; i += 1) {
    hash ^= key.charCodeAt(i);
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  const normalized = hash / 0xffffffff;
  return (normalized * 2 - 1) * amplitude;
}

/** Build a half violin path. side: 'left' = extends left from center, 'right' = extends right from center. */
function buildHalfViolinPath(
  values: number[],
  xCenter: number,
  yToPx: (rank: number) => number,
  maxHalfWidth: number,
  rankMin: number,
  rankMax: number,
  side: "left" | "right",
): string {
  if (values.length === 0) return "";

  const sorted = [...values].sort((a, b) => a - b);
  const mean = sorted.reduce((acc, value) => acc + value, 0) / sorted.length;
  const variance = sorted.reduce((acc, value) => acc + (value - mean) * (value - mean), 0) / sorted.length;
  const std = Math.sqrt(Math.max(0, variance));

  const kdeValues = std < 0.08 ? sorted.flatMap((value) => [value - 0.28, value, value + 0.28]) : sorted;
  const kdeSorted = [...kdeValues].sort((a, b) => a - b);

  const low = kdeSorted[0];
  const high = kdeSorted[kdeSorted.length - 1];
  const iqr = quantile(kdeSorted, 0.75) - quantile(kdeSorted, 0.25);
  const bandwidth = Math.max(0.28, Math.min(1.8, iqr > 0 ? iqr * 0.4 : 0.5));

  const rawSpan = high - low;
  const effectiveSpan = Math.max(rawSpan + bandwidth * 2.0, 0.9);
  const center = (low + high) / 2;
  const sampleLow = Math.max(rankMin, center - effectiveSpan / 2);
  const sampleHigh = Math.min(rankMax, center + effectiveSpan / 2);
  const sampleCount = 48;

  const sampleRanks: number[] = [];
  const densities: number[] = [];
  for (let i = 0; i < sampleCount; i += 1) {
    const rankValue = sampleLow + ((sampleHigh - sampleLow) * i) / Math.max(1, sampleCount - 1);
    sampleRanks.push(rankValue);

    let density = 0;
    for (const value of kdeSorted) {
      const z = (value - rankValue) / bandwidth;
      density += Math.exp(-0.5 * z * z);
    }
    densities.push(density);
  }

  const peak = Math.max(...densities, 1e-9);
  const sign = side === "left" ? -1 : 1;
  let path = `M ${xCenter.toFixed(2)} ${yToPx(sampleRanks[0]).toFixed(2)}`;

  for (let i = 1; i < sampleRanks.length; i += 1) {
    const widthRatio = densities[i] / peak;
    const halfWidth = maxHalfWidth * (0.08 + 0.92 * widthRatio);
    const x = xCenter + sign * halfWidth;
    const y = yToPx(sampleRanks[i]);
    path += ` L ${x.toFixed(2)} ${y.toFixed(2)}`;
  }

  path += ` L ${xCenter.toFixed(2)} ${yToPx(sampleRanks[sampleRanks.length - 1]).toFixed(2)} Z`;

  return path;
}

function formatRank(value: number): string {
  return Number.isInteger(value) ? `${value}` : value.toFixed(2);
}

export function NormalizedRankingPlot({ plot, className, theme = "dark" }: NormalizedRankingPlotProps) {
  const isLightTheme = theme === "light";
  const containerRef = useRef<HTMLDivElement>(null);
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);

  const parsed = useMemo(() => {
    const methods = parseStringArray(plot.data["methods"]) ?? [];
    const indicatorValues = parseStringArray(plot.data["indicator_values"]) ?? [];
    const matrix = parseMatrix(plot.data["matrix"]) ?? {};
    const rankMin = parseFiniteNumber(plot.data["rank_min"]) ?? 1;
    const rankMax = parseFiniteNumber(plot.data["rank_max"]) ?? Math.max(2, methods.length);

    const points: PointDatum[] = [];
    const summaries: MethodSummary[] = [];

    for (const method of methods) {
      const values: number[] = [];
      for (const indicator of indicatorValues) {
        const rank = matrix[indicator]?.[method];
        if (!Number.isFinite(rank)) continue;
        values.push(rank);
        points.push({ method, indicator, rank });
      }
      if (values.length === 0) continue;

      const sorted = [...values].sort((a, b) => a - b);
      summaries.push({
        method,
        values: sorted,
        q1: quantile(sorted, 0.25),
        median: quantile(sorted, 0.5),
        q3: quantile(sorted, 0.75),
        low: sorted[0],
        high: sorted[sorted.length - 1],
        mean: sorted.reduce((acc, value) => acc + value, 0) / sorted.length,
        k: sorted.length,
      });
    }

    return { methods, points, summaries, rankMin, rankMax };
  }, [plot.data]);

  const hasData = parsed.methods.length > 0 && parsed.summaries.length > 0;
  const methodCount = Math.max(1, parsed.methods.length);
  const rankTickStart = Math.ceil(parsed.rankMin);
  const rankTickEnd = Math.floor(parsed.rankMax);
  const rankTickCount = Math.max(1, rankTickEnd - rankTickStart + 1);

  const leftMargin = 72;
  const rightMargin = 20;
  const topMargin = 18;
  const bottomMargin = 102;
  const width = 980;
  const height = Math.max(430, 280 + rankTickCount * 22);
  const plotLeft = leftMargin;
  const plotRight = width - rightMargin;
  const plotTop = topMargin;
  const plotBottom = height - bottomMargin;
  const plotHeight = plotBottom - plotTop;
  const step = (plotRight - plotLeft) / methodCount;

  const rankSpan = Math.max(1e-9, parsed.rankMax - parsed.rankMin);
  const yToPx = (rank: number) => plotTop + ((rank - parsed.rankMin) / rankSpan) * plotHeight;

  const panelBg = isLightTheme ? "#ebebeb" : "#132841";
  const axisColor = isLightTheme ? "#1f2937" : "#ffffff";
  const gridStroke = isLightTheme ? "#d0d0d0" : "rgba(255,255,255,0.35)";
  const panelBorder = isLightTheme ? "#b8b8b8" : "rgba(255,255,255,0.45)";
  const boxFill = isLightTheme ? "#ffffff" : "rgba(255,255,255,0.16)";
  const labelFill = isLightTheme ? "#f5f5f5" : "rgba(15,23,42,0.90)";
  const labelStroke = isLightTheme ? "#6b7280" : "rgba(255,255,255,0.45)";
  const meanFill = "#b91c1c";
  const meanStroke = "#7f1d1d";

  const showTooltip = (event: MouseEvent<SVGElement>, title: string, lines: string[]) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    setTooltip({
      x: event.clientX - rect.left + 12,
      y: event.clientY - rect.top + 12,
      title,
      lines,
    });
  };

  const hideTooltip = () => setTooltip(null);

  const tooltipClassName = isLightTheme
    ? "pointer-events-none absolute z-20 min-w-44 rounded-lg border border-slate-300 bg-white p-3 shadow-lg"
    : "pointer-events-none absolute z-20 min-w-44 rounded-lg border border-border bg-card/95 p-3 shadow-lg backdrop-blur-sm";
  const tooltipTitleClassName = isLightTheme ? "font-semibold text-slate-900" : "font-semibold text-foreground";
  const tooltipLineClassName = isLightTheme ? "text-xs text-slate-700" : "text-xs text-muted-foreground";

  return (
    <div
      ref={containerRef}
      className={cn(className)}
      style={{
        width: "100%",
        maxHeight: "80vh",
        overflowX: "hidden",
        overflowY: "auto",
        backgroundColor: panelBg,
        borderRadius: 10,
        padding: 8,
        border: `1px solid ${panelBorder}`,
      }}
    >
      {!hasData ? (
        <div className={cn("text-center text-sm py-8", isLightTheme ? "text-slate-700" : "text-white")}>
          No deep normalized ranking data available.
        </div>
      ) : (
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Normalized ranking over individual phenotypes">
          <rect x={0} y={0} width={width} height={height} fill={panelBg} />

          <line x1={plotLeft} y1={plotBottom} x2={plotRight} y2={plotBottom} stroke={panelBorder} strokeWidth={1.2} />

          {Array.from({ length: Math.max(0, rankTickEnd - rankTickStart + 1) }).map((_, idx) => {
            const tick = rankTickStart + idx;
            if (tick <= parsed.rankMin || tick >= parsed.rankMax) return null;
            const y = yToPx(tick);
            return (
              <g key={`y-tick-${tick}`}>
                <line x1={plotLeft} y1={y} x2={plotRight} y2={y} stroke={gridStroke} strokeWidth={1} />
                <text x={plotLeft - 10} y={y + 4} textAnchor="end" fontSize={17} fill={axisColor} fontWeight={600}>
                  {tick}
                </text>
              </g>
            );
          })}

          {parsed.summaries.map((summary, index) => {
            const methodColor = METHOD_COLORS[index % METHOD_COLORS.length];
            const xCenter = plotLeft + step * (index + 0.5);
            const boxWidth = Math.max(18, Math.min(30, step * 0.34));
            const violinHalfW = Math.max(6, Math.min(14, step * 0.14));
            const yLow = yToPx(summary.low);
            const yHigh = yToPx(summary.high);
            const yQ1 = yToPx(summary.q1);
            const yQ3 = yToPx(summary.q3);
            const yMedian = yToPx(summary.median);
            const yMean = yToPx(summary.mean);
            const violinOffset = violinHalfW + 8;
            const violinSpineX = xCenter + violinOffset;
            const violinPath = buildHalfViolinPath(
              summary.values,
              violinSpineX,
              yToPx,
              violinHalfW,
              parsed.rankMin,
              parsed.rankMax,
              "right",
            );

            const labelWidth = 100;
            const labelX = Math.max(plotLeft, Math.min(plotRight - labelWidth, xCenter + 12));
            const labelY = yMean - 10;

            return (
              <g key={`method-${summary.method}`}>
                <line x1={xCenter} y1={plotTop} x2={xCenter} y2={plotBottom} stroke={gridStroke} strokeWidth={0.8} />

                {violinPath ? (
                  <path
                    d={violinPath}
                    fill={methodColor}
                    fillOpacity={isLightTheme ? 0.35 : 0.45}
                    stroke={axisColor}
                    strokeOpacity={0.8}
                    strokeWidth={1.2}
                    onMouseMove={(event) =>
                      showTooltip(event, summary.method, [
                        `Distribution width reflects density over phenotype-specific ranks`,
                        `Range: ${formatRank(summary.low)}-${formatRank(summary.high)}`,
                      ])
                    }
                    onMouseLeave={hideTooltip}
                  />
                ) : null}

                <line x1={xCenter} y1={yLow} x2={xCenter} y2={yHigh} stroke={axisColor} strokeWidth={1.4} />
                <line x1={xCenter - boxWidth / 2} y1={yLow} x2={xCenter + boxWidth / 2} y2={yLow} stroke={axisColor} strokeWidth={1.4} />
                <line x1={xCenter - boxWidth / 2} y1={yHigh} x2={xCenter + boxWidth / 2} y2={yHigh} stroke={axisColor} strokeWidth={1.4} />

                <rect
                  x={xCenter - boxWidth / 2}
                  y={Math.min(yQ1, yQ3)}
                  width={boxWidth}
                  height={Math.max(1, Math.abs(yQ3 - yQ1))}
                  fill={isLightTheme ? "#ffffff" : "#cbd5e1"}
                  stroke={axisColor}
                  strokeWidth={1.5}
                  onMouseMove={(event) =>
                    showTooltip(event, summary.method, [
                      `Q1: ${formatRank(summary.q1)}`,
                      `Median: ${formatRank(summary.median)}`,
                      `Q3: ${formatRank(summary.q3)}`,
                    ])
                  }
                  onMouseLeave={hideTooltip}
                />
                <line x1={xCenter - boxWidth / 2} y1={yMedian} x2={xCenter + boxWidth / 2} y2={yMedian} stroke={axisColor} strokeWidth={1.6} />

                {parsed.points
                  .filter((point) => point.method === summary.method)
                  .map((point) => {
                    const jitterXAmplitude = Math.max(2, step * 0.04);
                    const jitterYAmplitude = Math.max(0.08, (parsed.rankMax - parsed.rankMin) * 0.06);
                    const jitterX = stableJitter(`${point.method}|${point.indicator}`, jitterXAmplitude);
                    const jitterY = stableJitter(`Y|${point.method}|${point.indicator}`, jitterYAmplitude);
                    const pointX = xCenter + jitterX;
                    const pointY = yToPx(point.rank + jitterY);
                    return (
                      <circle
                        key={`point-${point.method}-${point.indicator}`}
                        cx={pointX}
                        cy={pointY}
                        r={4.0}
                        fill={methodColor}
                        fillOpacity={0.95}
                        stroke={panelBg}
                        strokeWidth={0.8}
                        onMouseMove={(event) =>
                          showTooltip(event, point.method, [
                            `Phenotype: ${point.indicator}`,
                            `Normalized rank: ${formatRank(point.rank)}`,
                          ])
                        }
                        onMouseLeave={hideTooltip}
                      />
                    );
                  })}

                <circle
                  cx={xCenter}
                  cy={yMean}
                  r={7.5}
                  fill={meanFill}
                  stroke={isLightTheme ? "#000000" : "#000000"}
                  strokeWidth={1.8}
                  onMouseMove={(event) =>
                    showTooltip(event, summary.method, [
                      `Mean rank: ${summary.mean.toFixed(2)}`,
                      `Q1-Q3: ${formatRank(summary.q1)}-${formatRank(summary.q3)}`,
                      `K: ${summary.k}`,
                    ])
                  }
                  onMouseLeave={hideTooltip}
                />

                <rect
                  x={labelX}
                  y={labelY}
                  width={labelWidth}
                  height={20}
                  rx={2.5}
                  ry={2.5}
                  fill={labelFill}
                  fillOpacity={0.03125}
                  stroke={labelStroke}
                  strokeWidth={1}
                />
                <text textAnchor="start" fill={axisColor} fontWeight={600}>
                  <tspan x={labelX + 6} y={labelY + 14} fontSize={17}>μ </tspan>
                  <tspan x={labelX + 8.5} y={labelY + 9} fontSize={16}>^</tspan>
                  <tspan x={labelX + 16} y={labelY + 18} fontSize={12.5}> mean</tspan>
                  <tspan x={labelX + 40} y={labelY + 14} fontSize={17}>{` = ${summary.mean.toFixed(2)}`}</tspan>
                </text>

                <text x={xCenter} y={plotBottom + 24} textAnchor="middle" fontSize={16.5} fill={axisColor} fontWeight={600}>
                  {summary.method}
                </text>
                <text x={xCenter} y={plotBottom + 46} textAnchor="middle" fontSize={16} fill={axisColor} fontWeight={600}>
                  {`(K = ${summary.k})`}
                </text>
              </g>
            );
          })}

          <text x={(plotLeft + plotRight) / 2} y={height - 20} textAnchor="middle" fontSize={18} fill={axisColor} fontWeight={600}>
            Methods
          </text>
          <text
            x={20}
            y={(plotTop + plotBottom) / 2}
            textAnchor="middle"
            fontSize={18}
            fill={axisColor}
            fontWeight={600}
            transform={`rotate(-90 20 ${(plotTop + plotBottom) / 2})`}
          >
            Ranking
          </text>
        </svg>
      )}

      {tooltip ? (
        <motion.div
          initial={{ opacity: 0, scale: 0.96 }}
          animate={{ opacity: 1, scale: 1 }}
          className={tooltipClassName}
          style={{ left: tooltip.x, top: tooltip.y }}
        >
          <p className={tooltipTitleClassName}>{tooltip.title}</p>
          <div className="mt-2 space-y-1">
            {tooltip.lines.map((line, index) => (
              <p key={`tip-line-${index}`} className={tooltipLineClassName}>
                {line}
              </p>
            ))}
          </div>
        </motion.div>
      ) : null}
    </div>
  );
}
