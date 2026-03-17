"use client";

import { useMemo } from "react";
import {
  ComposedChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { motion } from "framer-motion";
import type { RankingItem } from "@/lib/api";

interface ForestPlotProps {
  items: RankingItem[];
  className?: string;
  theme?: "dark" | "light";
}

const CHART_BG = "#132841";
const CHART_BG_LIGHT = "#ffffff";
const AXIS_COLOR = "#e2e8f0";
const AXIS_COLOR_LIGHT = "#0f172a";
const PREFERRED_PRS_METHOD_ORDER = [
  "C+T",
  "LDpred",
  "lassosum",
  "PRS-CS",
  "PRS-CS-auto",
  "SBayesR",
  "SCT",
  "DBSLMM",
  "LDpred2",
  "LDpred2-auto",
  "LDpred2-inf",
  "LDpred-funct",
  "lassosum2",
];
const PREFERRED_PRS_METHOD_SET = new Set(PREFERRED_PRS_METHOD_ORDER);

function applyPreferredPrsMethodOrder<T extends { name: string }>(items: T[]): T[] {
  const indexByName = new Map(items.map((item, index) => [item.name, index]));
  const preferredIndices = PREFERRED_PRS_METHOD_ORDER
    .map((name) => indexByName.get(name))
    .filter((index): index is number => index !== undefined);
  if (preferredIndices.length < 2) return items;
  return [
    ...preferredIndices.map((index) => items[index]),
    ...items.filter((item) => !PREFERRED_PRS_METHOD_SET.has(item.name)),
  ];
}

// Custom tooltip component for Forest Plot
const CustomTooltip = ({
  active,
  payload,
  theme = "dark",
}: {
  active?: boolean;
  payload?: unknown[];
  theme?: "dark" | "light";
}) => {
  if (!active || !payload || !payload.length) return null;

  const isLightTheme = theme === "light";
  const data = (payload[0] as { payload: ForestPlotDataItem }).payload;
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className={
        isLightTheme
          ? "bg-white border border-slate-300 rounded-lg p-3 shadow-lg"
          : "bg-card/95 backdrop-blur-sm border border-border rounded-lg p-3 shadow-lg"
      }
    >
      <p className={isLightTheme ? "font-semibold text-slate-900" : "font-semibold text-foreground"}>{data.name}</p>
      <div className="mt-2 space-y-1 text-sm">
        <p className={isLightTheme ? "text-slate-700" : "text-muted-foreground"}>
          Point Estimate: <span className={isLightTheme ? "text-slate-900 font-mono" : "text-primary font-mono"}>#{data.rank}</span>
        </p>
        <p className={isLightTheme ? "text-slate-700" : "text-muted-foreground"}>
          Confidence Interval:{" "}
          <span className={isLightTheme ? "text-slate-900 font-mono" : "text-foreground font-mono"}>
            [{data.ci_lower}, {data.ci_upper}]
          </span>
        </p>
        <p className={isLightTheme ? "text-slate-700" : "text-muted-foreground"}>
          CI Width: <span className={isLightTheme ? "text-slate-900 font-mono" : "text-foreground font-mono"}>{data.ci_width}</span>
        </p>
        <p className={isLightTheme ? "text-slate-700" : "text-muted-foreground"}>
          Score (θ̂):{" "}
          <span className={isLightTheme ? "text-slate-900 font-mono" : "text-foreground font-mono"}>
            {data.theta_hat.toFixed(4)}
          </span>
        </p>
      </div>
    </motion.div>
  );
};

interface ForestPlotDataItem {
  name: string;
  rank: number;
  theta_hat: number;
  ci_lower: number;
  ci_upper: number;
  ci_width: number;
  // For rendering CI line
  ciRange: [number, number];
}

/**
 * Forest Plot - displays ranking confidence intervals
 * Common visualization in statistical analysis for showing point estimates with CIs
 */
export function ForestPlot({ items, className, theme = "dark" }: ForestPlotProps) {
  const isLightTheme = theme === "light";

  // Prepare data for the forest plot - sort by rank
  const chartData: ForestPlotDataItem[] = useMemo(() => {
    return applyPreferredPrsMethodOrder([...items].sort((a, b) => a.rank - b.rank))
      .map((item) => {
        const ciLower = Math.round(item.ci_two_sided[0]);
        const ciUpper = Math.round(item.ci_two_sided[1]);
        return {
          name: item.name,
          rank: item.rank,
          theta_hat: item.theta_hat,
          ci_lower: ciLower,
          ci_upper: ciUpper,
          ci_width: ciUpper - ciLower,
          ciRange: [ciLower, ciUpper] as [number, number],
        };
      });
  }, [items]);

  // Calculate domain for X axis (rank-based)
  const { minRank, maxRank } = useMemo(() => {
    return {
      minRank: 0.5,
      maxRank: items.length + 0.5,
    };
  }, [items]);

  // Dynamic height based on number of items
  const chartHeight = Math.max(320, items.length * 42 + 96);
  const yAxisWidth = useMemo(() => {
    const maxNameLength = items.reduce((max, item) => Math.max(max, item.name.length), 0);
    return Math.min(220, Math.max(90, maxNameLength * 7 + 18));
  }, [items]);

  const chartBg = isLightTheme ? CHART_BG_LIGHT : CHART_BG;
  const axisColor = isLightTheme ? AXIS_COLOR_LIGHT : AXIS_COLOR;
  const gridStroke = isLightTheme ? "rgba(15,23,42,0.12)" : "rgba(226,232,240,0.25)";
  const axisStroke = isLightTheme ? "rgba(15,23,42,0.30)" : "rgba(226,232,240,0.5)";
  const referenceLineStroke = isLightTheme ? "rgba(15,23,42,0.45)" : "rgba(226,232,240,0.6)";
  const markerStroke = isLightTheme ? "#ffffff" : "#132841";

  return (
    <div
      className={className}
      style={{
        width: "100%",
        minHeight: chartHeight,
        backgroundColor: chartBg,
        borderRadius: 12,
        padding: 12,
        border: isLightTheme ? "1px solid rgba(15,23,42,0.16)" : undefined,
      }}
    >
      <ResponsiveContainer width="100%" height={chartHeight - 52}>
        <ComposedChart
          data={chartData}
          layout="vertical"
          margin={{ top: 24, right: 28, left: 16, bottom: 26 }}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            stroke={gridStroke}
            opacity={0.9}
            horizontal={true}
            vertical={true}
          />
          <XAxis
            type="number"
            domain={[minRank, maxRank]}
            tick={{ fill: axisColor, fontSize: 12, fontWeight: 600 }}
            axisLine={{ stroke: axisStroke }}
            tickLine={{ stroke: axisStroke }}
            label={{
              value: "Rank (95% CI)",
              position: "bottom",
              fill: axisColor,
              fontSize: 12,
              fontWeight: 600,
              offset: 16,
            }}
            tickFormatter={(value) => Math.round(value).toString()}
          />
          <YAxis
            type="category"
            dataKey="name"
            tick={{ fill: axisColor, fontSize: 12, fontWeight: 600 }}
            axisLine={{ stroke: axisStroke }}
            tickLine={{ stroke: axisStroke }}
            width={yAxisWidth}
          />
          <Tooltip content={<CustomTooltip theme={theme} />} />

          {/* Reference line at median rank */}
          <ReferenceLine
            x={(items.length + 1) / 2}
            stroke={referenceLineStroke}
            strokeDasharray="3 3"
            opacity={0.9}
          />

          {/* CI bars - rendered as horizontal lines for each item */}
          {chartData.map((item) => (
            <ReferenceLine
              key={`ci-${item.name}`}
              segment={[
                { x: item.ci_lower, y: item.name },
                { x: item.ci_upper, y: item.name },
              ]}
              stroke={axisColor}
              strokeWidth={3}
              opacity={0.8}
            />
          ))}

          {/* Point estimates (diamonds) with rank labels */}
          <Scatter
            dataKey="rank"
            fill={axisColor}
            shape={(props: { cx?: number; cy?: number; payload?: ForestPlotDataItem }) => {
              const cx = props.cx ?? 0;
              const cy = props.cy ?? 0;
              const payload = props.payload;
              if (!payload) {
                return null;
              }
              // Diamond shape for point estimate, with rank label above - both white like axes
              return (
                <g>
                  <polygon
                    points={`${cx},${cy - 8} ${cx + 6},${cy} ${cx},${cy + 8} ${cx - 6},${cy}`}
                    fill={axisColor}
                    stroke={markerStroke}
                    strokeWidth={1.5}
                  />
                  <text
                    x={cx}
                    y={cy - 14}
                    textAnchor="middle"
                    dominantBaseline="auto"
                    fill={axisColor}
                    fontSize={11}
                    fontWeight={600}
                  >
                    {payload.rank}
                  </text>
                </g>
              );
            }}
          />
        </ComposedChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div
        className={
          isLightTheme
            ? "mt-2 flex items-center justify-center gap-6 text-xs font-semibold text-slate-600"
            : "mt-2 flex items-center justify-center gap-6 text-xs font-semibold text-slate-300"
        }
      >
        <div className="flex items-center gap-2">
          <svg width="16" height="16" viewBox="0 0 16 16">
            <polygon
              points="8,2 14,8 8,14 2,8"
              fill={axisColor}
              stroke={isLightTheme ? "rgba(15,23,42,0.40)" : "rgba(226,232,240,0.6)"}
              strokeWidth={1}
            />
          </svg>
          <span>Point Estimate</span>
        </div>
        <div className="flex items-center gap-2">
          <div className={isLightTheme ? "h-1 w-4 rounded bg-slate-500/90" : "h-1 w-4 rounded bg-slate-400/90"} />
          <span>95% Confidence Interval</span>
        </div>
      </div>
    </div>
  );
}
