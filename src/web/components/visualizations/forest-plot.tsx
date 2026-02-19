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
}

const CHART_BG = "#132841";
const AXIS_COLOR = "#e2e8f0";

// Custom tooltip component for Forest Plot
const CustomTooltip = ({ active, payload }: { active?: boolean; payload?: unknown[] }) => {
  if (!active || !payload || !payload.length) return null;

  const data = (payload[0] as { payload: ForestPlotDataItem }).payload;
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="bg-card/95 backdrop-blur-sm border border-border rounded-lg p-3 shadow-lg"
    >
      <p className="font-semibold text-foreground">{data.name}</p>
      <div className="mt-2 space-y-1 text-sm">
        <p className="text-muted-foreground">
          Point Estimate: <span className="text-primary font-mono">#{data.rank}</span>
        </p>
        <p className="text-muted-foreground">
          Confidence Interval: <span className="text-foreground font-mono">[{data.ci_lower}, {data.ci_upper}]</span>
        </p>
        <p className="text-muted-foreground">
          CI Width: <span className="text-foreground font-mono">{data.ci_width}</span>
        </p>
        <p className="text-muted-foreground">
          Score (θ̂): <span className="text-foreground font-mono">{data.theta_hat.toFixed(4)}</span>
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
export function ForestPlot({ items, className }: ForestPlotProps) {
  // Prepare data for the forest plot - sort by rank
  const chartData: ForestPlotDataItem[] = useMemo(() => {
    return [...items]
      .sort((a, b) => a.rank - b.rank)
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

  return (
    <div
      className={className}
      style={{
        width: "100%",
        minHeight: chartHeight,
        backgroundColor: CHART_BG,
        borderRadius: 12,
        padding: 12,
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
            stroke="rgba(226,232,240,0.25)"
            opacity={0.9}
            horizontal={true}
            vertical={true}
          />
          <XAxis
            type="number"
            domain={[minRank, maxRank]}
            tick={{ fill: "#e2e8f0", fontSize: 12, fontWeight: 600 }}
            axisLine={{ stroke: "rgba(226,232,240,0.5)" }}
            tickLine={{ stroke: "rgba(226,232,240,0.5)" }}
            label={{
              value: "Rank (95% CI)",
              position: "bottom",
              fill: "#e2e8f0",
              fontSize: 12,
              fontWeight: 600,
              offset: 16,
            }}
            tickFormatter={(value) => Math.round(value).toString()}
          />
          <YAxis
            type="category"
            dataKey="name"
            tick={{ fill: "#f1f5f9", fontSize: 12, fontWeight: 600 }}
            axisLine={{ stroke: "rgba(226,232,240,0.5)" }}
            tickLine={{ stroke: "rgba(226,232,240,0.5)" }}
            width={yAxisWidth}
          />
          <Tooltip content={<CustomTooltip />} />

          {/* Reference line at median rank */}
          <ReferenceLine
            x={(items.length + 1) / 2}
            stroke="rgba(226,232,240,0.6)"
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
              stroke={AXIS_COLOR}
              strokeWidth={3}
              opacity={0.8}
            />
          ))}

          {/* Point estimates (diamonds) with rank labels */}
          <Scatter
            dataKey="rank"
            fill={AXIS_COLOR}
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
                    fill={AXIS_COLOR}
                    stroke="#132841"
                    strokeWidth={1.5}
                  />
                  <text
                    x={cx}
                    y={cy - 14}
                    textAnchor="middle"
                    dominantBaseline="auto"
                    fill={AXIS_COLOR}
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
      <div className="mt-2 flex items-center justify-center gap-6 text-xs font-semibold text-slate-300">
        <div className="flex items-center gap-2">
          <svg width="16" height="16" viewBox="0 0 16 16">
            <polygon
              points="8,2 14,8 8,14 2,8"
              fill={AXIS_COLOR}
              stroke="rgba(226,232,240,0.6)"
              strokeWidth={1}
            />
          </svg>
          <span>Point Estimate</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="h-1 w-4 rounded bg-slate-400/90" />
          <span>95% Confidence Interval</span>
        </div>
      </div>
    </div>
  );
}
