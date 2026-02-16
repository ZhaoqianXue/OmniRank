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

const CHART_BG = "#070e19";

// Color scale from light blue (best) to deep blue (worst)
const getColor = (rank: number, total: number) => {
  const ratio = (rank - 1) / Math.max(1, total - 1);
  const r = Math.round(225 + ratio * (28 - 225));
  const g = Math.round(239 + ratio * (68 - 239));
  const b = Math.round(255 + ratio * (116 - 255));
  return `rgb(${r}, ${g}, ${b})`;
};

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
          margin={{ top: 12, right: 28, left: 16, bottom: 26 }}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="rgba(255,255,255,0.45)"
            opacity={0.45}
            horizontal={true}
            vertical={true}
          />
          <XAxis
            type="number"
            domain={[minRank, maxRank]}
            tick={{ fill: "#f7fbff", fontSize: 12, fontWeight: 700 }}
            axisLine={{ stroke: "rgba(255,255,255,0.8)" }}
            tickLine={{ stroke: "rgba(255,255,255,0.8)" }}
            label={{
              value: "Rank (95% CI)",
              position: "bottom",
              fill: "#f7fbff",
              fontSize: 12,
              fontWeight: 700,
              offset: 16,
            }}
            tickFormatter={(value) => Math.round(value).toString()}
          />
          <YAxis
            type="category"
            dataKey="name"
            tick={{ fill: "#ffffff", fontSize: 12, fontWeight: 700 }}
            axisLine={{ stroke: "rgba(255,255,255,0.8)" }}
            tickLine={{ stroke: "rgba(255,255,255,0.8)" }}
            width={yAxisWidth}
          />
          <Tooltip content={<CustomTooltip />} />

          {/* Reference line at median rank */}
          <ReferenceLine
            x={(items.length + 1) / 2}
            stroke="rgba(255,255,255,0.92)"
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
              stroke={getColor(item.rank, items.length)}
              strokeWidth={3}
              opacity={0.7}
            />
          ))}

          {/* Point estimates (diamonds) */}
          <Scatter
            dataKey="rank"
            fill="hsl(var(--foreground))"
            shape={(props: { cx?: number; cy?: number; payload?: ForestPlotDataItem }) => {
              const cx = props.cx ?? 0;
              const cy = props.cy ?? 0;
              const payload = props.payload;
              if (!payload) {
                return null;
              }
              const color = getColor(payload.rank, items.length);
              // Diamond shape for point estimate
              return (
                <g>
                  <polygon
                    points={`${cx},${cy - 8} ${cx + 6},${cy} ${cx},${cy + 8} ${cx - 6},${cy}`}
                    fill={color}
                    stroke="hsl(var(--background))"
                    strokeWidth={1.5}
                  />
                </g>
              );
            }}
          />
        </ComposedChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div className="mt-2 flex items-center justify-center gap-6 text-xs font-semibold text-[#f8fbff]">
        <div className="flex items-center gap-2">
          <svg width="16" height="16" viewBox="0 0 16 16">
            <polygon
              points="8,2 14,8 8,14 2,8"
              fill="#f8fbff"
              stroke="rgba(8,34,61,0.75)"
              strokeWidth={1}
            />
          </svg>
          <span>Point Estimate</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="h-1 w-4 rounded bg-[#f8fbff]/90" />
          <span>95% Confidence Interval</span>
        </div>
      </div>
    </div>
  );
}
