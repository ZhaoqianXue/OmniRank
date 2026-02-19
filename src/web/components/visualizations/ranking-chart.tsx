"use client";

import { useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from "recharts";
import { motion } from "framer-motion";
import type { RankingItem } from "@/lib/api";

interface RankingChartProps {
  items: RankingItem[];
  className?: string;
  theme?: "dark" | "light";
}

const CHART_BG = "#132841";
const CHART_BG_LIGHT = "#ffffff";

// Color scale from light blue (best) to deep blue (worst)
const getColor = (rank: number, total: number) => {
  const ratio = (rank - 1) / Math.max(1, total - 1);
  const r = Math.round(225 + ratio * (26 - 225));
  const g = Math.round(239 + ratio * (66 - 239));
  const b = Math.round(255 + ratio * (115 - 255));
  return `rgb(${r}, ${g}, ${b})`;
};

// Custom tooltip component
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
  const data = (payload[0] as { payload: RankingItem }).payload;
  const ciLeft = Math.round(data.ci_two_sided[0]);
  const ciRight = Math.round(data.ci_two_sided[1]);
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
          Rank: <span className={isLightTheme ? "text-slate-900 font-mono" : "text-primary font-mono"}>#{data.rank}</span>
        </p>
        <p className={isLightTheme ? "text-slate-700" : "text-muted-foreground"}>
          Score: <span className={isLightTheme ? "text-slate-900 font-mono" : "text-foreground font-mono"}>{data.theta_hat.toFixed(4)}</span>
        </p>
        <p className={isLightTheme ? "text-slate-700" : "text-muted-foreground"}>
          Confidence Interval: <span className={isLightTheme ? "text-slate-900 font-mono" : "text-foreground font-mono"}>[{ciLeft}, {ciRight}]</span>
        </p>
      </div>
    </motion.div>
  );
};

export function RankingChart({ items, className, theme = "dark" }: RankingChartProps) {
  const isLightTheme = theme === "light";

  // Prepare data for the chart - sort by rank
  const chartData = useMemo(() => {
    return [...items].sort((a, b) => a.rank - b.rank);
  }, [items]);

  const chartHeight = useMemo(() => Math.max(300, items.length * 42 + 72), [items.length]);
  const yAxisWidth = useMemo(() => {
    const maxNameLength = items.reduce((max, item) => Math.max(max, item.name.length), 0);
    return Math.min(220, Math.max(90, maxNameLength * 7 + 18));
  }, [items]);

  // Calculate domain for Y axis
  const { minScore, maxScore } = useMemo(() => {
    if (items.length === 0) {
      return { minScore: -1, maxScore: 1 };
    }
    const scores = items.map((i) => i.theta_hat);
    const min = Math.min(...scores);
    const max = Math.max(...scores);
    const span = Math.max(0.2, max - min);
    const padding = span * 0.12;
    return {
      minScore: min - padding,
      maxScore: max + padding,
    };
  }, [items]);

  const chartBg = isLightTheme ? CHART_BG_LIGHT : CHART_BG;
  const axisTickColor = isLightTheme ? "#0f172a" : "#e2e8f0";
  const yAxisTickColor = isLightTheme ? "#111827" : "#f1f5f9";
  const gridStroke = isLightTheme ? "rgba(15,23,42,0.12)" : "rgba(226,232,240,0.25)";
  const axisStroke = isLightTheme ? "rgba(15,23,42,0.30)" : "rgba(226,232,240,0.5)";
  const referenceLineStroke = isLightTheme ? "rgba(15,23,42,0.45)" : "rgba(226,232,240,0.6)";

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
      <ResponsiveContainer width="100%" height={chartHeight - 24}>
        <BarChart
          data={chartData}
          layout="vertical"
          margin={{ top: 12, right: 28, left: 16, bottom: 20 }}
          barCategoryGap={8}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            stroke={gridStroke}
            opacity={0.9}
          />
          <XAxis
            type="number"
            domain={[minScore, maxScore]}
            tick={{ fill: axisTickColor, fontSize: 12, fontWeight: 600 }}
            axisLine={{ stroke: axisStroke }}
            tickLine={{ stroke: axisStroke }}
            label={{
              value: "Score (θ̂)",
              position: "bottom",
              fill: axisTickColor,
              fontSize: 12,
              fontWeight: 600,
            }}
          />
          <YAxis
            type="category"
            dataKey="name"
            tick={{ fill: yAxisTickColor, fontSize: 12, fontWeight: 600 }}
            axisLine={{ stroke: axisStroke }}
            tickLine={{ stroke: axisStroke }}
            width={yAxisWidth}
          />
          <Tooltip content={<CustomTooltip theme={theme} />} />
          <ReferenceLine
            x={0}
            stroke={referenceLineStroke}
            strokeDasharray="3 3"
            opacity={0.9}
          />
          <Bar
            dataKey="theta_hat"
            radius={[0, 4, 4, 0]}
            animationDuration={800}
            animationEasing="ease-out"
            barSize={22}
          >
            {chartData.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={getColor(entry.rank, items.length)}
                fillOpacity={0.95}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
