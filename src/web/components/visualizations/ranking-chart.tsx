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
}

const CHART_BG = "#070e19";

// Color scale from light blue (best) to deep blue (worst)
const getColor = (rank: number, total: number) => {
  const ratio = (rank - 1) / Math.max(1, total - 1);
  const r = Math.round(225 + ratio * (26 - 225));
  const g = Math.round(239 + ratio * (66 - 239));
  const b = Math.round(255 + ratio * (115 - 255));
  return `rgb(${r}, ${g}, ${b})`;
};

// Custom tooltip component
const CustomTooltip = ({ active, payload }: { active?: boolean; payload?: unknown[] }) => {
  if (!active || !payload || !payload.length) return null;

  const data = (payload[0] as { payload: RankingItem }).payload;
  const ciLeft = Math.round(data.ci_two_sided[0]);
  const ciRight = Math.round(data.ci_two_sided[1]);
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="bg-card/95 backdrop-blur-sm border border-border rounded-lg p-3 shadow-lg"
    >
      <p className="font-semibold text-foreground">{data.name}</p>
      <div className="mt-2 space-y-1 text-sm">
        <p className="text-muted-foreground">
          Rank: <span className="text-primary font-mono">#{data.rank}</span>
        </p>
        <p className="text-muted-foreground">
          Score: <span className="text-foreground font-mono">{data.theta_hat.toFixed(4)}</span>
        </p>
        <p className="text-muted-foreground">
          Confidence Interval: <span className="text-foreground font-mono">[{ciLeft}, {ciRight}]</span>
        </p>
      </div>
    </motion.div>
  );
};

export function RankingChart({ items, className }: RankingChartProps) {
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
      <ResponsiveContainer width="100%" height={chartHeight - 24}>
        <BarChart
          data={chartData}
          layout="vertical"
          margin={{ top: 12, right: 28, left: 16, bottom: 20 }}
          barCategoryGap={8}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="rgba(255,255,255,0.45)"
            opacity={0.45}
          />
          <XAxis
            type="number"
            domain={[minScore, maxScore]}
            tick={{ fill: "#f7fbff", fontSize: 12, fontWeight: 700 }}
            axisLine={{ stroke: "rgba(255,255,255,0.8)" }}
            tickLine={{ stroke: "rgba(255,255,255,0.8)" }}
            label={{
              value: "Score (θ̂)",
              position: "bottom",
              fill: "#f7fbff",
              fontSize: 12,
              fontWeight: 700,
            }}
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
          <ReferenceLine
            x={0}
            stroke="rgba(255,255,255,0.9)"
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
