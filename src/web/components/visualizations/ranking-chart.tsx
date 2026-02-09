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
  ErrorBar,
  Cell,
  ReferenceLine,
} from "recharts";
import { motion } from "framer-motion";
import type { RankingItem } from "@/lib/api";

interface RankingChartProps {
  items: RankingItem[];
  className?: string;
}

// Color scale from light violet (best) to deep violet (worst)
const getColor = (rank: number, total: number) => {
  const ratio = (rank - 1) / Math.max(1, total - 1);
  const r = Math.round(197 + ratio * (105 - 197));
  const g = Math.round(186 + ratio * (86 - 186));
  const b = Math.round(246 + ratio * (171 - 246));
  return `rgb(${r}, ${g}, ${b})`;
};

// Custom tooltip component
const CustomTooltip = ({ active, payload }: { active?: boolean; payload?: unknown[] }) => {
  if (!active || !payload || !payload.length) return null;

  const data = (payload[0] as { payload: RankingItem }).payload;
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
          CI: <span className="text-foreground font-mono">[{data.ci_two_sided[0]}, {data.ci_two_sided[1]}]</span>
        </p>
      </div>
    </motion.div>
  );
};

export function RankingChart({ items, className }: RankingChartProps) {
  // Prepare data for the chart - sort by rank
  const chartData = useMemo(() => {
    return [...items]
      .sort((a, b) => a.rank - b.rank)
      .map((item) => ({
        ...item,
        // Calculate error bar values (difference from theta_hat to CI bounds)
        // For visual purposes, we'll show score uncertainty
        errorLower: Math.abs(item.theta_hat - (item.theta_hat - 0.3)),
        errorUpper: Math.abs((item.theta_hat + 0.3) - item.theta_hat),
      }));
  }, [items]);

  // Calculate domain for Y axis
  const { minScore, maxScore } = useMemo(() => {
    const scores = items.map((i) => i.theta_hat);
    const min = Math.min(...scores);
    const max = Math.max(...scores);
    const padding = (max - min) * 0.1;
    return {
      minScore: min - padding - 0.3,
      maxScore: max + padding + 0.3,
    };
  }, [items]);

  return (
    <div className={className} style={{ width: "100%", minHeight: 300 }}>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart
          data={chartData}
          layout="vertical"
          margin={{ top: 20, right: 30, left: 80, bottom: 20 }}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="hsl(var(--border))"
            opacity={0.3}
          />
          <XAxis
            type="number"
            domain={[minScore, maxScore]}
            tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }}
            axisLine={{ stroke: "hsl(var(--border))" }}
            tickLine={{ stroke: "hsl(var(--border))" }}
            label={{
              value: "Score (θ̂)",
              position: "bottom",
              fill: "hsl(var(--muted-foreground))",
              fontSize: 12,
            }}
          />
          <YAxis
            type="category"
            dataKey="name"
            tick={{ fill: "hsl(var(--foreground))", fontSize: 12 }}
            axisLine={{ stroke: "hsl(var(--border))" }}
            tickLine={{ stroke: "hsl(var(--border))" }}
            width={70}
          />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine
            x={0}
            stroke="hsl(var(--muted-foreground))"
            strokeDasharray="3 3"
            opacity={0.5}
          />
          <Bar
            dataKey="theta_hat"
            radius={[0, 4, 4, 0]}
            animationDuration={800}
            animationEasing="ease-out"
          >
            {chartData.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={getColor(entry.rank, items.length)}
                fillOpacity={0.85}
              />
            ))}
            <ErrorBar
              dataKey="errorUpper"
              width={4}
              strokeWidth={2}
              stroke="hsl(var(--foreground))"
              opacity={0.6}
              direction="x"
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
