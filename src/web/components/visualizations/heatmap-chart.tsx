"use client";

import { useMemo } from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import type { RankingItem, RankingResults } from "@/lib/api";

interface HeatmapChartProps {
  results: RankingResults;
  className?: string;
}

// Color interpolation for heatmap
const getHeatmapColor = (value: number) => {
  // value: 0 (lose) -> 0.5 (tie) -> 1 (win)
  // Color scale: Purple (lose) -> Gray (tie) -> Cyan (win)
  if (value < 0.5) {
    // Purple to gray
    const ratio = value * 2;
    const r = Math.round(139 + ratio * (128 - 139));
    const g = Math.round(92 + ratio * (128 - 92));
    const b = Math.round(246 + ratio * (128 - 246));
    return `rgb(${r}, ${g}, ${b})`;
  } else {
    // Gray to cyan
    const ratio = (value - 0.5) * 2;
    const r = Math.round(128 + ratio * (0 - 128));
    const g = Math.round(128 + ratio * (240 - 128));
    const b = Math.round(128 + ratio * (255 - 128));
    return `rgb(${r}, ${g}, ${b})`;
  }
};

interface MatrixCell {
  rowItem: string;
  colItem: string;
  winRate: number;
  nComparisons: number;
}

export function HeatmapChart({ results, className }: HeatmapChartProps) {
  // Build matrix from pairwise data or generate from items
  const { matrix, items } = useMemo(() => {
    const itemNames = results.items.map((i) => i.name).sort((a, b) => {
      const rankA = results.items.find((i) => i.name === a)?.rank ?? 0;
      const rankB = results.items.find((i) => i.name === b)?.rank ?? 0;
      return rankA - rankB;
    });

    // Create matrix data
    const matrixData: MatrixCell[][] = [];
    
    // Build lookup from pairwise_matrix
    const pairwiseLookup = new Map<string, { winRate: number; nComparisons: number }>();
    for (const pair of results.pairwise_matrix) {
      pairwiseLookup.set(`${pair.item_a}|${pair.item_b}`, {
        winRate: pair.win_rate_a,
        nComparisons: pair.n_comparisons,
      });
      // Reverse pair
      pairwiseLookup.set(`${pair.item_b}|${pair.item_a}`, {
        winRate: 1 - pair.win_rate_a,
        nComparisons: pair.n_comparisons,
      });
    }

    // Generate matrix
    for (let i = 0; i < itemNames.length; i++) {
      const row: MatrixCell[] = [];
      for (let j = 0; j < itemNames.length; j++) {
        if (i === j) {
          // Diagonal - same item
          row.push({
            rowItem: itemNames[i],
            colItem: itemNames[j],
            winRate: 0.5,
            nComparisons: 0,
          });
        } else {
          const key = `${itemNames[i]}|${itemNames[j]}`;
          const pair = pairwiseLookup.get(key);
          if (pair) {
            row.push({
              rowItem: itemNames[i],
              colItem: itemNames[j],
              winRate: pair.winRate,
              nComparisons: pair.nComparisons,
            });
          } else {
            // Generate synthetic win rate from scores
            const scoreI = results.items.find((item) => item.name === itemNames[i])?.theta_hat ?? 0;
            const scoreJ = results.items.find((item) => item.name === itemNames[j])?.theta_hat ?? 0;
            // Sigmoid-like conversion from score difference to win probability
            const diff = scoreI - scoreJ;
            const winRate = 1 / (1 + Math.exp(-diff));
            row.push({
              rowItem: itemNames[i],
              colItem: itemNames[j],
              winRate,
              nComparisons: 0,
            });
          }
        }
      }
      matrixData.push(row);
    }

    return { matrix: matrixData, items: itemNames };
  }, [results]);

  const cellSize = Math.min(60, 400 / items.length);

  return (
    <div className={cn("flex flex-col items-center", className)}>
      {/* Legend */}
      <div className="flex items-center gap-4 mb-4 text-xs text-muted-foreground">
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 rounded" style={{ backgroundColor: getHeatmapColor(0) }} />
          <span>Row loses</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 rounded" style={{ backgroundColor: getHeatmapColor(0.5) }} />
          <span>Tie</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 rounded" style={{ backgroundColor: getHeatmapColor(1) }} />
          <span>Row wins</span>
        </div>
      </div>

      {/* Matrix container */}
      <div className="overflow-auto max-w-full">
        <div className="inline-block">
          {/* Column headers */}
          <div className="flex" style={{ marginLeft: cellSize + 8 }}>
            {items.map((item, idx) => (
              <div
                key={`col-${idx}`}
                className="text-xs text-muted-foreground overflow-hidden"
                style={{
                  width: cellSize,
                  height: cellSize,
                  writingMode: "vertical-rl",
                  textOrientation: "mixed",
                  transform: "rotate(180deg)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "flex-end",
                  paddingBottom: 4,
                }}
              >
                <span className="truncate">{item}</span>
              </div>
            ))}
          </div>

          {/* Matrix rows */}
          {matrix.map((row, rowIdx) => (
            <div key={`row-${rowIdx}`} className="flex items-center">
              {/* Row header */}
              <div
                className="text-xs text-muted-foreground text-right pr-2 truncate"
                style={{ width: cellSize + 8 }}
              >
                {items[rowIdx]}
              </div>
              
              {/* Cells */}
              {row.map((cell, colIdx) => (
                <motion.div
                  key={`cell-${rowIdx}-${colIdx}`}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: (rowIdx * items.length + colIdx) * 0.01 }}
                  className={cn(
                    "flex items-center justify-center border border-border/20 relative group cursor-pointer",
                    rowIdx === colIdx && "opacity-30"
                  )}
                  style={{
                    width: cellSize,
                    height: cellSize,
                    backgroundColor: getHeatmapColor(cell.winRate),
                  }}
                >
                  {/* Value on hover */}
                  <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 bg-black/60 transition-opacity">
                    <span className="text-xs font-mono text-white">
                      {rowIdx === colIdx ? "-" : (cell.winRate * 100).toFixed(0) + "%"}
                    </span>
                  </div>
                </motion.div>
              ))}
            </div>
          ))}
        </div>
      </div>

      {/* Caption */}
      <p className="text-xs text-muted-foreground mt-4 text-center">
        Cell color shows row item&apos;s win rate against column item
      </p>
    </div>
  );
}
