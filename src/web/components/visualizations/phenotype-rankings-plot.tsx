"use client";

import { motion } from "framer-motion";
import { useMemo, useRef, useState, type MouseEvent } from "react";
import { cn } from "@/lib/utils";
import type { PlotSpec } from "@/lib/api";

interface PhenotypeRankingsPlotProps {
  plot: PlotSpec;
  className?: string;
  theme?: "dark" | "light";
}

interface TooltipState {
  x: number;
  y: number;
  title: string;
  lines: string[];
}

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

function parseMatrixRows(value: unknown): Array<Array<number | null>> | null {
  if (!Array.isArray(value)) return null;
  const rows: Array<Array<number | null>> = [];
  for (const rowValue of value) {
    if (!Array.isArray(rowValue)) return null;
    const row: Array<number | null> = [];
    for (const cellValue of rowValue) {
      if (cellValue == null) {
        row.push(null);
        continue;
      }
      const parsed = parseFiniteNumber(cellValue);
      row.push(parsed);
    }
    rows.push(row);
  }
  return rows;
}

function interpolate(start: [number, number, number], end: [number, number, number], ratio: number): string {
  const bounded = Math.max(0, Math.min(1, ratio));
  const red = Math.round(start[0] + (end[0] - start[0]) * bounded);
  const green = Math.round(start[1] + (end[1] - start[1]) * bounded);
  const blue = Math.round(start[2] + (end[2] - start[2]) * bounded);
  return `rgb(${red}, ${green}, ${blue})`;
}

function rankColor(value: number, rankMin: number, rankMax: number): string {
  if (rankMax <= rankMin) return "rgb(245,158,11)";
  const ratio = (value - rankMin) / (rankMax - rankMin);
  // Lower rank (better) -> warm orange, higher rank (worse) -> blue.
  if (ratio <= 0.5) {
    return interpolate([245, 158, 11], [107, 70, 193], ratio * 2);
  }
  return interpolate([107, 70, 193], [30, 64, 175], (ratio - 0.5) * 2);
}

export function PhenotypeRankingsPlot({ plot, className, theme = "dark" }: PhenotypeRankingsPlotProps) {
  const isLightTheme = theme === "light";
  const containerRef = useRef<HTMLDivElement>(null);
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);

  const parsed = useMemo(() => {
    const methods = parseStringArray(plot.data["methods"]) ?? [];
    const indicatorValues = parseStringArray(plot.data["indicator_values"]) ?? [];
    const matrixRows = parseMatrixRows(plot.data["matrix_rows"]) ?? [];
    const rankMin = parseFiniteNumber(plot.data["rank_min"]) ?? 1;
    const rankMax = parseFiniteNumber(plot.data["rank_max"]) ?? Math.max(2, methods.length);
    return { methods, indicatorValues, matrixRows, rankMin, rankMax };
  }, [plot.data]);

  const hasData =
    parsed.methods.length > 0 &&
    parsed.indicatorValues.length > 0 &&
    parsed.matrixRows.length === parsed.indicatorValues.length;

  const cellW = 26;
  const cellH = 20;
  const leftMargin = 28;
  const topMargin = 16;
  const bottomMargin = 92;

  const methodCount = Math.max(1, parsed.methods.length);
  const rowCount = Math.max(1, parsed.indicatorValues.length);
  const maxLabelLength = parsed.indicatorValues.reduce((max, value) => Math.max(max, value.length), 8);
  const rowLabelWidth = Math.min(460, Math.max(220, maxLabelLength * 7 + 20));

  const heatmapLeft = leftMargin;
  const heatmapTop = topMargin;
  const heatmapWidth = methodCount * cellW;
  const heatmapHeight = rowCount * cellH;

  const rowLabelX = heatmapLeft + heatmapWidth + 10;
  const colorbarX = rowLabelX + rowLabelWidth + 20;
  const colorbarWidth = 16;
  const colorbarHeight = Math.max(120, Math.min(220, heatmapHeight - 4));
  const colorbarY = heatmapTop + Math.max(0, (heatmapHeight - colorbarHeight) / 2);
  const yAxisLabelX = colorbarX + colorbarWidth + 44;

  const width = yAxisLabelX + 24;
  const height = heatmapTop + heatmapHeight + bottomMargin;

  const panelBg = isLightTheme ? "#ebebeb" : "#132841";
  const axisColor = isLightTheme ? "#1f2937" : "#ffffff";
  const gridStroke = isLightTheme ? "#c7c7c7" : "rgba(255,255,255,0.35)";
  const panelBorder = isLightTheme ? "#b8b8b8" : "rgba(255,255,255,0.45)";
  const missingFill = isLightTheme ? "#bfbfbf" : "rgba(148,163,184,0.55)";

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
        backgroundColor: panelBg,
        borderRadius: 10,
        padding: 8,
        border: `1px solid ${panelBorder}`,
      }}
    >
      {!hasData ? (
        <div className={cn("text-center text-sm py-8", isLightTheme ? "text-slate-700" : "text-white")}>
          No phenotype ranking heatmap data available.
        </div>
      ) : (
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto" role="img" aria-label="Phenotype rankings heatmap">
          <rect x={0} y={0} width={width} height={height} fill={panelBg} />

          {parsed.indicatorValues.map((indicator, rowIndex) => {
            const rowY = heatmapTop + rowIndex * cellH;
            const row = parsed.matrixRows[rowIndex] ?? [];

            return (
              <g key={`row-${indicator}`}>
                {parsed.methods.map((method, colIndex) => {
                  const x = heatmapLeft + colIndex * cellW;
                  const value = row[colIndex];
                  const fill = typeof value === "number" ? rankColor(value, parsed.rankMin, parsed.rankMax) : missingFill;

                  return (
                    <rect
                      key={`cell-${indicator}-${method}`}
                      x={x}
                      y={rowY}
                      width={cellW}
                      height={cellH}
                      fill={fill}
                      stroke={gridStroke}
                      strokeWidth={0.6}
                      onMouseMove={(event) =>
                        showTooltip(event, `${indicator} | ${method}`, [
                          typeof value === "number"
                            ? `Normalized rank: ${value.toFixed(2)}`
                            : "Normalized rank: unavailable",
                        ])
                      }
                      onMouseLeave={hideTooltip}
                    />
                  );
                })}

                <text x={rowLabelX} y={rowY + cellH / 2 + 4} textAnchor="start" fontSize={11} fill={axisColor} fontWeight={600}>
                  {indicator}
                </text>
              </g>
            );
          })}

          {parsed.methods.map((method, colIndex) => {
            const x = heatmapLeft + colIndex * cellW + cellW * 0.5;
            const y = heatmapTop + heatmapHeight + 14;
            return (
              <text
                key={`method-${method}`}
                x={x}
                y={y}
                textAnchor="start"
                fontSize={10.5}
                fill={axisColor}
                fontWeight={600}
                transform={`rotate(45 ${x} ${y})`}
              >
                {method}
              </text>
            );
          })}

          {Array.from({ length: 40 }).map((_, index) => {
            const ratio = index / 39;
            const value = parsed.rankMin + ratio * (parsed.rankMax - parsed.rankMin);
            const y = colorbarY + ratio * colorbarHeight;
            return (
              <rect
                key={`bar-${index}`}
                x={colorbarX}
                y={y}
                width={colorbarWidth}
                height={colorbarHeight / 40 + 1}
                fill={rankColor(value, parsed.rankMin, parsed.rankMax)}
                stroke="none"
              />
            );
          })}

          {[parsed.rankMin, (parsed.rankMin + parsed.rankMax) / 2, parsed.rankMax].map((tick) => {
            const ratio = (tick - parsed.rankMin) / Math.max(1e-9, parsed.rankMax - parsed.rankMin);
            const y = colorbarY + ratio * colorbarHeight;
            return (
              <text key={`tick-${tick}`} x={colorbarX + colorbarWidth + 8} y={y + 4} textAnchor="start" fontSize={11} fill={axisColor} fontWeight={600}>
                {Number.isInteger(tick) ? `${tick}` : tick.toFixed(2)}
              </text>
            );
          })}

          <text x={colorbarX} y={colorbarY - 8} textAnchor="start" fontSize={11} fill={axisColor} fontWeight={600}>
            Rank scale
          </text>

          <text
            x={yAxisLabelX}
            y={heatmapTop + heatmapHeight / 2}
            textAnchor="middle"
            fontSize={12}
            fill={axisColor}
            fontWeight={600}
            transform={`rotate(-90 ${yAxisLabelX} ${heatmapTop + heatmapHeight / 2})`}
          >
            Phenotypes
          </text>

          <text x={heatmapLeft + heatmapWidth / 2} y={height - 20} textAnchor="middle" fontSize={12} fill={axisColor} fontWeight={600}>
            Methods
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
