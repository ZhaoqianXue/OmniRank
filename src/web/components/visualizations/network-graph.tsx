"use client";

import { useRef, useEffect, useState, useMemo } from "react";
import dynamic from "next/dynamic";
import { motion } from "framer-motion";
import type { RankingResults } from "@/lib/api";

// Dynamic import to avoid SSR issues with react-force-graph
const ForceGraph2D = dynamic(
  () => import("react-force-graph-2d"),
  {
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center h-full">
        <div className="text-muted-foreground">Loading graph...</div>
      </div>
    ),
  }
);

interface NetworkGraphProps {
  results: RankingResults;
  className?: string;
}

// Color scale based on rank
const getNodeColor = (rank: number, total: number) => {
  const ratio = (rank - 1) / Math.max(1, total - 1);
  const r = Math.round(197 + ratio * (105 - 197));
  const g = Math.round(186 + ratio * (86 - 186));
  const b = Math.round(246 + ratio * (171 - 246));
  return `rgb(${r}, ${g}, ${b})`;
};

// Link color based on win rate
const getLinkColor = (winRate: number) => {
  const alpha = Math.abs(winRate - 0.5) * 2;
  return `rgba(152, 132, 229, ${0.1 + alpha * 0.4})`;
};

export function NetworkGraph({ results, className }: NetworkGraphProps) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const graphRef = useRef<any>(null);
  const [dimensions, setDimensions] = useState({ width: 400, height: 400 });
  const containerRef = useRef<HTMLDivElement>(null);

  // Update dimensions on resize
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const { width, height } = containerRef.current.getBoundingClientRect();
        setDimensions({ width, height });
      }
    };

    updateDimensions();
    window.addEventListener("resize", updateDimensions);
    return () => window.removeEventListener("resize", updateDimensions);
  }, []);

  // Build graph data
  const graphData = useMemo(() => {
    const nodes = results.items.map((item) => ({
      id: item.name,
      name: item.name,
      rank: item.rank,
      score: item.theta_hat,
      val: 15 - item.rank * 1.5,
      color: getNodeColor(item.rank, results.items.length),
    }));

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const links: any[] = [];

    // Build links from pairwise matrix or generate from scores
    if (results.pairwise_matrix.length > 0) {
      for (const pair of results.pairwise_matrix) {
        if (Math.abs(pair.win_rate_a - 0.5) > 0.05) {
          links.push({
            source: pair.item_a,
            target: pair.item_b,
            value: Math.abs(pair.win_rate_a - 0.5) * 2,
            winRate: pair.win_rate_a,
          });
        }
      }
    } else {
      // Generate links from score differences
      for (let i = 0; i < results.items.length; i++) {
        for (let j = i + 1; j < results.items.length; j++) {
          const itemA = results.items[i];
          const itemB = results.items[j];
          const diff = Math.abs(itemA.theta_hat - itemB.theta_hat);

          if (diff > 0.3) {
            const winRate = itemA.theta_hat > itemB.theta_hat ? 0.7 : 0.3;
            links.push({
              source: itemA.name,
              target: itemB.name,
              value: Math.min(1, diff / 2),
              winRate,
            });
          }
        }
      }
    }

    return { nodes, links };
  }, [results]);

  // Custom node rendering
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const nodeCanvasObject = (node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
    const label = node.name;
    const fontSize = 12 / globalScale;
    const nodeSize = node.val;

    // Draw node circle with glow
    ctx.beginPath();
    ctx.arc(node.x, node.y, nodeSize, 0, 2 * Math.PI);
    ctx.fillStyle = node.color;
    ctx.shadowColor = node.color;
    ctx.shadowBlur = 15;
    ctx.fill();
    ctx.shadowBlur = 0;

    // Draw border
    ctx.strokeStyle = "rgba(255, 255, 255, 0.3)";
    ctx.lineWidth = 1 / globalScale;
    ctx.stroke();

    // Draw label
    ctx.font = `${fontSize}px sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
    ctx.fillText(label, node.x, node.y + nodeSize + fontSize);

    // Draw rank badge
    ctx.font = `bold ${fontSize}px sans-serif`;
    ctx.fillStyle = "#0b101e";
    ctx.fillText(`#${node.rank}`, node.x, node.y);
  };

  // Custom link rendering
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const linkCanvasObject = (link: any, ctx: CanvasRenderingContext2D) => {
    const source = link.source;
    const target = link.target;

    if (!source.x || !source.y || !target.x || !target.y) return;

    ctx.beginPath();
    ctx.moveTo(source.x, source.y);
    ctx.lineTo(target.x, target.y);
    ctx.strokeStyle = getLinkColor(link.winRate || 0.5);
    ctx.lineWidth = (link.value || 0.5) * 3;
    ctx.stroke();
  };

  return (
    <motion.div
      ref={containerRef}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className={className}
      style={{ width: "100%", height: "100%" }}
    >
      <ForceGraph2D
        ref={graphRef}
        graphData={graphData}
        width={dimensions.width}
        height={dimensions.height}
        backgroundColor="transparent"
        nodeCanvasObject={nodeCanvasObject}
        linkCanvasObject={linkCanvasObject}
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        nodePointerAreaPaint={(node: any, color: string, ctx: CanvasRenderingContext2D) => {
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(node.x, node.y, node.val + 5, 0, 2 * Math.PI);
          ctx.fill();
        }}
        linkDirectionalParticles={2}
        linkDirectionalParticleWidth={2}
        linkDirectionalParticleSpeed={0.005}
        d3AlphaDecay={0.02}
        d3VelocityDecay={0.3}
        cooldownTime={3000}
        onEngineStop={() => {
          if (graphRef.current?.centerAt) {
            graphRef.current.centerAt(0, 0, 500);
          }
        }}
      />
    </motion.div>
  );
}
