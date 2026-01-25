"use client";

import { useState, useEffect } from "react";
import { Play, Loader2, ExternalLink } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { cn } from "@/lib/utils";
import type { InferredSchema, AnalysisConfig } from "@/lib/api";

interface RankingPreviewBubbleProps {
  schema: InferredSchema;
  onStartAnalysis: (config: AnalysisConfig) => void;
  isAnalyzing?: boolean;
  className?: string;
}

// Timeline step component
function TimelineStep({
  title,
  children,
  isLast = false,
}: {
  title: string;
  children: React.ReactNode;
  isLast?: boolean;
}) {
  return (
    <div className="flex gap-3">
      {/* Timeline indicator */}
      <div className="flex flex-col items-center">
        <div className="w-2.5 h-2.5 rounded-full bg-muted-foreground/40" />
        {!isLast && <div className="w-0.5 flex-1 bg-muted-foreground/20 mt-1.5" />}
      </div>
      {/* Content */}
      <div className="flex-1 pb-4">
        <h4 className="text-sm font-semibold mb-2">{title}</h4>
        <div className="space-y-1.5">{children}</div>
      </div>
    </div>
  );
}

// Config row component
function ConfigRow({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-muted-foreground whitespace-nowrap">{label}:</span>
      <div className="text-xs font-mono bg-muted/60 px-2 py-0.5 rounded">{children}</div>
    </div>
  );
}

export function RankingPreviewBubble({
  schema,
  onStartAnalysis,
  isAnalyzing = false,
  className,
}: RankingPreviewBubbleProps) {
  const [bigbetter, setBigbetter] = useState<0 | 1>(1);
  const [bootstrapIterations, setBootstrapIterations] = useState(2000);
  const [randomSeed, setRandomSeed] = useState(42);
  const [expanded, setExpanded] = useState(false);

  // Sync bigbetter with schema when it changes
  useEffect(() => {
    if (schema) {
      setBigbetter(schema.bigbetter as 0 | 1);
    }
  }, [schema]);

  const handleStartAnalysis = () => {
    const config: AnalysisConfig = {
      bigbetter,
      bootstrap_iterations: bootstrapIterations,
      random_seed: randomSeed,
    };
    onStartAnalysis(config);
  };

  // Calculate estimated runtime
  const estimatedRuntime = `~${Math.max(1, Math.ceil(schema.ranking_items.length * 0.3))} seconds`;

  return (
    <div className={cn(
      "bg-white dark:bg-zinc-800 border border-border/50 rounded-2xl rounded-bl-sm shadow-sm max-w-sm",
      className
    )}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border/40">
        <h3 className="text-base font-semibold">Ranking Preview</h3>
        <button
          onClick={() => setExpanded(!expanded)}
          className="text-muted-foreground hover:text-foreground transition-colors"
        >
          <ExternalLink className="h-4 w-4" />
        </button>
      </div>

      {/* Content */}
      <div className="px-4 py-3">
        {/* DataQuality */}
        <TimelineStep title="DataQuality">
          <ConfigRow label="Missing values">0%</ConfigRow>
          <ConfigRow label="Data quality">Great, ready to run perfectly</ConfigRow>
          <ConfigRow label="Estimated runtime">{estimatedRuntime}</ConfigRow>
        </TimelineStep>

        {/* RankingConfig */}
        <TimelineStep title="RankingConfig">
          <ConfigRow label="Ranking items number">{schema.ranking_items.length}</ConfigRow>
          <div className="space-y-1">
            <span className="text-xs text-muted-foreground">Ranking items name:</span>
            <div className="text-xs font-mono bg-muted/60 px-2 py-1 rounded leading-relaxed break-words">
              {schema.ranking_items.join(", ")}
            </div>
          </div>
        </TimelineStep>

        {/* ParameterSetup */}
        <TimelineStep title="ParameterSetup" isLast>
          {/* Ranking direction */}
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">Ranking direction:</span>
            {expanded ? (
              <div className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground">
                  {bigbetter === 1 ? "Higher" : "Lower"}
                </span>
                <Switch
                  checked={bigbetter === 1}
                  onCheckedChange={(checked) => setBigbetter(checked ? 1 : 0)}
                  className="scale-75"
                />
              </div>
            ) : (
              <span className="text-xs font-mono bg-muted/60 px-2 py-0.5 rounded">...</span>
            )}
          </div>

          {/* Bootstrap iterations */}
          {expanded ? (
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">Bootstrap iterations:</span>
                <span className="text-xs font-mono bg-muted/60 px-2 py-0.5 rounded">
                  {bootstrapIterations}
                </span>
              </div>
              <Slider
                value={[bootstrapIterations]}
                onValueChange={([value]) => setBootstrapIterations(value)}
                min={100}
                max={5000}
                step={100}
                className="w-full"
              />
            </div>
          ) : (
            <ConfigRow label="Bootstrap iterations">{bootstrapIterations}</ConfigRow>
          )}

          {/* Random seed */}
          {expanded ? (
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">Random seed:</span>
                <span className="text-xs font-mono bg-muted/60 px-2 py-0.5 rounded">
                  {randomSeed}
                </span>
              </div>
              <Slider
                value={[randomSeed]}
                onValueChange={([value]) => setRandomSeed(value)}
                min={1}
                max={9999}
                step={1}
                className="w-full"
              />
            </div>
          ) : (
            <ConfigRow label="Random seed">{randomSeed}</ConfigRow>
          )}
        </TimelineStep>
      </div>

      {/* Start Ranking Button */}
      <div className="px-4 pb-4">
        <Button
          onClick={handleStartAnalysis}
          disabled={isAnalyzing}
          variant="outline"
          className="w-full"
          size="lg"
        >
          {isAnalyzing ? (
            <>
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              Analyzing...
            </>
          ) : (
            <>
              <Play className="h-4 w-4 mr-2" />
              Start Ranking
            </>
          )}
        </Button>
      </div>
    </div>
  );
}
