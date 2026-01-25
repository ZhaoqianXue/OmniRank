"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Play, Settings, Info, ChevronDown, ChevronUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { InferredSchema, AnalysisConfig } from "@/lib/api";

interface ConfigPanelProps {
  schema: InferredSchema;
  onStartAnalysis: (config: AnalysisConfig) => void;
  isAnalyzing?: boolean;
  className?: string;
}

export function ConfigPanel({
  schema,
  onStartAnalysis,
  isAnalyzing = false,
  className,
}: ConfigPanelProps) {
  const [bigbetter, setBigbetter] = useState<0 | 1>(schema.bigbetter as 0 | 1);
  const [bootstrapIterations, setBootstrapIterations] = useState(2000);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [randomSeed, setRandomSeed] = useState(42);

  const handleStartAnalysis = () => {
    const config: AnalysisConfig = {
      bigbetter,
      bootstrap_iterations: bootstrapIterations,
      random_seed: randomSeed,
    };
    onStartAnalysis(config);
  };

  return (
    <Card className={cn("bg-card/80 backdrop-blur-sm", className)}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Settings className="h-5 w-5 text-primary" />
            <CardTitle className="text-lg">Analysis Configuration</CardTitle>
          </div>
          <Badge variant="outline" className="text-xs">
            {schema.format}
          </Badge>
        </div>
        <CardDescription>
          {schema.ranking_items.length} items to rank • Confidence: {(schema.confidence * 100).toFixed(0)}%
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Detected Items */}
        <div>
          <Label className="text-sm font-medium mb-2 block">Ranking Items</Label>
          <div className="flex flex-wrap gap-1.5">
            {schema.ranking_items.slice(0, 8).map((item) => (
              <Badge key={item} variant="secondary" className="text-xs">
                {item}
              </Badge>
            ))}
            {schema.ranking_items.length > 8 && (
              <Badge variant="outline" className="text-xs">
                +{schema.ranking_items.length - 8} more
              </Badge>
            )}
          </div>
        </div>

        {/* Indicator Column */}
        {schema.indicator_col && (
          <div>
            <Label className="text-sm font-medium mb-2 block">
              Indicator: {schema.indicator_col}
            </Label>
            <div className="flex flex-wrap gap-1.5">
              {schema.indicator_values.map((value) => (
                <Badge key={value} variant="outline" className="text-xs">
                  {value}
                </Badge>
              ))}
            </div>
          </div>
        )}

        {/* Big Better Toggle */}
        <div className="flex items-center justify-between">
          <div className="space-y-0.5">
            <Label htmlFor="bigbetter" className="text-sm font-medium">
              Higher is Better
            </Label>
            <p className="text-xs text-muted-foreground">
              {bigbetter === 1 ? "Higher scores indicate better performance" : "Lower scores indicate better performance"}
            </p>
          </div>
          <Switch
            id="bigbetter"
            checked={bigbetter === 1}
            onCheckedChange={(checked) => setBigbetter(checked ? 1 : 0)}
          />
        </div>

        {/* Advanced Settings Toggle */}
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
        >
          {showAdvanced ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
          Advanced Settings
        </button>

        {/* Advanced Settings */}
        <motion.div
          initial={false}
          animate={{ height: showAdvanced ? "auto" : 0, opacity: showAdvanced ? 1 : 0 }}
          className="overflow-hidden"
        >
          <div className="space-y-4 pt-2">
            {/* Bootstrap Iterations */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-sm font-medium">Bootstrap Iterations</Label>
                <span className="text-sm text-muted-foreground">{bootstrapIterations}</span>
              </div>
              <Slider
                value={[bootstrapIterations]}
                onValueChange={([value]) => setBootstrapIterations(value)}
                min={100}
                max={5000}
                step={100}
                className="w-full"
              />
              <p className="text-xs text-muted-foreground">
                More iterations = more accurate confidence intervals, but slower
              </p>
            </div>

            {/* Random Seed */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-sm font-medium">Random Seed</Label>
                <span className="text-sm text-muted-foreground">{randomSeed}</span>
              </div>
              <Slider
                value={[randomSeed]}
                onValueChange={([value]) => setRandomSeed(value)}
                min={1}
                max={9999}
                step={1}
                className="w-full"
              />
              <p className="text-xs text-muted-foreground">
                Same seed = reproducible results
              </p>
            </div>
          </div>
        </motion.div>

        {/* Info Box */}
        <div className="flex items-start gap-2 p-3 rounded-lg bg-primary/5 border border-primary/20">
          <Info className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
          <p className="text-xs text-muted-foreground">
            OmniRank uses spectral ranking with optimal weights. Step 2 refinement will be automatically triggered if data heterogeneity is detected.
          </p>
        </div>

        {/* Start Button */}
        <Button
          onClick={handleStartAnalysis}
          disabled={isAnalyzing}
          className="w-full glow-cyan"
          size="lg"
        >
          {isAnalyzing ? (
            <>
              <span className="animate-pulse">Analyzing...</span>
            </>
          ) : (
            <>
              <Play className="h-4 w-4 mr-2" />
              Start Analysis
            </>
          )}
        </Button>
      </CardContent>
    </Card>
  );
}
