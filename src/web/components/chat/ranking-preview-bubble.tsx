"use client";

import { useState } from "react";
import { Play, Loader2, Settings2, Check, X, ChevronDown, ChevronRight, Eye, EyeOff } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import type {
  AnalysisConfig,
  FormatValidationResult,
  QualityValidationResult,
  SemanticSchema,
  ValidationWarning,
} from "@/lib/api";

interface RankingPreviewBubbleProps {
  schema: SemanticSchema;
  detectedFormat?: "pointwise" | "pairwise" | "multiway";
  formatResult?: FormatValidationResult | null;
  qualityResult?: QualityValidationResult | null;
  warnings?: ValidationWarning[];
  onStartAnalysis: (config: AnalysisConfig) => void;
  isAnalyzing?: boolean;
  isCompleted?: boolean;
  isReportVisible?: boolean;
  onToggleReport?: () => void;
  className?: string;
}

// Display row component (read-only)
function DisplayRow({
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

// Section component
function Section({
  title,
  children,
  showLine = true,
}: {
  title: string;
  children: React.ReactNode;
  showLine?: boolean;
}) {
  return (
    <div className="flex gap-3">
      <div className="flex flex-col items-center">
        <div className="w-2.5 h-2.5 rounded-full bg-muted-foreground/40" />
        {showLine && <div className="w-0.5 flex-1 bg-muted-foreground/20 mt-1.5" />}
      </div>
      <div className="flex-1 pb-4">
        <h4 className="text-sm font-semibold mb-2">{title}</h4>
        <div className="space-y-1.5">{children}</div>
      </div>
    </div>
  );
}

export function RankingPreviewBubble({
  schema,
  detectedFormat,
  formatResult,
  qualityResult,
  warnings = [],
  onStartAnalysis,
  isAnalyzing = false,
  isCompleted = false,
  isReportVisible = true,
  onToggleReport,
  className,
}: RankingPreviewBubbleProps) {
  const hasIndicator = Boolean(schema.indicator_col && schema.indicator_values.length > 0);

  // Configuration state
  const [bigbetter, setBigbetter] = useState<0 | 1>(schema.bigbetter as 0 | 1);
  const [selectedItems, setSelectedItems] = useState<string[]>([...schema.ranking_items]);
  const [indicatorCol, setIndicatorCol] = useState<string | null>(schema.indicator_col);
  const [selectedIndicatorValues, setSelectedIndicatorValues] = useState<string[]>([
    ...schema.indicator_values,
  ]);
  const [bootstrapIterations, setBootstrapIterations] = useState(2000);
  const [randomSeed, setRandomSeed] = useState(42);
  
  // Dialog state
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  
  // Temporary state for dialog editing
  const [tempBigbetter, setTempBigbetter] = useState<0 | 1>(1);
  const [tempSelectedItems, setTempSelectedItems] = useState<string[]>([]);
  const [tempIndicatorEnabled, setTempIndicatorEnabled] = useState<boolean>(Boolean(schema.indicator_col));
  const [tempSelectedIndicatorValues, setTempSelectedIndicatorValues] = useState<string[]>([]);
  const [tempBootstrapIterations, setTempBootstrapIterations] = useState(2000);
  const [tempRandomSeed, setTempRandomSeed] = useState(42);

  // Open dialog and sync temp state
  const handleOpenDialog = () => {
    setTempBigbetter(bigbetter);
    setTempSelectedItems([...selectedItems]);
    setTempIndicatorEnabled(indicatorCol !== null);
    setTempSelectedIndicatorValues([...selectedIndicatorValues]);
    setTempBootstrapIterations(bootstrapIterations);
    setTempRandomSeed(randomSeed);
    setShowAdvanced(false);
    setIsDialogOpen(true);
  };

  // Save changes from dialog
  const handleSaveConfig = () => {
    const nextIndicatorCol = tempIndicatorEnabled ? schema.indicator_col : null;
    setBigbetter(tempBigbetter);
    setSelectedItems(tempSelectedItems);
    setIndicatorCol(nextIndicatorCol);
    setSelectedIndicatorValues(tempIndicatorEnabled ? tempSelectedIndicatorValues : []);
    setBootstrapIterations(tempBootstrapIterations);
    setRandomSeed(tempRandomSeed);
    setIsDialogOpen(false);
  };

  // Toggle item selection
  const toggleItem = (item: string) => {
    setTempSelectedItems(prev => 
      prev.includes(item) 
        ? prev.filter(i => i !== item)
        : [...prev, item]
    );
  };

  // Toggle indicator value selection
  const toggleIndicatorValue = (value: string) => {
    setTempSelectedIndicatorValues(prev =>
      prev.includes(value)
        ? prev.filter(v => v !== value)
        : [...prev, value]
    );
  };

  // Select/deselect all items
  const selectAllItems = () => setTempSelectedItems([...schema.ranking_items]);
  const deselectAllItems = () => setTempSelectedItems([]);

  // Select/deselect all indicator values
  const selectAllIndicators = () => setTempSelectedIndicatorValues([...schema.indicator_values]);
  const deselectAllIndicators = () => setTempSelectedIndicatorValues([]);

  const handleStartAnalysis = () => {
    const useIndicator = indicatorCol !== null;
    const config: AnalysisConfig = {
      bigbetter,
      indicator_col: useIndicator ? indicatorCol : null,
      selected_items: selectedItems.length === schema.ranking_items.length ? undefined : selectedItems,
      selected_indicator_values: useIndicator
        ? (selectedIndicatorValues.length === schema.indicator_values.length ? undefined : selectedIndicatorValues)
        : undefined,
      bootstrap_iterations: bootstrapIterations,
      random_seed: randomSeed,
    };
    onStartAnalysis(config);
  };

  // Calculate estimated runtime
  const estimatedRuntime = `~${Math.max(1, Math.ceil(selectedItems.length * 0.3))} seconds`;

  // Check if config has been modified from defaults
  const isModified = 
    bigbetter !== schema.bigbetter ||
    indicatorCol !== schema.indicator_col ||
    selectedItems.length !== schema.ranking_items.length ||
    selectedIndicatorValues.length !== schema.indicator_values.length ||
    bootstrapIterations !== 2000 ||
    randomSeed !== 42;

  return (
    <>
      <div className={cn(
        "bg-white dark:bg-zinc-800 border border-border/50 rounded-2xl rounded-bl-sm shadow-sm max-w-sm w-full",
        className
      )}>
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-border/40">
          <div className="flex items-center gap-2">
            <h3 className="text-base font-semibold">Ranking Preview</h3>
            {isModified && (
              <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
                Modified
              </Badge>
            )}
          </div>
          <button
            onClick={handleOpenDialog}
            className="text-muted-foreground hover:text-foreground transition-colors"
            title="Configure analysis"
          >
            <Settings2 className="h-4 w-4" />
          </button>
        </div>

        {/* Content (Read-only display) */}
        <div className="px-4 py-3">
          {/* Data Schema */}
          <Section title="Data Schema">
            <DisplayRow label="Format">
              <span className={cn(
                detectedFormat === "pairwise" ? "text-blue-600 dark:text-blue-400" : "text-green-600 dark:text-green-400"
              )}>
                {detectedFormat || "inferred"}
              </span>
            </DisplayRow>
            <DisplayRow label="Format Check">
              <span className={cn(formatResult ? (formatResult.is_ready ? "text-green-600" : "text-yellow-600") : "text-muted-foreground")}>
                {formatResult ? (formatResult.is_ready ? "PASS" : "Needs Attention") : "N/A"}
              </span>
            </DisplayRow>
            <DisplayRow label="Quality Check">
              <span className={cn(qualityResult ? (qualityResult.is_valid ? "text-green-600" : "text-red-600") : "text-muted-foreground")}>
                {qualityResult ? (qualityResult.is_valid ? "PASS" : "Blocking Errors") : "N/A"}
              </span>
            </DisplayRow>
            <DisplayRow label="Data Quality">
              <span className={cn(
                warnings.length === 0 ? "text-green-600" : 
                warnings.some(w => w.severity === "error") ? "text-red-600" : "text-yellow-600"
              )}>
                {warnings.length === 0 ? "Good" : 
                 warnings.some(w => w.severity === "error") ? "Issues Found" : "Warnings"}
              </span>
            </DisplayRow>
            {warnings.length > 0 && (
              <div className="text-xs text-muted-foreground space-y-0.5 mt-1">
                {warnings.map((w, i) => (
                  <div key={i} className={cn(
                    "flex items-start gap-1",
                    w.severity === "error" ? "text-red-600" : "text-yellow-600"
                  )}>
                    <span>{w.severity === "error" ? "✕" : "⚠"}</span>
                    <span>{w.message}</span>
                  </div>
                ))}
              </div>
            )}
            <DisplayRow label="Direction">
              {bigbetter === 1 ? "Higher is better" : "Lower is better"}
            </DisplayRow>
          </Section>

          {/* Ranking Items */}
          <Section title="Ranking Items">
            <DisplayRow label="Selected">
              {selectedItems.length} / {schema.ranking_items.length}
            </DisplayRow>
            <div className="text-xs font-mono bg-muted/60 px-2 py-1 rounded leading-relaxed break-words">
              {selectedItems.join(", ")}
            </div>
          </Section>

          {/* Ranking Indicator (only if indicator exists) */}
          {hasIndicator && (
            <Section title="Ranking Indicator">
              <DisplayRow label="Column">{indicatorCol || "Disabled"}</DisplayRow>
              <DisplayRow label="Selected">
                {indicatorCol ? selectedIndicatorValues.length : 0} / {schema.indicator_values.length}
              </DisplayRow>
              {indicatorCol && selectedIndicatorValues.length > 0 && (
                <div className="text-xs font-mono bg-muted/60 px-2 py-0.5 rounded">
                  {selectedIndicatorValues.join(", ")}
                </div>
              )}
            </Section>
          )}

          {/* Parameters */}
          <Section title="Parameters" showLine={false}>
            <DisplayRow label="Bootstrap">{bootstrapIterations}</DisplayRow>
            <DisplayRow label="Seed">{randomSeed}</DisplayRow>
            <DisplayRow label="Est. runtime">{estimatedRuntime}</DisplayRow>
          </Section>
        </div>

        {/* Action Button - changes based on state */}
        <div className="px-4 pb-4">
          {isCompleted ? (
            // After ranking is completed: Show/Hide Report toggle
            <Button
              onClick={onToggleReport}
              variant="outline"
              className="w-full"
              size="lg"
            >
              {isReportVisible ? (
                <>
                  <EyeOff className="h-4 w-4 mr-2" />
                  Hide Report
                </>
              ) : (
                <>
                  <Eye className="h-4 w-4 mr-2" />
                  Show Report
                </>
              )}
            </Button>
          ) : (
            // Before/during analysis: Start Ranking button
            <>
              <Button
                onClick={handleStartAnalysis}
                disabled={isAnalyzing || selectedItems.length < 2}
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
              {selectedItems.length < 2 && (
                <p className="text-xs text-destructive mt-1 text-center">
                  Select at least 2 items to rank
                </p>
              )}
            </>
          )}
        </div>
      </div>

      {/* Configuration Dialog */}
      <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Configure Analysis</DialogTitle>
          </DialogHeader>
          
          <ScrollArea className="max-h-[60vh] pr-4">
            <div className="space-y-6 py-4">
              {/* Ranking Direction */}
              <div className="space-y-2">
                <h4 className="text-sm font-medium">Ranking Direction</h4>
                <div className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
                  <span className="text-sm">
                    {tempBigbetter === 1 ? "Higher is better" : "Lower is better"}
                  </span>
                  <Switch
                    checked={tempBigbetter === 1}
                    onCheckedChange={(checked) => setTempBigbetter(checked ? 1 : 0)}
                  />
                </div>
                <p className="text-xs text-muted-foreground">
                  {tempBigbetter === 1 
                    ? "Items with higher scores/win rates rank higher"
                    : "Items with lower scores (e.g., errors, latency) rank higher"}
                </p>
              </div>

              {/* Ranking Items Selection */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <h4 className="text-sm font-medium">Ranking Items</h4>
                  <div className="flex gap-1">
                    <Button variant="ghost" size="sm" className="h-6 text-xs" onClick={selectAllItems}>
                      All
                    </Button>
                    <Button variant="ghost" size="sm" className="h-6 text-xs" onClick={deselectAllItems}>
                      None
                    </Button>
                  </div>
                </div>
                <div className="flex flex-wrap gap-1.5 p-3 rounded-lg bg-muted/50 max-h-32 overflow-y-auto">
                  {schema.ranking_items.map((item) => (
                    <Badge
                      key={item}
                      variant={tempSelectedItems.includes(item) ? "default" : "outline"}
                      className="cursor-pointer text-xs"
                      onClick={() => toggleItem(item)}
                    >
                      {tempSelectedItems.includes(item) && <Check className="h-3 w-3 mr-1" />}
                      {item}
                    </Badge>
                  ))}
                </div>
                <p className="text-xs text-muted-foreground">
                  {tempSelectedItems.length} of {schema.ranking_items.length} items selected
                </p>
              </div>

              {/* Indicator Values Selection (if applicable) */}
              {hasIndicator && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <h4 className="text-sm font-medium">Indicator: {schema.indicator_col}</h4>
                    <Switch checked={tempIndicatorEnabled} onCheckedChange={setTempIndicatorEnabled} />
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {tempIndicatorEnabled ? "Enabled for segmented ranking." : "Disabled (global ranking only)."}
                  </p>
                  <div className="flex items-center justify-between">
                    <div className="flex gap-1">
                      <Button variant="ghost" size="sm" className="h-6 text-xs" onClick={selectAllIndicators}>
                        All
                      </Button>
                      <Button variant="ghost" size="sm" className="h-6 text-xs" onClick={deselectAllIndicators}>
                        None
                      </Button>
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-1.5 p-3 rounded-lg bg-muted/50">
                    {schema.indicator_values.map((value) => (
                      <Badge
                        key={value}
                        variant={tempSelectedIndicatorValues.includes(value) ? "default" : "outline"}
                        className={cn("cursor-pointer text-xs", !tempIndicatorEnabled && "pointer-events-none opacity-50")}
                        onClick={() => toggleIndicatorValue(value)}
                      >
                        {tempSelectedIndicatorValues.includes(value) && <Check className="h-3 w-3 mr-1" />}
                        {value}
                      </Badge>
                    ))}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Filter data by selected indicator values
                  </p>
                </div>
              )}

              {/* Advanced Settings (Collapsible) */}
              <div className="space-y-2">
                <button
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className="flex items-center gap-2 text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
                >
                  {showAdvanced ? (
                    <ChevronDown className="h-4 w-4" />
                  ) : (
                    <ChevronRight className="h-4 w-4" />
                  )}
                  Advanced Settings
                </button>
                
                {showAdvanced && (
                  <div className="space-y-4 pl-6 pt-2">
                    {/* Bootstrap Iterations */}
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm">Bootstrap Iterations</span>
                        <span className="text-sm font-mono bg-muted px-2 py-0.5 rounded">
                          {tempBootstrapIterations}
                        </span>
                      </div>
                      <Slider
                        value={[tempBootstrapIterations]}
                        onValueChange={([value]) => setTempBootstrapIterations(value)}
                        min={100}
                        max={5000}
                        step={100}
                      />
                      <p className="text-xs text-muted-foreground">
                        More iterations = more accurate CIs, longer runtime
                      </p>
                    </div>

                    {/* Random Seed */}
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm">Random Seed</span>
                        <Input
                          type="number"
                          value={tempRandomSeed}
                          onChange={(e) => {
                            const val = parseInt(e.target.value) || 1;
                            setTempRandomSeed(Math.max(1, Math.min(999999, val)));
                          }}
                          className="w-24 h-7 text-sm font-mono text-right"
                          min={1}
                          max={999999}
                        />
                      </div>
                      <p className="text-xs text-muted-foreground">
                        Set for reproducible results
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </ScrollArea>

          <DialogFooter className="gap-2">
            <Button variant="outline" onClick={() => setIsDialogOpen(false)}>
              <X className="h-4 w-4 mr-1" />
              Cancel
            </Button>
            <Button onClick={handleSaveConfig} disabled={tempSelectedItems.length < 2}>
              <Check className="h-4 w-4 mr-1" />
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
