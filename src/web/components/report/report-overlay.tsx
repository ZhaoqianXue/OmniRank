"use client";

import { useRef, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Download, FileText, BarChart3, Table, TrendingUp, ArrowUp, ArrowDown, Tag, AlertCircle, CheckCircle, Info, HelpCircle, MessageCircleQuestion, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { useState } from "react";
import { RankingChart } from "@/components/visualizations";
import { ForestPlot } from "@/components/visualizations/forest-plot";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { RankingResults, InferredSchema, AnalysisConfig, RankingItem } from "@/lib/api";

/**
 * Fallback insights component when LLM report is unavailable
 */
function FallbackInsights({ results }: { results: RankingResults }) {
  const items = results.items;
  const metadata = results.metadata;

  // Identify statistical ties (overlapping CIs)
  const ties = useMemo(() => {
    const groups: string[][] = [];
    const used = new Set<string>();

    for (let i = 0; i < Math.min(items.length, 10); i++) {
      const itemA = items[i];
      if (used.has(itemA.name)) continue;

      const group = [itemA.name];
      for (let j = i + 1; j < Math.min(items.length, 10); j++) {
        const itemB = items[j];
        if (used.has(itemB.name)) continue;

        // Check if CIs overlap
        const aLower = itemA.ci_two_sided[0];
        const aUpper = itemA.ci_two_sided[1];
        const bLower = itemB.ci_two_sided[0];
        const bUpper = itemB.ci_two_sided[1];

        if (!(aUpper < bLower || bUpper < aLower)) {
          group.push(itemB.name);
        }
      }

      if (group.length > 1) {
        group.forEach(name => used.add(name));
        groups.push(group);
      }
    }
    return groups;
  }, [items]);

  const topItem = items[0];
  const secondItem = items.length > 1 ? items[1] : null;
  const scoreGap = secondItem ? topItem.theta_hat - secondItem.theta_hat : 0;
  const ciWidth = topItem.ci_two_sided[1] - topItem.ci_two_sided[0];
  const confidence = ciWidth <= 2 ? "high" : ciWidth <= 4 ? "moderate" : "low";

  return (
    <div className="space-y-4 text-sm">
      {/* Executive Summary */}
      <div>
        <h3 className="font-semibold text-foreground flex items-center gap-2 mb-2">
          <CheckCircle className="h-4 w-4 text-green-500" />
          Executive Summary
        </h3>
        <p className="text-muted-foreground">
          Based on <strong className="text-foreground">{metadata.n_comparisons.toLocaleString()}</strong> comparisons
          among <strong className="text-foreground">{metadata.n_items}</strong> items,
          <strong className="text-foreground"> {topItem.name}</strong> emerges as the top-ranked item
          with a preference score of {topItem.theta_hat.toFixed(3)}.
          The ranking confidence is <strong className="text-foreground">{confidence}</strong> (CI width: {ciWidth}).
        </p>
      </div>

      {/* Statistical Significance */}
      <div>
        <h3 className="font-semibold text-foreground flex items-center gap-2 mb-2">
          <Info className="h-4 w-4 text-blue-500" />
          Statistical Significance
        </h3>
        {ties.length > 0 ? (
          <div className="space-y-1">
            <p className="text-muted-foreground">The following items have overlapping confidence intervals and may be statistically indistinguishable:</p>
            <ul className="list-disc list-inside text-muted-foreground ml-2">
              {ties.map((group, i) => (
                <li key={i}><strong className="text-foreground">{group.join(", ")}</strong></li>
              ))}
            </ul>
          </div>
        ) : (
          <p className="text-muted-foreground">
            All top items have non-overlapping confidence intervals, indicating statistically significant rank differences.
          </p>
        )}
      </div>

      {/* Performance Gap */}
      {secondItem && (
        <div>
          <h3 className="font-semibold text-foreground flex items-center gap-2 mb-2">
            <BarChart3 className="h-4 w-4 text-purple-500" />
            Performance Gap
          </h3>
          <p className="text-muted-foreground">
            The gap between #1 (<strong className="text-foreground">{topItem.name}</strong>) and
            #2 (<strong className="text-foreground">{secondItem.name}</strong>) is <strong className="text-foreground">{scoreGap.toFixed(3)}</strong> in preference score.
            {scoreGap > 0.5 ? " This indicates a substantial performance difference." :
              scoreGap > 0.2 ? " This indicates a moderate performance difference." :
                " The top two items are relatively close in performance."}
          </p>
        </div>
      )}

      {/* Data Quality */}
      <div>
        <h3 className="font-semibold text-foreground flex items-center gap-2 mb-2">
          <AlertCircle className="h-4 w-4 text-amber-500" />
          Data Quality
        </h3>
        <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-2">
          <li>
            Sparsity ratio: <strong className="text-foreground">{metadata.sparsity_ratio.toFixed(2)}</strong>
            {metadata.sparsity_ratio >= 1.0 ? " (sufficient data)" : " (sparse data, interpret with caution)"}
          </li>
          <li>
            Standard analysis methodology applied
          </li>
        </ul>
      </div>
    </div>
  );
}

interface ReportOverlayProps {
  isVisible: boolean;
  results: RankingResults | null;
  schema: InferredSchema | null;
  config: AnalysisConfig | null;
  onClose: () => void;
  onSendMessage?: (message: string) => void;
  className?: string;
}

/**
 * Interactive section wrapper that shows suggested questions
 * Questions are always visible as a collapsible panel below the section
 */
interface InteractiveSectionProps {
  title: string;
  questions: string[];
  onAskQuestion?: (question: string) => void;
  children: React.ReactNode;
  className?: string;
}

function InteractiveSection({
  title,
  questions,
  onAskQuestion,
  children,
  className,
}: InteractiveSectionProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className={cn("relative", className)}>
      {children}

      {/* Always visible question panel */}
      {questions.length > 0 && onAskQuestion && (
        <div className="mt-3 no-print">
          {/* Toggle button */}
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="flex items-center gap-2 text-xs text-primary hover:text-primary/80 transition-colors group"
          >
            <div className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-full bg-primary/5 border border-primary/20 hover:bg-primary/10 hover:border-primary/30 transition-all">
              <MessageCircleQuestion className="h-3.5 w-3.5" />
              <span className="font-medium">Ask about this</span>
              <ChevronRight className={cn(
                "h-3 w-3 transition-transform duration-200",
                isExpanded && "rotate-90"
              )} />
            </div>
          </button>

          {/* Expandable questions panel */}
          {isExpanded && (
            <div className="mt-2 p-3 bg-primary/5 border border-primary/10 rounded-lg animate-in slide-in-from-top-2 duration-200">
              <p className="text-xs text-muted-foreground mb-2">
                Click a question to send it to the chat:
              </p>
              <div className="space-y-1.5">
                {questions.map((q, i) => (
                  <button
                    key={i}
                    onClick={() => {
                      onAskQuestion(q);
                      setIsExpanded(false);
                    }}
                    className="w-full flex items-start gap-2 p-2 text-left text-sm rounded-md bg-white dark:bg-zinc-900 border border-border/50 hover:border-primary/50 hover:bg-primary/5 transition-all group"
                  >
                    <ChevronRight className="h-4 w-4 text-primary mt-0.5 group-hover:translate-x-0.5 transition-transform" />
                    <span className="flex-1">{q}</span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function ReportOverlay({
  isVisible,
  results,
  schema,
  config,
  onClose,
  onSendMessage,
  className,
}: ReportOverlayProps) {
  const reportRef = useRef<HTMLDivElement>(null);

  // Use LLM-generated questions from backend, with fallback
  const sectionQuestions = useMemo(() => {
    if (!results?.items) return {};

    // If backend provided LLM-generated questions, use them
    if (results.section_questions) {
      return {
        rankings: results.section_questions.rankings || [],
        insights: results.section_questions.insights || [],
        scoreDistribution: results.section_questions.score_distribution || [],
        confidenceIntervals: results.section_questions.confidence_intervals || [],
      };
    }

    // Fallback to context-aware default questions
    const items = results.items;
    const topItem = items[0]?.name || "the top item";
    const secondItem = items[1]?.name || "the second item";
    const metadata = results.metadata;

    return {
      rankings: [
        `Is ${topItem} significantly better than ${secondItem}?`,
        "Which items have the most reliable rankings?",
        "Are there any statistical ties in the top rankings?",
      ],
      insights: [
        "How should I interpret the confidence intervals?",
        "What makes the top-ranked item stand out?",
        "How does the data quality affect these results?",
      ],
      scoreDistribution: [
        "What does the score distribution tell us?",
        "Are there distinct performance tiers among the items?",
        "Which items have the largest performance gaps?",
      ],
      confidenceIntervals: [
        "Which items have the most uncertainty in their ranking?",
        "How do the confidence intervals compare across items?",
        "Can I trust the ranking of items with wide CIs?",
      ],
    };
  }, [results]);

  // Export report as PDF using native browser print
  // This utilizes the @media print styles in globals.css
  const handleExportPDF = () => {
    window.print();
  };

  if (!results) return null;

  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          className={cn(
            "absolute inset-0 z-50 bg-background/95 backdrop-blur-sm rounded-lg overflow-hidden print-content-parent",
            className
          )}
        >
          {/* Header with action buttons - Hidden when printing */}
          <div className="absolute top-0 left-0 right-0 flex items-center justify-between px-4 py-3 bg-background/80 backdrop-blur-sm border-b border-border/40 z-10 no-print">
            <div className="flex items-center gap-2">
              <FileText className="h-5 w-5 text-primary" />
              <h2 className="text-lg font-semibold">Ranking Report</h2>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={handleExportPDF}
                className="h-8"
              >
                <Download className="h-4 w-4 mr-1.5" />
                Export PDF
              </Button>
              <Button
                variant="ghost"
                size="icon"
                onClick={onClose}
                className="h-8 w-8 hover:bg-destructive/10 hover:text-destructive"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          </div>

          {/* Report content - scrollable like PDF */}
          <div className="absolute inset-0 top-14 overflow-auto">
            <div
              ref={reportRef}
              className="min-h-full p-6 bg-white dark:bg-zinc-900"
            >
              <div className="max-w-4xl mx-auto space-y-8">
                {/* Title section */}
                <div className="text-center border-b border-border/40 pb-6">
                  <h1 className="text-2xl font-bold mb-2">OmniRank Analysis Report</h1>
                  <p className="text-muted-foreground text-sm">
                    Generated on {new Date().toLocaleDateString("en-US", {
                      year: "numeric",
                      month: "long",
                      day: "numeric",
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </p>
                </div>

                {/* ========================================== */}
                {/* Section 1: Complete Rankings Table */}
                {/* ========================================== */}
                <InteractiveSection
                  title="Rankings"
                  questions={sectionQuestions.rankings || []}
                  onAskQuestion={onSendMessage}
                >
                  <section className="space-y-4">
                    <div className="flex items-center gap-2">
                      <Table className="h-5 w-5 text-primary" />
                      <h2 className="text-lg font-semibold">Complete Rankings</h2>
                    </div>

                    {/* Summary stats - contextual information */}
                    <div className={cn(
                      "grid gap-3",
                      schema?.indicator_col ? "grid-cols-2 md:grid-cols-4" : "grid-cols-3"
                    )}>
                      <div className="p-3 bg-muted/40 rounded-lg">
                        <div className="flex items-center gap-1.5 mb-1">
                          {(config?.bigbetter ?? schema?.bigbetter ?? 1) === 1 ? (
                            <ArrowUp className="h-3.5 w-3.5 text-green-500" />
                          ) : (
                            <ArrowDown className="h-3.5 w-3.5 text-red-500" />
                          )}
                          <p className="text-xs text-muted-foreground uppercase tracking-wide">Direction</p>
                        </div>
                        <p className="text-lg font-semibold">
                          {(config?.bigbetter ?? schema?.bigbetter ?? 1) === 1 ? "Higher is Better" : "Lower is Better"}
                        </p>
                      </div>
                      <div className="p-3 bg-muted/40 rounded-lg">
                        <p className="text-xs text-muted-foreground uppercase tracking-wide mb-1">Sample Count</p>
                        <p className="text-xl font-bold">{results.metadata?.n_comparisons || "-"}</p>
                      </div>
                      <div className="p-3 bg-muted/40 rounded-lg">
                        <p className="text-xs text-muted-foreground uppercase tracking-wide mb-1">Ranking Items</p>
                        <p className="text-xl font-bold">{results.items.length}</p>
                      </div>
                      {schema?.indicator_col && (
                        <div className="p-3 bg-muted/40 rounded-lg">
                          <div className="flex items-center gap-1.5 mb-1">
                            <Tag className="h-3.5 w-3.5 text-primary" />
                            <p className="text-xs text-muted-foreground uppercase tracking-wide">Indicator</p>
                          </div>
                          <p className="text-lg font-semibold truncate" title={schema.indicator_col}>
                            {schema.indicator_col}
                          </p>
                        </div>
                      )}
                    </div>

                    {/* Full rankings table */}
                    <TooltipProvider delayDuration={200}>
                      <div className="border border-border rounded-lg overflow-hidden">
                        <table className="w-full text-sm">
                          <thead className="bg-muted/50">
                            <tr>
                              <th className="px-4 py-3 text-left font-medium">
                                <Tooltip>
                                  <TooltipTrigger asChild>
                                    <span className="inline-flex items-center gap-1 cursor-help">
                                      Rank
                                      <HelpCircle className="h-3 w-3 text-muted-foreground" />
                                    </span>
                                  </TooltipTrigger>
                                  <TooltipContent side="top">
                                    <p>The final ranking position of this item (1 = best)</p>
                                  </TooltipContent>
                                </Tooltip>
                              </th>
                              <th className="px-4 py-3 text-left font-medium">
                                <Tooltip>
                                  <TooltipTrigger asChild>
                                    <span className="inline-flex items-center gap-1 cursor-help">
                                      Item
                                      <HelpCircle className="h-3 w-3 text-muted-foreground" />
                                    </span>
                                  </TooltipTrigger>
                                  <TooltipContent side="top">
                                    <p>The name or identifier of the ranked item</p>
                                  </TooltipContent>
                                </Tooltip>
                              </th>
                              <th className="px-4 py-3 text-center font-medium">
                                <Tooltip>
                                  <TooltipTrigger asChild>
                                    <span className="inline-flex items-center gap-1 cursor-help">
                                      95% CI
                                      <HelpCircle className="h-3 w-3 text-muted-foreground" />
                                    </span>
                                  </TooltipTrigger>
                                  <TooltipContent side="top" className="max-w-sm">
                                    <p>95% Confidence Interval for the rank. We are 95% confident the true rank falls within this range. Narrower intervals indicate more reliable rankings.</p>
                                  </TooltipContent>
                                </Tooltip>
                              </th>
                              <th className="px-4 py-3 text-right font-medium">
                                <Tooltip>
                                  <TooltipTrigger asChild>
                                    <span className="inline-flex items-center gap-1 cursor-help justify-end">
                                      Score (θ̂)
                                      <HelpCircle className="h-3 w-3 text-muted-foreground" />
                                    </span>
                                  </TooltipTrigger>
                                  <TooltipContent side="top" className="max-w-sm">
                                    <p>Estimated preference score (theta). Higher scores indicate stronger preference. The difference between two scores indicates the odds ratio of one item beating another.</p>
                                  </TooltipContent>
                                </Tooltip>
                              </th>
                            </tr>
                          </thead>
                          <tbody>
                            {[...results.items]
                              .sort((a, b) => a.rank - b.rank)
                              .map((item, index) => (
                                <tr
                                  key={item.name}
                                  className={cn(
                                    "border-t border-border/40",
                                    index % 2 === 0 ? "bg-background" : "bg-muted/20",
                                    item.rank === 1 && "bg-primary/5"
                                  )}
                                >
                                  <td className="px-4 py-3 font-mono">
                                    <span className={cn(
                                      "inline-flex items-center justify-center w-8 h-6 rounded text-xs font-semibold",
                                      item.rank === 1 && "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400",
                                      item.rank === 2 && "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300",
                                      item.rank === 3 && "bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-400",
                                      item.rank > 3 && "bg-muted text-muted-foreground"
                                    )}>
                                      #{item.rank}
                                    </span>
                                  </td>
                                  <td className="px-4 py-3 font-medium">{item.name}</td>
                                  <td className="px-4 py-3 text-center font-mono text-muted-foreground">
                                    [{item.ci_two_sided[0]}, {item.ci_two_sided[1]}]
                                  </td>
                                  <td className="px-4 py-3 text-right font-mono">{item.theta_hat.toFixed(4)}</td>
                                </tr>
                              ))}
                          </tbody>
                        </table>
                      </div>
                    </TooltipProvider>
                  </section>
                </InteractiveSection>

                {/* ========================================== */}
                {/* Section 2: Analysis Report (LLM-generated) */}
                {/* ========================================== */}
                <InteractiveSection
                  title="Analysis"
                  questions={sectionQuestions.insights || []}
                  onAskQuestion={onSendMessage}
                >
                  <section className="space-y-4">
                    <div className="flex items-center gap-2">
                      <TrendingUp className="h-5 w-5 text-primary" />
                      <h2 className="text-lg font-semibold">Analysis & Insights</h2>
                    </div>

                    {results.report && results.report.trim() ? (
                      <div className="prose prose-sm dark:prose-invert max-w-none bg-muted/20 rounded-lg p-6">
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          components={{
                            // Custom table styling
                            table: ({ children }) => (
                              <div className="overflow-x-auto my-4">
                                <table className="min-w-full border border-border rounded-lg overflow-hidden">
                                  {children}
                                </table>
                              </div>
                            ),
                            thead: ({ children }) => (
                              <thead className="bg-muted/50">{children}</thead>
                            ),
                            th: ({ children }) => (
                              <th className="px-4 py-2 text-left font-medium text-sm border-b border-border">
                                {children}
                              </th>
                            ),
                            td: ({ children }) => (
                              <td className="px-4 py-2 text-sm border-b border-border/40">
                                {children}
                              </td>
                            ),
                            // Custom heading styling
                            h1: ({ children }) => (
                              <h1 className="text-xl font-bold mt-6 mb-3 text-foreground">{children}</h1>
                            ),
                            h2: ({ children }) => (
                              <h2 className="text-lg font-semibold mt-5 mb-2 text-foreground">{children}</h2>
                            ),
                            h3: ({ children }) => (
                              <h3 className="text-base font-semibold mt-4 mb-2 text-foreground">{children}</h3>
                            ),
                            // List styling
                            ul: ({ children }) => (
                              <ul className="list-disc list-inside space-y-1 my-2 text-muted-foreground">{children}</ul>
                            ),
                            ol: ({ children }) => (
                              <ol className="list-decimal list-inside space-y-1 my-2 text-muted-foreground">{children}</ol>
                            ),
                            li: ({ children }) => (
                              <li className="text-sm">{children}</li>
                            ),
                            // Paragraph styling
                            p: ({ children }) => (
                              <p className="text-sm text-muted-foreground leading-relaxed my-2">{children}</p>
                            ),
                            // Strong/bold styling
                            strong: ({ children }) => (
                              <strong className="font-semibold text-foreground">{children}</strong>
                            ),
                          }}
                        >
                          {results.report}
                        </ReactMarkdown>
                      </div>
                    ) : (
                      <div className="bg-muted/20 rounded-lg p-6">
                        <FallbackInsights results={results} />
                      </div>
                    )}
                  </section>
                </InteractiveSection>

                {/* ========================================== */}
                {/* Section 3: Visualization */}
                {/* ========================================== */}
                <section className="space-y-6">
                  <div className="flex items-center gap-2">
                    <BarChart3 className="h-5 w-5 text-primary" />
                    <h2 className="text-lg font-semibold">Rankings Visualization</h2>
                  </div>

                  {/* Chart 1: Score Distribution Bar Chart */}
                  <InteractiveSection
                    title="Score Distribution"
                    questions={sectionQuestions.scoreDistribution || []}
                    onAskQuestion={onSendMessage}
                  >
                    <div className="space-y-2">
                      <h3 className="text-sm font-medium text-muted-foreground">Score Distribution</h3>
                      <div className="border border-border rounded-lg p-4 bg-card">
                        <RankingChart items={results.items} className="w-full" />
                      </div>
                    </div>
                  </InteractiveSection>

                  {/* Chart 2: Forest Plot with Confidence Intervals */}
                  <InteractiveSection
                    title="Confidence Intervals"
                    questions={sectionQuestions.confidenceIntervals || []}
                    onAskQuestion={onSendMessage}
                  >
                    <div className="space-y-2">
                      <h3 className="text-sm font-medium text-muted-foreground">Ranking Confidence Intervals</h3>
                      <div className="border border-border rounded-lg p-4 bg-card">
                        <ForestPlot items={results.items} className="w-full" />
                      </div>
                    </div>
                  </InteractiveSection>
                </section>

                {/* Footer */}
                <div className="text-center text-muted-foreground text-xs py-6 border-t border-border/40">
                  <p>Generated by OmniRank - Spectral Ranking Analysis Platform</p>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
