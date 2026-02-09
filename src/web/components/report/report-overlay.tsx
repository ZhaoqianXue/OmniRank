"use client";

import { useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Download, FileText, MessageSquareQuote, X } from "lucide-react";
import ReactMarkdown from "react-markdown";
import rehypeRaw from "rehype-raw";
import rehypeSanitize, { defaultSchema } from "rehype-sanitize";
import remarkGfm from "remark-gfm";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  artifactUrl,
  type AnalysisConfig,
  type ArtifactDescriptor,
  type PlotSpec,
  type QuotePayload,
  type RankingResults,
  type ReportOutput,
  type SemanticSchema,
} from "@/lib/api";

interface ReportOverlayProps {
  isVisible: boolean;
  sessionId: string | null;
  results: RankingResults | null;
  reportOutput: ReportOutput | null;
  plots: PlotSpec[];
  artifacts: ArtifactDescriptor[];
  schema: SemanticSchema | null;
  config: AnalysisConfig | null;
  onClose: () => void;
  onSendMessage?: (message: string, quotes?: QuotePayload[]) => void | Promise<void>;
  className?: string;
}

interface QuoteDraft {
  text: string;
  blockId?: string;
  kind?: string;
  x: number;
  y: number;
}

const reportSanitizeSchema = {
  ...defaultSchema,
  tagNames: [...(defaultSchema.tagNames || []), "section"],
  attributes: {
    ...(defaultSchema.attributes || {}),
    section: ["data-omni-block-id", "data-omni-kind"],
    img: [...(((defaultSchema.attributes || {}).img as Array<string | [string, ...string[]]>) || []), "src", "alt", "title"],
  },
};

export function ReportOverlay({
  isVisible,
  sessionId,
  results,
  reportOutput,
  plots,
  artifacts,
  schema,
  config,
  onClose,
  onSendMessage,
  className,
}: ReportOverlayProps) {
  const contentRef = useRef<HTMLDivElement>(null);
  const [quoteDraft, setQuoteDraft] = useState<QuoteDraft | null>(null);

  const markdown = reportOutput?.markdown || results?.report || "No report available.";

  const figureUrls = useMemo(() => {
    if (!sessionId) {
      return new Map<string, string>();
    }

    const map = new Map<string, string>();
    for (const plot of plots) {
      const artifact = artifacts.find((item) => item.kind === "figure" && item.title === plot.type);
      if (artifact) {
        map.set(plot.block_id, artifactUrl(sessionId, artifact.artifact_id));
      }
    }
    return map;
  }, [artifacts, plots, sessionId]);

  const artifactPathToUrl = useMemo(() => {
    if (!sessionId) {
      return new Map<string, string>();
    }
    const map = new Map<string, string>();
    for (const artifact of artifacts) {
      const url = artifactUrl(sessionId, artifact.artifact_id);
      map.set(artifact.title, url);
      map.set(artifact.artifact_id, url);
      if (artifact.kind === "figure") {
        map.set(artifact.title, url);
      }
    }
    for (const reportArtifact of reportOutput?.artifacts || []) {
      const match = artifacts.find(
        (item) => item.kind === reportArtifact.kind && item.title === reportArtifact.title
      );
      if (match) {
        const url = artifactUrl(sessionId, match.artifact_id);
        map.set(reportArtifact.path, url);
        const filename = reportArtifact.path.split("/").pop() || reportArtifact.path;
        map.set(filename, url);
      }
    }
    return map;
  }, [artifacts, reportOutput?.artifacts, sessionId]);

  const handleExportPdf = () => {
    window.print();
  };

  const handleMouseUp = () => {
    const selection = window.getSelection();
    if (!selection || selection.rangeCount === 0 || selection.isCollapsed) {
      setQuoteDraft(null);
      return;
    }

    const text = selection.toString().trim();
    if (!text) {
      setQuoteDraft(null);
      return;
    }

    const range = selection.getRangeAt(0);
    const rect = range.getBoundingClientRect();

    const anchorNode = selection.anchorNode;
    if (!anchorNode || !contentRef.current || !contentRef.current.contains(anchorNode)) {
      setQuoteDraft(null);
      return;
    }

    const element =
      anchorNode instanceof Element ? anchorNode : (anchorNode.parentElement as Element | null);
    const section = element?.closest("section[data-omni-block-id]");

    setQuoteDraft({
      text,
      blockId: section?.getAttribute("data-omni-block-id") || undefined,
      kind: section?.getAttribute("data-omni-kind") || undefined,
      x: rect.left + rect.width / 2,
      y: rect.top - 8,
    });
  };

  const handleQuote = async () => {
    if (!quoteDraft || !onSendMessage) {
      return;
    }

    const payload: QuotePayload = {
      quoted_text: quoteDraft.text,
      block_id: quoteDraft.blockId,
      kind: quoteDraft.kind,
      source: "report",
    };

    await onSendMessage("Please answer based on this quoted report excerpt.", [payload]);

    const selection = window.getSelection();
    if (selection) {
      selection.removeAllRanges();
    }
    setQuoteDraft(null);
  };

  if (!results && !reportOutput) {
    return null;
  }

  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          className={cn(
            "absolute inset-0 z-50 bg-card/95 backdrop-blur-sm rounded-lg overflow-hidden",
            className
          )}
        >
          <div className="absolute top-0 left-0 right-0 flex items-center justify-between px-4 py-3 bg-card/90 border-b border-border/40 z-10">
            <div className="flex items-center gap-2">
              <FileText className="h-5 w-5 text-primary" />
              <h2 className="text-lg font-semibold">Single-Page Report</h2>
            </div>
            <div className="flex items-center gap-2">
              <Button variant="outline" size="sm" onClick={handleExportPdf}>
                <Download className="h-4 w-4 mr-1.5" />
                Export PDF
              </Button>
              <Button variant="ghost" size="icon" onClick={onClose}>
                <X className="h-4 w-4" />
              </Button>
            </div>
          </div>

          <div className="absolute inset-0 top-14 overflow-auto" onMouseUp={handleMouseUp}>
            <div ref={contentRef} className="max-w-4xl mx-auto p-6 pb-24">
              <div className="mb-6 text-sm text-muted-foreground">
                <p>
                  Schema: <strong>{schema ? `${schema.ranking_items.length} items` : "N/A"}</strong>
                </p>
                <p>
                  Config: <strong>B={config?.bootstrap_iterations ?? 2000}</strong>, <strong>seed={config?.random_seed ?? 42}</strong>
                </p>
              </div>

              <div className="prose prose-sm dark:prose-invert max-w-none">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[rehypeRaw, [rehypeSanitize, reportSanitizeSchema]]}
                  components={{
                    img: ({ src, alt }) => {
                      const source = typeof src === "string" ? src : "";
                      const filename = source.split("/").pop() || source;
                      const normalizedSrc =
                        artifactPathToUrl.get(source) ||
                        artifactPathToUrl.get(filename) ||
                        Array.from(figureUrls.entries()).find(([, url]) => source.includes(url))?.[1] ||
                        source;
                      // eslint-disable-next-line @next/next/no-img-element
                      return <img src={normalizedSrc} alt={alt || "report figure"} className="w-full rounded border" />;
                    },
                  }}
                >
                  {markdown}
                </ReactMarkdown>
              </div>

              {plots.length > 0 && sessionId && (
                <section className="mt-8 space-y-4">
                  <h3 className="text-base font-semibold">Deterministic SVG Figures</h3>
                  {plots.map((plot) => {
                    const artifact = artifacts.find((item) => item.kind === "figure" && item.title === plot.type);
                    const src = artifact ? artifactUrl(sessionId, artifact.artifact_id) : "";
                    return (
                      <section
                        key={plot.block_id}
                        data-omni-block-id={plot.block_id}
                        data-omni-kind="figure"
                        className="border rounded-lg p-3"
                      >
                        <p className="text-sm font-medium">{plot.caption_plain}</p>
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        {src ? <img src={src} alt={plot.type} className="mt-2 w-full rounded" /> : <p className="mt-2 text-sm">Figure unavailable.</p>}
                        <p className="mt-2 text-xs text-muted-foreground">{plot.caption_academic}</p>
                      </section>
                    );
                  })}
                </section>
              )}
            </div>
          </div>

          {quoteDraft && (
            <div
              className="fixed z-[60]"
              style={{ left: quoteDraft.x, top: quoteDraft.y, transform: "translate(-50%, -100%)" }}
            >
              <Button size="sm" className="shadow-lg" onClick={handleQuote}>
                <MessageSquareQuote className="h-4 w-4 mr-1.5" />
                Quote
              </Button>
            </div>
          )}
        </motion.div>
      )}
    </AnimatePresence>
  );
}
