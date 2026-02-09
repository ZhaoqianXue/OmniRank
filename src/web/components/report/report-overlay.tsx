"use client";

import { useMemo, useRef, useState, type ComponentPropsWithoutRef, type ReactNode } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  BookOpen,
  ChevronDown,
  ChevronUp,
  Download,
  FileText,
  Info,
  MessageSquareQuote,
  X,
} from "lucide-react";
import ReactMarkdown, { type Components } from "react-markdown";
import rehypeRaw from "rehype-raw";
import rehypeSanitize, { defaultSchema } from "rehype-sanitize";
import remarkGfm from "remark-gfm";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import {
  artifactUrl,
  type AnalysisConfig,
  type ArtifactDescriptor,
  type HintSpec,
  type PlotSpec,
  type QuotePayload,
  type RankingResults,
  type ReportOutput,
  type SemanticSchema,
} from "@/lib/api";

/* -------------------------------------------------------------------------- */
/* Types                                                                       */
/* -------------------------------------------------------------------------- */

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

/* -------------------------------------------------------------------------- */
/* Sanitisation schema                                                         */
/* -------------------------------------------------------------------------- */

const reportSanitizeSchema = {
  ...defaultSchema,
  tagNames: [...(defaultSchema.tagNames || []), "section"],
  attributes: {
    ...(defaultSchema.attributes || {}),
    section: ["data-omni-block-id", "data-omni-kind"],
    img: [
      ...(((defaultSchema.attributes || {}).img as Array<string | [string, ...string[]]>) || []),
      "src",
      "alt",
      "title",
    ],
  },
};

/* -------------------------------------------------------------------------- */
/* Section kind styles                                                         */
/* -------------------------------------------------------------------------- */

const SECTION_STYLES: Record<string, string> = {
  summary:
    "relative bg-primary/[0.04] border border-primary/20 rounded-xl p-6 my-6 shadow-[0_0_24px_-6px_rgba(152,132,229,0.12)]",
  result: "my-6",
  table: "my-6",
  figure: "rounded-xl border border-border/40 bg-card/40 p-4 my-6",
  comparison:
    "bg-muted/20 border border-border/40 rounded-lg p-5 my-6",
  method: "my-6",
  limitation:
    "border-l-4 border-amber-500/50 bg-amber-500/[0.03] rounded-r-lg pl-5 pr-4 py-4 my-6",
  repro: "bg-muted/20 border border-border/30 rounded-lg p-5 my-6 font-mono text-xs leading-relaxed",
};

/* -------------------------------------------------------------------------- */
/* Custom Markdown components                                                  */
/* -------------------------------------------------------------------------- */

function buildMarkdownComponents(
  artifactPathToUrl: Map<string, string>,
  figureUrls: Map<string, string>,
): Components {
  return {
    /* ── Sections (kind-aware styling) ─────────────────────────────────── */
    section: (props: ComponentPropsWithoutRef<"section"> & { children?: ReactNode }) => {
      const { children, ...rest } = props;
      const kind = (rest as Record<string, unknown>)["data-omni-kind"] as string | undefined;
      const blockId = (rest as Record<string, unknown>)["data-omni-block-id"] as string | undefined;

      if (!kind) return <section {...rest}>{children}</section>;

      return (
        <section
          data-omni-block-id={blockId}
          data-omni-kind={kind}
          className={cn(SECTION_STYLES[kind] || "my-4")}
        >
          {children}
        </section>
      );
    },

    /* ── Headings ──────────────────────────────────────────────────────── */
    h1: ({ children }) => (
      <h1 className="text-2xl font-bold pb-4 mb-2 border-b border-primary/30">
        <span className="gradient-text">{children}</span>
      </h1>
    ),
    h2: ({ children }) => (
      <h2 className="text-lg font-semibold text-foreground mt-0 mb-3 flex items-center gap-2">
        <span className="inline-block h-5 w-1 rounded-full bg-primary" />
        {children}
      </h2>
    ),
    h3: ({ children }) => (
      <h3 className="text-base font-semibold text-foreground/90 mt-0 mb-2">
        {children}
      </h3>
    ),

    /* ── Tables ────────────────────────────────────────────────────────── */
    table: ({ children }) => (
      <div className="overflow-x-auto rounded-lg border border-border/40 my-4">
        <table className="w-full text-sm border-collapse">{children}</table>
      </div>
    ),
    thead: ({ children }) => (
      <thead className="bg-primary/[0.06] border-b border-border/40">{children}</thead>
    ),
    th: ({ children }) => (
      <th className="px-4 py-2.5 text-left text-[11px] font-semibold text-primary uppercase tracking-wider whitespace-nowrap">
        {children}
      </th>
    ),
    tbody: ({ children }) => (
      <tbody className="divide-y divide-border/20">{children}</tbody>
    ),
    tr: ({ children }) => (
      <tr className="hover:bg-muted/20 transition-colors">{children}</tr>
    ),
    td: ({ children }) => (
      <td className="px-4 py-2 text-sm whitespace-nowrap">{children}</td>
    ),

    /* ── Horizontal rule (section divider) ─────────────────────────────── */
    hr: () => (
      <div className="my-8 flex items-center gap-3">
        <div className="flex-1 h-px bg-border/40" />
        <div className="h-1.5 w-1.5 rounded-full bg-primary/40" />
        <div className="flex-1 h-px bg-border/40" />
      </div>
    ),

    /* ── Block elements ────────────────────────────────────────────────── */
    blockquote: ({ children }) => (
      <blockquote className="border-l-4 border-primary/30 pl-4 py-1 my-4 text-muted-foreground">
        {children}
      </blockquote>
    ),
    p: ({ children }) => (
      <p className="text-sm leading-relaxed mb-3 text-foreground/90">{children}</p>
    ),
    ul: ({ children }) => (
      <ul className="space-y-1.5 my-3 list-none pl-0">{children}</ul>
    ),
    ol: ({ children }) => (
      <ol className="space-y-1.5 my-3 list-decimal pl-5">{children}</ol>
    ),
    li: ({ children }) => (
      <li className="flex items-start gap-2 text-sm leading-relaxed">
        <span className="mt-[7px] h-1.5 w-1.5 rounded-full bg-primary/50 shrink-0" />
        <span className="flex-1">{children}</span>
      </li>
    ),

    /* ── Inline elements ───────────────────────────────────────────────── */
    strong: ({ children }) => (
      <strong className="font-semibold text-foreground">{children}</strong>
    ),
    em: ({ children }) => (
      <em className="italic text-muted-foreground">{children}</em>
    ),
    code: ({ children }) => (
      <code className="bg-muted/50 px-1.5 py-0.5 rounded text-xs font-mono text-primary/80">
        {children}
      </code>
    ),

    /* ── Images (artifact URL resolution) ──────────────────────────────── */
    img: ({ src, alt }) => {
      const source = typeof src === "string" ? src : "";
      const filename = source.split("/").pop() || source;
      const normalizedSrc =
        artifactPathToUrl.get(source) ||
        artifactPathToUrl.get(filename) ||
        Array.from(figureUrls.entries()).find(([, url]) => source.includes(url))?.[1] ||
        source;
      /* eslint-disable-next-line @next/next/no-img-element */
      return (
        <img
          src={normalizedSrc}
          alt={alt || "report figure"}
          className="w-full rounded-lg border border-border/30 my-2"
        />
      );
    },
  };
}

/* -------------------------------------------------------------------------- */
/* Glossary panel                                                              */
/* -------------------------------------------------------------------------- */

function GlossaryPanel({ hints }: { hints: HintSpec[] }) {
  const [open, setOpen] = useState(false);

  if (!hints || hints.length === 0) return null;

  return (
    <div className="mt-8 border border-border/30 rounded-lg overflow-hidden">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-4 py-3 text-sm font-medium text-foreground/80 hover:bg-muted/20 transition-colors"
      >
        <span className="flex items-center gap-2">
          <BookOpen className="h-4 w-4 text-primary/70" />
          Terms and Definitions
        </span>
        {open ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
      </button>
      {open && (
        <div className="px-4 pb-4 grid gap-3 sm:grid-cols-2">
          {hints.map((hint) => (
            <div
              key={hint.hint_id}
              className="rounded-lg border border-border/20 bg-muted/10 p-3"
            >
              <div className="flex items-center gap-1.5 mb-1">
                <Info className="h-3.5 w-3.5 text-primary/60" />
                <span className="text-xs font-semibold text-foreground/80">{hint.title}</span>
                <Badge variant="outline" className="text-[10px] px-1.5 py-0 h-4 ml-auto">
                  {hint.kind}
                </Badge>
              </div>
              <p className="text-xs text-muted-foreground leading-relaxed">{hint.body}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* ReportOverlay                                                               */
/* -------------------------------------------------------------------------- */

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
  const hints = reportOutput?.hints || [];

  /* ── Artifact URL maps ────────────────────────────────────────────────── */

  const figureUrls = useMemo(() => {
    if (!sessionId) return new Map<string, string>();
    const map = new Map<string, string>();
    for (const plot of plots) {
      const artifact = artifacts.find((a) => a.kind === "figure" && a.title === plot.type);
      if (artifact) map.set(plot.block_id, artifactUrl(sessionId, artifact.artifact_id));
    }
    return map;
  }, [artifacts, plots, sessionId]);

  const artifactPathToUrl = useMemo(() => {
    if (!sessionId) return new Map<string, string>();
    const map = new Map<string, string>();
    for (const a of artifacts) {
      const url = artifactUrl(sessionId, a.artifact_id);
      map.set(a.title, url);
      map.set(a.artifact_id, url);
    }
    for (const ra of reportOutput?.artifacts || []) {
      const match = artifacts.find((a) => a.kind === ra.kind && a.title === ra.title);
      if (match) {
        const url = artifactUrl(sessionId, match.artifact_id);
        map.set(ra.path, url);
        const filename = ra.path.split("/").pop() || ra.path;
        map.set(filename, url);
      }
    }
    return map;
  }, [artifacts, reportOutput?.artifacts, sessionId]);

  /* ── Markdown components (memoised) ──────────────────────────────────── */

  const mdComponents = useMemo(
    () => buildMarkdownComponents(artifactPathToUrl, figureUrls),
    [artifactPathToUrl, figureUrls],
  );

  /* ── Handlers ────────────────────────────────────────────────────────── */

  const handleExportPdf = () => window.print();

  const handleMouseUp = () => {
    const sel = window.getSelection();
    if (!sel || sel.rangeCount === 0 || sel.isCollapsed) {
      setQuoteDraft(null);
      return;
    }
    const text = sel.toString().trim();
    if (!text) {
      setQuoteDraft(null);
      return;
    }
    const range = sel.getRangeAt(0);
    const rect = range.getBoundingClientRect();
    const anchor = sel.anchorNode;
    if (!anchor || !contentRef.current || !contentRef.current.contains(anchor)) {
      setQuoteDraft(null);
      return;
    }
    const el = anchor instanceof Element ? anchor : (anchor.parentElement as Element | null);
    const section = el?.closest("section[data-omni-block-id]");
    setQuoteDraft({
      text,
      blockId: section?.getAttribute("data-omni-block-id") || undefined,
      kind: section?.getAttribute("data-omni-kind") || undefined,
      x: rect.left + rect.width / 2,
      y: rect.top - 8,
    });
  };

  const handleQuote = async () => {
    if (!quoteDraft || !onSendMessage) return;
    await onSendMessage("Please answer based on this quoted report excerpt.", [
      {
        quoted_text: quoteDraft.text,
        block_id: quoteDraft.blockId,
        kind: quoteDraft.kind,
        source: "report",
      },
    ]);
    window.getSelection()?.removeAllRanges();
    setQuoteDraft(null);
  };

  if (!results && !reportOutput) return null;

  /* ── Render ──────────────────────────────────────────────────────────── */

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
            className,
          )}
        >
          {/* ── Header bar ────────────────────────────────────────────── */}
          <div className="absolute top-0 left-0 right-0 flex items-center justify-between px-4 py-3 bg-card/90 border-b border-border/40 z-10">
            <div className="flex items-center gap-2">
              <FileText className="h-5 w-5 text-primary" />
              <h2 className="text-lg font-semibold">OmniRank Report</h2>
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

          {/* ── Scrollable content ────────────────────────────────────── */}
          <div className="absolute inset-0 top-14 overflow-auto" onMouseUp={handleMouseUp}>
            <div ref={contentRef} className="max-w-4xl mx-auto p-6 pb-24">
              {/* Meta badges */}
              <div className="flex flex-wrap items-center gap-2 mb-6">
                {schema && (
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Badge variant="secondary" className="text-xs gap-1">
                          {schema.ranking_items.length} items
                        </Badge>
                      </TooltipTrigger>
                      <TooltipContent>Number of items in the ranking</TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                )}
                {config && (
                  <>
                    <Badge variant="outline" className="text-xs gap-1 font-mono">
                      B={config.bootstrap_iterations ?? 2000}
                    </Badge>
                    <Badge variant="outline" className="text-xs gap-1 font-mono">
                      seed={config.random_seed ?? 42}
                    </Badge>
                  </>
                )}
                {typeof reportOutput?.key_findings?.n_clusters === "number" && (
                  <Badge variant="secondary" className="text-xs gap-1">
                    {String(reportOutput.key_findings.n_clusters)} tier(s)
                  </Badge>
                )}
              </div>

              {/* ── Markdown report ──────────────────────────────────── */}
              <div className="report-content">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[rehypeRaw, [rehypeSanitize, reportSanitizeSchema]]}
                  components={mdComponents}
                >
                  {markdown}
                </ReactMarkdown>
              </div>

              {/* ── Glossary / Hints ─────────────────────────────────── */}
              <GlossaryPanel hints={hints} />
            </div>
          </div>

          {/* ── Quote fab ─────────────────────────────────────────────── */}
          {quoteDraft && (
            <div
              className="fixed z-[60]"
              style={{
                left: quoteDraft.x,
                top: quoteDraft.y,
                transform: "translate(-50%, -100%)",
              }}
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
