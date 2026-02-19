"use client";

import { isValidElement, useMemo, useRef, useState, type ComponentPropsWithoutRef, type ReactNode } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { MessageSquareQuote, Moon, Sun, X } from "lucide-react";
import ReactMarkdown, { type Components } from "react-markdown";
import rehypeRaw from "rehype-raw";
import rehypeSanitize, { defaultSchema } from "rehype-sanitize";
import remarkGfm from "remark-gfm";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ForestPlot, RankingChart } from "@/components/visualizations";
import { cn } from "@/lib/utils";
import {
  artifactUrl,
  type AnalysisConfig,
  type ArtifactDescriptor,
  type HintSpec,
  type PlotSpec,
  type QuotePayload,
  type RankingItem,
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
  onQuoteToInput?: (quote: QuotePayload) => void;
  className?: string;
}

interface QuoteDraft {
  text: string;
  blockId?: string;
  kind?: string;
  x: number;
  y: number;
}

type ReportTheme = "dark" | "light";

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
    "relative bg-primary/[0.04] border border-primary/20 rounded-xl p-6 my-6 shadow-[0_0_24px_-6px_rgba(106,159,217,0.12)]",
  result: "my-6",
  table: "my-6",
  figure:
    "rounded-2xl border border-primary/20 bg-gradient-to-br from-primary/[0.06] via-card/80 to-card/40 p-5 my-7 shadow-[0_16px_40px_-24px_rgba(0,0,0,0.35)] [&>p:first-of-type]:text-[11px] [&>p:first-of-type]:uppercase [&>p:first-of-type]:tracking-wide [&>p:first-of-type]:text-primary/80 [&>p:first-of-type]:font-semibold [&>p:last-of-type]:text-xs [&>p:last-of-type]:text-muted-foreground [&>p:last-of-type]:leading-relaxed",
  comparison:
    "bg-muted/20 border border-border/40 rounded-lg p-5 my-6",
  method: "my-6",
  limitation:
    "border-l-4 border-[#FFD700]/50 bg-[#FFD700]/[0.03] rounded-r-lg pl-5 pr-4 py-4 my-6",
  repro: "bg-muted/20 border border-border/30 rounded-lg p-5 my-6 font-mono text-xs leading-relaxed",
};

const LIGHT_SECTION_STYLES: Record<string, string> = {
  summary: "relative bg-slate-50 border border-slate-200 rounded-xl p-6 my-6 shadow-[0_0_24px_-6px_rgba(15,23,42,0.06)]",
  result: "my-6",
  table: "my-6",
  figure:
    "rounded-2xl border border-slate-200 bg-white p-5 my-7 shadow-[0_16px_40px_-24px_rgba(15,23,42,0.20)] [&>p:first-of-type]:text-[11px] [&>p:first-of-type]:uppercase [&>p:first-of-type]:tracking-wide [&>p:first-of-type]:text-slate-500 [&>p:first-of-type]:font-semibold [&>p:last-of-type]:text-xs [&>p:last-of-type]:text-slate-600 [&>p:last-of-type]:leading-relaxed",
  comparison: "bg-slate-50 border border-slate-200 rounded-lg p-5 my-6",
  method: "my-6",
  limitation: "border-l-4 border-amber-400 bg-amber-50 rounded-r-lg pl-5 pr-4 py-4 my-6",
  repro: "bg-slate-50 border border-slate-200 rounded-lg p-5 my-6 font-mono text-xs leading-relaxed",
};

/* -------------------------------------------------------------------------- */
/* Custom Markdown components                                                  */
/* -------------------------------------------------------------------------- */

function parseStringArray(value: unknown): string[] | null {
  if (!Array.isArray(value)) return null;
  const out: string[] = [];
  for (const entry of value) {
    if (typeof entry !== "string") return null;
    out.push(entry);
  }
  return out;
}

function parseNumberArray(value: unknown): number[] | null {
  if (!Array.isArray(value)) return null;
  const out: number[] = [];
  for (const entry of value) {
    const numberValue = typeof entry === "number" ? entry : Number(entry);
    if (!Number.isFinite(numberValue)) return null;
    out.push(numberValue);
  }
  return out;
}

function rankingItemsFromPlot(plot: PlotSpec): RankingItem[] | null {
  const names = parseStringArray(plot.data["names"]);
  if (!names || names.length === 0) return null;

  const thetaHat = parseNumberArray(plot.data["theta_hat"]) ?? parseNumberArray(plot.data["scores"]);
  const rankPoint = parseNumberArray(plot.data["rank_point"]) ?? parseNumberArray(plot.data["ranks"]);
  const ciLower = parseNumberArray(plot.data["ci_lower"]) ?? parseNumberArray(plot.data["rank_ci_lower"]);
  const ciUpper = parseNumberArray(plot.data["ci_upper"]) ?? parseNumberArray(plot.data["rank_ci_upper"]);

  if (!thetaHat || !rankPoint || !ciLower || !ciUpper) return null;
  if (
    thetaHat.length !== names.length ||
    rankPoint.length !== names.length ||
    ciLower.length !== names.length ||
    ciUpper.length !== names.length
  ) {
    return null;
  }

  return names.map((name, index) => {
    const lower = Math.round(ciLower[index]);
    const upper = Math.round(ciUpper[index]);
    const rank = Math.round(rankPoint[index]);
    return {
      name,
      theta_hat: thetaHat[index],
      rank,
      ci_lower: lower,
      ci_upper: upper,
      ci_two_sided: [lower, upper],
    };
  });
}

function buildMarkdownComponents(
  artifactPathToUrl: Map<string, string>,
  figureUrls: Map<string, string>,
  plotsBySource: Map<string, PlotSpec>,
  rankingItems: RankingItem[],
  theme: ReportTheme,
  reportMetaBadges?: ReactNode,
): Components {
  const isLightTheme = theme === "light";

  const getHeadingText = (node: ReactNode): string => {
    if (typeof node === "string" || typeof node === "number") {
      return String(node);
    }
    if (Array.isArray(node)) {
      return node.map((child) => getHeadingText(child)).join("");
    }
    if (isValidElement<{ children?: ReactNode }>(node)) {
      return getHeadingText(node.props.children);
    }
    return "";
  };

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
          className={cn((isLightTheme ? LIGHT_SECTION_STYLES : SECTION_STYLES)[kind] || "my-4")}
        >
          {children}
        </section>
      );
    },

    /* ── Headings ──────────────────────────────────────────────────────── */
    h1: ({ children }) => {
      const headingText = getHeadingText(children).replace(/\s+/g, " ").trim();
      const isOmniRankReportTitle = headingText === "OmniRank Report";

      return (
        <header className="pb-4 mb-2 border-b border-primary/30 flex flex-wrap items-start justify-between gap-3">
          <h1 className={cn("text-2xl font-bold leading-tight", isLightTheme ? "text-slate-900" : "text-white")}>
            {isOmniRankReportTitle ? (
              <>
                <span className={cn(isLightTheme ? "text-slate-900" : "text-white")}>Omni</span>
                <span className={cn(isLightTheme ? "text-slate-900" : "text-white")}>Rank</span>
                <span className={cn(isLightTheme ? "text-slate-900" : "text-white")}> Report</span>
              </>
            ) : (
              <span className={cn(isLightTheme ? "text-slate-900" : "text-white")}>{children}</span>
            )}
          </h1>
          {reportMetaBadges ? (
            <div className="flex flex-wrap items-center justify-end gap-2">{reportMetaBadges}</div>
          ) : null}
        </header>
      );
    },
    h2: ({ children }) => (
      <h2 className={cn("text-lg font-semibold mt-0 mb-3 flex items-center gap-2", isLightTheme ? "text-slate-900" : "text-foreground")}>
        <span className="inline-block h-5 w-1 rounded-full bg-primary" />
        {children}
      </h2>
    ),
    h3: ({ children }) => (
      <h3 className={cn("text-base font-semibold mt-0 mb-2", isLightTheme ? "text-slate-900" : "text-foreground/90")}>
        {children}
      </h3>
    ),

    /* ── Tables ────────────────────────────────────────────────────────── */
    table: ({ children }) => (
      <div
        className={cn(
          "overflow-x-auto rounded-xl my-4 shadow-sm",
          isLightTheme ? "border border-slate-200 bg-white" : "border border-primary/25 bg-[#132841]",
        )}
      >
        <div className="min-w-max">
          <table className="w-full text-xs">{children}</table>
        </div>
      </div>
    ),
    thead: ({ children }) => (
      <thead className={cn("sticky top-0 z-10", isLightTheme ? "bg-slate-100" : "bg-primary/15")}>{children}</thead>
    ),
    th: ({ children }) => (
      <th
        className={cn(
          "px-3 py-2 text-left font-semibold border-b whitespace-nowrap",
          isLightTheme ? "text-slate-900 border-slate-200 bg-slate-100" : "text-slate-200 border-primary/25 bg-primary/15",
        )}
      >
        {children}
      </th>
    ),
    tbody: ({ children }) => <tbody>{children}</tbody>,
    tr: ({ children }) => (
      <tr
        className={cn(
          "border-b last:border-0 transition-colors",
          isLightTheme ? "border-slate-200 hover:bg-slate-50" : "border-primary/20 hover:bg-primary/10",
        )}
      >
        {children}
      </tr>
    ),
    td: ({ children }) => (
      <td className={cn("px-3 py-1.5 whitespace-nowrap", isLightTheme ? "text-slate-800" : "text-slate-300")}>
        {children}
      </td>
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
      <blockquote
        className={cn(
          "border-l-4 pl-4 py-1 my-4",
          isLightTheme ? "border-slate-300 text-slate-700" : "border-primary/30 text-muted-foreground",
        )}
      >
        {children}
      </blockquote>
    ),
    p: ({ children, node }) => {
      const hasBlockAstChild = (() => {
        const paragraph = node as { children?: Array<{ tagName?: string }> } | undefined;
        if (!paragraph?.children) return false;
        const blockLikeTags = new Set(["div", "section", "table", "figure", "ul", "ol", "blockquote", "img"]);
        return paragraph.children.some((child) => {
          const tagName = child?.tagName;
          return typeof tagName === "string" && blockLikeTags.has(tagName);
        });
      })();

      const hasBlockDescendant = (node: ReactNode): boolean => {
        if (node == null) return false;
        if (Array.isArray(node)) return node.some(hasBlockDescendant);
        if (typeof node === "string" || typeof node === "number") return false;
        if (!isValidElement<{ children?: ReactNode }>(node)) return false;
        const el = node as React.ReactElement<{ children?: ReactNode }>;
        if (typeof el.type === "string") {
          if (["div", "section", "table", "figure", "ul", "ol", "blockquote", "img"].includes(el.type)) return true;
        }
        const markdownTag = (el.props as { node?: { tagName?: string } }).node?.tagName;
        if (
          typeof markdownTag === "string" &&
          ["div", "section", "table", "figure", "ul", "ol", "blockquote", "img"].includes(markdownTag)
        ) {
          return true;
        }
        return hasBlockDescendant(el.props?.children);
      };
      const hasBlockChild = hasBlockAstChild || hasBlockDescendant(children);

      const className = cn("text-sm leading-relaxed mb-3", isLightTheme ? "text-slate-800" : "text-foreground/90");
      return hasBlockChild ? (
        <div className={className}>{children}</div>
      ) : (
        <p className={className}>{children}</p>
      );
    },
    ul: ({ children }) => (
      <ul className="space-y-1.5 my-3 list-none pl-0">{children}</ul>
    ),
    ol: ({ children }) => (
      <ol className="space-y-1.5 my-3 list-decimal pl-5">{children}</ol>
    ),
    li: ({ children }) => (
      <li className={cn("flex items-start gap-2 text-sm leading-relaxed", isLightTheme ? "text-slate-800" : "text-foreground/90")}>
        <span className={cn("mt-[7px] h-1.5 w-1.5 rounded-full shrink-0", isLightTheme ? "bg-slate-400" : "bg-primary/50")} />
        <span className="flex-1">{children}</span>
      </li>
    ),

    /* ── Inline elements ───────────────────────────────────────────────── */
    strong: ({ children }) => (
      <strong className={cn("font-semibold", isLightTheme ? "text-slate-900" : "text-foreground")}>{children}</strong>
    ),
    em: ({ children }) => (
      <em className={cn("italic", isLightTheme ? "text-slate-600" : "text-muted-foreground")}>{children}</em>
    ),
    code: ({ children }) => (
      <code
        className={cn(
          "px-1.5 py-0.5 rounded text-xs font-mono",
          isLightTheme ? "bg-slate-100 text-slate-900" : "bg-muted/50 text-primary/80",
        )}
      >
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

      const normalizedFilename = normalizedSrc.split("/").pop() || normalizedSrc;
      const matchedPlot =
        plotsBySource.get(source) ||
        plotsBySource.get(filename) ||
        plotsBySource.get(normalizedSrc) ||
        plotsBySource.get(normalizedFilename);

      if (matchedPlot) {
        const interactiveItems =
          rankingItems.length > 0 ? rankingItems : rankingItemsFromPlot(matchedPlot) || [];

        if (interactiveItems.length > 0) {
          if (matchedPlot.type === "ranking_bar") {
            return (
              <div className="my-2 overflow-hidden rounded-xl p-0">
                <RankingChart items={interactiveItems} className="w-full" theme={theme} />
              </div>
            );
          }
          if (matchedPlot.type === "ci_forest") {
            return (
              <div className="my-2 overflow-hidden rounded-xl p-0">
                <ForestPlot items={interactiveItems} className="w-full" theme={theme} />
              </div>
            );
          }
        }
      }

      return (
        <img
          src={normalizedSrc}
          alt={alt || "report figure"}
          className="my-2 w-full rounded-xl border border-border/30"
        />
      );
    },
  };
}

/* -------------------------------------------------------------------------- */
/* Glossary panel                                                              */
/* -------------------------------------------------------------------------- */

function GlossaryPanel({ hints, theme }: { hints: HintSpec[]; theme: ReportTheme }) {
  if (!hints || hints.length === 0) return null;
  const isLightTheme = theme === "light";

  return (
    <section className="my-6">
      <h2 className={cn("text-lg font-semibold mt-0 mb-3 flex items-center gap-2", isLightTheme ? "text-slate-900" : "text-foreground")}>
        <span className="inline-block h-5 w-1 rounded-full bg-primary" />
        Terms and Definitions
      </h2>
      <div className="space-y-5">
        {hints.map((hint) => (
          <div key={hint.hint_id}>
            <div className="flex items-center gap-2 mb-1.5">
              <h3 className={cn("text-base font-semibold mt-0 mb-0", isLightTheme ? "text-slate-900" : "text-foreground/90")}>
                {hint.title}
              </h3>
              <Badge
                variant="outline"
                className={cn(
                  "text-[10px] px-1.5 py-0 h-4 shrink-0",
                  isLightTheme ? "border-slate-300 text-slate-700 bg-white" : "border-border text-muted-foreground",
                )}
              >
                {hint.kind}
              </Badge>
            </div>
            <p className={cn("text-sm leading-relaxed", isLightTheme ? "text-slate-800" : "text-foreground/90")}>{hint.body}</p>
          </div>
        ))}
      </div>
    </section>
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
  onQuoteToInput,
  className,
}: ReportOverlayProps) {
  const contentRef = useRef<HTMLDivElement>(null);
  const [quoteDraft, setQuoteDraft] = useState<QuoteDraft | null>(null);
  const [reportTheme, setReportTheme] = useState<ReportTheme>("dark");
  const isLightTheme = reportTheme === "light";

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

  const plotsBySource = useMemo(() => {
    const map = new Map<string, PlotSpec>();
    const plotsByBlockId = new Map(plots.map((plot) => [plot.block_id, plot] as const));

    for (const plot of plots) {
      const source = plot.svg_path;
      const filename = source.split("/").pop() || source;
      map.set(source, plot);
      map.set(filename, plot);
    }

    for (const [blockId, url] of figureUrls) {
      const plot = plotsByBlockId.get(blockId);
      if (!plot) continue;
      map.set(url, plot);
      const filename = url.split("/").pop() || url;
      map.set(filename, plot);
    }

    for (const [source, url] of artifactPathToUrl) {
      const filename = source.split("/").pop() || source;
      const plot = map.get(source) || map.get(filename);
      if (!plot) continue;
      map.set(url, plot);
      const urlFilename = url.split("/").pop() || url;
      map.set(urlFilename, plot);
    }

    return map;
  }, [artifactPathToUrl, figureUrls, plots]);

  const reportMetaBadges = useMemo(
    () => (
      <>
        {schema && (
          <Badge
            variant="secondary"
            className={cn("text-xs gap-1", isLightTheme ? "bg-slate-100 border border-slate-300 text-slate-800" : "")}
          >
            {schema.ranking_items.length} items
          </Badge>
        )}
        {config && (
          <>
            <Badge
              variant="outline"
              className={cn("text-xs gap-1 font-mono", isLightTheme ? "border-slate-300 text-slate-800 bg-white" : "")}
            >
              B={config.bootstrap_iterations ?? 2000}
            </Badge>
            <Badge
              variant="outline"
              className={cn("text-xs gap-1 font-mono", isLightTheme ? "border-slate-300 text-slate-800 bg-white" : "")}
            >
              seed={config.random_seed ?? 42}
            </Badge>
          </>
        )}
      </>
    ),
    [config, isLightTheme, schema],
  );

  /* ── Markdown components ─────────────────────────────────────────────── */

  const mdComponents = useMemo(
    () =>
      buildMarkdownComponents(
        artifactPathToUrl,
        figureUrls,
        plotsBySource,
        results?.items || [],
        reportTheme,
        reportMetaBadges,
      ),
    [artifactPathToUrl, figureUrls, plotsBySource, reportMetaBadges, reportTheme, results?.items],
  );
  const renderedMarkdown = useMemo(
    () => (
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeRaw, [rehypeSanitize, reportSanitizeSchema]]}
        components={mdComponents}
      >
        {markdown}
      </ReactMarkdown>
    ),
    [markdown, mdComponents],
  );

  /* ── Handlers ────────────────────────────────────────────────────────── */

  const handleToggleTheme = () => {
    setReportTheme((prev) => (prev === "dark" ? "light" : "dark"));
  };

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
    const nextDraft: QuoteDraft = {
      text,
      blockId: section?.getAttribute("data-omni-block-id") || undefined,
      kind: section?.getAttribute("data-omni-kind") || undefined,
      x: rect.left + rect.width / 2,
      y: rect.top - 8,
    };
    setQuoteDraft((prev) => {
      if (
        prev &&
        prev.text === nextDraft.text &&
        prev.blockId === nextDraft.blockId &&
        prev.kind === nextDraft.kind
      ) {
        return prev;
      }
      return nextDraft;
    });
  };

  const handleQuote = () => {
    if (!quoteDraft || !onQuoteToInput) return;
    onQuoteToInput({
      quoted_text: quoteDraft.text,
      block_id: quoteDraft.blockId,
      kind: quoteDraft.kind,
      source: "report",
    });
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
            "absolute inset-0 z-50 backdrop-blur-sm rounded-lg overflow-hidden",
            isLightTheme ? "bg-white/95 text-slate-900" : "bg-card/95 text-foreground",
            className,
          )}
        >
          {/* ── Header bar ────────────────────────────────────────────── */}
          <div
            className={cn(
              "absolute top-0 left-0 right-0 flex items-center justify-between px-4 py-3 z-10",
              isLightTheme ? "bg-white/95 border-b border-slate-200" : "bg-card/90 border-b border-border/40",
            )}
          >
            <Button
              variant="outline"
              size="icon-sm"
              onClick={handleToggleTheme}
              className={cn(
                "rounded-full border shadow-sm",
                isLightTheme
                  ? "border-slate-300 bg-white text-slate-700 hover:bg-slate-100 hover:border-slate-400"
                  : "border-primary/30 bg-primary/10 text-primary hover:bg-primary/15 hover:border-primary/50",
              )}
              aria-label={isLightTheme ? "Switch report to dark mode" : "Switch report to light mode"}
              title={isLightTheme ? "Switch to dark mode" : "Switch to light mode"}
            >
              {isLightTheme ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </Button>
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={onClose}
              className={cn(
                "rounded-full border",
                isLightTheme
                  ? "border-slate-300 bg-white hover:bg-slate-100 hover:border-slate-400 text-slate-700"
                  : "border-border/60 bg-background/80 hover:bg-muted/70 hover:border-border/90",
              )}
              aria-label="Close report"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>

          {/* ── Scrollable content ────────────────────────────────────── */}
          <div className="absolute inset-0 top-14" onMouseUp={handleMouseUp}>
            <ScrollArea className="h-full">
              <div ref={contentRef} className={cn("max-w-4xl mx-auto p-6 pb-24", isLightTheme ? "text-slate-900" : "")}>
                {/* ── Markdown report ──────────────────────────────────── */}
                <div className="report-content">
                  {renderedMarkdown}
                </div>

                {/* ── Glossary / Hints ─────────────────────────────────── */}
                <GlossaryPanel hints={hints} theme={reportTheme} />
              </div>
            </ScrollArea>
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
