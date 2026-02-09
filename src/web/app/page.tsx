"use client";

import { useEffect, useState, type MouseEvent } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  ArrowRight,
  Bot,
  ChevronRight,
  CirclePlay,
  Database,
  FileText,
  FlaskConical,
  Github,
  Trophy,
  type LucideIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { HeroScene } from "@/components/landing/hero-scene";

interface IntroModule {
  id: string;
  label: string;
  duration: string;
  title: string;
  summary: string;
}

interface FeatureStory {
  title: string;
  description: string;
  icon: LucideIcon;
  visual: string;
}

interface UseCase {
  title: string;
  summary: string;
}

interface WorkflowStep {
  title: string;
  description: string;
}

const introModules: IntroModule[] = [
  {
    id: "intro",
    label: "1",
    duration: "2 mins",
    title: "Introduction",
    summary: "What OmniRank does.",
  },
  {
    id: "pairwise",
    label: "2",
    duration: "1 min",
    title: "Pairwise Ranking",
    summary: "Head-to-head comparisons.",
  },
  {
    id: "pointwise",
    label: "3",
    duration: "1 min",
    title: "Pointwise Scoring",
    summary: "Score-based ranking.",
  },
  {
    id: "multiway",
    label: "4",
    duration: "2 mins",
    title: "Multiway Ranking",
    summary: "Listwise rank inference.",
  },
  {
    id: "report",
    label: "5",
    duration: "1 min",
    title: "Report Export",
    summary: "Single-page report output.",
  },
];

const featureStories: FeatureStory[] = [
  {
    title: "AI-guided Data Checks",
    description: "Schema validation before compute.",
    icon: Bot,
    visual: "Validation",
  },
  {
    title: "Deterministic Spectral Ranking",
    description: "Point estimate + confidence interval.",
    icon: FlaskConical,
    visual: "Inference",
  },
  {
    title: "Research-ready Workflow",
    description: "One workspace for chat, ranking, report.",
    icon: FileText,
    visual: "Workspace",
  },
];

const useCases: UseCase[] = [
  {
    title: "LLM Pairwise Evaluation",
    summary: "Coding, math, and writing model matchups.",
  },
  {
    title: "Benchmark Suite Ranking",
    summary: "Accuracy tables into robust ordering.",
  },
  {
    title: "Human Preference Studies",
    summary: "Large pairwise studies with uncertainty.",
  },
  {
    title: "Product Tournament Decisions",
    summary: "Weekly A/B and multi-option ranking.",
  },
];

const workflowSteps: WorkflowStep[] = [
  { title: "Open Workspace", description: "Start a new run." },
  { title: "Load CSV", description: "Pairwise, pointwise, or multiway." },
  { title: "Run Analysis", description: "Validate, rank, and bootstrap." },
  { title: "Export Report", description: "Share rank + CI evidence." },
];

const navItems = [
  { id: "introduction", label: "Videos" },
  { id: "features", label: "Features" },
  { id: "use-cases", label: "Use Cases" },
  { id: "how-to-use", label: "How to Use" },
] as const;

export default function LandingPage() {
  const [activeIntroId, setActiveIntroId] = useState(introModules[0].id);
  const [activeUseCaseIndex, setActiveUseCaseIndex] = useState(0);
  const [isNavFloating, setIsNavFloating] = useState(false);

  const activeIntro = introModules.find((module) => module.id === activeIntroId) ?? introModules[0];
  const activeUseCase = useCases[activeUseCaseIndex] ?? useCases[0];

  const scrollToSection = (sectionId: string) => {
    const section = document.getElementById(sectionId);
    if (!section) {
      return;
    }

    section.scrollIntoView({ behavior: "smooth", block: "start" });
    if (window.location.hash) {
      window.history.replaceState(null, "", window.location.pathname + window.location.search);
    }
  };

  const handleSectionButton = (event: MouseEvent<HTMLButtonElement>, sectionId: string) => {
    event.preventDefault();
    scrollToSection(sectionId);
  };

  useEffect(() => {
    const updateNavState = () => {
      setIsNavFloating(window.scrollY > 36);
    };

    updateNavState();
    window.addEventListener("scroll", updateNavState, { passive: true });
    return () => {
      window.removeEventListener("scroll", updateNavState);
    };
  }, []);

  return (
    <main className="relative min-h-screen overflow-x-hidden text-foreground">
      <HeroScene className="fixed inset-0 -z-30 opacity-90" />
      <div className="pointer-events-none fixed inset-0 -z-20 bg-[radial-gradient(circle_at_20%_18%,rgba(152,132,229,0.24),transparent_40%),radial-gradient(circle_at_80%_14%,rgba(197,184,246,0.16),transparent_46%),radial-gradient(circle_at_50%_84%,rgba(16,25,46,0.88),transparent_62%)]" />
      <div className="pointer-events-none fixed inset-0 -z-10 bg-gradient-to-b from-background/86 via-background/94 to-background" />

      <div
        className={cn(
          "fixed inset-x-0 z-50 flex justify-center px-4 transition-all duration-500 md:px-6",
          isNavFloating ? "top-4" : "top-0",
        )}
      >
        <nav
          className={cn(
            "w-full transition-all duration-500 ease-out",
            isNavFloating
              ? "max-w-5xl rounded-full border border-border/55 bg-background/52 px-5 backdrop-blur-xl shadow-[0_18px_42px_rgba(0,0,0,0.35)]"
              : "max-w-7xl bg-transparent",
          )}
        >
          <div className={cn("flex items-center justify-between", isNavFloating ? "h-14" : "h-16")}>
            <Link href="/" className="text-xl font-semibold tracking-wide text-foreground">
              Omni<span className="text-primary">Rank</span>
            </Link>

            <Link
              href="/workspace"
              target="_blank"
              rel="noopener noreferrer"
              className={cn(
                "inline-flex items-center gap-1 rounded-full text-sm font-semibold transition-colors",
                isNavFloating
                  ? "bg-primary px-3.5 py-2 text-primary-foreground hover:bg-primary/90"
                  : "bg-primary/90 px-3.5 py-2 text-primary-foreground hover:bg-primary",
              )}
            >
              Online Workspace
              <ArrowRight className="h-4 w-4" />
            </Link>
          </div>
        </nav>
      </div>

      <div className="relative z-10 text-foreground">
        <section className="min-h-screen px-4 pb-24 pt-36 md:px-6 md:pt-40">
          <div className="mx-auto flex w-full max-w-7xl flex-col items-center text-center">
            <motion.div
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7, ease: "easeOut" }}
              className="w-full translate-y-5 md:translate-y-7"
            >
              <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-primary/35 bg-card/70 px-4 py-1.5 text-base uppercase tracking-[0.22em] text-primary">
                <Database className="h-3.5 w-3.5" />
                Agentic Ranking Platform
              </div>

              <h1 className="text-balance text-6xl font-bold leading-tight md:text-8xl">
                Omni<span className="text-primary">Rank</span>
              </h1>

              <p className="mx-auto mt-4 max-w-2xl text-lg text-muted-foreground md:text-xl">
                LLM Agent base ranking platform with reliable results.
              </p>

              <div className="mx-auto mt-8 grid w-full max-w-3xl grid-cols-1 gap-3 sm:grid-cols-2">
                <Link
                  href="/workspace"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex min-h-12 items-center justify-center gap-2 rounded-full bg-primary px-5 py-3 text-lg font-semibold text-primary-foreground transition-colors hover:bg-primary/90"
                >
                  <ArrowRight className="h-4 w-4" />
                  Online Workspace
                </Link>
                <Link
                  href="/workspace"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex min-h-12 items-center justify-center gap-2 rounded-full bg-primary px-5 py-3 text-lg font-semibold text-primary-foreground transition-colors hover:bg-primary/90"
                >
                  <Trophy className="h-4 w-4" />
                  LLM Leaderboard
                </Link>
              </div>

              <div className="mx-auto mt-4 flex w-full max-w-3xl flex-wrap justify-center gap-2">
                <a
                  href="https://arxiv.org"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 rounded-full border border-border/60 bg-card/70 px-3 py-1.5 text-base text-muted-foreground transition-colors hover:text-foreground"
                >
                  <FileText className="h-3.5 w-3.5" />
                  Preprint
                </a>
                <a
                  href="https://github.com"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 rounded-full border border-border/60 bg-card/70 px-3 py-1.5 text-base text-muted-foreground transition-colors hover:text-foreground"
                >
                  <Github className="h-3.5 w-3.5" />
                  GitHub
                </a>
              </div>
            </motion.div>
          </div>
        </section>

        <section id="introduction" className="min-h-screen px-4 py-28 md:px-6">
          <div className="mx-auto w-full max-w-6xl">
            <h2 className="mb-8 text-center text-3xl font-bold md:text-4xl">Introduction</h2>

            <div className="rounded-3xl border border-border/55 bg-card/70 p-4 shadow-2xl shadow-black/20 backdrop-blur-lg md:p-6">
              <div className="grid gap-4 lg:grid-cols-[320px,1fr]">
                <div className="space-y-2">
                  {introModules.map((module) => {
                    const isActive = module.id === activeIntro.id;
                    return (
                      <button
                        key={module.id}
                        type="button"
                        onClick={() => setActiveIntroId(module.id)}
                        className={cn(
                          "flex w-full items-start gap-3 rounded-xl border px-3 py-3 text-left transition-colors",
                          isActive
                            ? "border-primary/60 bg-primary/18 text-foreground"
                            : "border-border/60 bg-background/55 text-muted-foreground hover:border-primary/35 hover:text-foreground",
                        )}
                      >
                        <span
                          className={cn(
                            "mt-0.5 inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-[11px] font-semibold",
                            isActive ? "bg-primary text-primary-foreground" : "bg-card text-muted-foreground",
                          )}
                        >
                          {module.label}
                        </span>
                        <span className="min-w-0">
                          <span className="block text-sm font-semibold">{module.title}</span>
                          <span className="mt-1 block text-xs opacity-80">{module.duration}</span>
                        </span>
                      </button>
                    );
                  })}
                </div>

                <div className="rounded-2xl border border-border/60 bg-background/65 p-5">
                  <div className="relative overflow-hidden rounded-xl border border-border/55 bg-card/70">
                    <div className="flex items-center gap-2 border-b border-border/45 px-3 py-2 text-xs text-muted-foreground">
                      <CirclePlay className="h-3.5 w-3.5 text-primary" />
                      Video Preview
                    </div>
                    <div className="grid min-h-[220px] place-items-center bg-gradient-to-br from-background/80 via-card/65 to-background/85">
                      <div className="text-center">
                        <p className="text-base font-semibold">{activeIntro.title}</p>
                        <p className="mt-1 text-xs text-muted-foreground">{activeIntro.duration}</p>
                      </div>
                    </div>
                  </div>

                  <p className="mt-4 text-sm text-muted-foreground md:text-base">{activeIntro.summary}</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section id="features" className="min-h-screen px-4 py-28 md:px-6">
          <div className="mx-auto w-full max-w-6xl space-y-14">
            {featureStories.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <motion.article
                  key={feature.title}
                  initial={{ opacity: 0, y: 18 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true, amount: 0.2 }}
                  transition={{ duration: 0.55, ease: "easeOut" }}
                  className={cn(
                    "grid items-center gap-8 rounded-3xl border border-border/55 bg-card/70 p-5 backdrop-blur-lg lg:grid-cols-2",
                    index % 2 === 1 && "lg:[&>*:first-child]:order-2",
                  )}
                >
                  <div>
                    <div className="mb-4 inline-flex h-10 w-10 items-center justify-center rounded-lg border border-primary/35 bg-primary/15 text-primary">
                      <Icon className="h-5 w-5" />
                    </div>
                    <h3 className="text-2xl font-semibold text-foreground md:text-3xl">{feature.title}</h3>
                    <p className="mt-3 text-sm text-muted-foreground md:text-base">{feature.description}</p>
                  </div>

                  <div className="rounded-2xl border border-border/60 bg-background/75 p-4">
                    <div className="mb-3 text-xs uppercase tracking-[0.2em] text-muted-foreground">{feature.visual}</div>
                    <div className="space-y-2">
                      <div className="h-2 w-2/3 rounded-full bg-primary/75" />
                      <div className="h-2 w-[82%] rounded-full bg-primary/45" />
                      <div className="h-2 w-1/2 rounded-full bg-primary/55" />
                    </div>
                  </div>
                </motion.article>
              );
            })}
          </div>
        </section>

        <section id="use-cases" className="min-h-screen px-4 py-28 md:px-6">
          <div className="mx-auto w-full max-w-6xl rounded-3xl border border-border/55 bg-card/70 p-4 shadow-2xl shadow-black/20 backdrop-blur-lg md:p-6">
            <div className="grid gap-4 lg:grid-cols-5">
              <div className="space-y-2 lg:col-span-3">
                {useCases.map((useCase, index) => {
                  const isActive = index === activeUseCaseIndex;
                  return (
                    <button
                      key={useCase.title}
                      type="button"
                      onClick={() => setActiveUseCaseIndex(index)}
                      className={cn(
                        "w-full rounded-xl border px-4 py-3 text-left transition-colors",
                        isActive
                          ? "border-primary/60 bg-primary/16 text-foreground"
                          : "border-border/60 bg-background/55 text-muted-foreground hover:border-primary/35 hover:text-foreground",
                      )}
                    >
                      <div className="flex items-start justify-between gap-3">
                        <span className="text-sm font-semibold md:text-base">{useCase.title}</span>
                        <ChevronRight className={cn("mt-0.5 h-4 w-4", isActive ? "text-primary" : "text-muted-foreground")} />
                      </div>
                    </button>
                  );
                })}
              </div>

              <div className="lg:col-span-2">
                <div className="h-full min-h-[300px] rounded-2xl border border-border/60 bg-background/70 p-5">
                  <div className="inline-flex items-center gap-2 rounded-full border border-primary/35 bg-primary/15 px-2.5 py-1 text-xs text-primary">
                    <FileText className="h-3.5 w-3.5" />
                    Active Scenario
                  </div>
                  <h3 className="mt-4 text-xl font-semibold">{activeUseCase.title}</h3>
                  <p className="mt-3 text-sm leading-relaxed text-muted-foreground">{activeUseCase.summary}</p>

                  <div className="mt-5 rounded-xl border border-border/50 bg-card/75 p-3">
                    <p className="text-xs uppercase tracking-wide text-muted-foreground">Signal</p>
                    <p className="mt-1 text-2xl font-bold text-primary">Stable rank + CI</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section id="how-to-use" className="min-h-screen px-4 py-28 md:px-6">
          <div className="mx-auto w-full max-w-6xl rounded-3xl border border-border/55 bg-card/70 p-4 shadow-2xl shadow-black/20 backdrop-blur-lg md:p-6">
            <h2 className="text-center text-3xl font-bold md:text-4xl">How to Use OmniRank</h2>

            <div className="mt-10 grid gap-10 md:mt-14 lg:grid-cols-2">
              <div className="space-y-6">
                {workflowSteps.map((step, index) => (
                  <div key={step.title} className="flex gap-4">
                    <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary text-xs font-semibold text-primary-foreground">
                      {index + 1}
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-foreground">{step.title}</h3>
                      <p className="mt-1 text-sm text-muted-foreground md:text-base">{step.description}</p>
                    </div>
                  </div>
                ))}
              </div>

              <div className="rounded-2xl border border-border/60 bg-background/70 p-4">
                <div className="mb-3 flex items-center gap-2 text-sm font-medium text-muted-foreground">
                  <CirclePlay className="h-4 w-4 text-primary" />
                  Tutorial Preview
                </div>
                <div className="grid min-h-[280px] place-items-center rounded-lg border border-border/45 bg-card/70">
                  <p className="text-sm text-muted-foreground">Watch demo in workspace</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        <footer className="border-t border-border/55 bg-card/70 py-10 backdrop-blur-xl">
          <div className="mx-auto flex w-full max-w-6xl flex-col items-center justify-between gap-5 px-4 text-sm text-muted-foreground md:flex-row md:px-6">
            <div className="inline-flex items-center gap-2">
              <span className="grid h-7 w-7 place-items-center rounded-md border border-primary/35 bg-primary/15 text-primary">
                <Database className="h-3.5 w-3.5" />
              </span>
              <span>OmniRank Platform</span>
            </div>

            <div className="flex flex-wrap items-center justify-center gap-5">
              {navItems.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  onClick={(event) => handleSectionButton(event, item.id)}
                  className="transition-colors hover:text-foreground"
                >
                  {item.label}
                </button>
              ))}
              <Link href="/workspace" target="_blank" rel="noopener noreferrer" className="transition-colors hover:text-foreground">
                Workspace
              </Link>
            </div>

            <p>© 2026 OmniRank</p>
          </div>
        </footer>
      </div>
    </main>
  );
}
