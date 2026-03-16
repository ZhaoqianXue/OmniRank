"use client";

import { useCallback, useEffect } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  ArrowRight,
  BarChart3,
  BookOpen,
  Brain,
  CirclePlay,
  FileSpreadsheet,
  FileText,
  Github,
  MessageSquareText,
  Trophy,
  type LucideIcon,
} from "lucide-react";
import { HeroScene } from "@/components/landing/hero-scene";
import { SiteNavbar } from "@/components/layout/site-navbar";

interface WorkflowStep {
  title: string;
  description: string;
}

interface KeyFeature {
  title: string;
  description: string;
  icon: LucideIcon;
}

const workflowSteps: WorkflowStep[] = [
  {
    title: "Upload Data",
    description: "Upload your data file to get started.",
  },
  {
    title: "Confirm Parameters",
    description: "Review the detected settings and adjust them if needed.",
  },
  {
    title: "View Report",
    description: "Review the generated rankings, charts, and key insights.",
  },
];

const keyFeatures: KeyFeature[] = [
  {
    title: "An Agentic Ranking Copilot",
    description:
      "OmniRank is an AI copilot for ranking analysis. You describe your objective in plain language, and the system plans the statistical workflow—from data interpretation to final report generation.",
    icon: MessageSquareText,
  },
  {
    title: "Automated Comparison Structure Identification",
    description:
      "The system automatically infers the comparison structure from raw data, identifying the compared items, the winner, and the variables used for stratified analysis.",
    icon: Brain,
  },
  {
    title: "Principled Ranking Analysis Engine",
    description:
      "At its core, OmniRank performs ranking estimation using spectral methods with built-in diagnostics and uncertainty quantification, ensuring that decisions are based on reliable signals.",
    icon: BarChart3,
  },
  {
    title: "Decision-Ready Outputs",
    description:
      "The system generates a complete analysis artifact, including interpretable rankings, uncertainty summaries, and exportable visuals that teams can review, share, and reuse in publications or product decisions.",
    icon: FileSpreadsheet,
  },
];

const SCROLL_TO_HOW_TO_USE_KEY = "omnirank:scroll-to-how-to-use";
const HOW_TO_USE_SCROLL_OFFSET = 24;

export default function LandingPage() {
  const performHowToUseScroll = useCallback(() => {
    const howToUseSection = document.getElementById("how-to-use");
    if (!howToUseSection) return;

    const targetTop = Math.max(0, howToUseSection.getBoundingClientRect().top + window.scrollY - HOW_TO_USE_SCROLL_OFFSET);
    window.scrollTo({ top: targetTop, behavior: "smooth" });

    const nextUrl = `${window.location.pathname}${window.location.search}`;
    window.history.replaceState(null, "", nextUrl);
  }, []);

  const scrollToHowToUse = useCallback(() => {
    performHowToUseScroll();
  }, [performHowToUseScroll]);

  useEffect(() => {
    const shouldScroll = window.sessionStorage.getItem(SCROLL_TO_HOW_TO_USE_KEY);
    if (shouldScroll !== "1") return;

    window.sessionStorage.removeItem(SCROLL_TO_HOW_TO_USE_KEY);
    window.requestAnimationFrame(() => {
      performHowToUseScroll();
    });
  }, [performHowToUseScroll]);

  return (
    <main className="relative min-h-screen overflow-x-hidden text-foreground">
      <HeroScene className="fixed inset-0 -z-30 opacity-90" />
      <div className="pointer-events-none fixed inset-0 -z-20 bg-[radial-gradient(circle_at_20%_18%,rgba(106,159,217,0.24),transparent_40%),radial-gradient(circle_at_80%_14%,rgba(159,194,232,0.16),transparent_46%),radial-gradient(circle_at_50%_84%,rgba(11,26,48,0.88),transparent_62%)]" />
      <div className="pointer-events-none fixed inset-0 -z-10 bg-gradient-to-b from-background/86 via-background/94 to-background" />
      <SiteNavbar />

      <div className="relative z-10 text-foreground">
        <section className="min-h-screen px-4 pb-12 pt-36 md:px-6 md:pb-14 md:pt-44">
          <div className="mx-auto flex w-full max-w-7xl flex-col items-center text-center">
            <motion.div
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7, ease: "easeOut" }}
              className="w-full translate-y-14 md:translate-y-16"
            >
              <h1 className="text-balance text-5xl font-bold leading-tight md:text-7xl">
                Omni<span className="text-primary">Rank</span>
              </h1>

              <p className="mx-auto mt-4 w-fit max-w-none whitespace-nowrap text-center font-[family-name:var(--font-space-mono)] text-sm tracking-[-0.04em] [word-spacing:-0.1em] text-muted-foreground md:text-base">
                An agentic AI platform for{" "}
                <a
                  href="https://arxiv.org/html/2308.02918"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary underline underline-offset-2 hover:text-primary/80"
                >
                  Ranking Analysis
                </a>{" "}
                developed by{" "}
                <a
                  href="https://jin93.github.io/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary underline underline-offset-2 hover:text-primary/80"
                >
                  Jin Lab
                </a>{" "}
                and{" "}
                <a
                  href="https://maxineyu.github.io/personal_web/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary underline underline-offset-2 hover:text-primary/80"
                >
                  Yu Lab
                </a>
              </p>

              <div className="mx-auto mt-8 grid w-full max-w-3xl grid-cols-1 gap-3 sm:grid-cols-3">
                <button
                  type="button"
                  onClick={scrollToHowToUse}
                  className="inline-flex min-h-12 items-center justify-center gap-2 rounded-full border border-primary/48 bg-card/26 px-5 py-3 text-base font-semibold text-primary transition-all duration-300 hover:bg-card/36 hover:border-primary/66"
                >
                  <BookOpen className="h-4 w-4 text-primary" />
                  User Guide
                </button>
                <Link
                  href="/workspace"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex min-h-12 items-center justify-center gap-2 rounded-full bg-primary/80 px-5 py-3 text-base font-semibold text-primary-foreground transition-colors duration-300 hover:bg-primary/72"
                >
                  <ArrowRight className="h-4 w-4" />
                  Start Ranking
                </Link>
                <Link
                  href="/leaderboard"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex min-h-12 items-center justify-center gap-2 rounded-full border border-primary/48 bg-card/26 px-5 py-3 text-base font-semibold text-primary transition-all duration-300 hover:bg-card/36 hover:border-primary/66"
                >
                  <Trophy className="h-4 w-4 text-primary" />
                  LLM Leaderboard
                </Link>
              </div>

              <div className="mx-auto mt-8 flex flex-wrap items-center justify-center gap-2">
                <a
                  href="https://arxiv.org/html/2308.02918"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex min-h-10 items-center justify-center gap-1.5 whitespace-nowrap rounded-full border border-border/60 bg-card/26 px-5 py-2.5 text-sm text-muted-foreground transition-colors hover:bg-card/36 hover:text-foreground"
                >
                  <FileText className="h-4 w-4" />
                  Paper
                </a>
                <a
                  href="https://github.com"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex min-h-10 items-center justify-center gap-1.5 whitespace-nowrap rounded-full border border-border/60 bg-card/26 px-5 py-2.5 text-sm text-muted-foreground transition-colors hover:bg-card/36 hover:text-foreground"
                >
                  <Github className="h-4 w-4" />
                  GitHub
                </a>
              </div>
            </motion.div>
          </div>
        </section>

        <section id="what-is-omnirank" className="scroll-mt-32 px-4 pb-16 pt-28 md:px-6 md:pb-16 md:pt-28">
          <div className="mx-auto w-full max-w-6xl">
            <motion.div
              initial={{ opacity: 0, y: 18 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.3 }}
              transition={{ duration: 0.5, ease: "easeOut" }}
              className="text-center"
            >
              <h2 className="mb-4 text-center text-3xl font-bold md:text-4xl">What is OmniRank</h2>
              <p className="mx-auto mb-8 max-w-2xl text-center text-muted-foreground md:mb-10">
                OmniRank is an end-to-end agentic AI platform for ranking analysis, designed to convert diverse comparison data into statistically grounded, decision-ready ranking reports.
              </p>
            </motion.div>

            <div className="grid items-start gap-8 md:grid-cols-2 md:gap-x-12 md:gap-y-8">
              {keyFeatures.map((feature, index) => {
                const Icon = feature.icon;
                return (
                  <motion.div
                    key={feature.title}
                    initial={{ opacity: 0, y: 18 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true, amount: 0.2 }}
                    transition={{ duration: 0.5, delay: index * 0.08, ease: "easeOut" }}
                    className="flex gap-4"
                  >
                    <div className="mt-1 inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-primary/15 text-primary">
                      <Icon className="h-5 w-5" />
                    </div>
                    <div>
                      <h3 className="text-xl font-semibold text-foreground">{feature.title}</h3>
                      <p className="mt-2 text-sm leading-relaxed text-muted-foreground md:text-base">
                        {feature.description}
                      </p>
                    </div>
                  </motion.div>
                );
              })}
            </div>

          </div>
        </section>

        <section id="how-to-use" className="scroll-mt-24 px-4 pb-8 pt-20 md:px-6 md:pb-10 md:pt-22">
          <div className="mx-auto w-full max-w-6xl">
            <h2 className="mb-10 text-center text-3xl font-bold md:mb-12 md:text-4xl">How to Use OmniRank</h2>

            <div className="grid items-start gap-8 lg:grid-cols-[minmax(0,0.82fr)_minmax(0,1.18fr)] lg:gap-8">
              <div className="max-w-md space-y-5 lg:justify-self-end">
                {workflowSteps.map((step, index) => (
                  <div key={step.title} className="flex gap-5">
                    <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-primary text-sm font-bold text-primary-foreground">
                      {index + 1}
                    </div>
                    <div className="pt-0.5">
                      <h3 className="text-lg font-semibold text-foreground">{step.title}</h3>
                      <p className="mt-1 text-sm leading-relaxed text-muted-foreground">{step.description}</p>
                    </div>
                  </div>
                ))}

                <div className="flex gap-5">
                  <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground">
                    <MessageSquareText className="h-4 w-4" />
                  </div>
                  <div className="pt-0.5">
                    <h3 className="text-lg font-semibold text-foreground">Ask OmniAgent Questions</h3>
                    <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
                      Ask for help anytime while you work.
                    </p>
                  </div>
                </div>
              </div>

              <div className="lg:sticky lg:top-24">
                <div className="overflow-hidden rounded-xl border border-border/60 bg-background/80 shadow-xl shadow-black/10">
                  <div className="relative aspect-video min-h-[320px] w-full bg-card/90 md:min-h-[380px]">
                    <div className="absolute inset-0 grid place-items-center">
                      <div className="text-center">
                        <div className="mx-auto mb-3 grid h-16 w-16 place-items-center rounded-full bg-primary/90 text-primary-foreground shadow-lg transition-transform hover:scale-105">
                          <CirclePlay className="h-8 w-8" />
                        </div>
                        <p className="text-sm text-muted-foreground">Video coming soon</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <div className="px-4 pb-10 pt-0 md:px-6 md:pb-12">
          <div className="mx-auto w-full max-w-6xl text-center">
            <Link
              href="/workspace"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 rounded-full bg-primary px-6 py-3 text-base font-semibold text-primary-foreground transition-colors hover:bg-primary/90"
            >
              <ArrowRight className="h-4 w-4" />
              Get Started with OmniRank
            </Link>
          </div>
        </div>

        <footer className="border-t border-border/55 bg-card/70 py-8 backdrop-blur-xl md:py-9">
          <div className="mx-auto flex w-full max-w-6xl flex-col items-center gap-3 px-4 text-center text-sm text-muted-foreground md:px-6">
            <div className="flex items-center justify-center">
              <img
                src="https://upload.wikimedia.org/wikipedia/commons/7/7c/Shield_of_the_University_of_Pennsylvania.svg"
                alt="UPenn"
                className="mr-4 h-6 w-auto shrink-0 sm:mr-5 sm:h-7"
              />
              <Link href="/" target="_blank" rel="noopener noreferrer" className="text-2xl font-semibold tracking-wide text-foreground">
                Omni<span className="text-primary">Rank</span>
              </Link>
              <img
                src="https://static.cdnlogo.com/logos/w/18/washington-university-in-st-louis.svg"
                alt="WUSTL"
                className="ml-1 h-8 w-auto shrink-0 sm:ml-2 sm:h-9"
              />
            </div>
            <p>© 2026 Jin Lab and Yu Lab. All rights reserved.</p>
          </div>
        </footer>
      </div>
    </main>
  );
}
