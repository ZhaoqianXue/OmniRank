"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import {
  ArrowRight,
  BarChart3,
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
    title: "Upload Your Data",
    description:
      "Upload your comparison data directly into OmniRank. The platform supports CSV files with various comparison formats including pointwise scores, pairwise outcomes, and multiway rankings.",
  },
  {
    title: "AI-Powered Schema Inference",
    description:
      "Let our intelligent agent automatically infer your data semantics. OmniRank detects whether higher values are better, identifies ranking items, and extracts stratification dimensions for segmented analysis.",
  },
  {
    title: "Spectral Ranking Analysis",
    description:
      "Execute statistically rigorous spectral ranking with automatic bootstrap confidence intervals. The platform validates data quality, checks graph connectivity, and applies minimax-optimal estimation methods.",
  },
  {
    title: "Review & Export",
    description:
      "Review the AI-generated ranking report with interactive visualizations, confidence intervals, and uncertainty quantification. Export publication-ready figures and share your analysis with collaborators.",
  },
];

const keyFeatures: KeyFeature[] = [
  {
    title: "Natural Language Interface",
    description:
      "The first natural language interface for spectral ranking inference. Democratizing access to minimax-optimal ranking methods for practitioners without statistical programming expertise.",
    icon: MessageSquareText,
  },
  {
    title: "Semantic Schema Inference",
    description:
      "AI automatically infers comparison data semantics including preference direction, ranking items, and stratification dimensions. Reducing user configuration burden while maintaining statistical rigor.",
    icon: Brain,
  },
  {
    title: "Statistical Rigor",
    description:
      "Integrated uncertainty quantification with automatic bootstrap confidence intervals based on Gaussian multiplier bootstrap method. Enabling statistically grounded decisions without manual implementation.",
    icon: BarChart3,
  },
  {
    title: "Flexible Data Formats",
    description:
      "Supports all common comparison data formats: pointwise scores, pairwise outcomes, and multiway rankings. Automatic format detection and preprocessing for seamless analysis.",
    icon: FileSpreadsheet,
  },
];

export default function LandingPage() {
  return (
    <main className="relative min-h-screen overflow-x-hidden text-foreground">
      <HeroScene className="fixed inset-0 -z-30 opacity-90" />
      <div className="pointer-events-none fixed inset-0 -z-20 bg-[radial-gradient(circle_at_20%_18%,rgba(152,132,229,0.24),transparent_40%),radial-gradient(circle_at_80%_14%,rgba(197,184,246,0.16),transparent_46%),radial-gradient(circle_at_50%_84%,rgba(16,25,46,0.88),transparent_62%)]" />
      <div className="pointer-events-none fixed inset-0 -z-10 bg-gradient-to-b from-background/86 via-background/94 to-background" />
      <SiteNavbar />

      <div className="relative z-10 text-foreground">
        <section className="min-h-screen px-4 pb-24 pt-44 md:px-6 md:pt-52">
          <div className="mx-auto flex w-full max-w-7xl flex-col items-center text-center">
            <motion.div
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7, ease: "easeOut" }}
              className="w-full translate-y-8 md:translate-y-10"
            >
              <h1 className="text-balance text-5xl font-bold leading-tight md:text-7xl">
                Omni<span className="text-primary">Rank</span>
              </h1>

              <p className="mx-auto mt-4 w-fit max-w-none whitespace-nowrap text-center font-[family-name:var(--font-space-mono)] text-sm text-muted-foreground md:text-base">
                An agentic AI platform for{" "}
                <a
                  href="https://doi.org/10.1287/opre.2023.0439"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary underline underline-offset-2 hover:text-primary/80"
                >
                  Spectral Ranking
                </a>{" "}
                Analysis Developed by{" "}
                <a
                  href="https://jin93.github.io/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary underline underline-offset-2 hover:text-primary/80"
                >
                  Jin Jin Lab
                </a>
              </p>

              <div className="mx-auto mt-8 grid w-full max-w-3xl grid-cols-1 gap-3 sm:grid-cols-2">
                <Link
                  href="/leaderboard"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex min-h-12 items-center justify-center gap-2 rounded-full border border-primary/65 bg-card/80 px-5 py-3 text-base font-semibold text-primary transition-all duration-300 hover:bg-primary/10 hover:border-primary"
                >
                  <Trophy className="h-4 w-4 text-primary" />
                  LLM Leaderboard
                </Link>
                <Link
                  href="/workspace"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex min-h-12 items-center justify-center gap-2 rounded-full bg-primary px-5 py-3 text-base font-semibold text-primary-foreground transition-colors hover:bg-primary/90"
                >
                  <ArrowRight className="h-4 w-4" />
                  Start Ranking
                </Link>
              </div>

              <div className="mx-auto mt-4 flex w-full max-w-3xl flex-wrap justify-center gap-2">
                <a
                  href="https://arxiv.org/html/2308.02918"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 rounded-full border border-border/60 bg-card/70 px-3 py-1.5 text-sm text-muted-foreground transition-colors hover:text-foreground"
                >
                  <FileText className="h-3.5 w-3.5" />
                  Preprint
                </a>
                <a
                  href="https://github.com"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 rounded-full border border-border/60 bg-card/70 px-3 py-1.5 text-sm text-muted-foreground transition-colors hover:text-foreground"
                >
                  <Github className="h-3.5 w-3.5" />
                  GitHub
                </a>
              </div>
            </motion.div>
          </div>
        </section>

        <section id="how-to-use" className="min-h-screen px-4 py-28 md:px-6">
          <div className="mx-auto w-full max-w-6xl">
            <h2 className="mb-12 text-center text-3xl font-bold md:mb-16 md:text-4xl">How to Use OmniRank</h2>

            <div className="grid items-start gap-12 lg:grid-cols-2 lg:gap-16">
              <div className="space-y-8">
                {workflowSteps.map((step, index) => (
                  <div key={step.title} className="flex gap-5">
                    <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-primary text-sm font-bold text-primary-foreground">
                      {index + 1}
                    </div>
                    <div className="pt-0.5">
                      <h3 className="text-lg font-semibold text-foreground">{step.title}</h3>
                      <p className="mt-2 text-sm leading-relaxed text-muted-foreground md:text-base">{step.description}</p>
                    </div>
                  </div>
                ))}
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
                <p className="mt-4 text-center text-sm text-muted-foreground">
                  Watch this tutorial to learn how to use OmniRank effectively for your data analysis needs.
                </p>
              </div>
            </div>
          </div>
        </section>

        <section id="why-omnirank" className="min-h-screen px-4 py-28 md:px-6">
          <div className="mx-auto w-full max-w-6xl">
            <motion.div
              initial={{ opacity: 0, y: 18 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.3 }}
              transition={{ duration: 0.5, ease: "easeOut" }}
              className="text-center"
            >
              <h2 className="mb-4 text-center text-3xl font-bold md:text-4xl">Why OmniRank</h2>
              <p className="mx-auto mb-10 max-w-2xl text-center text-muted-foreground md:mb-12">
                A bridge that democratizes spectral ranking inference for domain experts without requiring linear algebra expertise or R programming skills.
              </p>
            </motion.div>

            <div className="rounded-3xl border border-border/55 bg-card/72 p-4 shadow-2xl shadow-black/20 backdrop-blur-xl md:p-6">
              <div className="grid gap-4 md:grid-cols-2">
                {keyFeatures.map((feature, index) => {
                  const Icon = feature.icon;
                  return (
                    <motion.article
                      key={feature.title}
                      initial={{ opacity: 0, y: 18 }}
                      whileInView={{ opacity: 1, y: 0 }}
                      viewport={{ once: true, amount: 0.2 }}
                      transition={{ duration: 0.5, delay: index * 0.08, ease: "easeOut" }}
                      className="group rounded-2xl border border-border/55 bg-background/72 p-5 transition-all duration-300 hover:-translate-y-0.5 hover:border-primary/45 hover:bg-background/86"
                    >
                      <div className="mb-4 inline-flex h-11 w-11 items-center justify-center rounded-xl border border-primary/30 bg-primary/15 text-primary transition-colors group-hover:bg-primary/25">
                        <Icon className="h-5 w-5" />
                      </div>
                      <h3 className="text-xl font-semibold text-foreground">{feature.title}</h3>
                      <p className="mt-3 text-sm leading-relaxed text-muted-foreground md:text-base">
                        {feature.description}
                      </p>
                    </motion.article>
                  );
                })}
              </div>
            </div>

            <div className="mt-12 text-center">
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
        </section>

        <footer className="border-t border-border/55 bg-card/70 py-10 backdrop-blur-xl">
          <div className="mx-auto flex w-full max-w-6xl flex-col items-center gap-3 px-4 text-center text-sm text-muted-foreground md:px-6">
            <div className="flex items-center justify-center">
              <img
                src="https://upload.wikimedia.org/wikipedia/commons/7/7c/Shield_of_the_University_of_Pennsylvania.svg"
                alt="UPenn"
                className="mr-4 h-6 w-auto shrink-0 sm:mr-5 sm:h-7"
              />
              <Link href="/" target="_blank" rel="noopener noreferrer" className="text-xl font-semibold tracking-wide text-foreground">
                Omni<span className="text-primary">Rank</span>
              </Link>
              <img
                src="https://static.cdnlogo.com/logos/w/18/washington-university-in-st-louis.svg"
                alt="WUSTL"
                className="ml-1 h-8 w-auto shrink-0 sm:ml-2 sm:h-9"
              />
            </div>
            <p>© 2026 Jin Jin Lab. All rights reserved.</p>
          </div>
        </footer>
      </div>
    </main>
  );
}
