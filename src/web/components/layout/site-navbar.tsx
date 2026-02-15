"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { ArrowRight, BookOpen, Trophy } from "lucide-react";
import { cn } from "@/lib/utils";

interface SiteNavbarProps {
  id?: string;
}

const SCROLL_TO_HOW_TO_USE_KEY = "omnirank:scroll-to-how-to-use";

export function SiteNavbar({ id }: SiteNavbarProps) {
  const [isNavFloating, setIsNavFloating] = useState(false);
  const pathname = usePathname();
  const router = useRouter();
  const isHomePage = pathname === "/";

  const handleHowToUseClick = useCallback(() => {
    if (isHomePage) {
      const howToUseSection = document.getElementById("how-to-use");
      if (!howToUseSection) return;

      howToUseSection.scrollIntoView({ behavior: "smooth", block: "start" });
      const nextUrl = `${window.location.pathname}${window.location.search}`;
      window.history.replaceState(null, "", nextUrl);
      return;
    }

    window.sessionStorage.setItem(SCROLL_TO_HOW_TO_USE_KEY, "1");
    router.push("/");
  }, [isHomePage, router]);

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
    <div
      id={id}
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
          <Link href="/" target="_blank" rel="noopener noreferrer" className="text-xl font-semibold tracking-wide text-foreground">
            Omni<span className="text-primary">Rank</span>
          </Link>

          {isNavFloating ? (
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleHowToUseClick}
                className={cn(
                  "inline-flex items-center gap-1.5 rounded-full border px-3.5 py-2 text-sm font-semibold transition-all duration-300",
                  "border-primary/65 bg-card/80 text-primary hover:bg-primary/10 hover:border-primary",
                )}
              >
                <BookOpen className="h-4 w-4 text-primary" />
                Usage Guide
              </button>

              <Link
                href="/workspace"
                target="_blank"
                rel="noopener noreferrer"
                className={cn(
                  "inline-flex items-center gap-1 rounded-full text-sm font-semibold transition-colors",
                  "bg-primary px-3.5 py-2 text-primary-foreground hover:bg-primary/90",
                )}
              >
                Start Ranking
                <ArrowRight className="h-4 w-4" />
              </Link>

              <Link
                href="/leaderboard"
                target="_blank"
                rel="noopener noreferrer"
                className={cn(
                  "inline-flex items-center gap-1.5 rounded-full border px-3.5 py-2 text-sm font-semibold transition-all duration-300",
                  "border-primary/65 bg-card/80 text-primary hover:bg-primary/10 hover:border-primary",
                )}
              >
                <Trophy className="h-4 w-4 text-primary" />
                LLM Leaderboard
              </Link>
            </div>
          ) : null}
        </div>
      </nav>
    </div>
  );
}
