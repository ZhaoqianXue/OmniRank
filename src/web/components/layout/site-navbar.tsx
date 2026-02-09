"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { ArrowRight, Trophy } from "lucide-react";
import { cn } from "@/lib/utils";

interface SiteNavbarProps {
  id?: string;
}

export function SiteNavbar({ id }: SiteNavbarProps) {
  const [isNavFloating, setIsNavFloating] = useState(false);
  const pathname = usePathname();
  const isLeaderboardPage = pathname === "/leaderboard";

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

          <div className="flex items-center gap-2">
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
              Start Ranking
              <ArrowRight className="h-4 w-4" />
            </Link>
          </div>
        </div>
      </nav>
    </div>
  );
}
