import type { Metadata } from "next";
import { Geist, Geist_Mono, JetBrains_Mono, Outfit } from "next/font/google";
import { Toaster } from "@/components/ui/sonner";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-jetbrains",
  subsets: ["latin"],
});

const outfit = Outfit({
  variable: "--font-outfit",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "OmniRank | Spectral Ranking Inference",
  description:
    "LLM Agent Platform for Statistically Rigorous Ranking Inference from Arbitrary Multiway Comparisons",
  keywords: ["ranking", "spectral", "LLM", "statistics", "inference"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${geistSans.variable} ${geistMono.variable} ${jetbrainsMono.variable} ${outfit.variable} antialiased min-h-screen bg-background`}
      >
        {children}
        <Toaster />
      </body>
    </html>
  );
}
