import { loadLeaderboardPageData } from "@/lib/leaderboard-data";
import LeaderboardClient from "@/app/leaderboard/leaderboard-client";

export const runtime = "nodejs";

export default function LeaderboardPage() {
  const initialData = loadLeaderboardPageData();
  return <LeaderboardClient initialData={initialData} />;
}

