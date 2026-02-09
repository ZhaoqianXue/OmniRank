import { NextRequest, NextResponse } from "next/server";
import { loadArenaCombinationMethods, loadHuggingFaceCombinationMethods } from "@/lib/leaderboard-data";

export const runtime = "nodejs";

function parseKeys(raw: string | null): string[] {
  if (!raw) {
    return [];
  }

  return raw
    .split(",")
    .map((value) => value.trim())
    .filter((value) => value.length > 0);
}

export async function GET(request: NextRequest) {
  const mode = request.nextUrl.searchParams.get("mode");
  const keys = parseKeys(request.nextUrl.searchParams.get("keys"));

  if (mode !== "arena" && mode !== "huggingface") {
    return NextResponse.json(
      { error: "Unsupported mode. Use 'arena' or 'huggingface'." },
      { status: 400 },
    );
  }

  try {
    const methods = mode === "arena"
      ? loadArenaCombinationMethods(keys)
      : loadHuggingFaceCombinationMethods(keys);

    return NextResponse.json({ methods }, { status: 200 });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to load combination results.";
    return NextResponse.json({ error: message }, { status: 400 });
  }
}

