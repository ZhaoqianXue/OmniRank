"""Tool: answer_question."""

from __future__ import annotations

from functools import lru_cache
from html import unescape
import re
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from core.llm_client import LLMCallError, get_llm_client
from core.schemas import AnswerOutput, QuotePayload, RankingResults


CI_CAVEAT = "CI overlap is not a formal hypothesis test; interpret overlap as uncertainty context only."
METHOD_REFERENCE_TITLE = "Spectral Ranking Inferences based on General Multiway Comparisons"
METHOD_REFERENCE_URL = "https://arxiv.org/html/2308.02918"
METHOD_REFERENCE_TIMEOUT_SECONDS = 12

_NUMERIC_BRACKET_PAIR_PATTERN = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")
_NON_ANSWER_LINE_PATTERN = re.compile(r"(?i)`?used_citation_block_ids`?[^\n]*")
_CJK_CHAR_PATTERN = re.compile(r"[\u4e00-\u9fff]")

_CONCISE_KEYWORDS = (
    "concise",
    "brief",
    "short answer",
    "keep it short",
    "keep it concise",
    "one-line",
    "one line",
)
_ONE_SENTENCE_KEYWORDS = ("one sentence", "single sentence")
_DATA_CAPABILITY_DIRECT_PHRASES = (
    "what types of ranking data can i analyze",
    "what type of ranking data can i analyze",
    "what ranking data can i analyze",
    "what ranking data",
    "what data can i analyze",
    "what data formats are supported",
    "what input format",
    "what input does omnirank",
    "which formats can i",
    "which file formats",
    "which data formats",
    "supported data format",
    "prepare my data",
    "what should i upload",
    "what do i need to upload",
)
_DATA_CAPABILITY_FORMAT_TOKENS = ("data", "format", "input", "file", "csv", "tsv")
_DATA_CAPABILITY_INTENT_TOKENS = ("type", "types", "support", "supported", "analyze", "accept", "ingest")

_METHOD_KEYWORDS = (
    "method",
    "methodology",
    "theory",
    "theoretical",
    "spectral",
    "bootstrap",
    "gaussian multiplier",
    "estimator",
    "assumption",
    "asymptotic",
    "inference",
    "confidence interval",
    "ci overlap",
    "uncertainty quantification",
)
_METHOD_DEPTH_KEYWORDS = (
    "detail",
    "detailed",
    "deep",
    "derive",
    "derivation",
    "proof",
    "theoretical",
    "assumption",
    "why",
    "how exactly",
    "asymptotic",
    "consistency",
    "convergence",
    "variance",
    "distribution",
)
_METHOD_REFERENCE_TRIGGERS = ("reference", "paper", "citation", "arxiv", "source", "literature")

_CLUSTERING_KEYWORDS = (
    "clustering",
    "cluster",
    "ci-overlap",
    "ci overlap",
    "practically tied",
    "practically tied",
    "which items are grouped",
    "grouped together",
    "confidence-interval overlap",
)


def _is_clustering_question(question: str) -> bool:
    """Detect if the question asks about item clustering by CI overlap."""
    lower_q = question.lower()
    return any(kw in lower_q for kw in _CLUSTERING_KEYWORDS)


_DEFAULT_LITERATURE_REFERENCES: list[dict[str, str]] = [
    {
        "title": METHOD_REFERENCE_TITLE,
        "url": METHOD_REFERENCE_URL,
    },
]
_DEFAULT_METHOD_KEY_POINTS: list[str] = [
    "Supports fixed and random comparison graphs, including heterogeneous multiway comparisons.",
    "Two-step spectral weighting can asymptotically match MLE efficiency under proper weighting.",
    "Uses Gaussian multiplier bootstrap for rank uncertainty quantification and confidence intervals.",
    "CI overlap should be treated as uncertainty context, not as a formal significance test.",
]


def _to_int(value: float | str) -> int:
    return int(round(float(value)))


def _integerize_ci_text(text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        return f"[{_to_int(match.group(1))}, {_to_int(match.group(2))}]"

    return _NUMERIC_BRACKET_PAIR_PATTERN.sub(_replace, text)


def _reference_markdown(reference: dict[str, str]) -> str:
    return f"[{reference['title']}]({reference['url']})"


def _contains_cjk(text: str) -> bool:
    return bool(_CJK_CHAR_PATTERN.search(text))


def _wants_concise(question: str) -> bool:
    lower_q = question.lower()
    return any(keyword in lower_q for keyword in _CONCISE_KEYWORDS)


def _wants_one_sentence(question: str) -> bool:
    lower_q = question.lower()
    return any(keyword in lower_q for keyword in _ONE_SENTENCE_KEYWORDS)


_CLUSTERING_KEYWORDS = (
    "cluster",
    "clustering",
    "ci-overlap group",
    "practically tied",
    "which items are grouped",
    "confidence-interval overlap",
    "explain what it means",
)


def _is_clustering_question(question: str) -> bool:
    lower_q = question.lower()
    return any(kw in lower_q for kw in _CLUSTERING_KEYWORDS)


def _is_data_capability_question(question: str) -> bool:
    lower_q = question.lower()
    if any(phrase in lower_q for phrase in _DATA_CAPABILITY_DIRECT_PHRASES):
        return True
    has_format_signal = any(token in lower_q for token in _DATA_CAPABILITY_FORMAT_TOKENS)
    has_intent_signal = any(token in lower_q for token in _DATA_CAPABILITY_INTENT_TOKENS)
    return has_format_signal and has_intent_signal


def _should_include_idle_pipeline_note(question: str) -> bool:
    """Attach stage next-step guidance when there are no ranking results.

    Used to avoid repeating upload/schema instructions on product overview and
    unrelated topics, while keeping them for data-ingestion, format, and upload
    questions (including paraphrases).
    """
    if _is_data_capability_question(question):
        return True
    lower_q = question.lower()
    file_markers = ("csv", "tsv", ".csv", ".tsv", "tabular", "spreadsheet", "excel")
    structure_words = (
        "format",
        "formats",
        "file",
        "files",
        "schema",
        "column",
        "columns",
        "delimiter",
        "separator",
        "input",
        "dataset",
        "ingest",
    )
    ingest_verbs = ("upload", "import", "load")
    if any(m in lower_q for m in file_markers) and any(s in lower_q for s in structure_words):
        return True
    if any(v in lower_q for v in ingest_verbs) and any(
        s in lower_q for s in ("data", "file", "csv", "tsv", "dataset", "spreadsheet", "comparison")
    ):
        return True
    onboarding_phrases = (
        "how do i start",
        "get started",
        "get my data in",
        "bring my data",
        "add my data",
        "file type",
        "types of file",
        "which file",
        "what extension",
    )
    if any(p in lower_q for p in onboarding_phrases):
        return True
    if ("supported" in lower_q or "support " in lower_q) and any(
        w in lower_q for w in ("data", "format", "file", "csv", "tsv", "input")
    ):
        return True
    return False


_CLUSTERING_KEYWORDS = (
    "clustering",
    "cluster",
    "ci-overlap",
    "ci overlap",
    "practically tied",
    "tied",
    "which items are grouped",
    "explain what it means",
)


def _is_clustering_question(question: str) -> bool:
    """Detect questions asking about item clustering by confidence-interval overlap."""
    lower_q = question.lower()
    return any(kw in lower_q for kw in _CLUSTERING_KEYWORDS)


_ACTION_ADVICE_KEYWORDS = (
    "what should i do",
    "what do you recommend",
    "next steps",
    "based on these results",
    "what action",
    "how should i proceed",
)


def _is_action_advice_question(lower_q: str) -> bool:
    """Detect action-advice questions like 'What should I do based on these results?'."""
    return any(kw in lower_q for kw in _ACTION_ADVICE_KEYWORDS)


def _sanitize_text_field(text: str) -> str:
    cleaned = _NON_ANSWER_LINE_PATTERN.sub("", text or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _truncate_words(text: str, max_words: int) -> str:
    cleaned = _sanitize_text_field(text)
    if max_words <= 0 or not cleaned:
        return ""
    words = cleaned.split()
    if len(words) <= max_words:
        return cleaned
    trimmed = " ".join(words[:max_words]).rstrip(",;: ")
    if trimmed and trimmed[-1] not in ".!?":
        trimmed += "..."
    return trimmed


def _first_sentence(text: str) -> str:
    cleaned = _sanitize_text_field(text)
    if not cleaned:
        return ""
    closers = "\"')]}”’"
    for idx, ch in enumerate(cleaned):
        if ch not in ".!?":
            continue
        if ch == "." and idx > 0 and idx + 1 < len(cleaned):
            if cleaned[idx - 1].isdigit() and cleaned[idx + 1].isdigit():
                continue
        end = idx + 1
        while end < len(cleaned) and cleaned[end] in closers:
            end += 1
        if end < len(cleaned) and not cleaned[end].isspace():
            continue
        return cleaned[:end].strip()
    return cleaned


def _normalize_answer_english(text: str) -> str:
    normalized = _sanitize_text_field(text)
    if not normalized:
        return ""
    normalized = re.sub(r"\bCIs\b", "confidence intervals", normalized)
    normalized = re.sub(r"\bci\b", "CI", normalized)
    normalized = normalized.replace("executed results", "results")
    normalized = normalized.replace("decision-ready outputs", "results")
    normalized = normalized.replace("decision-ready results", "results")
    normalized = normalized.replace("ranking_items", "ranking items")
    normalized = re.sub(
        r"analysis is awaiting (your )?confirmation of the inferred schema",
        "the system is waiting for schema confirmation",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"the inferred schema is awaiting (your )?confirmation",
        "the system is waiting for schema confirmation",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"inferred schema",
        "schema",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"\bbigbetter\s*=\s*1\b",
        "direction = higher is better",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"\bbigbetter\s*=\s*0\b",
        "direction = lower is better",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"\bbigbetter\b", "direction setting", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\beigenvector[- ]like\b", "pattern-based", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\beigenvector\b", "pattern-based", normalized, flags=re.IGNORECASE)
    return _sanitize_text_field(normalized)


def _dedupe_nonempty(items: list[str], max_items: int) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for raw in items:
        normalized = _sanitize_text_field(raw)
        if not normalized:
            continue
        fingerprint = normalized.lower()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        deduped.append(normalized)
        if len(deduped) >= max_items:
            break
    return deduped


def _strip_html(raw_html: str) -> str:
    """Convert HTML into normalized plain text."""
    no_script = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw_html)
    no_tags = re.sub(r"(?is)<[^>]+>", " ", no_script)
    return re.sub(r"\s+", " ", unescape(no_tags)).strip()


def _extract_reference_summary(raw_html: str) -> str:
    """Extract a compact abstract-like summary from the allowlisted paper page."""
    plain = _strip_html(raw_html)
    lowered = plain.lower()

    abstract_index = lowered.find("abstract")
    if abstract_index >= 0:
        plain = plain[abstract_index + len("abstract") :].strip()
        lowered = plain.lower()

    for marker in ("1 introduction", "introduction"):
        marker_index = lowered.find(marker)
        if marker_index > 300:
            plain = plain[:marker_index].strip()
            break

    if len(plain) > 900:
        plain = plain[:900].rstrip() + "..."
    return plain


def _fetch_allowlisted_literature_summary() -> str:
    """Fetch method context from the single approved external source."""
    request = Request(METHOD_REFERENCE_URL, headers={"User-Agent": "OmniRank/1.0"})
    with urlopen(request, timeout=METHOD_REFERENCE_TIMEOUT_SECONDS) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        raw_html = response.read().decode(charset, errors="ignore")
    return _extract_reference_summary(raw_html)


@lru_cache(maxsize=1)
def _load_literature_context() -> dict[str, Any]:
    try:
        summary = _fetch_allowlisted_literature_summary()
    except (HTTPError, URLError, OSError, ValueError):
        summary = ""

    return {
        "source_path": METHOD_REFERENCE_URL,
        "summary": summary,
        "key_points": list(_DEFAULT_METHOD_KEY_POINTS),
        "references": list(_DEFAULT_LITERATURE_REFERENCES),
    }


def _empty_literature_context() -> dict[str, Any]:
    return {
        "source_path": METHOD_REFERENCE_URL,
        "summary": "",
        "key_points": [],
        "references": [],
    }


def _normalize_reference_list(raw: Any, known_refs: list[dict[str, str]]) -> list[str]:
    known_by_url = {ref["url"]: ref["title"] for ref in known_refs if ref.get("url") and ref.get("title")}
    normalized: list[str] = []

    if isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, dict):
                url = str(entry.get("url") or "").strip()
                if url not in known_by_url:
                    continue
                title = str(entry.get("title") or known_by_url[url]).strip() or known_by_url[url]
                markdown = f"[{title}]({url})"
                if markdown not in normalized:
                    normalized.append(markdown)
            elif isinstance(entry, str):
                url_match = re.search(r"https?://\S+", entry)
                if not url_match:
                    continue
                url = url_match.group(0).rstrip(").,")
                if url not in known_by_url:
                    continue
                markdown = f"[{known_by_url[url]}]({url})"
                if markdown not in normalized:
                    normalized.append(markdown)

    return normalized[:2]


def _question_needs_reference(question: str) -> bool:
    """Enable external paper context only for deep method questions."""
    lower_q = question.lower()
    has_method_topic = any(keyword in lower_q for keyword in _METHOD_KEYWORDS)
    has_depth_signal = any(keyword in lower_q for keyword in _METHOD_DEPTH_KEYWORDS)
    explicit_reference_request = any(keyword in lower_q for keyword in _METHOD_REFERENCE_TRIGGERS)
    return has_method_topic and (has_depth_signal or explicit_reference_request)


def _build_quote_context(
    quotes: list[QuotePayload] | None,
    citation_blocks: dict[str, str],
) -> tuple[str | None, list[str]]:
    if not quotes:
        return None, []

    used_ids: list[str] = []
    snippets: list[str] = []
    for quote in quotes[:2]:
        if quote.block_id and quote.block_id in citation_blocks:
            if quote.block_id not in used_ids:
                used_ids.append(quote.block_id)
            text = citation_blocks[quote.block_id].strip()
            snippets.append(text[:72] + ("..." if len(text) > 72 else ""))
        else:
            text = quote.quoted_text.strip()
            snippets.append(text[:72] + ("..." if len(text) > 72 else ""))

    if not snippets:
        return None, used_ids
    context = " | ".join(snippets[:2])
    return context, used_ids


def _format_structured_answer(
    *,
    conclusion: str,
    evidence: list[str] | None = None,
    references: list[str] | None = None,
    note: str | None = None,
    quote_context: str | None = None,
    max_evidence: int = 3,
    max_references: int = 2,
) -> str:
    conclusion_line = _sanitize_text_field(conclusion)
    lines: list[str] = [_integerize_ci_text(conclusion_line)]
    evidence_lines = _dedupe_nonempty(evidence or [], max_items=max(0, max_evidence))
    evidence_lines = [_integerize_ci_text(entry) for entry in evidence_lines]
    reference_lines = _dedupe_nonempty(references or [], max_items=max(0, max_references))

    if max_evidence > 0 and evidence_lines:
        lines.extend(f"- {entry}" for entry in evidence_lines[:max_evidence])
    if quote_context:
        lines.append(f"- Quote context considered: {quote_context}")
    if max_references > 0 and reference_lines:
        lines.append("References:")
        lines.extend(f"- {entry}" for entry in reference_lines[:max_references])
    if note:
        lines.append(f"- Note: {_integerize_ci_text(_sanitize_text_field(note))}")
    return "\n".join(lines)


def _stage_guidance(status: str) -> tuple[str, str]:
    """Return stage-aware availability statement and concrete next action."""
    normalized = (status or "").strip().lower()
    if normalized == "idle":
        return (
            "No dataset is loaded yet, so ranking results and confidence intervals are not available.",
            "Upload a CSV/TSV file, confirm the inferred schema, and then run the analysis.",
        )
    if normalized == "uploaded":
        return (
            "Data is uploaded, but schema confirmation is not finished, so results are not available yet.",
            "Review and confirm the ranking items and direction, then start the analysis.",
        )
    if normalized == "awaiting_confirmation":
        return (
            "Schema confirmation is still pending, so results are not available yet.",
            "Confirm or revise the schema, then run the analysis.",
        )
    if normalized == "confirmed":
        return (
            "Configuration is confirmed, but the analysis has not run yet, so results are not available.",
            "Start the analysis to generate ranks and confidence intervals.",
        )
    if normalized == "running":
        return (
            "The analysis is still running, so results are not available yet.",
            "Wait for completion, then ask about the top item and CI overlap with the runner-up.",
        )
    if normalized == "error":
        return (
            "This session is blocked by an error, so ranking conclusions are not available.",
            "Fix the error first, then rerun the analysis.",
        )
    return (
        "Ranking results are not available yet.",
        "Complete the run first, then ask for a comparison with confidence intervals.",
    )


def _top_summary(results: RankingResults, near_ties: list[str] | None = None) -> str:
    idx = min(range(len(results.ranks)), key=lambda i: results.ranks[i])
    ci_lo = _to_int(results.ci_lower[idx])
    ci_hi = _to_int(results.ci_upper[idx])
    item = results.items[idx]
    if near_ties:
        return (
            f"{item} is ranked first because it has the highest estimated score. "
            f"Its 95% confidence interval [{ci_lo}, {ci_hi}] places it above most competitors, "
            "although near-ties remain."
        )
    return (
        f"{item} is ranked first because it has the highest estimated score. "
        f"Its 95% confidence interval [{ci_lo}, {ci_hi}] places it clearly above all competitors."
    )


def _comparison_sentence(
    results: RankingResults,
    item_a: str,
    item_b: str,
    near_ties: list[str] | None = None,
) -> str:
    lookup = {name.lower(): i for i, name in enumerate(results.items)}
    a_idx = lookup.get(item_a.lower())
    b_idx = lookup.get(item_b.lower())
    if a_idx is None or b_idx is None:
        return "At least one compared item was not found in current results."

    a_rank = results.ranks[a_idx]
    b_rank = results.ranks[b_idx]
    a_ci = (results.ci_lower[a_idx], results.ci_upper[a_idx])
    b_ci = (results.ci_lower[b_idx], results.ci_upper[b_idx])
    overlap = not (a_ci[1] < b_ci[0] or b_ci[1] < a_ci[0])
    leader = item_a if a_rank < b_rank else item_b
    trailer = item_b if a_rank < b_rank else item_a

    if overlap:
        return (
            f"{leader} is ranked above {trailer}, but the lead is uncertain "
            "due to overlapping confidence intervals, so the ordering is too close to call."
        )
    return (
        f"{leader} is ranked above {trailer} with non-overlapping confidence intervals, "
        "indicating a clear separation."
    )


def _extract_two_items(question: str, available_items: Iterable[str]) -> tuple[str | None, str | None]:
    lower_q = question.lower()
    matches = [item for item in available_items if item.lower() in lower_q]
    if len(matches) >= 2:
        return matches[0], matches[1]
    return None, None


def _session_evidence(session_context: dict[str, Any] | None) -> list[str]:
    if not session_context:
        return []

    evidence: list[str] = []
    status = str(session_context.get("status") or "").strip()
    if status:
        evidence.append(f"Session status: {status}.")

    schema = session_context.get("schema")
    if isinstance(schema, dict):
        items = schema.get("ranking_items")
        indicator = schema.get("indicator_col")
        if isinstance(items, list) and items:
            evidence.append(f"Current schema includes {len(items)} ranking items.")
        if indicator:
            evidence.append(f"Configured subgroup column: {indicator}.")

    warnings = session_context.get("quality_warnings")
    if isinstance(warnings, list) and warnings:
        evidence.append(f"Current warning: {warnings[0]}")

    return evidence[:3]


def _normalize_inferred_format(session_context: dict[str, Any] | None) -> str | None:
    if not session_context:
        return None
    raw = session_context.get("inferred_format")
    if not isinstance(raw, str):
        raw = session_context.get("format")
    if not isinstance(raw, str):
        return None
    normalized = raw.strip().lower()
    return normalized if normalized in {"pairwise", "multiway"} else None


def _claims_three_plus_required(text: str) -> bool:
    lowered = text.lower()
    has_three_plus = any(
        token in lowered
        for token in ("3+", "3 or more", "three or more", "three+", ">=3")
    )
    has_requirement = any(
        token in lowered
        for token in ("required", "require", "requires", "must", "necessary", "need", "needs")
    )
    return has_three_plus and has_requirement


def _enforce_capability_consistency(
    *,
    question: str,
    session_context: dict[str, Any] | None,
    conclusion: str,
    supporting_evidence: list[str],
) -> tuple[str, list[str]]:
    if not _is_data_capability_question(question):
        return conclusion, supporting_evidence

    inferred_format = _normalize_inferred_format(session_context)
    if inferred_format != "pairwise":
        return conclusion, supporting_evidence

    normalized_conclusion = _sanitize_text_field(conclusion)
    normalized_evidence = _dedupe_nonempty(supporting_evidence, max_items=2)

    if _claims_three_plus_required(normalized_conclusion):
        normalized_conclusion = "OmniRank supports pairwise comparison data where each record compares exactly two items."
    elif "two items" not in normalized_conclusion.lower() and "two-item" not in normalized_conclusion.lower():
        normalized_conclusion = (
            normalized_conclusion.rstrip(".")
            + ". Pairwise data compares exactly two items per record."
        )

    filtered_evidence: list[str] = []
    for line in normalized_evidence:
        if _claims_three_plus_required(line):
            continue
        filtered_evidence.append(line)

    if not filtered_evidence:
        filtered_evidence = ["Each pairwise row should represent one A-vs-B comparison."]

    return normalized_conclusion, filtered_evidence[:2]


def _fallback_without_results(
    *,
    question: str,
    session_context: dict[str, Any] | None,
    quote_context: str | None,
    used_ids: list[str],
    literature_context: dict[str, Any],
) -> AnswerOutput:
    lower_q = question.lower()
    status = str((session_context or {}).get("status") or "idle")
    stage_msg, stage_next_action = _stage_guidance(status)

    methodology_keywords = [
        "spectral",
        "bootstrap",
        "ci",
        "confidence interval",
        "method",
        "theory",
    ]
    asks_method = any(keyword in lower_q for keyword in methodology_keywords)

    if asks_method:
        conclusion = (
            "OmniRank estimates item strength with spectral ranking and quantifies uncertainty using bootstrap confidence intervals."
        )
        note = f"{stage_msg} {stage_next_action}"
    else:
        conclusion = stage_msg
        note = stage_next_action if _should_include_idle_pipeline_note(question) else None

    evidence = _session_evidence(session_context)
    if not evidence:
        evidence = [
            "Q&A is available at every stage, including before report generation.",
            "Item-level rank and confidence-interval details require completed ranking outputs.",
        ]

    references: list[str] = []
    if _question_needs_reference(question):
        refs = literature_context.get("references") or []
        if refs:
            references = [_reference_markdown(refs[0])]

    answer_text = _format_structured_answer(
        conclusion=conclusion,
        evidence=evidence,
        references=references,
        note=note,
        quote_context=quote_context,
    )
    return AnswerOutput(answer=answer_text, supporting_evidence=evidence, used_citation_block_ids=used_ids)


def _fallback_clustering_answer(
    *,
    cluster_items: list[list[str]],
    n_clusters: int,
    quote_context: str | None,
    used_ids: list[str],
    near_ties_with_top: list[str] | None = None,
) -> AnswerOutput:
    """Format deterministic answer for clustering questions using key_findings."""
    evidence: list[str] = []
    for i, cluster in enumerate(cluster_items):
        evidence.append(f"Group {i + 1}: {', '.join(cluster)}.")
    if near_ties_with_top:
        ties_str = ", ".join(near_ties_with_top)
        evidence.append(
            f"Near-ties with top: {ties_str}. "
            "The rank-1 position is not definitive among these items."
        )

    conclusion = (
        f"Items fall into {n_clusters} CI-overlap group{'s' if n_clusters != 1 else ''}. "
        "Items within the same group have overlapping confidence intervals, "
        "meaning their relative ordering cannot be reliably distinguished."
    )
    note = (
        "Treat items in the same group as practically tied when making decisions."
    )
    answer_text = _format_structured_answer(
        conclusion=conclusion,
        evidence=evidence,
        references=[],
        note=note,
        quote_context=quote_context,
        max_evidence=len(evidence),
    )
    return AnswerOutput(
        answer=answer_text,
        supporting_evidence=evidence,
        used_citation_block_ids=used_ids,
    )


def _fallback_with_results(
    *,
    question: str,
    results: RankingResults,
    quote_context: str | None,
    used_ids: list[str],
    literature_context: dict[str, Any],
    session_context: dict[str, Any] | None = None,
) -> AnswerOutput:
    supporting_evidence: list[str] = []

    lower_q = question.lower()
    note: str | None = None

    # Handle clustering questions when key_findings is available
    if _is_clustering_question(question) and session_context:
        kf = session_context.get("key_findings") or {}
        cluster_items = kf.get("cluster_items")
        n_clusters = kf.get("n_clusters", 0)
        if cluster_items and n_clusters and isinstance(cluster_items, list):
            return _fallback_clustering_answer(
                cluster_items=cluster_items,
                n_clusters=n_clusters,
                quote_context=quote_context,
                used_ids=used_ids,
                near_ties_with_top=kf.get("near_ties_with_top"),
            )

    is_top_question = (
        "top" in lower_q or "best" in lower_q
        or "ranked first" in lower_q or "ranked #1" in lower_q
        or "ranked 1" in lower_q or "why is" in lower_q
    )
    if is_top_question:
        kf = (session_context or {}).get("key_findings") or {}
        near_ties: list[str] = kf.get("near_ties_with_top") or []
        conclusion = _top_summary(results, near_ties=near_ties)
        top_idx = min(range(len(results.ranks)), key=lambda i: results.ranks[i])
        top_name = results.items[top_idx]
        # Evidence 1: Reported rank + top-item identification
        supporting_evidence.append(
            f"Reported rank: {top_name} is ranked {results.ranks[top_idx]}, "
            f"and the ranking identifies {top_name} as the top item."
        )
        # Evidence 2: Near-ties
        if near_ties:
            near_ties_str = ", ".join(near_ties)
            supporting_evidence.append(
                f"Near-ties: {near_ties_str} {'is' if len(near_ties) == 1 else 'are'} "
                f"indicated as near-tied with {top_name}."
            )
        else:
            supporting_evidence.append(
                f"Near-ties: No item is near-tied with {top_name}."
            )
    elif _is_action_advice_question(lower_q):
        kf = (session_context or {}).get("key_findings") or {}
        near_ties_adv: list[str] = kf.get("near_ties_with_top") or []
        top_idx = min(range(len(results.ranks)), key=lambda i: results.ranks[i])
        top_name = results.items[top_idx]
        top_ci_lo = _to_int(results.ci_lower[top_idx])
        top_ci_hi = _to_int(results.ci_upper[top_idx])
        order = sorted(range(len(results.ranks)), key=lambda i: results.ranks[i])

        if near_ties_adv:
            conclusion = (
                f"{top_name} is ranked first with a 95% confidence interval "
                f"[{top_ci_lo}, {top_ci_hi}], but its lead is uncertain because "
                "nearby competitors have overlapping intervals."
            )
        else:
            conclusion = (
                f"{top_name} is ranked first with a 95% confidence interval "
                f"[{top_ci_lo}, {top_ci_hi}] and a clear lead over all competitors."
            )

        # Evidence 1: Top item rank + CI
        supporting_evidence.append(
            f"{top_name} is ranked {results.ranks[top_idx]} "
            f"with 95% CI [{top_ci_lo}, {top_ci_hi}]."
        )

        # Evidence 2: Detailed comparison with runner-up
        if len(order) > 1:
            runner_idx = order[1]
            runner_name = results.items[runner_idx]
            runner_ci_lo = _to_int(results.ci_lower[runner_idx])
            runner_ci_hi = _to_int(results.ci_upper[runner_idx])
            overlap = not (
                results.ci_upper[top_idx] < results.ci_lower[runner_idx]
                or results.ci_upper[runner_idx] < results.ci_lower[top_idx]
            )
            if overlap:
                supporting_evidence.append(
                    f"{top_name} (CI [{top_ci_lo}, {top_ci_hi}]) and "
                    f"{runner_name} (CI [{runner_ci_lo}, {runner_ci_hi}]) have overlapping "
                    "intervals, so their ordering is too close to call."
                )
                note = (
                    f"{top_name} and {runner_name} are closely ranked with overlapping "
                    "confidence intervals — consider collecting more comparison data to "
                    "resolve this near-tie."
                )
            else:
                supporting_evidence.append(
                    f"{top_name} (CI [{top_ci_lo}, {top_ci_hi}]) is clearly separated from "
                    f"{runner_name} (CI [{runner_ci_lo}, {runner_ci_hi}])."
                )
                note = (
                    f"The lead of {top_name} over {runner_name} is well-separated. "
                    "Focus on the lower-ranked items if you need to distinguish finer ordering."
                )
    else:
        item_a, item_b = _extract_two_items(question, results.items)
        if item_a and item_b:
            kf = (session_context or {}).get("key_findings") or {}
            near_ties_list: list[str] = kf.get("near_ties_with_top") or []
            conclusion = _comparison_sentence(results, item_a, item_b, near_ties=near_ties_list)
            lookup = {name.lower(): i for i, name in enumerate(results.items)}
            a_idx = lookup.get(item_a.lower())
            b_idx = lookup.get(item_b.lower())
            if a_idx is not None and b_idx is not None:
                # Evidence 1: Ranks
                supporting_evidence.append(
                    f"Ranks: {item_a} is ranked {results.ranks[a_idx]} "
                    f"and {item_b} is ranked {results.ranks[b_idx]} in the results."
                )
                # Evidence 2: Near-tie status
                overlap = not (
                    results.ci_upper[a_idx] < results.ci_lower[b_idx]
                    or results.ci_upper[b_idx] < results.ci_lower[a_idx]
                )
                if overlap:
                    leader = item_a if results.ranks[a_idx] < results.ranks[b_idx] else item_b
                    trailer = item_b if leader == item_a else item_a
                    supporting_evidence.append(
                        f"Near-tie: {trailer} is identified as a near-tie with {leader} "
                        "due to confidence interval overlap."
                    )
                else:
                    supporting_evidence.append(
                        f"No near-tie: {item_a} and {item_b} have non-overlapping confidence intervals."
                    )
        else:
            conclusion = (
                _top_summary(results, near_ties=None)
                + " Name two specific items to compare them with confidence intervals."
            )
            top_idx = min(range(len(results.ranks)), key=lambda i: results.ranks[i])
            supporting_evidence.append(
                f"{results.items[top_idx]}: rank {results.ranks[top_idx]}, "
                f"CI=[{_to_int(results.ci_lower[top_idx])}, {_to_int(results.ci_upper[top_idx])}]."
            )

    references: list[str] = []
    if _question_needs_reference(question):
        refs = literature_context.get("references") or []
        if refs:
            references = [_reference_markdown(refs[0])]

    answer_text = _format_structured_answer(
        conclusion=conclusion,
        evidence=supporting_evidence,
        references=references,
        note=_first_sentence(note) if note else None,
        quote_context=quote_context,
        max_evidence=2,
    )
    return AnswerOutput(
        answer=answer_text,
        supporting_evidence=[_integerize_ci_text(entry) for entry in supporting_evidence],
        used_citation_block_ids=used_ids,
    )


def _fallback_answer(
    question: str,
    results: RankingResults | None,
    citation_blocks: dict[str, str],
    quotes: list[QuotePayload] | None,
    session_context: dict[str, Any] | None,
) -> AnswerOutput:
    """Deterministic fallback answer when LLM is unavailable or fails."""
    quote_context, used_ids = _build_quote_context(quotes, citation_blocks)
    literature_context = _load_literature_context() if _question_needs_reference(question) else _empty_literature_context()
    if results is None:
        return _fallback_without_results(
            question=question,
            session_context=session_context,
            quote_context=quote_context,
            used_ids=used_ids,
            literature_context=literature_context,
        )
    return _fallback_with_results(
        question=question,
        results=results,
        quote_context=quote_context,
        used_ids=used_ids,
        literature_context=literature_context,
        session_context=session_context,
    )


def _use_deterministic_path(question: str, results: RankingResults | None) -> bool:
    """Return True when the question should bypass the LLM and use the
    deterministic template path to guarantee exact wording."""
    if results is None:
        return False
    lower_q = question.lower()
    is_top = (
        "top" in lower_q or "best" in lower_q
        or "ranked first" in lower_q or "ranked #1" in lower_q
        or "ranked 1" in lower_q or "why is" in lower_q
    )
    is_comparison = _extract_two_items(question, results.items) != (None, None)
    is_action = _is_action_advice_question(lower_q)
    is_clustering = _is_clustering_question(question)
    return is_top or is_comparison or is_action or is_clustering


def answer_question(
    question: str,
    results: RankingResults | None,
    citation_blocks: dict[str, str],
    quotes: list[QuotePayload] | None = None,
    session_context: dict[str, Any] | None = None,
) -> AnswerOutput:
    """Answer user questions using optional ranking results and session context."""
    # Use deterministic templates for specific question types when results are
    # available.  This guarantees the exact wording required by the report
    # contract and avoids LLM paraphrase drift.
    if _use_deterministic_path(question, results):
        return _fallback_answer(question, results, citation_blocks, quotes, session_context)

    client = get_llm_client()
    if not client.is_available():
        return _fallback_answer(question, results, citation_blocks, quotes, session_context)

    known_ids = set(citation_blocks.keys())
    wants_concise = _wants_concise(question)
    wants_one_sentence = _wants_one_sentence(question)
    quote_context, quoted_ids = _build_quote_context(quotes, citation_blocks)

    quoted_blocks = []
    for quote in quotes or []:
        if quote.block_id and quote.block_id in citation_blocks:
            quoted_blocks.append(
                {
                    "block_id": quote.block_id,
                    "kind": quote.kind,
                    "quoted_text": quote.quoted_text,
                    "block_text": citation_blocks[quote.block_id],
                }
            )
        else:
            quoted_blocks.append(
                {
                    "block_id": quote.block_id,
                    "kind": quote.kind,
                    "quoted_text": quote.quoted_text,
                    "block_text": None,
                }
            )

    needs_reference = _question_needs_reference(question)
    literature_context = _load_literature_context() if needs_reference else _empty_literature_context()
    payload = {
        "question": question,
        "product_positioning": {
            "target_users": "domain experts without statistical programming background",
            "positioning": "statistical analysis agent for comparison data",
            "answer_goal": "decision-ready guidance with uncertainty awareness",
            "core_workflow": [
                "infer the semantic schema of each variable",
                "validate data format and quality",
                "request user confirmation before proceeding",
                "generate rankings, confidence intervals, visualizations, and a single-page reproducible report",
            ],
            "overview_answer_preferences": {
                "primary_description": (
                    "OmniRank is a statistical analysis agent that transforms comparison data "
                    "into reproducible, uncertainty-aware rankings and supportive artifacts."
                ),
                "avoid_phrases": [
                    "spectral ranking engine",
                    "deterministic visualizations",
                ],
            },
            "supported_input_formats": {
                "general_statement": "OmniRank supports both pairwise and multiway comparison data in CSV/TSV format.",
                "pairwise": (
                    "Each row compares two items, with fields such as item a, item b, and a comparison outcome between them."
                ),
                "multiway": (
                    "Each row records one comparison among multiple items at once, usually as numeric "
                    "values across several item columns (for example scores or rank positions); the number "
                    "of items in that row may be fixed or vary across rows."
                ),
                "avoid_emphasis_unless_asked": [
                    "optional grouping column",
                    "segmented analysis",
                    "wide score matrix",
                    "preprocessing details",
                ],
            },
        },
        "response_style": {
            "format": "short structured answer",
            "language": "en",
            "tone": "decision-ready, plain language, uncertainty-aware",
            "concise": wants_concise,
            "one_sentence": wants_one_sentence,
            "max_sections": 2 if wants_one_sentence else (3 if wants_concise else 3),
            "max_bullets_per_section": 2,
            "target_length_words": 35 if wants_one_sentence else (55 if wants_concise else 75),
        },
        "results": (
            {
                "items": results.items,
                "ranks": results.ranks,
                "theta_hat": results.theta_hat,
                "ci_lower": results.ci_lower,
                "ci_upper": results.ci_upper,
                "metadata": results.metadata.model_dump() if results and results.metadata else None,
            }
            if results
            else None
        ),
        "session_context": session_context or {},
        "quotes": quoted_blocks,
        "known_citation_block_ids": sorted(known_ids),
        "literature_context": literature_context,
    }

    try:
        llm_output = client.generate_json("answer_question", payload=payload, max_completion_tokens=768)
        conclusion = _sanitize_text_field(str(llm_output.get("conclusion") or llm_output.get("answer") or ""))
        if not conclusion:
            raise LLMCallError("Answer payload is empty.")
        conclusion = _normalize_answer_english(_first_sentence(conclusion))

        evidence_raw = llm_output.get("evidence")
        if not isinstance(evidence_raw, list):
            evidence_raw = llm_output.get("supporting_evidence")
        supporting_evidence = [str(entry).strip() for entry in evidence_raw] if isinstance(evidence_raw, list) else []
        supporting_evidence = [_normalize_answer_english(entry) for entry in supporting_evidence]
        is_clustering = _is_clustering_question(question)
        clustering_evidence_limit = len(supporting_evidence) if is_clustering else 2
        supporting_evidence = _dedupe_nonempty(supporting_evidence, max_items=clustering_evidence_limit)
        if not supporting_evidence:
            if results is None and not wants_one_sentence:
                stage_evidence = _session_evidence(session_context)
                supporting_evidence = stage_evidence[:1] if stage_evidence else []

        conclusion, supporting_evidence = _enforce_capability_consistency(
            question=question,
            session_context=session_context,
            conclusion=conclusion,
            supporting_evidence=supporting_evidence,
        )

        note_value = llm_output.get("note")
        note = _sanitize_text_field(str(note_value)) if isinstance(note_value, str) and note_value.strip() else None
        note = _normalize_answer_english(_first_sentence(note)) if note else None
        stage_next_action: str | None = None
        if results is None:
            _, stage_next_action = _stage_guidance(str((session_context or {}).get("status") or "idle"))
            # Prefer deterministic plain-language next step over verbose LLM phrasing.
            note = (
                stage_next_action
                if _should_include_idle_pipeline_note(question)
                else None
            )
        elif wants_concise:
            note = None

        # Keep citation trace strict: only quote-provided ids are returned.
        used_ids = [quote_id for quote_id in quoted_ids if quote_id in known_ids]

        references = _normalize_reference_list(llm_output.get("references"), literature_context.get("references") or [])
        if not needs_reference:
            references = []
        elif not references:
            ref_candidates = literature_context.get("references") or []
            if ref_candidates:
                references = [_reference_markdown(ref_candidates[0])]

        if wants_one_sentence:
            conclusion = _truncate_words(conclusion, 34)
        else:
            if is_clustering:
                conclusion_limit = 45
            elif results is None:
                conclusion_limit = 36
            else:
                conclusion_limit = 42
            if needs_reference and not wants_concise:
                conclusion_limit = 48
            conclusion = _truncate_words(conclusion, conclusion_limit)
            if results is None and conclusion.endswith("...") and len(conclusion.split()) < 10:
                stage_msg, _ = _stage_guidance(str((session_context or {}).get("status") or "idle"))
                conclusion = _normalize_answer_english(_first_sentence(stage_msg))
            evidence_word_limit = 30 if not is_clustering else 40
            supporting_evidence = [_truncate_words(entry, evidence_word_limit) for entry in supporting_evidence]
            clustering_trunc_limit = len(supporting_evidence) if is_clustering else 2
            supporting_evidence = _dedupe_nonempty(supporting_evidence, max_items=clustering_trunc_limit)
            note_fingerprint = ""
            if note:
                note_word_limit = 35 if is_clustering else (18 if (wants_concise or results is not None) else 24)
                note = _truncate_words(note, note_word_limit)
                note_fingerprint = _sanitize_text_field(note).lower()
            if note_fingerprint and any(
                note_fingerprint == _sanitize_text_field(x).lower()
                for x in [conclusion, *supporting_evidence]
            ):
                note = None
            if note and note.endswith("..."):
                if (
                    results is None
                    and stage_next_action
                    and _should_include_idle_pipeline_note(question)
                ):
                    note = _sanitize_text_field(stage_next_action)
                else:
                    note = None

        if wants_one_sentence:
            conclusion = _first_sentence(conclusion)
            supporting_evidence = []
            references = []
            note = None
            quote_context_local = None
            max_evidence = 0
            max_references = 0
        else:
            quote_context_local = None if wants_concise else quote_context
            max_evidence = len(supporting_evidence) if is_clustering else 2
            max_references = 1 if (wants_concise and references) else 2

        answer_text = _format_structured_answer(
            conclusion=conclusion,
            evidence=supporting_evidence,
            references=references,
            note=note,
            quote_context=quote_context_local,
            max_evidence=max_evidence,
            max_references=max_references,
        )

        # English-only output policy.
        if _contains_cjk(answer_text):
            return _fallback_answer(question, results, citation_blocks, quotes, session_context)

        return AnswerOutput(
            answer=answer_text,
            supporting_evidence=[_integerize_ci_text(entry) for entry in supporting_evidence],
            used_citation_block_ids=used_ids,
        )
    except (LLMCallError, ValueError, TypeError, KeyError):
        return _fallback_answer(question, results, citation_blocks, quotes, session_context)
