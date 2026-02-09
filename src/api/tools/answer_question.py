"""Tool: answer_question."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
from typing import Any, Iterable

from core.llm_client import LLMCallError, get_llm_client
from core.schemas import AnswerOutput, QuotePayload, RankingResults


CI_CAVEAT = "CI overlap is not a formal hypothesis test; interpret overlap as uncertainty context only."
PROJECT_ROOT = Path(__file__).resolve().parents[3]
LITERATURE_PATH = PROJECT_ROOT / ".agent" / "literature" / "spectral_ranking_inferences.md"

_NUMERIC_BRACKET_PAIR_PATTERN = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")

_DEFAULT_LITERATURE_REFERENCES: list[dict[str, str]] = [
    {
        "title": "Fan et al. (2022b) Ranking inferences based on the top choice of multiway comparisons",
        "url": "https://arxiv.org/abs/2211.11957",
    },
    {
        "title": "Gao et al. (2021) Uncertainty quantification in the Bradley-Terry-Luce model",
        "url": "https://arxiv.org/abs/2110.03874",
    },
    {
        "title": "Fan et al. (2022a) Uncertainty quantification of MLE for entity ranking with covariates",
        "url": "https://arxiv.org/abs/2212.09961",
    },
]


def _to_int(value: float | str) -> int:
    return int(round(float(value)))


def _integerize_ci_text(text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        return f"[{_to_int(match.group(1))}, {_to_int(match.group(2))}]"

    return _NUMERIC_BRACKET_PAIR_PATTERN.sub(_replace, text)


def _reference_markdown(reference: dict[str, str]) -> str:
    return f"[{reference['title']}]({reference['url']})"


def _extract_section(text: str, start_marker: str, end_marker: str) -> str:
    start = text.find(start_marker)
    if start < 0:
        return ""
    start += len(start_marker)
    end = text.find(end_marker, start)
    if end < 0:
        end = len(text)
    return text[start:end].strip()


def _parse_references_from_literature(text: str) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("*"):
            continue
        link_match = re.search(r"\[(https?://[^\]]+)\]\((https?://[^)]+)\)", stripped)
        if not link_match:
            continue
        url = link_match.group(2).strip()
        if not url:
            continue
        title_part = stripped.lstrip("*").strip()
        title_part = title_part.split("[", 1)[0].strip().rstrip(",.")
        title_part = re.sub(r"\s+Preprint.*$", "", title_part).strip(" ,.")
        if not title_part:
            continue
        refs.append({"title": title_part, "url": url})

    unique_refs: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for ref in refs:
        if ref["url"] in seen_urls:
            continue
        seen_urls.add(ref["url"])
        unique_refs.append(ref)
    return unique_refs


def _prioritize_references(refs: list[dict[str, str]]) -> list[dict[str, str]]:
    ranking_keywords = ("ranking", "spectral", "bradley-terry", "plackett", "entity ranking")

    def _score(ref: dict[str, str]) -> tuple[int, str]:
        title = ref.get("title", "").lower()
        is_ranking_related = any(keyword in title for keyword in ranking_keywords)
        return (0 if is_ranking_related else 1, title)

    return sorted(refs, key=_score)


@lru_cache(maxsize=1)
def _load_literature_context() -> dict[str, Any]:
    if not LITERATURE_PATH.exists():
        return {
            "source_path": str(LITERATURE_PATH),
            "summary": "",
            "references": list(_DEFAULT_LITERATURE_REFERENCES),
        }

    content = LITERATURE_PATH.read_text(encoding="utf-8")
    abstract = _extract_section(content, "## Abstract", "## 1.")
    compact_abstract = re.sub(r"\s+", " ", abstract).strip()
    if len(compact_abstract) > 900:
        compact_abstract = compact_abstract[:900].rstrip() + "..."

    parsed_refs = _parse_references_from_literature(content)
    if parsed_refs:
        references = _prioritize_references(parsed_refs)
        known_urls = {ref["url"] for ref in references}
        for default_ref in _DEFAULT_LITERATURE_REFERENCES:
            if default_ref["url"] not in known_urls:
                references.append(default_ref)
    else:
        references = list(_DEFAULT_LITERATURE_REFERENCES)

    key_points = [
        "Supports fixed and random comparison graphs, including heterogeneous multiway comparisons.",
        "Two-step spectral weighting can asymptotically match MLE efficiency under proper weighting.",
        "Uses Gaussian multiplier bootstrap for rank uncertainty quantification and confidence intervals.",
        "CI overlap should be treated as uncertainty context, not as a formal significance test.",
    ]

    return {
        "source_path": str(LITERATURE_PATH),
        "summary": compact_abstract,
        "key_points": key_points,
        "references": references[:8],
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


def _question_needs_reference(question: str, results: RankingResults | None) -> bool:
    lower_q = question.lower()
    keywords = [
        "method",
        "theory",
        "spectral",
        "bootstrap",
        "ci",
        "confidence interval",
        "why",
        "原理",
        "方法",
        "置信区间",
        "文献",
    ]
    return results is None or any(keyword in lower_q for keyword in keywords)


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
            snippets.append(text[:140] + ("..." if len(text) > 140 else ""))
        else:
            text = quote.quoted_text.strip()
            snippets.append(text[:140] + ("..." if len(text) > 140 else ""))

    if not snippets:
        return None, used_ids
    context = " | ".join(snippets)
    return context, used_ids


def _format_structured_answer(
    *,
    conclusion: str,
    evidence: list[str] | None = None,
    references: list[str] | None = None,
    note: str | None = None,
    quote_context: str | None = None,
) -> str:
    lines: list[str] = [f"Conclusion: {_integerize_ci_text(conclusion.strip())}"]
    evidence_lines = [_integerize_ci_text(entry.strip()) for entry in (evidence or []) if entry and entry.strip()]
    reference_lines = [entry.strip() for entry in (references or []) if entry and entry.strip()]

    if evidence_lines:
        lines.append("Evidence:")
        lines.extend(f"- {entry}" for entry in evidence_lines[:3])
    if reference_lines:
        lines.append("References:")
        lines.extend(f"- {entry}" for entry in reference_lines[:2])
    if note:
        lines.append(f"Note: {_integerize_ci_text(note.strip())}")
    if quote_context:
        lines.append(f"Quote context considered: {_integerize_ci_text(quote_context.strip())}")
    return "\n".join(lines)


def _top_summary(results: RankingResults) -> str:
    idx = min(range(len(results.ranks)), key=lambda i: results.ranks[i])
    return (
        f"Top-ranked item is {results.items[idx]} with rank {results.ranks[idx]} "
        f"and theta_hat={results.theta_hat[idx]:.4f}."
    )


def _comparison_sentence(results: RankingResults, item_a: str, item_b: str) -> str:
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
    lead = item_a if a_rank < b_rank else item_b
    relation = (
        "confidence intervals overlap, so separation between the two items remains uncertain"
        if overlap
        else f"{lead} appears stronger based on rank and non-overlapping intervals"
    )
    caveat = f" {CI_CAVEAT}" if overlap else ""
    return (
        f"{item_a} rank={a_rank}, CI=[{_to_int(a_ci[0])}, {_to_int(a_ci[1])}]; "
        f"{item_b} rank={b_rank}, CI=[{_to_int(b_ci[0])}, {_to_int(b_ci[1])}]. "
        f"Interpretation: {relation}.{caveat}"
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
            evidence.append(f"Schema includes {len(items)} ranking items.")
        if indicator:
            evidence.append(f"Indicator column: {indicator}.")

    warnings = session_context.get("quality_warnings")
    if isinstance(warnings, list) and warnings:
        evidence.append(f"Current quality warning: {warnings[0]}")

    return evidence[:3]


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
    stage_msg = {
        "idle": "No dataset is loaded yet, so item-level ranking and CI answers are not available.",
        "uploaded": "Data is uploaded but semantic/schema confirmation is not finished yet.",
        "awaiting_confirmation": "Schema confirmation is pending; ranking has not run yet.",
        "confirmed": "Configuration is confirmed; run analysis to generate ranking, theta_hat, and integer CIs.",
        "running": "Analysis is still running; wait for completion to answer item-level ranking questions.",
        "error": "Session is currently in error state; fix the blocking issue before requesting ranking-specific results.",
    }.get(status, "Ranking results are not available yet for item-level inference.")

    methodology_keywords = [
        "spectral",
        "bootstrap",
        "ci",
        "confidence interval",
        "method",
        "theory",
        "原理",
        "方法",
        "置信区间",
    ]
    asks_method = any(keyword in lower_q for keyword in methodology_keywords)

    if asks_method:
        conclusion = (
            "OmniRank uses spectral ranking to estimate latent preference scores (theta_hat), "
            "then uses Gaussian multiplier bootstrap for uncertainty quantification."
        )
        note = (
            "This session has not produced ranking outputs yet, so I cannot provide item-specific rank/CI values."
        )
    else:
        conclusion = stage_msg
        note = "After running analysis, I can answer item-level comparisons with integer CI bounds."

    evidence = _session_evidence(session_context)
    if not evidence:
        evidence = [
            "Q&A is available at every stage, including before report generation.",
            "Item-level rank and CI details require completed ranking outputs.",
        ]

    references: list[str] = []
    if _question_needs_reference(question, None):
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


def _fallback_with_results(
    *,
    question: str,
    results: RankingResults,
    quote_context: str | None,
    used_ids: list[str],
    literature_context: dict[str, Any],
) -> AnswerOutput:
    supporting_evidence: list[str] = []

    lower_q = question.lower()
    note: str | None = None
    if "top" in lower_q or "best" in lower_q:
        conclusion = _top_summary(results)
        top_idx = min(range(len(results.ranks)), key=lambda i: results.ranks[i])
        supporting_evidence.append(
            f"{results.items[top_idx]}: rank={results.ranks[top_idx]}, "
            f"theta_hat={results.theta_hat[top_idx]:.4f}, "
            f"CI=[{_to_int(results.ci_lower[top_idx])}, {_to_int(results.ci_upper[top_idx])}]"
        )
    else:
        item_a, item_b = _extract_two_items(question, results.items)
        if item_a and item_b:
            conclusion = _comparison_sentence(results, item_a, item_b)
            lookup = {name.lower(): i for i, name in enumerate(results.items)}
            a_idx = lookup.get(item_a.lower())
            b_idx = lookup.get(item_b.lower())
            if a_idx is not None and b_idx is not None:
                supporting_evidence.append(
                    f"{item_a}: rank={results.ranks[a_idx]}, "
                    f"theta_hat={results.theta_hat[a_idx]:.4f}, "
                    f"CI=[{_to_int(results.ci_lower[a_idx])}, {_to_int(results.ci_upper[a_idx])}]"
                )
                supporting_evidence.append(
                    f"{item_b}: rank={results.ranks[b_idx]}, "
                    f"theta_hat={results.theta_hat[b_idx]:.4f}, "
                    f"CI=[{_to_int(results.ci_lower[b_idx])}, {_to_int(results.ci_upper[b_idx])}]"
                )
                overlap = not (
                    results.ci_upper[a_idx] < results.ci_lower[b_idx]
                    or results.ci_upper[b_idx] < results.ci_lower[a_idx]
                )
                if overlap:
                    note = CI_CAVEAT
        else:
            conclusion = (
                _top_summary(results)
                + " Mention two item names to get a pairwise CI-aware comparison."
            )
            top_idx = min(range(len(results.ranks)), key=lambda i: results.ranks[i])
            supporting_evidence.append(
                f"{results.items[top_idx]} currently leads by rank with theta_hat={results.theta_hat[top_idx]:.4f}."
            )

    references: list[str] = []
    if _question_needs_reference(question, results):
        refs = literature_context.get("references") or []
        if refs:
            references = [_reference_markdown(refs[0])]

    answer_text = _format_structured_answer(
        conclusion=conclusion,
        evidence=supporting_evidence,
        references=references,
        note=note,
        quote_context=quote_context,
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
    literature_context = _load_literature_context()
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
    )


def answer_question(
    question: str,
    results: RankingResults | None,
    citation_blocks: dict[str, str],
    quotes: list[QuotePayload] | None = None,
    session_context: dict[str, Any] | None = None,
) -> AnswerOutput:
    """Answer user questions using optional ranking results and session context."""
    client = get_llm_client()
    if not client.is_available():
        return _fallback_answer(question, results, citation_blocks, quotes, session_context)

    known_ids = set(citation_blocks.keys())
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

    literature_context = _load_literature_context()
    payload = {
        "question": question,
        "response_style": {
            "format": "short structured answer",
            "max_sections": 4,
            "max_bullets_per_section": 3,
            "target_length_words": 140,
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
        llm_output = client.generate_json("answer_question", payload=payload, max_completion_tokens=640)
        conclusion = str(llm_output.get("conclusion") or llm_output.get("answer") or "").strip()
        if not conclusion:
            raise LLMCallError("Answer payload is empty.")

        evidence_raw = llm_output.get("evidence")
        if not isinstance(evidence_raw, list):
            evidence_raw = llm_output.get("supporting_evidence")
        supporting_evidence = [str(entry).strip() for entry in evidence_raw] if isinstance(evidence_raw, list) else []
        supporting_evidence = [entry for entry in supporting_evidence if entry][:3]
        if not supporting_evidence:
            if results is None:
                supporting_evidence = _session_evidence(session_context) or [
                    "No ranking results yet; this answer is based on session stage and methodology context."
                ]
            else:
                supporting_evidence = ["Derived from ranking scores and confidence intervals."]

        note_value = llm_output.get("note")
        note = str(note_value).strip() if isinstance(note_value, str) and note_value.strip() else None

        used_ids_raw = llm_output.get("used_citation_block_ids")
        used_ids: list[str] = []
        if isinstance(used_ids_raw, list):
            used_ids = [str(entry) for entry in used_ids_raw if str(entry) in known_ids]

        for quote_id in quoted_ids:
            if quote_id not in used_ids:
                used_ids.append(quote_id)

        references = _normalize_reference_list(llm_output.get("references"), literature_context.get("references") or [])
        if _question_needs_reference(question, results) and not references:
            ref_candidates = literature_context.get("references") or []
            if ref_candidates:
                references = [_reference_markdown(ref_candidates[0])]

        answer_text = _format_structured_answer(
            conclusion=conclusion,
            evidence=supporting_evidence,
            references=references,
            note=note,
            quote_context=quote_context,
        )

        return AnswerOutput(
            answer=answer_text,
            supporting_evidence=[_integerize_ci_text(entry) for entry in supporting_evidence],
            used_citation_block_ids=used_ids,
        )
    except (LLMCallError, ValueError, TypeError, KeyError):
        return _fallback_answer(question, results, citation_blocks, quotes, session_context)
