"""Tool: answer_question."""

from __future__ import annotations

from typing import Iterable

from core.llm_client import LLMCallError, get_llm_client
from core.schemas import AnswerOutput, QuotePayload, RankingResults


CI_CAVEAT = "CI overlap is not a formal hypothesis test; interpret overlap as uncertainty context only."


def _to_int(value: float) -> int:
    return int(round(float(value)))


def _format_structured_answer(
    *,
    conclusion: str,
    evidence: list[str] | None = None,
    note: str | None = None,
    quote_context: str | None = None,
) -> str:
    lines: list[str] = [f"Conclusion: {conclusion.strip()}"]
    evidence_lines = [entry.strip() for entry in (evidence or []) if entry and entry.strip()]
    if evidence_lines:
        lines.append("Evidence:")
        lines.extend(f"- {entry}" for entry in evidence_lines[:3])
    if note:
        lines.append(f"Note: {note.strip()}")
    if quote_context:
        lines.append(f"Quote context considered: {quote_context.strip()}")
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


def _fallback_answer(
    question: str,
    results: RankingResults,
    citation_blocks: dict[str, str],
    quotes: list[QuotePayload] | None,
) -> AnswerOutput:
    """Deterministic fallback answer when LLM is unavailable."""
    supporting_evidence: list[str] = []
    used_ids: list[str] = []

    quote_context = ""
    if quotes:
        quote_lines = []
        for quote in quotes:
            if quote.block_id and quote.block_id in citation_blocks:
                used_ids.append(quote.block_id)
                quote_lines.append(f"Quoted block ({quote.block_id}): {citation_blocks[quote.block_id]}")
            else:
                quote_lines.append(f"Quoted text: {quote.quoted_text}")
        quote_context = " ".join(quote_lines)

    lower_q = question.lower()
    note: str | None = None
    if "top" in lower_q or "best" in lower_q:
        answer = _top_summary(results)
        top_idx = min(range(len(results.ranks)), key=lambda i: results.ranks[i])
        supporting_evidence.append(
            f"{results.items[top_idx]}: rank={results.ranks[top_idx]}, "
            f"theta_hat={results.theta_hat[top_idx]:.4f}, "
            f"CI=[{_to_int(results.ci_lower[top_idx])}, {_to_int(results.ci_upper[top_idx])}]"
        )
    else:
        item_a, item_b = _extract_two_items(question, results.items)
        if item_a and item_b:
            answer = _comparison_sentence(results, item_a, item_b)
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
            answer = (
                _top_summary(results)
                + " Ask a pairwise comparison by mentioning two item names for a more specific answer."
            )
            top_idx = min(range(len(results.ranks)), key=lambda i: results.ranks[i])
            supporting_evidence.append(
                f"{results.items[top_idx]} currently leads by rank with theta_hat={results.theta_hat[top_idx]:.4f}."
            )
    answer = _format_structured_answer(
        conclusion=answer,
        evidence=supporting_evidence,
        note=note,
        quote_context=quote_context or None,
    )

    return AnswerOutput(
        answer=answer,
        supporting_evidence=supporting_evidence,
        used_citation_block_ids=used_ids,
    )


def answer_question(
    question: str,
    results: RankingResults,
    citation_blocks: dict[str, str],
    quotes: list[QuotePayload] | None = None,
) -> AnswerOutput:
    """Answer follow-up questions using current results and optional quote context."""
    client = get_llm_client()
    if not client.is_available():
        return _fallback_answer(question, results, citation_blocks, quotes)

    known_ids = set(citation_blocks.keys())
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

    payload = {
        "question": question,
        "response_style": {
            "format": "short structured answer",
            "max_sections": 3,
            "max_bullets_per_section": 3,
            "target_length_words": 120,
        },
        "results": {
            "items": results.items,
            "ranks": results.ranks,
            "theta_hat": results.theta_hat,
            "ci_lower": results.ci_lower,
            "ci_upper": results.ci_upper,
            "metadata": results.metadata.model_dump() if results.metadata else None,
        },
        "quotes": quoted_blocks,
        "known_citation_block_ids": sorted(known_ids),
    }

    try:
        llm_output = client.generate_json("answer_question", payload=payload, max_completion_tokens=520)
        answer = str(llm_output.get("answer") or "").strip()
        if not answer:
            raise LLMCallError("Answer payload is empty.")

        supporting = llm_output.get("supporting_evidence")
        supporting_evidence = [str(entry) for entry in supporting] if isinstance(supporting, list) else []
        if not supporting_evidence:
            supporting_evidence = ["Derived from ranking scores and confidence intervals."]

        used_ids_raw = llm_output.get("used_citation_block_ids")
        used_ids: list[str] = []
        if isinstance(used_ids_raw, list):
            used_ids = [str(entry) for entry in used_ids_raw if str(entry) in known_ids]

        for quote in quotes or []:
            if quote.block_id and quote.block_id in known_ids and quote.block_id not in used_ids:
                used_ids.append(quote.block_id)

        return AnswerOutput(
            answer=answer,
            supporting_evidence=supporting_evidence,
            used_citation_block_ids=used_ids,
        )
    except (LLMCallError, ValueError, TypeError, KeyError) as exc:
        raise LLMCallError(f"LLM question answering failed: {exc}") from exc
