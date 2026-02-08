"""Tool: answer_question."""

from __future__ import annotations

from typing import Iterable

from core.schemas import AnswerOutput, QuotePayload, RankingResults


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
    relation = "statistically indistinguishable by CI overlap" if overlap else f"{lead} appears stronger"

    return (
        f"{item_a} rank={a_rank}, CI=[{a_ci[0]:.2f},{a_ci[1]:.2f}]; "
        f"{item_b} rank={b_rank}, CI=[{b_ci[0]:.2f},{b_ci[1]:.2f}]. "
        f"Interpretation: {relation}."
    )


def _extract_two_items(question: str, available_items: Iterable[str]) -> tuple[str | None, str | None]:
    lower_q = question.lower()
    matches = [item for item in available_items if item.lower() in lower_q]
    if len(matches) >= 2:
        return matches[0], matches[1]
    return None, None


def answer_question(
    question: str,
    results: RankingResults,
    citation_blocks: dict[str, str],
    quotes: list[QuotePayload] | None = None,
) -> AnswerOutput:
    """Answer follow-up questions using current results and optional quote context."""
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
    if "top" in lower_q or "best" in lower_q:
        answer = _top_summary(results)
        supporting_evidence.append(answer)
    else:
        item_a, item_b = _extract_two_items(question, results.items)
        if item_a and item_b:
            answer = _comparison_sentence(results, item_a, item_b)
            supporting_evidence.append(answer)
        else:
            answer = (
                _top_summary(results)
                + " Ask a pairwise comparison by mentioning two item names for a more specific answer."
            )
            supporting_evidence.append("General summary based on current ranking results.")

    if quote_context:
        answer = f"{answer}\n\nQuote context considered: {quote_context}"

    return AnswerOutput(
        answer=answer,
        supporting_evidence=supporting_evidence,
        used_citation_block_ids=used_ids,
    )
