"""
Analyst Agent

Responsible for:
- Function 1: Report & Visualization Generation
  - Report Synthesis: Structured reports with executive summary, rankings, methodology, insights
  - Visualization Production: Rank plots, heatmaps, distributions, topology (handled by frontend)
- Function 2: Interactive User Q&A
  - Context-aware answers using session memory and spectral ranking knowledge
- Error Diagnosis
  - Diagnose errors and suggest fixes
"""

import logging
import os
from typing import Optional

from openai import OpenAI

from core.schemas import (
    RankingResults,
    RankingItem,
    AgentType,
    SectionQuestions,
)
from core.session_memory import SessionMemory, TraceType

logger = logging.getLogger(__name__)


class AnalystAgent:
    """
    Analyst Agent: Transforms ranking results into insights and handles user interaction.
    
    Function 1: Report & Visualization Generation
    - Generate structured reports with: executive summary, detailed rankings,
      methodology notes, and actionable insights
    - Provide data for visualizations (actual rendering handled by frontend)
    
    Function 2: Interactive User Q&A
    - Answer follow-up questions using session memory and spectral ranking knowledge
    - Provide statistically grounded answers with CI interpretation
    """
    
    # ==========================================================================
    # Knowledge Layer: Domain expertise embedded in system prompt
    # Following OpenAI's Structured System Instructions pattern
    # ==========================================================================
    SPECTRAL_KNOWLEDGE = """
## Spectral Ranking Knowledge Base

### Core Logic (Simple Terms)
Think of spectral ranking as "voting": comparisons are votes.
- If A beats B, A gets points from B.
- Beating a strong opponent (who beats many others) is worth more than beating a weak one.
- The final score (theta) reflects the global consensus power of each item.

### Understanding Confidence Intervals (The "Range of Truth")
A rank is never just a single number; it's a range of possibilities.
- **95% CI**: We are 95% sure the true rank falls within this range.
- **Overlap**: If the ranges of Item A and Item B overlap, they are effectively tied. We cannot say specifically who is better with statistical certainty.
- **No Overlap**: If A's range is entirely higher than B's range, A is the clear, statistically significant winner.

### Dealing with Comparing Data Quality
- **Sparsity Ratio (Data Quantity)**: Like pixel density in an image.
  - ≥ 1.0 (High Res): We have enough data for a clear picture.
  - < 1.0 (Low Res): The picture is grainy; rankings are rough estimates.
- **Heterogeneity (Fairness of Comparison Counts)**:
  - Balanced: Everyone played roughly the same number of matches.
  - High Heterogeneity: Some items played 100 times, others only 2.

### Analysis Methods
- **Initial Analysis (Standard)**: Good for balanced data.
- **Refined Analysis (Weighted)**: Automatically triggered when data is messy (highly heterogeneous). It gives smarter weights to comparisons to fix biases caused by uneven data, ensuring the most accurate possible ranking.

### Communicating Results
- Avoid jargon like "stationary distribution" or "Markov chain" unless asked.
- Focus on: **Is A essentially better than B?** (Check CI overlap).
- Use phrases like "statistically indistinguishable" instead of "null hypothesis not rejected."
"""

    # ==========================================================================
    # Report Generation Templates
    # ==========================================================================
    REPORT_SYSTEM_PROMPT = """You are an expert statistical analyst specializing in ranking inference.
Your task is to generate an insightful, professional analysis report that helps users understand their ranking results.

{knowledge}

## Report Structure Requirements
Generate a report with the following EXACT sections in this order:

### 1. Executive Summary
Write 2-3 sentences that answer:
- Who/what ranks at the top? Is this ranking statistically confident?
- Are there any surprising findings or notable patterns?
- What's the main takeaway for decision-making?

### 2. Statistical Significance Analysis
This is the MOST IMPORTANT section. Focus on:
- **Clear Winners**: Items with non-overlapping CIs from others (statistically significant superiority)
- **Statistical Ties**: Groups of items whose CIs overlap (cannot be reliably distinguished)
- **Interpretation Guide**: Explain what "non-overlapping CIs" means in plain language
  (e.g., "Items A and B have overlapping confidence intervals, meaning we cannot confidently say one is better than the other based on this data")

### 3. Performance Gaps
Analyze the score distributions:
- Gap between top performer and second place
- Gap between top tier and bottom tier
- Are rankings tightly clustered or clearly separated?
- Which positions have the highest uncertainty (widest CIs)?

### 4. Data Quality & Methodology
Provide context about the analysis:
- Sample size and comparison count
- Data quality considerations (sparse data, heterogeneous comparisons)
- Any data quality considerations (sparse data, heterogeneous comparisons)
- Confidence level interpretation (95% CI means: if we repeated this analysis many times, 95% of the intervals would contain the true rank)

### 5. Practical Recommendations
Provide actionable guidance:
- Which rankings can be trusted for decision-making?
- Where should caution be exercised?
- What additional data might help clarify uncertain rankings?
- Specific recommendations based on the domain (if identifiable from item names)

## Writing Guidelines
- Use markdown formatting with clear headers
- Be specific: use actual item names and numbers
- Explain statistical concepts in plain language
- Focus on insights, not just data description
- Total length: 500-800 words
- Use bullet points for clarity
- Highlight the most important findings
"""

    def __init__(self):
        """Initialize Analyst Agent."""
        self.name = "analyst"
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.client = OpenAI(api_key=api_key)
            # Use gpt-5-mini as per project rules (rules.md)
            self.model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
            self.enabled = True
        else:
            logger.warning("OPENAI_API_KEY not set, Analyst Agent disabled")
            self.enabled = False
    
    def _format_ranking_table(self, items: list[RankingItem], top_n: int = 10) -> str:
        """Format ranking results as a detailed text table."""
        lines = ["| Rank | Item | Score | 95% CI | CI Width |", 
                 "|:----:|:-----|:-----:|:------:|:--------:|"]
        
        for item in items[:top_n]:
            ci_width = item.ci_two_sided[1] - item.ci_two_sided[0]
            ci_str = f"[{item.ci_two_sided[0]}, {item.ci_two_sided[1]}]"
            lines.append(f"| {item.rank} | {item.name} | {item.theta_hat:.3f} | {ci_str} | {ci_width} |")
        
        if len(items) > top_n:
            lines.append(f"| ... | ({len(items) - top_n} more items) | ... | ... | ... |")
        
        return "\n".join(lines)
    
    def _identify_statistical_ties(self, items: list[RankingItem]) -> list[list[str]]:
        """Identify groups of items with overlapping CIs (statistical ties)."""
        groups = []
        used = set()
        
        for i, item_a in enumerate(items):
            if item_a.name in used:
                continue
            group = [item_a.name]
            for j, item_b in enumerate(items[i+1:], i+1):
                if item_b.name in used:
                    continue
                # Check if CIs overlap
                if not (item_a.ci_two_sided[1] < item_b.ci_two_sided[0] or 
                        item_b.ci_two_sided[1] < item_a.ci_two_sided[0]):
                    group.append(item_b.name)
            if len(group) > 1:
                for name in group:
                    used.add(name)
                groups.append(group)
        
        return groups
    
    def _analyze_data_quality(self, metadata) -> dict:
        """Analyze data quality indicators for reporting."""
        quality = {
            "sparsity_status": "sufficient" if metadata.sparsity_ratio >= 1.0 else "sparse",
            "heterogeneity_status": "high" if metadata.heterogeneity_index > 0.5 else "balanced",
        }
        return quality
    
    def generate_report(
        self,
        results: RankingResults,
        session: SessionMemory,
    ) -> str:
        """
        Generate a comprehensive structured report from ranking results.
        
        Function 1: Report Synthesis
        - Executive summary with key findings
        - Detailed rankings with CIs and significance
        - Methodology notes
        - Actionable insights
        
        Args:
            results: Ranking analysis results
            session: Session context
            
        Returns:
            Structured markdown report string
        """
        if not self.enabled:
            return self._generate_fallback_report(results)
        
        # Prepare comprehensive context for LLM
        ranking_table = self._format_ranking_table(results.items, top_n=15)
        metadata = results.metadata
        quality = self._analyze_data_quality(metadata)
        ties = self._identify_statistical_ties(results.items[:10])
        
        # Build detailed context
        context = f"""## Analysis Context

### Data Summary
- **Items Ranked**: {metadata.n_items}
- **Total Comparisons**: {metadata.n_comparisons}
- **Sparsity Ratio**: {metadata.sparsity_ratio:.2f} ({"sufficient data" if quality["sparsity_status"] == "sufficient" else "sparse data, interpret with caution"})
- **Heterogeneity Index**: {metadata.heterogeneity_index:.2f} ({"high variation in comparison counts" if quality["heterogeneity_status"] == "high" else "balanced comparison distribution"})
- **Analysis Method**: Spectral ranking with bootstrap uncertainty quantification
- **Runtime**: {metadata.runtime_sec:.2f} seconds

### Complete Rankings
{ranking_table}

### Statistical Ties Detected
{self._format_ties(ties) if ties else "No statistical ties detected among top items - all rankings are statistically distinguishable."}

### Top vs Bottom Gap
- Top item ({results.items[0].name}): score = {results.items[0].theta_hat:.3f}
- Bottom item ({results.items[-1].name}): score = {results.items[-1].theta_hat:.3f}
- Score range: {results.items[0].theta_hat - results.items[-1].theta_hat:.3f}

### Session Context
- Filename: {session.filename if session.filename else "Unknown"}
"""

        # Build system prompt with knowledge layer
        system_prompt = self.REPORT_SYSTEM_PROMPT.format(knowledge=self.SPECTRAL_KNOWLEDGE)
        
        user_prompt = f"""Based on the analysis context below, generate a comprehensive ranking analysis report.

{context}

Generate a well-structured report following the required format (Executive Summary, Rankings Overview, Key Findings, Methodology Notes, Actionable Insights)."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=2048,  # Increased for comprehensive reports
            )
            
            report = response.choices[0].message.content.strip()
            
            # Log trace
            session.add_trace(
                TraceType.REPORT_GENERATION,
                {
                    "status": "completed", 
                    "tokens": response.usage.total_tokens if response.usage else 0,
                    "model": self.model,
                },
                agent=AgentType.ANALYST,
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            session.add_trace(
                TraceType.REPORT_GENERATION,
                {"status": "failed", "error": str(e)},
                agent=AgentType.ANALYST,
                success=False,
                error_message=str(e),
            )
            return self._generate_fallback_report(results)
    
    def _format_ties(self, ties: list[list[str]]) -> str:
        """Format statistical ties for display."""
        if not ties:
            return ""
        lines = []
        for i, group in enumerate(ties, 1):
            lines.append(f"- Group {i}: {', '.join(group)} (overlapping confidence intervals)")
        return "\n".join(lines)
    
    def _generate_fallback_report(self, results: RankingResults) -> str:
        """Generate a structured report without LLM when API is unavailable."""
        items = results.items
        metadata = results.metadata
        
        # Analyze data quality
        quality = self._analyze_data_quality(metadata)
        ties = self._identify_statistical_ties(items[:10])
        
        top_5 = items[:5]
        
        report = f"""## Ranking Analysis Report

### 1. Executive Summary

Based on {metadata.n_comparisons} comparisons among {metadata.n_items} items, **{items[0].name}** emerges as the top-ranked item with a preference score of {items[0].theta_hat:.3f}. The ranking confidence interval [{items[0].ci_two_sided[0]}, {items[0].ci_two_sided[1]}] indicates {"high confidence" if items[0].ci_two_sided[1] - items[0].ci_two_sided[0] <= 2 else "moderate confidence"} in this position.

### 2. Rankings Overview

| Rank | Item | Score | 95% CI |
|:----:|:-----|:-----:|:------:|
"""
        for item in top_5:
            ci_str = f"[{item.ci_two_sided[0]}, {item.ci_two_sided[1]}]"
            report += f"| {item.rank} | {item.name} | {item.theta_hat:.3f} | {ci_str} |\n"
        
        if len(items) > 5:
            report += f"| ... | ({len(items) - 5} more items) | ... | ... |\n"
        
        report += f"""
### 3. Key Findings

"""
        # Add statistical ties information
        if ties:
            report += "**Statistical Ties Detected:**\n"
            for i, group in enumerate(ties, 1):
                report += f"- {', '.join(group)} have overlapping confidence intervals and may be statistically indistinguishable\n"
        else:
            report += "- All top items have non-overlapping confidence intervals, indicating statistically significant rank differences\n"
        
        # Add score gap analysis
        score_gap = items[0].theta_hat - items[1].theta_hat if len(items) > 1 else 0
        report += f"- The gap between #1 ({items[0].name}) and #2 ({items[1].name if len(items) > 1 else 'N/A'}) is {score_gap:.3f} in preference score\n"
        
        report += f"""
### 4. Methodology Notes

- **Data Quality**: {"Sufficient comparison data (sparsity ratio >= 1.0)" if quality["sparsity_status"] == "sufficient" else "Limited comparison data - interpret results with caution"}
- **Data Balance**: {"High heterogeneity in comparison counts" if quality["heterogeneity_status"] == "high" else "Balanced comparison distribution"}
- **Analysis Type**: Spectral ranking analysis with bootstrap-based confidence intervals
- **Computation Time**: {metadata.runtime_sec:.2f} seconds

### 5. Actionable Insights

- The top-ranked item **{items[0].name}** demonstrates consistent superior performance across comparisons
- {"Consider the statistical ties when making decisions - items with overlapping CIs may be practically equivalent" if ties else "Rankings are statistically distinguishable, supporting confident decision-making"}
- {"Additional comparison data may help narrow confidence intervals for more precise rankings" if quality["sparsity_status"] == "sparse" else "Data quantity is sufficient for reliable inference"}
"""
        
        return report
    
    def generate_suggested_questions(self, results: RankingResults) -> list[str]:
        """
        Generate suggested questions for the user based on ranking results.
        
        These questions help users explore their results through the Q&A interface.
        
        Args:
            results: Ranking analysis results
            
        Returns:
            List of 3 suggested questions
        """
        items = results.items
        metadata = results.metadata
        
        # Get top and second items
        top_item = items[0].name if items else "the top item"
        second_item = items[1].name if len(items) > 1 else "the second item"
        
        # Identify statistical ties
        ties = self._identify_statistical_ties(items[:6])
        has_ties = len(ties) > 0
        
        # Build context-aware questions
        questions = []
        
        # Question 1: About the top-ranked item
        questions.append(f"Is {top_item} significantly better than {second_item}?")
        
        # Question 2: About statistical ties or confidence
        if has_ties:
            tie_items = ties[0][:2]
            questions.append(f"Are {tie_items[0]} and {tie_items[1]} statistically different?")
        else:
            questions.append("Which rankings can I trust for decision-making?")
        
        # Question 3: About methodology or data quality
        if metadata.sparsity_ratio < 1.0:
            questions.append("How does sparse data affect my ranking results?")
        else:
            questions.append("How should I interpret the confidence intervals?")
        
        return questions
    
    def generate_section_questions(self, results: RankingResults) -> SectionQuestions:
        """
        Generate LLM-powered questions for each report section.
        
        Uses the LLM to create context-aware, insightful questions that help users
        explore different aspects of their ranking results.
        
        Args:
            results: Ranking analysis results
            
        Returns:
            SectionQuestions with questions for each report section
        """
        items = results.items
        metadata = results.metadata
        
        # Build context for LLM
        top_items = [item.name for item in items[:5]]
        top_item = items[0].name if items else "the top item"
        second_item = items[1].name if len(items) > 1 else "the second item"
        last_item = items[-1].name if items else "the last item"
        
        # Identify key characteristics
        ties = self._identify_statistical_ties(items[:6])
        quality = self._analyze_data_quality(metadata)
        
        # If LLM not available, return context-aware fallback questions
        if not self.enabled:
            return self._generate_fallback_section_questions(results, ties, quality)
        
        prompt = f"""Based on the following ranking analysis results, generate insightful questions for each section of the report.

## Analysis Context
- Top ranked items: {', '.join(top_items)}
- Total items: {len(items)}
- Comparisons analyzed: {metadata.n_comparisons}
- Data quality: {quality["sparsity_status"]}
- Heterogeneity: {quality["heterogeneity_status"]}
- Analysis method: Spectral ranking with bootstrap confidence intervals
- Statistical ties detected: {"Yes, between " + str(ties[0][:3]) if ties else "None"}

## Generate Questions
For each section, generate 3 specific, contextual questions that would help a user understand their results better. Questions should:
- Reference actual item names when relevant
- Be specific to the data characteristics
- Help users make decisions based on the rankings

Return your response in this exact JSON format (no markdown, just JSON):
{{
  "rankings": ["question1", "question2", "question3"],
  "insights": ["question1", "question2", "question3"],
  "score_distribution": ["question1", "question2", "question3"],
  "confidence_intervals": ["question1", "question2", "question3"]
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a statistical analyst helping users understand ranking results. Generate helpful, specific questions."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=800,
                response_format={"type": "json_object"},
            )
            
            content = response.choices[0].message.content
            if content:
                import json
                data = json.loads(content)
                return SectionQuestions(
                    rankings=data.get("rankings", [])[:3],
                    insights=data.get("insights", [])[:3],
                    score_distribution=data.get("score_distribution", [])[:3],
                    confidence_intervals=data.get("confidence_intervals", [])[:3],
                )
        except Exception as e:
            logger.error(f"Failed to generate section questions: {e}")
        
        # Fallback to context-aware questions
        return self._generate_fallback_section_questions(results, ties, quality)
    
    def _generate_fallback_section_questions(
        self, 
        results: RankingResults, 
        ties: list[list[str]], 
        quality: dict
    ) -> SectionQuestions:
        """Generate context-aware fallback questions when LLM is unavailable."""
        items = results.items
        metadata = results.metadata
        
        top_item = items[0].name if items else "the top item"
        second_item = items[1].name if len(items) > 1 else "the second item"
        
        return SectionQuestions(
            rankings=[
                f"Is {top_item} significantly better than {second_item}?",
                "Which items have the most reliable rankings?",
                "Are there any statistical ties I should consider?",
            ],
            insights=[
                "How should I interpret these ranking results?",
                f"What makes {top_item} stand out from the others?",
                "What are the key limitations of this analysis?",
            ],
            score_distribution=[
                "What does the score distribution tell us about performance?",
                "Are there distinct performance tiers among the items?",
                "Which items have similar performance levels?",
            ],
            confidence_intervals=[
                "Which rankings have the most uncertainty?",
                "How do I know if two items are truly different?",
                "What would improve the confidence in these rankings?",
            ],
        )
    
    def answer_question(
        self,
        question: str,
        results: RankingResults,
        session: SessionMemory,
    ) -> str:
        """
        Answer a user question about the analysis.
        
        Function 2: Interactive User Q&A
        - Combines session memory with spectral ranking knowledge
        - Provides statistically grounded answers with CI interpretation
        
        Args:
            question: User's question
            results: Ranking results
            session: Session context
            
        Returns:
            Answer string
        """
        if not self.enabled:
            return "I'm unable to answer questions at this time. Please check the ranking results directly."
        
        # Build comprehensive context from session
        context = session.to_context_dict()
        ranking_table = self._format_ranking_table(results.items, top_n=15)
        quality = self._analyze_data_quality(results.metadata)
        ties = self._identify_statistical_ties(results.items[:10])
        
        system_prompt = f"""You are an expert statistical consultant known for explaining complex data simply.
Your goal is to answer questions about ranking results using the "Bottom Line Up Front" (BLUF) method.

{self.SPECTRAL_KNOWLEDGE}

## Response Structure
1. **Direct Answer**: Start with a clear "Yes", "No", or direct observation.
2. **Evidence**: Cite specific data (e.g., "Item A's score is 1.5 vs Item B's 1.2", "CIs overlap by 20%").
3. **Plain English Explanation**: Explain *why* this matters using simple analogies (e.g., "Statistically, it's a tie").

## Rules
- **Be Concise**: Max 3 paragraphs.
- **Reference Session Data**: Always mention specific item names, ranks, or values from the user's data.
- **Strictly Interpret CIs**: Never say A is better than B if their CIs overlap, even if A's score is higher. Call it a "statistical tie".
"""

        user_prompt = f"""## Analysis Context

**Data Summary:**
- File: {context.get('filename', 'Unknown')}
- Items ranked: {results.metadata.n_items}
- Total comparisons: {results.metadata.n_comparisons}
- Data quality: {"Sufficient" if quality["sparsity_status"] == "sufficient" else "Sparse"}
- Data balance: {"High heterogeneity" if quality["heterogeneity_status"] == "high" else "Well-balanced"}
- Analysis type: Spectral ranking with bootstrap uncertainty

**Current Rankings:**
{ranking_table}

**Statistical Ties (overlapping CIs):**
{self._format_ties(ties) if ties else "None detected among top items"}

---

**User Question:**
{question}

Please provide a clear, accurate answer based on the ranking results above."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=1024,
            )
            
            content = response.choices[0].message.content
            if content is None or content.strip() == "":
                return "I'm unable to generate an answer at this time. Please try rephrasing your question."
            return content.strip()
            
        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            return f"I encountered an error while processing your question: {str(e)}"
    
    def answer_general_question(self, question: str) -> str:
        """
        Answer a general question about OmniRank without session context.
        
        Used in pre-upload stage when user asks about:
        - How the system works
        - What data formats are supported
        - Methodology explanations
        - Getting started guidance
        
        Args:
            question: User's question
            
        Returns:
            Answer string
        """
        if not self.enabled:
            return "I'm unable to answer questions at this time. Please upload data to get started with analysis."
        
        system_prompt = f"""You are OmniRank Assistant. Your goal is to get the user to upload data and start analyzing.
Keep answers short, encouraging, and focused on value.

{self.SPECTRAL_KNOWLEDGE}

## Response Style
- **Concise**: 2-3 sentences per concept.
- **Action-Oriented**: End with "Ready to try? Upload your [format] file to see..."
- **Analogy-Driven**: Use "voting", "matches", or "races" to explain comparisons.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                max_completion_tokens=800,
            )
            
            content = response.choices[0].message.content
            if content is None or content.strip() == "":
                return "I'm unable to generate an answer at this time. Please try rephrasing your question."
            return content.strip()
            
        except Exception as e:
            logger.error(f"General question answering failed: {e}")
            return f"I encountered an error while processing your question: {str(e)}"
    
    def answer_schema_question(
        self,
        question: str,
        session: SessionMemory,
    ) -> str:
        """
        Answer a question about data schema before analysis is run.
        
        Used in post-schema stage when user asks about:
        - Data format interpretation
        - Configuration options
        - What the analysis will do
        - Schema-related questions
        
        Args:
            question: User's question
            session: Session context with schema info
            
        Returns:
            Answer string
        """
        if not self.enabled:
            return "I'm unable to answer questions at this time. Please configure and run your analysis."
        
        # Get schema context
        schema = session.inferred_schema
        schema_info = ""
        if schema:
            schema_info = f"""
**Detected Data Format:** {schema.format}
**Ranking Items:** {', '.join(schema.ranking_items[:10])} ({len(schema.ranking_items)} total)
**Ranking Direction:** {"Higher is better" if schema.bigbetter == 1 else "Lower is better"}
"""
            if schema.indicator_col:
                schema_info += f"**Indicator Column:** {schema.indicator_col} ({', '.join(schema.indicator_values[:5])})"
        
        system_prompt = f"""You are OmniRank Assistant. The user has uploaded data and is looking at the config screen.
Help them understand what their choices mean for the final result.

{self.SPECTRAL_KNOWLEDGE}

## Current Context
- **Schema**: {schema_info}

## Response Style
- **Anticipatory**: "If you choose X, the analysis will focus on..."
- **Clarifying**: "This format means we're comparing..."
- **Concise**: Get straight to the point. No fluff.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                max_completion_tokens=800,
            )
            
            content = response.choices[0].message.content
            if content is None or content.strip() == "":
                return "I'm unable to generate an answer at this time. Please try rephrasing your question."
            return content.strip()
            
        except Exception as e:
            logger.error(f"Schema question answering failed: {e}")
            return f"I encountered an error while processing your question: {str(e)}"
    
    def diagnose_error(
        self,
        error: Exception,
        session: SessionMemory,
    ) -> str:
        """
        Diagnose an error and provide actionable suggestions.
        
        Error Diagnosis Function:
        - Classifies error type (DATA_ERROR vs EXECUTION_ERROR)
        - Provides clear explanation and actionable fixes
        - Suggests preventive measures
        
        Args:
            error: The exception that occurred
            session: Session context for debugging
            
        Returns:
            Diagnostic message with suggestions
        """
        error_str = str(error)
        error_type = type(error).__name__
        
        # Get recent error traces
        error_traces = session.get_error_traces()
        trace_info = [
            {"type": t.trace_type.value, "message": t.error_message}
            for t in error_traces[-3:]
        ]
        
        if not self.enabled:
            return self._generate_fallback_diagnosis(error_type, error_str)
        
        system_prompt = """You are a technical support assistant for OmniRank, a spectral ranking analysis system.
Your role is to diagnose errors and provide clear, actionable solutions.

## Error Classification
- DATA_ERROR: Issues with input data format, missing values, insufficient data
- EXECUTION_ERROR: Issues with computation, memory, timeout, R script failures

## Response Guidelines
1. First, classify the error type
2. Explain what went wrong in plain language
3. Provide specific, step-by-step fixes
4. Suggest how to prevent similar errors
5. Be concise and practical - users need quick solutions
"""

        user_prompt = f"""## Error Information

**Error Type:** {error_type}
**Error Message:** {error_str}

**Recent Error Traces:**
{trace_info}

**Session Context:**
- Filename: {session.filename}
- Status: {session.status.value}

Please diagnose this error and provide:
1. A clear explanation of what went wrong
2. Specific, actionable steps to fix the issue
3. Suggestions for preventing similar errors"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=1024,
            )

            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error diagnosis failed: {e}")
            return self._generate_fallback_diagnosis(error_type, error_str)
    
    def _generate_fallback_diagnosis(self, error_type: str, error_str: str) -> str:
        """Generate structured error diagnosis without LLM."""
        # Error classification and suggestions
        error_info = {
            "RScriptNotFoundError": {
                "classification": "EXECUTION_ERROR",
                "explanation": "The R script required for spectral ranking computation could not be found.",
                "fix": "Ensure R is installed and the spectral ranking scripts are in the correct location (src/spectral_ranking/).",
            },
            "RExecutionError": {
                "classification": "EXECUTION_ERROR",
                "explanation": "The R script encountered an error during computation.",
                "fix": "Check that your data file is properly formatted with valid numeric values and required columns.",
            },
            "RTimeoutError": {
                "classification": "EXECUTION_ERROR",
                "explanation": "The analysis exceeded the maximum allowed computation time.",
                "fix": "Try reducing bootstrap iterations (e.g., from 2000 to 500) or use a smaller dataset.",
            },
            "ROutputParseError": {
                "classification": "EXECUTION_ERROR",
                "explanation": "Failed to parse the output from the R computation.",
                "fix": "This may indicate a script error. Check the data format and try again.",
            },
            "ValueError": {
                "classification": "DATA_ERROR",
                "explanation": "Invalid input parameters were provided.",
                "fix": "Verify your configuration settings match the data (e.g., bigbetter direction, selected items).",
            },
            "FileNotFoundError": {
                "classification": "DATA_ERROR",
                "explanation": "The specified data file could not be found.",
                "fix": "Re-upload your data file and try again.",
            },
        }
        
        info = error_info.get(error_type, {
            "classification": "UNKNOWN_ERROR",
            "explanation": "An unexpected error occurred during analysis.",
            "fix": "Please check your data format and try again. If the problem persists, contact support.",
        })
        
        return f"""## Error Diagnosis

**Classification:** {info["classification"]}

**Error Type:** {error_type}

**What Went Wrong:**
{info["explanation"]}

**Details:** {error_str}

**How to Fix:**
{info["fix"]}

**Prevention:**
- Ensure data is properly formatted before upload
- Start with default settings and adjust only if needed
- For large datasets, consider reducing bootstrap iterations"""
