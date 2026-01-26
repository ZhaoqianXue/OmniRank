"""
Analyst Agent

Responsible for:
- Generating natural language reports from ranking results
- Answering user questions about the analysis
- Diagnosing errors and suggesting fixes
"""

import logging
import os
from typing import Optional

from openai import OpenAI

from core.schemas import (
    RankingResults,
    RankingItem,
    AgentType,
)
from core.session_memory import SessionMemory, TraceType

logger = logging.getLogger(__name__)


class AnalystAgent:
    """
    Analyst Agent: Generates reports and answers questions about results.
    
    Uses LLM to:
    1. Generate natural language summaries of ranking results
    2. Answer user questions about the analysis
    3. Diagnose errors and provide actionable suggestions
    """
    
    # Knowledge Layer: Domain expertise for ranking inference
    # NOTE: This is internal knowledge for the LLM. When generating user-facing content,
    # avoid technical jargon like "Step 1", "Step 2", "spectral", "vanilla" etc.
    SPECTRAL_KNOWLEDGE = """
## Ranking Analysis Knowledge Base

### Core Concept
The ranking method estimates preference scores from comparison data using 
statistically rigorous inference. Each item receives a score (theta) representing
its overall preference level.

### Confidence Intervals
- CI bounds indicate rank uncertainty; wider CIs suggest less reliable rankings
- Overlapping CIs between items mean they may be statistically indistinguishable
- 95% CI: We are 95% confident the true rank falls within this range

### Refined Analysis
- When data is imbalanced (some items compared more than others), the system
  automatically applies a refined analysis to improve accuracy
- High heterogeneity (uneven comparison counts) triggers this refinement
- The refinement uses optimal weighting to account for data imbalance

### Data Quality Considerations
- More comparisons lead to narrower confidence intervals
- Sparse data leads to wider confidence intervals (less certainty)
- Items with no comparisons to others cannot be ranked relative to them

### Comparing Items
- Use confidence intervals to determine if rank differences are significant
- If CI_A and CI_B overlap, A and B are statistically tied
- Score difference indicates magnitude of preference

### IMPORTANT FOR RESPONSES
When answering user questions, use accessible language. Avoid internal terms like:
- "Step 1", "Step 2", "vanilla", "spectral method"
- Instead say: "initial analysis", "refined analysis", "ranking method"
"""
    
    def __init__(self):
        """Initialize Analyst Agent."""
        self.name = "analyst"
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.client = OpenAI(api_key=api_key)
            self.model = os.getenv("OPENAI_MODEL", "gpt-5-nano")
            self.enabled = True
        else:
            logger.warning("OPENAI_API_KEY not set, Analyst Agent disabled")
            self.enabled = False
    
    def _format_ranking_table(self, items: list[RankingItem], top_n: int = 10) -> str:
        """Format ranking results as a text table."""
        lines = ["| Rank | Item | Score | CI (95%) |", "|------|------|-------|----------|"]
        
        for item in items[:top_n]:
            ci_str = f"[{item.ci_two_sided[0]}, {item.ci_two_sided[1]}]"
            lines.append(f"| {item.rank} | {item.name} | {item.theta_hat:.3f} | {ci_str} |")
        
        if len(items) > top_n:
            lines.append(f"| ... | ({len(items) - top_n} more items) | ... | ... |")
        
        return "\n".join(lines)
    
    def generate_report(
        self,
        results: RankingResults,
        session: SessionMemory,
    ) -> str:
        """
        Generate a natural language report from ranking results.
        
        Args:
            results: Ranking analysis results
            session: Session context
            
        Returns:
            Natural language report string
        """
        if not self.enabled:
            return self._generate_fallback_report(results)
        
        # Format data for prompt
        ranking_table = self._format_ranking_table(results.items)
        metadata = results.metadata
        
        prompt = f"""You are a statistical analysis expert. Generate a clear, professional report summarizing these spectral ranking results.

**Analysis Summary:**
- Number of items ranked: {metadata.n_items}
- Total comparisons: {metadata.n_comparisons}
- Data balance: {'Heterogeneous (refined analysis applied)' if metadata.step2_triggered else 'Well-balanced'}
- Runtime: {metadata.runtime_sec:.2f}s

**Ranking Results (sorted by rank):**
{ranking_table}

Generate a report that:
1. States the top-ranked items clearly
2. Notes any close competitions (overlapping confidence intervals)
3. Comments on data quality if relevant
4. Uses professional but accessible language

Keep the report concise (3-5 paragraphs). Do not use overly technical jargon. Do NOT mention internal technical terms like "Step 1", "Step 2", or "spectral method"."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=500,
            )
            
            report = response.choices[0].message.content.strip()
            
            # Log trace
            session.add_trace(
                TraceType.REPORT_GENERATION,
                {"status": "completed", "tokens": response.usage.total_tokens if response.usage else 0},
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
    
    def _generate_fallback_report(self, results: RankingResults) -> str:
        """Generate a basic report without LLM."""
        items = results.items
        metadata = results.metadata
        
        top_3 = items[:3]
        
        report = f"""## Ranking Analysis Report

**Summary:**
- {metadata.n_items} items were ranked based on {metadata.n_comparisons} comparisons.
- Analysis completed in {metadata.runtime_sec:.2f} seconds.

**Top Rankings:**
"""
        for item in top_3:
            report += f"- **#{item.rank} {item.name}** (score: {item.theta_hat:.3f}, CI: [{item.ci_two_sided[0]}, {item.ci_two_sided[1]}])\n"
        
        if metadata.step2_triggered:
            report += "\n*Note: Advanced refinement was applied for improved precision due to data heterogeneity.*"
        
        return report
    
    def answer_question(
        self,
        question: str,
        results: RankingResults,
        session: SessionMemory,
    ) -> str:
        """
        Answer a user question about the analysis.
        
        Args:
            question: User's question
            results: Ranking results
            session: Session context
            
        Returns:
            Answer string
        """
        if not self.enabled:
            return "I'm unable to answer questions at this time. Please check the ranking results directly."
        
        # Build context from session
        context = session.to_context_dict()
        ranking_table = self._format_ranking_table(results.items)
        
        prompt = f"""You are an expert statistical analyst specializing in ranking inference.

{self.SPECTRAL_KNOWLEDGE}

**Analysis Context:**
- File: {context.get('filename', 'Unknown')}
- Format: {context.get('schema', {}).get('format', 'Unknown')}
- Items ranked: {results.metadata.n_items}
- Data heterogeneity: {results.metadata.heterogeneity_index:.3f}
- Refined analysis: {'Yes (due to data imbalance)' if results.metadata.step2_triggered else 'No (data well-balanced)'}

**Current Results:**
{ranking_table}

**User Question:**
{question}

Provide a clear, accurate answer using your ranking analysis expertise.
- Reference confidence intervals when comparing items
- Explain statistical significance when relevant
- If the question cannot be answered from the data, explain why
- IMPORTANT: Do NOT mention internal technical terms like "Step 1", "Step 2", "vanilla spectral", or similar implementation details. Use user-friendly language."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=500,  # gpt-5-nano requires >= 500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            return f"I encountered an error while processing your question: {str(e)}"
    
    def diagnose_error(
        self,
        error: Exception,
        session: SessionMemory,
    ) -> str:
        """
        Diagnose an error and provide actionable suggestions.
        
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
        
        prompt = f"""You are a technical support assistant for a spectral ranking analysis system.

An error occurred during analysis:
- Error type: {error_type}
- Error message: {error_str}

Recent error traces:
{trace_info}

Session context:
- Filename: {session.filename}
- Status: {session.status.value}

Provide:
1. A clear explanation of what went wrong
2. Specific, actionable steps to fix the issue
3. Suggestions for preventing similar errors

Be concise and practical."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=500,  # gpt-5-nano requires >= 500
            )

            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error diagnosis failed: {e}")
            return self._generate_fallback_diagnosis(error_type, error_str)
    
    def _generate_fallback_diagnosis(self, error_type: str, error_str: str) -> str:
        """Generate basic error diagnosis without LLM."""
        suggestions = {
            "RScriptNotFoundError": "Please ensure R is installed and the spectral ranking scripts are in the correct location.",
            "RExecutionError": "The R script encountered an error. Check that your data file is properly formatted.",
            "RTimeoutError": "The analysis timed out. Try reducing the number of bootstrap iterations or using a smaller dataset.",
            "ROutputParseError": "Failed to parse R output. This may indicate a script error.",
            "ValueError": "Invalid input detected. Please verify your configuration settings.",
        }
        
        suggestion = suggestions.get(
            error_type,
            "Please check your data format and try again. If the problem persists, contact support."
        )
        
        return f"""**Error:** {error_type}

**Details:** {error_str}

**Suggestion:** {suggestion}"""
