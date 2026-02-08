"""Single-agent orchestrator for OmniRank fixed tool pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from agents.prompt_loader import load_system_prompt
from core.schemas import (
    AnswerOutput,
    ConfirmationResult,
    EngineConfig,
    InferResponse,
    QuotePayload,
    RunResponse,
    SessionStatus,
)
from core.session_memory import SessionMemory
from tools import ToolRegistry, build_tool_registry


PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)


class OmniRankAgent:
    """Single context-window agent with fixed tool calling pipeline."""

    def __init__(self, registry: ToolRegistry | None = None):
        self.registry = registry or build_tool_registry()
        self.system_prompt = load_system_prompt()
        self.model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None

        # Stage gating to prevent tool misuse.
        self.allowed_tools: dict[str, set[str]] = {
            "infer": {
                "read_data_file",
                "infer_semantic_schema",
                "validate_data_format",
                "preprocess_data",
                "validate_data_quality",
            },
            "confirm": {"request_user_confirmation", "validate_data_format", "validate_data_quality"},
            "run": {"execute_spectral_ranking", "generate_visualizations", "generate_report"},
            "question": {"answer_question"},
        }

    def _assert_stage_tool(self, stage: str, tool_name: str) -> None:
        """Enforce stage-scoped tool access."""
        if tool_name not in self.allowed_tools.get(stage, set()):
            raise RuntimeError(f"Tool '{tool_name}' is not allowed in stage '{stage}'.")

    def _call_tool(self, stage: str, session: SessionMemory, tool_name: str, **kwargs: Any) -> Any:
        """Call one tool with append-only trace recording."""
        self._assert_stage_tool(stage, tool_name)
        tool = self.registry.get(tool_name)
        try:
            output = tool(**kwargs)
            output_payload = output.model_dump() if hasattr(output, "model_dump") else output
            session.add_tool_call(tool_name=tool_name, inputs=kwargs, outputs=output_payload, success=True)
            return output
        except Exception as exc:  # noqa: BLE001
            session.add_tool_call(
                tool_name=tool_name,
                inputs=kwargs,
                outputs={},
                success=False,
                error=str(exc),
            )
            raise

    def _optional_llm_stage_note(self, stage: str, context: dict[str, Any]) -> None:
        """Optional model call for stage-level reasoning trace.

        This keeps gpt-5-mini involved while preserving deterministic tool execution fallback.
        """
        if self.client is None:
            return
        self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": f"Stage: {stage}. Keep fixed pipeline. Context: {context}",
                },
            ],
            max_completion_tokens=300,
        )

    def infer(self, session: SessionMemory, user_hints: str | None = None) -> InferResponse:
        """Run read->infer->format-loop->quality validation pipeline."""
        if not session.current_file_path:
            return InferResponse(success=False, requires_confirmation=False, error="No uploaded file found for session.")

        self._optional_llm_stage_note("infer", {"file": session.current_file_path})

        read_result = self._call_tool(
            "infer",
            session,
            "read_data_file",
            file_path=session.current_file_path,
        )
        if not read_result.success or not read_result.data:
            session.status = SessionStatus.ERROR
            session.error = read_result.error or "Failed to read data file"
            return InferResponse(success=False, requires_confirmation=False, error=session.error)

        session.data_summary = read_result.data

        schema_result = self._call_tool(
            "infer",
            session,
            "infer_semantic_schema",
            data_summary=read_result.data,
            file_path=session.current_file_path,
            user_hints=user_hints,
        )

        if not schema_result.success or not schema_result.schema:
            session.status = SessionStatus.ERROR
            session.error = schema_result.error or "Failed to infer schema"
            return InferResponse(success=False, requires_confirmation=False, error=session.error)

        session.inferred_schema = schema_result.schema

        current_file_path = session.current_file_path
        format_result = self._call_tool(
            "infer",
            session,
            "validate_data_format",
            file_path=current_file_path,
            schema=schema_result.schema,
        )

        # FORMAT VALIDATION LOOP
        max_loops = 3
        loop_count = 0
        while (not format_result.is_ready) and format_result.fixable:
            if loop_count >= max_loops:
                session.status = SessionStatus.ERROR
                session.error = "Exceeded maximum preprocessing loops."
                return InferResponse(
                    success=False,
                    schema_result=schema_result,
                    format_result=format_result,
                    requires_confirmation=False,
                    error=session.error,
                )

            preprocess_result = self._call_tool(
                "infer",
                session,
                "preprocess_data",
                file_path=current_file_path,
                schema=schema_result.schema,
                output_dir=str((session.current_file_path and os.path.dirname(session.current_file_path)) or ""),
            )
            current_file_path = preprocess_result.preprocessed_csv_path
            session.current_file_path = current_file_path

            format_result = self._call_tool(
                "infer",
                session,
                "validate_data_format",
                file_path=current_file_path,
                schema=schema_result.schema,
            )
            loop_count += 1

        session.format_validation_result = format_result
        if not format_result.is_ready:
            session.status = SessionStatus.ERROR
            session.error = "Data format validation failed and is not fixable."
            return InferResponse(
                success=False,
                schema_result=schema_result,
                format_result=format_result,
                requires_confirmation=False,
                error=session.error,
            )

        quality_result = self._call_tool(
            "infer",
            session,
            "validate_data_quality",
            file_path=current_file_path,
            schema=schema_result.schema,
        )

        session.quality_validation_result = quality_result
        if not quality_result.is_valid:
            session.status = SessionStatus.ERROR
            session.error = "Data quality validation failed."
            return InferResponse(
                success=False,
                data_summary=session.data_summary,
                schema_result=schema_result,
                format_result=format_result,
                quality_result=quality_result,
                requires_confirmation=False,
                error=session.error,
            )

        session.status = SessionStatus.AWAITING_CONFIRMATION
        session.error = None
        return InferResponse(
            success=True,
            data_summary=session.data_summary,
            schema_result=schema_result,
            format_result=format_result,
            quality_result=quality_result,
            preprocessed_path=session.current_file_path,
            requires_confirmation=True,
        )

    def confirm(
        self,
        session: SessionMemory,
        confirmed: bool,
        confirmed_schema,
        user_modifications: list[str],
        B: int,
        seed: int,
    ) -> ConfirmationResult:
        """Persist user confirmation and schema updates."""
        if session.status not in {SessionStatus.AWAITING_CONFIRMATION, SessionStatus.INFERRED, SessionStatus.CONFIRMED}:
            raise RuntimeError(f"Session is not ready for confirmation in state {session.status.value}.")

        if session.inferred_schema is None or session.format_validation_result is None or session.quality_validation_result is None:
            raise RuntimeError("Cannot confirm before successful infer phase.")

        self._optional_llm_stage_note("confirm", {"session_id": session.session_id})

        confirmation = self._call_tool(
            "confirm",
            session,
            "request_user_confirmation",
            proposed_schema=session.inferred_schema,
            format_result=session.format_validation_result,
            quality_result=session.quality_validation_result,
            confirmed=confirmed,
            confirmed_schema=confirmed_schema,
            user_modifications=user_modifications,
            B=B,
            seed=seed,
        )

        if not confirmation.confirmed:
            session.status = SessionStatus.AWAITING_CONFIRMATION
            return confirmation

        session.confirmed_schema = confirmation.confirmed_schema
        session.config = EngineConfig(
            csv_path=session.current_file_path or session.original_file_path or "",
            bigbetter=confirmation.confirmed_schema.bigbetter,
            selected_items=confirmation.confirmed_schema.ranking_items,
            selected_indicator_values=confirmation.confirmed_schema.indicator_values,
            B=confirmation.B,
            seed=confirmation.seed,
            r_script_path="src/spectral_ranking/spectral_ranking.R",
        )
        session.status = SessionStatus.CONFIRMED
        session.error = None

        return confirmation

    def run(self, session: SessionMemory, selected_items: list[str] | None, selected_indicator_values: list[str] | None) -> RunResponse:
        """Execute engine + visualization + report generation."""
        if session.status not in {SessionStatus.CONFIRMED, SessionStatus.COMPLETED}:
            return RunResponse(success=False, error=f"Session is not runnable in state {session.status.value}.")
        if session.config is None:
            return RunResponse(success=False, error="Missing confirmed engine config.")

        self._optional_llm_stage_note("run", {"session_id": session.session_id})

        session.status = SessionStatus.RUNNING
        session.config.selected_items = selected_items if selected_items else session.config.selected_items
        session.config.selected_indicator_values = (
            selected_indicator_values if selected_indicator_values else session.config.selected_indicator_values
        )

        work_dir = os.path.dirname(session.current_file_path or session.original_file_path or "")
        execution = self._call_tool(
            "run",
            session,
            "execute_spectral_ranking",
            config=session.config,
            session_work_dir=work_dir,
        )
        session.add_execution_trace(execution.trace)

        if not execution.success or not execution.results:
            session.status = SessionStatus.ERROR
            session.error = execution.error or "Engine execution failed"
            return RunResponse(success=False, config=session.config, execution=execution, error=session.error)

        session.current_results = execution.results

        artifact_dir = os.path.join(work_dir, "artifacts")
        viz_output = self._call_tool(
            "run",
            session,
            "generate_visualizations",
            results=execution.results,
            viz_types=["ranking_bar", "ci_forest"],
            artifact_dir=artifact_dir,
        )
        session.visualization_output = viz_output

        report = self._call_tool(
            "run",
            session,
            "generate_report",
            results=execution.results,
            session_meta={
                "B": session.config.B,
                "seed": session.config.seed,
                "current_file_path": session.current_file_path,
                "r_script_path": session.config.r_script_path,
            },
            plots=viz_output.plots,
        )
        session.report_output = report
        session.citation_blocks.extend(report.citation_blocks)

        for plot in viz_output.plots:
            session.register_artifact(kind="figure", path=plot.svg_path, title=plot.type, mime_type="image/svg+xml")

        for artifact in report.artifacts:
            session.register_artifact(
                kind=artifact.kind,
                path=artifact.path,
                title=artifact.title,
                mime_type=artifact.mime_type,
            )

        session.status = SessionStatus.COMPLETED
        session.error = None

        return RunResponse(
            success=True,
            config=session.config,
            execution=execution,
            visualizations=viz_output,
            report=report,
        )

    def answer(self, session: SessionMemory, question: str, quotes: list[QuotePayload] | None = None) -> AnswerOutput:
        """Answer follow-up question with quote-aware support."""
        if session.current_results is None:
            raise RuntimeError("No ranking results available for question answering.")

        self._optional_llm_stage_note("question", {"question": question})

        citation_lookup = {}
        if session.report_output is not None:
            citation_lookup = {
                block.block_id: block.text
                for block in session.report_output.citation_blocks
            }

        answer = self._call_tool(
            "question",
            session,
            "answer_question",
            question=question,
            results=session.current_results,
            citation_blocks=citation_lookup,
            quotes=quotes or [],
        )

        return answer
