#!/usr/bin/env python3
"""
Comprehensive Test Suite for Function 1: Format Recognition & Standardization.

This script tests the Data Agent's ability to:
1. Correctly identify data formats (Pointwise, Pairwise, Multiway)
2. Assess engine compatibility
3. Trigger standardization only when necessary
4. Infer correct bigbetter direction
5. Identify ranking items and indicator columns

Usage:
    python test_function1.py [--verbose] [--dataset DATASET_NAME]
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "api"))

import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

from agents.data_agent import DataAgent
from core.schemas import DataFormat, InferredSchema


@dataclass
class TestCase:
    """A single test case with expected results."""
    name: str
    filepath: Path
    expected_format: DataFormat
    expected_bigbetter: int  # 1 = higher is better, 0 = lower is better
    expected_engine_compatible: bool = True
    expected_standardization_needed: bool = False
    description: str = ""
    
    
@dataclass
class TestResult:
    """Result of a single test case."""
    test_case: TestCase
    passed: bool
    schema: Optional[InferredSchema]
    format_correct: bool
    bigbetter_correct: bool
    engine_compatible_correct: bool
    standardization_correct: bool
    errors: list[str] = field(default_factory=list)
    duration_ms: float = 0.0


def define_test_cases(base_dir: Path) -> list[TestCase]:
    """Define all test cases with expected results."""
    
    test_cases = [
        # =====================================================================
        # Pointwise Format Tests
        # =====================================================================
        TestCase(
            name="pointwise_llm_benchmark",
            filepath=base_dir / "pointwise" / "standard_llm_benchmark.csv",
            expected_format=DataFormat.POINTWISE,
            expected_bigbetter=1,  # Accuracy scores, higher is better
            description="Standard LLM benchmark with accuracy scores [0,1]"
        ),
        TestCase(
            name="pointwise_product_ratings",
            filepath=base_dir / "pointwise" / "product_ratings.csv",
            expected_format=DataFormat.POINTWISE,
            expected_bigbetter=1,  # Ratings, higher is better
            description="Product ratings [1,5] scale"
        ),
        TestCase(
            name="pointwise_latency",
            filepath=base_dir / "pointwise" / "latency_measurements.csv",
            expected_format=DataFormat.POINTWISE,
            expected_bigbetter=0,  # Latency, lower is better
            description="Server latency measurements (lower is better)"
        ),
        
        # =====================================================================
        # Pairwise Format Tests
        # =====================================================================
        TestCase(
            name="pairwise_llm_comparison",
            filepath=base_dir / "pairwise" / "standard_llm_comparison.csv",
            expected_format=DataFormat.PAIRWISE,
            expected_bigbetter=1,  # 1 = winner
            description="Standard pairwise LLM comparison with Task indicator"
        ),
        TestCase(
            name="pairwise_ab_test",
            filepath=base_dir / "pairwise" / "ab_test_results.csv",
            expected_format=DataFormat.PAIRWISE,
            expected_bigbetter=1,
            description="A/B test results without indicator column"
        ),
        TestCase(
            name="pairwise_tournament",
            filepath=base_dir / "pairwise" / "tournament_brackets.csv",
            expected_format=DataFormat.PAIRWISE,
            expected_bigbetter=1,
            description="Tournament brackets with 10 teams"
        ),
        
        # =====================================================================
        # Multiway Format Tests
        # =====================================================================
        TestCase(
            name="multiway_horse_race",
            filepath=base_dir / "multiway" / "horse_race_results.csv",
            expected_format=DataFormat.MULTIWAY,
            expected_bigbetter=0,  # Rank 1 is best, so lower is better
            description="Horse race results with rank positions"
        ),
        TestCase(
            name="multiway_game_rankings",
            filepath=base_dir / "multiway" / "multiplayer_game_rankings.csv",
            expected_format=DataFormat.MULTIWAY,
            expected_bigbetter=0,  # Position 1 = winner
            description="Multi-player game rankings"
        ),
        TestCase(
            name="multiway_paper_rankings",
            filepath=base_dir / "multiway" / "paper_rankings_topk.csv",
            expected_format=DataFormat.MULTIWAY,
            expected_bigbetter=0,  # Rank 1 is best
            description="Academic paper rankings (partial/top-k)"
        ),
        
        # =====================================================================
        # Edge Case Tests
        # =====================================================================
        TestCase(
            name="edge_columns_with_spaces",
            filepath=base_dir / "edge_cases" / "columns_with_spaces.csv",
            expected_format=DataFormat.POINTWISE,
            expected_bigbetter=1,
            expected_engine_compatible=False,  # Spaces in column names
            expected_standardization_needed=True,
            description="Column names with spaces (should trigger standardization)"
        ),
        TestCase(
            name="edge_special_chars",
            filepath=base_dir / "edge_cases" / "columns_with_special_chars.csv",
            expected_format=DataFormat.POINTWISE,
            expected_bigbetter=1,
            expected_engine_compatible=False,  # Special chars in column names
            expected_standardization_needed=True,
            description="Column names with special characters"
        ),
        TestCase(
            name="edge_minimal_data",
            filepath=base_dir / "edge_cases" / "minimal_data.csv",
            expected_format=DataFormat.POINTWISE,
            expected_bigbetter=1,
            description="Minimal valid data (2 columns, 5 rows)"
        ),
        TestCase(
            name="edge_ambiguous_format",
            filepath=base_dir / "edge_cases" / "ambiguous_format.csv",
            expected_format=DataFormat.POINTWISE,  # Most likely interpretation
            expected_bigbetter=1,  # Ratings interpretation
            description="Ambiguous data that could be pointwise or multiway"
        ),
        TestCase(
            name="edge_mixed_numeric",
            filepath=base_dir / "edge_cases" / "mixed_numeric_types.csv",
            expected_format=DataFormat.POINTWISE,
            expected_bigbetter=1,
            description="Mixed integer and float numeric columns"
        ),
        TestCase(
            name="edge_many_non_numeric",
            filepath=base_dir / "edge_cases" / "many_non_numeric_columns.csv",
            expected_format=DataFormat.POINTWISE,
            expected_bigbetter=1,
            description="Many non-numeric columns (engine should auto-filter)"
        ),
    ]
    
    return test_cases


def run_single_test(agent: DataAgent, test_case: TestCase, verbose: bool = False) -> TestResult:
    """Run a single test case and return the result."""
    import time
    
    errors = []
    schema = None
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Test: {test_case.name}")
        print(f"File: {test_case.filepath}")
        print(f"Expected: format={test_case.expected_format.value}, bigbetter={test_case.expected_bigbetter}")
        print(f"{'='*60}")
    
    # Read the file
    try:
        with open(test_case.filepath, "rb") as f:
            content = f.read()
    except Exception as e:
        errors.append(f"Failed to read file: {e}")
        return TestResult(
            test_case=test_case,
            passed=False,
            schema=None,
            format_correct=False,
            bigbetter_correct=False,
            engine_compatible_correct=False,
            standardization_correct=False,
            errors=errors,
        )
    
    # Run the Data Agent
    start_time = time.time()
    try:
        schema, warnings, explanation = agent.process(
            content=content,
            filename=test_case.filepath.name,
        )
        duration_ms = (time.time() - start_time) * 1000
    except Exception as e:
        errors.append(f"Agent processing failed: {e}")
        return TestResult(
            test_case=test_case,
            passed=False,
            schema=None,
            format_correct=False,
            bigbetter_correct=False,
            engine_compatible_correct=False,
            standardization_correct=False,
            errors=errors,
            duration_ms=(time.time() - start_time) * 1000,
        )
    
    # Evaluate results
    format_correct = schema.format == test_case.expected_format
    bigbetter_correct = schema.bigbetter == test_case.expected_bigbetter
    engine_compatible_correct = schema.engine_compatible == test_case.expected_engine_compatible
    standardization_correct = schema.standardization_needed == test_case.expected_standardization_needed
    
    # For format, bigbetter - these are critical
    # For engine_compatible, standardization - more lenient (LLM may have different assessment)
    passed = format_correct and bigbetter_correct
    
    if verbose:
        print(f"\n--- Results ---")
        print(f"Format: {schema.format.value} (expected: {test_case.expected_format.value}) {'✓' if format_correct else '✗'}")
        print(f"BigBetter: {schema.bigbetter} (expected: {test_case.expected_bigbetter}) {'✓' if bigbetter_correct else '✗'}")
        print(f"Engine Compatible: {schema.engine_compatible} (expected: {test_case.expected_engine_compatible}) {'✓' if engine_compatible_correct else '○'}")
        print(f"Standardization Needed: {schema.standardization_needed} (expected: {test_case.expected_standardization_needed}) {'✓' if standardization_correct else '○'}")
        print(f"Confidence: {schema.confidence:.2f}")
        print(f"Ranking Items: {schema.ranking_items[:5]}{'...' if len(schema.ranking_items) > 5 else ''}")
        print(f"Indicator: {schema.indicator_col}")
        print(f"Duration: {duration_ms:.1f}ms")
        
        if warnings:
            print(f"\nWarnings: {[w.message for w in warnings]}")
        
        if not format_correct:
            errors.append(f"Format mismatch: got {schema.format.value}, expected {test_case.expected_format.value}")
        if not bigbetter_correct:
            errors.append(f"BigBetter mismatch: got {schema.bigbetter}, expected {test_case.expected_bigbetter}")
    
    return TestResult(
        test_case=test_case,
        passed=passed,
        schema=schema,
        format_correct=format_correct,
        bigbetter_correct=bigbetter_correct,
        engine_compatible_correct=engine_compatible_correct,
        standardization_correct=standardization_correct,
        errors=errors,
        duration_ms=duration_ms,
    )


def run_all_tests(verbose: bool = False, filter_name: Optional[str] = None) -> dict:
    """Run all test cases and return summary."""
    
    base_dir = Path(__file__).parent
    test_cases = define_test_cases(base_dir)
    
    # Filter if specified
    if filter_name:
        test_cases = [tc for tc in test_cases if filter_name.lower() in tc.name.lower()]
        if not test_cases:
            print(f"No test cases match filter: {filter_name}")
            return {}
    
    print(f"\n{'='*70}")
    print(f"Function 1: Format Recognition & Standardization Test Suite")
    print(f"{'='*70}")
    print(f"Test Cases: {len(test_cases)}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Initialize Data Agent
    print("\nInitializing Data Agent...")
    agent = DataAgent()
    
    if not agent.enabled:
        print("WARNING: LLM not available, tests will use fallback heuristics")
    
    # Run tests
    results: list[TestResult] = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] Running: {test_case.name}...", end=" ")
        result = run_single_test(agent, test_case, verbose=verbose)
        results.append(result)
        
        if not verbose:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"{status} ({result.duration_ms:.0f}ms)")
    
    # Summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    
    format_correct = sum(1 for r in results if r.format_correct)
    bigbetter_correct = sum(1 for r in results if r.bigbetter_correct)
    engine_compatible_correct = sum(1 for r in results if r.engine_compatible_correct)
    standardization_correct = sum(1 for r in results if r.standardization_correct)
    
    print(f"\nOverall: {passed}/{len(results)} passed ({100*passed/len(results):.1f}%)")
    print(f"\nBreakdown:")
    print(f"  Format Detection:      {format_correct}/{len(results)} ({100*format_correct/len(results):.1f}%)")
    print(f"  BigBetter Inference:   {bigbetter_correct}/{len(results)} ({100*bigbetter_correct/len(results):.1f}%)")
    print(f"  Engine Compatibility:  {engine_compatible_correct}/{len(results)} ({100*engine_compatible_correct/len(results):.1f}%)")
    print(f"  Standardization:       {standardization_correct}/{len(results)} ({100*standardization_correct/len(results):.1f}%)")
    
    total_duration = sum(r.duration_ms for r in results)
    print(f"\nTotal Duration: {total_duration:.1f}ms (avg: {total_duration/len(results):.1f}ms per test)")
    
    if failed > 0:
        print(f"\n--- FAILED TESTS ---")
        for r in results:
            if not r.passed:
                print(f"\n{r.test_case.name}:")
                for err in r.errors:
                    print(f"  - {err}")
    
    # Save results to JSON
    results_file = base_dir / "test_results.json"
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(results),
            "format_accuracy": format_correct / len(results),
            "bigbetter_accuracy": bigbetter_correct / len(results),
        },
        "results": [
            {
                "name": r.test_case.name,
                "passed": r.passed,
                "format_correct": r.format_correct,
                "bigbetter_correct": r.bigbetter_correct,
                "engine_compatible_correct": r.engine_compatible_correct,
                "standardization_correct": r.standardization_correct,
                "detected_format": r.schema.format.value if r.schema else None,
                "detected_bigbetter": r.schema.bigbetter if r.schema else None,
                "confidence": r.schema.confidence if r.schema else None,
                "duration_ms": r.duration_ms,
                "errors": r.errors,
            }
            for r in results
        ]
    }
    
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    return results_data


def main():
    parser = argparse.ArgumentParser(description="Test Function 1: Format Recognition & Standardization")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output for each test")
    parser.add_argument("--dataset", "-d", type=str, help="Filter tests by dataset name")
    args = parser.parse_args()
    
    results = run_all_tests(verbose=args.verbose, filter_name=args.dataset)
    
    # Exit with non-zero if any tests failed
    if results and results.get("summary", {}).get("failed", 0) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
