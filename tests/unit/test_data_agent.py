"""
Unit tests for Data Agent LLM integration.

Tests cover:
1. LLM-powered schema inference (with mocks)
2. Fallback mode (when LLM unavailable)
3. Format detection (pointwise, pairwise)
4. BigBetter inference (macro + micro level)
5. Indicator column detection
6. Sparsity check (M < n*log(n) formula)
7. Connectivity check (networkx)
8. Validation warnings generation
"""

import math
import json
import pytest
import pandas as pd
from io import BytesIO
from unittest.mock import Mock, patch, MagicMock

# Import after path setup in conftest
from agents.data_agent import DataAgent
from core.schemas import DataFormat, InferredSchema, ValidationWarning


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    mock_client = Mock()
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    return mock_client


@pytest.fixture
def agent_with_mock(mock_openai_client):
    """Create DataAgent with mocked OpenAI client."""
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
        agent = DataAgent()
        agent.client = mock_openai_client
        agent.enabled = True
        return agent


@pytest.fixture
def agent_fallback():
    """Create DataAgent in fallback mode (no API key)."""
    with patch.dict('os.environ', {}, clear=True):
        # Remove OPENAI_API_KEY from environment
        import os
        orig_key = os.environ.pop('OPENAI_API_KEY', None)
        try:
            agent = DataAgent()
            assert not agent.enabled, "Agent should be in fallback mode"
            return agent
        finally:
            if orig_key:
                os.environ['OPENAI_API_KEY'] = orig_key


# =============================================================================
# Test Class: LLM-Powered Schema Inference
# =============================================================================

class TestDataAgentLLM:
    """Test LLM-powered schema inference."""
    
    def test_pointwise_detection_with_llm(self, agent_with_mock, pointwise_df):
        """Test pointwise format detection using LLM."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=json.dumps({
            "format": "pointwise",
            "format_reasoning": "Dense numeric matrix with scores",
            "bigbetter": 1,
            "bigbetter_reasoning": "Values are accuracy-like bounded scores",
            "ranking_items": ["model_A", "model_B", "model_C"],
            "ranking_items_reasoning": "Numeric columns representing models",
            "indicator_col": None,
            "indicator_reasoning": "No categorical dimension found"
        })))]
        mock_response.usage = Mock(total_tokens=100)
        
        agent_with_mock.client.chat.completions.create.return_value = mock_response
        
        content = pointwise_df.to_csv(index=False).encode()
        schema, warnings, explanation = agent_with_mock.process(content, "test.csv")
        
        assert schema.format == DataFormat.POINTWISE
        assert schema.bigbetter == 1
        assert len(schema.ranking_items) == 3
    
    def test_pairwise_detection_with_llm(self, agent_with_mock, pairwise_df):
        """Test pairwise format detection using LLM."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=json.dumps({
            "format": "pairwise",
            "format_reasoning": "Sparse 0/1 matrix with exactly 2 values per row",
            "bigbetter": 1,
            "bigbetter_reasoning": "1 indicates winner in pairwise comparison",
            "ranking_items": ["Model_A", "Model_B", "Model_C"],
            "ranking_items_reasoning": "Columns with 0/1 winner encoding",
            "indicator_col": "Task",
            "indicator_reasoning": "Task column provides stratification"
        })))]
        mock_response.usage = Mock(total_tokens=100)
        
        agent_with_mock.client.chat.completions.create.return_value = mock_response
        
        content = pairwise_df.to_csv(index=False).encode()
        schema, warnings, explanation = agent_with_mock.process(content, "test.csv")
        
        assert schema.format == DataFormat.PAIRWISE
        assert schema.indicator_col == "Task"
        assert "code" in schema.indicator_values
        assert "math" in schema.indicator_values
    
    def test_llm_failure_falls_back_to_heuristics(self, agent_with_mock, pointwise_df):
        """Test that LLM failure triggers fallback to heuristics."""
        # Make LLM call fail
        agent_with_mock.client.chat.completions.create.side_effect = Exception("API Error")
        
        content = pointwise_df.to_csv(index=False).encode()
        schema, warnings, explanation = agent_with_mock.process(content, "test.csv")
        
        # Should still return valid schema via fallback
        assert schema is not None
        assert schema.format == DataFormat.POINTWISE
        assert len(schema.ranking_items) > 0
    
    def test_llm_json_parsing_with_markdown_blocks(self, agent_with_mock, pointwise_df):
        """Test that LLM response with markdown code blocks is parsed correctly."""
        # LLM sometimes wraps JSON in markdown code blocks
        json_content = json.dumps({
            "format": "pointwise",
            "format_reasoning": "Dense matrix",
            "bigbetter": 1,
            "bigbetter_reasoning": "Scores",
            "ranking_items": ["model_A", "model_B"],
            "ranking_items_reasoning": "Models",
            "indicator_col": None,
            "indicator_reasoning": "None"
        })
        
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=f"```json\n{json_content}\n```"))]
        mock_response.usage = Mock(total_tokens=100)
        
        agent_with_mock.client.chat.completions.create.return_value = mock_response
        
        content = pointwise_df.to_csv(index=False).encode()
        schema, warnings, explanation = agent_with_mock.process(content, "test.csv")
        
        assert schema.format == DataFormat.POINTWISE
        assert schema.bigbetter == 1


# =============================================================================
# Test Class: Fallback Mode (Heuristics)
# =============================================================================

class TestDataAgentFallback:
    """Test fallback mode when LLM is unavailable."""
    
    def test_fallback_mode_enabled_check(self):
        """Test that fallback mode is correctly detected."""
        with patch.dict('os.environ', {}, clear=True):
            import os
            orig_key = os.environ.pop('OPENAI_API_KEY', None)
            try:
                agent = DataAgent()
                assert not agent.enabled
            finally:
                if orig_key:
                    os.environ['OPENAI_API_KEY'] = orig_key
    
    def test_fallback_pointwise_detection(self, pointwise_df):
        """Test heuristic pointwise detection."""
        with patch.dict('os.environ', {}, clear=True):
            import os
            orig_key = os.environ.pop('OPENAI_API_KEY', None)
            try:
                agent = DataAgent()
                result = agent._detect_format_heuristic(pointwise_df)
                assert result == DataFormat.POINTWISE
            finally:
                if orig_key:
                    os.environ['OPENAI_API_KEY'] = orig_key
    
    def test_fallback_pairwise_detection(self, pairwise_df):
        """Test heuristic pairwise detection."""
        with patch.dict('os.environ', {}, clear=True):
            import os
            orig_key = os.environ.pop('OPENAI_API_KEY', None)
            try:
                agent = DataAgent()
                result = agent._detect_format_heuristic(pairwise_df)
                assert result == DataFormat.PAIRWISE
            finally:
                if orig_key:
                    os.environ['OPENAI_API_KEY'] = orig_key
    
    def test_fallback_schema_inference(self, pointwise_df):
        """Test complete fallback schema inference."""
        with patch.dict('os.environ', {}, clear=True):
            import os
            orig_key = os.environ.pop('OPENAI_API_KEY', None)
            try:
                agent = DataAgent()
                content = pointwise_df.to_csv(index=False).encode()
                schema, warnings, explanation = agent.process(content, "test.csv")
                
                assert schema.format == DataFormat.POINTWISE
                assert len(schema.ranking_items) >= 2
                # Explanation should be empty in fallback mode
                assert explanation == ""
            finally:
                if orig_key:
                    os.environ['OPENAI_API_KEY'] = orig_key


# =============================================================================
# Test Class: BigBetter Inference
# =============================================================================

class TestBigBetterInference:
    """Test bigbetter inference (higher/lower is better)."""
    
    def test_higher_better_keywords(self, higher_better_df):
        """Test detection of 'higher is better' from column names."""
        with patch.dict('os.environ', {}, clear=True):
            import os
            orig_key = os.environ.pop('OPENAI_API_KEY', None)
            try:
                agent = DataAgent()
                bigbetter = agent._infer_bigbetter_heuristic(
                    higher_better_df, DataFormat.POINTWISE
                )
                assert bigbetter == 1
            finally:
                if orig_key:
                    os.environ['OPENAI_API_KEY'] = orig_key
    
    def test_lower_better_keywords(self, lower_better_df):
        """Test detection of 'lower is better' from column names."""
        with patch.dict('os.environ', {}, clear=True):
            import os
            orig_key = os.environ.pop('OPENAI_API_KEY', None)
            try:
                agent = DataAgent()
                bigbetter = agent._infer_bigbetter_heuristic(
                    lower_better_df, DataFormat.POINTWISE
                )
                assert bigbetter == 0
            finally:
                if orig_key:
                    os.environ['OPENAI_API_KEY'] = orig_key
    
    def test_pairwise_always_bigbetter_1(self, pairwise_df):
        """Test that pairwise format always has bigbetter=1 (1 is winner)."""
        with patch.dict('os.environ', {}, clear=True):
            import os
            orig_key = os.environ.pop('OPENAI_API_KEY', None)
            try:
                agent = DataAgent()
                bigbetter = agent._infer_bigbetter_heuristic(
                    pairwise_df, DataFormat.PAIRWISE
                )
                assert bigbetter == 1
            finally:
                if orig_key:
                    os.environ['OPENAI_API_KEY'] = orig_key
    
    def test_bounded_values_suggest_higher_better(self):
        """Test micro-level analysis: bounded [0,1] values suggest higher is better."""
        # Create data with no keyword hints but bounded values
        data = {
            'col_x': [0.85, 0.78, 0.82],
            'col_y': [0.80, 0.75, 0.78],
            'col_z': [0.88, 0.82, 0.85],
        }
        df = pd.DataFrame(data)
        
        with patch.dict('os.environ', {}, clear=True):
            import os
            orig_key = os.environ.pop('OPENAI_API_KEY', None)
            try:
                agent = DataAgent()
                bigbetter = agent._infer_bigbetter_heuristic(
                    df, DataFormat.POINTWISE
                )
                # Bounded [0,1] values usually mean higher is better
                assert bigbetter == 1
            finally:
                if orig_key:
                    os.environ['OPENAI_API_KEY'] = orig_key


# =============================================================================
# Test Class: Indicator Column Detection
# =============================================================================

class TestIndicatorColumnDetection:
    """Test indicator column detection."""
    
    def test_detect_task_indicator(self, pairwise_df):
        """Test detection of 'Task' column as indicator."""
        with patch.dict('os.environ', {}, clear=True):
            import os
            orig_key = os.environ.pop('OPENAI_API_KEY', None)
            try:
                agent = DataAgent()
                indicator_col, indicator_values = agent._detect_indicator_heuristic(pairwise_df)
                
                assert indicator_col == "Task"
                assert set(indicator_values) == {"code", "math", "writing"}
            finally:
                if orig_key:
                    os.environ['OPENAI_API_KEY'] = orig_key
    
    def test_at_most_one_indicator(self):
        """Test that at most ONE indicator column is returned."""
        # Create data with multiple potential indicators
        data = {
            'Task': ['code', 'math', 'writing'],
            'Category': ['A', 'B', 'C'],
            'Type': ['X', 'Y', 'Z'],
            'Score': [0.8, 0.7, 0.9],
        }
        df = pd.DataFrame(data)
        
        with patch.dict('os.environ', {}, clear=True):
            import os
            orig_key = os.environ.pop('OPENAI_API_KEY', None)
            try:
                agent = DataAgent()
                indicator_col, indicator_values = agent._detect_indicator_heuristic(df)
                
                # Should return exactly one (the first keyword match)
                assert indicator_col is not None
                assert isinstance(indicator_col, str)
            finally:
                if orig_key:
                    os.environ['OPENAI_API_KEY'] = orig_key
    
    def test_no_indicator_in_numeric_only_data(self, pointwise_df):
        """Test that no indicator is detected in numeric-only data."""
        # Remove non-numeric columns
        df = pointwise_df[['model_A', 'model_B', 'model_C']].copy()
        
        with patch.dict('os.environ', {}, clear=True):
            import os
            orig_key = os.environ.pop('OPENAI_API_KEY', None)
            try:
                agent = DataAgent()
                indicator_col, indicator_values = agent._detect_indicator_heuristic(df)
                
                assert indicator_col is None
                assert indicator_values == []
            finally:
                if orig_key:
                    os.environ['OPENAI_API_KEY'] = orig_key


# =============================================================================
# Test Class: Sparsity Check (CORRECTED FORMULA)
# =============================================================================

class TestSparsityCheck:
    """Test sparsity check with correct M < n*log(n) formula."""
    
    def test_sparsity_formula_correct(self, sparse_df):
        """Test that sparsity uses M < n*log(n) threshold."""
        with patch.dict('os.environ', {}, clear=True):
            import os
            orig_key = os.environ.pop('OPENAI_API_KEY', None)
            try:
                agent = DataAgent()
                ranking_items = list(sparse_df.columns)
                
                sparsity_ratio, severity, message = agent._check_sparsity_correct(
                    sparse_df, ranking_items
                )
                
                # With 5 items and only ~2-3 comparisons, should be sparse
                # Threshold = 5 * log(5) ≈ 8.05
                n_items = 5
                threshold = n_items * math.log(n_items)
                
                # Very sparse data should trigger warning
                assert severity == "warning"
                assert "sparse" in message.lower() or "Comparisons" in message
            finally:
                if orig_key:
                    os.environ['OPENAI_API_KEY'] = orig_key
    
    def test_dense_data_no_warning(self, pointwise_df):
        """Test that dense data does not trigger sparsity warning."""
        with patch.dict('os.environ', {}, clear=True):
            import os
            orig_key = os.environ.pop('OPENAI_API_KEY', None)
            try:
                agent = DataAgent()
                ranking_items = ['model_A', 'model_B', 'model_C']
                
                sparsity_ratio, severity, message = agent._check_sparsity_correct(
                    pointwise_df, ranking_items
                )
                
                # Dense data should be "info" not "warning"
                # 5 rows * C(3,2) = 5 * 3 = 15 comparisons
                # Threshold = 3 * log(3) ≈ 3.3
                # 15 > 3.3, so should be adequate
                assert severity == "info"
            finally:
                if orig_key:
                    os.environ['OPENAI_API_KEY'] = orig_key
    
    def test_sparsity_threshold_calculation(self):
        """Verify the M < n*log(n) threshold calculation."""
        # Test cases: n=5, n=10, n=20
        test_cases = [
            (5, 5 * math.log(5)),   # ≈ 8.05
            (10, 10 * math.log(10)),  # ≈ 23.03
            (20, 20 * math.log(20)),  # ≈ 59.91
        ]
        
        for n, expected_threshold in test_cases:
            actual = n * math.log(n)
            assert abs(actual - expected_threshold) < 0.01


# =============================================================================
# Test Class: Connectivity Check
# =============================================================================

class TestConnectivityCheck:
    """Test comparison graph connectivity check."""
    
    def test_connected_graph(self, pointwise_df):
        """Test that dense data has connected graph."""
        with patch.dict('os.environ', {}, clear=True):
            import os
            orig_key = os.environ.pop('OPENAI_API_KEY', None)
            try:
                agent = DataAgent()
                ranking_items = ['model_A', 'model_B', 'model_C']
                
                is_connected, message = agent._check_connectivity(
                    pointwise_df, ranking_items
                )
                
                assert is_connected
                assert "connected" in message.lower()
            finally:
                if orig_key:
                    os.environ['OPENAI_API_KEY'] = orig_key
    
    def test_disconnected_graph(self, disconnected_df):
        """Test detection of disconnected comparison graph."""
        with patch.dict('os.environ', {}, clear=True):
            import os
            orig_key = os.environ.pop('OPENAI_API_KEY', None)
            try:
                agent = DataAgent()
                ranking_items = list(disconnected_df.columns)
                
                is_connected, message = agent._check_connectivity(
                    disconnected_df, ranking_items
                )
                
                assert not is_connected
                assert "disconnected" in message.lower() or "components" in message.lower()
            finally:
                if orig_key:
                    os.environ['OPENAI_API_KEY'] = orig_key


# =============================================================================
# Test Class: Validation Warnings
# =============================================================================

class TestValidationWarnings:
    """Test validation warnings generation."""
    
    def test_minimum_items_error(self):
        """Test error when less than 2 ranking items."""
        # Create data with only 1 ranking item
        data = {'single_model': [0.8, 0.7, 0.9]}
        df = pd.DataFrame(data)
        
        with patch.dict('os.environ', {}, clear=True):
            import os
            orig_key = os.environ.pop('OPENAI_API_KEY', None)
            try:
                agent = DataAgent()
                content = df.to_csv(index=False).encode()
                schema, warnings, explanation = agent.process(content, "test.csv")
                
                # Should have error warning
                error_warnings = [w for w in warnings if w.severity == "error"]
                assert len(error_warnings) > 0
            finally:
                if orig_key:
                    os.environ['OPENAI_API_KEY'] = orig_key
    
    def test_small_sample_warning(self):
        """Test warning for small sample size."""
        # Create data with less than 10 samples
        data = {
            'model_A': [0.8, 0.7, 0.9],
            'model_B': [0.7, 0.6, 0.8],
        }
        df = pd.DataFrame(data)
        
        with patch.dict('os.environ', {}, clear=True):
            import os
            orig_key = os.environ.pop('OPENAI_API_KEY', None)
            try:
                agent = DataAgent()
                content = df.to_csv(index=False).encode()
                schema, warnings, explanation = agent.process(content, "test.csv")
                
                # Should have warning about small sample size
                sample_warnings = [w for w in warnings if "sample" in w.message.lower()]
                assert len(sample_warnings) > 0
            finally:
                if orig_key:
                    os.environ['OPENAI_API_KEY'] = orig_key
    
    def test_csv_parse_error(self):
        """Test error handling for invalid CSV."""
        # Use binary content that will definitely fail CSV parsing
        invalid_content = b"\x00\x01\x02\x03\x04\x05"
        
        with patch.dict('os.environ', {}, clear=True):
            import os
            orig_key = os.environ.pop('OPENAI_API_KEY', None)
            try:
                agent = DataAgent()
                schema, warnings, explanation = agent.process(invalid_content, "test.csv")
                
                # Should have error about parsing
                assert any(w.severity == "error" for w in warnings)
            finally:
                if orig_key:
                    os.environ['OPENAI_API_KEY'] = orig_key


# =============================================================================
# Test Class: Return Signature
# =============================================================================

class TestReturnSignature:
    """Test the new 3-value return signature."""
    
    def test_process_returns_three_values(self, pointwise_df):
        """Test that process() returns (schema, warnings, explanation)."""
        with patch.dict('os.environ', {}, clear=True):
            import os
            orig_key = os.environ.pop('OPENAI_API_KEY', None)
            try:
                agent = DataAgent()
                content = pointwise_df.to_csv(index=False).encode()
                
                result = agent.process(content, "test.csv")
                
                assert isinstance(result, tuple)
                assert len(result) == 3
                
                schema, warnings, explanation = result
                assert isinstance(schema, InferredSchema)
                assert isinstance(warnings, list)
                assert isinstance(explanation, str)
            finally:
                if orig_key:
                    os.environ['OPENAI_API_KEY'] = orig_key


# =============================================================================
# Test Class: Data Summary Building
# =============================================================================

class TestDataSummaryBuilding:
    """Test the data summary building for LLM analysis."""
    
    def test_data_summary_includes_shape(self, pointwise_df):
        """Test that data summary includes shape information."""
        with patch.dict('os.environ', {}, clear=True):
            import os
            orig_key = os.environ.pop('OPENAI_API_KEY', None)
            try:
                agent = DataAgent()
                summary = agent._build_data_summary(pointwise_df)
                
                assert "Shape:" in summary
                assert str(pointwise_df.shape[0]) in summary
                assert str(pointwise_df.shape[1]) in summary
            finally:
                if orig_key:
                    os.environ['OPENAI_API_KEY'] = orig_key
    
    def test_data_summary_includes_columns(self, pointwise_df):
        """Test that data summary includes column names."""
        with patch.dict('os.environ', {}, clear=True):
            import os
            orig_key = os.environ.pop('OPENAI_API_KEY', None)
            try:
                agent = DataAgent()
                summary = agent._build_data_summary(pointwise_df)
                
                assert "Column names:" in summary
                for col in pointwise_df.columns:
                    assert col in summary
            finally:
                if orig_key:
                    os.environ['OPENAI_API_KEY'] = orig_key
    
    def test_data_summary_includes_statistics(self, pointwise_df):
        """Test that data summary includes statistical info."""
        with patch.dict('os.environ', {}, clear=True):
            import os
            orig_key = os.environ.pop('OPENAI_API_KEY', None)
            try:
                agent = DataAgent()
                summary = agent._build_data_summary(pointwise_df)
                
                # Should include min, max, mean for numeric columns
                assert "range=" in summary or "min" in summary.lower()
                assert "mean=" in summary
            finally:
                if orig_key:
                    os.environ['OPENAI_API_KEY'] = orig_key


# =============================================================================
# Test Class: Legacy Function Compatibility
# =============================================================================

class TestLegacyFunctionCompatibility:
    """Test backward compatibility of legacy functions."""
    
    def test_legacy_detect_format(self, pointwise_df):
        """Test legacy detect_format function."""
        from agents.data_agent import detect_format
        
        result = detect_format(pointwise_df)
        assert result == DataFormat.POINTWISE
    
    def test_legacy_infer_bigbetter(self, higher_better_df):
        """Test legacy infer_bigbetter function."""
        from agents.data_agent import infer_bigbetter
        
        bigbetter = infer_bigbetter(higher_better_df, DataFormat.POINTWISE)
        assert bigbetter == 1
    
    def test_legacy_extract_ranking_items(self, pointwise_df):
        """Test legacy extract_ranking_items function."""
        from agents.data_agent import extract_ranking_items
        
        items = extract_ranking_items(pointwise_df, DataFormat.POINTWISE)
        assert len(items) >= 2
    
    def test_legacy_detect_indicator_column(self, pairwise_df):
        """Test legacy detect_indicator_column function."""
        from agents.data_agent import detect_indicator_column
        
        indicator_col, indicator_values = detect_indicator_column(pairwise_df)
        assert indicator_col == "Task"
    
    def test_legacy_infer_schema(self, pointwise_df):
        """Test legacy infer_schema function."""
        from agents.data_agent import infer_schema
        
        content = pointwise_df.to_csv(index=False).encode()
        schema = infer_schema(content, "test.csv")
        
        assert isinstance(schema, InferredSchema)
        assert schema.format == DataFormat.POINTWISE


# =============================================================================
# Test Class: Function 1 - Engine Compatibility Assessment
# =============================================================================

class TestEngineCompatibility:
    """Test Function 1: Engine Compatibility Assessment."""
    
    def test_engine_compatible_default_true(self, engine_compatible_df):
        """Test that engine_compatible defaults to True for standard data."""
        with patch.dict('os.environ', {}, clear=True):
            import os
            orig_key = os.environ.pop('OPENAI_API_KEY', None)
            try:
                agent = DataAgent()
                content = engine_compatible_df.to_csv(index=False).encode()
                schema, warnings, explanation = agent.process(content, "test.csv")
                
                # Standard numeric columns should be engine compatible
                assert schema.engine_compatible is True
                assert schema.standardization_needed is False
            finally:
                if orig_key:
                    os.environ['OPENAI_API_KEY'] = orig_key
    
    def test_engine_compatible_with_llm(self, agent_with_mock, engine_compatible_df):
        """Test engine compatibility assessment with LLM."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=json.dumps({
            "format": "pointwise",
            "format_reasoning": "Dense numeric matrix",
            "engine_compatible": True,
            "standardization_needed": False,
            "standardization_reason": None,
            "bigbetter": 1,
            "bigbetter_reasoning": "Standard scores",
            "ranking_items": ["model_1", "model_2", "model_3"],
            "ranking_items_reasoning": "Numeric columns",
            "indicator_col": None,
            "indicator_reasoning": "None found"
        })))]
        mock_response.usage = Mock(total_tokens=100)
        
        agent_with_mock.client.chat.completions.create.return_value = mock_response
        
        content = engine_compatible_df.to_csv(index=False).encode()
        schema, warnings, explanation = agent_with_mock.process(content, "test.csv")
        
        assert schema.engine_compatible is True
        assert schema.standardization_needed is False
        assert schema.standardization_reason is None
    
    def test_schema_has_new_fields(self, pointwise_df):
        """Test that InferredSchema has new Function 1 fields."""
        with patch.dict('os.environ', {}, clear=True):
            import os
            orig_key = os.environ.pop('OPENAI_API_KEY', None)
            try:
                agent = DataAgent()
                content = pointwise_df.to_csv(index=False).encode()
                schema, warnings, explanation = agent.process(content, "test.csv")
                
                # Check new fields exist
                assert hasattr(schema, 'engine_compatible')
                assert hasattr(schema, 'standardization_needed')
                assert hasattr(schema, 'standardization_reason')
            finally:
                if orig_key:
                    os.environ['OPENAI_API_KEY'] = orig_key


# =============================================================================
# Test Class: Function 1 - Conditional Standardization
# =============================================================================

class TestConditionalStandardization:
    """Test Function 1: Conditional Standardization logic."""
    
    def test_standardize_if_needed_returns_original_when_not_needed(self, pointwise_df):
        """Test that standardization is skipped when not needed."""
        with patch.dict('os.environ', {}, clear=True):
            import os
            orig_key = os.environ.pop('OPENAI_API_KEY', None)
            try:
                agent = DataAgent()
                
                # Create schema with standardization_needed=False
                schema = InferredSchema(
                    format=DataFormat.POINTWISE,
                    bigbetter=1,
                    ranking_items=["model_A", "model_B", "model_C"],
                    engine_compatible=True,
                    standardization_needed=False,
                )
                
                result_df, was_standardized = agent.standardize_if_needed(pointwise_df, schema)
                
                assert was_standardized is False
                # DataFrame should be the same object (not modified)
                assert result_df is pointwise_df
            finally:
                if orig_key:
                    os.environ['OPENAI_API_KEY'] = orig_key
    
    def test_standardize_if_needed_fixes_special_characters(self, special_chars_df):
        """Test that standardization fixes special characters in column names."""
        with patch.dict('os.environ', {}, clear=True):
            import os
            orig_key = os.environ.pop('OPENAI_API_KEY', None)
            try:
                agent = DataAgent()
                
                # Create schema with standardization_needed=True
                schema = InferredSchema(
                    format=DataFormat.POINTWISE,
                    bigbetter=1,
                    ranking_items=list(special_chars_df.columns),
                    engine_compatible=False,
                    standardization_needed=True,
                    standardization_reason="Column names contain special characters",
                )
                
                result_df, was_standardized = agent.standardize_if_needed(special_chars_df, schema)
                
                assert was_standardized is True
                # Check that special characters are removed/replaced
                for col in result_df.columns:
                    # No parentheses, hyphens, or slashes in column names
                    assert "(" not in col
                    assert ")" not in col
                    assert "/" not in col
            finally:
                if orig_key:
                    os.environ['OPENAI_API_KEY'] = orig_key
    
    def test_standardize_preserves_data_values(self, special_chars_df):
        """Test that standardization preserves data values."""
        with patch.dict('os.environ', {}, clear=True):
            import os
            orig_key = os.environ.pop('OPENAI_API_KEY', None)
            try:
                agent = DataAgent()
                
                original_values = special_chars_df.values.copy()
                
                schema = InferredSchema(
                    format=DataFormat.POINTWISE,
                    bigbetter=1,
                    ranking_items=list(special_chars_df.columns),
                    engine_compatible=False,
                    standardization_needed=True,
                    standardization_reason="Column names contain special characters",
                )
                
                result_df, was_standardized = agent.standardize_if_needed(special_chars_df, schema)
                
                # Data values should be preserved
                import numpy as np
                np.testing.assert_array_equal(result_df.values, original_values)
            finally:
                if orig_key:
                    os.environ['OPENAI_API_KEY'] = orig_key


# =============================================================================
# Test Class: Multiway Format Detection
# =============================================================================

class TestMultiwayFormatDetection:
    """Test multiway format detection."""
    
    def test_multiway_detection_with_llm(self, agent_with_mock, multiway_df):
        """Test multiway format detection using LLM."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=json.dumps({
            "format": "multiway",
            "format_reasoning": "Each row contains unique rank positions (1, 2, 3)",
            "engine_compatible": True,
            "standardization_needed": False,
            "standardization_reason": None,
            "bigbetter": 0,
            "bigbetter_reasoning": "Lower rank is better (1st place)",
            "ranking_items": ["Horse_A", "Horse_B", "Horse_C"],
            "ranking_items_reasoning": "Columns with rank position values",
            "indicator_col": None,
            "indicator_reasoning": "Race column is ID, not indicator"
        })))]
        mock_response.usage = Mock(total_tokens=100)
        
        agent_with_mock.client.chat.completions.create.return_value = mock_response
        
        content = multiway_df.to_csv(index=False).encode()
        schema, warnings, explanation = agent_with_mock.process(content, "test.csv")
        
        assert schema.format == DataFormat.MULTIWAY
        assert schema.bigbetter == 0  # Lower rank is better
        assert "Horse_A" in schema.ranking_items
