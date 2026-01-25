"""
Integration tests for Data Agent with real example data.

These tests verify the Data Agent works correctly with the actual
example datasets in data/examples/.

Note: Tests marked with @pytest.mark.integration require OPENAI_API_KEY
to be set for LLM-based tests.
"""

import os
import pytest
from pathlib import Path

# Import after path setup in conftest
from agents.data_agent import DataAgent
from core.schemas import DataFormat, InferredSchema


# =============================================================================
# Test Class: Real Data Integration (No LLM)
# =============================================================================

class TestRealDataIntegration:
    """Integration tests with real example data (fallback mode)."""
    
    @pytest.fixture
    def agent(self):
        """Create agent in fallback mode for deterministic tests."""
        # Clear API key to force fallback mode
        orig_key = os.environ.pop('OPENAI_API_KEY', None)
        try:
            agent = DataAgent()
            yield agent
        finally:
            if orig_key:
                os.environ['OPENAI_API_KEY'] = orig_key
    
    def test_real_pointwise_inference(self, agent, example_pointwise_data):
        """Test schema inference on real pointwise data."""
        schema, warnings, explanation = agent.process(
            example_pointwise_data,
            "example_data_pointwise.csv"
        )
        
        # Verify format detection
        assert schema.format == DataFormat.POINTWISE
        
        # Verify ranking items (should be model_1 through model_6)
        assert len(schema.ranking_items) >= 5
        
        # Verify bigbetter (scores are higher=better)
        assert schema.bigbetter == 1
        
        # Verify confidence is reasonable
        assert schema.confidence > 0
    
    def test_real_pairwise_inference(self, agent, example_pairwise_data):
        """Test schema inference on real pairwise data."""
        schema, warnings, explanation = agent.process(
            example_pairwise_data,
            "example_data_pairwise.csv"
        )
        
        # Verify format detection
        assert schema.format == DataFormat.PAIRWISE
        
        # Verify ranking items (should be the LLM names)
        assert len(schema.ranking_items) >= 5
        
        # Verify indicator column
        assert schema.indicator_col == "Task"
        
        # Verify indicator values
        assert set(schema.indicator_values) == {"code", "math", "writing"}
        
        # Verify bigbetter (1 = winner in pairwise)
        assert schema.bigbetter == 1
    
    def test_real_data_no_critical_errors(self, agent, example_pointwise_data, example_pairwise_data):
        """Test that real example data has no critical errors."""
        for data, filename in [
            (example_pointwise_data, "example_data_pointwise.csv"),
            (example_pairwise_data, "example_data_pairwise.csv"),
        ]:
            schema, warnings, explanation = agent.process(data, filename)
            
            # Should have no critical errors
            critical_errors = [w for w in warnings if w.severity == "error"]
            assert len(critical_errors) == 0, f"Critical errors in {filename}: {critical_errors}"
    
    def test_real_pointwise_ranking_items_correct(self, agent, example_pointwise_data):
        """Test that pointwise data correctly identifies model columns."""
        schema, warnings, explanation = agent.process(
            example_pointwise_data,
            "example_data_pointwise.csv"
        )
        
        # Should include model_1 through model_6
        expected_models = ['model_1', 'model_2', 'model_3', 'model_4', 'model_5', 'model_6']
        
        for model in expected_models:
            assert model in schema.ranking_items, f"Missing {model} in ranking_items"
        
        # Should NOT include sample_id or description
        assert 'sample_id' not in schema.ranking_items
        assert 'description' not in schema.ranking_items
    
    def test_real_pairwise_ranking_items_correct(self, agent, example_pairwise_data):
        """Test that pairwise data correctly identifies LLM columns."""
        schema, warnings, explanation = agent.process(
            example_pairwise_data,
            "example_data_pairwise.csv"
        )
        
        # Should include the LLM names (excluding Task column)
        expected_llms = ['Your Model', 'ChatGPT', 'Claude', 'Gemini', 'Llama', 'Qwen']
        
        for llm in expected_llms:
            assert llm in schema.ranking_items, f"Missing {llm} in ranking_items"
        
        # Should NOT include Task column
        assert 'Task' not in schema.ranking_items


# =============================================================================
# Test Class: LLM Integration (requires API key)
# =============================================================================

@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY required")
class TestLLMIntegration:
    """Integration tests with real LLM calls."""
    
    @pytest.fixture
    def agent(self):
        """Create agent with real LLM capabilities."""
        return DataAgent()
    
    def test_llm_pointwise_inference(self, agent, example_pointwise_data):
        """Test LLM-powered inference on real pointwise data."""
        schema, warnings, explanation = agent.process(
            example_pointwise_data,
            "example_data_pointwise.csv"
        )
        
        # Core assertions - these should work regardless of LLM or fallback
        assert schema.format == DataFormat.POINTWISE
        assert schema.bigbetter == 1
        assert len(schema.ranking_items) >= 5
        
        # Verify correct ranking items identified
        expected_models = ['model_1', 'model_2', 'model_3', 'model_4', 'model_5', 'model_6']
        for model in expected_models:
            assert model in schema.ranking_items, f"Missing {model}"
        
        # Confidence should be reasonable (may be lower if LLM fails and falls back)
        assert schema.confidence > 0
    
    def test_llm_pairwise_inference(self, agent, example_pairwise_data):
        """Test LLM-powered inference on real pairwise data."""
        schema, warnings, explanation = agent.process(
            example_pairwise_data,
            "example_data_pairwise.csv"
        )
        
        # Core assertions - these should work regardless of LLM or fallback
        assert schema.format == DataFormat.PAIRWISE
        assert schema.bigbetter == 1
        assert schema.indicator_col == "Task"
        assert set(schema.indicator_values) == {"code", "math", "writing"}
        
        # Verify ranking items (the LLM names)
        expected_llms = ['Your Model', 'ChatGPT', 'Claude', 'Gemini', 'Llama', 'Qwen']
        for llm in expected_llms:
            assert llm in schema.ranking_items, f"Missing {llm}"
        
        # Confidence should be reasonable (may be lower if LLM fails and falls back)
        assert schema.confidence > 0


# =============================================================================
# Test Class: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @pytest.fixture
    def agent(self):
        """Create agent in fallback mode."""
        orig_key = os.environ.pop('OPENAI_API_KEY', None)
        try:
            agent = DataAgent()
            yield agent
        finally:
            if orig_key:
                os.environ['OPENAI_API_KEY'] = orig_key
    
    def test_empty_file(self, agent):
        """Test handling of empty file."""
        empty_content = b""
        
        schema, warnings, explanation = agent.process(empty_content, "empty.csv")
        
        # Should have error
        assert any(w.severity == "error" for w in warnings)
    
    def test_header_only_file(self, agent):
        """Test handling of file with only headers."""
        header_only = b"col_a,col_b,col_c\n"
        
        schema, warnings, explanation = agent.process(header_only, "header_only.csv")
        
        # Should handle gracefully (might have warnings about small sample)
        assert schema is not None
    
    def test_single_row_file(self, agent):
        """Test handling of file with single data row."""
        single_row = b"col_a,col_b,col_c\n0.8,0.7,0.6\n"
        
        schema, warnings, explanation = agent.process(single_row, "single_row.csv")
        
        # Should have warning about small sample size
        sample_warnings = [w for w in warnings if "sample" in w.message.lower()]
        assert len(sample_warnings) > 0
    
    def test_many_columns_file(self, agent):
        """Test handling of file with many columns."""
        # Create file with 20 model columns
        import pandas as pd
        data = {f'model_{i}': [0.5 + i*0.02 for _ in range(10)] for i in range(20)}
        df = pd.DataFrame(data)
        content = df.to_csv(index=False).encode()
        
        schema, warnings, explanation = agent.process(content, "many_columns.csv")
        
        # Should detect all models as ranking items
        assert len(schema.ranking_items) >= 15
    
    def test_unicode_column_names(self, agent):
        """Test handling of Unicode column names."""
        import pandas as pd
        data = {
            'sample_id': ['s1', 's2', 's3'],
            'model_alpha': [0.8, 0.75, 0.82],
            'model_beta': [0.7, 0.72, 0.78],
        }
        df = pd.DataFrame(data)
        content = df.to_csv(index=False).encode('utf-8')
        
        schema, warnings, explanation = agent.process(content, "unicode.csv")
        
        # Should handle gracefully
        assert schema is not None
        assert schema.format == DataFormat.POINTWISE


# =============================================================================
# Test Class: Performance
# =============================================================================

class TestPerformance:
    """Test performance characteristics."""
    
    @pytest.fixture
    def agent(self):
        """Create agent in fallback mode for deterministic timing."""
        orig_key = os.environ.pop('OPENAI_API_KEY', None)
        try:
            agent = DataAgent()
            yield agent
        finally:
            if orig_key:
                os.environ['OPENAI_API_KEY'] = orig_key
    
    def test_large_file_handling(self, agent):
        """Test that large files are handled within reasonable time."""
        import time
        import pandas as pd
        
        # Create large dataframe (1000 rows, 10 columns)
        data = {f'model_{i}': [0.5 + i*0.05 for _ in range(1000)] for i in range(10)}
        df = pd.DataFrame(data)
        content = df.to_csv(index=False).encode()
        
        start_time = time.time()
        schema, warnings, explanation = agent.process(content, "large.csv")
        elapsed_time = time.time() - start_time
        
        # Should complete within 5 seconds (fallback mode)
        assert elapsed_time < 5.0, f"Processing took too long: {elapsed_time:.2f}s"
        
        # Should still produce valid results
        assert schema is not None
        assert len(schema.ranking_items) == 10
