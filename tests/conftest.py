"""
Pytest configuration and shared fixtures.
"""

import sys
from pathlib import Path

import pytest
import pandas as pd

# Add src/api to path for imports
API_PATH = Path(__file__).parent.parent / "src" / "api"
sys.path.insert(0, str(API_PATH))


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def example_pointwise_data():
    """Load example pointwise data."""
    data_path = Path(__file__).parent.parent / "data" / "examples" / "example_data_pointwise.csv"
    return data_path.read_bytes()


@pytest.fixture
def example_pairwise_data():
    """Load example pairwise data."""
    data_path = Path(__file__).parent.parent / "data" / "examples" / "example_data_pairwise.csv"
    return data_path.read_bytes()


@pytest.fixture
def pointwise_df():
    """Create a sample pointwise DataFrame for testing."""
    data = {
        'sample_id': ['s1', 's2', 's3', 's4', 's5'],
        'model_A': [0.85, 0.78, 0.92, 0.88, 0.75],
        'model_B': [0.72, 0.81, 0.79, 0.85, 0.70],
        'model_C': [0.68, 0.75, 0.71, 0.80, 0.65],
        'description': ['desc1', 'desc2', 'desc3', 'desc4', 'desc5'],
    }
    return pd.DataFrame(data)


@pytest.fixture
def pairwise_df():
    """Create a sample pairwise DataFrame for testing."""
    data = {
        'Task': ['code', 'math', 'writing', 'code', 'math'],
        'Model_A': [1, 0, None, 1, None],
        'Model_B': [0, 1, 1, None, 0],
        'Model_C': [None, None, 0, 0, 1],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sparse_df():
    """Create a sparse DataFrame for sparsity testing."""
    # Very sparse: only 2-3 comparisons total
    data = {
        'model_A': [0.8, None, None, None, None],
        'model_B': [0.7, 0.75, None, None, None],
        'model_C': [None, None, 0.65, None, None],
        'model_D': [None, None, None, 0.7, None],
        'model_E': [None, None, None, None, 0.6],
    }
    return pd.DataFrame(data)


@pytest.fixture
def disconnected_df():
    """Create a DataFrame with disconnected comparison graph."""
    # Model_A and Model_B form one component
    # Model_C and Model_D form another component
    data = {
        'Model_A': [0.8, 0.75, None, None],
        'Model_B': [0.7, 0.72, None, None],
        'Model_C': [None, None, 0.65, 0.68],
        'Model_D': [None, None, 0.60, 0.62],
    }
    return pd.DataFrame(data)


@pytest.fixture
def lower_better_df():
    """Create a DataFrame where lower is better (error/loss metrics)."""
    data = {
        'sample_id': ['s1', 's2', 's3'],
        'error_rate': [0.15, 0.22, 0.18],
        'loss_value': [0.32, 0.45, 0.38],
        'latency_ms': [120, 180, 150],
    }
    return pd.DataFrame(data)


@pytest.fixture
def higher_better_df():
    """Create a DataFrame where higher is better (score/accuracy metrics)."""
    data = {
        'sample_id': ['s1', 's2', 's3'],
        'accuracy_score': [0.85, 0.78, 0.82],
        'f1_score': [0.80, 0.75, 0.78],
        'precision': [0.88, 0.82, 0.85],
    }
    return pd.DataFrame(data)


@pytest.fixture
def multiway_candidate_df():
    """Create a DataFrame that might look like multiway format."""
    # Placeholder for future multiway detection tests
    data = {
        'Race_ID': [1, 2, 3],
        'Rank_1': ['A', 'B', 'C'],
        'Rank_2': ['B', 'C', 'A'],
        'Rank_3': ['C', 'A', 'B'],
    }
    return pd.DataFrame(data)
