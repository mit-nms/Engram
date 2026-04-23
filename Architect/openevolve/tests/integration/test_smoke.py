"""
Smoke tests for integration testing - fast tests that validate basic functionality
These run in CI to ensure core components work without requiring slow LLM calls
"""

import pytest
import tempfile
from pathlib import Path

from openevolve import run_evolution, evolve_function, evolve_code
from openevolve.config import Config, LLMModelConfig


class TestSmoke:
    """Fast smoke tests for CI"""

    def test_library_api_validation(self):
        """Test library API gives proper error messages when not configured"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
# EVOLVE-BLOCK-START
def solve(x):
    return x * 2
# EVOLVE-BLOCK-END
""")
            program_file = f.name
        
        def simple_evaluator(path):
            return {"score": 0.5, "combined_score": 0.5}
        
        # Test that library API properly validates LLM configuration
        with pytest.raises(ValueError, match="No LLM models configured"):
            run_evolution(
                initial_program=program_file,
                evaluator=simple_evaluator,
                iterations=1
            )
        
        # Clean up
        Path(program_file).unlink()

    def test_config_validation(self):
        """Test configuration validation works"""
        config = Config()
        
        # Test that default config has proper structure
        assert hasattr(config, 'llm')
        assert hasattr(config, 'database')
        assert hasattr(config, 'evaluator')
        assert hasattr(config, 'prompt')
        
        # Test defaults
        assert config.max_iterations > 0
        assert config.database.in_memory is True
        assert config.llm.retries >= 0

    def test_llm_config_creation(self):
        """Test that LLM configuration can be created properly"""
        config = Config()
        
        # Test adding a model configuration
        config.llm.models = [
            LLMModelConfig(
                name="test-model",
                api_key="test-key", 
                api_base="http://localhost:8000/v1",
                weight=1.0,
                timeout=60,
                retries=0
            )
        ]
        
        assert len(config.llm.models) == 1
        assert config.llm.models[0].name == "test-model"
        assert config.llm.models[0].retries == 0

    def test_evolution_result_structure(self):
        """Test that EvolutionResult has the expected structure"""
        from openevolve.api import EvolutionResult
        from openevolve.database import Program
        
        # Test creating an EvolutionResult
        result = EvolutionResult(
            best_program=None,
            best_score=0.85,
            best_code="def test(): pass",
            metrics={"accuracy": 0.85, "speed": 100},
            output_dir="/tmp/test"
        )
        
        assert result.best_score == 0.85
        assert result.best_code == "def test(): pass"
        assert result.metrics["accuracy"] == 0.85
        assert result.output_dir == "/tmp/test"
        assert "0.8500" in str(result)  # Test __repr__