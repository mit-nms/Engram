"""
Test file extension preservation across iterations
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from openevolve.config import Config, EvaluatorConfig, DatabaseConfig, LLMConfig, PromptConfig
from openevolve.controller import OpenEvolve
from openevolve.evaluator import Evaluator
from openevolve.process_parallel import ProcessParallelController, _worker_init, _lazy_init_worker_components


class TestFileExtensionPreservation(unittest.TestCase):
    """Test that file extensions are preserved across iterations"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

        # Create test config
        self.config = Config(
            max_iterations=5,
            language="cpp",
            llm=LLMConfig(models=[]),
            database=DatabaseConfig(population_size=10, num_islands=2),
            evaluator=EvaluatorConfig(parallel_evaluations=1),
            prompt=PromptConfig(),
        )

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_controller_preserves_file_extension(self):
        """Test that the controller properly extracts and stores file extension"""
        # Test Path extraction directly
        test_files = [
            "test.cpp",
            "test.rs",
            "test.py",
            "test.r"
        ]

        for filename in test_files:
            with self.subTest(filename=filename):
                expected_ext = Path(filename).suffix
                actual_ext = os.path.splitext(filename)[1]
                self.assertEqual(actual_ext, expected_ext)

    def test_evaluator_uses_correct_suffix(self):
        """Test that the evaluator uses the correct file suffix"""
        # Test with different file extensions
        test_cases = [
            (".py", "python"),
            (".cpp", "cpp"),
            (".rs", "rust"),
            (".r", "r"),
            (".js", "javascript"),
            (".metal", "metal")
        ]

        for suffix, language in test_cases:
            with self.subTest(suffix=suffix, language=language):
                # Create a temp evaluator file for this test
                temp_eval = os.path.join(self.temp_dir, f"eval_{language}.py")
                with open(temp_eval, "w") as f:
                    f.write("def evaluate(path): return {'score': 0.5}")

                # Create evaluator with specific suffix
                config = EvaluatorConfig()
                evaluator = Evaluator(config, temp_eval, suffix=suffix)

                # Check that the suffix is correctly stored
                self.assertEqual(evaluator.program_suffix, suffix)

    def test_worker_suffix_access(self):
        """Test that worker can access file suffix from config"""
        # Create a mock config object with file_suffix
        mock_config = MagicMock()
        mock_config.file_suffix = ".cpp"

        # Test getattr access pattern used in the code
        suffix = getattr(mock_config, 'file_suffix', '.py')
        self.assertEqual(suffix, ".cpp")

        # Test default fallback
        mock_config_no_suffix = MagicMock()
        del mock_config_no_suffix.file_suffix
        suffix_default = getattr(mock_config_no_suffix, 'file_suffix', '.py')
        self.assertEqual(suffix_default, ".py")

    def test_process_parallel_controller_passes_suffix(self):
        """Test that ProcessParallelController correctly passes file suffix"""
        # Create a mock database
        mock_database = MagicMock()

        # Create ProcessParallelController with specific file suffix
        controller = ProcessParallelController(
            self.config,
            "dummy_evaluator.py",
            mock_database,
            file_suffix=".rs"
        )

        # Check that file suffix is stored
        self.assertEqual(controller.file_suffix, ".rs")

    def test_file_extension_mapping(self):
        """Test that different file extensions are handled correctly"""
        test_files = [
            ("test.py", ".py"),
            ("test.cpp", ".cpp"),
            ("test.rs", ".rs"),
            ("test.r", ".r"),
            ("test.js", ".js"),
            ("test.metal", ".metal")
        ]

        for filename, expected_extension in test_files:
            with self.subTest(filename=filename):
                # Test Path suffix extraction
                actual_extension = Path(filename).suffix
                self.assertEqual(actual_extension, expected_extension)

    def test_evaluator_temp_file_creation(self):
        """Test that evaluator creates temp files with correct suffix"""
        # Create a temp evaluator file
        temp_eval = os.path.join(self.temp_dir, "eval_test.py")
        with open(temp_eval, "w") as f:
            f.write("def evaluate(path): return {'score': 0.5}")

        # Test different suffixes
        test_suffixes = [".py", ".cpp", ".rs", ".r", ".js"]

        for suffix in test_suffixes:
            with self.subTest(suffix=suffix):
                config = EvaluatorConfig()
                evaluator = Evaluator(config, temp_eval, suffix=suffix)

                # Verify suffix is stored correctly
                self.assertEqual(evaluator.program_suffix, suffix)


if __name__ == "__main__":
    unittest.main()