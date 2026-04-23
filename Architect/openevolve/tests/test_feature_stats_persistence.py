"""
Unit tests for feature_stats persistence in ProgramDatabase checkpoints
"""

import json
import os
import tempfile
import shutil
import unittest
from unittest.mock import patch

from openevolve.database import ProgramDatabase, Program
from openevolve.config import DatabaseConfig


class TestFeatureStatsPersistence(unittest.TestCase):
    """Test feature_stats are correctly saved and loaded in checkpoints"""

    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.config = DatabaseConfig(
            db_path=self.test_dir,
            feature_dimensions=["score", "custom_metric1", "custom_metric2"],
            feature_bins=10,
        )

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)

    def test_feature_stats_saved_and_loaded(self):
        """Test that feature_stats are correctly saved and loaded from checkpoints"""
        # Create database and add programs to build feature_stats
        db1 = ProgramDatabase(self.config)

        programs = []
        for i in range(5):
            program = Program(
                id=f"test_prog_{i}",
                code=f"# Test program {i}",
                metrics={
                    "combined_score": 0.1 + i * 0.2,
                    "custom_metric1": 10 + i * 20,
                    "custom_metric2": 100 + i * 50,
                },
            )
            programs.append(program)
            db1.add(program)

        # Verify feature_stats were built
        self.assertIn("score", db1.feature_stats)
        self.assertIn("custom_metric1", db1.feature_stats)
        self.assertIn("custom_metric2", db1.feature_stats)

        # Store original feature_stats for comparison
        original_stats = {
            dim: {"min": stats["min"], "max": stats["max"], "values": stats["values"].copy()}
            for dim, stats in db1.feature_stats.items()
        }

        # Save checkpoint
        db1.save(self.test_dir, iteration=42)

        # Load into new database
        db2 = ProgramDatabase(self.config)
        db2.load(self.test_dir)

        # Verify feature_stats were loaded correctly
        self.assertEqual(len(db2.feature_stats), len(original_stats))

        for dim, original in original_stats.items():
            self.assertIn(dim, db2.feature_stats)
            loaded = db2.feature_stats[dim]

            self.assertAlmostEqual(loaded["min"], original["min"], places=5)
            self.assertAlmostEqual(loaded["max"], original["max"], places=5)
            self.assertEqual(loaded["values"], original["values"])

    def test_empty_feature_stats_handling(self):
        """Test handling of empty feature_stats"""
        db1 = ProgramDatabase(self.config)

        # Save without any programs (empty feature_stats)
        db1.save(self.test_dir, iteration=1)

        # Load and verify
        db2 = ProgramDatabase(self.config)
        db2.load(self.test_dir)

        self.assertEqual(db2.feature_stats, {})

    def test_backward_compatibility_missing_feature_stats(self):
        """Test loading checkpoints that don't have feature_stats (backward compatibility)"""
        # Create a checkpoint manually without feature_stats
        os.makedirs(self.test_dir, exist_ok=True)

        # Create metadata without feature_stats (simulating old checkpoint)
        metadata = {
            "island_feature_maps": [{}],  # Updated to new format
            "islands": [[]],
            "archive": [],
            "best_program_id": None,
            "island_best_programs": [None],
            "last_iteration": 10,
            "current_island": 0,
            "island_generations": [0],
            "last_migration_generation": 0,
            # Note: no "feature_stats" key
        }

        with open(os.path.join(self.test_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        # Load should work without errors
        db = ProgramDatabase(self.config)
        db.load(self.test_dir)

        # feature_stats should be empty but not None
        self.assertEqual(db.feature_stats, {})

    def test_feature_stats_serialization_edge_cases(self):
        """Test feature_stats serialization handles edge cases correctly"""
        db = ProgramDatabase(self.config)

        # Test with various edge cases
        db.feature_stats = {
            "normal_case": {"min": 1.0, "max": 10.0, "values": [1.0, 5.0, 10.0]},
            "single_value": {"min": 5.0, "max": 5.0, "values": [5.0]},
            "large_values_list": {
                "min": 0.0,
                "max": 200.0,
                "values": list(range(200)),  # Should be truncated to 100
            },
            "empty_values": {"min": 0.0, "max": 1.0, "values": []},
        }

        # Test serialization
        serialized = db._serialize_feature_stats()

        # Check that large values list was truncated
        self.assertLessEqual(len(serialized["large_values_list"]["values"]), 100)

        # Test deserialization
        deserialized = db._deserialize_feature_stats(serialized)

        # Verify structure is maintained
        self.assertIn("normal_case", deserialized)
        self.assertIn("single_value", deserialized)
        self.assertIn("large_values_list", deserialized)
        self.assertIn("empty_values", deserialized)

        # Verify types are correct
        for dim, stats in deserialized.items():
            self.assertIsInstance(stats["min"], float)
            self.assertIsInstance(stats["max"], float)
            self.assertIsInstance(stats["values"], list)

    def test_feature_stats_preservation_during_load(self):
        """Test that feature_stats ranges are preserved when loading from checkpoint"""
        # Create database with programs
        db1 = ProgramDatabase(self.config)

        test_programs = []

        for i in range(3):
            program = Program(
                id=f"stats_test_{i}",
                code=f"# Stats test {i}",
                metrics={
                    "combined_score": 0.2 + i * 0.3,
                    "custom_metric1": 20 + i * 30,
                    "custom_metric2": 200 + i * 100,
                },
            )
            test_programs.append(program)
            db1.add(program)

        # Record original feature ranges
        original_ranges = {}
        for dim, stats in db1.feature_stats.items():
            original_ranges[dim] = {"min": stats["min"], "max": stats["max"]}

        # Save checkpoint
        db1.save(self.test_dir, iteration=50)

        # Load into new database
        db2 = ProgramDatabase(self.config)
        db2.load(self.test_dir)

        # Verify feature ranges are preserved
        for dim, original_range in original_ranges.items():
            self.assertIn(dim, db2.feature_stats)
            loaded_stats = db2.feature_stats[dim]

            self.assertAlmostEqual(
                loaded_stats["min"],
                original_range["min"],
                places=5,
                msg=f"Min value changed for {dim}: {original_range['min']} -> {loaded_stats['min']}",
            )
            self.assertAlmostEqual(
                loaded_stats["max"],
                original_range["max"],
                places=5,
                msg=f"Max value changed for {dim}: {original_range['max']} -> {loaded_stats['max']}",
            )

        # Test that adding a new program within existing ranges doesn't break anything
        new_program = Program(
            id="range_test",
            code="# Program to test range stability",
            metrics={
                "combined_score": 0.35,  # Within existing range
                "custom_metric1": 35,  # Within existing range
                "custom_metric2": 250,  # Within existing range
            },
        )

        # Adding this program should not cause issues
        db2.add(new_program)
        new_coords = db2._calculate_feature_coords(new_program)

        # Should get valid coordinates
        self.assertEqual(len(new_coords), len(self.config.feature_dimensions))
        for coord in new_coords:
            self.assertIsInstance(coord, int)
            self.assertGreaterEqual(coord, 0)

    def test_feature_stats_with_numpy_types(self):
        """Test that numpy types are correctly handled in serialization"""
        import numpy as np

        db = ProgramDatabase(self.config)

        # Simulate feature_stats with numpy types
        db.feature_stats = {
            "numpy_test": {
                "min": np.float64(1.5),
                "max": np.float64(9.5),
                "values": [np.float64(x) for x in [1.5, 5.0, 9.5]],
            }
        }

        # Test serialization doesn't fail
        serialized = db._serialize_feature_stats()

        # Verify numpy types were converted to Python types
        self.assertIsInstance(serialized["numpy_test"]["min"], float)
        self.assertIsInstance(serialized["numpy_test"]["max"], float)

        # Test deserialization
        deserialized = db._deserialize_feature_stats(serialized)
        self.assertIsInstance(deserialized["numpy_test"]["min"], float)
        self.assertIsInstance(deserialized["numpy_test"]["max"], float)

    def test_malformed_feature_stats_handling(self):
        """Test handling of malformed feature_stats during deserialization"""
        db = ProgramDatabase(self.config)

        # Test with malformed data
        malformed_data = {
            "valid_entry": {"min": 1.0, "max": 10.0, "values": [1.0, 5.0, 10.0]},
            "invalid_entry": "this is not a dict",
            "missing_keys": {
                "min": 1.0
                # missing "max" and "values"
            },
        }

        with patch("openevolve.database.logger") as mock_logger:
            deserialized = db._deserialize_feature_stats(malformed_data)

        # Should have valid entry and skip invalid ones
        self.assertIn("valid_entry", deserialized)
        self.assertNotIn("invalid_entry", deserialized)
        self.assertIn("missing_keys", deserialized)  # Should be created with defaults

        # Should have logged warning for invalid entry
        mock_logger.warning.assert_called()


if __name__ == "__main__":
    unittest.main()
