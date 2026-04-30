"""
Integration tests for island migration functionality with real LLM inference
"""

import pytest
import asyncio
from openevolve.controller import OpenEvolve


class TestMigrationWithLLM:
    """Test island migration with real LLM generation"""

    @pytest.mark.asyncio
    async def test_island_migration_no_duplicates_real_evolution(
        self,
        optillm_server,
        evolution_config,
        test_program_file,
        test_evaluator_file,
        evolution_output_dir
    ):
        """Test that migration doesn't create duplicate chains with real evolution"""
        # Configure for migration testing
        evolution_config.database.num_islands = 3
        evolution_config.database.migration_interval = 4
        evolution_config.database.migration_rate = 0.3
        evolution_config.max_iterations = 12
        evolution_config.evaluator.parallel_evaluations = 3  # One per island
        
        controller = OpenEvolve(
            initial_program_path=str(test_program_file),
            evaluation_file=str(test_evaluator_file),
            config=evolution_config,
            output_dir=str(evolution_output_dir)
        )
        
        await controller.run(iterations=12)
        
        # Verify no _migrant_ suffixes (our fix working)
        all_program_ids = list(controller.database.programs.keys())
        migrant_suffix_programs = [pid for pid in all_program_ids if "_migrant" in pid]
        assert len(migrant_suffix_programs) == 0, \
            f"Found programs with _migrant suffix: {migrant_suffix_programs}"
        
        # Verify no duplicate program IDs in feature maps
        all_mapped_ids = []
        for island_map in controller.database.island_feature_maps:
            all_mapped_ids.extend(island_map.values())
        
        # Check for duplicates
        unique_mapped_ids = set(all_mapped_ids)
        assert len(all_mapped_ids) == len(unique_mapped_ids), \
            "Found duplicate program IDs across island feature maps"
        
        # Verify migration metadata exists if migration occurred
        programs_with_migration_data = [
            p for p in controller.database.programs.values() 
            if p.metadata.get("migrant", False)
        ]
        
        print(f"Total programs: {len(controller.database.programs)}")
        print(f"Programs with migration data: {len(programs_with_migration_data)}")
        print(f"Last migration generation: {controller.database.last_migration_generation}")
        
        # If enough generations passed, migration should have been attempted
        if controller.database.last_migration_generation > 0:
            print("Migration was attempted at least once")
            # Verify migrant programs have clean UUIDs, not _migrant_ suffixes
            for migrant in programs_with_migration_data:
                assert "_migrant" not in migrant.id, \
                    f"Migrant program {migrant.id} has _migrant suffix"

    @pytest.mark.asyncio
    async def test_per_island_map_elites_isolation(
        self,
        optillm_server,
        evolution_config,
        test_program_file,
        test_evaluator_file,
        evolution_output_dir
    ):
        """Test that per-island MAP-Elites works correctly with migration"""
        evolution_config.database.num_islands = 3
        evolution_config.database.migration_interval = 5
        evolution_config.max_iterations = 10
        
        controller = OpenEvolve(
            initial_program_path=str(test_program_file),
            evaluation_file=str(test_evaluator_file),
            config=evolution_config,
            output_dir=str(evolution_output_dir)
        )
        
        await controller.run(iterations=10)
        
        # Check that each island has its own feature map
        assert len(controller.database.island_feature_maps) == 3, \
            "Should have 3 island feature maps"
        
        # Verify that programs exist in their assigned islands
        for island_idx, island_map in enumerate(controller.database.island_feature_maps):
            print(f"Island {island_idx}: {len(island_map)} programs in feature map")
            
            # Check that each program in the feature map exists in the database
            for coord, program_id in island_map.items():
                assert program_id in controller.database.programs, \
                    f"Program {program_id} in island {island_idx} not found in database"
                
                # Verify the program's island assignment matches
                program = controller.database.programs[program_id]
                program_island = program.metadata.get("island", 0)
                assert program_island == island_idx, \
                    f"Program {program_id} island mismatch: in map {island_idx} but metadata says {program_island}"

    @pytest.mark.asyncio
    async def test_migration_preserves_program_quality(
        self,
        optillm_server,
        evolution_config,
        test_program_file,
        test_evaluator_file,
        evolution_output_dir
    ):
        """Test that migration preserves program content and metrics"""
        evolution_config.database.num_islands = 2
        evolution_config.database.migration_interval = 6
        evolution_config.database.migration_rate = 0.5
        evolution_config.max_iterations = 8
        
        controller = OpenEvolve(
            initial_program_path=str(test_program_file),
            evaluation_file=str(test_evaluator_file),
            config=evolution_config,
            output_dir=str(evolution_output_dir)
        )
        
        await controller.run(iterations=8)
        
        # Find programs marked as migrants
        migrant_programs = [
            p for p in controller.database.programs.values() 
            if p.metadata.get("migrant", False)
        ]
        
        print(f"Found {len(migrant_programs)} migrant programs")
        
        for migrant in migrant_programs:
            # Verify migrant has a parent
            assert migrant.parent_id is not None, f"Migrant {migrant.id} should have parent_id"
            
            # Verify parent exists in database
            parent = controller.database.get(migrant.parent_id)
            if parent:  # Parent might have been replaced in MAP-Elites
                # Compare core properties that should be preserved
                assert migrant.language == parent.language, "Language should be preserved"
                # Code might be identical or evolved, we don't enforce exact match
                assert migrant.metrics is not None, "Migrant should have metrics"
            
            # Verify migrant is properly integrated (has island assignment)
            assert "island" in migrant.metadata, "Migrant should have island assignment"
            
            # Most importantly: no _migrant_ suffix
            assert "_migrant" not in migrant.id, f"Migrant {migrant.id} should not have _migrant suffix"

    @pytest.mark.asyncio
    async def test_migration_timing_logic(
        self,
        optillm_server,
        evolution_config,
        test_program_file,
        test_evaluator_file,
        evolution_output_dir
    ):
        """Test that migration timing logic works correctly"""
        evolution_config.database.num_islands = 2
        evolution_config.database.migration_interval = 3
        evolution_config.max_iterations = 6
        
        controller = OpenEvolve(
            initial_program_path=str(test_program_file),
            evaluation_file=str(test_evaluator_file),
            config=evolution_config,
            output_dir=str(evolution_output_dir)
        )
        
        # Track island generations during evolution
        initial_generations = controller.database.island_generations.copy()
        print(f"Initial island generations: {initial_generations}")
        
        await controller.run(iterations=6)
        
        final_generations = controller.database.island_generations.copy()
        final_migration_gen = controller.database.last_migration_generation
        
        print(f"Final island generations: {final_generations}")
        print(f"Last migration generation: {final_migration_gen}")
        
        # Basic sanity checks
        assert all(gen >= 0 for gen in final_generations), "All generations should be non-negative"
        assert final_migration_gen >= 0, "Last migration generation should be non-negative"
        
        # If any island advanced beyond migration interval, migration should have been considered
        max_generation = max(final_generations)
        if max_generation >= evolution_config.database.migration_interval:
            # Migration may or may not have happened (depends on island population), 
            # but the system should have at least considered it
            print(f"Migration should have been considered (max gen: {max_generation})")

    @pytest.mark.asyncio
    async def test_single_island_no_migration(
        self,
        optillm_server,
        evolution_config,
        test_program_file,
        test_evaluator_file,
        evolution_output_dir
    ):
        """Test that single island setup doesn't attempt migration"""
        evolution_config.database.num_islands = 1
        evolution_config.database.migration_interval = 3
        evolution_config.max_iterations = 8
        
        controller = OpenEvolve(
            initial_program_path=str(test_program_file),
            evaluation_file=str(test_evaluator_file),
            config=evolution_config,
            output_dir=str(evolution_output_dir)
        )
        
        await controller.run(iterations=8)
        
        # With single island, no migration should occur
        assert controller.database.last_migration_generation == 0, \
            "Single island should not perform migration"
        
        # All programs should be on island 0
        for program in controller.database.programs.values():
            program_island = program.metadata.get("island", 0)
            assert program_island == 0, f"Program {program.id} should be on island 0, found on island {program_island}"
        
        # No migrant programs should exist
        migrant_programs = [p for p in controller.database.programs.values() if p.metadata.get("migrant", False)]
        assert len(migrant_programs) == 0, "Single island should not create migrant programs"