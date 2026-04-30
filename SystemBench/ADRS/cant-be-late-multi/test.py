#!/usr/bin/env python3
"""
Simple test script for the Cant-Be-Late multi-region problem.
Runs the stage-2 evaluator on a chosen strategy file.
"""

import os
import sys
from typing import Optional

# Add the current directory to path to import evaluator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluator import evaluate


def main(initial_program_name: Optional[str] = None) -> int:
    print("=" * 70)
    print("Testing Cant-Be-Late Multi-Region Problem")
    print("=" * 70)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    if initial_program_name is None:
        initial_program_path = os.path.join(script_dir, "initial_program.py")
    else:
        initial_program_path = os.path.join(script_dir, initial_program_name)

    if not os.path.exists(initial_program_path):
        print(f"ERROR: program not found at {initial_program_path}")
        return 1

    print(f"\nUsing program: {initial_program_path}")
    print("-" * 70)

    try:
        results = evaluate(initial_program_path)

        if not isinstance(results, dict):
            # Evaluator for this problem always returns a dict
            raise TypeError(f"Unexpected result type from evaluate_stage2: {type(results)}")

        combined_score = results.get("combined_score", 0.0)

        print("\n" + "=" * 70)
        print("Evaluation Results")
        print("=" * 70)
        print(f"Combined Score: {combined_score:.6f}")
        for key, value in results.items():
            if key == "combined_score":
                continue
            if isinstance(value, (int, float)):
                print(f"{key}: {value}")
        print("\n" + "=" * 70)

        if combined_score > -1e8:
            print("✅ Test COMPLETED: evaluation ran successfully")
            return 0
        else:
            print("❌ Test FAILED: combined_score indicates failure")
            return 1
    except Exception as e:
        print(f"\n❌ Test FAILED with exception: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(main(sys.argv[1]))
    sys.exit(main())

