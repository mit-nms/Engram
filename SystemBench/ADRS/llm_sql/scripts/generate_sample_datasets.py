"""Generate sample datasets for agent exploration.

Produces two folders:
  - sample_data/          : columns after pre-merge (matches evaluator logic)
  - sample_data_original/ : raw columns with no merging applied
"""
import os
import pandas as pd
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from evaluator import _apply_column_merges, TEST_FILES, COL_MERGES

SAMPLE_SIZE = 300
OUTPUT_DIR = os.path.join(parent_dir, "deepagents_files", "sample_data")
OUTPUT_DIR_ORIGINAL = os.path.join(parent_dir, "deepagents_files", "sample_data_original")


def generate_samples():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for csv_path, merge_spec in zip(TEST_FILES, COL_MERGES):
        dataset_name = os.path.basename(csv_path).replace('.csv', '')
        print(f"Processing {dataset_name} (merged)...")

        df = pd.read_csv(csv_path, low_memory=False)
        df = _apply_column_merges(df, merge_spec)

        if len(df) > SAMPLE_SIZE:
            sample_df = df.sample(n=SAMPLE_SIZE, random_state=42)
        else:
            sample_df = df

        output_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_sample.csv")
        sample_df.to_csv(output_path, index=False)
        print(f"  Saved {len(sample_df)} rows, {len(sample_df.columns)} columns")


def generate_samples_original():
    os.makedirs(OUTPUT_DIR_ORIGINAL, exist_ok=True)

    for csv_path in TEST_FILES:
        dataset_name = os.path.basename(csv_path).replace('.csv', '')
        print(f"Processing {dataset_name} (original)...")

        df = pd.read_csv(csv_path, low_memory=False)

        if len(df) > SAMPLE_SIZE:
            sample_df = df.sample(n=SAMPLE_SIZE, random_state=42)
        else:
            sample_df = df

        output_path = os.path.join(OUTPUT_DIR_ORIGINAL, f"{dataset_name}_sample.csv")
        sample_df.to_csv(output_path, index=False)
        print(f"  Saved {len(sample_df)} rows, {len(sample_df.columns)} columns")


if __name__ == "__main__":
    generate_samples()
    generate_samples_original()
