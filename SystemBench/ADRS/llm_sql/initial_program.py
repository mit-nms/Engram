import pandas as pd
from solver import Algorithm


class Evolved(Algorithm):
    """
    Simple baseline: Sort columns by uniqueness (low-cardinality first),
    then sort rows lexicographically.
    """

    def reorder(self, df: pd.DataFrame, **kwargs):
        if df.empty:
            return df

        # Order columns by number of unique values (ascending)
        cols_by_uniqueness = sorted(df.columns, key=lambda c: df[c].nunique())

        # Reorder columns and sort rows lexicographically
        result = df[cols_by_uniqueness].sort_values(
            by=cols_by_uniqueness,
            kind="mergesort"
        )

        return result
