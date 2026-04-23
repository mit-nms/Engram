"""
Utility functions for the Architect module.
"""

import os
from typing import List
from rich.console import Console
from rich.table import Table


def make_directories(base_dir, task_type):
    """Create necessary directories for results if they don't exist.

    Args:
        base_dir: Base directory for results
        task_type: Type of task (used for subdirectory)

    Returns:
        Tuple of (code_dir, log_dir, plot_dir)
    """
    # Create directories within the Architect folder, organized by task type
    log_dir = os.path.join(base_dir, task_type, "logs")
    plot_dir = os.path.join(base_dir, task_type, "plots")

    # Create directories if they don't exist
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    return log_dir, plot_dir


def print_table(rows: List[List[str]], columns: List[str] = None, **kwargs):
    table = Table(show_header=columns is not None, **kwargs)
    if columns:
        for col in columns:
            table.add_column(col, justify="left")
    for row in rows:
        if len(row) == 0:
            table.add_section()
        else:
            table.add_row(*row)
    Console().print(table)
