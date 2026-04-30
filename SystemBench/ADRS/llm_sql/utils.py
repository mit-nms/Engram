from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from typing import List, Tuple

class TrieNode:
    def __init__(self):
        self.children = {}
        self.end_of_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.end_of_word = True

    def longest_common_prefix(self, word):
        node = self.root
        common_prefix_length = 0
        for char in word:
            if char in node.children:
                common_prefix_length += len(char)
                node = node.children[char]
            else:
                break
        return common_prefix_length

def calculate_length(value):
    val = 0
    if isinstance(value, bool):
        val = 4  # length of 'True' or 'False'
    elif isinstance(value, (int, float)):
        val = len(str(value))
    elif isinstance(value, str):
        val = len(value)
    else:
        val = 0
    return val**2

def evaluate_df_prefix_hit_cnt(df: pd.DataFrame) -> Tuple[int, int]:
    """
    Function to evaluate the prefix hit count of a DataFrame.
    Uses vectorized string creation for better performance.
    """

    def max_overlap(trie, row_string):
        return min(len(row_string), trie.longest_common_prefix(row_string))

    # Pre-compute all row strings at once (vectorized) - much faster than iterrows()
    row_strings = df.fillna("").astype(str).agg(''.join, axis=1).tolist()

    trie = Trie()
    total_prefix_hit_count = 0
    total_string_length = 0

    for row_string in row_strings:
        total_string_length += len(row_string)
        row_prefix_hit_count = max_overlap(trie, row_string)
        trie.insert(row_string)
        total_prefix_hit_count += row_prefix_hit_count

    total_prefix_hit_rate = total_prefix_hit_count / total_string_length if total_string_length > 0 else 0
    assert total_prefix_hit_count <= total_string_length
    return total_prefix_hit_count, total_prefix_hit_rate * 100
