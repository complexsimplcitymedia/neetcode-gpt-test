from collections import defaultdict
from typing import List


class Solution:

    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        tokens = list(corpus)
        merges = []

        for _ in range(num_merges):
            if len(tokens) < 2:
                break

            # Count frequency of all adjacent token pairs
            pair_counts = defaultdict(int)
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] += 1

            if not pair_counts:
                break

            # Find the most frequent pair
            # Minimize negative count (max frequency), then minimize pair lexicographically
            best_pair = min(
                pair_counts.keys(), key=lambda p: (-pair_counts[p], p)
            )

            # Merge all non-overlapping occurrences left to right
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and (tokens[i], tokens[i + 1]) == best_pair
                ):
                    new_tokens.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens
            merges.append([best_pair[0], best_pair[1]])

        return merges