from typing import List
from collections import Counter


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        tokens = list(corpus)
        merges = []

        for _ in range(num_merges):
            if len(tokens) < 2:
                break

            pairs = Counter(zip(tokens, tokens[1:]))
            if not pairs:
                break

            # Find max frequency, breaking ties by smallest lexicographical pair
            best_pair = min(pairs.keys(), key=lambda p: (-pairs[p], p))

            new_tokens = []
            i = 0
            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == best_pair[0]
                    and tokens[i + 1] == best_pair[1]
                ):
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            merges.append([best_pair[0], best_pair[1]])
            tokens = new_tokens

        return merges