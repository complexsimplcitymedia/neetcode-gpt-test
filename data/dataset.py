import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]], List[List[str]]]:
        # 1. Tokenize by splitting on whitespace
        tokens = raw_dataset.split()

        # 2. Set manual seed and generate batch_size random start indices
        torch.manual_seed(0)
        max_start = len(tokens) - context_length
        start_indices = torch.randint(0, max_start, (batch_size,)).tolist()

        # 3. Build X and Y sequence batches
        X = [tokens[i : i + context_length] for i in start_indices]
        Y = [tokens[i + 1 : i + 1 + context_length] for i in start_indices]

        return X, Y