from typing import Tuple
import torch
from torchtyping import TensorType


class Solution:

    def create_batches(
        self, data: TensorType[int], context_length: int, batch_size: int
    ) -> Tuple[TensorType[int], TensorType[int]]:
        # data: 1D tensor of encoded text (integer token IDs)
        # context_length: number of tokens in each training example
        # batch_size: number of examples per batch

        # Set manual seed for reproducibility
        torch.manual_seed(0)

        # Generate random start indices: high = len(data) - context_length (exclusive)
        max_start = len(data) - context_length
        start_indices = torch.randint(0, max_start, (batch_size,))

        # Extract sequence windows for input X and target Y
        X = torch.stack(
            [data[i : i + context_length] for i in start_indices]
        )
        Y = torch.stack(
            [data[i + 1 : i + 1 + context_length] for i in start_indices]
        )

        return X, Y