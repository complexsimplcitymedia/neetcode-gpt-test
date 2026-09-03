from typing import List
import torch
import torch.nn as nn
from torchtyping import TensorType


class Solution:

    def get_dataset(
        self, positive: List[str], negative: List[str]
    ) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        all_sentences = positive + negative
        unique_words = sorted(
            list({word for sentence in all_sentences for word in sentence.split()})
        )
        vocab = {word: i + 1 for i, word in enumerate(unique_words)}

        # 2. Encode each sentence by replacing words with their IDs
        encoded_tensors = []
        for sentence in all_sentences:
            ids = [vocab[word] for word in sentence.split()]
            encoded_tensors.append(torch.tensor(ids, dtype=torch.float32))

        # 3. & 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence
        padded_dataset = nn.utils.rnn.pad_sequence(
            encoded_tensors, batch_first=True, padding_value=0.0
        )

        return padded_dataset
        