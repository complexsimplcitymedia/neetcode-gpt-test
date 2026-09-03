from typing import Dict, List, Tuple


class Solution:

    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Find unique characters, sort them alphabetically
        unique_chars = sorted(set(text))

        # Build string-to-index (stoi) and index-to-string (itos) mappings
        stoi = {ch: i for i, ch in enumerate(unique_chars)}
        itos = {i: ch for i, ch in enumerate(unique_chars)}

        return stoi, itos

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Map each character in the string to its integer ID
        return [stoi[ch] for ch in text]

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Reconstruct the string from token IDs
        return "".join(itos[i] for i in ids)