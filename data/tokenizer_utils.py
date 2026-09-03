from typing import List, Dict

class Solution:
    
    def _greedy_tokenize(self, text: str, vocab: Dict[str, int]) -> List[str]:
        tokens = []
        i = 0
        n = len(text)
        while i < n:
            matched = False
            # Find the longest matching prefix starting at index i
            for j in range(n, i, -1):
                sub = text[i:j]
                if sub in vocab:
                    tokens.append(sub)
                    i = j
                    matched = True
                    break
            if not matched:
                # Fallback to single character if no sub-phrase is in vocab
                tokens.append(text[i])
                i += 1
        return tokens

    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number string representation using greedy left-to-right longest match.
        return [self._greedy_tokenize(str(num), vocab) for num in numbers]

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        return len(self._greedy_tokenize(text, vocab))

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        words = text.split()
        if not words:
            return 0.0
        
        num_tokens = self.count_tokens(text, vocab)
        return round(num_tokens / len(words), 4)