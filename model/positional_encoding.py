import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        # Create an empty matrix for the positional encodings
        pe = np.zeros((seq_len, d_model))
        
        # Create a column vector for positions: shape (seq_len, 1)
        positions = np.arange(seq_len)[:, np.newaxis]
        
        # Create a row vector for the dimension indices (2i): shape (d_model/2,)
        # These are the even indices: 0, 2, 4, ...
        _2i = np.arange(0, d_model, 2)
        
        # Calculate the denominator: 10000^(2i / d_model)
        denominator = 10000 ** (_2i / d_model)
        
        # Compute the sine and cosine values using broadcasting
        pe[:, 0::2] = np.sin(positions / denominator)
        pe[:, 1::2] = np.cos(positions / denominator)
        
        return np.round(pe, 5)