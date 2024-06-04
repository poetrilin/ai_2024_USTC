
import math
import torch
from torch import nn


def get_positional_encoding(d_model: int, seq_len: int = 5000):
    '''
    @param d_model: int: number of features in the query, key, and value vectors
    @param seq_len: int: maximum length of the input sequence
    @return encodings: torch.Tensor: positional encodings of shape (seq_len, d_model)
    '''
    encodings = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
    two_i = torch.arange(0, d_model, 2, dtype=torch.float32)
    div_term = torch.exp(two_i*(-math.log(10000.0)/d_model))
    encodings[:, 0::2] = torch.sin(position*div_term)
    encodings[:, 1::2] = torch.cos(position*div_term)
    encodings = encodings.unsqueeze(0).requires_grad_(False)
    return encodings


class AddPositionEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout_prob: float = 0.1):
        """
        d_model: number of features in the query, key, and value vectors
        seq_len: maximum length of the input sequence
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.register_buffer('positional_encoding',
                             get_positional_encoding(d_model, seq_len), False)

    def forward(self, x: torch.Tensor):
        pe = self.positional_encoding.detach().requires_grad_(False)
        x = x + pe
        x = self.dropout(x)
        return x
