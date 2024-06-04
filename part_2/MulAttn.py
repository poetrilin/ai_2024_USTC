import torch
import torch.nn as nn
import torch.nn.functional as F


class HeadAttention(nn.Module):
    def __init__(self, seq_len: int, embed_size: int, hidden_size: int,
                 bias: bool = True, dropout_prob: float = 0.1):
        super().__init__()
        # embed_size: dimension for input embedding vector
        # hidden_size: dimension for hidden vector. eg. x:(..., embed_size) --to_q--> query_vector:(..., hidden_size)

        # a triangular bool matrix for mask
        self.register_buffer("tril", torch.tril(
            torch.ones(seq_len, seq_len)))  # mask

        # TODO: init three matrix, to_q, to_k, to_v.
        self.to_q = nn.Linear(embed_size, hidden_size, bias=bias)
        self.to_k = nn.Linear(embed_size, hidden_size, bias=bias)
        self.to_v = nn.Linear(embed_size, hidden_size, bias=True)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inputs):
        # input: (batch_size, seq_len, embed_size)
        # return (batch_size, seq_len, hidden_size)
        # TODO: implement the attention mechanism
        q = self.to_q(inputs)
        k = self.to_k(inputs)
        v = self.to_v(inputs)
        scale = 1 / torch.sqrt(torch.tensor(k.size(-1), dtype=torch.float32))
        # (batch_size, seq_len, seq_len)
        attention = torch.matmul(q, k.transpose(-2, -1))*scale

        attention = attention.masked_fill(self.tril == 0, float('-inf'))
        attention = F.softmax(attention, dim=-1)

        attention = self.dropout(attention)
        out = attention@v
        return out, attention[-1]


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int, head_size: int, seq_len: int, embed_size: int):
        # n_heads is the number of head attention
        # head_size is the hidden_size in each HeadAttention
        super().__init__()
        head_size = embed_size // n_heads

        self.n_heads = n_heads
        self.head_size = head_size
        self.heads = nn.ModuleList(
            [HeadAttention(seq_len, embed_size, head_size) for _ in range(n_heads)])
        self.projection = nn.Linear(n_heads*head_size, embed_size)

    def forward(self, inputs):
        # input: (batch_size, seq_len, embed_size), make sure embed_size=n_heads x head_size
        # return: (batch_size, seq_len, embed_size), attn: (seq_len, seq_len)

        assert inputs.size(-1) == self.n_heads * self.head_size

        heads_out = [head(inputs)[0] for head in self.heads]
        # attn取最后一个head
        attn = self.heads[-1](inputs)[1]
        out = torch.cat(heads_out, dim=-1)
        out = self.projection(out)
        return out, attn
