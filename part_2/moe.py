import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


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
        attention = torch.matmul(q, k.transpose(-2, -1))*scale

        attention = attention.masked_fill(self.tril == 0, float('-inf'))
        attention = F.softmax(attention, dim=-1)

        attention = self.dropout(attention)
        out = attention@v
        return out


class MultiHeadAttention(nn.Module):
    # MultiHeadAttention is consist of many HeadAttention output.
    # concat all this head attention output o_i, then merge them with a projection matrix W_o, as [o_1, o_2, ...] x W_o
    # The reason for using multi-head attention is that we want each head to be able to extract different features
    def __init__(self, n_heads: int, head_size: int, seq_len: int, embed_size: int):
        # n_heads is the number of head attention
        # head_size is the hidden_size in each HeadAttention
        super().__init__()
        head_size = embed_size // n_heads
        # TODO: implement heads and projection
        self.n_heads = n_heads
        self.head_size = head_size
        self.heads = nn.ModuleList(
            [HeadAttention(seq_len, embed_size, head_size) for _ in range(n_heads)])
        self.projection = nn.Linear(n_heads*head_size, embed_size)

    def forward(self, inputs):
        # input: (batch_size, seq_len, embed_size), make sure embed_size=n_heads x head_size
        # return: (batch_size, seq_len, embed_size)
        # TODO:
        assert inputs.size(-1) == self.n_heads * self.head_size

        heads_out = [head(inputs) for head in self.heads]
        out = torch.cat(heads_out, dim=-1)
        out = self.projection(out)
        return out


class Expert(nn.Module):
    '''
    FFN module, which is a two-layer feed-forward network with activation function.
    '''

    def __init__(self, embed_size: int, dropout_prob: float = 0.1):
        super().__init__()
        # TODO: init two linear layer
        self.layer1 = nn.Linear(embed_size, 4*embed_size)
        self.layer2 = nn.Linear(4*embed_size, embed_size)
        self.activation = nn.GELU()  # GELU  or ReLU
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inputs):
        # inputs: (batch_size, seq_len, embed_size)
        # -> mid: (batch_size, seq_len, 4 x embed_size)
        # -> outputs: (batch_size, seq_len, embed_size)
        mid = self.layer1(inputs)
        mid = self.activation(mid)
        mid = self.dropout(mid)
        out = self.layer2(mid)
        return out

# First define the top k router module


class TopkRouter(nn.Module):
    def __init__(self, embed_size: int, num_experts: int, active_experts: int):
        ''' 
        TODO
        @embed_size : dimension of embedding 
        @num_experts : how many Experts per layer
        @active_experts: only active_experts out of num_experts are selected to process Embeddings per token.
        '''
        super().__init__()
        self.num_experts = num_experts
        self.active_experts = active_experts
        self.score_layer = nn.Linear(embed_size, num_experts)

    def get_scores(self, inputs):
        """
        socre[0:seq] = MLP(inputs[0:seq])
        @input  inputs: (batch_size, seq_len, embed_size)
        @return scores: (batch_size, seq_len, num_experts)
        """
        out = self.score_layer(inputs)
        out = F.softmax(out, dim=-1)
        return out

    def forward(self, inputs):
        ''' TODO
        完成这部分时，注意使用Softmax()对router_output做标准化。同时注意这部分所用操作的可导性。
        inputs is the output tensor from multihead self attention block, shape (B:batch size, T: seq_len, C: embed_size)
        return:
        --- 
        router_output: normalized weight of Experts, 即教程中的 \alpha
        indices:   index of selected Experts, 即教程中的 index
        '''
        scores = self.get_scores(inputs)
        # mask,只保留active_experts个最大值
        topk_value, top_k_indices = scores.topk(self.active_experts, dim=-1)
        mask = torch.zeros_like(scores).scatter_(
            dim=-1, index=top_k_indices, value=1.0)
        indices = mask
        masked_scores = scores * mask

        # for 0-> -inf , softmax -> 0
        masked_scores = masked_scores.masked_fill(mask == 0, float('-inf'))
        router_output = F.softmax(masked_scores, dim=-1)
        return router_output, indices


class SparseMoE(nn.Module):
    def __init__(self, embed_size: int, num_experts: int, active_experts: int):
        super().__init__()
        self.router = TopkRouter(embed_size, num_experts, active_experts)
        self.experts = nn.ModuleList(
            [Expert(embed_size) for _ in range(num_experts)])

    def forward(self, inputs):
        # TODO
        # 1. get router_output  from router
        # (batch_size, seq_len, num_experts)
        experts_weights, _ = self.router(inputs)
        # 2. get expert_output from experts
        # [num_experts, (batch_size, seq_len, embed_size)]
        expert_outputs = [expert(inputs) for expert in self.experts]
        # 3. merge expert_output with router_output
        # (batch_size, seq_len, embed_size, num_experts)
        expert_outputs = torch.stack(expert_outputs, dim=-1)
        if torch.isnan(expert_outputs).any():
            expert_outputs[
                torch.isnan(expert_outputs)
            ] = 0
        # Combine expert outputs and gating scores
        moe_output = torch.sum(
            experts_weights.unsqueeze(-2) * expert_outputs, dim=-1
        )
        # 4. return the final output
        # (batch_size, seq_len, embed_size)
        return moe_output


class Block(nn.Module):
    # Transformer basic block, consist of MultiHeadAttention, FeedForward and layer normalization
    def __init__(self, embed_size: int, n_heads: int, seq_len: int, num_experts: int, active_experts: int):
        super().__init__()
        # TODO: implement block structure
        super().__init__()
        self.attn = MultiHeadAttention(
            n_heads, embed_size, seq_len, embed_size)
        self.layer_norm = nn.LayerNorm(embed_size)
        self.moe = SparseMoE(embed_size, num_experts, active_experts)

    def forward(self, inputs):
        # input: (batch_size, seq_len, embed_size)
        # TODO: forward with residual connection
        resi = inputs
        x = self.attn(inputs)
        x = x + resi  # add residual connection
        x = self.layer_norm(x)
        add_normed = x

        ##### MoE #####
        x = self.moe(x)
        x = x + add_normed
        x = self.layer_norm(x)
        return x


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


class SparseMoETransformer(nn.Module):
    ''' 
    Transformer decoder, consist of 
    token embedding layer and position_embedding(position_embedding 可以理解为对位置编码，感兴趣的同学可以查阅原文，这里可以看为vocab_len = seq_len的Embedding)
    a stack of Transformer basic block
    a layernorm and output linear layer
    '''

    def __init__(self, vocab_size: int, seq_len: int, embed_size: int, n_layers: int, n_heads: int, num_experts: int, active_experts: int):
        # vocab_size is the number of word in vocabulary dict
        # seq_len is the sequence length/sentence length
        # embed_size is the embedding vector dimension
        super().__init__()
        # TODO:
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = AddPositionEncoding(embed_size, seq_len)
        self.blocks = nn.ModuleList([Block(embed_size, n_heads=n_heads, seq_len=seq_len,
                                    num_experts=num_experts, active_experts=active_experts) for _ in range(n_layers)])
        self.to_out = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.LayerNorm(embed_size),
            nn.Linear(embed_size, vocab_size),
        )
        self.seq_len = seq_len
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, inputs, labels=None):
        # labels: the (ground) true output
        # TODO: implement the forward function of the transformer

        # inputs:(batch_size, seq_len, )
        batch_size, seq_len, = inputs.shape
        # embedding:(batch_size, seq_len, embed_size)
        embedding = self.embedding(inputs)

        # add positional encoding
        pe_x = self.positional_encoding(embedding)
        x = self.norm(pe_x)

        # attens:(batch_size, seq_len, embed_size)
        for block in self.blocks:
            x = block(x)
        # logits:(batch_size, seq_len, vocab_size)
        logits = self.to_out(x)
        # compute the loss

        if labels is None:
            loss = None
        else:
            batch_size, seq_len, vocab_size = logits.shape
            logits = logits.view(batch_size * seq_len, vocab_size)
            labels = labels.view(batch_size * seq_len)
            loss = F.cross_entropy(logits, labels)
        return logits, loss

    def generate(self, inputs, max_new_tokens, tokenizer):
        inputs = tokenizer.encode(inputs).clone().detach().unsqueeze(0)
        device = next(self.parameters()).device
        inputs = inputs.to(device)
        nul_char = tokenizer.encode('#')[0]
        if inputs.size(1) > self.seq_len:
            inputs = inputs[:, :self.seq_len]
        elif inputs.size(1) < self.seq_len:
            # 左边padding
            inputs = F.pad(inputs, (self.seq_len - inputs.size(1), 0),
                           value=nul_char)
        generated = inputs  # shape: (batch_size=1, seq_len=21)
        for _ in range(max_new_tokens):
            if generated.size(1) > self.seq_len:
                generated_input = generated[:, -self.seq_len:]
            else:
                generated_input = generated
            logits, _ = self.forward(generated_input)
            last_logits = logits[:, -1, :]
            next_token_ids = torch.argmax(last_logits, dim=-1)
            next_token_ids = next_token_ids.unsqueeze(-1)
            generated = torch.cat([generated, next_token_ids], dim=1)
        return generated
