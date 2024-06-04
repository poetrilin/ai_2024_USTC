import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from MulAttn import MultiHeadAttention


class Expert(nn.Module):
    '''
    FFN module, which is a two-layer feed-forward network with activation function.
    '''

    def __init__(self, embed_size: int, dropout_prob: float = 0.1):
        super().__init__()

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
        x, attn = self.attn(inputs)
        x = x + resi  # add residual connection
        x = self.layer_norm(x)
        add_normed = x

        ##### MoE #####
        x = self.moe(x)
        x = x + add_normed
        x = self.layer_norm(x)
        return x, attn
