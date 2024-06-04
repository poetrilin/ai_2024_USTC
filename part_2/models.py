
import torch
import torch.nn as nn
import torch.nn.functional as F
from PosEncoding import AddPositionEncoding


class SparseMoETransformer(nn.Module):
    ''' 
    Transformer decoder, consist of 
    token embedding layer and position_embedding(position_embedding 可以理解为对位置编码，感兴趣的同学可以查阅原文，这里可以看为vocab_len = seq_len的Embedding)
    a stack of Transformer basic block
    a layernorm and output linear layer
    '''

    def __init__(self,
                 vocab_size: int, seq_len: int,
                 embed_size: int,
                 n_layers: int,
                 n_heads: int,
                 num_experts: int, active_experts: int,
                 verbose: bool = False):
        # vocab_size is the number of word in vocabulary dict
        # seq_len is the sequence length/sentence length
        # embed_size is the embedding vector dimension
        super().__init__()
        # TODO:
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = AddPositionEncoding(embed_size, seq_len)
        self.blocks = nn.ModuleList([Block(embed_size, n_heads=n_heads, seq_len=seq_len,
                                    num_experts=num_experts, active_experts=active_experts) for _ in range(n_layers)])
        self.attention_list = []
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

        self.attention_list = []
        # attens:(batch_size, seq_len, embed_size)
        for block in self.blocks:
            x, attn = block(x)
            self.attention_list.append(attn)
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

    def get_attn_list(self):
        return self.attention_list

    def top_k_sampling(logits: torch.Tensor, k: int) -> int:
        """
        Apply Top-k sampling to select the next token.

        Args:
            logits (torch.Tensor): Logits of the model's output.
            k (int): Number of top tokens to consider for sampling.

        Returns:
            int: The index of the sampled token.
        """
        values, indices = torch.topk(logits, k)
        distribution = torch.softmax(values, dim=-1)
        sampled_index = torch.multinomial(distribution, 1).item()
        return indices[sampled_index].item()

    def generate(
        self,
        inputs: str,
        max_new_tokens: int,
        tokenizer: Tokenizer,
        k: int = 5,
        visualize: bool = False,
    ) -> str:
        """
        Generate text using a Top-k sampling strategy.

        Args:
            inputs (str): The input text to the model.
            max_new_tokens (int): The maximum number of new tokens to generate.
            tokenizer: Tokenizer): The tokenizer used to encode and decode text.
            k (int): Number of top tokens to consider for sampling (default is 5).

        Returns:s
            str: The generated text.
        """
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
