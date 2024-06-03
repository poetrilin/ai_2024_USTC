

- Positional Encoding
$$
PE_{(pos, 2i)} = sin(pos/10000^{2i/d_{model}}) \\
PE_{(pos, 2i+1)} = cos(pos/10000^{2i/d_{model}})
$$

```python
    encodings = torch.zeros(seq_len, dim_embed)
    position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
    two_i = torch.arange(0, dim_embed, 2, dtype=torch.float32)  # 0, 2, 4, 6, 8, ..., 2*dim_embed
    div_term = torch.exp(two_i*(-math.log(10000.0)/dim_embed)) # 10000^{2i/d_{model}}
    encodings[:, 0::2] = torch.sin(position*div_term) 
    encodings[:, 1::2] = torch.cos(position*div_term)
    encodings = encodings.unsqueeze(1).requires_grad_(False)
    return encodings
```



```shell
I could pick my lance to
That the shall be this souls of the storment,
And therefore the selfs and the stand of the state,
And the shall be the sent the sent of the state,
And the stribung to the straight of the strange of the sea
child to me this so the seal the stand of the s
```