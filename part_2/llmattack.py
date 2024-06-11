import gc
import os
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from livelossplot import PlotLosses  # pip install livelossplot

# if torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
device = torch.device("cpu")
# Set the random seed for NumPy
np.random.seed(20)

# Set the random seed for PyTorch
torch.manual_seed(20)

# If you are using CUDA (i.e., a GPU), also set the seed for it
torch.cuda.manual_seed_all(20)

model_path = "models/TinyStories-33M/"
if os.path.exists(model_path):
    print("Model exists")
else:
    raise FileExistsError("Model not found")

target = "This is great! I love living on the wild side!"

# 如果你的电脑运行起来比较慢，考虑适当调整下面的参数

num_steps = 100  # 500
adv_string_init = "!"*200
adv_prefix = adv_string_init
batch_size = 512

topk = 256

if device == "cuda":
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(device).eval()
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
    ).to(device).eval()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def get_embedding_matrix(model):
    return model.transformer.wte.weight


def get_embeddings(model, input_ids):
    return model.transformer.wte(input_ids)


def token_gradients(model, input_ids, input_slice, target_slice, loss_slice):
    """
    Computes gradients of the loss with respect to the coordinates.

    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    input_ids : torch.Tensor
        The input sequence in the form of token ids. Shape: (seq_len,)
    input_slice : slice
        The slice of the input sequence for which gradients need to be computed.  Shape:
    target_slice : slice
        The slice of the input sequence to be used as targets.
    loss_slice : slice
        The slice of the logits to be used for computing the loss.

    Returns
    -------
    torch.Tensor
        The gradients of each token in the input_slice with respect to the loss.
    """

    embed_weights = get_embedding_matrix(model)

    # ==================需要你实现的部分==================
    # TODO: input_ids 是整个输入的 token_id, 但是我们只需要计算 input_slice 的梯度
    # 1. 先定义一个 zero tensor，shape 为 (input_slice_len, vocab_size)
    # vocab_size 是词表大小，思考词表大小对应模型的什么矩阵的哪一维
    vocab_size, _ = embed_weights.shape
    one_hot = torch.zeros(
        input_ids[input_slice].shape[0], vocab_size, device=model.device, dtype=embed_weights.dtype
    )

    # TODO: 2. 将 one_hot 中对应的 token_id 的位置置为 1
    one_hot = one_hot.scatter_(1,
                               input_ids[input_slice].unsqueeze(1),
                               torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype))
    one_hot.requires_grad = True

    # TODO: 3. 将 one_hot 乘以 embedding 矩阵，得到 input_slice 的 embedding，注意我们需要梯度
    # (1, seq_len, embedding_size)
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)

    embeds = get_embeddings(model, input_ids.unsqueeze(
        0)).detach()  # (1, seq_len, embedding_size)

    # TODO: 4. 用 input_embeds 替换 embedding 的对应部分（可以拼接），拿到 logits 之后和 target 进行 loss 计算

    full_embeds = torch.cat(
        [
            embeds[:, :input_slice.start, :],
            input_embeds,
            embeds[:, input_slice.stop:, :]
        ],
        dim=1)
    # (1, seq_len, vocab_size)
    logits = model(inputs_embeds=full_embeds).logits
    targets = input_ids[target_slice]
    loss = nn.CrossEntropyLoss()(logits[0, loss_slice, :], targets)

    # ==================需要实现的部分结束==================

    loss.backward()

    grad = one_hot.grad.clone()
    grad = grad / grad.norm(dim=-1, keepdim=True)

    return grad


def sample_control(control_toks, grad, batch_size, topk=topk):
    """
    Use the gradients to sample new control tokens.
    """
    # 拿到梯度之后，我们可以使用梯度来采样新的控制 token
    # ==================需要你实现的部分==================
    control_toks = control_toks.to(grad.device)
    # TODO: 重复 batch_size 次（随机采样的次数） -> (batch_size, len(control_toks))
    original_control_toks = control_toks.repeat(batch_size, 1)

    # TODO: 生成 batch_size 个新的 token 位置作为采样的位置，允许复选
    new_token_pos = torch.arange(
        0,
        len(control_toks),
        len(control_toks) / batch_size,
        device=grad.device
    )
    new_token_pos = new_token_pos.type(torch.int64)

    # TODO: 利用梯度的 topk 来获取每个 token 位置上梯度最大的 topk 个 token 的索引
    # https://pytorch.org/docs/stable/generated/torch.topk.html
    top_indices = (-grad).topk(topk, dim=1).indices
    # TODO: 从 top_indices 中的 new_token_pos （作为 index）随机采样一个 topk token 的索引，作为新的 token
    new_token_val = torch.gather(
        top_indices[new_token_pos], 1,
        torch.randint(0, topk, (batch_size, 1),
                      device=grad.device)
    )

    # TODO: 得到新的 control tokens
    new_control_toks = original_control_toks.scatter_(
        1, new_token_pos.unsqueeze(-1), new_token_val)

    # ==================需要实现的部分结束==================

    return new_control_toks


def get_filtered_cands(tokenizer, control_cand, filter_cand=True, curr_control=None):
    cands, count = [], 0
    for i in range(control_cand.shape[0]):
        decoded_str = tokenizer.decode(
            control_cand[i], skip_special_tokens=True)
        if filter_cand:
            if decoded_str != curr_control \
                    and len(tokenizer(decoded_str, add_special_tokens=False).input_ids) == len(control_cand[i]):
                cands.append(decoded_str)
            else:
                count += 1
        else:
            cands.append(decoded_str)

    if filter_cand:
        cands = cands + [cands[-1]] * (len(control_cand) - len(cands))
    return cands


def get_logits(*, model, tokenizer, input_ids, control_slice, test_controls, return_ids=False, batch_size=512):

    if isinstance(test_controls[0], str):
        max_len = control_slice.stop - control_slice.start
        test_ids = [
            torch.tensor(tokenizer(
                control, add_special_tokens=False).input_ids[:max_len], device=model.device)
            for control in test_controls
        ]
        pad_tok = 0
        while pad_tok in input_ids or any([pad_tok in ids for ids in test_ids]):
            pad_tok += 1
        nested_ids = torch.nested.nested_tensor(test_ids)
        test_ids = torch.nested.to_padded_tensor(
            nested_ids, pad_tok, (len(test_ids), max_len))
    else:
        raise ValueError(
            f"test_controls must be a list of strings, got {type(test_controls)}")

    if not (test_ids[0].shape[0] == control_slice.stop - control_slice.start):
        raise ValueError((
            f"test_controls must have shape "
            f"(n, {control_slice.stop - control_slice.start}), "
            f"got {test_ids.shape}"
        ))
    locs = torch.arange(control_slice.start, control_slice.stop).repeat(
        test_ids.shape[0], 1).to(model.device)  # (n, control_slice_len)
    ids = torch.scatter(
        input_ids.unsqueeze(0).repeat(test_ids.shape[0], 1).to(model.device),
        1,
        locs,
        test_ids
    )
    if pad_tok >= 0:
        attn_mask = (ids != pad_tok).type(ids.dtype)
    else:
        attn_mask = None

    if return_ids:
        del locs, test_ids
        gc.collect()
        return forward(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size), ids
    else:
        del locs, test_ids
        logits = forward(model=model, input_ids=ids,
                         attention_mask=attn_mask, batch_size=batch_size)
        del ids
        gc.collect()
        return logits


def forward(*, model, input_ids, attention_mask, batch_size=512):

    logits = []
    for i in range(0, input_ids.shape[0], batch_size):

        batch_input_ids = input_ids[i:i+batch_size]
        if attention_mask is not None:
            batch_attention_mask = attention_mask[i:i+batch_size]
        else:
            batch_attention_mask = None

        logits.append(model(input_ids=batch_input_ids,
                      attention_mask=batch_attention_mask).logits)

        gc.collect()

    del batch_input_ids, batch_attention_mask

    return torch.cat(logits, dim=0)


def target_loss(logits, ids, target_slice):
    crit = nn.CrossEntropyLoss(reduction='none')
    loss_slice = slice(target_slice.start-1, target_slice.stop-1)
    loss = crit(logits[:, loss_slice, :].transpose(1, 2), ids[:, target_slice])
    return loss.mean(dim=-1)


def is_success(prompt):
    # ==================需要你实现的部分==================
    # TODO: 这部分需要你实现判断是否攻击成功的逻辑。用你的 prompt 进行一次推理，
    # 然后判断输出中是否包含 target，如果包含则返回 True，否则返回 False
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_length=100,
                            num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # ==================需要实现的部分结束==================
    print(output_text)
    if output_text.lower().find(target.lower()) != -1:
        return True
    return False


plotlosses = PlotLosses()

# ==========需要你实现的部分==========
# TODO: 定义 adv_slice, target_slice, loss_slice
adv_slice = slice(
    0, len(tokenizer(adv_prefix, add_special_tokens=False).input_ids))
target_slice = slice(adv_slice.stop, adv_slice.stop +
                     len(tokenizer(target, add_special_tokens=False).input_ids))
loss_slice = slice(target_slice.start-1, target_slice.stop-1)
# ==========需要实现的部分结束==========

for i in range(num_steps):

    input_ids = tokenizer.encode(
        adv_prefix+target, add_special_tokens=False, return_tensors='pt').squeeze(0)
    input_ids = input_ids.to(device)

    coordinate_grad = token_gradients(model,
                                      input_ids,
                                      adv_slice,
                                      target_slice,
                                      loss_slice)
    with torch.no_grad():
        adv_prefix_tokens = input_ids[adv_slice].to(device)
        new_adv_prefix_toks = sample_control(adv_prefix_tokens,
                                             coordinate_grad,
                                             batch_size)
        new_adv_prefix = get_filtered_cands(tokenizer,
                                            new_adv_prefix_toks,
                                            filter_cand=True,
                                            curr_control=adv_prefix)  # [batch_size, len(adv_prefix)]
        logits, ids = get_logits(model=model,
                                 tokenizer=tokenizer,
                                 input_ids=input_ids,
                                 control_slice=adv_slice,
                                 test_controls=new_adv_prefix,
                                 return_ids=True,
                                 batch_size=batch_size)
        losses = target_loss(logits, ids, target_slice)
        best_new_adv_prefix_id = losses.argmin()
        best_new_adv_prefix = new_adv_prefix[best_new_adv_prefix_id]
        current_loss = losses[best_new_adv_prefix_id]
        adv_prefix = best_new_adv_prefix
    plotlosses.update({'Loss': current_loss.detach().cpu().numpy()})
    plotlosses.send()
    plotlosses.update({'Loss': current_loss.detach().cpu().numpy()})
    plotlosses.send()
    print(f"Current Prefix:{best_new_adv_prefix}", end='\r')
    if is_success(best_new_adv_prefix):
        break
    del coordinate_grad, adv_prefix_tokens
    gc.collect()
    torch.cuda.empty_cache()

if is_success(best_new_adv_prefix):
    print("SUCCESS:", best_new_adv_prefix)
else:
    raise ValueError("Attack failed")
