import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np

from dataset import Tokenizer, ShakespeareDataset, create_dataloader
from moe import SparseMoETransformer

LEARNING_RATE = 5e-4
WEIGHT_DECAY = 2e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, dataloader, epoch, device):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    model.train()
    total_loss = 0
    for i, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
        # TODO: implement the training process, and compute the training loss and validation loss
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        logits, loss = model(inputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f'Epoch {epoch} Loss: {total_loss / len(dataloader)}')
    return total_loss / len(dataloader)


def validate(model, dataloader, epoch, device):
    model.eval()
    # TODO: 实现验证函数。与训练函数类似，但不需要计算梯度。
    total_loss = 0
    for i, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        logits, loss = model(inputs, targets)
        total_loss += loss.item()
    print(f'Epoch {epoch} Loss: {total_loss / len(dataloader)}')
    return total_loss / len(dataloader)


# 训练模型
def run(model, train_dataloader, valid_dataloader, device, epochs=10):
    train_loss_list = []
    valid_loss_list = []
    for epoch in range(epochs):
        train_loss = train(model, train_dataloader, epoch, device)
        valid_loss = validate(model, valid_dataloader, epoch, device)
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        print(
            f'Epoch {epoch} Train Loss: {train_loss}, Valid Loss: {valid_loss}')
    return train_loss_list, valid_loss_list


if __name__ == '__main__':
    print(f"device: {device}")
    BATCH_SIZE = 128
    EPOCHS = 10
    token_path = 'input.txt'
    tokenizer = Tokenizer(dataPath=token_path)
    SEQ_LEN = 256
    train_oader, val_loader = create_dataloader(
        'input.txt', tokenizer, chunk_size=SEQ_LEN, batch_size=BATCH_SIZE)
    model = SparseMoETransformer(vocab_size=len(tokenizer.char2index), seq_len=SEQ_LEN,
                                 embed_size=64, n_layers=3, n_heads=8, num_experts=8, active_experts=2).to(device)
    # 判断是否有训好的模型
    if os.path.exists('./output/model.pth'):
        model.load_state_dict(torch.load('./output/model.pth'))
    else:
        print('No model found, training from scratch')
        train_loss_list, valid_loss_list = run(
            model, train_oader, val_loader, device=device, epochs=EPOCHS)
        # 保存模型
        torch.save(model.state_dict(), '/output/model.pth')
        # 存list为np,因为是丢远程跑所以存下来
        np.save('/output/train_loss_list.npy', np.array(train_loss_list))
        np.save('/output/valid_loss_list.npy', np.array(valid_loss_list))
    pad_char = '#'
    ans = tokenizer.decode(model.generate(
        "I could pick my lance ", max_new_tokens=256, tokenizer=tokenizer)[0].tolist())
    print(ans.strip(pad_char))
