'''可视化attention map'''
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


def plot_attention_map(attention_map, save_path: str = None):
    """
    Plot the attention map
    Args:
        attention_map: 2D numpy array
        save_path: str, path to save the plot
    """
    plt.figure(figsize=(15, 15))
    plt.imshow(attention_map, cmap='viridis')
    plt.colorbar()
    plt.title('Attention Map')

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == '__main__':

    if os.path.exists('./output/attn_list.pth'):
        attn_list = torch.load('./output/attn_list.pth')
        # torch to numpy
        attn_list = [attn.cpu().detach().numpy() for attn in attn_list]
        # 取最后 20x20个 grid
        attn_list = [attn[-20:, -20:] for attn in attn_list]
        for i, attn in tqdm(enumerate(attn_list)):
            plot_attention_map(attn, f'./output/img/attention_map_{i}.png')

    else:
        raise FileNotFoundError('No model found, please train the model first')
