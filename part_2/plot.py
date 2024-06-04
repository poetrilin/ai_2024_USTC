import numpy as np
import matplotlib.pyplot as plt
import torch


def plot(train_loss_list, valid_loss_list, save_path: str = None):
    plt.figure()
    plt.plot(train_loss_list, label='train loss')
    plt.plot(valid_loss_list, label='valid loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == '__main__':
    # 读取训练和验证的loss
    train_loss_list = np.load('./output/train_loss_list.npy')
    valid_loss_list = np.load('./output/valid_loss_list.npy')
    print(train_loss_list)
    save_path = './output/loss_plot.png'
    plot(train_loss_list, valid_loss_list, save_path)
