import torch
from d2l.torch import d2l
from torch import nn
from torch.optim import Adam
from dataset_to_vocab import get_loader
from vocab import *
import matplotlib.pyplot as plt
import math

Vocab()


# 计算训练时正确率
def get_train_accuracy(net, train_loader, device=None):
    if isinstance(net, nn.Module):
        net.eval()
    accurate_cnt = 0
    with torch.no_grad():
        for tokens_idx, sentiments in train_loader:
            tokens_idx = tokens_idx.to(device)
            sentiments = sentiments.to(device)
            sentiments_pred = net(tokens_idx)
            pred = sentiments_pred.data.max(1, keepdim=True)[1]  # 获取最大值的位置,[batch_size,1]
            accurate_cnt += pred.eq(sentiments.data.view_as(pred)).sum()

    return accurate_cnt / len(train_loader.dataset)


# 计算验证集、测试集正确率
def get_eval_accuracy(net, sentiments_pred, sentiments):
    if isinstance(net, nn.Module):
        net.eval()

    accurate_cnt = 0
    with torch.no_grad():
        pred = sentiments_pred.data.max(1, keepdim=True)[1]  # 获取最大值的位置,[batch_size,1]
        accurate_cnt += pred.eq(sentiments.data.view_as(pred)).sum()

    return accurate_cnt


def train_epoch(net, train_loader, loss, optimizer, device):
    metric = d2l.Accumulator(2)  # 训练损失之和,词元数量
    for tokens_idx, sentiments in train_loader:
        optimizer.zero_grad()
        tokens_idx = tokens_idx.to(device)
        sentiments = sentiments.to(device)
        sentiments_pred = net(tokens_idx)
        l = loss(sentiments_pred, sentiments).mean()
        l.backward()
        optimizer.step()

        metric.add(l * sentiments.numel(), sentiments.numel())

    # loss
    return metric[0] / metric[1]


def train(net, train_loader, num_epoch, checkpoint_path, device):
    loss = nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters())
    epoch_loss_list = []
    epoches_accuracy_list = []
    eval_loss_list = []
    eval_accuracy_list = []
    for epoch in range(num_epoch):
        net.train()
        epoch_loss = train_epoch(net, train_loader, loss, optimizer, device)
        epoch_accuracy = get_train_accuracy(net, train_loader, device)
        eval_loss, eval_accuracy = eval(net, 'validate', device)

        epoches_accuracy_list.append(epoch_accuracy)
        epoch_loss_list.append(epoch_loss)
        eval_accuracy_list.append(eval_accuracy)
        eval_loss_list.append(eval_loss)

        print(f'train: epoch {epoch + 1} loss: {epoch_loss} accuracy: {epoch_accuracy}')
        print(f'validate: epoch {epoch + 1} loss: {eval_loss} accuracy: {eval_accuracy}\n')

    # 保存模型参数
    checkpoint = {"model_state_dict": net.state_dict()}
    torch.save(checkpoint, checkpoint_path)
    show_descent(num_epoch, epoch_loss_list, epoches_accuracy_list, eval_loss_list, eval_accuracy_list)


def show_descent(num_epoch, *args):
    labels = ['train_loss', 'train_accuracy', 'validate_loss', 'validate_accuracy']
    x_axis = [i for i in range(num_epoch)]
    args_len = len(args)
    row = math.ceil(math.sqrt(args_len))
    col = math.ceil(args_len / row)
    i = 1
    for list in args:
        y_axis = list
        plt.subplot(row, col, i)
        plt.tight_layout()
        plt.plot(torch.tensor(x_axis).cpu().numpy(), torch.tensor(y_axis).cpu().numpy(), label=labels[i - 1])
        i = i + 1
        plt.xlabel('epoch')
        plt.legend()

    plt.show()


def eval(net, pattern, device):
    net.eval()
    eval_loss = 0
    accurate_cnt = 0
    loss = nn.CrossEntropyLoss()
    eval_loader, imdb_ds_eval = get_loader(batch_size=100, pattern=pattern)
    with torch.no_grad():
        for tokens_idx, sentiments in eval_loader:
            tokens_idx = tokens_idx.to(device)
            sentiments = sentiments.to(device)
            sentiments_pred = net(tokens_idx)
            eval_loss += (loss(sentiments_pred, sentiments).mean()) * sentiments.numel()
            accurate_cnt += get_eval_accuracy(net, sentiments_pred, sentiments)

    eval_loss /= len(eval_loader.dataset)
    accuracy = accurate_cnt / len(eval_loader.dataset)
    return eval_loss, accuracy


if __name__ == '__main__':
    epoch = 5
    loss1 = torch.tensor([0.56, 0.50, 0.45, 0.44, 0.39])
    loss1 = loss1.to(torch.device('cuda'))
    show_descent(epoch, loss1, loss1, loss1, loss1, loss1)
