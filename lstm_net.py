import torch
from d2l.torch import d2l
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from dataset_to_vocab import get_loader
from vocab import *

device = torch.device('cuda')
Vocab()

class LSTM_net(nn.Module):
    def __init__(self, vocab_size, embedding_dim, padding_idx=1):
        super(LSTM_net, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=padding_idx).to()
        self.lstm = nn.LSTM(input_size=200, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True, dropout=0.5)
        self.fc1 = nn.Linear(64 * 2, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, tokens):
        token_embeded = self.embedding(tokens)  # input embeded :[batch_size, max_word, 200]

        output, (h_n, c_n) = self.lstm(token_embeded)  # h_n :[4,batch_size,hidden_size]
        # out :[batch_size,hidden_size*2]
        out = torch.cat([h_n[-1, :, :], h_n[-2, :, :]], dim=-1)  # 拼接正向最后一个输出和反向最后一个输出
        out_fc1 = self.fc1(out)
        out_fc1_relu = F.relu(out_fc1)
        out_fc2 = self.fc2(out_fc1_relu)  # out :[batch_size,2]
        return F.log_softmax(out_fc2, dim=-1)


def train_epoch(net, train_iter, loss, optimizer, device):
    metric = d2l.Accumulator(2)  # 训练损失之和,词元数量
    for tokens_idx, sentiments in train_iter:
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



def train(net, train_iter, num_epochs, device):
    loss = nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters())
    for i in range(num_epochs):
        epoch_loss = train_epoch(net, train_iter, loss, optimizer, device)
        print(f'epoch : {i} loss: {epoch_loss}\n')

    net_path = "./net/lstm_model.pkl"
    torch.save(net, net_path)
    # 保存模型参数
    state_dict_path = "./net/lstm_model_state_dict.pkl"
    net_state_dict = net.state_dict()
    torch.save(net_state_dict, state_dict_path)


def predict(net):
    test_loss = 0
    accurate_cnt = 0
    net.eval()
    loss = nn.CrossEntropyLoss()
    test_iter, imdb_ds_test = get_loader(pattern='test')
    with torch.no_grad():
        for tokens_idx, sentiments in test_iter:
            tokens_idx = tokens_idx.to(device)
            sentiments = sentiments.to(device)
            sentiments_pred = net(tokens_idx)
            test_loss += loss(sentiments_pred, sentiments).mean()
            pred = sentiments_pred.data.max(1, keepdim=True)[1]  # 获取最大值的位置,[batch_size,1]
            accurate_cnt += pred.eq(sentiments.data.view_as(pred)).sum()
    test_loss /= len(test_iter.dataset)
    print('\nTest set: Avg. loss: {:.4f}, \n'
          'Accuracy {}/{}: ({:.2f}%)\n'.format(test_loss, accurate_cnt, len(test_iter.dataset), 100. * accurate_cnt / len(test_iter.dataset)))


if __name__ == '__main__':
    train_iter, imdb_ds = get_loader(pattern='train')
    num_epochs = 5
    # 训练
    net = LSTM_net(imdb_ds.get_vocab_size(), embedding_dim=200)
    net = net.to(device)
    #train(net, train_iter, num_epochs=num_epochs, device=device)

    net_path = "./net/lstm_model.pkl"
    net = torch.load(net_path)
    predict(net)