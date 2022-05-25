import torch
from torch import nn
import train_eval
import net
from dataset_to_vocab import get_loader
from vocab import Vocab

device = torch.device('cuda')
Vocab()

# 记得改模型保存路径
if __name__ == '__main__':
    # 超参
    # num_epoches batch_size max_word embedding_dim hidden_size num_layers 双向 dropout
    train_iter, imdb_ds = get_loader(batch_size=100, pattern='train')
    num_epochs = 8
    # 训练
    net_path = "./net/gru_model.pkl"
    state_dict_path = "./net/gru_model_state_dict.pkl"
    # net_path = "./net/lstm_model.pkl"
    # state_dict_path = "./net/lstm_model_state_dict.pkl"
    # rnn_layer = nn.RNN(input_size=200, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True, dropout=0.5)
    rnn_layer = nn.GRU(input_size=200, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True, dropout=0.5)
    # rnn_layer = nn.LSTM(input_size=200, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True, dropout=0.5)
    net = net.rnn_net(rnn_layer=rnn_layer, vocab_size=imdb_ds.get_vocab_size(), embedding_dim=200).to(device)
    train_eval.train(net, train_iter, num_epochs=num_epochs, net_path=net_path, state_dict_path=state_dict_path,
                     device=device)

    net = torch.load(net_path)
    loss, accuracy = train_eval.eval(net, 'test', device)
    print(f'\nloss is {loss} accuracy is {accuracy}')
