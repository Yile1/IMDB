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
    # num_epoch batch_size max_word embedding_dim hidden_size num_layers 双向 dropout
    num_epoch_lstm = 5
    # num_epoch_gru = 5
    batch_size = 100
    # num_epoch_rnn = 4
    # batch_size_rnn = 50
    train_iter, imdb_ds = get_loader(batch_size=batch_size, pattern='train')
    # 训练
    checkpoint_path = "./net/lstm_checkpoint.pkl"
    # 两层rnn方便加dropout
    # rnn_layer = nn.RNN(input_size=200, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True, dropout=0.5)
    # rnn_layer = nn.GRU(input_size=200, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True, dropout=0.5)
    rnn_layer = nn.LSTM(input_size=200, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True, dropout=0.5)
    net = net.rnn_net(rnn_layer=rnn_layer, vocab_size=imdb_ds.get_vocab_size(), embedding_dim=200).to(device)
    train_eval.train(net, train_iter, num_epoch=num_epoch_lstm, checkpoint_path=checkpoint_path,
                     device=device)

    net = torch.load(checkpoint_path)['model']
    loss, accuracy = train_eval.eval(net, 'test', device)
    print(f'\nloss is {loss} accuracy is {accuracy}')
