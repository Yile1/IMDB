import torch
from torch import nn
import torch.nn.functional as F

class rnn_net(nn.Module):
    def __init__(self, rnn_layer, vocab_size, embedding_dim, padding_idx=1):
        super(rnn_net, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=padding_idx).to()
        self.rnn_layer = rnn_layer
        self.fc1 = nn.Linear(64 * 2, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, tokens):
        token_embeded = self.embedding(tokens)  # token_embeded :[batch_size, max_word, 200]

        # output(seq_len, batch, hidden_size * num_directions)
        # hn(num_layers * num_directions, batch, hidden_size)
        # cn(num_layers * num_directions, batch, hidden_size)
        if not isinstance(self.rnn_layer, nn.LSTM):
            output, h_n = self.rnn_layer(token_embeded)
        else:
            output, (h_n, c_n) = self.rnn_layer(token_embeded)  # h_n :[4,batch_size,hidden_size]
        # out :[batch_size,hidden_size*2]
        out = torch.cat([h_n[-1, :, :], h_n[-2, :, :]], dim=-1)  # 拼接正向最后一个输出和反向最后一个输出
        out_fc1 = self.fc1(out)
        out_fc1_relu = F.relu(out_fc1)
        out_fc2 = self.fc2(out_fc1_relu)  # out :[batch_size,2]
        return F.log_softmax(out_fc2, dim=-1)