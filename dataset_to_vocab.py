# 将dataset中的token弄出来
# 每个token中的每一个句子都利用vocab转为index
# dataloader
import torch
import dataset
from torch.utils.data import DataLoader
from vocab import *

Vocab()

class transform:
    @property
    def vocab(self):
        return self._vocab

    def __init__(self, pattern):
        self.ds = dataset.IMDBDataSet(pattern)
        self.tokens = self.ds.get_tokens()
        self.sentiments = self.ds.get_sentiments()
        self._vocab = pickle.load(open("./vocab/vocab.pkl", "rb"))

        self.tokens_idx = []
        for token in self.tokens:
            self.tokens_idx.append(self._vocab.token_to_idx(token, max_word=200))

    def __getitem__(self, idx):
        return self.tokens_idx[idx], self.sentiments[idx]

    def __len__(self):
        return len(self.sentiments)

    def get_vocab_size(self):
        return len(self._vocab.dict)


def get_loader(batch_size, pattern):
    imdb_ds = transform(pattern=pattern)
    # 调用__getitem__函数
    return DataLoader(imdb_ds, batch_size=batch_size, shuffle=False, collate_fn=my_collate), imdb_ds


# 每个句子不一样会报错
def my_collate(batch):
    token, sentiment = zip(*batch)
    token = torch.LongTensor(token)
    sentiment = torch.LongTensor(sentiment)
    return token, sentiment


if __name__ == "__main__":
    data_loader, imdb_ds = get_loader(pattern='train')
    for tokens_idx, sentiments in data_loader:
        print(tokens_idx)
        print(sentiments)
    print(1)
