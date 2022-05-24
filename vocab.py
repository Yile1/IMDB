import collections
import os
import pickle


class Vocab:
    UNK = "<UNK>"  # 表示未知字符
    PAD = "<PAD>"  # 填充符

    @property
    def token_freqs(self):
        return self._token_freqs
    
    def __init__(self, tokens=None, min_freq=0):
        self.dict = {
            self.UNK: 0,
            self.PAD: 1
        }
        if tokens is None:
            tokens = []
        # 按出现频率排序
        freqs = self.count_freq(tokens)
        self._token_freqs = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            self.dict[token] = len(self.dict)
        
        self.idx_to_token = dict(zip(self.dict.values(), self.dict.keys()))
    
    def __len__(self):
        return len(self.dict)

    def __getitem__(self, token):
        # 由token（key）得到index
        return self.dict[token]

    # 由index得到token
    def get_token(self, idx):
        return self.value_to_token[idx]

    def count_freq(self, tokens):
        if len(tokens) == 0 or isinstance(tokens[0], list):
            # 将词元列表展平成一个列表
            tokens = [token for word_list in tokens for token in word_list]
        return collections.Counter(tokens)
    
    def token_to_idx(self, review, max_word=None):
        if len(review) > max_word:
            review = review[:max_word]
        else:
            review = review + [self.PAD] * (max_word - len(review))

        return [self.dict.get(i, 1) for i in review]

if __name__ == "__main__":
    from dataset import *
    tokens = [["叶光", "叶光", "大", "冤种"],
             ["叶光", "大便", "吃", "大便"]]
    vocab = Vocab(tokens, 1)
    print(vocab.dict)
    review = ["大", "冤种", "叶光", "天天", "吃", "大便"]
    print(vocab.token_to_idx(review, 6))

    # tokens = IMDBDataSet(pattern='train').get_tokens()
    # vocab = Vocab(tokens)
    # if not os.path.exists("./vocab"):
    #     os.makedirs("./vocab")
    # pickle.dump(vocab, open("./vocab/vocab.pkl", "wb"))
    # voc_model = pickle.load(open("./vocab/vocab.pkl", "rb"))
    # print(voc_model)