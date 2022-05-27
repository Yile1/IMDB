# coding=utf-8
import pandas as pd
import re

class IMDBDataSet:
    def __init__(self, pattern='train'):
        filePath = "./data/IMDB Dataset.csv"
        self.data = pd.read_csv(filePath, encoding='utf-8')

        if pattern == 'train':
            begin_line = 0
            end_line = 30000
        elif pattern == 'validate':
            begin_line = 30000
            end_line = 40000
        else:
            begin_line = 40000
            end_line = 50001

        self.reviews = self.data.iloc[begin_line:end_line, 0].tolist()
        self.sentiments = self.data.iloc[begin_line:end_line, 1].tolist()
        self.sentiments = [1 if sentiment == 'positive' else 0 for sentiment in self.sentiments]
        self.tokens = []
        self.tokenize()


    def __getitem__(self, idx):
        return self.tokens[idx], self.sentiments[idx]

    def __len__(self):
        return len(self.reviews)

    def tokenize(self):
        fileters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.', '/', ':', ';', '<', '=', '>',
                    '\?', '@', '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”',
                    '“', ]
        # 把大写转化为小写
        self.reviews = [review.lower() for review in self.reviews]
        for review in self.reviews:
            review = re.sub("<br />", " ", review)
            review = re.sub("|".join(fileters), " ", review)
            self.tokens.append([i for i in review.split(" ") if len(i) > 0])

    def get_tokens(self):
        return self.tokens

    def get_sentiments(self):
        return self.sentiments

