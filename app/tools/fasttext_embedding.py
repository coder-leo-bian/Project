import gensim
from gensim.models import FastText
from gensim.models.word2vec import LineSentence
import os


class TrainWordEmbedding:
    def __init__(self, data_path='/Users/bj/Desktop/Documents/project_01_data/cut_result'):
        self.data_path = data_path

    def train_embedding(self, save_path):
        model = FastText(LineSentence(self.data_path), window=5, size=35, iter=10, min_count=1)
        model.save(save_path)


if __name__ == '__main__':
    embedding = TrainWordEmbedding()
    embedding.train_embedding('/Users/bj/Desktop/Documents/project_01_data/project_02_fasttext_model/')
