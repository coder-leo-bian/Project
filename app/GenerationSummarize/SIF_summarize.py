
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
import re
import gensim
import os
# from textrank4zh import TextRank4Sentence
from tools.deal_text import cut
from tools.base_function import cosine_similar
import warnings
"""https://www.zhongxiaoping.cn/2019/02/25/SIF%E7%AE%97%E6%B3%95%E8%A7%A3%E6%9E%90/#wu-sif-suan-fa-dai-ma-bu-zou sif算法解析"""

warnings.filterwarnings('ignore')

if os.path.exists('/root/.flag'):
    WORD_VECTOR = '/root/word2vec/THUCNews_word2Vec_128.model'
elif os.path.exists('/Volumes/Samsung_T5/'):
    WORD_VECTOR = "/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_word2Vec/THUCNews_word2Vec_128.model"
elif os.path.exists('/Users/haha'):
    WORD_VECTOR = '/Users/haha/Desktop/Project_01/static/save_file/save_mode2'
elif os.path.exists('/Users/bj') :
    WORD_VECTOR = '/Users/bj/Desktop/Documents/Project_01/static/save_file/save_mode2'

W2V_MODEL = gensim.models.Word2Vec.load(WORD_VECTOR)


class SIFSummarization:
    def __init__(self, doc_, title_=None):
        self.model_word_vector = W2V_MODEL
        self.doc_ = doc_
        self.title_ = title_
        self.words = cut(doc_) # 对整篇文章进行分词
        self.counter = Counter(self.words)   # 对分词结果进行Counter，方便计算词频

    def get_word_frequency(self, word):
        return self.counter[word] / len(self.words)

    def sentence_to_vec(self, sentence_list, embedding_size=100, a: float = 1e-3):
        sentence_set = []
        for sentence in sentence_list:
            vs = np.zeros(embedding_size)  # add all word2vec values into one vector for the sentence
            word_list = cut(sentence)
            sentence_length = len(word_list)
            # 这个就是初步的句子向量的计算方法
            try:
                if word_list and sentence_length:
                    for word in word_list:
                        if word in self.model_word_vector:
                            a_value = a / (a + self.get_word_frequency(word))  # smooth inverse frequency, SIF
                            vs = np.add(vs, np.multiply(a_value, self.model_word_vector[word]))  # vs += sif * word_vector
                        else:
                            continue

                    vs = np.divide(vs, sentence_length)  # weighted average
                    sentence_set.append(vs)  # add to our existing re-calculated set of sentences
                else:
                    continue
            except:
                continue
        # calculate PCA of this sentence set,计算主成分
        pca = PCA()
        # 使用PCA方法进行训练
        pca.fit(np.array(sentence_set))
        # 返回具有最大方差的的成分的第一个,也就是最大主成分,
        # components_也就是特征个数/主成分个数,最大的一个特征值
        u = pca.components_[0]  # the PCA vector
        # 构建投射矩阵
        u = np.multiply(u, np.transpose(u))  # u x uT
        # judge the vector need padding by wheather the number of sentences less than embeddings_size
        # 判断是否需要填充矩阵,按列填充
        if len(u) < embedding_size:
            for i in range(embedding_size - len(u)):
                # 列相加
                u = np.append(u, 0)  # add needed extension for multiplication below

        # resulting sentence vectors, vs = vs -u x uT x vs
        sentence_vecs = []
        for vs in sentence_set:
            sub = np.multiply(u, vs)
            sentence_vecs.append(np.subtract(vs, sub))
        return sentence_vecs

    def compute_similar_by_cosine(self, sentence_vector_list):
        doc_sentence = sentence_vector_list.pop(-1)
        square_doc = np.sqrt(np.sum(np.square(doc_sentence)))
        similar = []
        for i, sentence_vector in enumerate(sentence_vector_list):
            up = np.dot(sentence_vector, doc_sentence)
            down = np.sqrt(np.sum(np.square(sentence_vector))) + square_doc
            similar.append(up / down)
        similar_ = {i: v for i, v in enumerate(similar)}
        return similar_

    def main(self, flags=1):
        """
        :param flags: 1 使用标题匹配文本相似度；其他值使用sif，每个句子和长文本进行相似度计算
        :return:
        """

        sentence = ''.join(re.findall(r'[^\s]', self.doc_))
        pattern = re.compile('[。？?!！.]')
        sentence_list = pattern.sub(' ', sentence).split()

        # sentence_list = self.doc_.split('。')

        if flags == 1:
            sentence_list.append(self.title_) # 长文本按句号切分句子

        else:
            sentence_list.append(self.doc_) # 将长文本作为句子
        sentence_vector_list = self.sentence_to_vec(sentence_list, embedding_size=128)  # 获得每个句子的句子向量
        special_vector = sentence_vector_list.pop(-1)  # 取出最后一个(标题或长文本)句子向量

        similar_ = []
        for vector in sentence_vector_list:
            similar_.append(cosine_similar(vector, special_vector))

        similar_ = {i: v for i, v in enumerate(similar_)} # 对应cosine value 和 index
        similar_ = sorted(similar_.items(), key=lambda x: x[1], reverse=True)  # 根据cosine value排序
        similar_ = sorted(similar_, key=lambda x: x[1], reverse=True) # 根据

        sorted_score = [i for i, v in similar_[: 3]]  # 取出前3个cosine value 最大的索引
        result = ''
        sorted_score.sort()
        for i in sorted_score:
            result += sentence_list[i]
            result += '。'
        return result, sorted_score, sentence_list
