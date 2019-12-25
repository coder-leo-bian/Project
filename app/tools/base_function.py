import numpy as np
import os
from sklearn.decomposition import PCA
import gensim
from tools.deal_text import cut
from collections import Counter
import pandas as pd

if os.path.exists('/root/flag_server'):
    WORD_VECTOR = '/root/project/Project_01/static/save_file/save_mode2'
elif os.path.exists('/Volumes/Samsung_T5/'):
    WORD_VECTOR = "/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_word2Vec/THUCNews_word2Vec_128.model"
elif os.path.exists('/Users/haha'):
    WORD_VECTOR = '/Users/haha/Desktop/Project_01/static/save_file/save_mode2'
elif os.path.exists('/Users/bj') :
    WORD_VECTOR = '/Users/bj/Desktop/Documents/Project_01/static/save_file/save_mode2'



def cosine_similar(vocter1, vocter2):
    up = np.dot(vocter1, vocter2)
    down = np.sqrt(np.sum(np.square(vocter1))) * np.sqrt(np.sum(np.square(vocter2)))
    return up / down

def sentence_to_vec(corpus, sentence_list, W2V_MODEL, embedding_size=128, a: float = 1e-3):
    # sif sentence embedding
    sentence_set = []
    words = cut(corpus)
    counters = Counter(words)
    for sentence in sentence_list:
        vs = np.zeros(embedding_size)  # add all word2vec values into one vector for the sentence
        word_list = cut(sentence)
        sentence_length = len(word_list)
        # 这个就是初步的句子向量的计算方法
        try:
            if word_list and sentence_length:
                for word in word_list:
                    if word in W2V_MODEL:
                        a_value = a / (a + (counters[word] / len(words)))  # smooth inverse frequency, SIF
                        vs = np.add(vs, np.multiply(a_value, W2V_MODEL[word]))  # vs += sif * word_vector
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


def save_to_csv(file_path=None, save_path=None, header= None, encoding='utf-8'):
    with open(file_path, 'r') as fr:
        lines = fr.readlines()
    lines = [line.replace('\n', '') for line in lines]
    df = pd.DataFrame(lines)
    df.to_csv(save_path, encoding=encoding, header=header)
