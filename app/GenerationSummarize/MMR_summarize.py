import re
import gensim
import os
import operator
import jieba
from app.tools.metrics import rouge_n
from app.tools.base_function import cosine_similar, sentence_to_vec
import warnings
import math
warnings.filterwarnings('ignore')


"""https://www.zhongxiaoping.cn/2019/02/25/SIF%E7%AE%97%E6%B3%95%E8%A7%A3%E6%9E%90/#wu-sif-suan-fa-dai-ma-bu-zou sif算法解析"""


if os.path.exists('/root/flag_server'):
    WORD_VECTOR = '/root/project/Project_01/static/save_file/save_mode2'
elif os.path.exists('/Volumes/Samsung_T5/'):
    WORD_VECTOR = "/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_word2Vec/THUCNews_word2Vec_128.model"
elif os.path.exists('/Users/haha'):
    WORD_VECTOR = '/Users/haha/Desktop/Project_01/static/save_file/save_mode2'
elif os.path.exists('/Users/bj'):
    WORD_VECTOR = '/Users/bj/Desktop/Documents/Project_01/static/save_file/save_mode2'

W2V_MODEL = gensim.models.Word2Vec.load(WORD_VECTOR)


class MMRSummarization:
    def __init__(self):
        self.word2vec = None
        self.text = None

    def sentence_to_vector(self, sentence_list):
        return sentence_to_vec(corpus=self.text, sentence_list=sentence_list, embedding_size=100, W2V_MODEL=self.word2vec)

    def compute_qdscore(self):
        sentence_list = self.split_sentence(self.text)  # [sen1, sen2, ...]
        sentence_list.append(''.join(sentence_list))
        QDsocre, index_solution = {}, {}
        sentence_list_vector = self.sentence_to_vector(sentence_list)
        special_vector = sentence_list_vector.pop(-1) # 长文本
        for inx, vector in enumerate(sentence_list_vector):
            qdscore = cosine_similar(vector, special_vector)
            QDsocre[sentence_list[inx]] = qdscore
            index_solution[sentence_list[inx]] = inx
        return QDsocre, index_solution, sentence_list_vector, sentence_list

    def split_sentence(self, sentence):
        sentence = ''.join(re.findall(r'[^\s]', sentence))
        pattern = re.compile('[。？?!！.]')
        split = pattern.sub(' ', sentence).split()
        return split

    def MMR(self, sentence=None, alpha=0.7, max_size=3):
        """
        main function: alpha * similar(Q, D) - (1-alpha) * max(similar(D_i, D_j))
        first step: compute Q, D similar
        second step: 选择第一步中得分最高的作为结果结果集中的一个子集
        third step: 遍历第一步的句子得分结果，得到每个句子的MMR得分。并取得最大MMR得分加入摘要集合中
        sentence: 文本内容
        stopwords: 停止词path
        embedding: 是否使用word2vec，默认onehot
        """
        self.text = sentence
        self.word2vec = W2V_MODEL
        QDsocre, index_solution, sentence_list_vector, sentence_list = self.compute_qdscore() #
        summarize_set = list()
        while max_size > 1:
            if not summarize_set:
                summarize_set.append(sentence_list[0])
                # max_score = sorted(QDsocre.items(), key=operator.itemgetter(1), reverse=True)[0][0]
                # summarize_set.append(max_score)
            MMRscore = {}
            for sen in QDsocre.keys():
                if sen not in summarize_set:
                    summarize_vectors = [sentence_list_vector[index_solution[summary_str]]
                                             for summary_str in summarize_set]
                    sen_vector = sentence_list_vector[index_solution[sen]]
                    mmrscore = alpha * QDsocre[sen] - ((1 - alpha) * max(
                        [cosine_similar(sen_vector, summarize_vector)
                         for summarize_vector in summarize_vectors]))
                    MMRscore[sen] = mmrscore
            max_mmrscore = sorted(MMRscore.items(), key=operator.itemgetter(1), reverse=True)[0][0]
            summarize_set.append(max_mmrscore)
            summarize_set_str = ''.join(summarize_set)
            max_size -= 1
        summarize = [(summ, index_solution[summ]) for summ in summarize_set]
        return '。'.join([s[0] for s in  sorted(summarize, key=lambda x: x[1])])


class TextAnalyse:
    def __init__(self, sentence):
        mmr = MMRSummarization()
        self.sentence = sentence
        self.sentence_list = mmr.split_sentence(sentence)
        self.sentence_words_list = [self.stop_word(dictwords=jieba.lcut(sen)) for sen in self.sentence_list]

    def stop_word(self, dictwords, path='../app/static/stopwords'):
        with open(path, 'r') as fr:
            stopwords = [line.replace('\n', '') for line in fr.read() if line != '\n']
        res = [word for word in dictwords if word not in stopwords]
        return res

    def analyse_tags_tfidf(self):
        # tf idf 提取关键词 count(sen_w)/count(sen) == tf;  count(doc)/count(doc_w) = idf;  tfidf = log(tf * idf)
        doc_count = len(self.sentence_words_list)
        word_counts = dict()
        seek = []
        for word_list in self.sentence_words_list:
            for word in word_list:
                if word not in seek:
                    seek.append(word)
                    if word in word_counts: word_counts[word] += 1
                    else: word_counts[word] = 1
        idfs_value = {word: math.log(doc_count / float(word_counts[word])) for word in word_counts}

    def analyse_tags_textrank(self):
        # textrank 提取关键词
        pass

    def topic(self):
        # 主题提取
        pass


if __name__ == '__main__':
    with open("../static/news.txt", "r", encoding='utf-8') as myfile:
        text = myfile.read().replace('\n', '')
    title_ = '中华人民共和国成立70周年庆祝活动总结会议在京举行 习近平亲切会见庆祝活动筹办工作有关方面代表'
    max_len = len(text) // 10
    mmr = MMRSummarization()
    res_summary = ''
    for sen, i in mmr.MMR(sentence=text, max_len=max_len):
        res_summary += ''.join(sen.split(' '))
        res_summary += '。'
    print(res_summary)


