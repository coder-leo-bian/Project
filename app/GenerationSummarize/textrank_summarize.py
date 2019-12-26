import networkx
import re
import gensim
import math
import jieba
import warnings
from tools.deal_text import cut, postags_words
"""https://www.zhongxiaoping.cn/2019/02/25/SIF%E7%AE%97%E6%B3%95%E8%A7%A3%E6%9E%90/#wu-sif-suan-fa-dai-ma-bu-zou sif算法解析"""
warnings.filterwarnings('ignore')


class TextRankSummarization:
    """
    利用textRank实现的文本摘要
    """
    def __init__(self):
        pass

    def get_connect_graph_by_text_rank(self, tokenized_text='', window=3):
        """building word connect graph """
        keywords_graph = networkx.Graph()
        tokeners = tokenized_text.split()
        for ii, t in enumerate(tokeners):
            word_tuples = [(tokeners[connect], t) for connect in range(ii - window, ii + window) if connect >= 0 and connect < len(tokeners)]
            keywords_graph.add_edges_from(word_tuples)
        return keywords_graph

    def split_sentence(self, sentence):
        """split"""
        sentence = ''.join(re.findall(r'[^\s]', sentence))
        pattern = re.compile('[。？?!！.]')
        split = pattern.sub(' ', sentence).split()
        return split

    def get_summarization_simple_with_text_rank(self, text, constrain=200):
        return self.get_summarization_simple(text, self.sentence_ranking_by_text_ranking, constrain)

    def sentence_ranking_by_text_ranking(self, split_sentence):
        """计算sentece的pagerank，并根据值的大小进行排序"""
        sentence_graph = self.get_connect_graph_by_text_rank(' '.join(split_sentence))
        ranking_sentence = networkx.pagerank(sentence_graph)
        ranking_sentence = sorted(ranking_sentence.items(), key=lambda x: x[1], reverse=True)
        return ranking_sentence

    def get_summarization_simple(self, text, score_fn, consitrain=200):
        # 根据textrank的大小排序，取得前200个字符
        sub_sentence = self.split_sentence(text)
        ranking_sentence = score_fn(sub_sentence)
        selected_text = set()
        current_text = ''
        for sen, _ in ranking_sentence:
            if len(current_text) < consitrain:
                current_text += sen
                selected_text.add(sen)
            else:
                break
        summarized = []
        for sen in sub_sentence:
            if sen in selected_text:
                summarized.append(sen)
        return summarized

    def punctuation_to_sentence(self, summarization, text):
        # 句子和标点符号的映射，待完善
        result = []
        punctuation = [',', '.', '。', '，']
        decode = [(m.group(), m.span()) for m in re.finditer('|'.join(summarization), text)]
        for sent, span in decode:
            for i in text[span[1]:]:
                if i in punctuation:
                    sent += i
                    result.append(sent)
                    break
        return result

    def get_result_simple(self, text):
        summarization = self.get_summarization_simple_with_text_rank(text, constrain=len(text) // 10)
        result = self.punctuation_to_sentence(summarization, text)
        result = (''.join(result)).split('。')
        return '。'.join(result[: -1]) + '。'


class TextRankKeyWords:
    def __init__(self, sentence, windows=3):
        self.sentence = sentence
        self.tokens = cut(sentence)  # [word1, word2, ...]
        self.windows = windows
        self.tokens = []
        self.get_tokens(sentence)

    def get_tokens(self, sentence=None):
        words = cut(sentence)
        tokens = []
        domain = ['a', 'n', 'nd', 'nh', 'ni', 'nl', 'ns', 'nt', 'nz', 'v']
        for idx, value in enumerate(postags_words(words)):
            if value in domain:
                tokens.append(words[idx])
        self.tokens = tokens

    def get_connect_graph_by_text_rank(self):
        graph = networkx.Graph()
        for idx, w in enumerate(self.tokens):
            words_tuple = [(self.tokens[i], w) for i in range(idx-self.windows, idx+self.windows)
                           if i >=0 and i < len(self.tokens)]
            graph.add_edges_from(words_tuple)
        return graph

    def analyse_tags_textrank(self, word_counts=20):
        # text rank 提取关键词
        """
        1. tokens = [word1, word2, ...]
        2. 去除停止词
        3. 词性标注 只留 名字 动词 形容词
        4. 根据window大小构建graph
        5. 利用PageRank计算得分
        :return:
        """
        graph = self.get_connect_graph_by_text_rank()
        score = networkx.pagerank(graph)
        return sorted(score.items(), key=lambda x: x[1], reverse=True)[:word_counts]


if __name__ == '__main__':
    sentence = """网易娱乐7月21日报道 林肯公园主唱查斯特·贝宁顿Chester Bennington于今天早上，在洛杉矶帕洛斯弗迪斯的一个私人庄园自缢身亡，年仅41岁。此消息已得到洛杉矶警方证实。洛杉矶警方透露，Chester的家人正在外地度假，
    Chester独自在家，上吊地点是家里的二楼。一说是一名音乐公司工作人员来家里找他时发现了尸体，也有人称是佣人最早发现其死
    亡。　　林肯公园另一位主唱麦克·信田确认了Chester Bennington自杀属实，并对此感到震惊和心痛，称稍后官方会发布声明。
    Chester昨天还在推特上转发了一条关于曼哈顿垃圾山的新闻。粉丝们纷纷在该推文下留言，不相信Chester已经走了。　　
    外媒猜测，Chester选择在7月20日自杀的原因跟他极其要好的朋友、Soundgarden(声音花园)乐队以及Audioslave乐队主唱C
    hris Cornell有关，因为7月20日是Chris Cornell的诞辰。而Chris Cornell于今年5月17日上吊自杀，享年52岁。Chris去
    世后，Chester还为他写下悼文。　　对于Chester的自杀，亲友表示震惊但不意外，因为Chester曾经透露过想自杀的念头，
    他曾表示自己童年时被虐待，导致他医生无法走出阴影，也导致他长期酗酒和嗑药来疗伤。目前，洛杉矶警方仍在调查Chester
    的死因。　　据悉，Chester与毒品和酒精斗争多年，年幼时期曾被成年男子性侵，导致常有轻生念头。Chester生前有过2段婚姻
    ，育有6个孩子。　　林肯公园在今年五月发行了新专辑《多一丝曙光One More Light》，成为他们第五张登顶Billboard排行榜
    的专辑。而昨晚刚刚发布新单《Talking To Myself》MV。"""
    ta = TextRankKeyWords(sentence)
    ta.analyse_tags_textrank()