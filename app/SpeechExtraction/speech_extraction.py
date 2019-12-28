from gensim.models import Word2Vec, word2vec
import numpy as np
from collections import defaultdict
import re, jieba
import os
import pyltp
jieba.load_userdict('static/wordsdict')
"""python 使用ltp: https://pyltp.readthedocs.io/zh_CN/latest/api.html"""

if os.path.exists('/Volumes/Samsung_T5/'):
    LTP_DATA_DIR = '/Volumes/Samsung_T5/AI/ltp_data'  # ltp模型目录的路径
elif os.path.exists('/root/.flag'):
    LTP_DATA_DIR = '/root/ltp_data'  # ltp模型目录的路径


def load_model(c):
    if c == 'pos':
        pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
        postagger = pyltp.Postagger()  # 初始化实例
        postagger.load(pos_model_path)  # 加载模型
        return postagger
    elif c == 'ner':
        ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体识别模型路径，模型名称为`pos.model`
        recognizer = pyltp.NamedEntityRecognizer()  # 初始化实例
        recognizer.load(ner_model_path)  # 加载模型
        return recognizer
    elif c == 'parser':
        par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`
        parser = pyltp.Parser()  # 初始化实例
        parser.load(par_model_path)  # 加载模型

        return parser


# 依存分析
class ParseDepend:
    """
    sentences: ["a b c", "d, e, f"]
    """
    def __init__(self, path='/root/ltp_data', sentences=None):
        self.LTP_DATA_DIR = LTP_DATA_DIR  # ltp模型目录的路径
        self.arcs = None
        self.sentences = self.deal_sentence(sentences)
        self.postags = [self.get_word_xing(sent) for sent in self.sentences]
        self.depends = [self.get_word_depend(self.sentences[i], self.postags[i]) for i in range(len(self.sentences))]
        self.ner = [self.get_ner(self.sentences[i], self.postags[i]) for i in range(len(self.sentences))]
        self.say_similar_words = self.get_say_similar()

    def deal_sentence(self, sentences):
        # 处理句子，并且分词
        sentence = ''.join(re.findall(r'[^\s]', sentences))
        pattern = re.compile('[。？?!！]')
        split = pattern.sub(' ', sentence).split()
        return [jieba.lcut(s) for s in split]

    def get_word_xing(self, words):
        # 词性标注
        # pos_model_path = os.path.join(self.LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
        # postagger = pyltp.Postagger()  # 初始化实例
        # postagger.load(pos_model_path)  # 加载模型
        # words = ['元芳', '你', '怎么', '看']  # 分词结果
        postagger = load_model('pos')
        postags = postagger.postag(words)  # 词性标注
        # print('\t'.join(postags))
        postagger.release()  # 释放模型
        return '\t'.join(postags).split()

    def get_word_depend(self, words, postags):
        # 依存分析
        # par_model_path = os.path.join(self.LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`
        # parser = pyltp.Parser()  # 初始化实例
        # parser.load(par_model_path)  # 加载模型
        # words = ['元芳', '你', '怎么', '看']
        # postags = ['nh', 'r', 'r', 'v']
        parser = load_model('parser')
        arcs = parser.parse(words, postags)  # 句法分析
        # print("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))
        parser.release()  # 释放模型
        return [(arc.relation, arc.head) for idx, arc in enumerate(arcs)]  # {'SBV': 4}

    def get_HED(self, words, arcs):
        # get HED
        root = None
        for i, arc in enumerate(arcs):
            if arc.relation == 'HED' and arc.head == 0:
                root = (i, arc.relation, words[i])
        return root

    def get_ner(self, words, postags):
        # 命名实体识别
        # ner_model_path = os.path.join(self.LTP_DATA_DIR, 'ner.model')  # 命名实体识别模型路径，模型名称为`pos.model`
        # recognizer = pyltp.NamedEntityRecognizer()  # 初始化实例
        # recognizer.load(ner_model_path)  # 加载模型
        recognizer = load_model('ner')
        netags = recognizer.recognize(words, postags)  # 命名实体识别
        recognizer.release()  # 释放模型
        # print('\t'.join(netags))
        return ' '.join(netags).split()

    def get_word(self, head, wtype, sentence, arcs):
        # get related word
        for i, arc in enumerate(arcs):
            if (arc.head - 1) == head and arc.relation == wtype:
                return sentence[i], i
        return 'nan', 'nan'

    def get_say_similar(self):
        with open('/Users/haha/Desktop/Project/app/static/say_word_similar', 'r') as fr:
            words = fr.readlines()
        return [word.replace('\n', '') for word in words]

    def get_main(self):
        """
        1. 查询谓语是否存在；2.查找主谓关系；3.确定命名实体；4.判断主语是否是人名，谓语动词是不是"说"相似的
        :return:
        """
        result = []
        for sen_idx, relate_words in enumerate(self.depends):
            for word_idx, relate_word in enumerate(relate_words):
                if 'SBV' not in relate_word: continue
                ZHUYU = self.sentences[sen_idx][word_idx]
                WEIYU = self.sentences[sen_idx][relate_word[1] - 1]
                QITA = self.sentences[sen_idx][relate_word[1]: ]
                if WEIYU in self.say_similar_words and QITA:
                    if result and  ''.join(QITA) in result[-1][-1]: continue
                    result.append([''.join(ZHUYU), ''.join(WEIYU),
                                   ''.join(QITA) if ''.join(QITA)[0] not in [',', '，'] else ''.join(QITA)[1: ]])
        if not result:
            result = ['不存在"说"相似的谓语动词']
        return {idx: val for idx, val in enumerate(result)}


if __name__ == '__main__':
    text = """31岁华裔女硕士在多伦多公寓身中多刀死亡今年9月2日，来自中国大陆、毕业于加拿大贵湖大学的31岁华裔女硕士田媛（音译）在多伦多市北约克区的一幢公寓楼中惨遭杀害，田媛双手被绑、身中多刀死于现场，田媛的男友吴胜当晚回屋发现女友尸体后立即报了警。据报道，现年31岁的田媛于2005年从中国大陆前往加拿大留学，曾先后就读于温莎大学和贵湖大学，田媛今年从贵湖大学毕业，获得了经济学硕士的学位，最近一直在找工作。田媛本来和哥哥在多伦多市北约克区4层楼高的柏文公寓内合租着一套房子，然而一个月前，田媛的哥哥开始前往美国工作，田媛经常一个人住在家中。田媛有一名叫做吴胜的36岁华裔男友，他也经常来田媛的公寓中住宿。然而今年9月2日傍晚，当田媛的男友吴胜下班后来到田媛的4楼公寓、用钥匙打开房门时，他震惊地看到田媛双手被绑，躺在地板上，身上血肉模糊，早已惨死多时。吴胜立即打电话报了警，警方赶到现场时，发现田媛身上被捅了至少5刀，她显然是死于严重的刀伤。警方调查发现，由于田媛的哥哥离开加拿大前往美国工作，一个月前，田媛曾在加拿大“加国无忧”网站和一家华文报纸上打出寻人合租广告，邀请有正当职业的单身男性或单身女性一起合租，希望将哥哥空下的那间卧室出租出去。在田媛遇害前，曾有好几名陌生人来看过田媛位于4楼的公寓。警方已经发出公告，希望近期到田媛家看过房子的租客能够主动和警方取得联系，提供调查线索。调查发现，长相美丽的田媛来加拿大的短短3年中，已经先后交过好几名男朋友。田媛遇害的可能原因目前有好几种：包括与潜在寻租客发生冲突，被害身亡；或者可能死于情杀等。多伦多警方呼吁：“不管大家想到什么，不管你们觉得这些事情是否很琐碎，都请告诉我们，因为往往是一些最容易被人忽视的东西反而会成为破案的关键线索。”欧阳"""
    pd = ParseDepend(path=LTP_DATA_DIR, sentences=text)
    r = pd.get_main()
    print(r)
