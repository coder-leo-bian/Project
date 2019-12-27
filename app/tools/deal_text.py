import jieba
import re
import os
from pyltp import Postagger


if os.path.exists('/root/.flag'):
    path = 'static/stopwords'
    parse_path = '/root/ltp_data'
else:
    if os.path.exists('/Users/bj'):
        path = '/Users/bj/Desktop/Documents/Project_01/static/stopwords'
        parse_path = '/Users/bj/Desktop/Documents/ltp_data'
    elif os.path.exists('/Users/haha'):
        path = '/Users/haha/Desktop//Project_01/static/stopwords'


def cut(sentence):
    pattern = re.compile('[\w+]')
    sentence = ''.join(re.findall(pattern, sentence))
    return stop_word(jieba.lcut(sentence))


def postags_words(words=None):
    # 词性标注
    postagger = Postagger()
    postagger.load(os.path.join(parse_path, 'pos.model'))
    postags = postagger.postag(words)
    print(postags)
    postagger.release()
    return postags


def stop_word(words):
    with open(path, 'r') as fr:
        stopwords = fr.readlines()
    stopwords = [word.replace('\n', '') for word in stopwords ]
    return [word for word in words if word not in stopwords]


def sentence_split(sentence):
    """
    :param sentence: 文本str
    :return: [sen1, sen2, ...]
    """
    sentence = ''.join(re.findall(r'[^\s]', sentence))
    pattern = re.compile('[。？?!！.]')
    split = pattern.sub(' ', sentence).split()
    return split


