import jieba
import re
import os


if os.path.exists('/root/.flag'):
    path = '../static/stopwords'
else:
    if os.path.exists('/Users/bj'):
        path = '/Users/bj/Desktop/Documents/Project_01/static/stopwords'
    elif os.path.exists('/Users/haha'):
        path = '/Users/haha/Desktop//Project_01/static/stopwords'


def cut(sentence):
    pattern = re.compile('[\w+]')
    sentence = ''.join(re.findall(pattern, sentence))
    return stop_word(jieba.lcut(sentence))


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


