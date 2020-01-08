from collections import Counter, defaultdict
import jieba
import pandas as pd

MAX_INPUT_SEQ_LENGTH = 500 # 输入文本最大长度
MAX_TARGET_SEQ_LENGTH = 50  # 摘要最大长度
MAX_INPUT_VOCAB_LEN = 5000  # 文本序列最常用的5000个词
MAX_TARGET_VOCAB_LEN = 2000 # 摘要序列最常用的2000个词


def fit_text(X, Y, input_seq_max_length=None, target_seq_max_length=None):
    if not input_seq_max_length:
        input_seq_max_length = MAX_INPUT_SEQ_LENGTH
    if not target_seq_max_length:
        target_seq_max_length = MAX_TARGET_SEQ_LENGTH
    input_word2idx = dict()
    input_idx2word = dict()
    target_word2idx = dict()
    target_idx2word = dict()

    normal_x = list()
    counter_x = list()
    for sent in X:
        sent_cut = jieba.lcut(sent)
        sent_length = len(sent_cut)
        if sent_length > input_seq_max_length:
            sent_cut = sent_cut[: input_seq_max_length]
        normal_x.append(sent_cut) # [[word1, word2, ...], [word1, word2, ...], ...]
        counter_x.extend(sent_cut) # [wowd1, word2, word1, word3, word4, ...]

    normal_y = list()
    counter_y = list()
    for sent in Y:
        sent_cut = jieba.lcut(sent)
        sent_length = len(sent_cut)
        if sent_length > target_seq_max_length:
            sent_cut = sent_cut[: target_seq_max_length - 2]
        sent_cut = ['<BOS>'] + sent_cut + ['<EOS>']
        normal_y.append(sent_cut)
        counter_y.extend(sent_cut)

    input_words = Counter([j for i in normal_x for j in i ])
    target_words = Counter([j for i in counter_y for j in i])

    for idx, w in enumerate(input_words.most_common(MAX_INPUT_VOCAB_LEN)):
        input_word2idx[w[0]] = idx + 2
        input_idx2word[idx+2] = w[0]
    input_word2idx['PAD'] = 1
    input_word2idx['UNK'] = 0
    input_idx2word[1] = 'PAD'
    input_idx2word[0] = 'UNK'

    for idx, w in enumerate(target_words.most_common(MAX_TARGET_VOCAB_LEN)):
        target_word2idx[w[0]] = idx + 3
        target_idx2word[idx + 3] = w[0]
    target_word2idx['UNK'] = 0
    target_word2idx['<BOS>'] = 1
    target_word2idx['<EOS>'] = 2
    target_idx2word[0] = 'UNK'
    target_idx2word[1] = '<BOS>'
    target_idx2word[2] = '<EOS>'

    num_input_tokens = len(input_word2idx)
    num_target_tokens = len(target_word2idx)

    config = dict()
    config['input_word2idx'] = input_word2idx
    config['input_idx2word'] = input_idx2word
    config['target_word2idx'] = target_word2idx
    config['target_idx2word'] = target_idx2word
    config['num_input_tokens'] = num_input_tokens
    config['num_target_tokens'] = num_target_tokens
    config['max_input_seq_length'] = MAX_INPUT_SEQ_LENGTH
    config['max_target_seq_length'] = MAX_TARGET_SEQ_LENGTH
    return config






