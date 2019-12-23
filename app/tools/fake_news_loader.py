from collections import Counter, defaultdict
import jieba
import pandas as pd

MAX_INPUT_SEQ_LENGTH = 500
MAX_TARGET_SEQ_LENGTH = 50
MAX_INPUT_VOCAB_LEN = 5000
MAX_TARGET_VOCAB_LEN = 3000

def fit_text(X, Y, input_seq_max_length=None, target_seq_max_length=None):
    if not input_seq_max_length:
        input_seq_max_length = MAX_INPUT_SEQ_LENGTH
    if not target_seq_max_length:
        target_seq_max_length = MAX_TARGET_SEQ_LENGTH
    input_word2idx = dict()
    input_idx2word = dict()
    target_word2idx = dict()
    target_idx2word = dict()



    input_words = Counter([w for s in X for w in s])
    target_words = Counter([w for s in Y for w in s])
    for idx, w in enumerate(input_words.most_common(MAX_INPUT_VOCAB_LEN)):
        input_word2idx[w[0]] = idx + 2
        input_idx2word[idx+2] = w[0]
    input_word2idx['PAD'] = 0
    input_word2idx['UNK'] = 1
    input_idx2word[0] = 'PAD'
    input_idx2word[1] = 'UNK'

    for idx, w in enumerate(target_words.most_common(MAX_TARGET_VOCAB_LEN)):
        target_word2idx[w[0]] = idx + 2
        target_idx2word[idx + 2] = w[0]
    target_word2idx['UNK'] = 0
    target_word2idx['PAD'] = 1
    target_idx2word[0] = 'UNK'
    target_idx2word[1] = 'PAD'

    num_input_tokens = len(input_word2idx)
    num_target_tokens = len(target_word2idx)

    input_embedding_bak = [[input_word2idx[w] if w in input_word2idx else input_word2idx['UNK']
                            for w in s] for s in X]
    input_embedding = []
    for ie in input_embedding_bak:
        if len(ie) < MAX_INPUT_SEQ_LENGTH:
            miss_len = MAX_INPUT_SEQ_LENGTH - len(ie)
            miss_num = [input_word2idx['PAD']] * miss_len
            ie.extend(miss_num)
        elif len(ie) > MAX_INPUT_SEQ_LENGTH:
            more_len = len(ie) - MAX_INPUT_SEQ_LENGTH
            ie = ie[: (len(ie)-more_len)]
        input_embedding.append(ie)

    target_embedding_bak = [[target_word2idx[w] if w in target_word2idx else target_word2idx['UNK']
                            for w in s] for s in Y]
    target_embedding = []
    for ie in target_embedding_bak:
        if len(ie) < MAX_TARGET_SEQ_LENGTH:
            miss_len = MAX_TARGET_SEQ_LENGTH - len(ie) - 1
            eos = ie.pop(-1)
            miss_num = [target_word2idx['PAD']] * miss_len
            ie.extend(miss_num)
            ie.append(eos)
        elif len(ie) > MAX_TARGET_SEQ_LENGTH:
            more_len = len(ie) - MAX_TARGET_SEQ_LENGTH
            ie = ie[: (len(ie)-more_len)]
        target_embedding.append(ie)

    config = dict()
    config['input_word2idx'] = input_word2idx
    config['input_idx2word'] = input_idx2word
    config['target_word2idx'] = target_word2idx
    config['target_idx2word'] = target_idx2word
    config['num_input_tokens'] = num_input_tokens
    config['num_target_tokens'] = num_target_tokens
    config['target_embedding'] = target_embedding
    config['input_embedding'] = input_embedding
    config['max_input_seq_length'] = MAX_INPUT_SEQ_LENGTH
    config['max_target_seq_length'] = MAX_TARGET_SEQ_LENGTH
    return config






