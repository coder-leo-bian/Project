from keras.models import Model
from keras.layers import LSTM, Embedding, Input, Dense
from keras.preprocessing.sequence import pad_sequences
import jieba
import numpy as np
import pandas as pd
from app.GenerationSummarize.attention import Encoder, Decoder, Attention
from app.tools.fake_news_loader import fit_text
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint


HIDDEN_UNITS = 128

"""<BOS> 开始字符，<EOS> 终止字符，<PAD> 1 补全字符， <UNK> 0 未知字符"""

class Seq2Seq:

    model_name = 'seq2seq'

    def __init__(self, config):
        self.version = config['version'] if 'version' in config else 0
        self.input_word2idx = config['input_word2idx']
        self.input_idx2word = config['input_idx2word']
        self.target_word2idx = config['target_word2idx']
        self.target_idx2word = config['target_idx2word']
        self.num_input_tokens = config['num_input_tokens']
        self.num_target_tokens = config['num_target_tokens']
        self.max_input_seq_length = config['max_input_seq_length']
        self.max_target_seq_length = config['max_target_seq_length']
        self.config = config
        self.encoder_model = None
        self.encoder_inputs = None
        self.decoder_model = None
        self.model = None
        self.encoder()
        self.decoder()

    def encoder(self):
        encoder_inputs = Input(shape=(None,), name='encoder_name')  # [32, 500] batch_size = 32, max_len=500
        # 输入(batch, input_length) 输出 (batch, max_target_seq_length, output_dim)
        encoder_embedding = Embedding(input_dim=self.num_input_tokens,
                                      output_dim=self.max_target_seq_length,
                                      input_length=self.max_input_seq_length)(encoder_inputs)  # [32, 500, 50]

        encoder_lstm = LSTM(HIDDEN_UNITS, return_state=True, name='encoder_lstm')  # [32, 10]
        encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_embedding)
        encoder_states = [encoder_state_h, encoder_state_c]
        self.encoder_model = Model(encoder_inputs, encoder_states)
        self.encoder_inputs = encoder_inputs

    def decoder(self):
        # decoder输入
        decoder_inputs = Input(shape=(None, self.max_target_seq_length), name='decoder_inputs')
        # decoder 使用的循环神经网络
        decoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True,
                            return_sequences=True, name='decoder_lstm')
        decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs)
        # 全连接层
        decoder_dense = Dense(units=self.max_target_seq_length, activation='softmax',
                                name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([self.encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(optimizer='categorical_crossentropy', loss='rmsprop')
        self.model = model
        self.decoder_model = Model(decoder_inputs, decoder_outputs)

        decoder_inputs_state = [Input(shape=(None, HIDDEN_UNITS)), Input(shape=(None, HIDDEN_UNITS))]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_inputs_state)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_inputs_state, [decoder_outputs] + decoder_states)

    @staticmethod
    def get_config_path(model_dir_path):
        return model_dir_path+ '/' + Seq2Seq.model_name + '-config.npy'

    @staticmethod
    def get_weight_path(model_dir_path):
        return model_dir_path + '/' + Seq2Seq.model_name + '-weight.h5'

    def content_encoding(self, contents):
        # 1. 切词 [word1, word2, word3, ...]; 2. word to idx
        X_contents = []
        unk = 0
        for content in contents:
            X_content = []
            content = jieba.lcut(content) if jieba.lcut(content) <= self.num_input_tokens else jieba.lcut(content)[: self.num_input_tokens+1]
            for c in content:
                if c in self.input_word2idx:
                    X_content.append(self.input_word2idx[c])
                else:
                    X_content.append(unk)
            X_contents.append(X_content)
        return np.array(X_contents)

    def target_encoding(self, targets):
        X_targets = []
        unk = 0
        for target in targets:
            X_target = []
            target = ['<BOS>'] + jieba.lcut(target) \
                if len(jieba.lcut(target)) <= self.num_target_tokens - 2 \
                else jieba.lcut(target)[: self.num_target_tokens - 1] + ['<EOS>']
            for t in target:
                if t in self.target_word2idx:
                    X_targets.append(self.target_word2idx[t])
                else:
                    X_target.append(unk)
            X_targets.append(X_target)
        return np.array(X_targets)

    def fit(self, X_train, Y_train, X_test, Y_test, batch_size=32, epochs=10, model_dir_path=None):
        # 1. 获取模型路径和权重路径并保存
        config_file_path = Seq2Seq.get_config_path(model_dir_path)
        weight_file_path = Seq2Seq.get_weight_path(model_dir_path)
        np.save(config_file_path. self.config)
        # 2. input 和 target 文本处理(输入长度 <eos> <bos>)
        X_train = self.content_encoding(X_train)
        X_test = self.content_encoding(X_test)
        # 3. generate_batch
        # 4. fit_generator
        # 5. 保存权重
        pass


class Seq2SeqAttention:
    model_name = 'seq2seq-attention'
    def __init__(self, config):
        # self.num_input_tokens = config['num_input_tokens']  # len(num_input_tokens) 5000
        # self.max_input_seq_length = config['max_input_seq_length']  # 500
        # self.num_target_tokens = config['num_target_tokens']  # len(num_target_tokens) 2000
        # self.max_target_seq_length = config['max_target_seq_length']  # 50
        # self.input_word2idx = config['input_word2idx']  # dict({'word': 1})
        # self.input_idx2word = config['input_idx2word']  # dict({1: word})
        # self.target_word2idx = config['target_word2idx']
        # self.target_idx2word = config['target_idx2word']

        self.version = config['version'] if 'version' in config else 0
        self.input_word2idx = config['input_word2idx']
        self.input_idx2word = config['input_idx2word']
        self.target_word2idx = config['target_word2idx']
        self.target_idx2word = config['target_idx2word']
        self.num_input_tokens = config['num_input_tokens']  # 5000
        self.num_target_tokens = config['num_target_tokens']    # 2000
        self.max_input_seq_length = config['max_input_seq_length']  # 500
        self.max_target_seq_length = config['max_target_seq_length']    # 50
        config['version'] = self.version
        self.config = config
        encoder = Encoder(self.num_input_tokens, self.max_input_seq_length)
        encoder_inputs = encoder.inputs
        encoder_outputs = encoder.forward()

        decoder = Decoder(self.num_target_tokens)
        decoder_inputs = decoder.inputs
        decoder_outputs = decoder.forward(encoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        print(model.summary())
        self.model = model

    @staticmethod
    def get_config_params(dir_path):
        return dir_path + Seq2SeqAttention.model_name + '-config.npy'

    @staticmethod
    def get_weights_params(dir_path):
        return dir_path + Seq2SeqAttention.model_name + '-weight.h5'

    @staticmethod
    def get_model_params(dir_path):
        return dir_path + Seq2SeqAttention.model_name + '-weight.h5'

    def generate_batch(self, x_samples, y_samples, batch_size):
        num_batches = len(x_samples) // batch_size  # 625
        while True:
            for batchIdx in range(0, num_batches):
                start = batchIdx * batch_size
                end = (batchIdx + 1) * batch_size
                # padding: 500, 每一次64个样本
                encoder_input_data_batch = pad_sequences(x_samples[start:end], self.max_input_seq_length)
                # （64， 50，2001）
                decoder_target_data_batch = np.zeros(shape=(batch_size, self.max_target_seq_length, self.num_target_tokens))
                # （64， 50，2001）
                decoder_input_data_batch = np.zeros(shape=(batch_size, self.max_target_seq_length, self.num_target_tokens))
                for lineIdx, target_words in enumerate(y_samples[start:end]):
                    for idx, w in enumerate(target_words):
                        w2idx = 0  # default [UNK]
                        if w in self.target_word2idx:
                            w2idx = self.target_word2idx[w]
                        if w2idx != 0:
                            decoder_input_data_batch[lineIdx, idx, w2idx] = 1
                            if idx > 0:
                                decoder_target_data_batch[lineIdx, idx - 1, w2idx] = 1
                yield [encoder_input_data_batch, decoder_input_data_batch], decoder_target_data_batch

    def target_to_encoding(self, y):
        unk = 0
        temp = list()
        bos = '<BOS>'
        eos = '<EOS>'
        for sen in y:
            words = list()
            for word in [bos] + sen.split(' '):
                if len(words) >= self.max_target_seq_length: break
                if word in self.target_word2idx:
                    words.append(self.target_word2idx[word])
                else:
                    words.append(unk)
            words.append(self.target_word2idx[eos])
            temp.append(words)
        return np.array(temp)

    def input_to_encoding(self, x):
        temp = []
        unk = 0
        for sen in x:
            words = []
            for word in sen.split(' '):
                if len(words) > self.max_input_seq_length: break
                if word in self.input_word2idx:
                    words.append(self.input_word2idx[word])
                else:
                    words.append(unk)
            temp.append(words)
        return np.array(temp)

    def fit(self, x_train, y_train, x_test, y_test, epochs=10, batch_size=16, model_dir_path=None):
        model_dir_path = model_dir_path if model_dir_path else './models/'
        self.config['version'] += 1
        config_file_path = Seq2SeqAttention.get_config_params(model_dir_path)
        weight_file_path = Seq2SeqAttention.get_weights_params(model_dir_path)
        np.save(config_file_path, self.config)

        Xtrain = self.input_to_encoding(x_train)
        Xtest = self.input_to_encoding(x_test)

        Ytrain = self.target_to_encoding(y_train)
        Ytest = self.target_to_encoding(y_test)

        train_batch = self.generate_batch(Xtrain, Ytrain, batch_size=batch_size)
        test_batch = self.generate_batch(Xtest, Ytest, batch_size=batch_size)

        checkpoint = ModelCheckpoint(weight_file_path)

        train_num_batches = Xtrain.shape[0] // batch_size
        test_num_batches = Xtest.shape[0] // batch_size
        print('config：', self.config)
        history = self.model.fit_generator(generator=train_batch, steps_per_epoch=train_num_batches,
                                           epochs=epochs,
                                           verbose=1, validation_data=test_batch, validation_steps=test_num_batches,
                                           callbacks=[checkpoint])
        self.model.save_weights(weight_file_path)

# {
#     "name": "陈怡",
#     "level": 1,
# },
# {
#     "name": "王总",
#     "level": 2,
# },
# {
#     "name": "宋婧",
#     "level": 2
# },
# {
#     "name": "方总",
#     "level": 3,
# },
#   {
#     "name": "楠楠",
#     "level": 0,
#   },
#
#   {
#     "name": "亚文",
#     "level": 0,
#   },
#   {
#     "name": "小美",
#     "level": 0,
#   },
#   {
#     "name": "王晶",
#     "level": 0,
#   },
#   {
#     "name": "刘言",
#     "level": 0,
#   },
#   {
#     "name": "Carmen",
#     "level": 0,
#   },
#   {
#     "name": "Katie",
#     "level": 0,
#   },
#
#   {
#     "name": "Harvey",
#     "level": 0,
#   },

if __name__ == '__main__':
    np.random.seed(42)
    data_dir_path = '/Users/haha/Desktop/NewsSet'
    report_dir_path = './reports'
    model_dir_path = './models'

    print('loading csv file ...')
    df_x = pd.read_csv('/Users/haha/Desktop/NewsSet' + '/train_text.csv')
    df_y = pd.read_csv('/Users/haha/Desktop/NewsSet' + '/train_label.csv')
    print('extract configuration from input texts ...')
    X = [' '.join(jieba.lcut(x)) for x in df_x.values[..., 1][: 200]]
    Y = [' '.join(jieba.lcut(y)) for y in df_y.values[..., 1][: 200]]
    config = fit_text(X, Y)
    print(config)
    # path = 'models/' + Seq2SeqAttention.model_name + '-config.npy'
    # np.save(path, config)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=7)
    # print()
    s2s = Seq2SeqAttention(config)
    s2s.fit(x_train, y_train, x_test, y_test, epochs=10)