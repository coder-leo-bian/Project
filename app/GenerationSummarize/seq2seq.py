from keras.models import Model
from keras.layers import LSTM, Embedding, Input, Dense
from keras.preprocessing.sequence import pad_sequences
import jieba
import numpy as np
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
