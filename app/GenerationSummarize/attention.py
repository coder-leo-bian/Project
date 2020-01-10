from keras import Model, Input, Sequential
from keras.layers import LSTM, Embedding, Dense, Bidirectional, \
    RepeatVector, Activation, Concatenate, Dot, Lambda, Softmax
import numpy as np
from keras import backend as K


HIDDEN_UNIT = 128
MAX_INPUT_SEQ_LENGTH = 500


class Encoder:
    def __init__(self, num_input_tokens, max_input_seq_length):
        # (64, 500)
        self.inputs = Input(shape=(None,), name='encoder_inputs')
        # (64, 500, 128)
        self.embedding_outputs = Embedding(input_dim=num_input_tokens,
                                           output_dim=HIDDEN_UNIT,
                                           input_length=max_input_seq_length,
                                           name='encoder_embedding')(self.inputs)
        # (64, 500, 128)
        self.rnn = Bidirectional(LSTM(HIDDEN_UNIT, return_sequences=True), merge_mode='concat', name='encoder_lstm')

    def forward(self):
        # encoder_outputs, encoder_state_h, encoder_state_c = encoder_rnn()
        return self.rnn(self.embedding_outputs)


class Attention:
    def __init__(self, attention_size=5):
        # self.model = Sequential([
        #     Dense(attention_size, input_shape=(64, 500, 296)),
        #     Activation('tanh'),
        #     Dense(1),
        # ])
        self.model = Sequential()
        self.model.add(Dense(attention_size, activation='tanh', input_dim=()))
        self.model.add(Dense(1))

    def attention_mechanism(self, encoder_context_state, decoder_hidden_state):
        """
        :param model:
        :param encoder_context_state: shape(64, 500, 256)
        :param decoder_hidden_state: shape(64, 50) 广播500次
        :return: 加权平均后的背景向量
        """
        # 对时间步序列进行广播
        decoder_hidden_state = RepeatVector(MAX_INPUT_SEQ_LENGTH)(decoder_hidden_state)
        # 对最后一列进行拼接
        enc_and_dec_state = Concatenate(axis=-1)([decoder_hidden_state, encoder_context_state])
        e = self.model(enc_and_dec_state)
        # 对时间步进行softmax
        alpha = Softmax(axis=1)(e)
        new_context = alpha * encoder_context_state
        return new_context


class Decoder:
    def __init__(self, num_target_tokens):
        # 64, 50, 2000
        self.inputs = Input(shape=(None, ), name='decoder_input')
        self.rnn = LSTM(HIDDEN_UNIT, return_state=True, name='decoder_lstm')
        self.dense = Dense(num_target_tokens, activation='softmax',name='decoder_dense')

    def forward(self, encoder_outputs, decoder_inputs):
        attention_outputs = Attention().attention_mechanism(encoder_outputs, decoder_inputs)
        rnn_layer = self.rnn(attention_outputs)
        dense_layer = self.dense(rnn_layer)
        return dense_layer


def build_model():
    config = {'num_input_tokens': 5000, 'max_input_seq_length': 500, 'num_target_tokens': 2000}
    encoder = Encoder(config['num_input_tokens'], config['max_input_seq_length'])
    encoder_inputs = encoder.inputs
    encoder_outputs = encoder.forward()

    decoder = Decoder(config['num_target_tokens'])
    decoder_inputs = decoder.inputs
    decoder_outputs = decoder.forward(encoder_outputs, decoder_inputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print(model.summary())

if __name__ == '__main__':
    build_model()







