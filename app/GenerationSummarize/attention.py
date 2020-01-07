from keras import Model, Input, Sequential
from keras.layers import LSTM, Embedding, Dense, Bidirectional, \
    RepeatVector, Activation, Concatenate, Dot, Lambda
import numpy as np
from keras import backend as K


HIDDEN_UNIT = 128


class Encoder:
    def __init__(self, num_input_tokens, max_input_seq_length):
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
    at_dot = Dot(axes=1)
    def __init__(self, attention_size=5):
        self.model = Sequential()
        self.model.add(Dense(attention_size, activation='tanh'))
        self.model.add(Dense(1))

    def forward(self, h, encoder_outputs):
        repeat = RepeatVector(500)(h)
        input_c_and_output_s = Concatenate(axis=-1)([encoder_outputs, repeat])
        self.model(input_c_and_output_s)
        attention = Activation('softmax')(input_c_and_output_s)
        context = Attention.at_dot([attention, encoder_outputs])
        return context

    def attention_layer(self, encoder_outputs, max_target_seq_length=50):
        lstm_layer = LSTM(HIDDEN_UNIT, return_state=True, name='at_LSTM_attention_layer')
        h = Lambda(lambda X: K.zeros(shape=(K.shape(X)[0], HIDDEN_UNIT)), name='h_attention_layer')(encoder_outputs)
        c = Lambda(lambda X: K.zeros(shape=(K.shape(X)[0], HIDDEN_UNIT)), name='c_attention_layer')(encoder_outputs)
        output = []
        for _ in range(max_target_seq_length):
            context = self.forward(h, encoder_outputs)
            h, _, c = lstm_layer(context, initial_state=[h, c])
            output.append(h)
        return output


class Decoder:
    def __init__(self, num_target_tokens):
        # None, None, 2000
        self.inputs = Input(shape=(None, num_target_tokens), name='decoder_input')
        self.attention = Attention()
        self.rnn = LSTM(HIDDEN_UNIT, return_state=True, name='decoder_lstm')
        self.dense = Dense(num_target_tokens, activation='softmax',name='decoder_dense')

    def forward(self, encoder_outputs):
        attention_outputs = self.attention.attention_layer(encoder_outputs)
        decoder_outputs = [self.dense(timestep) for timestep in attention_outputs]
        return decoder_outputs


if __name__ == '__main__':
    config = {'num_input_tokens': 5000, 'max_input_seq_length': 500, 'num_target_tokens': 2000}
    encoder = Encoder(config['num_input_tokens'], config['max_input_seq_length'])
    encoder_inputs = encoder.inputs
    encoder_outputs = encoder.forward()

    decoder = Decoder(config['num_target_tokens'])
    decoder_inputs = decoder.inputs
    decoder_outputs = decoder.forward(encoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    print(model.summary())







