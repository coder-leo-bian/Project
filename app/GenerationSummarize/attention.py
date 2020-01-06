from keras import Model, Input, Sequential
from keras.layers import LSTM, Embedding, Dense


HIDDEN_UNIT = 100


class Encoder:
    def __init__(self, num_input_tokens, max_input_seq_length):
        self.inputs = Input(shape=(None,), name='encoder_inputs')
        self.embedding_outputs = Embedding(input_dim=num_input_tokens,
                                           output_dim=HIDDEN_UNIT,
                                           input_length=max_input_seq_length,
                                           name='encoder_embedding')(self.inputs)
        self.rnn = LSTM(HIDDEN_UNIT, return_state=True, name='encoder_lstm')

    def forward(self):
        # encoder_outputs, encoder_state_h, encoder_state_c = encoder_rnn()
        return self.rnn(self.embedding_ouputs)


class Attention:
    def __init__(self, attention_size=5):
        self.model = Sequential()
        self.model.add(Dense(attention_size, activation='tanh'))
        self.model.add(Dense(1))

    def forward(self):
        pass


class Decoder:
    def __init__(self, num_target_tokens, max_target_seq_length=None):
        self.inputs = Input(shape=(None, num_target_tokens), name='decoder_input')
        self.rnn = LSTM(HIDDEN_UNIT, return_state=True, name='decoder_lstm')
        self.dense = Dense(num_target_tokens, activation='softmax', name='decoder_dense')

    def forward(self):
        decoder_outputs, decoder_state_h, decoder_state_c = self.rnn(self.inputs)
        decoder_outputs = self.dense(decoder_outputs)
        return decoder_outputs, decoder_state_h, decoder_state_c


if __name__ == '__main__':
    config = {'num_input_tokens': 5000, 'max_input_seq_length': 500, 'num_target_tokens': 2000}
    encoder = Encoder(config['num_input_tokens'], config['max_input_seq_length'])
    encoder_inputs = encoder.inputs
    encoder_outputs, encoder_state_h, encoder_state_c = encoder.forward()
    encoder_state = [encoder_state_h, encoder_state_c]

    decoder = Decoder(config['num_target_tokens'])
    decoder_inputs = decoder.inputs
    decoder_outputs, decoder_state_h, decoder_state_c = decoder.forward()

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    print(model.summary())







